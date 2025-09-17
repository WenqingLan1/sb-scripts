#!/usr/bin/env python3
"""
generate_report.py
Compare two SuperBench markdown result files and generate an HTML report.

Usage:
    python generate_report.py <baseline.md> <compare.md> [-o report.html] [--include-tables]
"""
import argparse
import sys
from pathlib import Path
import html
import plotly.graph_objects as go
import plotly.io as pio
import re

# Edit this list to ignore metrics containing any of these tokens (case-insensitive).
# Example: 'correctness' will ignore 'gpu-copy-bw:correctness' and similar metrics.
IGNORE_TOKENS = [
    "correctness",
]

def parse_file(filepath: Path):
    """
    Parse a markdown-style results file into {section: {metric: mean_value}}.
    Preserves insertion order of sections and metrics.
    """
    data = {}
    current = None
    if not filepath.exists():
        return data
    with filepath.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("## "):
                current = line[3:].strip()
                data[current] = {}
                continue
            if line.startswith("# "):
                current = line[2:].strip()
                data.setdefault(current, {})
                continue

            if "|" in line:
                parts = [p.strip() for p in line.split("|")]
                parts = [p for p in parts if p != ""]
                if len(parts) < 2:
                    continue
                # Common pattern: metric | mean | value
                if len(parts) >= 3 and parts[1].lower() in ("mean", "avg", "average"):
                    metric = parts[0]
                    try:
                        value = float(parts[2])
                        if current is None:
                            current = "Global"
                            data.setdefault(current, {})
                        if metric not in data[current]:
                            data[current][metric] = value
                    except Exception:
                        pass
                    continue
                # Find 'mean' token and take next column as value
                found = False
                for i, p in enumerate(parts[:-1]):
                    if "mean" in p.lower():
                        metric = parts[0]
                        try:
                            value = float(parts[i+1])
                            if current is None:
                                current = "Global"
                                data.setdefault(current, {})
                            if metric not in data[current]:
                                data[current][metric] = value
                        except Exception:
                            pass
                        found = True
                        break
                if found:
                    continue
                # Last resort: metric,value (2 columns)
                if len(parts) == 2:
                    metric = parts[0]
                    try:
                        value = float(parts[1])
                        if current is None:
                            current = "Global"
                            data.setdefault(current, {})
                        if metric not in data[current]:
                            data[current][metric] = value
                    except Exception:
                        pass
    return data

def group_key(metric: str) -> str:
    """
    Group by the leading token of the metric, stripping trailing sizes/variants.
    Examples:
      'gpu-stream:perf' -> 'gpu-stream'
      'model-benchmarks:gpt@small-fp8' -> 'model-benchmarks'
      'gemm-flops' -> 'gemm'
      'nvbandwidth 128' -> 'nvbandwidth'
      'nvbandwidth(128MB)' -> 'nvbandwidth'
    """
    m = str(metric).strip()
    # quick explicit prefixes mapping (optional) - keep small if used
    PREFIX_MAP = {
        # 'cublaslt-gemm': 'cublas',  # example if you want custom remaps
    }
    if m in PREFIX_MAP:
        return PREFIX_MAP[m]
    # capture leading alnum/-,_ sequence (stop at whitespace, parentheses, digits-only suffix, or other punctuation)
    mo = re.match(r'^([A-Za-z][A-Za-z0-9_\-]*)', m)
    if mo:
        return mo.group(1)
    # fallback to previous separators logic
    if ":" in m:
        return m.split(":", 1)[0]
    if "@" in m:
        return m.split("@", 1)[0]
    if "-" in m:
        return m.split("-", 1)[0]
    return m

def should_ignore_metric(metric: str) -> bool:
    if not IGNORE_TOKENS:
        return False
    m = metric.lower()
    for tok in IGNORE_TOKENS:
        if tok.lower() in m:
            return True
    return False

def fmt(v):
    if v is None:
        return ""
    try:
        # pretty-print integers without decimals, floats with 3 decimals
        if float(v).is_integer():
            return str(int(v))
        return f"{float(v):,.3f}"
    except Exception:
        return str(v)

def build_table_html(metrics, baseline_vals, compare_vals, percent_diffs):
    """
    Build a small HTML table for a group.
    """
    rows = []
    rows.append("<table border='1' cellpadding='6' cellspacing='0'>")
    rows.append("<thead><tr><th>Metric</th><th>Baseline</th><th>Compare</th><th>% diff</th></tr></thead>")
    rows.append("<tbody>")
    for m, b, c, pd in zip(metrics, baseline_vals, compare_vals, percent_diffs):
        mb = html.escape(m)
        bb = html.escape(fmt(b))
        cc = html.escape(fmt(c))
        pd_s = "" if pd is None else f"{pd:.2f}%"
        rows.append(f"<tr><td>{mb}</td><td style='text-align:right'>{bb}</td><td style='text-align:right'>{cc}</td><td style='text-align:right'>{pd_s}</td></tr>")
    rows.append("</tbody></table>")
    return "\n".join(rows)

def build_report(baseline_path, compare_path, out_path, include_tables=False):
    baseline_data = parse_file(Path(baseline_path))
    compare_data = parse_file(Path(compare_path))
    label_baseline = baseline_path
    label_compare = compare_path

    # Section ordering: baseline sections first, then compare-only sections
    sections = []
    for s in baseline_data.keys():
        if s not in sections:
            sections.append(s)
    for s in compare_data.keys():
        if s not in sections:
            sections.append(s)

    html_snippets = []
    for sec in sections:
        # Determine metric order: baseline metrics in order, then compare-only metrics
        baseline_metrics = [m for m in list(baseline_data.get(sec, {}).keys()) if not should_ignore_metric(m)]
        compare_metrics = [m for m in list(compare_data.get(sec, {}).keys()) if not should_ignore_metric(m)]
        metrics = []
        for m in baseline_metrics:
            if m not in metrics:
                metrics.append(m)
        for m in compare_metrics:
            if m not in metrics:
                metrics.append(m)
        if not metrics:
            continue

        # Group metrics by group_key preserving order (first occurrence)
        groups = []
        group_map = {}
        for m in metrics:
            g = group_key(m)
            if g not in group_map:
                group_map[g] = []
                groups.append(g)
            group_map[g].append(m)

        # Render each group: optional HTML table + charts
        html_snippets.append(f"<h2>{html.escape(sec)}</h2>")
        for g in groups:
            group_metrics = [m for m in group_map[g] if not should_ignore_metric(m)]
            if not group_metrics:
                continue

            # prepare values
            baseline_vals = [baseline_data.get(sec, {}).get(m) for m in group_metrics]
            compare_vals = [compare_data.get(sec, {}).get(m) for m in group_metrics]
            percent_diffs = []
            for b, c in zip(baseline_vals, compare_vals):
                if b is None or c is None:
                    percent_diffs.append(None)
                else:
                    try:
                        percent_diffs.append(((c - b) / b) * 100 if b != 0 else None)
                    except Exception:
                        percent_diffs.append(None)

            # Small table (optional)
            html_snippets.append(f"<h3>Group: {html.escape(g)}</h3>")
            if include_tables:
                html_snippets.append(build_table_html(group_metrics, baseline_vals, compare_vals, percent_diffs))

            # Charts
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(x=group_metrics, y=baseline_vals, name=label_baseline))
            fig_bar.add_trace(go.Bar(x=group_metrics, y=compare_vals, name=label_compare))
            fig_bar.update_layout(
                title=f"[{sec}] {g} - Metric Comparison",
                xaxis_title="Metric",
                yaxis_title="Mean Value",
                barmode="group",
                template="plotly_white",
                height=420
            )

            fig_diff = go.Figure()
            fig_diff.add_trace(go.Scatter(x=group_metrics, y=percent_diffs, mode="lines+markers",
                                          name=f"% diff ({label_compare} vs {label_baseline})",
                                          line=dict(color="orange")))
            fig_diff.add_shape(type='line', x0=-0.5, x1=len(group_metrics)-0.5, y0=0, y1=0,
                               line=dict(color='gray', dash='dash'))
            fig_diff.update_layout(
                title=f"[{sec}] {g} - Percent Difference ({label_compare} vs {label_baseline})",
                xaxis_title="Metric",
                yaxis_title="Percent Difference (%)",
                template="plotly_white",
                height=380
            )

            html_snippets.append(pio.to_html(fig_bar, full_html=False, include_plotlyjs=False))
            html_snippets.append(pio.to_html(fig_diff, full_html=False, include_plotlyjs=False))

    table_note = " (tables included)" if include_tables else " (charts only)"
    ignore_note = f" (ignored: {', '.join(IGNORE_TOKENS)})" if IGNORE_TOKENS else ""
    final_html = f"""<html>
<head>
  <title>Metrics comparison{table_note}{ignore_note}</title>
  <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
  <style>
    body {{ font-family: Arial; padding: 20px; }}
    .plot-container {{ margin-bottom: 60px; }}
    table {{ border-collapse: collapse; margin-bottom: 12px; }}
    th {{ background:#eee; }}
  </style>
</head>
<body>
  <h1>Plotly Metrics Comparison Report{table_note}{ignore_note}</h1>
  <p><strong>Baseline:</strong> {html.escape(label_baseline)}<br/><strong>Compare:</strong> {html.escape(label_compare)}</p>
  {"".join(f'<div class="plot-container">{plot}</div>' for plot in html_snippets)}
</body>
</html>"""

    Path(out_path).write_text(final_html, encoding="utf-8")
    print(f"✔ Report written to: {out_path}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("baseline", help="Baseline markdown summary (used for ordering)")
    p.add_argument("compare", help="Compare markdown summary")
    p.add_argument("-o", "--output", default="comparison_report.html")
    p.add_argument("--include-tables", action="store_true",
                   help="Include numeric HTML tables for each metric group in the report.")
    args = p.parse_args()
    if not Path(args.baseline).exists():
        print("Baseline file not found:", args.baseline, file=sys.stderr); sys.exit(2)
    if not Path(args.compare).exists():
        print("Compare file not found:", args.compare, file=sys.stderr); sys.exit(2)
    build_report(args.baseline, args.compare, args.output, include_tables=args.include_tables)

if __name__ == "__main__":
    main()
