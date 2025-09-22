#!/usr/bin/env python3
"""
generate_report.py
Compare two SuperBench markdown result files and generate an HTML report.

Supports all SuperBench benchmark types including:
- Model benchmarks (BERT, GPT, LSTM, ResNet, VGG, DenseNet, LLaMA)
- MICRO1 benchmarks (CUBLAS, cuBLASLt, cuDNN, GEMM-FLOPS including fp64/int8, kernel launch)
- MICRO2 benchmarks (CPU memory, GPU burn, NCCL bandwidth, matmul, sharding)
- Memory bandwidth tests (DTOD, GPUMEM, DTOH/HTOD via SM/DMA)
- GPU-STREAM, nvbandwidth, cpu-stream benchmarks
- IB, DISK benchmarks

Usage:
    python generate_report.py <baseline.md> <compare.md> [-o report.html] [--include-tables]
"""
import argparse
import sys
import json
from pathlib import Path
import html
import plotly.graph_objects as go
import plotly.io as pio
import re
from collections import defaultdict
from statistics import mean

# Edit this list to ignore metrics containing any of these tokens (case-insensitive).
# Example: 'correctness' will ignore 'gpu-copy-bw:correctness' and similar metrics.
# Note: Correctness tests and gpu-burn are typically pass/fail and may not be meaningful for performance comparison.
IGNORE_TOKENS = [
    "correctness",
    "gpu-burn",
    "lstm"
]

def normalize_gpu_metric_name(metric: str) -> str:
    """
    Normalize GPU metrics by removing GPU-specific numbering to enable averaging across GPUs.
    E.g., 'gpu-stream:perf/STREAM_ADD_double_gpu_0_buffer_4294967296_block_1024_bw'
    becomes 'gpu-stream:perf/STREAM_ADD_double_buffer_4294967296_block_1024_bw'
    Also handles 'gpu-burn/gpu_0_pass' -> 'gpu-burn/gpu_pass'
    """
    # Pattern to match gpu_<number> and remove it, handling both middle and end positions
    normalized = re.sub(r'_gpu_\d+_', '_', metric)  # Middle: _gpu_0_ -> _
    normalized = re.sub(r'_gpu_\d+$', '', normalized)  # End: _gpu_0 -> ''
    normalized = re.sub(r'/gpu_\d+_', '/', normalized)  # After slash: /gpu_0_ -> /
    normalized = re.sub(r'gpu_\d+_', '', normalized)  # Start: gpu_0_ -> ''
    return normalized

def aggregate_gpu_metrics(section_data: dict) -> dict:
    """
    Aggregate metrics across multiple GPUs by averaging values for metrics that differ only by GPU number.
    """
    # Group metrics by their normalized names
    metric_groups = defaultdict(list)
    
    for metric, value in section_data.items():
        if 'gpu_' in metric and ('gpu-stream' in metric or 'gpu-burn' in metric):
            normalized_name = normalize_gpu_metric_name(metric)
            metric_groups[normalized_name].append(value)
        else:
            # Keep non-GPU metrics as-is
            metric_groups[metric].append(value)
    
    # Average the grouped metrics
    aggregated = {}
    for normalized_metric, values in metric_groups.items():
        if len(values) > 1:
            # Multiple values to average
            aggregated[normalized_metric] = mean(values)
        else:
            # Single value, keep as-is
            aggregated[normalized_metric] = values[0]
    
    return aggregated

def parse_file(filepath: Path):
    """
    Parse a markdown-style results file into {section: {metric: mean_value}}.
    Preserves insertion order of sections and metrics.
    """
  # Notes:
  # - The parser is intentionally permissive because different summary files
  #   may format tables differently (e.g., extra pipes, missing headers).
  # - We look for common 'mean' column labels and fall back to simple
  #   two-column rows when possible. Non-numeric cells are ignored.
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
    
    # Aggregate GPU metrics across multiple GPUs
    for section in data:
        data[section] = aggregate_gpu_metrics(data[section])
    
    return data

def group_key(metric: str) -> str:
    """
    Group by the leading token of the metric, stripping trailing sizes/variants.
    Enhanced to handle new benchmark types including fp64 and int8 GEMM operations.
    """
    # Normalize input to a string and trim whitespace.
    # We aim to extract a stable "group" prefix for related metrics.
    m = str(metric).strip()
    # Enhanced explicit prefixes mapping for better grouping
    PREFIX_MAP = {
        # Group NCCL bandwidth tests by operation type
        'nccl-bw:nvlink-allgather': 'nccl-allgather',
        'nccl-bw:nvlink-alltoall': 'nccl-alltoall', 
        'nccl-bw:nvlink-broadcast': 'nccl-broadcast',
        'nccl-bw:nvlink-reduce': 'nccl-reduce',
        'nccl-bw:nvlink-reducescatter': 'nccl-reducescatter',
        'nccl-bw:nvlink': 'nccl-allreduce',  # allreduce is the base nvlink test
        # Group GPU copy bandwidth tests by direction and method
        'gpu-copy-bw:correctness': 'gpu-copy-correctness',
        'gpu-copy-bw:perf': 'gpu-copy-perf',
        # Group GPU stream tests 
        'gpu-stream:perf': 'gpu-stream',
        # Group CPU stream tests by socket
        'cpu-stream:cross-socket0': 'cpu-stream-socket0',
        'cpu-stream:cross-socket1': 'cpu-stream-socket1',
    }
    
    # Check for exact matches first
    if m in PREFIX_MAP:
        return PREFIX_MAP[m]
    
    # Check for prefix matches in order of specificity
    for prefix, group in sorted(PREFIX_MAP.items(), key=len, reverse=True):
        if m.startswith(prefix):
            return group
    
    # Capture a leading token that starts with a letter and continues with
    # letters, digits, underscores or dashes. This catches common metric
    # identifiers like 'gpu-stream', 'nvbandwidth', 'gemm-flops', etc.
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


MAX_WRAP_CHUNKS = 10

def wrap_chunks(s: str, width: int = 50, max_chunks: int = MAX_WRAP_CHUNKS):
    """
    Break a long string into a list of chunk strings suitable for putting into
    `customdata` so `hovertemplate` can compose them with `<br>` between chunks.

    Behaviour summary:
    - Prefer splitting on whitespace to preserve whole words when possible.
    - If a single word/token exceeds `width`, break that token into
      width-sized slices so extremely long identifiers are still wrapped.
    - Always return exactly `max_chunks` entries (pad with empty strings)
      so `customdata` rows have a stable shape.
    """
    if s is None:
        return [""] * max_chunks
    text = str(s)
    parts = []
    cur = ""
    # Split while preserving whitespace groups so we can rebuild lines
    for token in re.split(r'(\s+)', text):
        if not token:
            continue
        tok = token.strip()
        if not tok:
            continue
        # If token itself is longer than width, flush any current buffer
        # then slice the long token into width-sized pieces.
        if len(tok) > width:
            if cur:
                parts.append(cur.strip())
                cur = ""
            t = tok
            for i in range(0, len(t), width):
                parts.append(t[i:i+width])
            continue
        # Otherwise attempt to append to the current line; if it would exceed
        # width, push current and start a new one.
        if len((cur + ' ' + tok).strip()) > width and cur:
            parts.append(cur.strip())
            cur = tok
        else:
            cur = (cur + ' ' + tok).strip()
    if cur:
        parts.append(cur.strip())
    # Escape HTML and normalize to fixed length
    parts = [html.escape(p) for p in parts][:max_chunks]
    if len(parts) < max_chunks:
        parts.extend([""] * (max_chunks - len(parts)))
    return parts


def wrap_text_single(s: str, width: int = 50, max_chunks: int = MAX_WRAP_CHUNKS) -> str:
    """
    Return a single HTML string with <br> between non-empty chunks.
    This avoids emitting empty lines when fewer chunks are used.
    """
    parts = wrap_chunks(s, width=width, max_chunks=max_chunks)
    # Remove empty parts and join with <br> so the hover shows no blank lines.
    parts = [p for p in parts if p]
    if not parts:
        return ""
    return "<br>".join(parts)

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
    plot_pairs = []
    id_counter = 0

    for sec in sections:
        # Determine metric order: baseline metrics then compare-only
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

        # Group metrics by group_key preserving order
        groups = []
        group_map = {}
        for m in metrics:
            g = group_key(m)
            if g not in group_map:
                group_map[g] = []
                groups.append(g)
            group_map[g].append(m)

        html_snippets.append(f"<h2>{html.escape(sec)}</h2>")
        for g in groups:
            group_metrics = [m for m in group_map[g] if not should_ignore_metric(m)]
            if not group_metrics:
                continue

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

            html_snippets.append(f"<h3>Group: {html.escape(g)}</h3>")
            if include_tables:
                html_snippets.append(build_table_html(group_metrics, baseline_vals, compare_vals, percent_diffs))

            # Charts
            id_counter += 1
            # Reserve a fixed plotting area height and separate bottom margin
            # to prevent long metric names (x-axis labels) from stealing plot
            # vertical space. Values lowered to reduce overall chart height
            # while keeping label area readable on typical monitors.
            plot_area_height = 360
            bottom_margin_for_labels = 120
            height_bar = plot_area_height + bottom_margin_for_labels

            plot_area_height_diff = 260
            bottom_margin_for_labels_diff = 100
            height_diff = plot_area_height_diff + bottom_margin_for_labels_diff
            bar_id = f"plot_bar_{id_counter}"
            diff_id = f"plot_diff_{id_counter}"

            # prepare wrapped hover text and customdata rows [report_name, wrapped_html]
            wrapped_single = [wrap_text_single(m, width=50, max_chunks=MAX_WRAP_CHUNKS) for m in group_metrics]
            custom_rows_baseline = [[label_baseline, ws] for ws in wrapped_single]
            custom_rows_compare = [[label_compare, ws] for ws in wrapped_single]

            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(x=group_metrics, y=baseline_vals, name=label_baseline,
                                     customdata=custom_rows_baseline,
                                     hovertemplate="%{customdata[0]}<br>%{customdata[1]}<br><b>%{y}</b><extra></extra>"))
            fig_bar.add_trace(go.Bar(x=group_metrics, y=compare_vals, name=label_compare,
                                     customdata=custom_rows_compare,
                                     hovertemplate="%{customdata[0]}<br>%{customdata[1]}<br><b>%{y}</b><extra></extra>"))
            # Increase chart height for better y-axis spacing/readability.
            # Place the legend horizontally above the plot so long file
            # path labels do not consume horizontal space on the right.
            fig_bar.update_traces(hoverlabel=dict(align='left', bgcolor='rgba(255,255,255,0.9)', bordercolor='black'))
            fig_bar.update_layout(
                title=f"[{sec}] {g} - Metric Comparison",
                xaxis_title="", yaxis_title="Mean Value",
                barmode="group", template="plotly_white",
                height=height_bar, hovermode="closest",
                legend=dict(orientation='h', y=1.08, x=0.01, xanchor='left'),
                margin=dict(t=120, b=bottom_margin_for_labels)
            )
            # Keep x-axis label area stable: rotate ticks and prevent pan/zoom changing
            # the x-axis label layout (fixedrange keeps labels readable during pan).
            fig_bar.update_xaxes(tickangle=45, automargin=False, fixedrange=False, title_standoff=40)
            fig_bar.update_traces(hoverlabel=dict(align='left'))

            diff_custom = [[f"% diff ({label_compare} vs {label_baseline})", ws] for ws in wrapped_single]
            fig_diff = go.Figure()
            fig_diff.add_trace(go.Scatter(x=group_metrics, y=percent_diffs, mode="lines+markers",
                                           name=f"% diff ({label_compare} vs {label_baseline})",
                                           line=dict(color="orange"),
                                           customdata=diff_custom,
                                           hovertemplate="%{customdata[0]}<br>%{customdata[1]}<br><b>%{y:.2f}%</b><extra></extra>"))
            fig_diff.add_shape(type='line', x0=-0.5, x1=len(group_metrics)-0.5, y0=0, y1=0,
                               line=dict(color='gray', dash='dash'))
            # Slightly larger diff plot to match bar chart vertical space.
            # Use a top-positioned horizontal legend to avoid horizontal clutter.
            fig_diff.update_traces(hoverlabel=dict(align='left', bgcolor='rgba(255,255,255,0.9)', bordercolor='black'))
            fig_diff.update_layout(
                title=f"[{sec}] {g} - Percent Difference ({label_compare} vs {label_baseline})",
                xaxis_title="", yaxis_title="Percent Difference (%)",
                template="plotly_white", height=height_diff, hovermode="closest",
                legend=dict(orientation='h', y=1.08, x=0.01, xanchor='left'),
                margin=dict(t=120, b=bottom_margin_for_labels_diff)
            )
            fig_diff.update_xaxes(tickangle=45, automargin=False, fixedrange=False, title_standoff=36)
            fig_diff.update_traces(hoverlabel=dict(align='left'))

            # Export HTML fragments and register pair for JS
            bar_html = fig_bar.to_html(full_html=False, include_plotlyjs=False, div_id=bar_id)
            diff_html = fig_diff.to_html(full_html=False, include_plotlyjs=False, div_id=diff_id)
            html_snippets.append(bar_html)
            html_snippets.append(diff_html)

            plot_pairs.append({
                "bar": bar_id,
                "diff": diff_id,
                "x": group_metrics,
                "ys": [baseline_vals, compare_vals],
                "diffs": percent_diffs
            })

    # Build JS that syncs zoom/pan and triggers y-autorange on visible data
    js_sync = """
<script>
document.addEventListener('DOMContentLoaded', function(){
  const pairs = REPLACE_PAIRS_JSON;

  function extractXRange(relayoutData){
    if(!relayoutData) return null;
    if(relayoutData['xaxis.range']) return relayoutData['xaxis.range'];
    let x0 = null, x1 = null;
    for(const k in relayoutData){
      if(k === '_sync_source') continue;
      const m = k.match(/^xaxis\.range\[(\d)\]$/);
      if(m){
        if(m[1]==='0') x0 = relayoutData[k];
        if(m[1]==='1') x1 = relayoutData[k];
      }
    }
    if(x0 !== null && x1 !== null) return [x0, x1];
    return null;
  }

  function visibleIndexRangeFromX(xr, xArray){
    if(!xr) return null;
    let a = xr[0], b = xr[1];
    if(typeof a === 'string' && typeof b === 'string' && xArray.indexOf(a) !== -1 && xArray.indexOf(b) !== -1){
      let i0 = xArray.indexOf(a), i1 = xArray.indexOf(b);
      if(i0 > i1){ let t = i0; i0 = i1; i1 = t; }
      return {start: i0, end: i1};
    }
    let na = parseFloat(a), nb = parseFloat(b);
    if(!isNaN(na) && !isNaN(nb)){
      let i0 = Math.max(0, Math.floor(Math.min(na, nb)));
      let i1 = Math.min(xArray.length - 1, Math.ceil(Math.max(na, nb)));
      return {start: i0, end: i1};
    }
    return null;
  }

  function computeRangeForValues(arrays, idxRange){
    if(!idxRange) return null;
    let ymin = Number.POSITIVE_INFINITY, ymax = Number.NEGATIVE_INFINITY;
    let found = false;
    for(let a=0; a<arrays.length; a++){
      const arr = arrays[a] || [];
      for(let i=idxRange.start; i<=idxRange.end && i < arr.length; i++){
        const v = arr[i];
        if(v === null || v === undefined) continue;
        const num = Number(v);
        if(Number.isFinite(num)){
          found = true;
          if(num < ymin) ymin = num;
          if(num > ymax) ymax = num;
        }
      }
    }
    if(!found) return null;
    if(ymin === ymax){
      if(ymin === 0){ ymin = -1; ymax = 1; }
      else { const pad = Math.abs(ymin) * 0.06; ymin -= pad; ymax += pad; }
    } else { const span = ymax - ymin; const pad = span * 0.06; ymin -= pad; ymax += pad; }
    return [ymin, ymax];
  }

  pairs.forEach(function(pair){
    const barDiv = document.getElementById(pair.bar);
    const diffDiv = document.getElementById(pair.diff);
    const xArray = pair.x || [];
    const ys = pair.ys || [[],[]];
    const diffs = pair.diffs || [];

    function onRelayout(sourceId, targetId, eventData){
      if(eventData && eventData['_sync_source']) return;
      const xr = extractXRange(eventData);
      const idxRange = visibleIndexRangeFromX(xr, xArray);
      if(idxRange){
                // Compute only the percent-diff range; for bar charts prefer Plotly's
                // built-in autorange so axis ticks are nicely rounded and match the
                // default behaviour seen in the older notebook.
                const diffRange = computeRangeForValues([diffs], idxRange);
                try{
                    let payload = {'xaxis.range': xr, '_sync_source': sourceId};
                    if(targetId.startsWith('plot_diff_')){
                        if(diffRange) payload['yaxis.range'] = diffRange;
                        else payload['yaxis.autorange'] = true;
                    } else {
                        // For bar charts, ask Plotly to autorange (nicer tick placement).
                        payload['yaxis.autorange'] = true;
                    }
                    Plotly.relayout(targetId, payload);
                }catch(e){}
                try{
                    if(sourceId === pair.bar && diffDiv){
                        if(diffRange) Plotly.relayout(pair.diff, {'yaxis.range': diffRange, '_sync_source': sourceId});
                        else Plotly.relayout(pair.diff, {'yaxis.autorange': true, '_sync_source': sourceId});
                    }
                    if(sourceId === pair.diff && barDiv){
                        // When the percent-diff trace is the source, let the bar chart
                        // autorange rather than imposing an explicit numeric range.
                        try{
                            Plotly.relayout(pair.bar, {'yaxis.autorange': true, '_sync_source': sourceId});
                        }catch(e){}
                    }
                }catch(e){}
      } else {
        try{ Plotly.relayout(targetId, {'yaxis.autorange': true, '_sync_source': sourceId}); }catch(e){}
      }
    }

    if(barDiv && barDiv.on){ barDiv.on('plotly_relayout', function(eventData){ onRelayout(pair.bar, pair.diff, eventData); }); }
    if(diffDiv && diffDiv.on){ diffDiv.on('plotly_relayout', function(eventData){ onRelayout(pair.diff, pair.bar, eventData); }); }
  });
});
</script>
"""
    js_sync = js_sync.replace("REPLACE_PAIRS_JSON", json.dumps(plot_pairs))

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
  {js_sync}
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
