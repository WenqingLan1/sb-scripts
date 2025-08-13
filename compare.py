#!/usr/bin/env python3
import json
import argparse
from pathlib import Path

def load_summary(path):
    data = {}
    for line in Path(path).open():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        
        # Group metrics by benchmark name (extracted from key prefixes)
        for key, value in obj.items():
            if "/" in key:
                # Extract benchmark name from key like "cpu-memory-bw-latency/return_code"
                benchmark_name = key.split("/")[0]
                metric_name = key.split("/", 1)[1]  # Get everything after first "/"
                
                # Skip return codes and correctness metrics as before
                if "return_code" in metric_name or "correctness" in metric_name or "monitor" in metric_name:
                    continue
                
                if benchmark_name not in data:
                    data[benchmark_name] = {}
                
                data[benchmark_name][metric_name] = value
            else:
                # Handle keys without "/" separator (like "node")
                if key not in ("return_code", "return_code_list") and "correctness" not in key and "monitor" not in key:
                    if "misc" not in data:
                        data["misc"] = {}
                    data["misc"][key] = value
    
    return data

def main():
    parser = argparse.ArgumentParser(
        description="Compare two results-summary.jsonl files")
    parser.add_argument("file1", help="first summary JSONL")
    parser.add_argument("file2", help="second summary JSONL")
    args = parser.parse_args()

    s1 = load_summary(args.file1)
    s2 = load_summary(args.file2)

    only1 = set(s1) - set(s2)
    only2 = set(s2) - set(s1)
    missing = sorted(only1 | only2)

    # intersect and filter out non-string keys and any with "correctness"
    common = sorted(set(s1) & set(s2))
    threshold = 0.05

    for bench in common:
         a = s1[bench]
         b = s2[bench]
         diffs = []
         keys = set(a) | set(b)
         for k in sorted(keys):
             va = a.get(k)
             vb = b.get(k)
             # skip if missing in either file
             if va is None or vb is None:
                 continue
             # normalize to numeric values
             if isinstance(va, list) and va:
                 v1 = va[0]
             elif isinstance(va, (int, float)):
                 v1 = va
             else:
                 continue
             if isinstance(vb, list) and vb:
                 v2 = vb[0]
             elif isinstance(vb, (int, float)):
                 v2 = vb
             else:
                 continue
             # skip identical
             if v1 == v2:
                 continue
             # compute relative change: file2 over file1
             try:
                 pct = (v2 - v1) / v1
             except (ZeroDivisionError, TypeError):
                 continue
             if abs(pct) > threshold:
                 diffs.append((k, va, vb, pct))
         if diffs:
             print(f"{bench}:")
             for k, va, vb, pct in diffs:
                 # show positive or negative change
                 print(f"  - {k}: {pct:+.1%}")
                 print(f"      {args.file1}: {va}")
                 print(f"      {args.file2}: {vb}")

    if missing:
        print(f"\nBenchmarks only in one file:")
        for bench in sorted(only1):
            print(f"  Only in {args.file1}: {bench}")
        for bench in sorted(only2):
            print(f"  Only in {args.file2}: {bench}")

if __name__ == "__main__":
    main()