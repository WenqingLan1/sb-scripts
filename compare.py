#!/usr/bin/env python3
import json
import argparse
from pathlib import Path

def load_summary(path):
    """Load benchmark data from the first line of a JSONL file."""
    with Path(path).open() as f:
        line = f.readline().strip()
        if not line:
            return {}
        
        obj = json.loads(line)
        data = {}
        
        # Group metrics by benchmark name
        for key, value in obj.items():
            if "/" in key:
                benchmark_name, metric_name = key.split("/", 1)
                
                # Skip unwanted benchmark types
                if any(skip in benchmark_name for skip in ["correctness", "monitor", "return_code"]):
                    continue
                
                if benchmark_name not in data:
                    data[benchmark_name] = {}
                data[benchmark_name][metric_name] = value
        
        return data

def main():
    parser = argparse.ArgumentParser(
        description="Compare two results-summary.jsonl files")
    parser.add_argument("file1", help="first summary JSONL")
    parser.add_argument("file2", help="second summary JSONL")
    args = parser.parse_args()

    data1 = load_summary(args.file1)
    data2 = load_summary(args.file2)

    # Find benchmarks that exist in only one file
    only1 = set(data1) - set(data2)
    only2 = set(data2) - set(data1)
    common = set(data1) & set(data2)
    
    threshold = 0.05

    # Compare common benchmarks
    for benchmark in sorted(common):
        metrics1 = data1[benchmark]
        metrics2 = data2[benchmark]
        
        # Check for metric differences
        metrics_only1 = set(metrics1) - set(metrics2)
        metrics_only2 = set(metrics2) - set(metrics1)
        
        if metrics_only1 or metrics_only2:
            print(f"{benchmark} - Missing metrics:")
            if metrics_only1:
                print(f"  Only in {args.file1}:")
                for metric in sorted(metrics_only1):
                    print(f"    - {metric}")
            if metrics_only2:
                print(f"  Only in {args.file2}:")
                for metric in sorted(metrics_only2):
                    print(f"    - {metric}")
            print()
        
        # Check for value differences in common metrics
        common_metrics = set(metrics1) & set(metrics2)
        diffs = []
        
        for metric in sorted(common_metrics):
            val1, val2 = metrics1[metric], metrics2[metric]
            
            # Skip non-numeric comparisons
            if not isinstance(val1, (int, float)) or not isinstance(val2, (int, float)):
                continue
                
            if val1 == val2:
                continue
                
            try:
                pct = (val2 - val1) / val1
                if abs(pct) > threshold:
                    diffs.append((metric, val1, val2, pct))
            except ZeroDivisionError:
                continue
        
        if diffs:
            print(f"{benchmark} - Value differences:")
            for metric, val1, val2, pct in diffs:
                print(f"  - {metric}: {pct:+.1%}")
                print(f"      {args.file1}: {val1}")
                print(f"      {args.file2}: {val2}")
            print()

    # Print missing benchmarks
    if only1 or only2:
        print("Benchmarks only in one file:")
        if only1:
            print(f"  Only in {args.file1}:")
            for benchmark in sorted(only1):
                print(f"    - {benchmark}")
        if only2:
            print(f"  Only in {args.file2}:")
            for benchmark in sorted(only2):
                print(f"    - {benchmark}")

if __name__ == "__main__":
    main()
