import json
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Check for non-zero return codes in benchmark results")
    parser.add_argument("file", help="Path to results-summary.jsonl file")
    args = parser.parse_args()

    fn = Path(args.file)
    data = json.loads(fn.read_text())

    errors = []
    for metric, val in data.items():
        if "/return_code" in metric and val != 0:
            errors.append((metric, val))

    if errors:
        print("Benchmarks with non-zero return codes:")
        for metric, code in errors:
            print(f"  • {metric}:{code}")
    else:
        print("✔ All benchmarks returned 0")

if __name__ == "__main__":
    main()
