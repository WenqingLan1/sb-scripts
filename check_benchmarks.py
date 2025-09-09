import argparse
import json
import sys
import yaml
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description="Verify that every benchmark enabled in a SuperBench YAML appears in a JSONL results file."
    )
    parser.add_argument(
        "yaml_file",
        type=Path,
        help="Path to the SuperBench YAML"
    )
    parser.add_argument(
        "jsonl_file",
        type=Path,
        help="Path to the results-summary.jsonl"
    )
    args = parser.parse_args()

    # 1) load enabled benchmarks
    cfg = yaml.safe_load(args.yaml_file.read_text())
    enabled = [
        b.strip()
        for b in cfg.get("superbench", {}).get("enable", [])
        if isinstance(b, str) and not b.startswith("#")
    ]

    # 2) collect all keys from the JSONL
    all_keys = set()
    for line in args.jsonl_file.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
            all_keys.update(rec.keys())
        except json.JSONDecodeError:
            continue

    # 3) check presence by prefix
    missing = []
    for bench in enabled:
        prefix = bench.rstrip("/") + "/"
        found = any(k.startswith(prefix) for k in all_keys)
        status = "OK" if found else "MISSING"
        print(f"{bench:35s} : {status}")
        if not found:
            missing.append(bench)

    if missing:
        print("\nMissing benchmarks:")
        for b in missing:
            print("  -", b)
        sys.exit(1)
    else:
        print("\nAll enabled benchmarks are present.")
        sys.exit(0)

if __name__ == "__main__":
    main()
