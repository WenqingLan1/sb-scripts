import json
from pathlib import Path

fn = Path(r"")
data = json.loads(fn.read_text())

errors = []
for metric, val in data.items():
    if "/return_code" in metric and val != 0:
        # strip off the “/return_code” suffix to get the benchmark name
        bench = metric.rsplit("/return_code", 1)[0]
        errors.append((bench, val))

if errors:
    print("Benchmarks with non-zero return codes:")
    for bench, code in errors:
        print(f"{bench} → return_code={code}")
else:
    print("All benchmarks returned 0")