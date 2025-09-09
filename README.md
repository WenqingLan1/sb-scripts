# SuperBench Benchmark Analysis Scripts

This repository contains two AI generated Python scripts for analyzing benchmark results from JSONL files.

## Scripts

### 1. `compare.py` - Compare Benchmark Results

Compares two benchmark result files and shows differences in metrics and values.

**Usage:**
```bash
python compare.py <file1> <file2>
```

**Example:**
```bash
# compares file2 over file1. results are signed.
python compare.py ./results-summary-1.jsonl ./results-summary-2.jsonl
```

**Output:**
- Missing metrics between files
- Value differences above 5% threshold
- Benchmarks that exist only in one file

**Sample Output:**
```
cpu-memory-bw-latency - Value differences:
  - mem_bandwidth_matrix_numa_0_1_bw: +5.2%
      file1: 7056.89
      file2: 7424.32

gpu-copy-bw:perf - Missing metrics:
  Only in file2:
    - new_metric_name
```

### 2. `check_return_code.py` - Check for Failed Benchmarks

Checks a single benchmark file for any non-zero return codes (indicating failures).

**Usage:**
```bash
python check_return_code.py <file>
```

**Example:**
```bash
python check_return_code.py ./results-summary.jsonl
```

**Output:**
- Lists all benchmarks with non-zero return codes
- Shows success message if all benchmarks passed

**Sample Output:**
```
Benchmarks with non-zero return codes:
  • cudnn-function/return_code:124
  • gemm-flops/return_code:124
```
### 3. `check_benchmarks.py` - Verify Enabled Benchmarks in Results

Checks that every benchmark listed under `superbench.enable` in a SuperBench YAML config appears in a JSONL results file.

**Usage:**
```bash
python check_benchmarks.py <config.yaml> <results-summary.jsonl>
```

**Example:**
```bash
python check_benchmarks.py gb200.yaml results-summary.jsonl
```

**Output:**
- Prints each enabled benchmark with `OK` if any key in the JSONL starts with `<benchmark>/`, or `MISSING` otherwise  
- Exits with status `0` if all are present, or `1` if any benchmarks are missing

**Sample Output:**
```
kernel-launch                       : OK
gemm-flops                          : OK
…
computation-communication-overlap   : MISSING
…
All enabled benchmarks are present.  # or lists
```
