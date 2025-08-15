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
