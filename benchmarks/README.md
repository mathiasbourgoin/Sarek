# Sarek Benchmarks

Performance benchmarking suite for collecting data across multiple machines.

**[📊 View Interactive Results](https://mathiasbourgoin.github.io/Sarek/benchmarks/)** | **[🤝 Contribute Your Benchmarks](CONTRIBUTING.md)**

## Overview

This system is designed for **multi-machine data collection** with easy aggregation. Each benchmark run produces self-contained JSON files with full system metadata that can be combined later.

Results are published on our [interactive benchmarks page](https://mathiasbourgoin.github.io/Sarek/benchmarks/) where you can compare performance across different GPUs and backends.

## Quick Start

### Running Benchmarks

```bash
# Build benchmarks
dune build benchmarks/bench_matrix_mul.exe

# Run matrix multiplication benchmark (GPU devices only by default)
dune exec benchmarks/bench_matrix_mul.exe -- \
  --sizes 256,512,1024,2048 \
  --iterations 20 \
  --warmup 5 \
  --output results/

# Include all devices (including slow CPU backends)
dune exec benchmarks/bench_matrix_mul.exe -- \
  --sizes 256,512 \
  --all-devices \
  --output results/

# Run on current machine and save to machine-specific directory
mkdir -p results/$(hostname)
dune exec benchmarks/bench_matrix_mul.exe -- \
  --sizes 512,1024,2048 \
  --output results/$(hostname)/
```

### Aggregating Results

```bash
# Build aggregation tools
dune build benchmarks/aggregate.exe benchmarks/to_csv.exe

# Combine results from multiple machines
dune exec benchmarks/aggregate.exe -- \
  aggregated_results.json \
  results/machine1/*.json \
  results/machine2/*.json \
  results/machine3/*.json

# Convert to CSV for spreadsheet analysis
dune exec benchmarks/to_csv.exe -- aggregated_results.json results.csv

# Or convert individual runs
dune exec benchmarks/to_csv.exe -- results/machine1/cachyos_matrix_mul_naive_256_*.json
```

## Data Format

Each benchmark run produces a **self-contained JSON file** with all metadata:

```json
{
  "benchmark": {
    "name": "matrix_mul_naive",
    "timestamp": "2026-01-10T14:19:00Z",
    "git_commit": "a1b2c3d4",
    "parameters": {
      "size": 1024,
      "block_size": 256,
      "iterations": 10,
      "warmup": 5
    }
  },
  "system": {
    "hostname": "gpu-workstation-01",
    "os": "Linux 6.1.0",
    "cpu": {"model": "AMD Ryzen 9 5950X", "cores": 16},
    "memory_gb": 64,
    "devices": [
      {
        "id": 0,
        "name": "NVIDIA RTX 3090",
        "framework": "CUDA",
        "compute_capability": "8.6",
        "memory_gb": 24,
        "driver": "550.54.14"
      }
    ]
  },
  "results": [
    {
      "device_id": 0,
      "device_name": "NVIDIA RTX 3090",
      "framework": "CUDA",
      "iterations": [1.234, 1.245, 1.238, ...],
      "mean_ms": 1.239,
      "stddev_ms": 0.005,
      "throughput_gflops": 891.2
    }
  ]
}
```

## Example Workflow

### Step 1: Collect Data on Each Machine

```bash
# Machine 1 (NVIDIA GPU)
dune exec benchmarks/bench_runner.exe -- --all --output results/nvidia/

# Machine 2 (AMD GPU)
dune exec benchmarks/bench_runner.exe -- --all --output results/amd/

# Machine 3 (Apple Silicon)
dune exec benchmarks/bench_runner.exe -- --all --output results/apple/
```

### Step 2: Aggregate

```bash
# Copy results to one location and combine
python3 benchmarks/aggregate.py results/**/*.json --output paper_data.json
```

### Step 3: Generate Plots

```bash
# Speedup comparison across all devices
python3 benchmarks/plot_speedup.py paper_data.json --output plots/
```
