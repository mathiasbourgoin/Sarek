# Sarek Benchmarks

Performance benchmarking suite for collecting data across multiple machines.

**[📊 View Interactive Results](https://mathiasbourgoin.github.io/Sarek/benchmarks/)** | **[🤝 Contribute Your Benchmarks](CONTRIBUTING.md)**

## Overview

This system is designed for **multi-machine data collection** with easy aggregation. Each benchmark run produces self-contained JSON files with full system metadata that can be combined later.

Results are published on our [interactive benchmarks page](https://mathiasbourgoin.github.io/Sarek/benchmarks/) where you can compare performance across different GPUs and backends.

## Quick Start

### Running All Benchmarks (Recommended)

The easiest way to run all benchmarks and update the web viewer:

```bash
# Run all benchmarks and update web data
make benchmarks

# Or directly:
./benchmarks/run_all_benchmarks.sh

# Results saved to results/run_TIMESTAMP/ and web data updated
```

This script will:
1. Build all benchmark executables
2. Run all 6 benchmarks with default sizes
3. Generate timestamped result files (including PPM images for Mandelbrot)
4. Update `gh-pages/benchmarks/data/latest.json`
5. Provide instructions for committing results

## Available Benchmarks

| Benchmark | Description | Metric | What It Tests |
|-----------|-------------|--------|---------------|
| **Matrix Multiplication** | Dense matrix multiply (naive) | GFLOPS | Compute-bound arithmetic intensity |
| **Vector Addition** | Element-wise addition | GB/s | Memory bandwidth ceiling |
| **Parallel Reduction** | Sum all array elements | GB/s | Shared memory & synchronization |
| **Transpose (Naive)** | Matrix transpose | GB/s | Memory access patterns baseline |
| **Transpose (Tiled)** | Optimized with shared memory | GB/s | Memory optimization impact (2-5× speedup) |
| **Mandelbrot Set** | Fractal generation | Mpixels/s | Arithmetic intensity & branch divergence |

### Running Individual Benchmarks

```bash
# Build all benchmarks
dune build benchmarks/bench_matrix_mul.exe benchmarks/bench_vector_add.exe \
           benchmarks/bench_reduction.exe benchmarks/bench_transpose.exe \
           benchmarks/bench_transpose_tiled.exe benchmarks/bench_mandelbrot.exe

# Run matrix multiplication benchmark (default: 256, 512, 1024, 2048 elements)
dune exec benchmarks/bench_matrix_mul.exe

# Run vector addition benchmark (default: 1M, 10M, 50M, 100M elements)
dune exec benchmarks/bench_vector_add.exe

# Run reduction benchmark (default: 1M, 10M, 50M, 100M elements)
dune exec benchmarks/bench_reduction.exe

# Run transpose benchmarks (default: 256, 512, 1024, 2048, 4096, 8192 - NxN matrices)
dune exec benchmarks/bench_transpose.exe          # Naive version
dune exec benchmarks/bench_transpose_tiled.exe    # Optimized with shared memory

# Run Mandelbrot benchmark (default: 512, 1024, 2048, 4096 - square images)
dune exec benchmarks/bench_mandelbrot.exe         # Generates PPM images
dune exec benchmarks/bench_mandelbrot.exe -- --no-images  # Skip image generation

# Custom sizes and iterations
dune exec benchmarks/bench_matrix_mul.exe -- \
  --sizes 512,1024,2048,4096 \
  --iterations 20 \
  --warmup 5 \
  --output benchmarks/results/

# Run on the current machine, into the directory the tooling actually reads.
#
# There is no machine directory to create and no label to type. The benchmark
# DERIVES its own label (<os>-<gpu-vendor>, e.g. linux-nvidia) and that label
# already prefixes every output filename, so results from different machines
# stay apart in one flat directory. This is also the layout the deduplicator
# scans -- it reads benchmarks/results/ directly and does not descend into
# subdirectories, so files written to results/<machine>/ are invisible to it.
#
# SAREK_BENCH_MACHINE overrides the label for a local run; it is refused if it
# equals the hostname, because a machine label is published and a hostname must
# not be. Tracked results must still carry a derived <os>-<vendor> label.
dune exec benchmarks/bench_matrix_mul.exe -- --output benchmarks/results/
dune exec benchmarks/bench_vector_add.exe -- --output benchmarks/results/
dune exec benchmarks/bench_reduction.exe -- --output benchmarks/results/
dune exec benchmarks/bench_transpose.exe -- --output benchmarks/results/
dune exec benchmarks/bench_transpose_tiled.exe -- --output benchmarks/results/
dune exec benchmarks/bench_mandelbrot.exe -- --output benchmarks/results/
```

### Publishing Results to Web Viewer

```bash
# Build web conversion tool
dune build benchmarks/to_web.exe

# Convert benchmark results to web format
dune exec benchmarks/to_web.exe -- \
  gh-pages/benchmarks/data/latest.json \
  benchmarks/results/*.json

# The web viewer will automatically display the results at:
# https://mathiasbourgoin.github.io/Sarek/benchmarks/
```

### Aggregating Results

```bash
# Build aggregation tools
dune build benchmarks/aggregate.exe benchmarks/to_csv.exe

# Combine results from multiple machines. They share one flat directory; the
# derived <os>-<vendor> label prefixing each filename is what keeps them apart.
dune exec benchmarks/aggregate.exe -- \
  aggregated_results.json \
  benchmarks/results/*.json

# Or restrict to particular machines by their label prefix
dune exec benchmarks/aggregate.exe -- \
  aggregated_results.json \
  benchmarks/results/linux-nvidia_*.json \
  benchmarks/results/darwin-apple_*.json

# Convert to CSV for spreadsheet analysis
dune exec benchmarks/to_csv.exe -- aggregated_results.json results.csv

# Or convert individual runs
dune exec benchmarks/to_csv.exe -- \
  benchmarks/results/linux-intel_matrix_mul_naive_256_*.json
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
    "machine": "linux-nvidia",
    "os": "Linux",
    "cpu": {"model": "AMD Ryzen 9 5950X", "cores": 16},
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

### Step 1: Collect Data on Each Machine (Easy Method)

```bash
# Machine 1 (NVIDIA GPU) - Run all benchmarks at once!
./benchmarks/run_all_benchmarks.sh
# Results saved to: results/run_TIMESTAMP/

# Machine 2 (AMD GPU)
./benchmarks/run_all_benchmarks.sh
# Results saved to: results/run_TIMESTAMP/

# Machine 3 (Apple Silicon)
./benchmarks/run_all_benchmarks.sh
# Results saved to: results/run_TIMESTAMP/
```

### Step 1 Alternative: Run Benchmarks Individually

The command is the SAME on every machine -- there is no per-machine directory
to name. Each benchmark derives its own `<os>-<gpu-vendor>` label and prefixes
its output filenames with it, so one flat `benchmarks/results/` holds all three
machines without collisions, and the deduplicator (which does not descend into
subdirectories) can see them.

```bash
# On each machine in turn -- machine 1 (NVIDIA), 2 (AMD), 3 (Apple Silicon):
dune exec benchmarks/bench_matrix_mul.exe -- --output benchmarks/results/
dune exec benchmarks/bench_vector_add.exe -- --output benchmarks/results/
dune exec benchmarks/bench_reduction.exe -- --output benchmarks/results/
dune exec benchmarks/bench_transpose.exe -- --output benchmarks/results/
dune exec benchmarks/bench_transpose_tiled.exe -- --output benchmarks/results/

# If two machines share an os AND a GPU vendor their runs merge under one
# label. SAREK_BENCH_MACHINE overrides the label for a local run (it is refused
# if it equals the hostname). Note that results COMMITTED to this repository
# must still be named after a derived <os>-<vendor> label --
# scripts/check-no-machine-identifiers.sh enforces that on tracked paths -- so
# use the override for local comparison, not to introduce a new public label.
```

### Step 2: Aggregate and Publish

```bash
# Combine all results
dune exec benchmarks/aggregate.exe -- \
  benchmarks/results/aggregated.json \
  benchmarks/results/*.json

# Convert to web format for GitHub Pages
dune exec benchmarks/to_web.exe -- \
  gh-pages/benchmarks/data/latest.json \
  benchmarks/results/*.json

# Commit and push to publish
git add gh-pages/benchmarks/data/latest.json
git commit -m "Update benchmark results"
git push
```

### Step 3: View Results

Visit the interactive web viewer at:
https://mathiasbourgoin.github.io/Sarek/benchmarks/

Select benchmarks from dropdown, filter by backend, and compare performance across devices.
