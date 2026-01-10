# Contributing Benchmark Results

We welcome benchmark contributions from the community! Your results help build a comprehensive performance database for Sarek across different hardware.

## Quick Start (Recommended)

### 1. Run All Benchmarks

The easiest way to contribute is to run all benchmarks at once:

```bash
# Clone the repository
git clone https://github.com/mathiasbourgoin/SPOC.git
cd SPOC

# Install dependencies
opam install --deps-only -y .

# Run ALL benchmarks (recommended!)
eval $(opam env)
./benchmarks/run_all_benchmarks.sh

# Or using make:
make benchmarks
```

This single command will:
- Build all benchmark executables
- Run 5 benchmarks: matrix_mul, vector_add, reduction, transpose, transpose_tiled
- Save results to `results/run_TIMESTAMP/`
- Update the web viewer data automatically

**Estimated time:** 5-15 minutes depending on your hardware.

### 2. Submit Your Results

```bash
# Create a new branch
git checkout -b benchmark-results-$(hostname)

# Add your result files (from the timestamped directory)
git add results/run_*/
git add gh-pages/benchmarks/data/latest.json

# Commit with descriptive message
git commit -m "Add benchmark results for $(hostname)

Hardware:
- GPU: [Your GPU model, e.g., NVIDIA RTX 4090]
- CPU: [Your CPU model, e.g., AMD Ryzen 9 7950X]
- RAM: [Memory size, e.g., 64GB DDR5-6000]
- Backend: [CUDA/OpenCL/Vulkan/Metal]
- OS: [e.g., Ubuntu 24.04, Windows 11, macOS 14]"

# Push and create PR
git push origin benchmark-results-$(hostname)
# Then create a Pull Request on GitHub
```

**🔍 Preview Your Results:** Once you create the PR, a GitHub Action will automatically:
1. Build a preview deployment of the benchmarks page with your results
2. Post a comment on your PR with a preview link
3. Update the preview when you push new commits

The preview URL will look like:
```
https://mathiasbourgoin.github.io/Sarek/preview/pr-XXX/benchmarks/
```

This lets you verify your results look correct before maintainers review the PR!

## What We're Looking For

### High Priority Hardware

- **NVIDIA GPUs:**
  - Consumer: RTX 4090, 4080, 4070, 3090, 3080, 3070, 3060
  - Professional: A100, A6000, A5000, A4000
  - Server: H100, H200, A100 80GB
  
- **AMD GPUs:**
  - Consumer: RX 7900 XTX, 7900 XT, 7800 XT, 7700 XT
  - Professional: Radeon Pro W7900, W7800
  - Server: MI300X, MI250X, MI210
  
- **Intel GPUs:**
  - Arc: A770 16GB, A770 8GB, A750, A380
  - Data Center: Max Series (Ponte Vecchio)
  
- **Apple Silicon:**
  - M1, M1 Pro, M1 Max, M1 Ultra
  - M2, M2 Pro, M2 Max, M2 Ultra
  - M3, M3 Pro, M3 Max
  - M4, M4 Pro, M4 Max
  
- **Mobile/Embedded:**
  - Qualcomm Adreno
  - ARM Mali
  - PowerVR

### Backend Coverage

We're interested in results from all backends:
- **CUDA** (NVIDIA GPUs)
- **OpenCL** (cross-platform)
- **Vulkan** (modern cross-platform)
- **Metal** (Apple devices)

## Guidelines

### Benchmark Settings

- **Use default settings** unless you have a specific reason not to
- **Run multiple iterations** (default: 20) for statistical accuracy
- **Include warmup runs** (default: 5) to exclude cold-start effects
- **Run on idle system** to avoid interference from other processes

### What to Submit

**DO submit:**
- ✅ JSON result files from `benchmarks/results/`
- ✅ Results from different backends on the same hardware
- ✅ Results from different GPUs/systems
- ✅ Results from interesting hardware (new GPUs, mobile, embedded)

**DON'T submit:**
- ❌ Modified benchmark code (unless fixing a bug - separate PR)
- ❌ Results from virtual machines (unless explicitly noted)
- ❌ Results from overclocked systems (unless explicitly noted)
- ❌ Incomplete result files

### File Naming

Result files are automatically named:
```
{hostname}_{benchmark}_{size}_{timestamp}.json
```

For example:
```
workstation_matrix_mul_naive_1024_2026-01-10T14-33-55.json
```

If you want to submit results with a more descriptive name, you can rename them:
```
rtx4090_matrix_mul_naive_1024_2026-01-10T14-33-55.json
m3max_matrix_mul_naive_1024_2026-01-10T14-33-55.json
```

## Viewing Results

Your results will appear on the [benchmarks page](https://mathiasbourgoin.github.io/Sarek/benchmarks/) after your PR is merged.

The page provides interactive charts where you can:
- Compare different backends
- Filter by device/GPU
- Toggle between benchmarks
- View detailed system information

## Advanced: Running Full Benchmark Suite

Once more benchmarks are implemented, you'll be able to run the full suite:

```bash
# Run all benchmarks (coming soon)
dune exec benchmarks/bench_runner.exe -- \
  --all \
  --output benchmarks/results/

# Run specific benchmarks
dune exec benchmarks/bench_runner.exe -- \
  --benchmark matrix_mul,vector_add,reduction \
  --output benchmarks/results/
```

## Troubleshooting

### No GPU detected

If benchmarks report "No devices available", check:

1. **Backend libraries installed:**
   ```bash
   # For CUDA
   opam install sarek-cuda
   
   # For OpenCL
   opam install sarek-opencl
   
   # For Vulkan
   opam install sarek-vulkan
   
   # For Metal (macOS only)
   opam install sarek-metal
   ```

2. **Drivers installed:**
   - NVIDIA: Install CUDA Toolkit
   - AMD: Install ROCm or Adrenalin drivers
   - Intel: Install oneAPI or Arc drivers
   - Apple: Metal is included with macOS

3. **Permissions:**
   - Some systems require user to be in `video` or `render` group
   - Check with `groups` command

### Benchmark crashes

If benchmarks crash:

1. **Try smaller sizes first:**
   ```bash
   dune exec benchmarks/bench_matrix_mul.exe -- --sizes 64,128,256
   ```

2. **Check memory:**
   - Large matrix sizes require significant GPU memory
   - 4096×4096 FP32 matrix = 64MB per matrix (192MB total for A, B, C)

3. **Report the issue:**
   - Include error messages
   - Include system info (GPU, driver version, OS)
   - Create a GitHub issue

## Questions?

- Open an issue on [GitHub](https://github.com/mathiasbourgoin/SPOC/issues)
- Check the [documentation](https://mathiasbourgoin.github.io/Sarek/)
- Ask on [discussions](https://github.com/mathiasbourgoin/SPOC/discussions)

## Thank You!

Your contributions help make Sarek better for everyone. Thank you for sharing your benchmark results! 🚀
