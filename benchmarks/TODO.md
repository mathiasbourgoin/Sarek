# Benchmarks TODO

This document tracks the implementation status of the Sarek benchmark suite.

## Infrastructure

- [x] Common library (statistics, timing, system info)
- [x] Backend loader with conditional GPU support
- [x] JSON output format with full system metadata
- [x] CSV conversion tool
- [x] Aggregation tool for multi-machine results
- [x] Web viewer (to_web.ml) for GitHub Pages
- [x] Interactive Chart.js visualization
- [x] PR preview deployment workflow
- [ ] Plotting tools (gnuplot or OCaml-based)
- [ ] LaTeX table generation
- [ ] Unified benchmark runner (bench_runner.exe)
- [ ] CI integration for performance regression tracking

## Core Performance Benchmarks

### Memory Bandwidth
- [x] **Vector Add** - Pure memory bandwidth test
  - ✅ Implemented in bench_vector_add.ml
  - ✅ Element-wise addition: `C[i] = A[i] + B[i]`
  - ✅ Measures memory bandwidth (GB/s)
  - ✅ Default sizes: 1M, 10M, 50M, 100M elements
  - ✅ CPU baseline verification
  
- [ ] **Vector Copy** - Memory transfer baseline
  - Simple copy: `B[i] = A[i]`
  - Baseline for memory operations
  
- [ ] **STREAM Triad** - Industry standard memory benchmark
  - `A[i] = B[i] + C[i] * scalar`
  - Compare against published STREAM results

### Linear Algebra
- [x] **Matrix Multiplication (naive)** - Basic dense linear algebra
  - ✅ Implemented in bench_matrix_mul.ml
  - ✅ CPU baseline verification
  - ✅ Throughput calculation (GFLOPS)
  - ✅ Bug fixed: correct kernel arguments (m, n, k)
  - ✅ Default sizes: 256, 512, 1024, 2048 elements
  
- [ ] **Matrix Multiplication (tiled)** - Shared memory optimization
  - Use shared memory for tiling
  - Show optimization impact
  - Compare naive vs tiled performance
  - Kernel exists in tests/e2e - adapt for benchmarking
  
- [ ] **Matrix Multiplication (optimized)** - Register blocking
  - Advanced optimizations
  - Multiple tile sizes
  - Compare against cuBLAS/clBLAS
  
- [ ] **Matrix Multiplication (optimized)** - Register blocking
  - Advanced optimizations
  - Multiple tile sizes
  - Compare against cuBLAS/clBLAS

- [ ] **GEMM (SGEMM/DGEMM)** - BLAS-level routines
  - Float32 and Float64 variants
  - Batched matrix multiply
  - Matrix sizes from 128x128 to 8192x8192

### Parallel Reduction
- [x] **Sum Reduction** - Basic parallel reduction
  - ✅ Implemented in bench_reduction.ml
  - ✅ Tree-based reduction with shared memory
  - ✅ Logarithmic reduction pattern (256 -> 128 -> 64 -> ... -> 1)
  - ✅ Default sizes: 1M, 10M, 50M, 100M elements
  - ✅ Verification passing on all sizes
  - ✅ Measures memory bandwidth (GB/s)
  
- [ ] **Min/Max Reduction** - Comparison-based reduction
  - Find minimum/maximum in array
  - Compare different reduction patterns
  - Kernel exists in tests/e2e - adapt for benchmarking
  
- [ ] **Dot Product** - Combined multiply-reduce
  - `sum(A[i] * B[i])`
  - Common in scientific computing
  - Kernel exists in tests/e2e - adapt for benchmarking

### Data Movement
- [ ] **Transpose** - Memory access pattern benchmark
  - Matrix transpose with/without shared memory
  - Measure impact of coalesced access
  - Square and rectangular matrices
  - Already exists in tests/e2e - adapt for benchmarking
  
- [ ] **Scan (Prefix Sum)** - Parallel scan algorithms
  - Inclusive and exclusive scan
  - Hillis-Steele and Blelloch algorithms
  - Already exists in tests/e2e - adapt for benchmarking
  
- [ ] **Gather/Scatter** - Irregular memory access
  - Index-based array operations
  - Measure random access performance

### Sorting and Searching
- [ ] **Bitonic Sort** - Parallel sorting network
  - In-place sorting
  - Size sweep (powers of 2)
  - Already exists in tests/e2e - adapt for benchmarking
  
- [ ] **Radix Sort** - Integer sorting
  - Multi-pass digit-based sort
  - Compare against std::sort CPU baseline
  
- [ ] **Histogram** - Binning with atomics
  - Various bin counts (64, 256, 1024)
  - Atomic operations performance
  - Already exists in tests/e2e - adapt for benchmarking

## Scientific Computing

### Stencil Computations
- [ ] **2D Jacobi** - Iterative stencil
  - 5-point stencil (up, down, left, right, center)
  - Heat diffusion simulation
  - Already exists in tests/e2e (test_stencil.ml) - adapt for benchmarking
  
- [ ] **3D Stencil** - 3D heat equation
  - 7-point or 27-point stencil
  - Volume data processing
  
- [ ] **Convolution 2D** - Image filtering
  - Separable and non-separable kernels
  - Various kernel sizes (3x3, 5x5, 7x7, 11x11)
  - Already exists in tests/e2e - adapt for benchmarking

### N-Body Simulation
- [ ] **N-Body (naive)** - O(N²) particle interactions
  - All-pairs gravitational forces
  - Particle counts: 1K, 4K, 16K, 64K
  - Already exists in tests/e2e (test_nbody_ppx.ml) - adapt for benchmarking
  
- [ ] **N-Body (optimized)** - Tiled computation
  - Shared memory optimization
  - Compare performance gains

### Monte Carlo Methods
- [ ] **Pi Estimation** - Simple Monte Carlo
  - Random point sampling
  - RNG performance measurement
  
- [ ] **Random Walk** - Stochastic simulation
  - Brownian motion simulation
  - Parallel RNG streams

## Graphics and Rendering

- [ ] **Mandelbrot Set** - Embarrassingly parallel
  - Complex number iteration
  - Already exists in tests/e2e - adapt for benchmarking
  
- [ ] **Ray Tracing** - Ray-sphere intersection
  - Basic ray tracing kernel
  - Already exists in tests/e2e (test_ray_ppx.ml) - adapt for benchmarking
  
- [ ] **Path Tracing** - Monte Carlo ray tracing
  - Global illumination
  - Multiple bounces

## Machine Learning Primitives

### Activation Functions
- [ ] **ReLU** - `max(0, x)`
- [ ] **Sigmoid** - `1 / (1 + exp(-x))`
- [ ] **Tanh** - Hyperbolic tangent
- [ ] **Softmax** - Normalization with exp

### Pooling Operations
- [ ] **Max Pooling** - 2x2 and 3x3 windows
- [ ] **Average Pooling** - Window averaging

### Normalization
- [ ] **Batch Normalization** - Mean/variance normalization
- [ ] **Layer Normalization** - Per-layer normalization

### Convolution (ML-style)
- [ ] **Im2Col + GEMM** - Standard CNN convolution
- [ ] **Winograd Convolution** - Fast convolution algorithm

## Microbenchmarks

### Atomic Operations
- [ ] **Atomic Add** - `atomicAdd` performance
- [ ] **Atomic CAS** - Compare-and-swap
- [ ] **Atomic Min/Max** - Comparison atomics

### Memory Hierarchy
- [ ] **Shared Memory Bank Conflicts** - Measure bank conflict impact
- [ ] **Register Spilling** - Register pressure effects
- [ ] **Cache Behavior** - L1/L2 cache utilization

### Synchronization
- [ ] **Barrier Synchronization** - Block-level barriers
  - Already exists in tests/e2e - adapt for benchmarking
- [ ] **Warp Shuffle** - Fast intra-warp communication (if available)

### Control Flow
- [ ] **Branch Divergence** - Warp divergence cost
- [ ] **Loop Unrolling** - Impact of pragma unroll

## FFT and Signal Processing

- [ ] **FFT 1D** - Fast Fourier Transform
  - Cooley-Tukey algorithm
  - Sizes: 1024, 4096, 16K, 64K points
  
- [ ] **FFT 2D** - Image frequency domain
  - 2D signal processing
  
- [ ] **Convolution (frequency domain)** - FFT-based convolution

## Advanced Features

### Multi-GPU
- [ ] **Weak Scaling** - Constant work per GPU
- [ ] **Strong Scaling** - Fixed work across GPUs
- [ ] **Peer-to-Peer Transfer** - Direct GPU-to-GPU copy

### Mixed Precision
- [ ] **FP16 Operations** - Half precision (if supported)
- [ ] **Mixed FP32/FP16** - Mixed precision training

### Compilation and Runtime
- [ ] **Kernel Compilation Time** - Measure compilation overhead
- [ ] **Kernel Launch Overhead** - Host-device sync cost
- [ ] **Memory Transfer Bandwidth** - Host ↔ Device transfer rates

## Implementation Priority

### Phase 1 (Core Benchmarks)
1. Matrix Multiplication (tiled) - Show optimization capability
2. Vector Add - Memory bandwidth baseline  
3. Reduction (sum) - Parallel patterns
4. Transpose - Memory coalescing
5. N-Body - Real-world code

### Phase 2 (Breadth)
6. Scan (prefix sum)
7. Histogram
8. Convolution
9. Stencil 2D
10. Sort (bitonic)

### Phase 3 (Depth)
11. FFT
12. GEMM (optimized)
13. Monte Carlo
14. Ray Tracing

### Phase 4 (ML/Specialized)
15. Activation functions
16. Pooling
17. Batch normalization
18. Microbenchmarks

## Notes

- All benchmarks should include CPU baseline for verification
- Each benchmark should measure throughput (GFLOPS, GB/s, elements/sec)
- Size sweep for scalability analysis
- Multiple optimization levels where applicable (naive, tiled, optimized)
- Backend comparison (CUDA, OpenCL, Vulkan, Metal)
- Statistical analysis (mean, stddev, median, min, max)
- Self-contained JSON output for multi-machine aggregation
