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
  
- [x] **Matrix Multiplication (tiled)** - Shared memory optimization
  - ✅ Implemented in bench_matrix_mul_tiled.ml
  - ✅ Uses 16×16 tiles with shared memory
  - ✅ Shows optimization impact
  - ✅ Compare naive vs tiled performance
  - ✅ Default sizes: 128, 256, 512, 1024, 2048, 4096
  
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
  
- [x] **Min/Max Reduction** - Comparison-based reduction
  - ✅ Implemented in bench_reduction_max.ml
  - ✅ Find maximum in array with tree reduction
  - ✅ Shared memory optimization
  - ✅ Default sizes: 1M, 10M, 50M, 100M elements
  
- [x] **Dot Product** - Combined multiply-reduce
  - ✅ Implemented in bench_dot_product.ml
  - ✅ `sum(A[i] * B[i])`
  - ✅ Common in scientific computing
  - ✅ Default sizes: 1M, 10M, 50M, 100M elements

### Data Movement
- [x] **Transpose (Naive)** - Memory access pattern benchmark
  - ✅ Implemented in bench_transpose.ml
  - ✅ Naive transpose kernel (1D thread indexing)
  - ✅ Measures memory bandwidth (GB/s)
  - ✅ Default sizes: 256, 512, 1024, 2048, 4096, 8192 (NxN matrices)
  - ✅ Verification with float32-aware tolerance
  - Results @ 8192: Arc GPU 10.19 GB/s (11% of peak, strided writes hurt)
  
- [x] **Transpose (Tiled)** - Optimized with shared memory
  - ✅ Implemented in bench_transpose_tiled.ml
  - ✅ Uses 16×16 tiles with shared memory
  - ✅ +1 padding to avoid bank conflicts
  - ✅ 2D thread blocks for optimal GPU utilization
  - ✅ Default sizes: 256, 512, 1024, 2048, 4096, 8192
  - Results @ 8192: Arc GPU 32.67 GB/s (3.21× speedup over naive!)
  - Shows excellent scaling: 0.87× @ 256 → 3.21× @ 8192
  - CPU benefits even more: 5.37× speedup @ 8192
  
- [x] **Scan (Prefix Sum)** - Parallel scan algorithms
  - ✅ Implemented in bench_scan.ml
  - ✅ Hillis-Steele parallel scan algorithm
  - ✅ Power-of-2 sizes: 64, 128, 256
  
- [x] **Gather/Scatter** - Irregular memory access
  - ✅ Implemented in bench_gather_scatter.ml
  - ✅ Index-based array operations (both gather and scatter)
  - ✅ Measure random access performance
  - ✅ Default sizes: 1M, 10M, 50M elements

### Sorting and Searching
- [x] **Bitonic Sort** - Parallel sorting network
  - ✅ Implemented in bench_bitonic_sort.ml
  - ✅ In-place sorting network
  - ✅ Size sweep (powers of 2): 1024, 4096, 16384
  
- [x] **Radix Sort** - Integer sorting
  - ✅ Implemented in bench_radix_sort.ml
  - ⚠️ Known issue #101: Segmentation fault
  - Multi-pass digit-based sort
  
- [x] **Histogram** - Binning with atomics
  - ✅ Implemented in bench_histogram.ml
  - ✅ 256 bins with atomic operations
  - ✅ Default sizes: 1M, 10M, 50M elements

## Scientific Computing

### Stencil Computations
- [x] **2D Jacobi** - Iterative stencil
  - ✅ Implemented in bench_stencil_2d.ml
  - ✅ 5-point stencil (up, down, left, right, center)
  - ✅ Heat diffusion / Laplace equation
  - ✅ Default sizes: 256×256, 512×512, 1024×1024, 2048×2048
  
- [ ] **3D Stencil** - 3D heat equation
  - 7-point or 27-point stencil
  - Volume data processing
  
- [x] **Convolution 2D** - Image filtering
  - ✅ Implemented in bench_conv2d.ml
  - ✅ 3×3 box blur kernel
  - ✅ Image processing workload
  - ✅ Default sizes: 256×256, 512×512, 1024×1024, 2048×2048

### N-Body Simulation
- [x] **N-Body (naive)** - O(N²) particle interactions
  - ✅ Implemented in bench_nbody.ml
  - ✅ All-pairs gravitational forces
  - ✅ Particle counts: 512, 1024, 2048, 4096
  - ✅ High arithmetic intensity benchmark
  
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

- [x] **Mandelbrot Set** - Embarrassingly parallel
  - ✅ Implemented in bench_mandelbrot.ml
  - ✅ Complex number iteration
  - ✅ Generates visualization images
  - ✅ Default sizes: 512×512, 1024×1024, 2048×2048
  
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

## Implementation Status Summary

### ✅ Completed (19 benchmarks)
1. ✅ Vector Add - Memory bandwidth baseline
2. ✅ Vector Copy - Memory transfer baseline
3. ✅ STREAM Triad - Industry standard memory benchmark
4. ✅ Matrix Multiplication (naive) - Basic dense linear algebra
5. ✅ Matrix Multiplication (tiled) - Shared memory optimization
6. ✅ Reduction (sum) - Parallel patterns
7. ✅ Reduction (max) - Comparison-based reduction
8. ✅ Dot Product - Combined multiply-reduce
9. ✅ Transpose (naive) - Memory coalescing
10. ✅ Transpose (tiled) - Optimized transpose
11. ✅ Scan (prefix sum) - Hillis-Steele algorithm
12. ✅ Gather/Scatter - Irregular memory access
13. ✅ Bitonic Sort - Parallel sorting network
14. ✅ Radix Sort - Integer sorting (⚠️ has segfault issue #101)
15. ✅ Histogram - Atomic operations
16. ✅ 2D Jacobi Stencil - Heat diffusion
17. ✅ 2D Convolution - Image filtering
18. ✅ N-Body - Gravitational simulation
19. ✅ Mandelbrot Set - Fractal generation

### 🚧 In Progress / Planned
- [ ] FFT 1D/2D
- [ ] GEMM (optimized with register blocking)
- [ ] 3D Stencil
- [ ] N-Body (optimized with tiling)
- [ ] Ray Tracing
- [ ] Monte Carlo methods
- [ ] ML primitives (activation functions, pooling, normalization)
- [ ] Microbenchmarks (atomics, barriers, bank conflicts)

## Notes

- All benchmarks should include CPU baseline for verification
- Each benchmark should measure throughput (GFLOPS, GB/s, elements/sec)
- Size sweep for scalability analysis
- Multiple optimization levels where applicable (naive, tiled, optimized)
- Backend comparison (CUDA, OpenCL, Vulkan, Metal)
- Statistical analysis (mean, stddev, median, min, max)
- Self-contained JSON output for multi-machine aggregation
