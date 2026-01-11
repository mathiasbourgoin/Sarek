# Comprehensive Benchmark Implementation Plan

This document provides a detailed roadmap for implementing all missing benchmarks in the Sarek framework. The plan considers existing test implementations, web viewer comparison strategies, and prioritization based on impact and dependencies.

## Current Status

### ✅ Implemented (6 benchmarks)
1. **Vector Add** - Pure memory bandwidth baseline
2. **Matrix Multiplication (Naive)** - Basic dense linear algebra
3. **Sum Reduction** - Tree-based parallel reduction
4. **Transpose (Naive)** - Unoptimized memory access patterns
5. **Transpose (Tiled)** - Shared memory optimization with bank conflict avoidance
6. **Mandelbrot Set** - Embarrassingly parallel compute-bound workload

All implemented benchmarks include:
- ✅ Backend code generation (CUDA/OpenCL/Vulkan/Metal)
- ✅ Descriptions with generated code tabs
- ✅ Web viewer integration
- ✅ CPU baseline verification
- ✅ Performance data collection

## Existing E2E Test Coverage

The following kernels already exist in `sarek/tests/e2e/` and can be adapted:

| Test File | Benchmarks Available | Adaptation Effort |
|-----------|---------------------|-------------------|
| `test_matrix_mul.ml` | Naive + Tiled MatMul | **Low** - already structured like benchmarks |
| `test_reduce.ml` | Sum, Max, Dot Product | **Low** - multiple reduction variants |
| `test_scan.ml` | Inclusive Prefix Sum (Hillis-Steele) | **Medium** - needs size sweep |
| `test_sort.ml` | Bitonic Sort, Odd-Even Merge Sort | **Medium** - already has runner structure |
| `test_histogram.ml` | Histogram with atomics | **Low** - straightforward adaptation |
| `test_stencil.ml` | 1D Stencil (3-point) | **Low** - needs 2D extension |
| `test_convolution.ml` | 1D Convolution (3-point) | **Medium** - needs 2D extension |
| `test_nbody_ppx.ml` | N-Body all-pairs forces | **Low** - well-structured |
| `test_ray_ppx.ml` | Ray-sphere intersection | **Low** - rendering kernel |
| `test_mandelbrot.ml` | Mandelbrot set (already done) | ✅ |
| `test_vector_add.ml` | Vector add (already done) | ✅ |
| `test_transpose.ml` | Transpose (already done) | ✅ |
| `test_barrier_converged.ml` | Barrier synchronization | **Medium** - microbenchmark |
| `test_math_intrinsics.ml` | Sin, cos, sqrt, exp, log | **Low** - math function throughput |
| `test_bitwise_ops.ml` | Bitwise operations | **Low** - microbenchmark |

## Web Viewer Comparison Strategy

### Approach 1: Same-Page Comparisons (Recommended)
Display naive vs optimized versions on the same benchmark page:
- **Matrix Multiplication**: Naive vs Tiled vs Highly Optimized (if implemented)
- **Transpose**: Already done (Naive vs Tiled) ✅
- **N-Body**: Naive O(N²) vs Tiled (shared memory optimization)
- **Sort**: Bitonic vs Odd-Even Merge vs Radix (if implemented)
- **Reduction**: Different patterns (tree vs sequential addressing)

**Implementation**: Each variant becomes a separate benchmark with naming convention:
- `matrix_mul_naive`
- `matrix_mul_tiled`
- `matrix_mul_optimized`

Web viewer can show them side-by-side in 4-panel comparison mode.

### Approach 2: Version Selector UI
Add a "Version" dropdown alongside backend selector:
- User selects benchmark → sees available versions (Naive, Tiled, Optimized)
- Charts update to compare selected version across backends
- More complex UI but cleaner benchmark list

**Recommendation**: Use Approach 1 initially. It's simpler and works with existing infrastructure.

## Priority Classification

### P0 - Critical (Foundational benchmarks everyone expects)
Core memory, compute, and parallel patterns that define GPU performance characteristics.

### P1 - High Priority (Complete core categories)
Fill out standard benchmark categories to provide comprehensive coverage.

### P2 - Medium Priority (Specialized workloads)
Domain-specific benchmarks (ML, graphics, scientific) and advanced optimizations.

### P3 - Low Priority (Microbenchmarks and edge cases)
Fine-grained performance analysis and feature demonstrations.

---

## Detailed Implementation Roadmap

### Phase 1: Core Optimizations & Comparisons (P0)

#### 1. Matrix Multiplication (Tiled) 
- **Priority**: P0
- **Category**: Linear Algebra
- **Existing Code**: ✅ `sarek/tests/e2e/test_matrix_mul.ml` lines 90-130
- **Effort**: 🟢 Low (2-3 hours)
- **Dependencies**: None
- **Key Features**:
  - Shared memory tiling (16×16 or 32×32 tiles)
  - Supersteps with barrier synchronization
  - Compare speedup vs naive version
- **Web Comparison**: Display alongside `matrix_mul_naive` for direct speedup visualization
- **Sizes**: 128, 256, 512, 1024, 2048, 4096 (square matrices)
- **Metrics**: GFLOPS, speedup ratio vs naive
- **Description Focus**: Explain tiling strategy, memory hierarchy benefits, barrier usage

#### 2. Vector Copy
- **Priority**: P0
- **Category**: Memory Bandwidth
- **Existing Code**: ❌ None (trivial kernel)
- **Effort**: 🟢 Very Low (1 hour)
- **Dependencies**: None
- **Key Features**:
  - Simplest possible kernel: `B[i] = A[i]`
  - Pure memory bandwidth baseline
  - Compare with Vector Add to show compute overhead
- **Web Comparison**: Show with Vector Add to demonstrate bandwidth vs compute+bandwidth
- **Sizes**: 1M, 10M, 50M, 100M, 500M elements
- **Metrics**: GB/s, comparison to peak memory bandwidth
- **Description Focus**: Memory bandwidth measurement methodology

#### 3. STREAM Triad
- **Priority**: P0
- **Category**: Memory Bandwidth
- **Existing Code**: ❌ None (simple kernel)
- **Effort**: 🟢 Low (1-2 hours)
- **Dependencies**: None
- **Key Features**:
  - Industry standard: `A[i] = B[i] + C[i] * scalar`
  - Compare against published STREAM benchmarks
  - 3 reads + 1 write = 4× data movement
- **Web Comparison**: Compare with Vector Add and Vector Copy
- **Sizes**: 1M, 10M, 50M, 100M, 500M elements
- **Metrics**: GB/s, STREAM score comparison
- **Description Focus**: Relation to STREAM benchmark, memory subsystem stress test

#### 4. Max Reduction
- **Priority**: P0
- **Category**: Parallel Reduction
- **Existing Code**: ✅ `sarek/tests/e2e/test_reduce.ml` lines 30-60
- **Effort**: 🟢 Very Low (1 hour)
- **Dependencies**: Sum Reduction (for comparison)
- **Key Features**:
  - Tree-based reduction with comparison instead of addition
  - Same algorithm structure as sum reduction
- **Web Comparison**: Show with Sum Reduction to demonstrate pattern reuse
- **Sizes**: 1M, 10M, 50M, 100M elements
- **Metrics**: GB/s, elements/sec
- **Description Focus**: Reduction pattern generalization

#### 5. Dot Product
- **Priority**: P0
- **Category**: Parallel Reduction
- **Existing Code**: ✅ `sarek/tests/e2e/test_reduce.ml` lines 60-90
- **Effort**: 🟢 Low (1-2 hours)
- **Dependencies**: Sum Reduction
- **Key Features**:
  - Combined multiply-reduce: `sum(A[i] * B[i])`
  - Two memory reads + arithmetic
  - Common in scientific computing and ML
- **Web Comparison**: Show with Sum Reduction
- **Sizes**: 1M, 10M, 50M, 100M elements
- **Metrics**: GFLOPS, GB/s
- **Description Focus**: Combining map and reduce operations

**Phase 1 Total**: 5 benchmarks, ~8-10 hours effort

---

### Phase 2: Data Movement & Scanning (P1)

#### 6. Prefix Sum (Scan)
- **Priority**: P1
- **Category**: Data Movement / Parallel Primitives
- **Existing Code**: ✅ `sarek/tests/e2e/test_scan.ml`
- **Effort**: 🟡 Medium (4-6 hours)
- **Dependencies**: None
- **Key Features**:
  - Inclusive scan using Hillis-Steele algorithm
  - Fundamental parallel primitive
  - Logarithmic steps with barriers
  - Work-efficient vs step-efficient trade-offs
- **Web Comparison**: Standalone, but show scaling characteristics
- **Sizes**: 256, 512, 1K, 2K, 4K, 8K, 16K elements (powers of 2)
- **Metrics**: GB/s, elements/sec, efficiency vs sequential
- **Description Focus**: Parallel scan algorithms, applications in sorting/packing
- **Notes**: Existing test limited to 256 elements, needs multi-block extension for larger sizes

#### 7. Gather/Scatter
- **Priority**: P1
- **Category**: Data Movement
- **Existing Code**: ❌ None
- **Effort**: 🟡 Medium (3-4 hours)
- **Dependencies**: None
- **Key Features**:
  - Irregular memory access patterns
  - Index-based array operations
  - Measure random vs sequential access performance
  - Two kernels: Gather (`output[i] = input[indices[i]]`) and Scatter (`output[indices[i]] = input[i]`)
- **Web Comparison**: Compare gather vs scatter performance
- **Sizes**: 1M, 10M, 50M elements
- **Metrics**: GB/s, comparison to sequential access
- **Description Focus**: Cache behavior, memory coalescing impact

**Phase 2 Total**: 2 benchmarks, ~7-10 hours effort

---

### Phase 3: Sorting Algorithms (P1)

#### 8. Bitonic Sort
- **Priority**: P1
- **Category**: Sorting
- **Existing Code**: ✅ `sarek/tests/e2e/test_sort.ml` lines 60-120
- **Effort**: 🟡 Medium (3-4 hours)
- **Dependencies**: None
- **Key Features**:
  - Parallel sorting network
  - O(n log² n) comparisons
  - Regular communication pattern ideal for GPUs
  - In-place sorting
- **Web Comparison**: Compare with CPU std::sort and other GPU sorts
- **Sizes**: 1K, 4K, 16K, 64K, 256K, 1M elements (powers of 2)
- **Metrics**: Elements/sec, sort rate (M keys/sec)
- **Description Focus**: Sorting networks, GPU-friendly algorithms

#### 9. Radix Sort
- **Priority**: P1
- **Category**: Sorting
- **Existing Code**: ❌ None
- **Effort**: 🔴 High (8-12 hours)
- **Dependencies**: Scan (for prefix sum in counting phase)
- **Key Features**:
  - Multi-pass digit-based sort
  - O(kn) where k is number of digits
  - Fastest for integer keys
  - Requires scan primitive
- **Web Comparison**: Compare with Bitonic Sort and CPU baseline
- **Sizes**: 1K, 4K, 16K, 64K, 256K, 1M elements
- **Metrics**: M keys/sec, comparison to CPU
- **Description Focus**: LSB vs MSB radix sort, counting sort phase

**Phase 3 Total**: 2 benchmarks, ~11-16 hours effort

---

### Phase 4: Histograms & Atomics (P1)

#### 10. Histogram
- **Priority**: P1
- **Category**: Atomic Operations / Data Analysis
- **Existing Code**: ✅ `sarek/tests/e2e/test_histogram.ml`
- **Effort**: 🟢 Low (2-3 hours)
- **Dependencies**: None
- **Key Features**:
  - Atomic increments for binning
  - Shared memory optimization possible
  - Various bin counts (64, 256, 1024)
  - Measures atomic operation performance
- **Web Comparison**: Compare bin counts and atomic contention
- **Sizes**: 1M, 10M, 50M elements
- **Metrics**: Elements/sec, atomic ops/sec
- **Description Focus**: Atomic operations, contention management

#### 11. Atomic Operations Microbenchmark
- **Priority**: P2
- **Category**: Microbenchmarks
- **Existing Code**: ❌ None (simple kernels)
- **Effort**: 🟡 Medium (3-4 hours)
- **Dependencies**: None
- **Key Features**:
  - Atomic Add, CAS (Compare-And-Swap), Min, Max
  - Measure throughput with varying contention levels
  - Global vs shared memory atomics
- **Web Comparison**: Compare atomic operation types
- **Sizes**: Vary contention (1 location, 16, 256, 4K locations)
- **Metrics**: Atomic ops/sec, latency
- **Description Focus**: Atomic operation costs, contention effects

**Phase 4 Total**: 2 benchmarks, ~5-7 hours effort

---

### Phase 5: Stencil Computations (P1)

#### 12. 2D Jacobi / 5-Point Stencil
- **Priority**: P1
- **Category**: Scientific Computing / Stencil
- **Existing Code**: ⚠️ `sarek/tests/e2e/test_stencil.ml` (1D only)
- **Effort**: 🟡 Medium (5-6 hours)
- **Dependencies**: None
- **Key Features**:
  - 5-point stencil (up, down, left, right, center)
  - Heat diffusion / Laplace equation solver
  - Iterative computation with ghost cells
  - Multiple optimization levels (naive, shared memory, register blocking)
- **Web Comparison**: Compare optimization levels
- **Sizes**: 128², 256², 512², 1024², 2048² grids
- **Metrics**: Cells/sec, GB/s, iterations/sec
- **Description Focus**: Stencil patterns, PDE solvers, optimization techniques

#### 13. 2D Convolution
- **Priority**: P1
- **Category**: Image Processing / Stencil
- **Existing Code**: ⚠️ `sarek/tests/e2e/test_convolution.ml` (1D only)
- **Effort**: 🟡 Medium (4-6 hours)
- **Dependencies**: None
- **Key Features**:
  - 2D image filtering
  - Various kernel sizes (3×3, 5×5, 7×7, 11×11)
  - Separable vs non-separable kernels
  - Shared memory tiling
- **Web Comparison**: Compare kernel sizes and separable vs non-separable
- **Sizes**: 512², 1024², 2048², 4096² images
- **Metrics**: Pixels/sec, GB/s
- **Description Focus**: Image filtering, separable convolution optimization

#### 14. 3D Stencil
- **Priority**: P2
- **Category**: Scientific Computing
- **Existing Code**: ❌ None
- **Effort**: 🔴 High (6-8 hours)
- **Dependencies**: 2D Stencil
- **Key Features**:
  - 7-point or 27-point stencil
  - 3D heat equation / wave equation
  - Volume data processing
- **Web Comparison**: Compare with 2D stencil
- **Sizes**: 64³, 128³, 256³, 512³ volumes
- **Metrics**: Cells/sec, GB/s
- **Description Focus**: 3D data layout, memory access patterns

**Phase 5 Total**: 3 benchmarks, ~15-20 hours effort

---

### Phase 6: N-Body Simulation (P1)

#### 15. N-Body (Naive)
- **Priority**: P1
- **Category**: Scientific Computing / Physics
- **Existing Code**: ✅ `sarek/tests/e2e/test_nbody_ppx.ml`
- **Effort**: 🟢 Low (2-3 hours)
- **Dependencies**: None
- **Key Features**:
  - All-pairs gravitational forces O(N²)
  - Embarrassingly parallel
  - High arithmetic intensity
  - Common benchmark for HPC
- **Web Comparison**: Show with optimized version
- **Sizes**: 1K, 2K, 4K, 8K, 16K, 32K particles
- **Metrics**: GFLOPS, interactions/sec
- **Description Focus**: N-body problem, gravitational simulation

#### 16. N-Body (Tiled/Shared Memory)
- **Priority**: P2
- **Category**: Scientific Computing
- **Existing Code**: ❌ None (needs implementation)
- **Effort**: 🔴 High (6-8 hours)
- **Dependencies**: N-Body Naive
- **Key Features**:
  - Tile particles into shared memory
  - Reduce global memory traffic
  - Show speedup from tiling
- **Web Comparison**: Direct comparison with naive version
- **Sizes**: Same as naive
- **Metrics**: GFLOPS, speedup ratio
- **Description Focus**: Tiling optimization, shared memory usage

**Phase 6 Total**: 2 benchmarks, ~8-11 hours effort

---

### Phase 7: Graphics & Rendering (P2)

#### 17. Ray Tracing (Basic)
- **Priority**: P2
- **Category**: Graphics / Rendering
- **Existing Code**: ✅ `sarek/tests/e2e/test_ray_ppx.ml`
- **Effort**: 🟡 Medium (3-4 hours)
- **Dependencies**: None
- **Key Features**:
  - Ray-sphere intersection
  - Basic shading (Lambertian)
  - Multiple spheres scene
  - Measures branching and math throughput
- **Web Comparison**: Standalone or with path tracing
- **Sizes**: 256², 512², 1024², 2048² images
- **Metrics**: Rays/sec, Mpixels/sec
- **Description Focus**: Ray tracing fundamentals, GPU ray casting

#### 18. Path Tracing
- **Priority**: P3
- **Category**: Graphics / Rendering
- **Existing Code**: ❌ None
- **Effort**: 🔴 Very High (12-16 hours)
- **Dependencies**: Ray Tracing
- **Key Features**:
  - Monte Carlo ray tracing
  - Global illumination
  - Multiple bounces
  - Importance sampling
- **Web Comparison**: Compare with ray tracing
- **Sizes**: 256², 512², 1024² images, 4-16 samples/pixel
- **Metrics**: Rays/sec, samples/sec
- **Description Focus**: Path tracing algorithm, Monte Carlo methods

**Phase 7 Total**: 2 benchmarks, ~15-20 hours effort

---

### Phase 8: Math & Intrinsics (P2)

#### 19. Math Intrinsics Benchmark
- **Priority**: P2
- **Category**: Microbenchmarks / Math
- **Existing Code**: ✅ `sarek/tests/e2e/test_math_intrinsics.ml`
- **Effort**: 🟢 Low (2-3 hours)
- **Dependencies**: None
- **Key Features**:
  - Sin, cos, tan, sqrt, exp, log, pow
  - Fast vs accurate math functions
  - Measure throughput of each operation
- **Web Comparison**: Compare operations side-by-side
- **Sizes**: 1M, 10M, 100M operations
- **Metrics**: GOps/sec, latency
- **Description Focus**: Math function performance, fast vs accurate

#### 20. ReLU / Sigmoid / Tanh (ML Activations)
- **Priority**: P2
- **Category**: Machine Learning
- **Existing Code**: ❌ None (trivial kernels)
- **Effort**: 🟢 Low (2-3 hours)
- **Dependencies**: None
- **Key Features**:
  - Common ML activation functions
  - Element-wise operations
  - Branch divergence analysis (ReLU)
- **Web Comparison**: Compare activation types
- **Sizes**: 1M, 10M, 100M elements
- **Metrics**: Elements/sec, GB/s
- **Description Focus**: Activation functions in neural networks

**Phase 8 Total**: 2 benchmarks, ~4-6 hours effort

---

### Phase 9: Machine Learning Primitives (P2)

#### 21. Max Pooling
- **Priority**: P2
- **Category**: Machine Learning
- **Existing Code**: ❌ None
- **Effort**: 🟡 Medium (3-4 hours)
- **Dependencies**: None
- **Key Features**:
  - 2×2 and 3×3 pooling windows
  - Common in CNNs
  - Memory access patterns
- **Web Comparison**: Compare window sizes
- **Sizes**: 512², 1024², 2048² images
- **Metrics**: Pixels/sec
- **Description Focus**: Pooling in CNNs

#### 22. Batch Normalization
- **Priority**: P3
- **Category**: Machine Learning
- **Existing Code**: ❌ None
- **Effort**: 🔴 High (6-8 hours)
- **Dependencies**: Reduction (for mean/variance)
- **Key Features**:
  - Mean and variance normalization
  - Per-channel statistics
  - Two-pass algorithm
- **Web Comparison**: Standalone
- **Sizes**: Vary batch size and channels
- **Metrics**: Elements/sec
- **Description Focus**: Normalization techniques in deep learning

**Phase 9 Total**: 2 benchmarks, ~9-12 hours effort

---

### Phase 10: FFT & Signal Processing (P2)

#### 23. FFT 1D (Cooley-Tukey)
- **Priority**: P2
- **Category**: Signal Processing
- **Existing Code**: ❌ None
- **Effort**: 🔴 Very High (12-16 hours)
- **Dependencies**: None
- **Key Features**:
  - Fast Fourier Transform
  - Radix-2 or radix-4
  - Bit-reversal permutation
  - Complex number arithmetic
- **Web Comparison**: Compare with CPU FFT libraries
- **Sizes**: 1K, 4K, 16K, 64K, 256K, 1M points (powers of 2)
- **Metrics**: GFLOPS, samples/sec
- **Description Focus**: FFT algorithm, spectral analysis

#### 24. FFT 2D
- **Priority**: P3
- **Category**: Signal Processing
- **Existing Code**: ❌ None
- **Effort**: 🔴 Very High (10-14 hours)
- **Dependencies**: FFT 1D
- **Key Features**:
  - 2D frequency domain
  - Row-column FFT decomposition
  - Image frequency analysis
- **Web Comparison**: Compare with 1D FFT
- **Sizes**: 256², 512², 1024², 2048² images
- **Metrics**: Mpixels/sec
- **Description Focus**: 2D FFT applications

**Phase 10 Total**: 2 benchmarks, ~22-30 hours effort

---

### Phase 11: Advanced GEMM (P2)

#### 25. GEMM (Highly Optimized)
- **Priority**: P2
- **Category**: Linear Algebra
- **Existing Code**: ❌ None
- **Effort**: 🔴 Very High (16-24 hours)
- **Dependencies**: Matrix Multiplication (Tiled)
- **Key Features**:
  - Register blocking (8×8 or 16×16 tiles)
  - Vector loads/stores
  - Multi-level tiling
  - Approach cuBLAS/clBLAS performance (50-70%)
- **Web Comparison**: Compare with naive and tiled versions
- **Sizes**: 128, 256, 512, 1024, 2048, 4096, 8192
- **Metrics**: GFLOPS, % of peak
- **Description Focus**: Deep optimization techniques

#### 26. Batched GEMM
- **Priority**: P3
- **Category**: Linear Algebra
- **Existing Code**: ❌ None
- **Effort**: 🟡 Medium (4-6 hours)
- **Dependencies**: GEMM Optimized
- **Key Features**:
  - Multiple small matrix multiplications
  - Common in ML inference
  - Measure batching efficiency
- **Web Comparison**: Compare batch sizes
- **Sizes**: Various batch×matrix configurations
- **Metrics**: GFLOPS, matrices/sec
- **Description Focus**: Batched operations for ML

**Phase 11 Total**: 2 benchmarks, ~20-30 hours effort

---

### Phase 12: Microbenchmarks (P3)

#### 27. Shared Memory Bank Conflicts
- **Priority**: P3
- **Category**: Microbenchmarks
- **Existing Code**: ❌ None
- **Effort**: 🟡 Medium (3-4 hours)
- **Dependencies**: None
- **Key Features**:
  - Various stride patterns
  - Measure bank conflict impact
  - Show padding benefits
- **Web Comparison**: Compare stride patterns
- **Sizes**: Fixed shared memory sizes with varying strides
- **Metrics**: GB/s, bank conflicts detected
- **Description Focus**: GPU memory hierarchy

#### 28. Branch Divergence
- **Priority**: P3
- **Category**: Microbenchmarks
- **Existing Code**: ⚠️ Partial in `test_bitwise_ops.ml`
- **Effort**: 🟡 Medium (3-4 hours)
- **Dependencies**: None
- **Key Features**:
  - Varying divergence patterns
  - Measure warp efficiency
  - Compare if-else vs min/max
- **Web Comparison**: Compare divergence levels
- **Sizes**: Vary divergence percentage
- **Metrics**: Throughput, warp efficiency
- **Description Focus**: SIMT execution model

#### 29. Barrier Synchronization
- **Priority**: P3
- **Category**: Microbenchmarks
- **Existing Code**: ✅ `sarek/tests/e2e/test_barrier_converged.ml`
- **Effort**: 🟢 Low (2 hours)
- **Dependencies**: None
- **Key Features**:
  - Measure barrier overhead
  - Various synchronization patterns
- **Web Comparison**: Standalone
- **Sizes**: Vary number of barriers
- **Metrics**: Barriers/sec, overhead
- **Description Focus**: Synchronization primitives

**Phase 12 Total**: 3 benchmarks, ~8-10 hours effort

---

### Phase 13: Memory Transfer (P2)

#### 30. Host-Device Transfer Bandwidth
- **Priority**: P2
- **Category**: System / Memory
- **Existing Code**: ❌ None (needs framework support)
- **Effort**: 🟡 Medium (4-6 hours)
- **Dependencies**: Infrastructure changes needed
- **Key Features**:
  - Measure H2D and D2H bandwidth
  - Pinned vs pageable memory
  - Various transfer sizes
- **Web Comparison**: Compare transfer directions and memory types
- **Sizes**: 1KB to 1GB
- **Metrics**: GB/s, latency
- **Description Focus**: PCIe bandwidth, memory transfer strategies

#### 31. Kernel Launch Overhead
- **Priority**: P3
- **Category**: System / Microbenchmarks
- **Existing Code**: ❌ None
- **Effort**: 🟢 Low (2-3 hours)
- **Dependencies**: None
- **Key Features**:
  - Measure empty kernel launch time
  - CPU-GPU synchronization cost
  - Impact of kernel complexity on launch
- **Web Comparison**: Compare backends
- **Sizes**: N/A (timing measurement)
- **Metrics**: Microseconds, launches/sec
- **Description Focus**: Runtime overhead

**Phase 13 Total**: 2 benchmarks, ~6-9 hours effort

---

### Phase 14: Monte Carlo Methods (P2)

#### 32. Pi Estimation (Monte Carlo)
- **Priority**: P2
- **Category**: Monte Carlo / Simulation
- **Existing Code**: ❌ None
- **Effort**: 🟡 Medium (3-4 hours)
- **Dependencies**: None
- **Key Features**:
  - Random point sampling in unit square
  - Parallel RNG streams
  - Convergence analysis
- **Web Comparison**: Standalone
- **Sizes**: 1M, 10M, 100M, 1B samples
- **Metrics**: Samples/sec, estimation accuracy
- **Description Focus**: Monte Carlo methods, parallel RNG

#### 33. Random Walk / Brownian Motion
- **Priority**: P3
- **Category**: Monte Carlo / Simulation
- **Existing Code**: ❌ None
- **Effort**: 🟡 Medium (4-5 hours)
- **Dependencies**: Pi Estimation
- **Key Features**:
  - Stochastic particle simulation
  - Parallel random number generation
  - Statistical analysis
- **Web Comparison**: Standalone
- **Sizes**: 100K, 1M particles, 100-1000 steps
- **Metrics**: Steps/sec, particles/sec
- **Description Focus**: Stochastic simulation

**Phase 14 Total**: 2 benchmarks, ~7-9 hours effort

---

## Summary by Priority

### P0 - Critical (5 benchmarks, ~8-10 hours)
1. Matrix Multiplication (Tiled) - Show optimization impact
2. Vector Copy - Memory bandwidth baseline
3. STREAM Triad - Industry standard
4. Max Reduction - Reduction pattern variant
5. Dot Product - Map-reduce combination

**Rationale**: These complete the foundational set that every GPU benchmark suite has. They demonstrate key concepts (tiling, memory bandwidth, reduction patterns) and enable direct comparisons with published results.

### P1 - High Priority (13 benchmarks, ~65-85 hours)
6. Prefix Sum (Scan)
7. Gather/Scatter
8. Bitonic Sort
9. Radix Sort
10. Histogram
11. 2D Jacobi / 5-Point Stencil
12. 2D Convolution
13. N-Body (Naive)
14. Vector Copy
15. STREAM Triad
16. Max Reduction
17. Dot Product
18. Atomic Operations Microbenchmark

**Rationale**: These fill out standard categories (sorting, scanning, stencils, atomics) that define comprehensive GPU benchmark coverage. Most have existing test code to adapt.

### P2 - Medium Priority (16 benchmarks, ~110-150 hours)
19. 3D Stencil
20. N-Body (Tiled)
21. Ray Tracing
22. Math Intrinsics
23. ReLU/Sigmoid/Tanh
24. Max Pooling
25. FFT 1D
26. GEMM (Highly Optimized)
27. Host-Device Transfer
28. Pi Estimation
29. And more...

**Rationale**: Domain-specific workloads (graphics, ML, signal processing) and advanced optimizations. Longer development time but important for specialized users.

### P3 - Low Priority (10 benchmarks, ~50-70 hours)
30. Path Tracing
31. Batch Normalization
32. FFT 2D
33. Batched GEMM
34. Shared Memory Bank Conflicts
35. Branch Divergence
36. Barrier Synchronization
37. Kernel Launch Overhead
38. Random Walk
39. And more...

**Rationale**: Microbenchmarks, edge cases, and highly specialized workloads. Important for deep analysis but not critical for general users.

---

## Recommended Implementation Order

### Sprint 1: Core Optimizations (2-3 weeks)
Focus on P0 benchmarks that show optimization impact and complete core categories:
1. Matrix Multiplication (Tiled) - **Critical for showing Sarek's optimization capabilities**
2. Vector Copy + STREAM Triad - **Complete memory bandwidth suite**
3. Max Reduction + Dot Product - **Complete reduction patterns**

**Deliverable**: 5 new benchmarks, all with same-page comparisons to naive versions

### Sprint 2: Data Movement & Sorting (2-3 weeks)
4. Prefix Sum (Scan) - **Fundamental parallel primitive**
5. Bitonic Sort - **Existing code, GPU-friendly sorting**
6. Gather/Scatter - **Irregular memory patterns**

**Deliverable**: 3 new benchmarks, introduces sorting and scan primitives

### Sprint 3: Stencils & N-Body (3-4 weeks)
7. 2D Jacobi/5-Point Stencil - **Scientific computing representative**
8. 2D Convolution - **Image processing workload**
9. N-Body (Naive) - **Physics simulation, existing test code**
10. Histogram + Atomic Ops - **Atomic operations showcase**

**Deliverable**: 4 new benchmarks, covers stencils and atomics

### Sprint 4: Sorting & Advanced (2-3 weeks)
11. Radix Sort - **Fast integer sorting**
12. N-Body (Tiled) - **Optimization comparison**
13. Ray Tracing - **Graphics workload**

**Deliverable**: 3 new benchmarks, advanced optimizations

### Sprint 5+: Specialized Workloads (Ongoing)
- FFT, GEMM optimization, ML primitives, microbenchmarks
- Implement based on user feedback and priorities

---

## Technical Implementation Guidelines

### Kernel Adaptation from E2E Tests

When adapting existing test kernels:

1. **Extract kernel definition** from test file
2. **Add to `generate_backend_code.ml`** in appropriate section
3. **Create benchmark runner** (`bench_<name>.ml`) with:
   - Size sweep logic
   - CPU baseline for verification
   - Timing and throughput calculations
   - JSON output formatting
4. **Create description markdown** (`benchmarks/descriptions/<name>.md`) with:
   - Problem explanation
   - Algorithm description
   - Kernel code with syntax highlighting
   - `<!-- GENERATED_CODE_TABS: name -->` marker
5. **Add to dune** executable list
6. **Update `run_all_benchmarks.sh`**
7. **Generate backend code**: `dune exec ./generate_backend_code.exe`
8. **Test locally** with all backends
9. **Commit and push** - CI will verify generated code

### Comparison Benchmark Naming

Use consistent naming for version comparisons:
- `<algorithm>_naive` - Unoptimized baseline
- `<algorithm>_tiled` - Shared memory tiling
- `<algorithm>_optimized` - Highly optimized variant
- `<algorithm>_<variant>` - Algorithm variants (e.g., `scan_hillis_steele`, `scan_blelloch`)

### Web Viewer Updates

For each new benchmark:
1. Description markdown automatically appears in selector
2. Generated code tabs work automatically via marker
3. Comparison views automatically include new data
4. No JavaScript changes needed unless adding new visualization types

### Performance Metrics

Each benchmark should report:
- **Primary metric**: GFLOPS, GB/s, elements/sec, etc.
- **Problem size**: Clear indication of N
- **Verification**: Pass/fail against CPU baseline
- **Backend**: CUDA, OpenCL, Vulkan, Metal
- **Device**: Full device name
- **Timestamp**: For historical tracking

### Optimization Levels

For benchmarks with multiple optimization levels:
1. **Always start with naive version** - Establishes baseline
2. **Document each optimization** - What changed and why
3. **Measure speedup** - Show improvement factor
4. **Explain trade-offs** - Complexity vs performance

---

## Infrastructure Improvements Needed

### Short-term
- ✅ Backend code generation (Done)
- ✅ Web viewer with comparisons (Done)
- ✅ CI integration (Done)
- ⚠️ **Multi-block scan** for large arrays (needed for Phase 2)
- ⚠️ **Complex number support** (needed for FFT in Phase 10)

### Medium-term
- 📊 **Performance regression tracking** - CI job that compares against baseline
- 📈 **Historical performance graphs** - Track performance over time
- 🔍 **Roofline model integration** - Show theoretical vs actual performance
- 🎯 **Automated optimization hints** - Suggest improvements based on metrics

### Long-term
- 🌐 **Multi-GPU benchmarks** - Weak/strong scaling
- 🔢 **Mixed precision** - FP16/FP32 comparisons
- 🏆 **Leaderboard** - Community-submitted results
- 📦 **Benchmark bundles** - Predefined suites (quick, standard, comprehensive)

---

## Testing Strategy

### Verification
Every benchmark must:
1. **Include CPU baseline** - Reference implementation in pure OCaml
2. **Compare results** - GPU output must match CPU within tolerance
3. **Test multiple sizes** - Ensure correctness scales
4. **Test all backends** - CUDA, OpenCL, Vulkan, Metal

### CI Integration
- ✅ Generated code verification (implemented)
- ⏳ Benchmark correctness tests (run on CI with small sizes)
- ⏳ Performance regression detection (compare against baseline)

### Local Testing
Before committing each benchmark:
```bash
# Generate backend code
make bench-generate-code

# Run benchmark with verification
dune exec ./bench_<name>.exe -- --verify

# Run on all backends
dune exec ./bench_<name>.exe -- --benchmark

# Copy generated code to gh-pages
cp benchmarks/descriptions/generated/<name>.md gh-pages/benchmarks/descriptions/generated/

# Test web viewer locally
cd gh-pages && bundle exec jekyll serve
```

---

## Effort Estimates

### Total Effort by Priority
- **P0**: 8-10 hours (5 benchmarks)
- **P1**: 65-85 hours (13 benchmarks)
- **P2**: 110-150 hours (16 benchmarks)
- **P3**: 50-70 hours (10 benchmarks)

**Grand Total**: ~233-315 hours (~6-8 weeks full-time)

### Reality Check
With part-time work (10-15 hours/week):
- **P0 completion**: 1 week
- **P0 + P1 completion**: 5-9 weeks
- **All priorities**: 16-32 weeks (4-8 months)

### Recommended Approach
1. **Complete P0** (Sprint 1) - Get core optimization comparisons working
2. **User feedback** - See what's most valuable
3. **Prioritize P1** based on feedback - Focus on high-demand benchmarks
4. **Incremental P2/P3** - Add specialized benchmarks as needed

---

## Success Metrics

### Phase 1 Success (P0 Complete)
- ✅ 11 total benchmarks (6 existing + 5 new)
- ✅ All core categories represented (memory, compute, reduction, optimization)
- ✅ Comparison benchmarks showing optimization impact
- ✅ Professional web viewer with all visualizations working

### Phase 2 Success (P1 Complete)
- ✅ 24 total benchmarks
- ✅ All standard GPU benchmark categories covered
- ✅ Multiple optimization levels demonstrated
- ✅ Comprehensive documentation

### Long-term Success
- 🎯 40+ benchmarks covering all major workload types
- 🎯 Referenced in academic papers comparing GPU frameworks
- 🎯 Community contributions of new benchmarks
- 🎯 Performance parity with or exceeding hand-written CUDA for core benchmarks

---

## Appendix: Quick Reference

### Benchmarks by Existing Code Availability

**✅ Ready to Adapt (11)**
- Matrix Mul (Tiled), Reduce (Max/Dot), Scan, Sort (Bitonic), Histogram, Stencil (1D→2D), Convolution (1D→2D), N-Body, Ray Tracing, Math Intrinsics, Barrier Sync

**❌ Need Implementation (23+)**
- Vector Copy, STREAM Triad, Gather/Scatter, Radix Sort, 3D Stencil, N-Body (Tiled), Path Tracing, ML Activations, Pooling, Batch Norm, FFT, Optimized GEMM, Atomics, Bank Conflicts, Branch Divergence, Transfer Bandwidth, Launch Overhead, Monte Carlo, etc.

### Benchmarks by Category

| Category | Count | Priority | Existing Code |
|----------|-------|----------|---------------|
| Memory Bandwidth | 3 | P0 | Partial |
| Linear Algebra | 4 | P0-P2 | Partial |
| Reduction | 3 | P0 | Yes |
| Data Movement | 3 | P1 | Partial |
| Sorting | 2 | P1 | Partial |
| Atomics | 2 | P1-P2 | Partial |
| Stencil | 3 | P1-P2 | Partial (1D only) |
| N-Body | 2 | P1-P2 | Partial (naive) |
| Graphics | 2 | P2-P3 | Partial (ray only) |
| Math/Intrinsics | 2 | P2 | Yes |
| ML Primitives | 3 | P2-P3 | No |
| FFT | 2 | P2-P3 | No |
| Microbenchmarks | 5 | P3 | Partial |
| Monte Carlo | 2 | P2-P3 | No |
| System/Transfer | 2 | P2-P3 | No |

---

## Next Steps

1. **Review and approve this plan** with project stakeholders
2. **Set up Sprint 1 timeline** and assign resources
3. **Create GitHub issues** for each benchmark (use this as template)
4. **Start with Matrix Mul (Tiled)** - Highest impact, existing code
5. **Document progress** in this file as benchmarks complete
6. **Iterate based on feedback** - Adjust priorities as users engage

---

*Document Version*: 1.0  
*Last Updated*: 2025-01-XX  
*Status*: Ready for Implementation  
*Branch*: benchmark-expansion
