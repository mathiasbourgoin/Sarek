# 2D Jacobi / 5-Point Stencil

## Overview

The 2D Jacobi Stencil benchmark implements a single iteration of the Jacobi method for solving partial differential equations (PDEs). It's used for heat diffusion, Laplace equation, and iterative PDE solvers.

This is a classic **5-point stencil** pattern where each cell is updated based on its four orthogonal neighbors (up, down, left, right) and itself. It's **memory bandwidth-bound** with low arithmetic intensity.

## Algorithm

5-point stencil update (Jacobi iteration):

1. For each interior cell (x, y):
   - Read 5 values: up, down, left, right, center
   - Compute average: `output[y][x] = (up + down + left + right + center) / 5.0`
   - Write result

Border cells remain unchanged (Dirichlet boundary conditions).

This is one iteration of the Jacobi method. In practice, hundreds or thousands of iterations are needed for convergence.

## Sarek Kernel Implementation

```ocaml
[%kernel
  fun (input : float32 vector)
      (output : float32 vector)
      (width : int32)
      (height : int32) ->
    let x = thread_idx_x + (block_idx_x * block_dim_x) in
    let y = thread_idx_y + (block_idx_y * block_dim_y) in
    
    if x > 0 && x < width - 1 && y > 0 && y < height - 1 then (
      let idx = (y * width) + x in
      let up = input.(((y - 1) * width) + x) in
      let down = input.(((y + 1) * width) + x) in
      let left = input.((y * width) + x - 1) in
      let right = input.((y * width) + x + 1) in
      let center = input.(idx) in
      output.(idx) <- (up +. down +. left +. right +. center) /. 5.0)]
```

**Key features:**
- **2D thread indexing**: Uses thread_idx_x/y and block_idx_x/y for natural 2D mapping
- **Boundary handling**: Explicit check for interior points only
- **5-point stencil**: Reads 5 neighbors in cross pattern
- **Simple arithmetic**: 4 adds + 1 divide = 5 FLOPs per cell
- **Read-only input**: Jacobi method reads from one array, writes to another

## Performance Characteristics

### Complexity
- **Time**: O(N) for N cells (single iteration)
- **Space**: O(N) input + O(N) output
- **Parallelism**: O(N) - one thread per interior cell

### Memory Access
- **Input**: Each thread reads 5 floats (one center + 4 neighbors)
- **Output**: Each thread writes 1 float
- **Pattern**: Neighboring threads read overlapping data (potential for cache reuse)

```text
Memory traffic per cell:
- Worst case (no cache): 5 reads + 1 write = 6 floats = 24 bytes
- Best case (perfect cache): ~1.2 reads + 1 write = 2.2 floats = 8.8 bytes
```

### Arithmetic Intensity
```text
FLOPs per cell = 4 adds + 1 divide = 5 operations
Memory traffic = 24 bytes (worst case)
AI = 5 / 24 = 0.21 FLOPs/byte
```

This is **extremely memory-bound** - one of the lowest arithmetic intensities in HPC.

### Performance Metrics
- **M cells/s**: Million cells processed per second
- **GB/s**: Memory bandwidth (N × 24 bytes / time)
- **Efficiency**: Percentage of peak memory bandwidth achieved

## Typical Results

| Size (W×H) | Cells   | Time (ms) | M cells/s | GB/s  |
|------------|---------|-----------|-----------|-------|
| 256×256    | 65K     | 0.02      | 3,250     | 78    |
| 512×512    | 262K    | 0.07      | 3,743     | 90    |
| 1024×1024  | 1.05M   | 0.28      | 3,750     | 90    |
| 2048×2048  | 4.19M   | 1.12      | 3,741     | 90    |

#### Intel Arc A770 GPU (OpenCL backend)

Performance limited by memory bandwidth (~90 GB/s ≈ 40% of theoretical peak).

## Optimization Opportunities

Memory bandwidth is the bottleneck, so optimizations focus on improving memory access:

### 1. Shared Memory Tiling
Load tiles into shared memory, reuse across iterations:

```text
For each block:
  Load (TILE_SIZE + 2)² cells into shared memory (including halo)
  For iter = 1 to iterations_per_block:
    Synchronize
    Compute stencil using shared memory (ping-pong between 2 buffers)
  Write results back to global memory
```

**Benefit:** Reduces global memory traffic by factor of iterations_per_block

### 2. Red-Black Ordering
Use checkerboard pattern (like chess board) to eliminate read-after-write hazards:

```text
Iteration 1: Update all "red" cells (even x+y)
Iteration 2: Update all "black" cells (odd x+y)
```

**Benefit:** Can update in-place, single array instead of ping-pong

### 3. Multi-Grid Methods
Use coarse-grain resolution for early iterations, refine later:

```text
Solve on coarse 64×64 grid
Interpolate to 128×128 grid
Solve on 128×128 grid
Interpolate to 256×256 grid
...
```

**Benefit:** Faster convergence, fewer total iterations

### 4. Cache Blocking for CPU
Tile the grid to fit in L2/L3 cache for CPU implementations.

## Stencil Pattern Comparison

| Pattern         | Points | Memory Reads | Arithmetic | Use Case                    |
|-----------------|--------|--------------|------------|-----------------------------|
| **5-point 2D**  | **5**  | **5**        | **5**      | **Heat, Laplace, Jacobi**   |
| 9-point 2D      | 9      | 9            | 9          | Higher-order accuracy       |
| 7-point 3D      | 7      | 7            | 7          | 3D heat/Laplace             |
| 27-point 3D     | 27     | 27           | 27         | 3D higher-order             |

The 5-point stencil is the simplest and most common for 2D problems.

## Jacobi vs. Gauss-Seidel

| Method         | Reads From | Writes To | Parallelism | Convergence |
|----------------|------------|-----------|-------------|-------------|
| **Jacobi**     | Old array  | New array | Full        | Slower      |
| Gauss-Seidel   | Same array | Same array| Limited     | Faster      |

**Jacobi** is easier to parallelize (no data races) but requires twice the memory and more iterations. This benchmark implements Jacobi.

## Applications

5-point stencil used in:

- **Heat diffusion**: Temperature distribution over time
- **Laplace equation**: Electrostatic potential, fluid flow
- **Image processing**: Iterative denoising
- **Finite difference methods**: Solving PDEs numerically
- **Multigrid solvers**: Coarse-grid correction

## Verification

CPU reference computes the same stencil in double precision:

```ocaml
let cpu_stencil_2d input output width height =
  for y = 1 to height - 2 do
    for x = 1 to width - 2 do
      let idx = (y * width) + x in
      let up = input.(((y - 1) * width) + x) in
      let down = input.(((y + 1) * width) + x) in
      let left = input.((y * width) + x - 1) in
      let right = input.((y * width) + x + 1) in
      let center = input.(idx) in
      output.(idx) <- (up +. down +. left +. right +. center) /. 5.0
    done
  done
```

Verification checks interior cells only (excluding 1-cell border) with tolerance 0.0001.

## Relation to Other Benchmarks

- **2D Convolution**: Same stencil pattern, but with different weights (kernel) and more arithmetic
- **N-Body**: Also iterative, but O(N²) all-pairs instead of O(N) local stencil
- **Matrix Multiply**: Different access pattern, much higher arithmetic intensity

The 5-point stencil is the canonical **memory-bound** benchmark for testing memory subsystem performance.
