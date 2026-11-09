# N-Body Simulation (Naive)

## Overview

The N-Body benchmark implements a naive all-pairs gravitational force calculation, a fundamental problem in computational physics and astronomy. Each particle computes the gravitational influence from all other particles, resulting in O(N²) complexity.

This is a **compute-intensive** benchmark with high arithmetic intensity (18 FLOPs per interaction), making it ideal for testing GPU compute performance rather than memory bandwidth.

## Algorithm

The naive N-body algorithm computes pairwise interactions:

1. For each particle *i*:
   - Initialize force accumulator (fx, fy, fz) = (0, 0, 0)
   - For each other particle *j* where j ≠ i:
     - Compute distance vector: **d** = **p**ⱼ - **pᵢ**
     - Compute distance squared: r² = dx² + dy² + dz²
     - Add softening factor: r² += ε (prevents singularities)
     - Compute inverse cube: inv = 1 / (r² × √r²)
     - Accumulate force: **f** += **d** × inv
   - Write final acceleration: **aᵢ** = **f**

The softening factor (ε = 1e-9) prevents division by zero when particles are very close.

## Sarek Kernel Implementation

```ocaml
[%kernel
  let module Types = struct
    type particle = {x : float32; y : float32; z : float32}
  end in
  let make_p (x : float32) (y : float32) (z : float32) : particle =
    {x; y; z}
  in
  fun (xs : float32 vector)
      (ys : float32 vector)
      (zs : float32 vector)
      (ax : float32 vector)
      (ay : float32 vector)
      (az : float32 vector)
      (n : int32)
    ->
    let tid = thread_idx_x + (block_idx_x * block_dim_x) in
    if tid < n then (
      let p = make_p xs.(tid) ys.(tid) zs.(tid) in
      let fx = mut 0.0 in
      let fy = mut 0.0 in
      let fz = mut 0.0 in
      let i = tid in
      for j = 0 to n - 1 do
        if j <> i then (
          let q = make_p xs.(j) ys.(j) zs.(j) in
          let dx = q.x -. p.x in
          let dy = q.y -. p.y in
          let dz = q.z -. p.z in
          let dist2 = (dx *. dx) +. (dy *. dy) +. (dz *. dz) +. 1e-9 in
          let inv = 1.0 /. (sqrt dist2 *. dist2) in
          fx := fx +. (dx *. inv) ;
          fy := fy +. (dy *. inv) ;
          fz := fz +. (dz *. inv))
      done ;
      ax.(tid) <- fx ;
      ay.(tid) <- fy ;
      az.(tid) <- fz)]
```

**Key features:**
- **Record types**: Uses Sarek's `module Types` to define particle structure
- **Embarrassingly parallel**: Each particle computes independently
- **Inner loop**: Iterates over all N particles (not parallelized)
- **Arithmetic intensity**: 18 FLOPs per interaction (3 subtracts, 3 multiplies for squares, 3 adds for dist2, 1 sqrt, 1 multiply and 1 divide for inv, 3 multiplies for forces, 3 adds for accumulation)

## Performance Characteristics

### Complexity
- **Time**: O(N²) interactions
- **Space**: O(N) - 6N floats input (x,y,z positions) + 3N output (ax,ay,az accelerations)
- **Parallelism**: O(N) - one thread per particle

### Memory Access
- **Input**: Each thread reads 3 floats for itself (coalesced), then N×3 floats for others (irregular)
- **Output**: Each thread writes 3 floats (coalesced)
- **Pattern**: Memory bandwidth is NOT the bottleneck - compute dominates

### Arithmetic Intensity
```text
FLOPs per particle = N × 18 operations
Total FLOPs = N² × 18
Memory traffic = 9N floats read + 3N floats write = 12N × 4 bytes = 48N bytes
AI = (18N²) / (48N) = 0.375N FLOPs/byte
```

For N=1024: AI ≈ 430 FLOPs/byte (extremely compute-intensive!)

### Performance Metrics
- **GFLOPS**: (N² × 18) / time_seconds / 1e9
- **G interactions/s**: N² / time_seconds / 1e9
- **Throughput**: Billion particle-particle interactions per second

## Typical Results

| Size | Time (ms) | GFLOPS | G interactions/s |
|------|-----------|--------|------------------|
| 512  | 0.14      | 37.4   | 1.87             |
| 1024 | 0.54      | 38.8   | 1.94             |
| 2048 | 2.15      | 39.0   | 1.95             |
| 4096 | 8.58      | 39.0   | 1.95             |

**Intel Arc A770 GPU (OpenCL backend)**

Performance scales **quadratically** with problem size, as expected for O(N²) algorithm.

## Optimization Opportunities

This is the **naive** implementation. Advanced optimizations include:

1. **Shared memory tiling**: Load tile of particles into shared memory, reuse N/tile times
2. **Symmetry**: Exploit Newton's 3rd law (Fᵢⱼ = -Fⱼᵢ) - saves 2× work but requires atomics
3. **Tree methods**: Barnes-Hut or Fast Multipole Method reduce to O(N log N)
4. **Blocked algorithms**: Improve cache locality

The naive implementation serves as a baseline for measuring raw compute throughput.

## Verification

CPU reference computes the same force calculation in double precision. Verification checks relative tolerance:

```ocaml
let check v1 v2 =
  let diff = abs_float (v1 -. v2) in
  diff <= 1e-3 || diff /. (abs_float v2 +. 1e-9) <= 1e-3
```

Tolerances account for:
- Single vs double precision differences
- Non-associative floating-point accumulation order
- Softening parameter differences
