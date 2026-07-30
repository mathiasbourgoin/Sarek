# Matrix Multiplication (Tiled)

**Category**: Linear Algebra  
**Optimization Level**: Shared Memory Tiling  
**Demonstrates**: Memory hierarchy optimization, barrier synchronization, performance improvement

## Overview

This benchmark implements **tiled matrix multiplication** using shared memory to reduce global memory traffic. It demonstrates a key GPU optimization technique: loading data into fast shared memory and reusing it across multiple threads.

Compared to the naive approach, tiling provides:
- **3-10× speedup** on typical GPUs
- Reduced global memory bandwidth requirements
- Better cache utilization

## Algorithm

The tiled algorithm divides matrices into **16×16 tiles** and processes them in shared memory:

1. **Load Phase**: Each thread block cooperatively loads one tile of A and one tile of B into shared memory
2. **Barrier Sync**: Wait for all threads to finish loading
3. **Compute Phase**: Each thread computes its partial sum using the tiles in shared memory
4. **Repeat**: Move to next tile pair until all tiles are processed

This reduces global memory accesses from O(N³) to O(N²).

## Matrix Multiplication (Tiled) Kernel

```ocaml
let matmul_tiled_kernel =
  [%kernel
    fun (a : float32 vector)
        (b : float32 vector)
        (c : float32 vector)
        (m : int32)
        (n : int32)
        (k : int32) ->
      (* Allocate shared memory tiles: 16×16 elements each *)
      let%shared (tile_a : float32) = 256l in
      let%shared (tile_b : float32) = 256l in
      
      (* Get thread position within block and grid *)
      let tx = thread_idx_x in
      let ty = thread_idx_y in
      let row = ty + (block_dim_y * block_idx_y) in
      let col = tx + (block_dim_x * block_idx_x) in
      
      (* Tiling configuration *)
      let tile_size = 16l in
      let num_tiles = (k + tile_size - 1l) / tile_size in
      let sum = mut 0.0 in
      
      (* Process tiles sequentially *)
      for t = 0 to num_tiles - 1l do
        (* Superstep 1: Load tile from matrix A *)
        let%superstep load_a =
          let a_col = (t * tile_size) + tx in
          if row < m && a_col < k then
            tile_a.((ty * tile_size) + tx) <- a.((row * k) + a_col)
          else 
            tile_a.((ty * tile_size) + tx) <- 0.0
        in
        
        (* Superstep 2: Load tile from matrix B *)
        let%superstep load_b =
          let b_row = (t * tile_size) + ty in
          if b_row < k && col < n then
            tile_b.((ty * tile_size) + tx) <- b.((b_row * n) + col)
          else 
            tile_b.((ty * tile_size) + tx) <- 0.0
        in
        
        (* Superstep 3: Compute partial sum using tiles *)
        let%superstep _compute =
          for i = 0 to tile_size - 1l do
            sum :=
              sum
              +. (tile_a.((ty * tile_size) + i) *. tile_b.((i * tile_size) + tx))
          done
        in
        ()
      done ;
      
      (* Write final result *)
      if row < m && col < n then 
        c.((row * n) + col) <- sum]
```

## Key Optimizations

### 1. Shared Memory Tiling
- Each 16×16 thread block uses **2KB of shared memory** (2 tiles × 256 floats × 4 bytes)
- Shared memory is **10-100× faster** than global memory
- Data reuse: Each tile element is used 16 times

### 2. Coalesced Memory Access
- Threads load consecutive elements (coalesced access pattern)
- Maximizes memory bus utilization

### 3. Barrier Synchronization
- `let%superstep` ensures all threads load data before computation
- Prevents race conditions on shared memory

### 4. 2D Thread Blocks
- Uses `dims2d 16 16` for natural matrix indexing
- Each thread computes one output element

## Performance Characteristics

**Arithmetic Intensity**: High (O(N) reuse per memory access)  
**Memory Pattern**: Coalesced reads, reused data  
**Parallelism**: O(N²) threads  
**Bottleneck**: Computation (with tiling), not memory

**Expected Speedup vs Naive**:
- Small matrices (128-512): 1.5-3×
- Medium matrices (1024-2048): 3-5×
- Large matrices (4096+): 5-10×

Speedup increases with matrix size because:
- Fixed shared memory overhead amortized
- Better occupancy on GPU
- More opportunities for data reuse

## Comparison with Naive

| Aspect | Naive | Tiled |
|--------|-------|-------|
| Global Memory Accesses | O(N³) | O(N²) |
| Memory Traffic | ~2N³ floats | ~2N² floats |
| Shared Memory | None | 2KB per block |
| Synchronization | None | 3 barriers per tile |
| Complexity | Simple | Moderate |
| Performance | Baseline | 3-10× faster |

## Use Cases

- **Linear algebra libraries** (BLAS level 3)
- **Machine learning** (neural network training/inference)
- **Scientific computing** (finite element methods, simulations)
- **Computer graphics** (transformations, projections)

## Further Optimizations

Advanced GEMM implementations add:
- **Register blocking** (8×8 or 16×16 register tiles)
- **Multi-level tiling** (registers → shared → L2 → global)
- **Prefetching** (overlap memory and compute)
- **Vector instructions** (use float4/vector loads)
- **Warp specialization** (separate load/compute warps)

High-performance libraries like cuBLAS achieve **80-90% of theoretical peak**.

---

<!-- GENERATED_CODE_TABS: matrix_mul_tiled -->

## References

- Volkov & Demmel, "Benchmarking GPUs to Tune Dense Linear Algebra" (2008)
- NVIDIA, "CUDA C++ Best Practices Guide"
- Hong & Kim, "An Analytical Model for a GPU Architecture with Memory-level and Thread-level Parallelism Awareness"
