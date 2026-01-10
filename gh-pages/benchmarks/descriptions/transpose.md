# Matrix Transpose

Matrix transpose operation: **OUT[j,i] = IN[i,j]**

Compares naive vs tiled implementations to demonstrate the dramatic impact of memory access patterns.

## Description

Transpose converts row-major to column-major layout (or vice versa). The naive version accesses memory inefficiently, while the tiled version uses shared memory to achieve coalesced access patterns.

## Why It Matters

Transpose is crucial for:
- **Linear Algebra**: Matrix operations often require transposed operands
- **FFT Algorithms**: Multi-dimensional FFTs require transpose operations
- **Neural Networks**: Weight matrix operations, attention mechanisms
- **Data Layout Conversion**: Converting between row-major and column-major

More importantly, transpose is a **teaching benchmark** that demonstrates:
- **Memory access patterns matter**: Same work, 2-5× performance difference
- **Shared memory benefits**: On-chip memory enables optimization
- **Bank conflict avoidance**: Careful memory layout prevents slowdowns
- **Coalesced access importance**: Global memory must be accessed sequentially

## Naive Kernel

```ocaml
[%kernel
  fun (input : float32 vector)
      (output : float32 vector)
      (width : int32) (height : int32) ->
    let open Std in
    let tid = global_thread_id in
    if tid < width * height then begin
      let col = tid mod width in
      let row = tid / width in
      let in_idx = (row * width) + col in
      let out_idx = (col * height) + row in
      output.(out_idx) <- input.(in_idx)
    end]
```

**Problem**: Input reads are coalesced, but output writes are strided (non-coalesced), causing poor memory performance.

## Tiled Kernel

```ocaml
[%kernel
  fun (input : float32 vector)
      (output : float32 vector)
      (width : int32) (height : int32) ->
    (* 16x17 tile with +1 padding to avoid bank conflicts *)
    let%shared (tile : float32) = 272l in
    let tile_size = 16l in
    let tx = thread_idx_x in
    let ty = thread_idx_y in
    let block_col = block_idx_x * tile_size in
    let block_row = block_idx_y * tile_size in
    
    (* Load tile from input (coalesced reads) *)
    let%superstep load =
      let read_col = block_col + tx in
      let read_row = block_row + ty in
      if read_row < height && read_col < width then
        tile.((ty * (tile_size + 1l)) + tx) <-
          input.((read_row * width) + read_col)
    in
    
    (* Write transposed tile (coalesced writes) *)
    let%superstep store =
      let write_col = block_row + tx in
      let write_row = block_col + ty in
      if write_row < height && write_col < width then
        output.((write_row * width) + write_col) <-
          tile.((tx * (tile_size + 1l)) + ty)
    in]
```

**Solution**: 
1. Load 16×16 tile into shared memory (coalesced reads)
2. Synchronize threads
3. Write transposed tile to output (coalesced writes)
4. Use 16×**17** layout to avoid bank conflicts

## Key Optimizations

- **Tiling**: Process 16×16 blocks instead of individual elements
- **Shared Memory**: Use fast on-chip memory as intermediate buffer
- **Bank Conflict Avoidance**: +1 padding ensures no two threads access same bank
- **Coalesced Access**: Both reads and writes are sequential

## Performance Characteristics

| Metric | Naive | Tiled |
|--------|-------|-------|
| Memory Pattern | Strided writes | Coalesced reads & writes |
| Shared Memory | None | 272 floats per block |
| Bank Conflicts | N/A | Zero (with padding) |
| Typical Performance | 100-200 GB/s | 400-700 GB/s |
| **Speedup** | **1×** | **2-5×** |

## Expected Results

The tiled version should be **2-5× faster** than naive, demonstrating that memory access patterns can have a bigger impact than algorithmic improvements.
