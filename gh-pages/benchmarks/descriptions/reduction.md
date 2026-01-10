# Parallel Reduction (Sum)

Tree-based parallel reduction computing the sum of all array elements.

## Description

Reduces an array of N elements to a single sum value using a logarithmic tree pattern with shared memory. Each thread block reduces 256 elements to 1, requiring multiple kernel launches for large arrays.

## Why It Matters

Parallel reduction is a fundamental pattern in GPU computing:
- **Statistics**: Mean, variance, min/max operations
- **Physics Simulations**: Total energy, center of mass calculations
- **Machine Learning**: Loss computation, gradient norms
- **Scientific Computing**: Dot products, norms, convergence checks

This benchmark tests:
- **Shared memory performance**: Heavy use of on-chip fast memory
- **Thread synchronization**: Multiple barrier operations
- **Memory hierarchy**: Optimal use of register → shared → global memory
- **Reduction pattern efficiency**: Logarithmic complexity vs linear

## Sarek Kernel

```ocaml
[%kernel
  fun (input : float32 vector)
      (output : float32 vector)
      (n : int32) ->
    let%shared (sdata : float32) = 256l in
    let tid = thread_idx_x in
    let gid = thread_idx_x + (block_dim_x * block_idx_x) in
    
    (* Load data into shared memory *)
    let%superstep load =
      if gid < n then sdata.(tid) <- input.(gid)
      else sdata.(tid) <- 0.0
    in
    
    (* Logarithmic reduction: 128 -> 64 -> 32 -> 16 -> 8 -> 4 -> 2 -> 1 *)
    let%superstep reduce128 =
      if tid < 128l then sdata.(tid) <- sdata.(tid) +. sdata.(tid + 128l)
    in
    let%superstep reduce64 =
      if tid < 64l then sdata.(tid) <- sdata.(tid) +. sdata.(tid + 64l)
    in
    let%superstep reduce32 =
      if tid < 32l then sdata.(tid) <- sdata.(tid) +. sdata.(tid + 32l)
    in
    (* ... continues through reduce16, reduce8, reduce4, reduce2, reduce1 ... *)
    
    if tid = 0l then output.(block_idx_x) <- sdata.(0l)]
```

<!-- GENERATED_CODE_TABS: reduction -->

## Key Features

- **Shared memory**: Uses `let%shared` for on-chip fast storage
- **Supersteps**: Uses `let%superstep` for explicit synchronization barriers
- **Logarithmic complexity**: O(log N) steps instead of O(N)
- **Block-level reduction**: Each block produces one partial sum

## Performance Characteristics

- **Memory Pattern**: One global read, one global write per element
- **Shared Memory Intensive**: 8 barrier synchronizations per block
- **Bank Conflict Free**: Sequential shared memory access pattern
- **Typical Performance**: 300-600 GB/s depending on GPU
- **Multi-pass**: Large arrays require multiple kernel launches

## Algorithm Explanation

1. **Load**: Each thread loads one element from global to shared memory
2. **Reduce**: Threads cooperatively reduce shared memory in log₂(256) = 8 steps
3. **Store**: Thread 0 writes block's result to global memory
4. **Repeat**: If output > 1 element, launch kernel again on partial sums
