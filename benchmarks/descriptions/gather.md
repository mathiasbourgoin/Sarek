# Gather

Indirect read operation: **output[i] = input[indices[i]]**

## Description

The gather operation reads elements from memory using an index array, where each thread loads from a potentially random location. This is the "read" half of indirect memory access patterns - the dual of scatter. Gather is fundamental to sparse matrix operations, graph algorithms, and data reorganization.

## Why It Matters

Gather operations are:
- **Indirect Access**: Tests random memory access performance
- **Read Coalescing**: Adjacent threads may access distant memory locations
- **Cache Performance**: Success depends heavily on cache behavior
- **Graph Algorithms**: Essential for traversing irregular data structures

This benchmark reveals how well the GPU handles irregular memory access patterns where the access pattern is determined at runtime rather than compile time.

## Sarek Kernel

```ocaml
[%kernel
  fun (input : int32 vector)
      (indices : int32 vector)
      (output : int32 vector)
      (n : int32) ->
    let open Std in
    let tid = global_thread_id in
    
    if tid < n then begin
      let idx = indices.(tid) in
      output.(tid) <- input.(idx)
    end]
```

<!-- GENERATED_CODE_TABS: gather -->

## Key Features

- **Indirect reads**: Memory address computed at runtime
- **Simple pattern**: Just load from computed address
- **No atomics needed**: Each output location written once
- **Cache-friendly**: Can benefit from spatial locality if indices are clustered

## Performance Characteristics

- **Memory Pattern**: Random reads, sequential writes
- **Bottleneck**: Cache miss rate and memory latency
- **Best case**: Indices are sequential or clustered (good cache hits)
- **Worst case**: Completely random indices (cache thrashing)

## Access Patterns

Performance varies dramatically with index distribution:
- **Sequential**: indices[i] = i → same as memcpy
- **Strided**: indices[i] = i × stride → predictable pattern
- **Random**: indices[i] = random() → worst case for caches
- **Clustered**: indices locally similar → some cache benefits

## Comparison with Regular Loads

- **Regular array access**: Predictable, coalesced, cached
- **Gather access**: Unpredictable, potentially scattered, cache-dependent
- **Performance gap**: 2-10× slower than coalesced access

## Expected Results

Modern GPUs should achieve:
- **Sequential pattern**: ~400-800 GB/s (near memory bandwidth)
- **Random pattern**: ~50-200 M elements/s
- **Highly variable**: Depends on input indices distribution
- **Cache benefits**: Can improve by 2-5× with good locality
