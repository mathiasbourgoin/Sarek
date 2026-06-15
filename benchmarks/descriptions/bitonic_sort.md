# Bitonic Sort

Data-oblivious parallel sorting algorithm: **Sort n elements with O(log²n) comparison stages**

## Description

Bitonic sort is a parallel sorting algorithm that performs the same sequence of compare-and-swap operations regardless of input values. This "data-oblivious" property makes it ideal for GPUs where all threads must follow the same control flow. The algorithm builds increasingly large sorted subsequences using a series of bitonic merge passes.

## Why It Matters

Bitonic sort is:
- **Data-Oblivious**: Same comparisons regardless of input (no divergence)
- **Regular Communication**: Predictable memory access patterns
- **Highly Parallel**: All comparisons in each stage can run in parallel
- **GPU-Friendly**: No branch divergence, perfect for SIMD architectures

While not the fastest sorting algorithm for CPUs (O(n log²n) vs O(n log n)), bitonic sort's regular structure makes it competitive on GPUs. It's used in graphics pipelines, parallel databases, and as a building block for other parallel algorithms.

## Sarek Kernel

```ocaml
[%kernel
  fun (data : int32 vector)
      (j : int32)
      (k : int32)
      (n : int32) ->
    let open Std in
    let tid = global_thread_id in
    let ixj = tid lxor j in (* XOR for bitonic partner *)
    
    if tid < ixj && ixj < n then begin
      let di = data.(tid) in
      let dij = data.(ixj) in
      
      (* Determine sort direction based on k *)
      let ascending = (tid land k) = 0l in
      
      (* Compare and swap if needed *)
      if (di > dij) = ascending then begin
        data.(tid) <- dij;
        data.(ixj) <- di
      end
    end]
```

<!-- GENERATED_CODE_TABS: bitonic_sort -->

## Key Features

- **Power-of-2 sizes**: Requires n = 2^k elements
- **Log²(n) passes**: For n=1024: 55 passes, for n=65536: 136 passes
- **Compare-exchange**: Simple atomic operation at each step
- **No branch divergence**: All threads perform same operations

## Performance Characteristics

- **Complexity**: O(n log²n) comparisons
- **Memory Pattern**: Strided access with increasing distances
- **Passes**: log(n) × (log(n)+1) / 2
- **Bottleneck**: Memory bandwidth for large arrays

## Algorithm Structure

Bitonic sort operates in nested loops:
```text
for k in [2, 4, 8, ..., n]:  (outer loop - merge size)
  for j in [k/2, k/4, ..., 1]:  (inner loop - comparison distance)
    Compare-and-swap elements at distance j within bitonic sequences of size k
```

Each (k,j) pair requires one kernel launch with all threads comparing in parallel.

## Size Considerations

Default sizes are smaller than other benchmarks due to O(n log²n) complexity:
- **n=1024**: 55 passes (10 × 11 / 2)
- **n=4096**: 78 passes (12 × 13 / 2)
- **n=16384**: 105 passes (14 × 15 / 2)
- **n=65536**: 136 passes (16 × 17 / 2)

## Expected Results

Modern GPUs should achieve:
- **Small arrays (n=1024)**: ~1-5 M elements/s
- **Medium arrays (n=16K)**: ~50-200 M elements/s
- **Large arrays (n=1M)**: ~100-500 M elements/s
- **Trade-off**: More passes but better memory coalescing for larger sizes
