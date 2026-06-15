# Histogram (256 bins)

Parallel histogram computation using atomic operations: **Count occurrences in 256 bins**

## Description

Computes a 256-bin histogram where each bin counts how many input elements fall into that bin's range. Uses shared memory to reduce global memory contention and atomic operations to handle concurrent updates from multiple threads. This is a fundamental operation in image processing, data analysis, and as a building block for radix sort.

## Why It Matters

Histogram computation is:
- **Atomic Operations**: Tests GPU atomic memory performance
- **Shared Memory**: Uses fast on-chip memory to reduce global contention
- **Irregular Access**: Random memory access patterns
- **Real-World Pattern**: Used in image processing, sorting, statistics

This benchmark demonstrates handling of memory contention through atomics and the effectiveness of shared memory optimization. It's particularly relevant for radix sort (histogram of digit occurrences) and image processing pipelines.

## Sarek Kernel

```ocaml
[%kernel
  fun (input : int32 vector)
      (histogram : int32 vector)
      (n : int32)
      (num_bins : int32) ->
    let open Std in
    let tid = local_thread_id in
    let gid = global_thread_id in
    
    (* Shared memory for local histogram *)
    let shared_hist = shared_array int32 256 in
    
    (* Initialize shared histogram to zero *)
    if tid < num_bins then
      shared_hist.(tid) <- 0l;
    barrier ();
    
    (* Compute local histogram with atomic add *)
    if gid < n then begin
      let bin = input.(gid) in
      atomic_add shared_hist.(bin) 1l
    end;
    barrier ();
    
    (* Merge local histogram to global *)
    if tid < num_bins then
      atomic_add histogram.(tid) shared_hist.(tid)]
```

<!-- GENERATED_CODE_TABS: histogram -->

## Key Features

- **256 bins**: Fixed size for efficient shared memory usage
- **Atomic operations**: Handle concurrent updates correctly
- **Shared memory optimization**: Reduces global memory contention
- **Two-level reduction**: Block-local then global merge

## Performance Characteristics

- **Bottleneck**: Atomic operation contention
- **Memory Pattern**: Random access to shared memory
- **Optimization**: Shared memory reduces global atomics by factor of block_size
- **Throughput**: Highly dependent on input distribution

## Algorithm Phases

1. **Initialize**: Zero out shared memory histogram (parallel)
2. **Local Count**: Each block builds histogram in shared memory (atomic)
3. **Merge**: Combine block histograms into global histogram (atomic)

## Atomic Operation Performance

The two-level approach reduces global atomic operations:
- **Without shared memory**: n atomic operations to global memory
- **With shared memory**: ~n/block_size atomic operations to global memory
- **Block size 256**: 256× reduction in global atomics

However, shared memory atomics still have contention when many threads update the same bin.

## Input Distribution Matters

Performance varies dramatically with input:
- **Uniform distribution**: Maximum contention (all bins equally)
- **Skewed distribution**: Some bins get more contention
- **Best case**: Each thread updates different bin (rare)
- **Worst case**: All threads update same bin (serialization)

## Expected Results

Modern GPUs should achieve:
- **Uniform input**: ~50-200 M elements/s
- **Highly dependent on**: Atomic operation throughput and input pattern
- **Memory bound**: Limited by shared memory and atomic performance
