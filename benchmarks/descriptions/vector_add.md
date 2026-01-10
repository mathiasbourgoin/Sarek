# Vector Addition

Element-wise vector addition: **C[i] = A[i] + B[i]**

## Description

The simplest parallel operation - each thread adds one element from vector A to the corresponding element in vector B and stores the result in vector C.

## Why It Matters

Vector addition is:
- **Memory-Bandwidth Bound**: Tests peak memory bandwidth of the GPU
- **Fundamental Pattern**: Used in countless algorithms (linear algebra, physics, ML)
- **Baseline Benchmark**: Establishes the lower bound for memory-intensive operations
- **Roofline Model**: Helps determine the memory bandwidth ceiling for other algorithms

This benchmark directly measures how fast data can be moved between GPU memory and compute units. The computational work is trivial (one addition), so performance is entirely limited by memory bandwidth.

## Sarek Kernel

```ocaml
[%kernel
  fun (a : float32 vector)
      (b : float32 vector)
      (c : float32 vector)
      (n : int32) ->
    let open Std in
    let tid = global_thread_id in
    if tid < n then
      c.(tid) <- a.(tid) +. b.(tid)]
```

<!-- GENERATED_CODE_TABS: vector_add -->

## Key Features

- **Ultra-simple**: Just 4 lines of actual computation
- **Perfectly parallel**: No dependencies between threads
- **Coalesced access**: All memory accesses are sequential and aligned
- **Memory bound**: Performance limited by bandwidth, not compute

## Performance Characteristics

- **Arithmetic Intensity**: 1 FLOP per 12 bytes transferred (read A, read B, write C)
- **Memory Pattern**: Perfectly coalesced sequential access
- **Typical Performance**: 200-900 GB/s depending on GPU
- **Bottleneck**: DRAM bandwidth, not compute units

## Expected Results

Modern GPUs should achieve:
- **NVIDIA RTX 4090**: ~800 GB/s
- **AMD RX 7900 XTX**: ~700 GB/s  
- **Intel Arc A770**: ~500 GB/s
- **Apple M3 Max**: ~400 GB/s
