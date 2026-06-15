# Prefix Sum (Inclusive Scan)

Parallel prefix sum using the Hillis-Steele algorithm: **output[i] = sum(input[0..i])**

## Description

Computes an inclusive prefix sum where each output element contains the sum of all input elements up to and including that position. This implementation uses the Hillis-Steele algorithm within a single block, limiting it to 256 elements but providing an elegant parallel solution with shared memory.

## Why It Matters

Prefix sum (scan) is:
- **Fundamental Primitive**: Building block for many parallel algorithms
- **Data-Dependent**: Each element depends on all previous elements
- **Communication Pattern**: Log(n) supersteps with distance-doubling
- **Shared Memory Intensive**: Tests shared memory bandwidth and synchronization

This benchmark demonstrates how to transform a sequential operation (accumulation) into a parallel one using the step-efficient Hillis-Steele algorithm. It's used in algorithms like radix sort, stream compaction, and lexical analysis.

## Sarek Kernel

```ocaml
[%kernel
  fun (input : int32 vector)
      (output : int32 vector)
      (n : int32) ->
    let open Std in
    let tid = local_thread_id in
    let shared_data = shared_array int32 256 in
    
    (* Load input to shared memory *)
    if tid < n then
      shared_data.(tid) <- input.(tid)
    else
      shared_data.(tid) <- 0l;
    
    barrier ();
    
    (* Hillis-Steele: log(n) supersteps *)
    let rec scan_step offset =
      if offset < 256 then begin
        let val_to_add =
          if tid >= offset && tid < n then
            shared_data.(tid - offset)
          else
            0l
        in
        barrier ();
        if tid < n then
          shared_data.(tid) <- shared_data.(tid) + val_to_add;
        barrier ();
        scan_step (offset * 2)
      end
    in
    scan_step 1;
    
    (* Write result *)
    if tid < n then
      output.(tid) <- shared_data.(tid)]
```

<!-- GENERATED_CODE_TABS: scan -->

## Key Features

- **Single-block**: Limited to 256 elements (block size)
- **Shared memory**: All communication happens in fast shared memory
- **Step-efficient**: O(log n) parallel steps with O(n log n) total work (sequential: O(n) work)
- **Distance-doubling**: Each superstep doubles the communication distance

## Performance Characteristics

- **Arithmetic Intensity**: High - many adds per memory transfer
- **Memory Pattern**: Shared memory access with bank conflicts possible
- **Supersteps**: log2(n) + 1 (9 steps for 256 elements)
- **Bottleneck**: Shared memory bandwidth and barrier synchronization

## Algorithm Notes

The Hillis-Steele algorithm performs log(n) parallel supersteps:
- **Step 1**: Each thread adds element at distance 1
- **Step 2**: Each thread adds element at distance 2
- **Step 4**: Each thread adds element at distance 4
- ...continues until distance >= n

For larger arrays, hierarchical approaches (multi-block scan) or work-efficient algorithms like Blelloch scan would be more appropriate.

## Expected Results

Performance is typically measured in M elements/s:
- **Single-block scan**: ~100-500 M elements/s
- **Limited by**: Shared memory bandwidth and synchronization overhead
- **Small sizes**: Extra overhead makes throughput metrics less meaningful
