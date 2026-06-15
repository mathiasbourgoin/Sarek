# Radix Sort Benchmark

## Description
Implements a single pass of radix sort using 8-bit digits (256 bins). This benchmarks the core histogram and scatter kernels, which are the fundamental building blocks of radix sort.

**Note:** A full stable multi-pass radix sort requires stable scattering, which is complex to implement efficiently with global atomics. This benchmark focuses on measuring the throughput of the histogram and scatter steps themselves, effectively performing a bucket sort on the least significant byte.

## Kernels
1. **Histogram**: Computes the frequency of each 8-bit digit (0-255). Uses shared memory atomics for high performance within a workgroup, then merges to global memory.
2. **Scatter**: Scatters elements into their respective bins based on prefix-sum offsets. Uses global atomics to determine the write position for each element.

## Performance Characteristics
- **Complexity**: O(n) for a single pass.
- **Memory Access**:
  - Histogram: Coalesced reads, atomic updates (shared + global).
  - Scatter: Coalesced reads, scattered writes (random access pattern depending on data distribution).
- **Synchronization**: Requires barrier synchronization between histogram and scatter phases.
- **Data Distribution**: Performance is sensitive to data distribution (conflicts in atomics). This benchmark uses uniform random data.

## Verification
Verifies that the output array is sorted according to the 8 least significant bits of each element.

## Sarek Kernels

### Histogram Kernel
```ocaml
[%kernel
  fun (input : int32 vector)
      (histogram : int32 vector)
      (n : int32)
      (shift : int32)
      (mask : int32) ->
    let open Std in
    let open Gpu in
    let%shared (local_hist : int32) = 256l in
    let tid = thread_idx_x in
    let gid = global_thread_id in
    (* Initialize local histogram *)
    let%superstep init = if tid < 256l then local_hist.(tid) <- 0l in
    (* Count in local histogram *)
    let%superstep[@divergent] count =
      if gid < n then begin
        let value = input.(gid) in
        let digit = (value lsr shift) land mask in
        let _old = atomic_add_int32 local_hist digit 1l in
        ()
      end
    in
    (* Merge to global histogram *)
    let%superstep[@divergent] merge =
      if tid < 256l then begin
        let _old = atomic_add_global_int32 histogram tid local_hist.(tid) in
        ()
      end
    in
    ()]
```

### Scatter Kernel
```ocaml
[%kernel
  fun (input : int32 vector)
      (output : int32 vector)
      (counters : int32 vector)
      (n : int32)
      (shift : int32)
      (mask : int32) ->
    let open Std in
    let open Gpu in
    let gid = global_thread_id in
    if gid < n then begin
      let value = input.(gid) in
      let digit = (value lsr shift) land mask in
      (* Atomically get and increment counter for this digit *)
      let pos = atomic_add_global_int32 counters digit 1l in
      output.(pos) <- value
    end]
```

<!-- GENERATED_CODE_TABS: radix_sort -->
