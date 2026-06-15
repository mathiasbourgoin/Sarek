# Scatter

Indirect write operation: **output[indices[i]] = input[i]**

## Description

The scatter operation writes elements to memory using an index array, where each thread stores to a potentially random location. This is the "write" half of indirect memory access patterns - the dual of gather. Unlike gather, scatter can have write conflicts when multiple threads try to write to the same location.

## Why It Matters

Scatter operations are:
- **Indirect Writes**: Tests random memory write performance
- **Write Conflicts**: Multiple threads may target same location
- **Atomic Required**: Must handle conflicts correctly (or accept race conditions)
- **Data Reorganization**: Essential for bucketing, partitioning, and sorting

This benchmark demonstrates the challenges of irregular write patterns where destination addresses are computed at runtime. Write conflicts make scatter more complex than gather.

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
      output.(idx) <- input.(tid)
    end]
```

<!-- GENERATED_CODE_TABS: scatter -->

## Key Features

- **Indirect writes**: Memory address computed at runtime
- **Write conflicts**: Multiple threads may write to same location
- **Race conditions**: Last write wins (non-deterministic without atomics)
- **Sequential reads**: Input access is coalesced

## Performance Characteristics

- **Memory Pattern**: Sequential reads, random writes
- **Bottleneck**: Write bandwidth and potential conflicts
- **Conflicts**: When indices overlap, only one write succeeds
- **Best case**: All indices unique (no conflicts)

## Write Conflict Handling

This benchmark uses simple non-atomic writes:
- **Benefit**: Maximum performance, no synchronization overhead
- **Drawback**: Race conditions when indices overlap
- **Result**: One of the conflicting values wins (unpredictable which)
- **Verification**: Checks that output values came from input (not corrupted)

For applications requiring all writes (like histogram), atomic operations would be needed.

## Access Patterns

Performance varies with index distribution:
- **Sequential**: indices[i] = i → same as memcpy
- **Unique random**: No conflicts, but poor write coalescing
- **Overlapping**: Multiple writes to same location (conflicts)
- **Clustered**: Better cache utilization but more conflicts

## Gather vs Scatter

- **Gather**: Read conflicts don't matter (same value read multiple times)
- **Scatter**: Write conflicts may lose data
- **Performance**: Often similar, but scatter can be slower due to write conflicts

## Expected Results

Modern GPUs should achieve:
- **No conflicts**: ~50-200 M elements/s
- **With conflicts**: Lower throughput due to serialization
- **Variable performance**: Depends heavily on indices distribution
- **Write bandwidth**: Limited by non-coalesced writes
