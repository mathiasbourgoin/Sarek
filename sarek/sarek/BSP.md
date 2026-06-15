# BSP Model and Supersteps

Sarek provides Bulk Synchronous Parallel (BSP) constructs for structured GPU programming with explicit synchronization.

## Overview

GPU kernels often require synchronization between phases:
1. Load data into shared memory
2. **Barrier** - ensure all threads have loaded
3. Compute using shared data
4. **Barrier** - ensure all threads have computed
5. Write results

The `superstep` construct enforces this pattern with compile-time safety.

## Syntax

### Superstep Blocks

```ocaml
let kernel = [%kernel fun shared input output ->
  (* Phase 1: Load *)
  superstep {
    shared.(thread_idx_x) <- input.(global_idx)
  };

  (* Phase 2: Compute - all threads see loaded data *)
  superstep {
    let sum = shared.(thread_idx_x) + shared.(thread_idx_x + 1) in
    output.(global_idx) <- sum
  }
]
```

Each `superstep { }` block:
- Executes its body
- Inserts an implicit `block_barrier` at the end
- Ensures all threads complete before the next superstep begins

### Barriers

Explicit barriers are also available:

```ocaml
Gpu.block_barrier  (* Synchronize all threads in block *)
Gpu.warp_barrier   (* Synchronize threads within a warp *)
```

## Convergence Safety

Sarek tracks warp convergence to prevent deadlocks.

### The Problem

```ocaml
(* DEADLOCK: Some threads wait at barrier, others skip it *)
if thread_idx_x < 16 then
  Gpu.block_barrier  (* Only half the threads reach this! *)
```

### Compile-Time Detection

Sarek detects barriers in diverged control flow:

```
Error: Barrier in potentially diverged control flow
  This barrier may not be reached by all threads, causing deadlock.
  Consider using 'superstep ~divergent { }' if intentional.
```

### Convergence States

| State | Meaning | Barrier Allowed |
|-------|---------|-----------------|
| Converged | All threads executing | Yes |
| Uniform | Threads may diverge but will reconverge | Yes (at reconvergence) |
| Diverged | Threads have diverged | No |

### Allowing Divergent Barriers

For advanced cases where divergence is intentional:

```ocaml
superstep ~divergent {
  if thread_idx_x = 0 then
    (* Only thread 0 does cleanup *)
    cleanup ()
}
(* Barrier still inserted, but divergence warning suppressed *)
```

## Warp-Level Operations

Some operations are warp-synchronous:

```ocaml
(* Warp shuffle - exchange data within warp *)
let value = Gpu.shfl_down value 1

(* Warp vote - check condition across warp *)
let all_positive = Gpu.all (x > 0)
let any_negative = Gpu.any (x < 0)
let ballot = Gpu.ballot (x > threshold)
```

These implicitly synchronize the warp.

## Best Practices

### Do

```ocaml
(* Structured phases with supersteps *)
superstep { load_phase () };
superstep { compute_phase () };
superstep { store_phase () }

(* Uniform conditions are safe *)
if block_idx_x = 0 then
  superstep { special_block_work () }
```

### Don't

```ocaml
(* Barrier depends on thread-varying condition *)
if thread_idx_x mod 2 = 0 then
  Gpu.block_barrier  (* ERROR: diverged barrier *)

(* Barrier in loop with thread-varying bound *)
for i = 0 to thread_idx_x do
  Gpu.block_barrier  (* ERROR: different iteration counts *)
done
```

## Implementation

The convergence checker (`Sarek_convergence.ml`) tracks:
- Control flow entry points
- Conditions involving thread indices
- Barrier locations
- Reconvergence points

Errors are reported at compile time, before GPU code generation.

## Example: Parallel Reduction

```ocaml
let reduce = [%kernel fun shared input output n ->
  (* Load into shared memory *)
  superstep {
    shared.(thread_idx_x) <-
      if global_idx < n then input.(global_idx) else 0l
  };

  (* Tree reduction *)
  let stride = ref (block_dim_x / 2) in
  while !stride > 0 do
    superstep {
      if thread_idx_x < !stride then
        shared.(thread_idx_x) <-
          shared.(thread_idx_x) + shared.(thread_idx_x + !stride)
    };
    stride := !stride / 2
  done;

  (* Write result *)
  if thread_idx_x = 0 then
    output.(block_idx_x) <- shared.(0)
]
```
