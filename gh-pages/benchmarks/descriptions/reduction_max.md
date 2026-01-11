# Max Reduction

**Category**: Parallel Reduction  
**Pattern**: Comparison-based reduction  
**Demonstrates**: Tree-based parallel reduction with comparison operations

## Overview

Max Reduction finds the maximum value in an array using parallel tree-based reduction. It demonstrates how the **reduction pattern generalizes** beyond arithmetic operations to comparison-based operations.

## Algorithm

Same tree structure as sum reduction, but with comparison instead of addition:

```
Step 1: Compare pairs (256 → 128 elements)
Step 2: Compare pairs (128 → 64 elements)  
...
Step 8: Final compare (2 → 1 element)
```

## Kernel

```ocaml
let reduce_max_kernel =
  [%kernel
    fun (input : float32 vector) (output : float32 vector) (n : int32) ->
      let%shared (sdata : float32) = 256l in
      (* Load into shared memory *)
      let%superstep load =
        if gid < n then sdata.(tid) <- input.(gid)
        else sdata.(tid) <- -1000000.0  (* Sentinel *)
      in
      (* Tree reduction with max comparison *)
      let%superstep reduce128 =
        if tid < 128l then
          if sdata.(tid + 128l) > sdata.(tid) then
            sdata.(tid) <- sdata.(tid + 128l)
      in
      (* ... 6 more reduction steps ... *)
      let%superstep write =
        if tid = 0l then output.(block_idx_x) <- sdata.(0l)
      in
      ()]
```

**Key Difference from Sum**: Uses `if b > a` instead of `+`

## Performance

Same as sum reduction: **memory-bound**, not compute-bound.

<!-- GENERATED_CODE_TABS: reduction_max -->
