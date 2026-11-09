# Dot Product

**Category**: Parallel Reduction  
**Pattern**: Map-Reduce combination  
**Demonstrates**: Element-wise multiply followed by sum reduction

## Overview

Dot Product combines two operations: **map** (element-wise multiply) and **reduce** (sum). This is a fundamental pattern in linear algebra, ML, and scientific computing.

Formula: `result = Σ(A[i] × B[i])` for i = 0 to N-1

## Algorithm

1. **Map phase**: Each thread computes `A[i] × B[i]`
2. **Reduce phase**: Tree-based sum (same as sum reduction)

## Kernel

```ocaml
let dot_product_kernel =
  [%kernel
    fun (a : float32 vector) (b : float32 vector) (output : float32 vector) (n : int32) ->
      let%shared (sdata : float32) = 256l in
      (* Map: multiply in load step *)
      let%superstep load =
        if gid < n then sdata.(tid) <- a.(gid) *. b.(gid)
        else sdata.(tid) <- 0.0
      in
      (* Reduce: tree-based sum (8 steps) *)
      (* ... same as sum reduction ... *)
]
```

**Key**: Map happens during load, reduce is standard sum pattern.

## Use Cases
- Vector similarity (cosine similarity)
- Neural network operations  
- Linear algebra (BLAS dot)
- Physics simulations

<!-- GENERATED_CODE_TABS: dot_product -->
