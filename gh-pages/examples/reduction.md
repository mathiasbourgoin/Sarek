---
layout: page
title: Parallel Reduction Example
---

# Parallel Reduction

Parallel reduction (e.g., sum, max, min) is a common pattern that requires coordination between threads. This example implements a tree-based reduction sum using shared memory.

## Kernel Code

Each thread block reduces a portion of the array into a single value. These partial sums are stored in the output array. A second kernel launch (or CPU loop) is usually required to sum the block results.

```ocaml
open Sarek

let%kernel reduce_sum (input : float32 vector) (output : float32 vector) (n : int32) =
  (* Shared memory for the block *)
  let%shared sdata = Array.create Float32 256 in
  
  let tid = thread_idx_x in
  let gid = get_global_id 0 in
  
  (* Load data: each thread loads one element *)
  (* For better performance, each thread could load/sum multiple elements first *)
  let%superstep load =
    if gid < n then 
      sdata.(tid) <- input.(gid)
    else 
      sdata.(tid) <- 0.0
  in
  
  (* Tree reduction in shared memory *)
  (* Unrolled for warp efficiency (simulated here with supersteps) *)
  let%superstep reduce128 =
    if tid < 128 then sdata.(tid) <- sdata.(tid) +. sdata.(tid + 128)
  in
  let%superstep reduce64 =
    if tid < 64 then sdata.(tid) <- sdata.(tid) +. sdata.(tid + 64)
  in
  (* ... unrolling continues ... *)
  let%superstep reduce2 =
    if tid < 2 then sdata.(tid) <- sdata.(tid) +. sdata.(tid + 2)
  in
  let%superstep reduce1 =
    if tid < 1 then sdata.(tid) <- sdata.(tid) +. sdata.(tid + 1)
  in
  
  (* Write result for this block *)
  if tid = 0 then
    output.(block_idx_x) <- sdata.(0)
```

## Host Code

```ocaml
let run_reduction () =
  let n = 1_000_000 in
  let block_size = 256 in
  let num_blocks = (n + block_size - 1) / block_size in
  
  let output = Vector.create Float32 num_blocks in
  
  (* First pass: Reduce array to `num_blocks` partial sums *)
  Execute.run reduce_sum
    ~device:(Device.get_default ())
    ~block:(256, 1, 1)
    ~grid:(num_blocks, 1, 1)
    [Vec input; Vec output; Int32 n];
    
  (* Second pass: Sum the partial results (on CPU for simplicity here) *)
  let partials = Vector.to_array output in
  let total = Array.fold_left (+.) 0.0 partials
```
