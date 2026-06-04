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

let reduce_sum_kernel =
  [%kernel
    fun (input : float32 vector) (output : float32 vector) (n : int32) ->
      (* Shared memory for the block - 256 elements *)
      let%shared (sdata : float32) = 256l in
      
      let tid = thread_idx_x in
      let gid = thread_idx_x + (block_dim_x * block_idx_x) in
      
      (* Load data: each thread loads one element *)
      let%superstep load =
        if gid < n then sdata.(tid) <- input.(gid) 
        else sdata.(tid) <- 0.0
      in
      
      (* Tree reduction in shared memory - unrolled for efficiency *)
      let%superstep reduce128 =
        if tid < 128l then sdata.(tid) <- sdata.(tid) +. sdata.(tid + 128l)
      in
      let%superstep reduce64 =
        if tid < 64l then sdata.(tid) <- sdata.(tid) +. sdata.(tid + 64l)
      in
      let%superstep reduce32 =
        if tid < 32l then sdata.(tid) <- sdata.(tid) +. sdata.(tid + 32l)
      in
      let%superstep reduce16 =
        if tid < 16l then sdata.(tid) <- sdata.(tid) +. sdata.(tid + 16l)
      in
      let%superstep reduce8 =
        if tid < 8l then sdata.(tid) <- sdata.(tid) +. sdata.(tid + 8l)
      in
      let%superstep reduce4 =
        if tid < 4l then sdata.(tid) <- sdata.(tid) +. sdata.(tid + 4l)
      in
      let%superstep reduce2 =
        if tid < 2l then sdata.(tid) <- sdata.(tid) +. sdata.(tid + 2l)
      in
      let%superstep reduce1 =
        if tid < 1l then sdata.(tid) <- sdata.(tid) +. sdata.(tid + 1l)
      in
      
      (* Write result for this block *)
      if tid = 0l then output.(block_idx_x) <- sdata.(0l)]
```

## Host Code

```ocaml
open Sarek
module Device = Spoc_core.Device
module Vector = Spoc_core.Vector

let run_reduction () =
  let n = 1_000_000 in
  let block_size = 256 in
  let num_blocks = (n + block_size - 1) / block_size in
  
  (* Get IR from kernel *)
  let _, kirc = reduce_sum_kernel in
  let ir = match kirc.Sarek.Kirc_types.body_ir with
    | Some ir -> ir | None -> failwith "No IR" in
  
  (* Create input and output vectors *)
  let input = Vector.create Vector.float32 n in
  let output = Vector.create Vector.float32 num_blocks in
  
  (* Initialize input data *)
  for i = 0 to n - 1 do
    Vector.set input i 1.0;
  done;
  
  (* Calculate grid dimensions *)
  let block = Execute.dims1d block_size in
  let grid = Execute.dims1d num_blocks in
  let dev = Device.best () in
  
  (* First pass: Reduce array to num_blocks partial sums *)
  Execute.run_vectors
    ~device:dev
    ~ir
    ~args:[Vec input; Vec output; Int n]
    ~block
    ~grid
    ();
  
  (* Second pass: Sum the partial results on CPU *)
  let partial_sums = Vector.to_array output in
  let final_sum = Array.fold_left (+.) 0.0 partial_sums in
  
  Printf.printf "Sum of %d elements: %f\n" n final_sum
