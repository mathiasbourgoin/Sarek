---
layout: page
title: Vector Addition Example
---

# Vector Addition

Vector addition is the "Hello World" of GPU programming. It demonstrates how to perform element-wise operations on arrays in parallel.

## Kernel Code

The kernel takes two input vectors `a` and `b`, and writes the result to `c`. Each thread processes one element.

```ocaml
open Sarek
module Std = Sarek_stdlib.Std

let vector_add_kernel =
  [%kernel
    fun (a : float32 vector)
        (b : float32 vector)
        (c : float32 vector)
        (n : int32) ->
      let open Std in
      let tid = global_thread_id in
      if tid < n then c.(tid) <- a.(tid) + b.(tid)]
```

## Host Code

The host code initializes the data, selects a device, and launches the kernel.

```ocaml
open Sarek
module Device = Spoc_core.Device
module Vector = Spoc_core.Vector

let () =
  (* Problem size *)
  let n = 1_000_000 in
  
  (* Get IR from kernel *)
  let _, kirc = vector_add_kernel in
  let ir = match kirc.Sarek.Kirc_types.body_ir with
    | Some ir -> ir | None -> failwith "No IR" in
  
  (* Create vectors *)
  let a = Vector.create Vector.float32 n in
  let b = Vector.create Vector.float32 n in
  let c = Vector.create Vector.float32 n in
  
  (* Initialize data *)
  for i = 0 to n - 1 do
    Vector.set a i (float_of_int i);
    Vector.set b i (float_of_int (i * 2));
  done;
  
  (* Select device *)
  let dev = Device.best () in
  
  (* Calculate grid dimensions *)
  let block_size = 256 in
  let grid_size = (n + block_size - 1) / block_size in
  let block = Execute.dims1d block_size in
  let grid = Execute.dims1d grid_size in
  
  (* Run kernel *)
  Execute.run_vectors
    ~device:dev
    ~ir
    ~args:[Vec a; Vec b; Vec c; Int n]
    ~block
    ~grid
    ();
    
  (* Verify result *)
  let result = Vector.get c 10 in
  Printf.printf "c[10] = %f (expected 30.0)\n" result
```

