---
layout: index_sample
title: Vector Addition Example
---

# Vector Addition

Vector addition is the "Hello World" of GPU programming. It demonstrates how to perform element-wise operations on arrays in parallel.

## Kernel Code

The kernel takes two input vectors `a` and `b`, and writes the result to `c`. Each thread processes one element.

```ocaml
open Sarek

let%kernel vector_add (a : float32 vector) (b : float32 vector) (c : float32 vector) (n : int32) =
  (* Get global thread ID *)
  let tid = get_global_id 0 in
  
  (* Check bounds to prevent out-of-bounds access *)
  if tid < n then
    (* Perform element-wise addition *)
    c.(tid) <- a.(tid) + b.(tid)
```

## Host Code

The host code initializes the data, selects a device, and launches the kernel.

```ocaml
let () =
  (* Problem size *)
  let n = 1_000_000 in
  
  (* Create vectors *)
  let a = Vector.create Float32 n in
  let b = Vector.create Float32 n in
  let c = Vector.create Float32 n in
  
  (* Initialize data *)
  for i = 0 to n - 1 do
    Vector.set a i (float_of_int i);
    Vector.set b i (float_of_int (i * 2));
  done;
  
  (* Select device *)
  let device = Device.get_default () in
  
  (* Calculate grid dimensions *)
  let block_size = 256 in
  let grid_size = (n + block_size - 1) / block_size in
  
  (* Run kernel *)
  Execute.run vector_add 
    ~device 
    ~grid:(grid_size, 1, 1) 
    ~block:(block_size, 1, 1) 
    [Vec a; Vec b; Vec c; Int32 (Int32.of_int n)];
    
  (* Verify result *)
  let result = Vector.get c 10 in
  Printf.printf "c[10] = %f (expected 30.0)\n" result
```

```