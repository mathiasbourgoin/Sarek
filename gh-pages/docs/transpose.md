---
layout: index_sample
title: Polymorphic Matrix Transpose
---

# Polymorphic Matrix Transpose

One of Sarek's most powerful features is **polymorphism**. Unlike C/CUDA where you often write separate kernels for `float`, `double`, and `int`, Sarek allows you to write generic functions that can be reused across different types.

This example demonstrates a generic transpose function that works on `int32`, `float32`, and even custom record types.

## 1. Generic Helper Function

First, we define a generic helper function using `[@sarek.module]`. This tells Sarek this function is intended for GPU execution but will be compiled only when used inside a kernel.

```ocaml
open Sarek

(* A generic transpose logic that works for any type 'a *)
let[@sarek.module] do_transpose (input : 'a vector) (output : 'a vector)
                                (width : int) (height : int) (tid : int) =
  let n = width * height in
  if tid < n then begin
    let col = tid mod width in
    let row = tid / width in
    let in_idx = (row * width) + col in
    let out_idx = (col * height) + row in
    
    (* Reads and writes 'a - works for floats, ints, structs... *)
    output.(out_idx) <- input.(in_idx)
  end
```

## 2. Defining Concrete Kernels

We can now define specific kernels that "monomorphize" (specialize) the generic helper for specific data types.

### For Basic Types (`float32`, `int32`)

```ocaml
(* Kernel specialized for Float32 *)
let%kernel transpose_float32 (input : float32 vector) (output : float32 vector)
                             (width : int32) (height : int32) =
  let tid = get_global_id 0 in
  do_transpose input output width height tid

(* Kernel specialized for Int32 *)
let%kernel transpose_int32 (input : int32 vector) (output : int32 vector)
                           (width : int32) (height : int32) =
  let tid = get_global_id 0 in
  do_transpose input output width height tid
```

### For Custom Records (`structs`)

Sarek also supports custom record types. We define the type with `[@@sarek.type]` and can immediately use our generic transpose logic on it.

```ocaml
(* Define a custom GPU-compatible record *)
type point3d = {
  x : float32;
  y : float32;
  z : float32
} [@@sarek.type]

(* Kernel specialized for Point3D records *)
let%kernel transpose_point3d (input : point3d vector) (output : point3d vector)
                             (width : int32) (height : int32) =
  let tid = get_global_id 0 in
  (* Sarek automatically handles the structure layout and memory copying *)
  do_transpose input output width height tid
```

## 3. Host Code

The host code looks standard, but notice how we handle the custom `point3d` vector.

```ocaml
let run_polymorphic_tests () =
  let width, height = 1024, 1024 in
  let n = width * height in
  let device = Device.get_default () in
  let block = (256, 1, 1) in
  let grid = ((n + 255)/256, 1, 1) in

  (* 1. Run Float32 Transpose *)
  let a = Vector.create Float32 n in
  let b = Vector.create Float32 n in
  (* ... init a ... *)
  Execute.run transpose_float32 ~device ~grid ~block 
    [Vec a; Vec b; Int32 width; Int32 height];

  (* 2. Run Custom Struct Transpose *)
  (* Create a vector for our custom type *)
  let points_in = Vector.create_custom 
    (module struct type t = point3d let size = 12 end) n in
  let points_out = Vector.create_custom 
    (module struct type t = point3d let size = 12 end) n in
    
  (* Initialize with OCaml records *)
  Vector.set points_in 0 { x=1.0; y=2.0; z=3.0 };
  
  (* Run the same logic on structs! *)
  Execute.run transpose_point3d ~device ~grid ~block 
    [Vec points_in; Vec points_out; Int32 width; Int32 height]
```

## Why this matters

In CUDA or OpenCL C, supporting `point3d` would require writing a new kernel `transpose_point3d_kernel` and manually handling the struct fields, or using complex C++ templates. In Sarek, the compiler handles the type specialization, structure layout, and memory access patterns automatically.