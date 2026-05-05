(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

[@@@warning "-33"]

(******************************************************************************
 * E2E test for Native plugin integration with new runtime
 *
 * This test verifies that:
 * 1. Native plugin can be used through the new runtime
 * 2. Kernels registered with Native_plugin.register_kernel are executed
 * 3. Results are correct
 *
 * NOTE ON BUFFER ARCHITECTURE (Phase 4 TODO):
 *
 * We removed the SPOC custom-vector path. When runtime custom buffers are ready,
 * reintroduce a dedicated test that exercises them. The desired design is a
 * single buffer type that:
 *
 * 1. Uses ctypes raw memory (works on all backends: CUDA, OpenCL, Native)
 * 2. Has an element descriptor:
 *    - Numeric: size + Bigarray kind (for float32, int32, etc.)
 *    - Custom: size + get/set functions (for records like point, float4, etc.)
 * 3. Is completely backend-agnostic - the buffer doesn't know about frameworks
 * 4. Supports multi-device: same buffer can have copies on multiple devices
 * 5. Tracks dirty state to know which copies need synchronization
 ******************************************************************************)

module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer
module Execute = Sarek.Execute

(* Force plugin initialization *)
let () = Sarek_native.Native_plugin.init ()

let size = 1024

[@@@warning "-32"]

type float32 = float

(* Custom record type for 2D points - PPX generates point_custom *)
type point = {x : float32; y : float32} [@@sarek.type]

let transform_points_kirc =
  snd
    [%kernel
      fun (src : point vector) (dst : point vector) (n : int32) ->
        let tid = thread_idx_x in
        if tid < n then
          let p = src.(tid) in
          dst.(tid) <- {x = p.x +. 1.0; y = p.y *. 2.0}]

(* A simple kernel function that adds vectors: c[i] = a[i] + b[i] *)
let get_float32_arg args arg_idx elem_idx =
  let open Spoc_framework.Framework_sig in
  match args.(arg_idx) with
  | EA_Vec (module V) -> (
      match V.get elem_idx with
      | Spoc_framework.Typed_value.TV_Scalar
          (Spoc_framework.Typed_value.SV ((module S), x)) -> (
          match S.to_primitive x with
          | Spoc_framework.Typed_value.PFloat f when S.name = "float32" -> f
          | Spoc_framework.Typed_value.PFloat _ ->
              failwith "Expected float32 scalar, got non-float32 float"
          | _ -> failwith "Expected float32 scalar")
      | _ -> failwith "Expected scalar vector element")
  | _ -> failwith (Printf.sprintf "Expected vector for arg %d" arg_idx)

let set_float32_arg args arg_idx elem_idx value =
  let open Spoc_framework.Framework_sig in
  match args.(arg_idx) with
  | EA_Vec (module V) ->
      V.set
        elem_idx
        (Spoc_framework.Typed_value.TV_Scalar
           (Spoc_framework.Typed_value.SV
              ((module Spoc_framework.Typed_value.Float32_type), value)))
  | _ -> failwith (Printf.sprintf "Expected vector for arg %d" arg_idx)

let vector_add_kernel args (gx, _gy, _gz) (bx, _by, _bz) =
  (* Extract arguments from exec_arg array *)
  let open Spoc_framework.Framework_sig in
  let n =
    match args.(3) with
    | EA_Int32 n -> Int32.to_int n
    | _ -> failwith "Expected int32 for arg 3"
  in

  (* Compute total threads and iterate *)
  let total_threads = gx * bx in
  for tid = 0 to total_threads - 1 do
    if tid < n then
      set_float32_arg
        args
        2
        tid
        (get_float32_arg args 0 tid +. get_float32_arg args 1 tid)
  done

(* Register the kernel *)
let () =
  Sarek_native.Native_plugin.register_kernel "vector_add" vector_add_kernel

let () =
  print_endline "=== Native Runtime Integration Test ===" ;

  (* Step 1: Initialize native devices *)
  print_endline "\n[1] Initializing Native devices..." ;
  let devices = Spoc_core.Device.init ~frameworks:["Native"] () in
  Printf.printf "Found %d Native device(s)\n" (Array.length devices) ;
  if Array.length devices = 0 then begin
    print_endline "No Native devices found - test failed" ;
    exit 1
  end ;
  let dev = devices.(0) in
  Printf.printf "Using device: %s (%s)\n" dev.name dev.framework ;

  (* Step 2: Check kernel is registered *)
  print_endline "\n[2] Checking kernel registration..." ;
  let registered = Sarek_native.Native_plugin.kernel_registered "vector_add" in
  Printf.printf "Kernel 'vector_add' registered: %b\n" registered ;
  if not registered then begin
    print_endline "Kernel not registered - test failed" ;
    exit 1
  end ;

  (* Step 3: Allocate buffers *)
  print_endline "\n[3] Allocating buffers..." ;
  let buf_a = Spoc_core.Runtime.alloc_float32 dev size in
  let buf_b = Spoc_core.Runtime.alloc_float32 dev size in
  let buf_c = Spoc_core.Runtime.alloc_float32 dev size in
  print_endline "Buffers allocated" ;

  (* Step 4: Initialize data *)
  print_endline "\n[4] Initializing data..." ;
  let host_a = Bigarray.Array1.create Bigarray.float32 Bigarray.c_layout size in
  let host_b = Bigarray.Array1.create Bigarray.float32 Bigarray.c_layout size in
  let host_c = Bigarray.Array1.create Bigarray.float32 Bigarray.c_layout size in
  for i = 0 to size - 1 do
    Bigarray.Array1.set host_a i (float_of_int i) ;
    Bigarray.Array1.set host_b i (float_of_int (i * 2)) ;
    Bigarray.Array1.set host_c i 0.0
  done ;
  Spoc_core.Runtime.to_device ~src:host_a ~dst:buf_a ;
  Spoc_core.Runtime.to_device ~src:host_b ~dst:buf_b ;
  Spoc_core.Runtime.to_device ~src:host_c ~dst:buf_c ;
  print_endline "Data initialized and transferred" ;

  (* Step 5: Execute kernel *)
  print_endline "\n[5] Executing kernel via new runtime..." ;
  let block = Spoc_core.Runtime.dims1d 256 in
  let grid = Spoc_core.Runtime.dims1d ((size + 255) / 256) in

  Spoc_core.Runtime.run
    dev
    ~name:"vector_add"
    ~source:"" (* Native doesn't need source *)
    ~args:
      [
        Spoc_core.Runtime.ArgBuffer buf_a;
        Spoc_core.Runtime.ArgBuffer buf_b;
        Spoc_core.Runtime.ArgBuffer buf_c;
        Spoc_core.Runtime.ArgInt32 (Int32.of_int size);
      ]
    ~grid
    ~block
    () ;
  print_endline "Kernel executed" ;

  (* Step 6: Verify results *)
  print_endline "\n[6] Verifying results..." ;
  Spoc_core.Runtime.from_device ~src:buf_c ~dst:host_c ;

  let errors = ref 0 in
  for i = 0 to size - 1 do
    let expected = float_of_int i +. float_of_int (i * 2) in
    let got = Bigarray.Array1.get host_c i in
    if abs_float (got -. expected) > 0.001 then begin
      if !errors < 5 then
        Printf.printf
          "  Mismatch at %d: expected %.2f, got %.2f\n"
          i
          expected
          got ;
      incr errors
    end
  done ;

  if !errors > 0 then begin
    Printf.printf "Total errors: %d\n" !errors ;
    print_endline "=== Test FAILED ===" ;
    exit 1
  end ;

  (* Cleanup *)
  Spoc_core.Runtime.free buf_a ;
  Spoc_core.Runtime.free buf_b ;
  Spoc_core.Runtime.free buf_c ;

  print_endline "Vector add test passed!" ;

  (* ============================================================ *)
  (* Part 2: Custom type test using runtime Vector + Execute           *)
  (* ============================================================ *)
  print_endline "\n[7] Testing custom types with runtime Vector..." ;

  let n_points = 100 in
  let src = Vector.create_custom point_custom n_points in
  let dst = Vector.create_custom point_custom n_points in

  for i = 0 to n_points - 1 do
    Vector.set src i {x = float_of_int i; y = float_of_int (i * 2)} ;
    Vector.set dst i {x = 0.0; y = 0.0}
  done ;

  let threads = min 256 n_points in
  let grid_x = (n_points + threads - 1) / threads in
  let ir =
    match transform_points_kirc.Sarek.Kirc_types.body_ir with
    | Some ir -> ir
    | None -> failwith "Kernel has no IR"
  in

  print_endline "\n[8] Executing custom type kernel..." ;
  Execute.run_vectors
    ~device:dev
    ~block:(Execute.dims1d threads)
    ~grid:(Execute.dims1d grid_x)
    ~ir
    ~args:
      [Execute.Vec src; Execute.Vec dst; Execute.Int32 (Int32.of_int n_points)]
    () ;
  Transfer.flush dev ;
  print_endline "Custom type kernel executed" ;

  print_endline "\n[9] Verifying custom type results..." ;
  let errors = ref 0 in
  for i = 0 to n_points - 1 do
    let expected_x = float_of_int i +. 1.0 in
    let expected_y = float_of_int (i * 2) *. 2.0 in
    let result = Vector.get dst i in
    if
      abs_float (result.x -. expected_x) > 1e-3
      || abs_float (result.y -. expected_y) > 1e-3
    then begin
      if !errors < 5 then
        Printf.printf
          "  Error at %d: expected {x=%.1f; y=%.1f}, got {x=%.1f; y=%.1f}\n"
          i
          expected_x
          expected_y
          result.x
          result.y ;
      incr errors
    end
  done ;

  if !errors > 0 then begin
    Printf.printf "Custom type test: %d errors\n" !errors ;
    print_endline "=== Test FAILED ===" ;
    exit 1
  end ;

  print_endline "Custom type test passed!" ;
  print_endline "\n=== All Tests PASSED ==="
