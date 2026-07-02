(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Regression test: set_arg* must honor the caller-supplied idx, not the call
 * order.
 *
 * Backends historically accumulated set_arg* calls by call order (reversed
 * lists / appends) and silently ignored idx. Every production caller happens
 * to call set_arg in ascending, contiguous, unique order, so the bug never
 * surfaced there -- but calling set_arg out of order (legal per the ARGS
 * signature, which takes an explicit idx) used to silently mis-bind
 * arguments. This test drives Native and Interpreter's low-level Kernel API
 * directly, out of order, and checks the kernel observes the *correct*
 * idx -> value binding.
 ******************************************************************************)

module Native_kernel = Sarek_native.Native_plugin_base.Native.Kernel
module Native_device = Sarek_native.Native_plugin_base.Native.Device

(** {1 Native backend} *)

(* Native kernels are raw OCaml functions over an exec_arg array; no PPX
   needed. sub(a, b) = a - b, order-sensitive so a wrong idx <-> value
   binding is observable in the result. *)
let native_result = ref 0l

let native_sub_fn args (_gx, _gy, _gz) (_bx, _by, _bz) =
  let open Spoc_framework.Framework_sig in
  let a = match args.(0) with EA_Int32 n -> n | _ -> failwith "arg 0" in
  let b = match args.(1) with EA_Int32 n -> n | _ -> failwith "arg 1" in
  native_result := Int32.sub a b

let () =
  Sarek_native.Native_plugin_base.register_kernel "sub_order_test" native_sub_fn

let test_native_out_of_order () =
  Native_device.init () ;
  let dev = Native_device.get 0 in
  let kernel = Native_kernel.compile dev ~name:"sub_order_test" ~source:"" in
  let args = Native_kernel.create_args () in
  (* Out of order: set idx 1 (b) before idx 0 (a). *)
  Native_kernel.set_arg_int32 args 1 2l ;
  Native_kernel.set_arg_int32 args 0 5l ;
  Native_kernel.launch
    kernel
    ~args
    ~grid:{Spoc_framework.Framework_sig.x = 1; y = 1; z = 1}
    ~block:{Spoc_framework.Framework_sig.x = 1; y = 1; z = 1}
    ~shared_mem:0
    ~stream:None ;
  let expected =
    3l
    (* a=5, b=2 => 5-2=3 *)
  in
  if !native_result <> expected then begin
    Printf.printf
      "[Native] FAIL: set_arg idx not honored: expected a-b=%ld, got %ld \
       (call-order binding would compute 2-5=-3)\n"
      expected
      !native_result ;
    exit 1
  end ;
  Printf.printf
    "[Native] out-of-order set_arg honors idx: OK (a-b=%ld)\n"
    !native_result

(** {1 Interpreter backend} *)

module Interp_kernel =
  Sarek_interpreter.Interpreter_plugin_base.Interpreter.Kernel

module Interp_device =
  Sarek_interpreter.Interpreter_plugin_base.Interpreter.Device

module Interp_memory =
  Sarek_interpreter.Interpreter_plugin_base.Interpreter.Memory

let sub_kirc =
  snd
    [%kernel
      fun (a : int32) (b : int32) (out : int32 vector) -> out.(0) <- a - b]

let test_interpreter_out_of_order () =
  let ir =
    match sub_kirc.Sarek.Kirc_types.body_ir with
    | Some ir -> ir
    | None -> failwith "sub_kirc has no IR"
  in
  Sarek_interpreter.Interpreter_plugin_base.register_kernel
    "sub_order_test_interp"
    ir ;
  Interp_device.init () ;
  let dev = Interp_device.get 0 in
  let kernel =
    Interp_kernel.compile dev ~name:"sub_order_test_interp" ~source:""
  in
  let out_buf = Interp_memory.alloc dev 1 Bigarray.int32 in
  let args = Interp_kernel.create_args () in
  (* Out of order: buffer (idx 2), then b (idx 1), then a (idx 0). *)
  Interp_kernel.set_arg_buffer args 2 out_buf ;
  Interp_kernel.set_arg_int32 args 1 2l ;
  Interp_kernel.set_arg_int32 args 0 5l ;
  Interp_kernel.launch
    kernel
    ~args
    ~grid:{Spoc_framework.Framework_sig.x = 1; y = 1; z = 1}
    ~block:{Spoc_framework.Framework_sig.x = 1; y = 1; z = 1}
    ~shared_mem:0
    ~stream:None ;
  let host = Bigarray.Array1.create Bigarray.int32 Bigarray.c_layout 1 in
  Interp_memory.device_to_host ~src:out_buf ~dst:host ;
  let got = Bigarray.Array1.get host 0 in
  let expected = 3l in
  if got <> expected then begin
    Printf.printf
      "[Interpreter] FAIL: set_arg idx not honored: expected a-b=%ld, got %ld\n"
      expected
      got ;
    exit 1
  end ;
  Printf.printf
    "[Interpreter] out-of-order set_arg honors idx: OK (a-b=%ld)\n"
    got

let () =
  test_native_out_of_order () ;
  test_interpreter_out_of_order () ;
  print_endline "=== set_arg idx-ordering regression test PASSED ==="
