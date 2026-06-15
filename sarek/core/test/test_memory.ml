(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Unit tests for Memory module
 *
 * Tests buffer types, allocation API, and transfer operations.
 * Note: Most operations require a backend, so we test structure and
 * error handling paths.
 ******************************************************************************)

(** {1 Helper to create fake device} *)

let make_fake_caps () : Spoc_framework.Framework_sig.capabilities =
  {
    max_threads_per_block = 256;
    max_block_dims = (256, 256, 64);
    max_grid_dims = (65535, 65535, 65535);
    shared_mem_per_block = 16384;
    total_global_mem = 1073741824L;
    compute_capability = (0, 0);
    supports_fp64 = true;
    supports_atomics = true;
    warp_size = 32;
    max_registers_per_block = 16384;
    clock_rate_khz = 1000000;
    multiprocessor_count = 4;
    is_cpu = true;
  }

let make_fake_device () : Spoc_core.Device.t =
  {
    id = 999;
    backend_id = 0;
    name = "Fake Device";
    framework = "NonExistentFramework";
    capabilities = make_fake_caps ();
  }

(** {1 Buffer Module Type Tests} *)

let test_buffer_module_type () =
  (* Verify BUFFER module type signature by checking we can define a module
     that satisfies it *)
  let module FakeBuffer : Spoc_core.Memory.BUFFER = struct
    let device = make_fake_device ()

    let size = 100

    let elem_size = 4

    let device_ptr = 0n

    let host_ptr_to_device _ptr ~byte_size:_ = ()

    let device_to_host_ptr _ptr ~byte_size:_ = ()

    let bind_to_kargs _kargs _idx = ()

    let free () = ()
  end in
  assert (FakeBuffer.size = 100) ;
  assert (FakeBuffer.elem_size = 4) ;
  print_endline "  BUFFER module type: OK"

(** {1 Allocation API Signature Tests} *)

let test_alloc_signature () =
  let _f :
      Spoc_core.Device.t ->
      int ->
      ('a, 'b) Bigarray.kind ->
      'a Spoc_core.Memory.buffer =
    Spoc_core.Memory.alloc
  in
  print_endline "  alloc signature: OK"

let test_alloc_custom_signature () =
  let _f :
      Spoc_core.Device.t ->
      size:int ->
      elem_size:int ->
      'a Spoc_core.Memory.buffer =
    Spoc_core.Memory.alloc_custom
  in
  print_endline "  alloc_custom signature: OK"

let test_free_signature () =
  let _f : 'a Spoc_core.Memory.buffer -> unit = Spoc_core.Memory.free in
  print_endline "  free signature: OK"

(** {1 Transfer API Signature Tests} *)

let test_host_to_device_signature () =
  let _f :
      src:('a, 'b, Bigarray.c_layout) Bigarray.Array1.t ->
      dst:'a Spoc_core.Memory.buffer ->
      unit =
    Spoc_core.Memory.host_to_device
  in
  print_endline "  host_to_device signature: OK"

let test_device_to_host_signature () =
  let _f :
      src:'a Spoc_core.Memory.buffer ->
      dst:('a, 'b, Bigarray.c_layout) Bigarray.Array1.t ->
      unit =
    Spoc_core.Memory.device_to_host
  in
  print_endline "  device_to_host signature: OK"

let test_host_ptr_to_device_signature () =
  let _f : src_ptr:unit Ctypes.ptr -> dst:'a Spoc_core.Memory.buffer -> unit =
    Spoc_core.Memory.host_ptr_to_device
  in
  print_endline "  host_ptr_to_device signature: OK"

let test_device_to_host_ptr_signature () =
  let _f : src:'a Spoc_core.Memory.buffer -> dst_ptr:unit Ctypes.ptr -> unit =
    Spoc_core.Memory.device_to_host_ptr
  in
  print_endline "  device_to_host_ptr signature: OK"

let test_device_to_device_signature () =
  let _f :
      src:'a Spoc_core.Memory.buffer -> dst:'a Spoc_core.Memory.buffer -> unit =
    Spoc_core.Memory.device_to_device
  in
  print_endline "  device_to_device signature: OK"

(** {1 Accessor API Signature Tests} *)

let test_size_signature () =
  let _f : 'a Spoc_core.Memory.buffer -> int = Spoc_core.Memory.size in
  print_endline "  size signature: OK"

let test_elem_size_signature () =
  let _f : 'a Spoc_core.Memory.buffer -> int = Spoc_core.Memory.elem_size in
  print_endline "  elem_size signature: OK"

let test_device_signature () =
  let _f : 'a Spoc_core.Memory.buffer -> Spoc_core.Device.t =
    Spoc_core.Memory.device
  in
  print_endline "  device signature: OK"

let test_device_ptr_signature () =
  let _f : 'a Spoc_core.Memory.buffer -> nativeint =
    Spoc_core.Memory.device_ptr
  in
  print_endline "  device_ptr signature: OK"

let test_bind_to_kargs_signature () =
  let _f :
      'a Spoc_core.Memory.buffer ->
      Spoc_framework.Framework_sig.kargs ->
      int ->
      unit =
    Spoc_core.Memory.bind_to_kargs
  in
  print_endline "  bind_to_kargs signature: OK"

(** {1 Error Handling Tests} *)

let test_alloc_unknown_framework () =
  let fake_device = make_fake_device () in
  let raised = ref false in
  (try
     let _ = Spoc_core.Memory.alloc fake_device 100 Bigarray.float32 in
     ()
   with Failure msg ->
     if String.sub msg 0 18 = "Unknown framework:" then raised := true) ;
  assert !raised ;
  print_endline "  alloc unknown framework error: OK"

let test_alloc_custom_unknown_framework () =
  let fake_device = make_fake_device () in
  let raised = ref false in
  (try
     let _ = Spoc_core.Memory.alloc_custom fake_device ~size:100 ~elem_size:4 in
     ()
   with Failure msg ->
     if String.sub msg 0 18 = "Unknown framework:" then raised := true) ;
  assert !raised ;
  print_endline "  alloc_custom unknown framework error: OK"

(** {1 Main} *)

let () =
  print_endline "Memory module tests:" ;
  test_buffer_module_type () ;
  test_alloc_signature () ;
  test_alloc_custom_signature () ;
  test_free_signature () ;
  test_host_to_device_signature () ;
  test_device_to_host_signature () ;
  test_host_ptr_to_device_signature () ;
  test_device_to_host_ptr_signature () ;
  test_device_to_device_signature () ;
  test_size_signature () ;
  test_elem_size_signature () ;
  test_device_signature () ;
  test_device_ptr_signature () ;
  test_bind_to_kargs_signature () ;
  test_alloc_unknown_framework () ;
  test_alloc_custom_unknown_framework () ;
  print_endline "All Memory module tests passed!"
