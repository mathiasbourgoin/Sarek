(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Unit tests for Kernel module
 *
 * Tests kernel compilation, argument handling, and launch API.
 * Note: Most operations require a backend, so we test structure and
 * error handling paths.
 ******************************************************************************)

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

(** {1 Module Type Tests} *)

let test_kernel_module_type () =
  (* Verify KERNEL module type signature by checking we can define a module
     that satisfies it *)
  let module FakeKernel : Spoc_core.Kernel.KERNEL = struct
    let device = make_fake_device ()

    let name = "test_kernel"

    let launch ~args:_ ~grid:_ ~block:_ ~shared_mem:_ = ()
  end in
  assert (FakeKernel.name = "test_kernel") ;
  assert (FakeKernel.device.name = "Fake Device") ;
  print_endline "  KERNEL module type: OK"

let test_args_module_type () =
  (* Verify ARGS module type signature by checking we can define a module
     that satisfies it *)
  let module FakeArgs : Spoc_core.Kernel.ARGS = struct
    let device = make_fake_device ()

    let kargs = Spoc_framework.Framework_sig.No_kargs

    let set_int32 _idx _v = ()

    let set_int64 _idx _v = ()

    let set_float32 _idx _v = ()

    let set_float64 _idx _v = ()

    let set_ptr _idx _ptr = ()
  end in
  assert (FakeArgs.device.name = "Fake Device") ;
  print_endline "  ARGS module type: OK"

(** {1 Compilation API Signature Tests} *)

let test_compile_signature () =
  let _f :
      Spoc_core.Device.t -> name:string -> source:string -> Spoc_core.Kernel.t =
    Spoc_core.Kernel.compile
  in
  print_endline "  compile signature: OK"

let test_compile_cached_signature () =
  let _f :
      Spoc_core.Device.t -> name:string -> source:string -> Spoc_core.Kernel.t =
    Spoc_core.Kernel.compile_cached
  in
  print_endline "  compile_cached signature: OK"

(** {1 Args API Signature Tests} *)

let test_create_args_signature () =
  let _f : Spoc_core.Device.t -> Spoc_core.Kernel.args =
    Spoc_core.Kernel.create_args
  in
  print_endline "  create_args signature: OK"

let test_set_arg_buffer_signature () =
  let _f : Spoc_core.Kernel.args -> int -> 'a Spoc_core.Memory.buffer -> unit =
    Spoc_core.Kernel.set_arg_buffer
  in
  print_endline "  set_arg_buffer signature: OK"

let test_set_arg_int32_signature () =
  let _f : Spoc_core.Kernel.args -> int -> int32 -> unit =
    Spoc_core.Kernel.set_arg_int32
  in
  print_endline "  set_arg_int32 signature: OK"

let test_set_arg_int64_signature () =
  let _f : Spoc_core.Kernel.args -> int -> int64 -> unit =
    Spoc_core.Kernel.set_arg_int64
  in
  print_endline "  set_arg_int64 signature: OK"

let test_set_arg_float32_signature () =
  let _f : Spoc_core.Kernel.args -> int -> float -> unit =
    Spoc_core.Kernel.set_arg_float32
  in
  print_endline "  set_arg_float32 signature: OK"

let test_set_arg_float64_signature () =
  let _f : Spoc_core.Kernel.args -> int -> float -> unit =
    Spoc_core.Kernel.set_arg_float64
  in
  print_endline "  set_arg_float64 signature: OK"

let test_set_arg_ptr_signature () =
  let _f : Spoc_core.Kernel.args -> int -> nativeint -> unit =
    Spoc_core.Kernel.set_arg_ptr
  in
  print_endline "  set_arg_ptr signature: OK"

(** {1 Launch API Signature Tests} *)

let test_launch_signature () =
  let _f :
      Spoc_core.Kernel.t ->
      args:Spoc_core.Kernel.args ->
      grid:Spoc_framework.Framework_sig.dims ->
      block:Spoc_framework.Framework_sig.dims ->
      ?shared_mem:int ->
      unit ->
      unit =
    Spoc_core.Kernel.launch
  in
  print_endline "  launch signature: OK"

let test_clear_cache_signature () =
  let _f : Spoc_core.Device.t -> unit = Spoc_core.Kernel.clear_cache in
  print_endline "  clear_cache signature: OK"

(** {1 Accessor Signature Tests} *)

let test_device_signature () =
  let _f : Spoc_core.Kernel.t -> Spoc_core.Device.t = Spoc_core.Kernel.device in
  print_endline "  device signature: OK"

let test_name_signature () =
  let _f : Spoc_core.Kernel.t -> string = Spoc_core.Kernel.name in
  print_endline "  name signature: OK"

let test_args_device_signature () =
  let _f : Spoc_core.Kernel.args -> Spoc_core.Device.t =
    Spoc_core.Kernel.args_device
  in
  print_endline "  args_device signature: OK"

let test_get_kargs_signature () =
  let _f : Spoc_core.Kernel.args -> Spoc_framework.Framework_sig.kargs =
    Spoc_core.Kernel.get_kargs
  in
  print_endline "  get_kargs signature: OK"

(** {1 Error Handling Tests} *)

let test_compile_unknown_framework () =
  let fake_device = make_fake_device () in
  let raised = ref false in
  (try
     let _ = Spoc_core.Kernel.compile fake_device ~name:"test" ~source:"" in
     ()
   with Failure msg ->
     if String.sub msg 0 18 = "Unknown framework:" then raised := true) ;
  assert !raised ;
  print_endline "  compile unknown framework error: OK"

let test_compile_cached_unknown_framework () =
  let fake_device = make_fake_device () in
  let raised = ref false in
  (try
     let _ =
       Spoc_core.Kernel.compile_cached fake_device ~name:"test" ~source:""
     in
     ()
   with Failure msg ->
     if String.sub msg 0 18 = "Unknown framework:" then raised := true) ;
  assert !raised ;
  print_endline "  compile_cached unknown framework error: OK"

let test_create_args_unknown_framework () =
  let fake_device = make_fake_device () in
  let raised = ref false in
  (try
     let _ = Spoc_core.Kernel.create_args fake_device in
     ()
   with Failure msg ->
     if String.sub msg 0 18 = "Unknown framework:" then raised := true) ;
  assert !raised ;
  print_endline "  create_args unknown framework error: OK"

let test_clear_cache_unknown_framework () =
  (* clear_cache should not raise for unknown framework - it's a no-op *)
  let fake_device = make_fake_device () in
  Spoc_core.Kernel.clear_cache fake_device ;
  print_endline "  clear_cache unknown framework (no-op): OK"

(** {1 Framework_sig.dims Tests} *)

let test_dims_usage () =
  let grid : Spoc_framework.Framework_sig.dims = {x = 128; y = 1; z = 1} in
  let block : Spoc_framework.Framework_sig.dims = {x = 256; y = 1; z = 1} in
  assert (grid.x = 128) ;
  assert (block.x = 256) ;
  print_endline "  dims usage: OK"

(** {1 Main} *)

let () =
  print_endline "Kernel module tests:" ;
  test_kernel_module_type () ;
  test_args_module_type () ;
  test_compile_signature () ;
  test_compile_cached_signature () ;
  test_create_args_signature () ;
  test_set_arg_buffer_signature () ;
  test_set_arg_int32_signature () ;
  test_set_arg_int64_signature () ;
  test_set_arg_float32_signature () ;
  test_set_arg_float64_signature () ;
  test_set_arg_ptr_signature () ;
  test_launch_signature () ;
  test_clear_cache_signature () ;
  test_device_signature () ;
  test_name_signature () ;
  test_args_device_signature () ;
  test_get_kargs_signature () ;
  test_compile_unknown_framework () ;
  test_compile_cached_unknown_framework () ;
  test_create_args_unknown_framework () ;
  test_clear_cache_unknown_framework () ;
  test_dims_usage () ;
  print_endline "All Kernel module tests passed!"
