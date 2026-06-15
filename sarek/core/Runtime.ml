(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Sarek Runtime - High-level Kernel Execution
 *
 * Provides the main API for executing Sarek kernels on GPU devices.
 * This bridges the PPX-generated IR with the new ctypes plugin architecture.
 *
 * Usage:
 *   let dev = Runtime.init_device () in
 *   Runtime.run kernel ~device:dev ~block:(256,1,1) ~grid:(n/256,1,1) args
 ******************************************************************************)

open Spoc_framework

(** Re-export framework dims helpers to avoid duplicate types *)
type dims = Framework_sig.dims = {x : int; y : int; z : int}

let dims1d = Framework_sig.dims_1d

let dims2d = Framework_sig.dims_2d

let dims3d = Framework_sig.dims_3d

let to_framework_dims (d : dims) = d

(** Initialize the runtime and get the best available device *)
let init_device ?framework () =
  let frameworks =
    match framework with Some f -> [f] | None -> ["CUDA"; "OpenCL"]
  in
  let _ = Device.init ~frameworks () in
  match Device.best () with
  | Some d -> d
  | None -> failwith "No GPU device available"

(** Get all available devices *)
let all_devices ?framework () =
  let frameworks =
    match framework with Some f -> [f] | None -> ["CUDA"; "OpenCL"]
  in
  Device.init ~frameworks ()

(** Kernel source cache: (device_framework, kernel_name, source_hash) ->
    compiled *)
let kernel_cache : (string * string * int, Kernel.t) Hashtbl.t =
  Hashtbl.create 32

(** Compile a kernel from source, with caching *)
let compile_kernel (device : Device.t) ~(name : string) ~(source : string) :
    Kernel.t =
  let key = (device.framework, name, Hashtbl.hash source) in
  match Hashtbl.find_opt kernel_cache key with
  | Some k -> k
  | None ->
      let k = Kernel.compile_cached device ~name ~source in
      Hashtbl.replace kernel_cache key k ;
      k

(** Clear the kernel cache *)
let clear_cache () = Hashtbl.clear kernel_cache

(** Argument builder - collects kernel arguments *)
type arg =
  | ArgBuffer : _ Memory.buffer -> arg
  | ArgInt32 : int32 -> arg
  | ArgInt64 : int64 -> arg
  | ArgFloat32 : float -> arg
  | ArgFloat64 : float -> arg

(** Create arguments from a list *)
let set_args (device : Device.t) (args : arg list) : Kernel.args =
  let kargs = Kernel.create_args device in
  List.iteri
    (fun i arg ->
      match arg with
      | ArgBuffer buf -> Kernel.set_arg_buffer kargs i buf
      | ArgInt32 v -> Kernel.set_arg_int32 kargs i v
      | ArgInt64 v -> Kernel.set_arg_int64 kargs i v
      | ArgFloat32 v -> Kernel.set_arg_float32 kargs i v
      | ArgFloat64 v -> Kernel.set_arg_float64 kargs i v)
    args ;
  kargs

(** Run a kernel with the given arguments *)
let run_kernel (kernel : Kernel.t) ~(args : Kernel.args) ~(grid : dims)
    ~(block : dims) ?(shared_mem = 0) () : unit =
  Kernel.launch
    kernel
    ~args
    ~grid:(to_framework_dims grid)
    ~block:(to_framework_dims block)
    ~shared_mem
    ()

(** High-level run function: compile (if needed) and execute *)
let run (device : Device.t) ~(name : string) ~(source : string)
    ~(args : arg list) ~(grid : dims) ~(block : dims) ?(shared_mem = 0) () :
    unit =
  let kernel = compile_kernel device ~name ~source in
  let kargs = set_args device args in
  run_kernel kernel ~args:kargs ~grid ~block ~shared_mem ()

(** Memory allocation shortcuts *)
let alloc_float32 device n = Memory.alloc device n Bigarray.float32

let alloc_float64 device n = Memory.alloc device n Bigarray.float64

let alloc_int32 device n = Memory.alloc device n Bigarray.int32

let alloc_int64 device n = Memory.alloc device n Bigarray.int64

(** Allocate a buffer for custom types with explicit element size *)
let alloc_custom = Memory.alloc_custom

(** Host-to-device transfer *)
let to_device = Memory.host_to_device

(** Device-to-host transfer *)
let from_device = Memory.device_to_host

(** Host-to-device transfer for custom types (raw pointer) *)
let to_device_ptr = Memory.host_ptr_to_device

(** Device-to-host transfer for custom types (raw pointer) *)
let from_device_ptr = Memory.device_to_host_ptr

(** Free a buffer *)
let free = Memory.free
