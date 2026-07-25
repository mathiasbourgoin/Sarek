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

(** Outer kernel memo, keyed by {!Spoc_framework.Compile_cache.make_key} so this
    layer keys exactly like the per-backend caches underneath it. Guarded
    against concurrent multi-domain access by [Spoc_framework.Guarded_cache]
    (this is the cross-backend entry point most multi-domain code hits).

    The key MUST include device identity. [Device.t.framework] is the backend
    {e name}, shared by every device of that backend, so keying on it alone
    aliases all of them — and [Kernel.compile_cached] closes the specific
    backend kernel into the [Kernel.t] it returns, so a hit would hand device B
    a kernel built for device A and silently produce wrong results.

    [destroy] is a no-op because the backend handles are owned by the
    per-backend caches reached through [Kernel.compile_cached] — this layer
    holds only memoization and must never release anything. That does NOT make
    this layer safe to leave alone during teardown: those backend caches release
    the handles these [Kernel.t] closures capture, so the layer has to be
    dropped whenever they are (see the [Cache_hooks] registrations below). *)
let kernel_cache : (string, Kernel.t) Spoc_framework.Guarded_cache.t =
  Spoc_framework.Guarded_cache.create ~size:32 ~destroy:(fun _ -> ()) ()

(** Clear the kernel cache *)
let clear_cache () = Spoc_framework.Guarded_cache.clear kernel_cache

let () =
  (* Participate in backend teardown. Both directions are handle-invalidating
     for this layer, and neither is observable from here without the hook. *)
  Spoc_framework.Cache_hooks.on_clear_all clear_cache ;
  Spoc_framework.Cache_hooks.on_device_destroy (fun backend_device_id ->
      (* The hook carries a BACKEND-local device index, while this cache groups
         by the global [Device.t.id]. Several backends can use the same index,
         and the notification does not say which fired, so evict every global id
         that could be it. Over-eviction here only drops memoization (the
         backend caches own the handles), so erring wide is free; erring narrow
         would keep serving a released handle. Reading [Device.devices] directly
         avoids triggering enumeration from inside a teardown path. *)
      Array.iter
        (fun (d : Device.t) ->
          if d.backend_id = backend_device_id then
            Spoc_framework.Guarded_cache.evict_device kernel_cache d.id)
        !Device.devices)

(** Compile a kernel from source, with caching *)
let compile_kernel (device : Device.t) ~(name : string) ~(source : string) :
    Kernel.t =
  let key =
    Spoc_framework.Compile_cache.make_key
    (* [id] is already globally unique across backends; the framework name is
         folded in so the key stays self-describing (and unambiguous even if a
         caller ever hands us a hand-built [Device.t]). [make_key] digests each
         component, so the separator cannot be spoofed. *)
      ~device:(Printf.sprintf "%s#%d" device.Device.framework device.Device.id)
      ~name
      ~source
      ()
  in
  Spoc_framework.Guarded_cache.find_or_build
    kernel_cache
    ~key
    ~device_id:device.Device.id
    (fun () -> Kernel.compile_cached device ~name ~source)

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
