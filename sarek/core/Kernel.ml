(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Sarek Runtime - Kernel Execution
 *
 * Provides unified kernel compilation and execution across backends.
 * Uses first-class modules to wrap backend-specific handles.
 ******************************************************************************)

open Spoc_framework
open Spoc_framework_registry

(** {1 Kernel Module Type} *)

(** A compiled kernel packages backend-specific kernel with its operations *)
module type KERNEL = sig
  val device : Device.t

  val name : string

  val launch :
    args:Framework_sig.kargs ->
    grid:Framework_sig.dims ->
    block:Framework_sig.dims ->
    shared_mem:int ->
    unit
end

type t = Kernel : (module KERNEL) -> t

(** {1 Args Module Type} *)

(** Kernel arguments builder packages backend-specific args with setters *)
module type ARGS = sig
  val device : Device.t

  val kargs : Framework_sig.kargs

  val set_int32 : int -> int32 -> unit

  val set_int64 : int -> int64 -> unit

  val set_float32 : int -> float -> unit

  val set_float64 : int -> float -> unit

  val set_ptr : int -> nativeint -> unit
end

type args = Args : (module ARGS) -> args

(** {1 Compilation} *)

(** Compile a kernel from source *)
let compile (device : Device.t) ~(name : string) ~(source : string) : t =
  match Framework_registry.find_backend device.framework with
  | None -> failwith ("Unknown framework: " ^ device.framework)
  | Some (module B : Framework_sig.BACKEND) ->
      let dev = B.Device.get device.backend_id in
      let k = B.Kernel.compile dev ~name ~source in
      Kernel
        (module struct
          let device = device

          let name = name

          let launch ~args ~grid ~block ~shared_mem =
            match B.unwrap_kargs args with
            | Some a ->
                B.Kernel.launch k ~args:a ~grid ~block ~shared_mem ~stream:None
            | None -> failwith "launch: backend mismatch"
        end : KERNEL)

(** Compile with caching *)
let compile_cached (device : Device.t) ~(name : string) ~(source : string) : t =
  match Framework_registry.find_backend device.framework with
  | None -> failwith ("Unknown framework: " ^ device.framework)
  | Some (module B : Framework_sig.BACKEND) ->
      let dev = B.Device.get device.backend_id in
      let k = B.Kernel.compile_cached dev ~name ~source in
      Kernel
        (module struct
          let device = device

          let name = name

          let launch ~args ~grid ~block ~shared_mem =
            match B.unwrap_kargs args with
            | Some a ->
                B.Kernel.launch k ~args:a ~grid ~block ~shared_mem ~stream:None
            | None -> failwith "launch: backend mismatch"
        end : KERNEL)

(** {1 Arguments} *)

(** Create an arguments object *)
let create_args (device : Device.t) : args =
  match Framework_registry.find_backend device.framework with
  | None -> failwith ("Unknown framework: " ^ device.framework)
  | Some (module B : Framework_sig.BACKEND) ->
      let a = B.Kernel.create_args () in
      Args
        (module struct
          let device = device

          let kargs = B.wrap_kargs a

          let set_int32 idx v = B.Kernel.set_arg_int32 a idx v

          let set_int64 idx v = B.Kernel.set_arg_int64 a idx v

          let set_float32 idx v = B.Kernel.set_arg_float32 a idx v

          let set_float64 idx v = B.Kernel.set_arg_float64 a idx v

          let set_ptr idx ptr = B.Kernel.set_arg_ptr a idx ptr
        end : ARGS)

(** Set buffer argument *)
let set_arg_buffer (Args (module A)) (idx : int) (buf : _ Memory.buffer) : unit
    =
  Memory.bind_to_kargs buf A.kargs idx

(** Set int32 argument *)
let set_arg_int32 (Args (module A)) (idx : int) (v : int32) : unit =
  A.set_int32 idx v

(** Set int64 argument *)
let set_arg_int64 (Args (module A)) (idx : int) (v : int64) : unit =
  A.set_int64 idx v

(** Set float32 argument *)
let set_arg_float32 (Args (module A)) (idx : int) (v : float) : unit =
  A.set_float32 idx v

(** Set float64 argument *)
let set_arg_float64 (Args (module A)) (idx : int) (v : float) : unit =
  A.set_float64 idx v

(** Set raw device pointer argument (CUDA only) *)
let set_arg_ptr (Args (module A)) (idx : int) (ptr : nativeint) : unit =
  A.set_ptr idx ptr

(** {1 Execution} *)

(** Launch a kernel *)
let launch (Kernel (module K)) ~(args : args) ~(grid : Framework_sig.dims)
    ~(block : Framework_sig.dims) ?(shared_mem = 0) () : unit =
  let (Args (module A)) = args in
  K.launch ~args:A.kargs ~grid ~block ~shared_mem

(** {1 Cache Management} *)

(** Clear all kernel caches.

    The layers ABOVE this one are dropped first: [B.Kernel.clear_cache] releases
    the backend handles (clReleaseKernel / cuModuleUnload / ...), and any outer
    memo — notably [Runtime.kernel_cache] — holds [Kernel.t] closures over
    exactly those handles. Leaving them in place means the next lookup hits the
    outer memo and launches through a released handle (segfault).

    The notification is fired on BOTH sides of the backend clear, and that is
    load-bearing rather than belt-and-braces. The pre-notification drops what is
    already memoized; it cannot cover a build that starts after it. An outer
    build that misses after the first notification snapshots the {e new}
    generation, compiles a handle that [B.Kernel.clear_cache] then releases, and
    on re-acquire finds the generation unchanged — so it installs a value over a
    dead handle and poisons the memo for the rest of the process, which is the
    exact failure [Guarded_cache]'s [~invalidated_by_clear] exists to prevent.
    The post-notification bumps the generation again, so any build spanning the
    release is rejected. The window it closes is the whole duration of the
    backend clear.

    Both notifications are also isolated from the backend clear. [Cache_hooks]
    re-raises the first failing listener, and a listener is only dropping
    memoization it does not own — letting that escape before
    [B.Kernel.clear_cache] would skip the release this function exists to
    perform and leak every backend handle. This mirrors the discipline
    [Cache_hooks.mli] imposes on the device-destroy path: finish the teardown,
    then re-raise. *)
let clear_cache (device : Device.t) : unit =
  let listener_exn = ref None in
  let notify () =
    try Cache_hooks.notify_clear_all ()
    with e -> if Option.is_none !listener_exn then listener_exn := Some e
  in
  notify () ;
  Fun.protect ~finally:notify (fun () ->
      match Framework_registry.find_backend device.framework with
      | None -> ()
      | Some (module B : Framework_sig.BACKEND) -> B.Kernel.clear_cache ()) ;
  Option.iter raise !listener_exn

(** {1 Accessors} *)

(** Get kernel device *)
let device (Kernel (module K)) : Device.t = K.device

(** Get kernel name *)
let name (Kernel (module K)) : string = K.name

(** Get args device *)
let args_device (Args (module A)) : Device.t = A.device

(** Get wrapped kargs for direct backend use *)
let get_kargs (Args (module A)) : Framework_sig.kargs = A.kargs
