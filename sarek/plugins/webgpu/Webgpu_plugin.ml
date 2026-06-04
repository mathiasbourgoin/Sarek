(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * WebGPU Plugin Backend - Inert Skeleton
 *
 * Implements the Framework_sig.BACKEND interface as an inert stub for the
 * in-browser WebGPU backend.  is_available () returns false so Device.init
 * never selects this backend during native execution.  All operations raise
 * "Spoc_webgpu: not yet implemented" - the real in-browser WebGPU runtime
 * is forthcoming.
 ******************************************************************************)

open Spoc_framework
open Spoc_framework_registry

let not_impl () =
  failwith
    "Spoc_webgpu: not yet implemented (in-browser WebGPU runtime is \
     forthcoming)"

(** Extend Framework_sig.kargs with WebGPU-specific variant *)
type Framework_sig.kargs += Webgpu_kargs of unit

(** WebGPU intrinsic registry - empty stub *)
module Webgpu_intrinsics : Framework_sig.INTRINSIC_REGISTRY = struct
  type intrinsic_impl = unit

  let table : (string, intrinsic_impl) Hashtbl.t = Hashtbl.create 4

  let register name impl = Hashtbl.replace table name impl

  let find name = Hashtbl.find_opt table name

  let list_all () =
    Hashtbl.fold (fun name _ acc -> name :: acc) table [] |> List.sort compare
end

(** WebGPU Backend - inert stub implementing Framework_sig.BACKEND *)
module Backend : Framework_sig.BACKEND = struct
  let name = "WebGPU"

  let version = (0, 1, 0)

  let is_available () = false

  module Device = struct
    type t = unit

    type id = int

    let init () = not_impl ()

    let count () = not_impl ()

    let get _i = not_impl ()

    let id _d = not_impl ()

    let name _d = not_impl ()

    let capabilities _d = not_impl ()

    let set_current _d = not_impl ()

    let synchronize _d = not_impl ()
  end

  module Stream = struct
    type t = unit

    let create _d = not_impl ()

    let destroy _s = not_impl ()

    let synchronize _s = not_impl ()

    let default _d = not_impl ()
  end

  module Memory = struct
    type 'a buffer = unit

    let alloc _d _n _kind = not_impl ()

    let alloc_custom _d ~size:_ ~elem_size:_ = not_impl ()

    let alloc_zero_copy _d _arr _kind = not_impl ()

    let free _buf = not_impl ()

    let host_to_device ~src:_ ~dst:_ = not_impl ()

    let device_to_host ~src:_ ~dst:_ = not_impl ()

    let host_ptr_to_device ~src_ptr:_ ~byte_size:_ ~dst:_ = not_impl ()

    let device_to_host_ptr ~src:_ ~dst_ptr:_ ~byte_size:_ = not_impl ()

    let device_to_device ~src:_ ~dst:_ = not_impl ()

    let size _buf = not_impl ()

    let device_ptr _buf = not_impl ()

    let is_zero_copy _buf = not_impl ()
  end

  module Event = struct
    type t = unit

    let create () = not_impl ()

    let destroy _e = not_impl ()

    let record _e _s = not_impl ()

    let synchronize _e = not_impl ()

    let elapsed ~start:_ ~stop:_ = not_impl ()
  end

  module Kernel = struct
    type t = unit

    type args = unit

    let compile _d ~name:_ ~source:_ = not_impl ()

    let compile_cached _d ~name:_ ~source:_ = not_impl ()

    let clear_cache () = not_impl ()

    let create_args () = not_impl ()

    let set_arg_buffer _a _i _buf = not_impl ()

    let set_arg_int32 _a _i _v = not_impl ()

    let set_arg_int64 _a _i _v = not_impl ()

    let set_arg_float32 _a _i _v = not_impl ()

    let set_arg_float64 _a _i _v = not_impl ()

    let set_arg_ptr _a _i _p = not_impl ()

    let launch _k ~args:_ ~grid:_ ~block:_ ~shared_mem:_ ~stream:_ = not_impl ()
  end

  let enable_profiling () = not_impl ()

  let disable_profiling () = not_impl ()

  (** Execution model: JIT - WebGPU compiles WGSL shaders at runtime *)
  let execution_model = Framework_sig.JIT

  (** Generate source - WGSL codegen not yet wired; returns None *)
  let generate_source ?block:_ (_ir : Sarek_ir_types.kernel) : string option =
    None

  (** Direct execution not applicable to JIT backend *)
  let execute_direct ~native_fn:_ ~ir:_ ~block:_ ~grid:_
      (_args : Framework_sig.exec_arg array) =
    not_impl ()

  module Intrinsics = Webgpu_intrinsics

  let supported_source_langs = []

  let run_source ~source:_ ~lang:_ ~kernel_name:_ ~block:_ ~grid:_ ~shared_mem:_
      (_args : Framework_sig.run_source_arg list) =
    not_impl ()

  let wrap_kargs args = Webgpu_kargs args

  let unwrap_kargs = function Webgpu_kargs args -> Some args | _ -> None
end

(** Auto-register backend when module is loaded. Priority 1 - lower than Native
    (10) and Interpreter (5). Guarded by is_available () so native Device.init
    never selects this. *)
let registered_backend =
  lazy
    (if Backend.is_available () then
       Framework_registry.register_backend
         ~priority:1
         (module Backend : Framework_sig.BACKEND))

let () = Lazy.force registered_backend

(** Force module initialization *)
let init () = Lazy.force registered_backend
