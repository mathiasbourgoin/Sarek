(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Metal Plugin - Backend Implementation
 *
 * Implements the unified Framework_sig.BACKEND interface for Metal devices.
 * Extends the Metal base with:
 * - Execution model discrimination (JIT)
 * - IR-based source generation via Sarek_ir_metal
 * - Intrinsic registry support
 ******************************************************************************)

open Spoc_framework
open Spoc_framework_registry

(** Reuse the existing Metal implementation *)
module Metal_base = struct
  include Metal_plugin_base.Metal
end

(** Extend Framework_sig.kargs with Metal-specific variant *)
type Framework_sig.kargs += Metal_kargs of Metal_base.Kernel.args

(** Metal-specific intrinsic implementation *)
type metal_intrinsic = {
  intr_name : string;
  intr_codegen : string;
  intr_convergence : Framework_sig.convergence;
}

(** Intrinsic registry for Metal-specific intrinsics *)
module Metal_intrinsics : Framework_sig.INTRINSIC_REGISTRY = struct
  type intrinsic_impl = metal_intrinsic

  let table : (string, intrinsic_impl) Hashtbl.t = Hashtbl.create 64

  let register name impl = Hashtbl.replace table name impl

  let find name = Hashtbl.find_opt table name

  let list_all () =
    Hashtbl.fold (fun name _ acc -> name :: acc) table [] |> List.sort compare

  (* Register standard Metal intrinsics *)
  let () =
    (* Thread indexing - Metal uses thread_position_in_threadgroup *)
    register
      "thread_id_x"
      {
        intr_name = "thread_id_x";
        intr_codegen = "tid.x";
        intr_convergence = Framework_sig.Divergent;
      } ;
    register
      "thread_id_y"
      {
        intr_name = "thread_id_y";
        intr_codegen = "tid.y";
        intr_convergence = Framework_sig.Divergent;
      } ;
    register
      "thread_id_z"
      {
        intr_name = "thread_id_z";
        intr_codegen = "tid.z";
        intr_convergence = Framework_sig.Divergent;
      } ;
    register
      "block_id_x"
      {
        intr_name = "block_id_x";
        intr_codegen = "bid.x";
        intr_convergence = Framework_sig.Uniform;
      } ;
    register
      "block_id_y"
      {
        intr_name = "block_id_y";
        intr_codegen = "bid.y";
        intr_convergence = Framework_sig.Uniform;
      } ;
    register
      "block_id_z"
      {
        intr_name = "block_id_z";
        intr_codegen = "bid.z";
        intr_convergence = Framework_sig.Uniform;
      } ;
    register
      "block_dim_x"
      {
        intr_name = "block_dim_x";
        intr_codegen = "tpg.x";
        intr_convergence = Framework_sig.Uniform;
      } ;
    register
      "block_dim_y"
      {
        intr_name = "block_dim_y";
        intr_codegen = "tpg.y";
        intr_convergence = Framework_sig.Uniform;
      } ;
    register
      "block_dim_z"
      {
        intr_name = "block_dim_z";
        intr_codegen = "tpg.z";
        intr_convergence = Framework_sig.Uniform;
      } ;
    register
      "grid_dim_x"
      {
        intr_name = "grid_dim_x";
        intr_codegen = "grid.x";
        intr_convergence = Framework_sig.Uniform;
      } ;
    register
      "grid_dim_y"
      {
        intr_name = "grid_dim_y";
        intr_codegen = "grid.y";
        intr_convergence = Framework_sig.Uniform;
      } ;
    register
      "grid_dim_z"
      {
        intr_name = "grid_dim_z";
        intr_codegen = "grid.z";
        intr_convergence = Framework_sig.Uniform;
      } ;
    register
      "global_thread_id"
      {
        intr_name = "global_thread_id";
        intr_codegen = "gid.x";
        intr_convergence = Framework_sig.Divergent;
      } ;

    (* Synchronization *)
    register
      "block_barrier"
      {
        intr_name = "block_barrier";
        intr_codegen = "threadgroup_barrier(mem_flags::mem_threadgroup)";
        intr_convergence = Framework_sig.Sync;
      } ;
    register
      "warp_barrier"
      {
        intr_name = "warp_barrier";
        intr_codegen = "simdgroup_barrier(mem_flags::mem_threadgroup)";
        intr_convergence = Framework_sig.Sync;
      } ;
    register
      "memory_fence"
      {
        intr_name = "memory_fence";
        intr_codegen = "threadgroup_barrier(mem_flags::mem_device)";
        intr_convergence = Framework_sig.Uniform;
      }
end

(** Metal Backend - implements BACKEND *)
module Backend : Framework_sig.BACKEND = struct
  (* Include all of BACKEND from Metal_base *)
  include Metal_base

  (** Execution model: Metal uses JIT compilation *)
  let execution_model = Framework_sig.JIT

  (** Generate Metal Shading Language source from Sarek IR.
      @param block Ignored - Metal specifies work-group size at launch *)
  let generate_source ?block:_ (ir : Sarek_ir_types.kernel) : string option =
    try
      let source = Sarek_ir_metal.generate_with_types ~types:ir.kern_types ir in
      Spoc_core.Log.debug Kernel ("Metal source:\n" ^ source) ;
      Some source
    with _ -> None

  (** Execute directly - not supported for JIT backend *)
  let execute_direct ~native_fn:_ ~ir:_ ~block:_ ~grid:_ _args =
    Metal_error.raise_error
      (Metal_error.feature_not_supported
         "direct execution (JIT backend only supports code generation)")

  (** Metal intrinsic registry *)
  module Intrinsics = Metal_intrinsics

  (** {2 External Kernel Support} *)

  (** Supported source languages: Metal only *)
  let supported_source_langs = [Framework_sig.OpenCL_Source]
  (* Note: Using OpenCL_Source for now as placeholder *)

  (** Execute external kernel from source *)
  let run_source ~source ~lang:_ ~kernel_name ~block ~grid ~shared_mem:_ args =
    (* Get current device (must be set by Execute before calling) *)
    let dev =
      match Metal_plugin_base.Metal.Device.get_current_device () with
      | Some d -> d
      | None ->
          Metal_error.raise_error (Metal_error.no_device_selected "run_source")
    in

    (* Compile and get kernel *)
    let compiled = Kernel.compile_cached dev ~name:kernel_name ~source in

    (* Set up kernel arguments *)
    let kargs = Kernel.create_args () in
    let wrapped_kargs = Metal_kargs kargs in
    List.iteri
      (fun i arg ->
        match arg with
        | Framework_sig.RSA_Buffer {binder; _} -> binder wrapped_kargs i
        | Framework_sig.RSA_Int32 n -> Kernel.set_arg_int32 kargs i n
        | Framework_sig.RSA_Int64 n -> Kernel.set_arg_int64 kargs i n
        | Framework_sig.RSA_Float32 f -> Kernel.set_arg_float32 kargs i f
        | Framework_sig.RSA_Float64 f -> Kernel.set_arg_float64 kargs i f)
      args ;

    (* Launch *)
    let stream = Stream.default dev in
    Kernel.launch
      compiled
      ~args:kargs
      ~grid
      ~block
      ~shared_mem:0
      ~stream:(Some stream)

  let wrap_kargs args = Metal_kargs args

  let unwrap_kargs = function Metal_kargs args -> Some args | _ -> None
end

(** Check if backend is disabled via environment variable *)
let is_disabled () =
  Sys.getenv_opt "SPOC_DISABLE_GPU" = Some "1"
  || Sys.getenv_opt "SPOC_DISABLE_METAL" = Some "1"

(** Backend registration - happens once when first needed *)
let registered_backend =
  lazy
    (if Backend.is_available () then
       Framework_registry.register_backend
         ~priority:95
         (* Higher priority than OpenCL on macOS *)
         (module Backend : Framework_sig.BACKEND)
     else ())

(** Auto-register backend when module is loaded, unless disabled *)
let () = if not (is_disabled ()) then Lazy.force registered_backend

(** Force module initialization *)
let init () = if not (is_disabled ()) then Lazy.force registered_backend

(** {1 Additional Metal-specific Functions} *)

(** Register a custom Metal intrinsic *)
let register_intrinsic = Metal_intrinsics.register

(** Look up a Metal intrinsic *)
let find_intrinsic = Metal_intrinsics.find

(** Generate Metal source with custom types *)
let generate_with_types = Sarek_ir_metal.generate_with_types

(** Generate Metal source for a kernel *)
let generate_source = Sarek_ir_metal.generate
