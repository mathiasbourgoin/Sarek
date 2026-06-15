(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * OpenCL Plugin - Backend Implementation
 *
 * Implements the unified Framework_sig.BACKEND interface for OpenCL devices.
 * Extends the OpenCL base with:
 * - Execution model discrimination (JIT)
 * - IR-based source generation via Sarek_ir_opencl
 * - Intrinsic registry support
 ******************************************************************************)

open Spoc_framework
open Spoc_framework_registry

(** Reuse the existing OpenCL implementation *)
module Opencl_base = struct
  include Opencl_plugin_base.Opencl
end

(** Extend Framework_sig.kargs with OpenCL-specific variant *)
type Framework_sig.kargs += Opencl_kargs of Opencl_base.Kernel.args

(** OpenCL-specific intrinsic implementation *)
type opencl_intrinsic = {
  intr_name : string;
  intr_codegen : string;
  intr_convergence : Framework_sig.convergence;
}

(** Intrinsic registry for OpenCL-specific intrinsics *)
module Opencl_intrinsics : Framework_sig.INTRINSIC_REGISTRY = struct
  type intrinsic_impl = opencl_intrinsic

  let table : (string, intrinsic_impl) Hashtbl.t = Hashtbl.create 64

  let register name impl = Hashtbl.replace table name impl

  let find name = Hashtbl.find_opt table name

  let list_all () =
    Hashtbl.fold (fun name _ acc -> name :: acc) table [] |> List.sort compare

  (* Register standard OpenCL intrinsics *)
  let () =
    (* Thread indexing - OpenCL uses get_local_id, get_group_id, etc. *)
    register
      "thread_id_x"
      {
        intr_name = "thread_id_x";
        intr_codegen = "get_local_id(0)";
        intr_convergence = Framework_sig.Divergent;
      } ;
    register
      "thread_id_y"
      {
        intr_name = "thread_id_y";
        intr_codegen = "get_local_id(1)";
        intr_convergence = Framework_sig.Divergent;
      } ;
    register
      "thread_id_z"
      {
        intr_name = "thread_id_z";
        intr_codegen = "get_local_id(2)";
        intr_convergence = Framework_sig.Divergent;
      } ;
    register
      "block_id_x"
      {
        intr_name = "block_id_x";
        intr_codegen = "get_group_id(0)";
        intr_convergence = Framework_sig.Uniform;
      } ;
    register
      "block_id_y"
      {
        intr_name = "block_id_y";
        intr_codegen = "get_group_id(1)";
        intr_convergence = Framework_sig.Uniform;
      } ;
    register
      "block_id_z"
      {
        intr_name = "block_id_z";
        intr_codegen = "get_group_id(2)";
        intr_convergence = Framework_sig.Uniform;
      } ;
    register
      "block_dim_x"
      {
        intr_name = "block_dim_x";
        intr_codegen = "get_local_size(0)";
        intr_convergence = Framework_sig.Uniform;
      } ;
    register
      "block_dim_y"
      {
        intr_name = "block_dim_y";
        intr_codegen = "get_local_size(1)";
        intr_convergence = Framework_sig.Uniform;
      } ;
    register
      "block_dim_z"
      {
        intr_name = "block_dim_z";
        intr_codegen = "get_local_size(2)";
        intr_convergence = Framework_sig.Uniform;
      } ;
    register
      "grid_dim_x"
      {
        intr_name = "grid_dim_x";
        intr_codegen = "get_num_groups(0)";
        intr_convergence = Framework_sig.Uniform;
      } ;
    register
      "grid_dim_y"
      {
        intr_name = "grid_dim_y";
        intr_codegen = "get_num_groups(1)";
        intr_convergence = Framework_sig.Uniform;
      } ;
    register
      "grid_dim_z"
      {
        intr_name = "grid_dim_z";
        intr_codegen = "get_num_groups(2)";
        intr_convergence = Framework_sig.Uniform;
      } ;
    register
      "global_thread_id"
      {
        intr_name = "global_thread_id";
        intr_codegen = "get_global_id(0)";
        intr_convergence = Framework_sig.Divergent;
      } ;
    register
      "global_size"
      {
        intr_name = "global_size";
        intr_codegen = "get_global_size(0)";
        intr_convergence = Framework_sig.Uniform;
      } ;

    (* Synchronization *)
    register
      "block_barrier"
      {
        intr_name = "block_barrier";
        intr_codegen = "barrier(CLK_LOCAL_MEM_FENCE)";
        intr_convergence = Framework_sig.Sync;
      } ;
    register
      "warp_barrier"
      {
        intr_name = "warp_barrier";
        intr_codegen = "sub_group_barrier(CLK_LOCAL_MEM_FENCE)";
        intr_convergence = Framework_sig.Sync;
      } ;
    register
      "memory_fence"
      {
        intr_name = "memory_fence";
        intr_codegen = "mem_fence(CLK_GLOBAL_MEM_FENCE)";
        intr_convergence = Framework_sig.Uniform;
      }
end

(** OpenCL Backend - implements BACKEND *)
module Backend : Framework_sig.BACKEND = struct
  (* Include all of BACKEND from Opencl_base *)
  include Opencl_base

  (** Execution model: OpenCL uses JIT compilation *)
  let execution_model = Framework_sig.JIT

  (** Generate OpenCL source from Sarek IR.
      @param block Ignored - OpenCL specifies work-group size at launch *)
  let generate_source ?block:_ (ir : Sarek_ir_types.kernel) : string option =
    try
      let source =
        Sarek_ir_opencl.generate_with_types ~types:ir.kern_types ir
      in
      (* Add FP64 pragma if kernel uses double precision *)
      let source =
        if Sarek_ir_analysis.kernel_uses_float64 ir then
          "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n\n" ^ source
        else source
      in
      Spoc_core.Log.debug Kernel ("OpenCL source:\n" ^ source) ;
      Some source
    with _ -> None

  (** Execute directly - not supported for JIT backend *)
  let execute_direct ~native_fn:_ ~ir:_ ~block:_ ~grid:_ _args =
    Opencl_error.raise_error
      (Opencl_error.unsupported_source_lang "direct execution")

  (** OpenCL intrinsic registry *)
  module Intrinsics = Opencl_intrinsics

  (** {2 External Kernel Support} *)

  (** Supported source languages: OpenCL only *)
  let supported_source_langs = [Framework_sig.OpenCL_Source]

  (** Execute external kernel from source *)
  let run_source ~source ~lang ~kernel_name ~block ~grid ~shared_mem args =
    match lang with
    | Framework_sig.OpenCL_Source ->
        (* Get current device (must be set by Execute before calling) *)
        let dev =
          match Opencl_plugin_base.Opencl.Device.get_current_device () with
          | Some d -> d
          | None ->
              Opencl_error.raise_error
                (Opencl_error.no_device_selected "run_source")
        in

        (* Compile and get kernel *)
        let compiled = Kernel.compile_cached dev ~name:kernel_name ~source in

        (* Set up kernel arguments using typed run_source_arg list *)
        let kargs = Kernel.create_args () in
        let wrapped_kargs = Opencl_kargs kargs in
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
          ~shared_mem
          ~stream:(Some stream)
    | Framework_sig.CUDA_Source ->
        Opencl_error.raise_error
          (Opencl_error.unsupported_source_lang "CUDA_Source")
    | Framework_sig.PTX ->
        Opencl_error.raise_error (Opencl_error.unsupported_source_lang "PTX")
    | Framework_sig.SPIR_V ->
        Opencl_error.raise_error (Opencl_error.unsupported_source_lang "SPIR_V")
    | Framework_sig.GLSL_Source ->
        Opencl_error.raise_error
          (Opencl_error.unsupported_source_lang "GLSL_Source")

  let wrap_kargs args = Opencl_kargs args

  let unwrap_kargs = function Opencl_kargs args -> Some args | _ -> None
end

(** Check if backend is disabled via environment variable. Checked at runtime to
    allow SPOC_DISABLE_* to work without rebuild. *)
let is_disabled () =
  Sys.getenv_opt "SPOC_DISABLE_GPU" = Some "1"
  || Sys.getenv_opt "SPOC_DISABLE_OPENCL" = Some "1"

(** Backend registration - happens once when first needed *)
let registered_backend =
  lazy
    (if Backend.is_available () then
       Framework_registry.register_backend
         ~priority:90
         (module Backend : Framework_sig.BACKEND))

(** Auto-register backend when module is loaded, unless disabled *)
let () = if not (is_disabled ()) then Lazy.force registered_backend

(** Force module initialization *)
let init () = if not (is_disabled ()) then Lazy.force registered_backend

(** {1 Additional OpenCL-specific Functions} *)

(** Register a custom OpenCL intrinsic *)
let register_intrinsic = Opencl_intrinsics.register

(** Look up an OpenCL intrinsic *)
let find_intrinsic = Opencl_intrinsics.find

(** Generate OpenCL source with custom types *)
let generate_with_types = Sarek_ir_opencl.generate_with_types

(** Generate OpenCL source for a kernel *)
let generate_source = Sarek_ir_opencl.generate

(** Generate OpenCL source with FP64 extension if needed *)
let generate_with_fp64 = Sarek_ir_opencl.generate_with_fp64
