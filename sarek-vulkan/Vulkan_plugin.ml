(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Vulkan Plugin - Backend Implementation
 *
 * Implements the unified Framework_sig.BACKEND interface for Vulkan devices.
 * Features:
 * - Execution model: JIT (GLSL -> SPIR-V -> Vulkan pipeline)
 * - IR-based source generation via Sarek_ir_glsl
 * - Intrinsic registry support for Vulkan-specific intrinsics
 *
 * Requirements:
 * - libvulkan.so.1 at runtime
 * - glslangValidator in PATH for GLSL to SPIR-V compilation
 ******************************************************************************)

open Spoc_framework
open Spoc_framework_registry

(** Reuse the existing Vulkan implementation *)
module Vulkan_base = struct
  include Vulkan_plugin_base.Vulkan
end

(** Extend Framework_sig.kargs with Vulkan-specific variant *)
type Framework_sig.kargs += Vulkan_kargs of Vulkan_base.Kernel.args

(** Vulkan-specific intrinsic implementation *)
type vulkan_intrinsic = {
  intr_name : string;
  intr_codegen : string;
  intr_convergence : Framework_sig.convergence;
}

(** Intrinsic registry for Vulkan-specific intrinsics *)
module Vulkan_intrinsics : Framework_sig.INTRINSIC_REGISTRY = struct
  type intrinsic_impl = vulkan_intrinsic

  let table : (string, intrinsic_impl) Hashtbl.t = Hashtbl.create 64

  let register name impl = Hashtbl.replace table name impl

  let find name = Hashtbl.find_opt table name

  let list_all () =
    Hashtbl.fold (fun name _ acc -> name :: acc) table [] |> List.sort compare

  (* Register standard Vulkan/GLSL compute intrinsics *)
  let () =
    (* Thread indexing - GLSL compute uses gl_GlobalInvocationID, etc. *)
    register
      "thread_id_x"
      {
        intr_name = "thread_id_x";
        intr_codegen = "gl_LocalInvocationID.x";
        intr_convergence = Framework_sig.Divergent;
      } ;
    register
      "thread_id_y"
      {
        intr_name = "thread_id_y";
        intr_codegen = "gl_LocalInvocationID.y";
        intr_convergence = Framework_sig.Divergent;
      } ;
    register
      "thread_id_z"
      {
        intr_name = "thread_id_z";
        intr_codegen = "gl_LocalInvocationID.z";
        intr_convergence = Framework_sig.Divergent;
      } ;
    register
      "block_id_x"
      {
        intr_name = "block_id_x";
        intr_codegen = "gl_WorkGroupID.x";
        intr_convergence = Framework_sig.Uniform;
      } ;
    register
      "block_id_y"
      {
        intr_name = "block_id_y";
        intr_codegen = "gl_WorkGroupID.y";
        intr_convergence = Framework_sig.Uniform;
      } ;
    register
      "block_id_z"
      {
        intr_name = "block_id_z";
        intr_codegen = "gl_WorkGroupID.z";
        intr_convergence = Framework_sig.Uniform;
      } ;
    register
      "block_dim_x"
      {
        intr_name = "block_dim_x";
        intr_codegen = "gl_WorkGroupSize.x";
        intr_convergence = Framework_sig.Uniform;
      } ;
    register
      "block_dim_y"
      {
        intr_name = "block_dim_y";
        intr_codegen = "gl_WorkGroupSize.y";
        intr_convergence = Framework_sig.Uniform;
      } ;
    register
      "block_dim_z"
      {
        intr_name = "block_dim_z";
        intr_codegen = "gl_WorkGroupSize.z";
        intr_convergence = Framework_sig.Uniform;
      } ;
    register
      "grid_dim_x"
      {
        intr_name = "grid_dim_x";
        intr_codegen = "gl_NumWorkGroups.x";
        intr_convergence = Framework_sig.Uniform;
      } ;
    register
      "grid_dim_y"
      {
        intr_name = "grid_dim_y";
        intr_codegen = "gl_NumWorkGroups.y";
        intr_convergence = Framework_sig.Uniform;
      } ;
    register
      "grid_dim_z"
      {
        intr_name = "grid_dim_z";
        intr_codegen = "gl_NumWorkGroups.z";
        intr_convergence = Framework_sig.Uniform;
      } ;
    register
      "global_thread_id"
      {
        intr_name = "global_thread_id";
        intr_codegen = "gl_GlobalInvocationID.x";
        intr_convergence = Framework_sig.Divergent;
      } ;
    register
      "global_size"
      {
        intr_name = "global_size";
        intr_codegen = "(gl_WorkGroupSize.x * gl_NumWorkGroups.x)";
        intr_convergence = Framework_sig.Uniform;
      } ;

    (* Synchronization *)
    register
      "block_barrier"
      {
        intr_name = "block_barrier";
        intr_codegen = "barrier()";
        intr_convergence = Framework_sig.Sync;
      } ;
    register
      "memory_fence"
      {
        intr_name = "memory_fence";
        intr_codegen = "memoryBarrier()";
        intr_convergence = Framework_sig.Uniform;
      }
end

(** Vulkan Backend - implements BACKEND *)
module Backend : Framework_sig.BACKEND = struct
  (* Include all of BACKEND from Vulkan_base *)
  include Vulkan_base

  (** Execution model: Vulkan uses JIT compilation (GLSL -> SPIR-V) *)
  let execution_model = Framework_sig.JIT

  (** Generate GLSL source from Sarek IR.
      @param block Workgroup dimensions - required for correct GLSL local_size
  *)
  let generate_source ?block (ir : Sarek_ir_types.kernel) : string option =
    let block_tuple =
      match block with
      | Some b -> Some (b.Framework_sig.x, b.Framework_sig.y, b.Framework_sig.z)
      | None -> None
    in
    try
      Some
        (Sarek_ir_glsl.generate_with_types
           ?block:block_tuple
           ~log:(fun s -> Spoc_core.Log.debugf Spoc_core.Log.Device "%s" s)
           ~types:ir.kern_types
           ir)
    with _ -> None

  (** Execute directly - not supported for JIT backend *)
  let execute_direct ~native_fn:_ ~ir:_ ~block:_ ~grid:_ _args =
    Vulkan_error.raise_error
      (Vulkan_error.feature_not_supported
         "direct execution (JIT backend only supports code generation)")

  (** Vulkan intrinsic registry *)
  module Intrinsics = Vulkan_intrinsics

  (** {2 External Kernel Support} *)

  (** Supported source languages: GLSL compute shaders *)
  let supported_source_langs = [Framework_sig.GLSL_Source]

  (** Execute external kernel from source *)
  let run_source ~source ~lang ~kernel_name ~block ~grid ~shared_mem args =
    ignore shared_mem ;
    match lang with
    | Framework_sig.GLSL_Source ->
        (* Get current device (must be set by Execute before calling) *)
        let dev =
          match Vulkan_plugin_base.Vulkan.Device.get_current_device () with
          | Some d -> d
          | None ->
              Vulkan_error.raise_error
                (Vulkan_error.no_device_selected "run_source")
        in

        (* Compile GLSL to pipeline *)
        let compiled = Kernel.compile_cached dev ~name:kernel_name ~source in

        (* Set up kernel arguments using typed run_source_arg list *)
        let kargs = Kernel.create_args () in
        let wrapped_kargs = Vulkan_kargs kargs in
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
    | Framework_sig.SPIR_V ->
        Vulkan_error.raise_error
          (Vulkan_error.feature_not_supported "SPIR-V direct loading")
    | Framework_sig.CUDA_Source ->
        Vulkan_error.raise_error (Vulkan_error.unsupported_source_lang "CUDA")
    | Framework_sig.OpenCL_Source ->
        Vulkan_error.raise_error (Vulkan_error.unsupported_source_lang "OpenCL")
    | Framework_sig.PTX ->
        Vulkan_error.raise_error (Vulkan_error.unsupported_source_lang "PTX")

  let wrap_kargs args = Vulkan_kargs args

  let unwrap_kargs = function Vulkan_kargs args -> Some args | _ -> None
end

(** Check if backend is disabled via environment variable. Checked at runtime to
    allow SPOC_DISABLE_* to work without rebuild. *)
let is_disabled () =
  Sys.getenv_opt "SPOC_DISABLE_GPU" = Some "1"
  || Sys.getenv_opt "SPOC_DISABLE_VULKAN" = Some "1"

(** Backend registration - happens once when first needed *)
let registered_backend =
  lazy
    (Spoc_core.Log.debug
       Spoc_core.Log.Device
       "Vulkan_plugin: checking availability" ;
     if Backend.is_available () then begin
       Spoc_core.Log.debug
         Spoc_core.Log.Device
         "Vulkan_plugin: Vulkan available, registering backend" ;
       try
         Framework_registry.register_backend
           ~priority:80
           (module Backend : Framework_sig.BACKEND) ;
         Spoc_core.Log.debug
           Spoc_core.Log.Device
           "Vulkan_plugin: registration succeeded"
       with exn ->
         Spoc_core.Log.debugf
           Spoc_core.Log.Device
           "Vulkan_plugin: registration failed with exception: %s"
           (Printexc.to_string exn)
     end
     else
       Spoc_core.Log.debug
         Spoc_core.Log.Device
         "Vulkan_plugin: Vulkan not available (missing libvulkan or \
          glslangValidator)")

(** Auto-register backend when module is loaded, unless disabled *)
let () = if not (is_disabled ()) then Lazy.force registered_backend

(** Force module initialization *)
let init () = if not (is_disabled ()) then Lazy.force registered_backend

(** {1 Additional Vulkan-specific Functions} *)

(** Register a custom Vulkan intrinsic *)
let register_intrinsic = Vulkan_intrinsics.register

(** Look up a Vulkan intrinsic *)
let find_intrinsic = Vulkan_intrinsics.find

(** Generate GLSL source with custom types *)
let generate_with_types = Sarek_ir_glsl.generate_with_types

(** Generate GLSL source for a kernel *)
let generate_source = Sarek_ir_glsl.generate

(** Check if glslangValidator is available *)
let glslang_available = Vulkan_api.glslang_available

(** Compile GLSL to SPIR-V *)
let compile_glsl_to_spirv = Vulkan_api.compile_glsl_to_spirv
