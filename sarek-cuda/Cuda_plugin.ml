(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * CUDA Plugin - Backend Implementation
 *
 * Implements the unified Framework_sig.BACKEND interface for CUDA devices.
 * Extends the CUDA base with:
 * - Execution model discrimination (JIT)
 * - IR-based source generation via Sarek_ir_cuda
 * - Intrinsic registry support
 ******************************************************************************)

open Spoc_framework
open Spoc_framework_registry

(** Reuse the existing CUDA implementation *)
module Cuda_base = struct
  include Cuda_plugin_base.Cuda
end

(** Extend Framework_sig.kargs with CUDA-specific variant *)
type Framework_sig.kargs += Cuda_kargs of Cuda_base.Kernel.args

(** CUDA-specific intrinsic implementation *)
type cuda_intrinsic = {
  intr_name : string;
  intr_codegen : string;
  intr_convergence : Framework_sig.convergence;
}

(** Intrinsic registry for CUDA-specific intrinsics *)
module Cuda_intrinsics : Framework_sig.INTRINSIC_REGISTRY = struct
  type intrinsic_impl = cuda_intrinsic

  let table : (string, intrinsic_impl) Hashtbl.t = Hashtbl.create 64

  let register name impl = Hashtbl.replace table name impl

  let find name = Hashtbl.find_opt table name

  let list_all () =
    Hashtbl.fold (fun name _ acc -> name :: acc) table [] |> List.sort compare

  (* Register standard CUDA intrinsics *)
  let () =
    (* Thread indexing *)
    register
      "thread_id_x"
      {
        intr_name = "thread_id_x";
        intr_codegen = "threadIdx.x";
        intr_convergence = Framework_sig.Divergent;
      } ;
    register
      "thread_id_y"
      {
        intr_name = "thread_id_y";
        intr_codegen = "threadIdx.y";
        intr_convergence = Framework_sig.Divergent;
      } ;
    register
      "thread_id_z"
      {
        intr_name = "thread_id_z";
        intr_codegen = "threadIdx.z";
        intr_convergence = Framework_sig.Divergent;
      } ;
    register
      "block_id_x"
      {
        intr_name = "block_id_x";
        intr_codegen = "blockIdx.x";
        intr_convergence = Framework_sig.Uniform;
      } ;
    register
      "block_id_y"
      {
        intr_name = "block_id_y";
        intr_codegen = "blockIdx.y";
        intr_convergence = Framework_sig.Uniform;
      } ;
    register
      "block_id_z"
      {
        intr_name = "block_id_z";
        intr_codegen = "blockIdx.z";
        intr_convergence = Framework_sig.Uniform;
      } ;
    register
      "block_dim_x"
      {
        intr_name = "block_dim_x";
        intr_codegen = "blockDim.x";
        intr_convergence = Framework_sig.Uniform;
      } ;
    register
      "block_dim_y"
      {
        intr_name = "block_dim_y";
        intr_codegen = "blockDim.y";
        intr_convergence = Framework_sig.Uniform;
      } ;
    register
      "block_dim_z"
      {
        intr_name = "block_dim_z";
        intr_codegen = "blockDim.z";
        intr_convergence = Framework_sig.Uniform;
      } ;
    register
      "global_thread_id"
      {
        intr_name = "global_thread_id";
        intr_codegen = "(threadIdx.x + blockIdx.x * blockDim.x)";
        intr_convergence = Framework_sig.Divergent;
      } ;

    (* Synchronization *)
    register
      "block_barrier"
      {
        intr_name = "block_barrier";
        intr_codegen = "__syncthreads()";
        intr_convergence = Framework_sig.Sync;
      } ;
    register
      "warp_barrier"
      {
        intr_name = "warp_barrier";
        intr_codegen = "__syncwarp()";
        intr_convergence = Framework_sig.Sync;
      } ;
    register
      "memory_fence"
      {
        intr_name = "memory_fence";
        intr_codegen = "__threadfence()";
        intr_convergence = Framework_sig.Uniform;
      }
end

(** CUDA Backend - implements BACKEND *)
module Backend : Framework_sig.BACKEND = struct
  (* Include all of BACKEND from Cuda_base *)
  include Cuda_base

  (** Execution model: CUDA uses JIT compilation *)
  let execution_model = Framework_sig.JIT

  (** Generate CUDA source from Sarek IR.
      @param block Ignored - CUDA specifies block size at launch *)
  let generate_source ?block:_ (ir : Sarek_ir_types.kernel) : string option =
    try Some (Sarek_ir_cuda.generate_with_types ~types:ir.kern_types ir)
    with _ -> None

  (** Execute directly - not supported for JIT backend *)
  let execute_direct ~native_fn:_ ~ir:_ ~block:_ ~grid:_ _args =
    Cuda_error.raise_error
      (Cuda_error.unsupported_source_lang "direct execution")

  (** CUDA intrinsic registry *)
  module Intrinsics = Cuda_intrinsics

  (** {2 External Kernel Support} *)

  let supported_source_langs = [Framework_sig.CUDA_Source; Framework_sig.PTX]

  let get_current_dev caller =
    match Cuda_plugin_base.Cuda.Device.get_current_device () with
    | Some d -> d
    | None -> Cuda_error.raise_error (Cuda_error.no_device_selected caller)

  let bind_args wrapped_kargs kargs args =
    List.iteri
      (fun i arg ->
        match arg with
        | Framework_sig.RSA_Buffer {binder; _} -> binder wrapped_kargs i
        | Framework_sig.RSA_Int32 n -> Kernel.set_arg_int32 kargs i n
        | Framework_sig.RSA_Int64 n -> Kernel.set_arg_int64 kargs i n
        | Framework_sig.RSA_Float32 f -> Kernel.set_arg_float32 kargs i f
        | Framework_sig.RSA_Float64 f -> Kernel.set_arg_float64 kargs i f)
      args

  (** Execute external kernel from source *)
  let run_source ~source ~lang ~kernel_name ~block ~grid ~shared_mem args =
    match lang with
    | Framework_sig.CUDA_Source ->
        let dev = get_current_dev "run_source:CUDA" in
        let compiled = Kernel.compile_cached dev ~name:kernel_name ~source in
        let kargs = Kernel.create_args () in
        bind_args (Cuda_kargs kargs) kargs args ;
        let stream = Stream.default dev in
        Kernel.launch
          compiled
          ~args:kargs
          ~grid
          ~block
          ~shared_mem
          ~stream:(Some stream)
    | Framework_sig.PTX ->
        let dev = get_current_dev "run_source:PTX" in
        let compiled = Kernel.load_from_ptx ~name:kernel_name ~ptx:source in
        let kargs = Kernel.create_args () in
        bind_args (Cuda_kargs kargs) kargs args ;
        let stream = Stream.default dev in
        Kernel.launch
          compiled
          ~args:kargs
          ~grid
          ~block
          ~shared_mem
          ~stream:(Some stream)
    | Framework_sig.OpenCL_Source ->
        Cuda_error.raise_error
          (Cuda_error.unsupported_source_lang "OpenCL_Source")
    | Framework_sig.SPIR_V ->
        Cuda_error.raise_error (Cuda_error.unsupported_source_lang "SPIR_V")
    | Framework_sig.GLSL_Source ->
        Cuda_error.raise_error
          (Cuda_error.unsupported_source_lang "GLSL_Source")

  let wrap_kargs args = Cuda_kargs args

  let unwrap_kargs = function Cuda_kargs args -> Some args | _ -> None
end

(** Check if backend is disabled via environment variable. Checked at runtime to
    allow SPOC_DISABLE_* to work without rebuild. *)
let is_disabled () =
  Sys.getenv_opt "SPOC_DISABLE_GPU" = Some "1"
  || Sys.getenv_opt "SPOC_DISABLE_CUDA" = Some "1"

(** Backend registration - happens once when first needed *)
let registered_backend =
  lazy
    (Spoc_core.Log.debug
       Spoc_core.Log.Device
       "Cuda_plugin: checking availability" ;
     if Backend.is_available () then begin
       Spoc_core.Log.debug
         Spoc_core.Log.Device
         "Cuda_plugin: CUDA available, registering backend" ;
       Framework_registry.register_backend
         ~priority:100
         (module Backend : Framework_sig.BACKEND)
     end
     else
       Spoc_core.Log.debug
         Spoc_core.Log.Device
         "Cuda_plugin: CUDA not available")

(** Auto-register backend when module is loaded, unless disabled *)
let () = if not (is_disabled ()) then Lazy.force registered_backend

(** Force module initialization *)
let init () = if not (is_disabled ()) then Lazy.force registered_backend

(** {1 Additional CUDA-specific Functions} *)

(** Register a custom CUDA intrinsic *)
let register_intrinsic = Cuda_intrinsics.register

(** Look up a CUDA intrinsic *)
let find_intrinsic = Cuda_intrinsics.find

(** Generate CUDA source with custom types *)
let generate_with_types = Sarek_ir_cuda.generate_with_types

(** Generate CUDA source for a kernel *)
let generate_source = Sarek_ir_cuda.generate
