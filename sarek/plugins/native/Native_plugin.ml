(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Native Plugin Backend Implementation
 *
 * Implements the Framework_sig.BACKEND interface for CPU execution.
 * Extends Native_plugin with:
 * - Execution model discrimination (Direct)
 * - Direct execution of pre-compiled OCaml functions
 * - Intrinsic registry support
 *
 * This plugin coexists with Native_plugin during the transition period.
 ******************************************************************************)

open Spoc_framework
open Spoc_framework_registry

(** Reuse the existing Native implementation *)
module Native_base = struct
  include Native_plugin_base.Native
end

(** Extend Framework_sig.kargs with Native-specific variant *)
type Framework_sig.kargs += Native_kargs of Native_base.Kernel.args

(** Native-specific intrinsic implementation *)
type native_intrinsic = {
  intr_name : string;
  intr_codegen : string;
  intr_convergence : Framework_sig.convergence;
}

(** Intrinsic registry for Native-specific intrinsics *)
module Native_intrinsics : Framework_sig.INTRINSIC_REGISTRY = struct
  type intrinsic_impl = native_intrinsic

  let table : (string, intrinsic_impl) Hashtbl.t = Hashtbl.create 64

  let register name impl = Hashtbl.replace table name impl

  let find name = Hashtbl.find_opt table name

  let list_all () =
    Hashtbl.fold (fun name _ acc -> name :: acc) table [] |> List.sort compare

  (* Register standard Native intrinsics - these are placeholders since
     Native execution uses pre-compiled OCaml functions *)
  let () =
    (* Thread indexing - in Native, these come from thread state *)
    register
      "thread_id_x"
      {
        intr_name = "thread_id_x";
        intr_codegen = "Sarek_cpu_runtime.thread_id_x ()";
        intr_convergence = Framework_sig.Divergent;
      } ;
    register
      "thread_id_y"
      {
        intr_name = "thread_id_y";
        intr_codegen = "Sarek_cpu_runtime.thread_id_y ()";
        intr_convergence = Framework_sig.Divergent;
      } ;
    register
      "thread_id_z"
      {
        intr_name = "thread_id_z";
        intr_codegen = "Sarek_cpu_runtime.thread_id_z ()";
        intr_convergence = Framework_sig.Divergent;
      } ;
    register
      "block_id_x"
      {
        intr_name = "block_id_x";
        intr_codegen = "Sarek_cpu_runtime.block_id_x ()";
        intr_convergence = Framework_sig.Uniform;
      } ;
    register
      "block_id_y"
      {
        intr_name = "block_id_y";
        intr_codegen = "Sarek_cpu_runtime.block_id_y ()";
        intr_convergence = Framework_sig.Uniform;
      } ;
    register
      "block_id_z"
      {
        intr_name = "block_id_z";
        intr_codegen = "Sarek_cpu_runtime.block_id_z ()";
        intr_convergence = Framework_sig.Uniform;
      } ;
    register
      "block_dim_x"
      {
        intr_name = "block_dim_x";
        intr_codegen = "Sarek_cpu_runtime.block_dim_x ()";
        intr_convergence = Framework_sig.Uniform;
      } ;
    register
      "block_dim_y"
      {
        intr_name = "block_dim_y";
        intr_codegen = "Sarek_cpu_runtime.block_dim_y ()";
        intr_convergence = Framework_sig.Uniform;
      } ;
    register
      "block_dim_z"
      {
        intr_name = "block_dim_z";
        intr_codegen = "Sarek_cpu_runtime.block_dim_z ()";
        intr_convergence = Framework_sig.Uniform;
      } ;
    register
      "global_thread_id"
      {
        intr_name = "global_thread_id";
        intr_codegen = "Sarek_cpu_runtime.global_thread_id ()";
        intr_convergence = Framework_sig.Divergent;
      } ;

    (* Synchronization - Native uses threadpool barriers *)
    register
      "block_barrier"
      {
        intr_name = "block_barrier";
        intr_codegen = "Sarek_cpu_runtime.block_barrier ()";
        intr_convergence = Framework_sig.Sync;
      } ;
    register
      "warp_barrier"
      {
        intr_name = "warp_barrier";
        intr_codegen = "()" (* No-op on CPU *);
        intr_convergence = Framework_sig.Sync;
      } ;
    register
      "memory_fence"
      {
        intr_name = "memory_fence";
        intr_codegen = "()" (* No-op on CPU with proper memory model *);
        intr_convergence = Framework_sig.Uniform;
      }
end

(** Native Backend - implements BACKEND *)
module Backend : Framework_sig.BACKEND = struct
  (* Include all of BACKEND from Native_base *)
  include Native_base

  (** Execution model: Native uses Direct execution *)
  let execution_model = Framework_sig.Direct

  (** Generate source - not used for Direct backend *)
  let generate_source ?block:_ (_ir : Sarek_ir_types.kernel) : string option =
    None

  (** Convert exec_arg to native_arg for typed native function *)
  let exec_arg_to_native (arg : Framework_sig.exec_arg) :
      Sarek_ir_types.native_arg =
    match arg with
    | Framework_sig.EA_Int32 n -> Sarek_ir_types.NA_Int32 n
    | Framework_sig.EA_Int64 n -> Sarek_ir_types.NA_Int64 n
    | Framework_sig.EA_Float32 f -> Sarek_ir_types.NA_Float32 f
    | Framework_sig.EA_Float64 f -> Sarek_ir_types.NA_Float64 f
    | Framework_sig.EA_Scalar ((module S), v) -> (
        (* Convert scalar via primitive *)
        match S.to_primitive v with
        | Typed_value.PInt32 n -> Sarek_ir_types.NA_Int32 n
        | Typed_value.PInt64 n -> Sarek_ir_types.NA_Int64 n
        | Typed_value.PFloat f -> Sarek_ir_types.NA_Float32 f
        | Typed_value.PBool b -> Sarek_ir_types.NA_Int32 (if b then 1l else 0l)
        | Typed_value.PBytes _ ->
            Native_error.(raise_error (feature_not_supported "PBytes type")))
    | Framework_sig.EA_Composite _ ->
        Native_error.(
          raise_error
            (feature_not_supported "custom composite types in execution"))
    | Framework_sig.EA_Vec (module V) ->
        (* Create NA_Vec with typed accessors that delegate to EXEC_VECTOR *)
        let get_as_f32 i =
          match V.get i with
          | Typed_value.TV_Scalar (Typed_value.SV ((module S), x)) -> (
              match S.to_primitive x with
              | Typed_value.PFloat f -> f
              | Typed_value.PInt32 n -> Int32.to_float n
              | Typed_value.PInt64 n -> Int64.to_float n
              | _ ->
                  Native_error.(
                    raise_error
                      (unsupported_construct "get_f32" "incompatible type")))
          | _ ->
              Native_error.(
                raise_error (unsupported_construct "get_f32" "not a scalar"))
        in
        let set_as_f32 i f =
          V.set
            i
            (Typed_value.TV_Scalar
               (Typed_value.SV ((module Typed_value.Float32_type), f)))
        in
        let get_as_f64 i =
          match V.get i with
          | Typed_value.TV_Scalar (Typed_value.SV ((module S), x)) -> (
              match S.to_primitive x with
              | Typed_value.PFloat f -> f
              | Typed_value.PInt32 n -> Int32.to_float n
              | Typed_value.PInt64 n -> Int64.to_float n
              | _ ->
                  Native_error.(
                    raise_error
                      (unsupported_construct "get_f64" "incompatible type")))
          | _ ->
              Native_error.(
                raise_error (unsupported_construct "get_f64" "not a scalar"))
        in
        let set_as_f64 i f =
          V.set
            i
            (Typed_value.TV_Scalar
               (Typed_value.SV ((module Typed_value.Float64_type), f)))
        in
        let get_as_i32 i =
          match V.get i with
          | Typed_value.TV_Scalar (Typed_value.SV ((module S), x)) -> (
              match S.to_primitive x with
              | Typed_value.PInt32 n -> n
              | Typed_value.PInt64 n -> Int64.to_int32 n
              | Typed_value.PFloat f -> Int32.of_float f
              | _ ->
                  Native_error.(
                    raise_error
                      (unsupported_construct "get_i32" "incompatible type")))
          | _ ->
              Native_error.(
                raise_error (unsupported_construct "get_i32" "not a scalar"))
        in
        let set_as_i32 i n =
          V.set
            i
            (Typed_value.TV_Scalar
               (Typed_value.SV ((module Typed_value.Int32_type), n)))
        in
        let get_as_i64 i =
          match V.get i with
          | Typed_value.TV_Scalar (Typed_value.SV ((module S), x)) -> (
              match S.to_primitive x with
              | Typed_value.PInt64 n -> n
              | Typed_value.PInt32 n -> Int64.of_int32 n
              | Typed_value.PFloat f -> Int64.of_float f
              | _ ->
                  Native_error.(
                    raise_error
                      (unsupported_construct "get_i64" "incompatible type")))
          | _ ->
              Native_error.(
                raise_error (unsupported_construct "get_i64" "not a scalar"))
        in
        let set_as_i64 i n =
          V.set
            i
            (Typed_value.TV_Scalar
               (Typed_value.SV ((module Typed_value.Int64_type), n)))
        in
        Sarek_ir_types.NA_Vec
          (Sarek_ir_types.NV
             {
               length = V.length;
               elem_size = V.elem_size;
               type_name = V.type_name;
               type_id = V.type_id;
               get_f32 = get_as_f32;
               set_f32 = set_as_f32;
               get_f64 = get_as_f64;
               set_f64 = set_as_f64;
               get_i32 = get_as_i32;
               set_i32 = set_as_i32;
               get_i64 = get_as_i64;
               set_i64 = set_as_i64;
               get_typed = V.get_typed;
               set_typed = V.set_typed;
               underlying_type_id = V.underlying_type_id;
               underlying = V.underlying;
             })

  (** Convert exec_arg array to native_arg array *)
  let exec_args_to_native (args : Framework_sig.exec_arg array) :
      Sarek_ir_types.native_arg array =
    Array.map exec_arg_to_native args

  (** Execute directly using native function from IR. Args contain vectors
      directly (not expanded buffer/length pairs). The native function uses
      native_arg accessors for element access. *)
  let execute_direct
      ~(native_fn :
         (block:Framework_sig.dims ->
         grid:Framework_sig.dims ->
         Framework_sig.exec_arg array ->
         unit)
         option) ~(ir : Sarek_ir_types.kernel option)
      ~(block : Framework_sig.dims) ~(grid : Framework_sig.dims)
      (args : Framework_sig.exec_arg array) : unit =
    (* First try native_fn if provided *)
    match native_fn with
    | Some fn -> fn ~block ~grid args
    | None -> (
        (* Fall back to IR native function *)
        match ir with
        | Some kernel -> (
            match kernel.kern_native_fn with
            | Some (Sarek_ir_types.NativeFn fn) ->
                (* Convert exec_args to native_arg for typed native function *)
                let native_args = exec_args_to_native args in
                let block_tuple = (block.x, block.y, block.z) in
                let grid_tuple = (grid.x, grid.y, grid.z) in
                fn
                  ~parallel:true
                  ~block:block_tuple
                  ~grid:grid_tuple
                  native_args
            | None ->
                Native_error.(
                  raise_error
                    (compilation_failed
                       ""
                       "no native function found in kernel IR")))
        | None ->
            Native_error.(
              raise_error (compilation_failed "" "kernel IR not available")))

  (** Native intrinsic registry *)
  module Intrinsics = Native_intrinsics

  (** {2 External Kernel Support} *)

  (** Native backend does not support external GPU sources *)
  let supported_source_langs = []

  (** External kernel execution not supported on Native backend *)
  let run_source ~source:_ ~lang ~kernel_name:_ ~block:_ ~grid:_ ~shared_mem:_
      (_args : Framework_sig.run_source_arg list) =
    let lang_str =
      match lang with
      | Framework_sig.CUDA_Source -> "CUDA"
      | Framework_sig.OpenCL_Source -> "OpenCL"
      | Framework_sig.PTX -> "PTX"
      | Framework_sig.SPIR_V -> "SPIR-V"
      | Framework_sig.GLSL_Source -> "GLSL"
    in
    Native_error.(raise_error (unsupported_source_lang lang_str))

  let wrap_kargs args = Native_kargs args

  let unwrap_kargs = function Native_kargs args -> Some args | _ -> None
end

(** Auto-register backend when module is loaded *)
let registered_backend =
  lazy
    (if Backend.is_available () then
       Framework_registry.register_backend
         ~priority:10
         (module Backend : Framework_sig.BACKEND))

let () = Lazy.force registered_backend

(** Force module initialization *)
let init () = Lazy.force registered_backend

(** {1 Additional Native-specific Functions} *)

(** Register a custom Native intrinsic *)
let register_intrinsic = Native_intrinsics.register

(** Look up a Native intrinsic *)
let find_intrinsic = Native_intrinsics.find

(** Execute a kernel directly with the registered function *)
let run_kernel_direct ~name
    ~(native_fn :
       Framework_sig.exec_arg array ->
       int * int * int ->
       int * int * int ->
       unit) ~(args : Framework_sig.exec_arg array) ~(grid : int * int * int)
    ~(block : int * int * int) : unit =
  (* Use the existing Native_plugin kernel registry *)
  ignore name ;
  native_fn args grid block

(** Register a kernel for direct execution (wraps
    Native_plugin_base.register_kernel) *)
let register_kernel = Native_plugin_base.register_kernel

(** Check if a kernel is registered *)
let kernel_registered = Native_plugin_base.kernel_registered

(** List all registered kernels *)
let list_kernels = Native_plugin_base.list_kernels
