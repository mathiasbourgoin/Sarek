(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Kirc kernel type (modern path)
 *
 * Defines a new kernel record type with lazy IR generation for the unified
 * execution architecture. Coexists with the existing Kirc module during
 * the transition period.
 *
 * Key differences from Kirc:
 * - Lazy IR: Sarek_ir.kernel is generated only when needed (JIT backends)
 * - Native function: Pre-compiled OCaml function for Direct backends
 * - Typed AST: Access to the typed AST for Custom backends
 * - Clean separation: No legacy Kirc_Ast.k_ext in the primary interface
 ******************************************************************************)

open Spoc_framework
open Spoc_framework_registry
open Spoc_core

(** {1 Kernel Type} *)

(** Phase 4 kernel record with lazy IR generation *)
type 'a kernel = {
  name : string;  (** Kernel name (used for compilation cache key) *)
  ir : Sarek_ir_types.kernel Lazy.t;
      (** Lazy IR generation - only forced for JIT backends *)
  native_fn :
    (block:Framework_sig.dims ->
    grid:Framework_sig.dims ->
    Framework_sig.exec_arg array ->
    unit)
    option;
      (** Pre-compiled OCaml function for Direct backends *)
  param_types : Sarek_ir_types.elttype list;
      (** Parameter types for argument marshalling *)
  extensions : Kirc_types.extension array;
      (** Extensions required (ExFloat32, ExFloat64) *)
}

(** {1 Constructors} *)

(** Create a kernel with both IR and native function *)
let make ~name ~ir ~native_fn ~param_types ?(extensions = [||]) () =
  {name; ir; native_fn; param_types; extensions}

(** Create a JIT-only kernel (no native function) *)
let make_jit ~name ~ir ~param_types ?(extensions = [||]) () =
  {name; ir; native_fn = None; param_types; extensions}

(** Create a Native-only kernel (no IR) *)
let make_native ~name ~native_fn ~param_types () =
  {
    name;
    ir = lazy (Kirc_error.raise_error (Kirc_error.No_ir {kernel_name = name}));
    native_fn = Some native_fn;
    param_types;
    extensions = [||];
  }

(** {1 Accessors} *)

(** Get the kernel name *)
let name k = k.name

(** Get the IR (forces evaluation if lazy) *)
let ir k = Lazy.force k.ir

(** Check if native function is available *)
let has_native k = Option.is_some k.native_fn

(** Get the native function (raises if not available) *)
let native_fn k =
  match k.native_fn with
  | Some fn -> fn
  | None ->
      Kirc_error.raise_error
        (Kirc_error.No_native_function
           {kernel_name = k.name; context = "native_fn accessor"})

(** Get parameter types *)
let param_types k = k.param_types

(** Get extensions *)
let extensions k = k.extensions

(** {1 Conversion from Legacy Kirc} *)

(** Convert exec_arg to native_arg. This bridges Framework_sig.exec_arg (typed
    execution args) to Sarek_ir_types.native_arg (native function args). *)
let exec_arg_to_native_arg (arg : Framework_sig.exec_arg) :
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
          Kirc_error.raise_error
            (Kirc_error.Unsupported_arg_type
               {
                 arg_type = "PBytes";
                 reason = "not supported in native_arg";
                 context = "exec_arg_to_native_arg";
               }))
  | Framework_sig.EA_Composite _ ->
      Kirc_error.raise_error
        (Kirc_error.Unsupported_arg_type
           {
             arg_type = "EA_Composite";
             reason = "composite types not yet supported";
             context = "exec_arg_to_native_arg";
           })
  | Framework_sig.EA_Vec (module V) ->
      (* Create NA_Vec with typed accessors *)
      let get_as_f32 i =
        match V.get i with
        | Typed_value.TV_Scalar (Typed_value.SV ((module S), x)) -> (
            match S.to_primitive x with
            | Typed_value.PFloat f -> f
            | Typed_value.PInt32 n -> Int32.to_float n
            | Typed_value.PInt64 n -> Int64.to_float n
            | prim ->
                Kirc_error.raise_error
                  (Kirc_error.Type_conversion_failed
                     {
                       from_type = Typed_value.primitive_type_name prim;
                       to_type = "float32";
                       index = Some i;
                       context = "get_f32";
                     }))
        | tv ->
            Kirc_error.raise_error
              (Kirc_error.Type_conversion_failed
                 {
                   from_type = Typed_value.typed_value_type_name tv;
                   to_type = "float32";
                   index = Some i;
                   context = "get_f32 (not a scalar)";
                 })
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
            | prim ->
                Kirc_error.raise_error
                  (Kirc_error.Type_conversion_failed
                     {
                       from_type = Typed_value.primitive_type_name prim;
                       to_type = "float64";
                       index = Some i;
                       context = "get_f64";
                     }))
        | tv ->
            Kirc_error.raise_error
              (Kirc_error.Type_conversion_failed
                 {
                   from_type = Typed_value.typed_value_type_name tv;
                   to_type = "float64";
                   index = Some i;
                   context = "get_f64 (not a scalar)";
                 })
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
            | prim ->
                Kirc_error.raise_error
                  (Kirc_error.Type_conversion_failed
                     {
                       from_type = Typed_value.primitive_type_name prim;
                       to_type = "int32";
                       index = Some i;
                       context = "get_i32";
                     }))
        | tv ->
            Kirc_error.raise_error
              (Kirc_error.Type_conversion_failed
                 {
                   from_type = Typed_value.typed_value_type_name tv;
                   to_type = "int32";
                   index = Some i;
                   context = "get_i32 (not a scalar)";
                 })
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
            | prim ->
                Kirc_error.raise_error
                  (Kirc_error.Type_conversion_failed
                     {
                       from_type = Typed_value.primitive_type_name prim;
                       to_type = "int64";
                       index = Some i;
                       context = "get_i64";
                     }))
        | tv ->
            Kirc_error.raise_error
              (Kirc_error.Type_conversion_failed
                 {
                   from_type = Typed_value.typed_value_type_name tv;
                   to_type = "int64";
                   index = Some i;
                   context = "get_i64 (not a scalar)";
                 })
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

(** {1 Execution} *)

(** Execute a kernel on a device with exec_arg array args. Note: Only works for
    Native backend. For JIT backends (CUDA/OpenCL), use run_with_args which
    provides properly typed arguments. *)
let run ~(device : Device.t) ~(block : Framework_sig.dims)
    ~(grid : Framework_sig.dims) ?(shared_mem = 0) (k : 'a kernel)
    (args : Framework_sig.exec_arg array) : unit =
  ignore shared_mem ;
  match device.framework with
  | "Native" -> (
      match k.native_fn with
      | Some fn -> fn ~block ~grid args
      | None ->
          Kirc_error.raise_error
            (Kirc_error.No_native_function
               {kernel_name = k.name; context = "run"}))
  | fw ->
      Kirc_error.raise_error
        (Kirc_error.Wrong_backend
           {
             expected = "Native";
             got = fw;
             operation = "run with exec_arg array";
           })

(** Execute a kernel with explicit typed arguments. Works for all backends
    (Native, CUDA, OpenCL). Uses plugin dispatch via Execute.run. *)
let run_with_args ~(device : Device.t) ~(block : Framework_sig.dims)
    ~(grid : Framework_sig.dims) ?(shared_mem = 0) (k : 'a kernel)
    (args : Execute.vector_arg list) : unit =
  Execute.run
    ~device
    ~name:k.name
    ~ir:(Some k.ir)
    ~native_fn:k.native_fn
    ~block
    ~grid
    ~shared_mem
    args

(** {1 Utilities} *)

(** Get generated source for a specific backend. Uses plugin's generate_source.
*)
let source_for_backend (k : 'a kernel) ~backend : string =
  match Framework_registry.find_backend backend with
  | Some (module B : Framework_sig.BACKEND) -> (
      let ir = Lazy.force k.ir in
      match B.generate_source ir with
      | Some source -> source
      | None ->
          Kirc_error.raise_error (Kirc_error.No_source_generation {backend}))
  | None -> Kirc_error.raise_error (Kirc_error.Backend_not_found {backend})

(** Check if kernel requires FP64 extension *)
let requires_fp64 k = Array.mem Kirc_types.ExFloat64 k.extensions

(** Check if kernel requires FP32 extension *)
let requires_fp32 k = Array.mem Kirc_types.ExFloat32 k.extensions

(** {1 Debugging} *)

(** Pretty-print the IR *)
let pp_ir fmt k =
  let ir = Lazy.force k.ir in
  Sarek_ir_pp.pp_kernel fmt ir

(** Get IR as string *)
let ir_to_string k =
  let buf = Buffer.create 1024 in
  let fmt = Format.formatter_of_buffer buf in
  pp_ir fmt k ;
  Format.pp_print_flush fmt () ;
  Buffer.contents buf
