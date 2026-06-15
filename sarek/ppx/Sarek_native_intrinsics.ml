(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Sarek PPX - GPU kernel DSL for OCaml
 *
 * Native Code Generation - Type and Intrinsic Mapping
 * ===================================================
 *
 * Maps Sarek types to OCaml core_type for type annotations.
 * Maps GPU intrinsics to their CPU runtime equivalents.
 *
 * See also:
 * - Sarek_native_helpers for utility functions
 * - Sarek_native_gen for expression and kernel generation
 ******************************************************************************)

open Ppxlib
open Sarek_types
open Sarek_env

(** {1 Type Mapping} *)

(** Generate a core type from a Sarek type (for type annotations).

    For scalar types (int32, float, etc), we generate explicit types to help
    OCaml infer the correct numeric type.

    For record types, we generate the type name to help OCaml resolve field
    accesses. The type name may be qualified (e.g., "Module.point").

    For vectors and other complex types, we use wildcards. *)
let rec core_type_of_typ ~loc typ : Ppxlib.core_type =
  match repr typ with
  (* Primitive types - only TUnit, TBool, TInt32 exist as primitives *)
  | TPrim TUnit -> [%type: unit]
  | TPrim TBool -> [%type: bool]
  | TPrim TInt32 -> [%type: int32]
  (* Registered types - numeric types are registered by stdlib *)
  | TReg Float32 -> [%type: float]
  | TReg Float64 -> [%type: float]
  | TReg Int -> [%type: int32]
  | TReg Int64 -> [%type: int64]
  (* Vector/array types - use wildcard to avoid scope issues *)
  | TVec _ -> [%type: _]
  | TArr _ -> [%type: _]
  (* Record types - for qualified types (Module.type), generate the type name
     to help resolve field accesses. For inline types (no module prefix),
     use wildcard to avoid scope issues. *)
  | TRecord (name, _fields) -> (
      match String.split_on_char '.' name with
      | [_simple_name] ->
          (* Inline type - use wildcard to avoid "type escapes scope" errors *)
          [%type: _]
      | parts ->
          (* Qualified type like "Module.type" - generate the type path *)
          let rec build_lid = function
            | [] -> failwith "empty type name"
            | [x] -> Lident x
            | x :: rest -> Ldot (build_lid rest, x)
          in
          let lid = build_lid (List.rev parts) in
          Ast_builder.Default.ptyp_constr ~loc {txt = lid; loc} [])
  (* Variant types - same as records *)
  | TVariant (name, _constrs) -> (
      match String.split_on_char '.' name with
      | [_simple_name] ->
          (* Inline type - use wildcard *)
          [%type: _]
      | parts ->
          let rec build_lid = function
            | [] -> failwith "empty type name"
            | [x] -> Lident x
            | x :: rest -> Ldot (build_lid rest, x)
          in
          let lid = build_lid (List.rev parts) in
          Ast_builder.Default.ptyp_constr ~loc {txt = lid; loc} [])
  | TTuple tys ->
      Ast_builder.Default.ptyp_tuple ~loc (List.map (core_type_of_typ ~loc) tys)
  | TFun _ | TVar _ | TReg _ -> [%type: _]

(** {1 Intrinsic Mapping}

    Map Sarek intrinsics to their OCaml equivalents. For cpu_kern, we call the
    OCaml implementations directly rather than generating GPU code. *)

(** Kernel generation mode for simple vs full execution - defined early for use
    below *)
type gen_mode =
  | FullMode  (** Standard mode - uses thread_state for all indices *)
  | Simple1DMode  (** Simple 1D - gid_x passed directly as int32 *)
  | Simple2DMode  (** Simple 2D - gid_x, gid_y passed directly *)
  | Simple3DMode  (** Simple 3D - gid_x, gid_y, gid_z passed directly *)

(** Variable names for simple mode global indices *)
let simple_gid_x = "__gid_x"

let simple_gid_y = "__gid_y"

let simple_gid_z = "__gid_z"

(** Map stdlib module paths to their runtime module paths. Sarek stdlib modules
    are available at both "Foo" and "Sarek_stdlib.Foo" paths at compile time. At
    runtime, the corresponding CPU implementations are in Sarek_cpu_runtime. *)
let map_stdlib_path = function
  | ["Float32"] | ["Sarek_stdlib"; "Float32"] ->
      ["Sarek"; "Sarek_cpu_runtime"; "Float32"]
  | ["Float64"] | ["Sarek_stdlib"; "Float64"] ->
      (* Float64 is just OCaml float, use stdlib *)
      ["Float"]
  | ["Int32"] | ["Sarek_stdlib"; "Int32"] -> ["Int32"]
  | ["Int64"] | ["Sarek_stdlib"; "Int64"] -> ["Int64"]
  (* Note: Gpu module functions stay as Gpu.* - only specific functions like
     block_barrier, global_idx, global_size are handled specially in gen_intrinsic_fun *)
  | path -> path

(** Generate intrinsic constant based on generation mode. For simple modes,
    direct global indices are used. For full mode, we access thread_state. *)
let gen_intrinsic_const ~loc ~gen_mode (ref : intrinsic_ref) =
  let const =
    match ref with
    | CorePrimitiveRef name -> name
    | IntrinsicRef (_, name) -> name
  in
  let open Sarek_native_helpers in
  match (gen_mode, const) with
  (* Full mode - always use thread_state *)
  | FullMode, "thread_idx_x" ->
      let state = evar ~loc state_var in
      [%expr [%e state].Sarek.Sarek_cpu_runtime.thread_idx_x]
  | FullMode, "thread_idx_y" ->
      let state = evar ~loc state_var in
      [%expr [%e state].Sarek.Sarek_cpu_runtime.thread_idx_y]
  | FullMode, "thread_idx_z" ->
      let state = evar ~loc state_var in
      [%expr [%e state].Sarek.Sarek_cpu_runtime.thread_idx_z]
  | FullMode, "block_dim_x" ->
      let state = evar ~loc state_var in
      [%expr [%e state].Sarek.Sarek_cpu_runtime.block_dim_x]
  | FullMode, "block_dim_y" ->
      let state = evar ~loc state_var in
      [%expr [%e state].Sarek.Sarek_cpu_runtime.block_dim_y]
  | FullMode, "block_dim_z" ->
      let state = evar ~loc state_var in
      [%expr [%e state].Sarek.Sarek_cpu_runtime.block_dim_z]
  | FullMode, "grid_dim_x" ->
      let state = evar ~loc state_var in
      [%expr [%e state].Sarek.Sarek_cpu_runtime.grid_dim_x]
  | FullMode, "grid_dim_y" ->
      let state = evar ~loc state_var in
      [%expr [%e state].Sarek.Sarek_cpu_runtime.grid_dim_y]
  | FullMode, "grid_dim_z" ->
      let state = evar ~loc state_var in
      [%expr [%e state].Sarek.Sarek_cpu_runtime.grid_dim_z]
  | FullMode, "global_idx_x" ->
      let state = evar ~loc state_var in
      [%expr Sarek.Sarek_cpu_runtime.global_idx_x [%e state]]
  | FullMode, "global_idx_y" ->
      let state = evar ~loc state_var in
      [%expr Sarek.Sarek_cpu_runtime.global_idx_y [%e state]]
  | FullMode, "global_idx_z" ->
      let state = evar ~loc state_var in
      [%expr Sarek.Sarek_cpu_runtime.global_idx_z [%e state]]
  | FullMode, "global_size_x" ->
      let state = evar ~loc state_var in
      [%expr Sarek.Sarek_cpu_runtime.global_size_x [%e state]]
  | FullMode, "global_size_y" ->
      let state = evar ~loc state_var in
      [%expr Sarek.Sarek_cpu_runtime.global_size_y [%e state]]
  | FullMode, "global_size_z" ->
      let state = evar ~loc state_var in
      [%expr Sarek.Sarek_cpu_runtime.global_size_z [%e state]]
  (* Aliases *)
  | FullMode, "global_thread_id" ->
      let state = evar ~loc state_var in
      [%expr Sarek.Sarek_cpu_runtime.global_idx_x [%e state]]
  | FullMode, "block_idx_x" ->
      let state = evar ~loc state_var in
      [%expr [%e state].Sarek.Sarek_cpu_runtime.block_idx_x]
  | FullMode, "block_idx_y" ->
      let state = evar ~loc state_var in
      [%expr [%e state].Sarek.Sarek_cpu_runtime.block_idx_y]
  | FullMode, "block_idx_z" ->
      let state = evar ~loc state_var in
      [%expr [%e state].Sarek.Sarek_cpu_runtime.block_idx_z]
  (* Simple 1D mode - use direct gid_x for global_idx_x *)
  | Simple1DMode, "global_idx_x" -> evar ~loc simple_gid_x
  | Simple1DMode, "global_thread_id" -> evar ~loc simple_gid_x
  | Simple1DMode, _ ->
      failwith
        (Printf.sprintf
           "Intrinsic %s not available in Simple1DMode (only global_idx_x)"
           const)
  (* Simple 2D mode - use direct gid_x, gid_y *)
  | Simple2DMode, "global_idx_x" -> evar ~loc simple_gid_x
  | Simple2DMode, "global_idx_y" -> evar ~loc simple_gid_y
  | Simple2DMode, "global_thread_id" -> evar ~loc simple_gid_x
  | Simple2DMode, _ ->
      failwith
        (Printf.sprintf
           "Intrinsic %s not available in Simple2DMode (only global_idx_x, \
            global_idx_y)"
           const)
  (* Simple 3D mode - use direct gid_x, gid_y, gid_z *)
  | Simple3DMode, "global_idx_x" -> evar ~loc simple_gid_x
  | Simple3DMode, "global_idx_y" -> evar ~loc simple_gid_y
  | Simple3DMode, "global_idx_z" -> evar ~loc simple_gid_z
  | Simple3DMode, "global_thread_id" -> evar ~loc simple_gid_x
  | Simple3DMode, _ ->
      failwith
        (Printf.sprintf
           "Intrinsic %s not available in Simple3DMode (only global_idx_x, \
            global_idx_y, global_idx_z)"
           const)
  | FullMode, _ ->
      failwith (Printf.sprintf "Unknown intrinsic constant: %s" const)

(** Generate intrinsic function call based on generation mode. For simple modes,
    we use thread_state for most operations. stdlib functions are called
    directly from their runtime implementations. *)
let gen_intrinsic_fun ~loc ~gen_mode (ref : intrinsic_ref) args =
  let open Sarek_native_helpers in
  (* Helper for simple mode global index functions *)
  let use_simple_gid_fn name =
    match (gen_mode, name) with
    | ( (Simple1DMode | Simple2DMode | Simple3DMode),
        ("global_idx" | "global_idx_x" | "global_thread_id") ) ->
        Some (evar ~loc simple_gid_x)
    | (Simple2DMode | Simple3DMode), "global_idx_y" ->
        Some (evar ~loc simple_gid_y)
    | Simple3DMode, "global_idx_z" -> Some (evar ~loc simple_gid_z)
    | _ -> None
  in
  let state = evar ~loc state_var in
  match ref with
  | CorePrimitiveRef name -> (
      match use_simple_gid_fn name with
      | Some e -> e
      | None -> (
          match name with
          | "block_barrier" | "warp_barrier" ->
              (* Call the barrier function from thread state *)
              [%expr [%e state].Sarek.Sarek_cpu_runtime.barrier ()]
          | "global_idx" | "global_thread_id" ->
              [%expr Sarek.Sarek_cpu_runtime.global_idx_x [%e state]]
          | "global_size" ->
              [%expr Sarek.Sarek_cpu_runtime.global_size_x [%e state]]
          | _ ->
              (* Try Gpu module *)
              let fn = evar_qualified ~loc ["Gpu"] name in
              Ast_builder.Default.pexp_apply
                ~loc
                fn
                (List.map (fun a -> (Nolabel, a)) args)))
  | IntrinsicRef (path, name) -> (
      match use_simple_gid_fn name with
      | Some e -> e
      | None -> (
          (* Check if this is a Gpu module function that maps to thread state *)
          match (path, name) with
          | (["Gpu"] | ["Sarek_stdlib"; "Gpu"]), "block_barrier" ->
              [%expr [%e state].Sarek.Sarek_cpu_runtime.barrier ()]
          | (["Gpu"] | ["Sarek_stdlib"; "Gpu"]), "global_idx" ->
              [%expr Sarek.Sarek_cpu_runtime.global_idx_x [%e state]]
          | (["Gpu"] | ["Sarek_stdlib"; "Gpu"]), "global_size" ->
              [%expr Sarek.Sarek_cpu_runtime.global_size_x [%e state]]
          | _ -> (
              (* Call the OCaml implementation from the stdlib module *)
              let fn = evar_qualified ~loc (map_stdlib_path path) name in
              match args with
              | [] -> fn
              | _ ->
                  Ast_builder.Default.pexp_apply
                    ~loc
                    fn
                    (List.map (fun a -> (Nolabel, a)) args))))
