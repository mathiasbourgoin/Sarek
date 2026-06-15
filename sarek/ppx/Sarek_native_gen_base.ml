(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

open Ppxlib
open Sarek_typed_ast
open Sarek_types

(* Import helpers and intrinsics *)
open Sarek_native_helpers
open Sarek_native_intrinsics

(** {1 Expression Generation} *)

(** Mutable variable tracking. We track which variable IDs are mutable so that
    TEVar can dereference them. *)
module IntSet = Set.Make (Int)

(** Set of inline type names for first-class module approach *)
module StringSet = Set.Make (String)

(** Expression generation context *)
type gen_context = {
  mut_vars : IntSet.t;
      (** Variable IDs that are mutable (need dereferencing) *)
  inline_types : StringSet.t;
      (** Type names that use first-class module accessors *)
  current_module : string option;
      (** Current module name for same-module type detection *)
  gen_mode : gen_mode;
      (** Generation mode - affects how thread indices are accessed *)
  use_native_arg : bool;
      (** When true, vector params are accessor objects (v#get i, v#set i x)
          instead of Vector.t (Vector.get v i, Vector.kernel_set v i x) *)
}

(** Empty generation context *)
let empty_ctx =
  {
    mut_vars = IntSet.empty;
    inline_types = StringSet.empty;
    current_module = None;
    gen_mode = FullMode;
    use_native_arg = false;
  }

(** Check if a qualified type name is from the current module. For
    "Test_registered_variant.color", returns true if current_module is
    "Test_registered_variant". *)
let is_same_module ctx type_name =
  match ctx.current_module with
  | None -> false
  | Some cur_mod -> (
      match String.rindex_opt type_name '.' with
      | Some idx ->
          let type_mod = String.sub type_name 0 idx in
          String.equal type_mod cur_mod
      | None -> false)

(** The name of the first-class module variable *)
let types_module_var = "__types"

(** {1 First-Class Module Name Helpers} *)

(** Generate accessor function name for a field getter *)
let field_getter_name type_name field_name : string =
  Printf.sprintf "get_%s_%s" type_name field_name

(** Generate constructor function name for a record *)
let record_maker_name type_name : string = Printf.sprintf "make_%s" type_name

(** Generate constructor function name for a variant constructor *)
let variant_ctor_name type_name ctor_name : string =
  Printf.sprintf "make_%s_%s" type_name ctor_name

let gen_literal ~loc (te : texpr) : expression =
  match te.te with
  | TEUnit -> [%expr ()]
  | TEBool b -> if b then [%expr true] else [%expr false]
  | TEInt n -> (
      (* Check the type annotation to generate the correct literal type.
         In GPU kernels, integer literals compared with int32 should be int32. *)
      match repr te.ty with
      | TReg Int | TPrim TInt32 ->
          [%expr Int32.of_int [%e Ast_builder.Default.eint ~loc n]]
      | TReg Int64 -> [%expr Int64.of_int [%e Ast_builder.Default.eint ~loc n]]
      | _ ->
          (* Default to plain int *)
          Ast_builder.Default.eint ~loc n)
  | TEInt32 n ->
      (* Generate int32 literal using Int32.of_int *)
      [%expr Int32.of_int [%e Ast_builder.Default.eint ~loc (Int32.to_int n)]]
  | TEInt64 n ->
      (* Generate int64 literal using Int64.of_int *)
      [%expr Int64.of_int [%e Ast_builder.Default.eint ~loc (Int64.to_int n)]]
  | TEFloat f | TEDouble f ->
      Ast_builder.Default.efloat ~loc (string_of_float f)
  | _ -> failwith "gen_literal: not a literal expression"

(** Generate variable reference (local, module-level, qualified) *)
let gen_variable ~loc ~ctx (name : string) (id : int) : expression =
  let var_e =
    if String.contains name '.' then
      (* Qualified name - build a proper Ldot path.
         For stdlib modules like "Float32.of_float", we need to map
         to the runtime path. *)
      let parts = String.split_on_char '.' name in
      (* Split into module path and function name *)
      let module_path, func_name =
        match List.rev parts with
        | fn :: rest -> (List.rev rest, fn)
        | [] ->
            failwith
              (Printf.sprintf
                 "Internal error: String.split_on_char returned empty list for \
                  '%s'"
                 name)
      in
      (* Map stdlib module paths to runtime locations *)
      let mapped_path = map_stdlib_path module_path in
      evar_qualified ~loc mapped_path func_name
    else evar ~loc name
  in
  if IntSet.mem id ctx.mut_vars then
    (* Mutable variable - dereference the ref *)
    [%expr ![%e var_e]]
  else var_e

let custom_descriptor_expr ?current_module ~loc type_name =
  let parts = String.split_on_char '.' type_name in
  match List.rev parts with
  | name :: modules ->
      let modules = List.rev modules in
      let modules =
        match (current_module, modules) with
        | Some current, [m] when String.equal current m -> []
        | _ -> modules
      in
      evar_qualified ~loc modules (name ^ "_custom")
  | [] -> failwith "custom_descriptor_expr: empty type name"

let is_inline_type inline_types type_name =
  match inline_types with
  | Some names -> StringSet.mem type_name names
  | None -> false

let sanitize_ident s =
  String.map
    (function
      | ('a' .. 'z' | 'A' .. 'Z' | '0' .. '9' | '_') as c -> c | _ -> '_')
    s

let inline_type_id_method type_name = "type_id_" ^ sanitize_ident type_name

let inline_vector_type_id_method type_name =
  "vector_type_id_" ^ sanitize_ident type_name

let inline_type_id_expr ~loc ~ids_var type_name =
  Ast_builder.Default.pexp_send
    ~loc
    (evar ~loc ids_var)
    {txt = inline_type_id_method type_name; loc}

let inline_vector_type_id_expr ~loc ~ids_var type_name =
  Ast_builder.Default.pexp_send
    ~loc
    (evar ~loc ids_var)
    {txt = inline_vector_type_id_method type_name; loc}

let vector_type_id_expr ?current_module ?inline_types
    ?(inline_ids_var = types_module_var) ~loc elem_ty =
  match repr elem_ty with
  | TReg Float32 -> [%expr Spoc_core.Vector.float32_vector_type_id]
  | TReg Float64 -> [%expr Spoc_core.Vector.float64_vector_type_id]
  | TReg Int | TPrim TInt32 -> [%expr Spoc_core.Vector.int32_vector_type_id]
  | TReg Int64 -> [%expr Spoc_core.Vector.int64_vector_type_id]
  | TReg Char -> [%expr Spoc_core.Vector.char_vector_type_id]
  | TReg (Custom "complex32") ->
      [%expr Spoc_core.Vector.complex32_vector_type_id]
  | TRecord (type_name, _) | TVariant (type_name, _) ->
      if is_inline_type inline_types type_name then
        inline_vector_type_id_expr ~loc ~ids_var:inline_ids_var type_name
      else
        [%expr
          [%e custom_descriptor_expr ?current_module ~loc type_name]
            .Spoc_core.Vector.vector_type_id]
  | _ -> [%expr failwith "unsupported vector type identity"]

let custom_type_id_expr ?current_module ?inline_types
    ?(inline_ids_var = types_module_var) ~loc elem_ty =
  match repr elem_ty with
  | TReg (Custom "complex32") -> [%expr Spoc_core.Vector.complex32_type_id]
  | TRecord (type_name, _) | TVariant (type_name, _) ->
      if is_inline_type inline_types type_name then
        inline_type_id_expr ~loc ~ids_var:inline_ids_var type_name
      else
        [%expr
          [%e custom_descriptor_expr ?current_module ~loc type_name]
            .Spoc_core.Vector.type_id]
  | _ -> [%expr failwith "unsupported custom type identity"]
