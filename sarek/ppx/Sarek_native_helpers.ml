(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Sarek PPX - GPU kernel DSL for OCaml
 *
 * Native Code Generation Helpers
 * ===============================
 *
 * Utility functions for generating native OCaml code from Sarek's typed AST.
 * Provides:
 * - Location conversions
 * - Identifier and variable name generation
 * - Default value generation for types
 *
 * See also:
 * - Sarek_native_intrinsics for type/intrinsic mapping
 * - Sarek_native_gen for expression and kernel generation
 ******************************************************************************)

open Ppxlib
open Sarek_types

(** {1 Location Conversion} *)

(** Convert Sarek_ast.loc to Ppxlib.location *)
let ppxlib_loc_of_sarek (l : Sarek_ast.loc) : location =
  let pos_start =
    {
      Lexing.pos_fname = l.loc_file;
      pos_lnum = l.loc_line;
      pos_bol = 0;
      pos_cnum = l.loc_col;
    }
  in
  let pos_end =
    {
      Lexing.pos_fname = l.loc_file;
      pos_lnum = l.loc_end_line;
      pos_bol = 0;
      pos_cnum = l.loc_end_col;
    }
  in
  {loc_start = pos_start; loc_end = pos_end; loc_ghost = false}

(** {1 Expression Helpers} *)

(** Helper to create an identifier expression *)
let evar ~loc name =
  Ast_builder.Default.pexp_ident ~loc {txt = Lident name; loc}

(** Helper to create a qualified identifier expression *)
let evar_qualified ~loc path name =
  match path with
  | [] -> evar ~loc name
  | first :: rest ->
      let lid =
        List.fold_left
          (fun acc m -> Ldot (acc, m))
          (Lident first)
          (rest @ [name])
      in
      Ast_builder.Default.pexp_ident ~loc {txt = lid; loc}

(** {1 Variable Naming} *)

(** Create a unique name for a variable by id *)
let var_name id = Printf.sprintf "__v%d" id

(** Create a unique name for a mutable variable by id *)
let mut_var_name id = Printf.sprintf "__m%d" id

(** Thread state variable name - bound in kernel wrapper *)
let state_var = "__state"

(** Shared memory variable name - bound in block wrapper *)
let shared_var = "__shared"

(** {1 Default Value Generation} *)

(** Generate default value expression for a given type. Used for array
    initialization and other contexts where a default is needed. *)
let rec default_value_for_type ~loc (ty : typ) : expression =
  match repr ty with
  | TPrim TUnit -> [%expr ()]
  | TPrim TBool -> [%expr false]
  | TPrim TInt32 -> [%expr 0l]
  | TReg Int -> [%expr 0l]
  | TReg Int64 -> [%expr 0L]
  | TReg Float32 -> [%expr 0.0]
  | TReg Float64 -> [%expr 0.0]
  | TReg Char -> [%expr '\000']
  | TRecord (_name, fields) ->
      (* Generate record with all fields set to their default values *)
      let field_defaults =
        List.map
          (fun (fname, fty) ->
            let default_val = default_value_for_type ~loc fty in
            ({txt = Lident fname; loc}, default_val))
          fields
      in
      Ast_builder.Default.pexp_record ~loc field_defaults None
  | TVariant (_name, constrs) -> (
      (* Use the first nullary constructor, or first constructor with default args *)
      match List.find_opt (fun (_, arg) -> Option.is_none arg) constrs with
      | Some (cname, None) ->
          Ast_builder.Default.pexp_construct ~loc {txt = Lident cname; loc} None
      | _ -> (
          (* If no nullary constructor, use first constructor with default value for its argument *)
          match constrs with
          | (cname, Some arg_ty) :: _ ->
              let arg_default = default_value_for_type ~loc arg_ty in
              Ast_builder.Default.pexp_construct
                ~loc
                {txt = Lident cname; loc}
                (Some arg_default)
          | _ ->
              (* No constructors? This shouldn't happen, but provide a failsafe *)
              [%expr failwith "Cannot create default value for empty variant"]))
  | TReg (Custom _name) ->
      (* For unknown custom types registered via [@sarek.type], we can't generate
         a default without knowing the type structure. This should be rare since
         most custom types are TRecord or TVariant. *)
      [%expr failwith "Cannot create default value for custom type"]
  | _ -> [%expr failwith "Cannot create default value for this type"]
