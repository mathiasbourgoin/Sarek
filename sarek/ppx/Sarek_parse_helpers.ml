(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

open Ppxlib

(** Convert ppxlib location to Sarek location *)
let loc_of_ppxlib = Sarek_ast.loc_of_ppxlib

(** Parse exception *)
exception Parse_error_exn of string * Location.t

let loc_to_sloc loc = Sarek_ast.loc_of_ppxlib loc

(** Parse a core_type to type_expr *)
let rec parse_type (ct : core_type) : Sarek_ast.type_expr =
  match ct.ptyp_desc with
  | Ptyp_constr ({txt; _}, args) ->
      let rec flatten = function
        | Longident.Lident s -> [s]
        | Longident.Ldot (li, s) -> flatten li @ [s]
        | Longident.Lapply _ -> []
      in
      let name = String.concat "." (flatten txt) in
      Sarek_ast.TEConstr (name, List.map parse_type args)
  | Ptyp_poly (_, ct) -> parse_type ct
  | Ptyp_var name -> Sarek_ast.TEVar name
  | Ptyp_arrow (_, t1, t2) -> Sarek_ast.TEArrow (parse_type t1, parse_type t2)
  | Ptyp_tuple ts -> Sarek_ast.TETuple (List.map parse_type ts)
  | _ -> Sarek_ast.TEConstr ("unknown", [])

let parse_record_fields labels =
  List.map
    (fun (ld : label_declaration) ->
      let name = ld.pld_name.txt in
      let ty = parse_type ld.pld_type in
      let is_mut =
        match ld.pld_mutable with Mutable -> true | Immutable -> false
      in
      (name, is_mut, ty))
    labels

let parse_variant_constructors constrs =
  let parse_arg = function
    | Pcstr_tuple [] -> None
    | Pcstr_tuple [arg] -> Some (parse_type arg)
    | Pcstr_tuple _ ->
        raise
          (Parse_error_exn
             ( "Constructors with multiple arguments are not supported",
               Location.none ))
    | Pcstr_record _ ->
        raise
          (Parse_error_exn
             ("Record constructors are not supported in kernels", Location.none))
  in
  List.map
    (fun (cd : constructor_declaration) ->
      let name = cd.pcd_name.txt in
      let arg = parse_arg cd.pcd_args in
      (name, arg))
    constrs

(** Extract type annotation from a Ppxlib pattern if present *)
let rec extract_type_from_pattern (pat : Ppxlib.pattern) :
    Sarek_ast.type_expr option =
  match pat.ppat_desc with
  | Ppat_constraint (_, ct) -> Some (parse_type ct)
  | Ppat_alias (p, _) -> extract_type_from_pattern p
  | _ -> None

(** Extract variable name from a Ppxlib pattern *)
let rec extract_name_from_pattern (pat : Ppxlib.pattern) : string option =
  match pat.ppat_desc with
  | Ppat_var {txt; _} -> Some txt
  | Ppat_constraint (p, _) -> extract_name_from_pattern p
  | Ppat_alias (_, {txt; _}) -> Some txt
  | Ppat_any -> Some "_"
  | _ -> None

(** Extract parameter from pparam_desc *)
let extract_param_from_pattern (pat : Ppxlib.pattern) : Sarek_ast.param =
  let name =
    match extract_name_from_pattern pat with
    | Some n -> n
    | None -> raise (Parse_error_exn ("Expected named parameter", pat.ppat_loc))
  in
  let ty =
    match extract_type_from_pattern pat with
    | Some t -> t
    | None ->
        raise
          (Parse_error_exn
             ("Kernel parameters must have type annotations", pat.ppat_loc))
  in
  {
    Sarek_ast.param_name = name;
    Sarek_ast.param_type = ty;
    Sarek_ast.param_loc = loc_of_ppxlib pat.ppat_loc;
  }

(** Parse a Ppxlib pattern to Sarek pattern *)
let rec parse_pattern (pat : Ppxlib.pattern) : Sarek_ast.pattern =
  let loc = loc_of_ppxlib pat.ppat_loc in
  let pat_desc =
    match pat.ppat_desc with
    | Ppat_any -> Sarek_ast.PAny
    | Ppat_var {txt; _} -> Sarek_ast.PVar txt
    | Ppat_constraint (p, _) -> (parse_pattern p).Sarek_ast.pat
    | Ppat_construct ({txt = Lident name; _}, None) ->
        Sarek_ast.PConstr (name, None)
    | Ppat_construct ({txt = Lident name; _}, Some (_, arg)) ->
        Sarek_ast.PConstr (name, Some (parse_pattern arg))
    | Ppat_tuple ps -> Sarek_ast.PTuple (List.map parse_pattern ps)
    | _ -> raise (Parse_error_exn ("Unsupported pattern", pat.ppat_loc))
  in
  {Sarek_ast.pat = pat_desc; Sarek_ast.pat_loc = loc}

(** Parse a binary operator *)
let parse_binop (op : string) : Sarek_ast.binop option =
  match op with
  | "+" | "+." -> Some Sarek_ast.Add
  | "-" | "-." -> Some Sarek_ast.Sub
  | "*" | "*." -> Some Sarek_ast.Mul
  | "/" | "/." -> Some Sarek_ast.Div
  | "mod" -> Some Sarek_ast.Mod
  | "=" -> Some Sarek_ast.Eq
  | "<>" | "!=" -> Some Sarek_ast.Ne
  | "<" | "<." -> Some Sarek_ast.Lt
  | "<=" | "<=." -> Some Sarek_ast.Le
  | ">" | ">." -> Some Sarek_ast.Gt
  | ">=" | ">=." -> Some Sarek_ast.Ge
  | "&&" -> Some Sarek_ast.And
  | "||" -> Some Sarek_ast.Or
  | "land" -> Some Sarek_ast.Land
  | "lor" -> Some Sarek_ast.Lor
  | "lxor" -> Some Sarek_ast.Lxor
  | "lsl" -> Some Sarek_ast.Lsl
  | "lsr" -> Some Sarek_ast.Lsr
  | "asr" -> Some Sarek_ast.Asr
  | _ -> None

(** Parse a unary operator *)
let parse_unop (op : string) : Sarek_ast.unop option =
  match op with
  | "-" | "-." | "~-" | "~-." -> Some Sarek_ast.Neg
  | "not" -> Some Sarek_ast.Not
  | "lnot" -> Some Sarek_ast.Lnot
  | _ -> None

module Ast_502 = Astlib.Ast_502
module To_502 =
  Ppxlib_ast__Versions.Convert
    (Ppxlib_ast__Versions.OCaml_current)
    (Ppxlib_ast__Versions.OCaml_502)
module From_502 =
  Ppxlib_ast__Versions.Convert
    (Ppxlib_ast__Versions.OCaml_502)
    (Ppxlib_ast__Versions.OCaml_current)

let expression_to_502 expr =
  expr |> Selected_ast.to_ocaml Expression |> To_502.copy_expression

let expression_of_502 expr =
  expr |> From_502.copy_expression |> Selected_ast.of_ocaml Expression

let pattern_of_502 pat =
  pat |> From_502.copy_pattern |> Selected_ast.of_ocaml Pattern

let case_of_502 case = case |> From_502.copy_case |> Selected_ast.of_ocaml Case

type fun_body = Fun_body of expression | Fun_cases of case list

let is_function_expression_502 expr =
  let module P = Ast_502.Parsetree in
  match (expression_to_502 expr).P.pexp_desc with
  | P.Pexp_function _ -> true
  | _ -> false

let same_position (a : Lexing.position) (b : Lexing.position) =
  String.equal a.pos_fname b.pos_fname
  && a.pos_lnum = b.pos_lnum && a.pos_bol = b.pos_bol && a.pos_cnum = b.pos_cnum

let same_location (a : Location.t) (b : Location.t) =
  same_position a.loc_start b.loc_start && same_position a.loc_end b.loc_end

let expression_at_loc (root : expression) (loc : Location.t) =
  let found = ref None in
  let seen_root = ref false in
  let finder =
    object
      inherit Ast_traverse.iter as super

      method! expression expr =
        match !found with
        | Some _ -> ()
        | None ->
            let is_root = not !seen_root in
            seen_root := true ;
            if (not is_root) && same_location expr.pexp_loc loc then
              found := Some expr
            else super#expression expr
    end
  in
  finder#expression root ;
  !found

let pattern_of_param (p : Ppxlib.pattern) : Ppxlib.pattern = p

let collect_fun_params (expr : expression) :
    Ppxlib.pattern list * fun_body option =
  let module P = Ast_502.Parsetree in
  let module A = Ast_502.Asttypes in
  let rec loop acc e =
    match e.P.pexp_desc with
    | P.Pexp_function (params, _, body) -> (
        let collect_param acc p =
          match p.P.pparam_desc with
          | P.Pparam_val (A.Nolabel, None, pat) -> pat :: acc
          | P.Pparam_val (_, _, pat) ->
              raise
                (Parse_error_exn
                   ( "Labelled parameters not supported in kernels",
                     pat.P.ppat_loc ))
          | P.Pparam_newtype name ->
              raise
                (Parse_error_exn
                   ( "Locally abstract type parameters not supported in kernels",
                     name.loc ))
        in
        let acc = List.fold_left collect_param acc params in
        match body with
        | P.Pfunction_body body_expr -> loop acc body_expr
        | P.Pfunction_cases (cases, _, _) ->
            ( List.rev_map pattern_of_502 acc,
              Some (Fun_cases (List.map case_of_502 cases)) ))
    | _ ->
        if acc = [] then ([], None)
        else
          let body_expr =
            match expression_at_loc expr e.P.pexp_loc with
            | Some original when not (is_function_expression_502 original) ->
                original
            | None -> expression_of_502 e
            | Some _ -> expression_of_502 e
          in
          (List.rev_map pattern_of_502 acc, Some (Fun_body body_expr))
  in
  loop [] (expression_to_502 expr)

let is_function_expression = is_function_expression_502
