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

(** A functor application inside a type path ([F(X).t]). [flatten] used to
    return [[]] for it, so the type came out named [""] — and an unresolvable
    name is turned into an empty placeholder record by
    [Sarek_types.type_of_type_expr], not into an error. *)
let functor_type_path_msg =
  "a functor application in a type path (`F(X).t`) is not supported in a \
   kernel: kernel types are resolved by NAME against the types registered with \
   [@@sarek.type], and a functor application has no name to register. Alias \
   the result outside the kernel (`module M = F(X)`) and write `M.t`."

(** A labelled or optional argument in an ARROW TYPE. [Ptyp_arrow]'s label used
    to be read as [_], so [x:int -> int] and [int -> int] parsed to the same
    [TEArrow] — the label was dropped from the type while the corresponding
    parameter is refused outright by [collect_fun_params]. *)
let labelled_arrow_type_msg =
  "a labelled or optional argument in a function type (`x:t -> u`, `?x:t -> \
   u`) is not supported in a kernel: kernel functions take positional \
   arguments only (a labelled PARAMETER is refused for the same reason), so \
   the label would have to be dropped from the type. Write `t -> u`."

(** Parse a core_type to type_expr.

    The final arm refuses rather than returning a placeholder: it used to answer
    [TEConstr ("unknown", [])] for every core_type shape not listed, and
    [Sarek_types.type_of_type_expr] maps an unrecognised constructor to
    [TRecord (name, [])] — an empty record type. So an annotation the parser
    could not read became a phantom type named ["unknown"] that unified with
    nothing and was reported, if at all, as a type error somewhere else
    (backlog-192). *)
let rec parse_type (ct : core_type) : Sarek_ast.type_expr =
  match ct.ptyp_desc with
  | Ptyp_constr ({txt; _}, args) ->
      let rec flatten = function
        | Longident.Lident s -> [s]
        | Longident.Ldot (li, s) -> flatten li @ [s]
        | Longident.Lapply _ ->
            raise (Parse_error_exn (functor_type_path_msg, ct.ptyp_loc))
      in
      let name = String.concat "." (flatten txt) in
      Sarek_ast.TEConstr (name, List.map parse_type args)
  (* [Ptyp_poly]'s quantifier list is deliberately not read: it binds the type
     variables that [Ptyp_var] below already carries by name, so nothing about
     the type is lost by looking through it. *)
  | Ptyp_poly (_, ct) -> parse_type ct
  | Ptyp_var name -> Sarek_ast.TEVar name
  | Ptyp_arrow (Nolabel, t1, t2) ->
      Sarek_ast.TEArrow (parse_type t1, parse_type t2)
  | Ptyp_arrow ((Labelled _ | Optional _), _, _) ->
      raise (Parse_error_exn (labelled_arrow_type_msg, ct.ptyp_loc))
  | Ptyp_tuple ts -> Sarek_ast.TETuple (List.map parse_type ts)
  | d ->
      raise
        (Parse_error_exn (Sarek_unsupported.core_type_refusal d, ct.ptyp_loc))

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

(** A GADT-style constructor declaration ([C : int -> t]). [pcd_res] and
    [pcd_vars] were never read, so the return type and the existential binders
    were dropped and the constructor was recorded as if it had been written
    [C of int] — for a parameterised type that is a different declaration. *)
let gadt_constructor_msg =
  "a GADT-style constructor declaration (`C : t -> u`, `C : type a. ...`) is \
   not supported in a kernel type: a variant is lowered to a tag plus one \
   payload whose type is fixed by the declaration, so a per-constructor return \
   type has nowhere to go. Write `C of t`."

let parse_variant_constructors constrs =
  let parse_arg loc = function
    | Pcstr_tuple [] -> None
    | Pcstr_tuple [arg] -> Some (parse_type arg)
    | Pcstr_tuple _ ->
        raise
          (Parse_error_exn
             ("Constructors with multiple arguments are not supported", loc))
    | Pcstr_record _ ->
        raise
          (Parse_error_exn
             ("Record constructors are not supported in kernels", loc))
  in
  List.map
    (fun (cd : constructor_declaration) ->
      (* [pcd_attributes] is not read here, and that is a residual drop rather
         than a justified omission. This helper is shared: for a TOP-LEVEL
         [@@sarek.type] declaration the same constructor also reaches OCaml,
         which is entitled to its attributes, so refusing here would break code
         that compiles today; but for a PAYLOAD-LOCAL type (consumed by the PPX,
         never seen by OCaml) nothing reads them at all. An earlier revision of
         this comment claimed the first case covered both. [pcd_res]/[pcd_vars]
         are a different matter — nothing reads them in either case, so they are
         refused. *)
      if cd.pcd_res <> None || cd.pcd_vars <> [] then
        raise (Parse_error_exn (gadt_constructor_msg, cd.pcd_loc)) ;
      let name = cd.pcd_name.txt in
      let arg = parse_arg cd.pcd_loc cd.pcd_args in
      (name, arg))
    constrs

(** Extract type annotation from a Ppxlib pattern if present *)
let rec extract_type_from_pattern (pat : Ppxlib.pattern) :
    Sarek_ast.type_expr option =
  match pat.ppat_desc with
  | Ppat_constraint (_, ct) -> Some (parse_type ct)
  | Ppat_alias (p, _) -> extract_type_from_pattern p
  | _ -> None

(** An [as] alias in a BINDER position ([let (p as x) = e], [fun (p as x) ->]).
    [extract_name_from_pattern] used to answer the alias name and throw the
    inner pattern away, so every name [p] bound was silently absent from the
    kernel environment and surfaced later as an unbound variable — pointing at
    the USE, not at the alias. *)
let alias_binder_msg =
  "an `as` alias in a kernel binder (`let (p as x) = e`, `fun (p as x) ->`) is \
   not supported: only the alias name is bound, and the names inside `p` would \
   silently not exist. Bind the value under one name and destructure it in a \
   `let` or a `match` of its own."

(** [let x :> t = e] — a coercion on a binding. *)
let binding_coercion_msg =
  "a coercion on a `let` binding (`let x :> t = e`) is not supported in a \
   kernel: there is no subtyping in the kernel type system, so a coercion has \
   nothing to mean. Use a type annotation (`let x : t = e`)."

(** [let f : type a. ...] — a locally abstract type on a binding. *)
let binding_univars_msg =
  "a locally abstract type on a `let` binding (`let f : type a. a -> a = ...`) \
   is not supported in a kernel: every kernel-local function is monomorphised \
   at its call sites, so it cannot carry a quantified type."

(** The declared type of a [let] binding, from EITHER spelling.

    [let (x : t) = e] puts the annotation in the PATTERN ([Ppat_constraint]).
    [let x : t = e] — the spelling almost everything in this tree uses — puts it
    in [pvb_constraint] instead, and that field was never read: the annotation
    was silently dropped, so a kernel-local [let sum : float = ...] was typed by
    inference with the declared width ignored, and an annotated module constant
    was dropped entirely by [Sarek_parse.parse_payload] for want of a type
    (backlog-192).

    Both are read here, pattern first. *)
let binding_type (vb : Ppxlib.value_binding) : Sarek_ast.type_expr option =
  let from_pattern =
    match vb.pvb_pat.ppat_desc with
    | Ppat_constraint (_, ct) -> Some (parse_type ct)
    | _ -> None
  in
  let from_constraint =
    match vb.pvb_constraint with
    | None -> None
    | Some (Pvc_constraint {locally_abstract_univars = []; typ}) ->
        Some (parse_type typ)
    | Some (Pvc_constraint {locally_abstract_univars = _ :: _; _}) ->
        raise (Parse_error_exn (binding_univars_msg, vb.pvb_loc))
    | Some (Pvc_coercion _) ->
        raise (Parse_error_exn (binding_coercion_msg, vb.pvb_loc))
  in
  match (from_pattern, from_constraint) with
  | None, None -> None
  | Some t, None | None, Some t -> Some t
  | Some _, Some _ ->
      (* [let (x : t1) : t2 = e] is legal OCaml and puts an annotation in BOTH
         places (checked with `ocamlc -stop-after parsing`). Preferring one and
         discarding the other is the defect this sweep is about, and nothing
         here checks that they agree, so the shape is refused. *)
      raise
        (Parse_error_exn
           ( "this binding is annotated twice (`let (x : t1) : t2 = e`), and \
              Sarek reads one annotation per binding — the other would be \
              discarded without being checked against it. Keep one.",
             vb.pvb_loc ))

(** Extract variable name from a Ppxlib pattern *)
let rec extract_name_from_pattern (pat : Ppxlib.pattern) : string option =
  match pat.ppat_desc with
  | Ppat_var {txt; _} -> Some txt
  | Ppat_constraint (p, _) -> extract_name_from_pattern p
  | Ppat_alias _ -> raise (Parse_error_exn (alias_binder_msg, pat.ppat_loc))
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

(** Existential type binders on a constructor pattern ([C (type a) p]). The
    binder list in [Ppat_construct]'s payload was read as [_], so the pattern
    parsed as the plain [C p] and the locally abstract type simply was not
    there. *)
let existential_pattern_msg =
  "an existential type binder in a constructor pattern (`C (type a) p`) is not \
   supported in a kernel: it comes with a GADT declaration, which a kernel \
   variant cannot be. Match the constructor without the binder."

(** Parse a Ppxlib pattern to Sarek pattern *)
let rec parse_pattern (pat : Ppxlib.pattern) : Sarek_ast.pattern =
  let loc = loc_of_ppxlib pat.ppat_loc in
  let pat_desc =
    match pat.ppat_desc with
    | Ppat_any -> Sarek_ast.PAny
    | Ppat_var {txt; _} -> Sarek_ast.PVar txt
    (* The annotation is deliberately looked THROUGH rather than read: a
       pattern's type is fixed by the scrutinee and by the constructor
       declaration, so the annotation cannot change what is lowered.

       An earlier revision added "this is what makes the documented
       [let ((a, b) : t) = e] spelling work", which is the opposite of what the
       code does: that spelling is REFUSED by [Sarek_parse.parse_let_form],
       which raises when [binding_type vb] is [Some] and reads the constraint
       straight off [pvb_pat]. Looking through it here is why the binding
       reaches the tuple branch at all — it is not what makes it succeed.
       Caught by CodeRabbit on #398. *)
    | Ppat_constraint (p, _) -> (parse_pattern p).Sarek_ast.pat
    | Ppat_construct ({txt = Lident name; _}, None) ->
        Sarek_ast.PConstr (name, None)
    | Ppat_construct ({txt = Lident name; _}, Some ([], arg)) ->
        Sarek_ast.PConstr (name, Some (parse_pattern arg))
    | Ppat_construct ({txt = Lident _; _}, Some (_ :: _, _)) ->
        raise (Parse_error_exn (existential_pattern_msg, pat.ppat_loc))
    | Ppat_tuple ps -> Sarek_ast.PTuple (List.map parse_pattern ps)
    | d ->
        raise
          (Parse_error_exn (Sarek_unsupported.pattern_refusal d, pat.ppat_loc))
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

(** A coercion in a function's return position ([fun x :> t -> e]). *)
let return_coercion_msg =
  "a coercion in a function's return position (`fun x :> t -> e`, `let f x :> \
   t = e`) is not supported in a kernel: there is no subtyping in the kernel \
   type system, so a coercion has nothing to mean. Use a type annotation (`: \
   t`)."

(** An annotation with fewer arrows than the function has parameters. Shared
    between the two places that put a declared type into a RESULT slot. *)
let annotation_arity_msg =
  "this annotation has fewer arrows than the function has parameters, so Sarek \
   cannot tell which part of it is the result type. Annotate the result \
   instead (`let f x : t = ...`), or give the full arrow type."

(** Strip [n] leading arrows off a type, or [None] if it has fewer than [n]. *)
let rec peel_arrows n (t : Sarek_ast.type_expr) : Sarek_ast.type_expr option =
  if n = 0 then Some t
  else match t with TEArrow (_, r) -> peel_arrows (n - 1) r | _ -> None

(** The declared RESULT type of a function expression, if it has one.

    [Pexp_function]'s [type_constraint option] slot is where OCaml >= 5.1 puts
    the [: t] of [let f (x : int32) : int32 = ...] — NOT in the pattern. It was
    read as [_] by [collect_fun_params], so every kernel helper's declared
    return type was silently discarded and the helper's result type came from
    inference alone (backlog-192).

    EVERY [Pexp_function] on the way down is inspected, not just the outermost.
    An earlier revision of this function read only the outermost and justified
    it by saying a nested one "belongs to that inner function and not to the
    binding" — which is false, because [collect_fun_params] below DESCENDS
    through [Pfunction_body] and merges the inner function's parameters into the
    binding's list. After that flattening there is no inner function left for
    the annotation to belong to, and
    [let f (x : int32) = fun (y : int32) : float32 -> ...] had its [float32]
    dropped while the flattened spelling of the same function had it honoured.
    Measured: the two spellings compiled to different verdicts, exit 0 versus a
    unification error.

    The LAST constraint on the way down wins, and it is peeled by the number of
    parameters collected AFTER it — because a constraint sitting above further
    parameters describes a FUNCTION type, not the flattened result.
    [let f (x : int32) : (int32 -> int32) = fun (y : int32) -> y] therefore
    yields [int32], not the arrow. Too few arrows is refused rather than
    half-applied. *)
let fun_return_type (expr : expression) : Sarek_ast.type_expr option =
  let module P = Ast_502.Parsetree in
  (* [found] is the last constraint seen and how many parameters have been
     collected since; it mirrors [collect_fun_params]'s descent exactly. *)
  let rec loop found params_since e =
    match e.P.pexp_desc with
    | P.Pexp_function (params, ct, body) -> (
        let params_since = params_since + List.length params in
        let found, params_since =
          match (ct, found) with
          | None, _ -> (found, params_since)
          | Some (P.Pconstraint c), None -> (Some c, 0)
          | Some (P.Pconstraint _), Some _ ->
              (* Two result annotations on one binding, e.g. [let f (x : 'a) :
                 ('a -> 'a) = fun (y : 'b) : 'b -> y]. Only one can reach the
                 single result slot, so keeping the inner one would discard the
                 relationship the outer one states between the parameters and
                 the result. Refuse instead of picking (found by the
                 cross-runtime review of this branch's first version, which
                 picked the inner one silently). *)
              raise
                (Parse_error_exn
                   ( "this binding carries two result annotations (one on an \
                      outer `fun` and one on an inner one). A kernel function \
                      has a single result type, so only one of them could be \
                      honoured and the other would be discarded without being \
                      checked against it. Keep one.",
                     expr.pexp_loc ))
          | Some (P.Pcoerce _), _ ->
              raise (Parse_error_exn (return_coercion_msg, expr.pexp_loc))
        in
        (* [Pfunction_cases] is refused by every caller through [Fun_cases]; it
           collects no further parameters here. *)
        match body with
        | P.Pfunction_body b -> loop found params_since b
        | P.Pfunction_cases _ -> (found, params_since))
    | _ -> (found, params_since)
  in
  match loop None 0 (expression_to_502 expr) with
  | None, _ -> None
  | Some ct, params_since -> (
      let ty =
        parse_type
          (From_502.copy_core_type ct |> Selected_ast.of_ocaml Core_type)
      in
      match peel_arrows params_since ty with
      | Some r -> Some r
      | None -> raise (Parse_error_exn (annotation_arity_msg, expr.pexp_loc)))

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
