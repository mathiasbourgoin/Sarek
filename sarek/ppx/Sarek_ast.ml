(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Sarek PPX - GPU kernel DSL for OCaml
 *
 * This module defines the framework-independent source AST for Sarek kernels.
 * It replaces the camlp4-coupled AST from the old implementation.
 ******************************************************************************)

(** Source locations *)
type loc = {
  loc_file : string;
  loc_line : int;
  loc_col : int;
  loc_end_line : int;
  loc_end_col : int;
}

let dummy_loc =
  {
    loc_file = "<dummy>";
    loc_line = 1;
    loc_col = 0;
    loc_end_line = 1;
    loc_end_col = 0;
  }

let loc_of_ppxlib (loc : Ppxlib.Location.t) : loc =
  {
    loc_file = loc.loc_start.pos_fname;
    loc_line = loc.loc_start.pos_lnum;
    loc_col = loc.loc_start.pos_cnum - loc.loc_start.pos_bol;
    loc_end_line = loc.loc_end.pos_lnum;
    loc_end_col = loc.loc_end.pos_cnum - loc.loc_end.pos_bol;
  }

let loc_to_ppxlib (loc : loc) : Ppxlib.Location.t =
  let open Lexing in
  let pos_start =
    {
      pos_fname = loc.loc_file;
      pos_lnum = loc.loc_line;
      pos_bol = 0;
      pos_cnum = loc.loc_col;
    }
  in
  let pos_end =
    {
      pos_fname = loc.loc_file;
      pos_lnum = loc.loc_end_line;
      pos_bol = 0;
      pos_cnum = loc.loc_end_col;
    }
  in
  {loc_start = pos_start; loc_end = pos_end; loc_ghost = false}

(** Memory spaces for arrays *)
type memspace =
  | Local  (** Thread-private memory *)
  | Shared  (** Block-shared memory *)
  | Global  (** Global device memory *)

(** Type syntax (what user writes, before inference) *)
type type_expr =
  | TEVar of string  (** 'a, 'b - type variables *)
  | TEConstr of string * type_expr list  (** int32, float32 vector, etc. *)
  | TEArrow of type_expr * type_expr  (** a -> b *)
  | TETuple of type_expr list  (** a * b *)

(** Binary operators - single constructor each, type resolved during typing *)
type binop =
  | Add
  | Sub
  | Mul
  | Div
  | Mod
  | And
  | Or
  | Eq
  | Ne
  | Lt
  | Le
  | Gt
  | Ge
  | Land
  | Lor
  | Lxor
  | Lsl
  | Lsr
  | Asr  (** Bitwise ops *)

(** Unary operators *)
type unop =
  | Neg  (** Arithmetic negation *)
  | Not  (** Logical not *)
  | Lnot  (** Bitwise not *)

(** Patterns for match expressions *)
type pattern = {pat : pattern_desc; pat_loc : loc}

and pattern_desc =
  | PAny  (** _ *)
  | PVar of string  (** x *)
  | PConstr of string * pattern option  (** Some x, None *)
  | PTuple of pattern list  (** (a, b) *)

(** Kernel parameters - defined before expr so ELetRec can reference it *)
type param = {param_name : string; param_type : type_expr; param_loc : loc}

(** Expressions *)
type expr = {e : expr_desc; expr_loc : loc}

and expr_desc =
  (* Literals *)
  | EUnit
  | EBool of bool
  | EInt of int  (** int literal (will be int32 in kernel) *)
  | EInt32 of int32
  | EInt64 of int64
  | EFloat of float  (** float32 *)
  | EDouble of float  (** float64 *)
  (* Variables and access *)
  | EVar of string
  | EVecGet of expr * expr  (** v.[i] - vector element access *)
  | EVecSet of expr * expr * expr  (** v.[i] <- x *)
  | EArrGet of expr * expr  (** a.(i) - array element access *)
  | EArrSet of expr * expr * expr  (** a.(i) <- x *)
  | EFieldGet of expr * string  (** r.field *)
  | EFieldSet of expr * string * expr  (** r.field <- x *)
  (* Operations *)
  | EBinop of binop * expr * expr
  | EUnop of unop * expr
  | EApp of expr * expr list  (** f x y *)
  | EAssign of string * expr  (** x := e *)
  (* Binding and control *)
  | ELet of string * type_expr option * expr * expr  (** let x : t = e1 in e2 *)
  | ELetRec of string * param list * type_expr option * expr * expr
      (** let rec f params : ret_ty = body in cont *)
  | ELetMut of string * type_expr option * expr * expr
      (** let mutable x = ... *)
  | EIf of expr * expr * expr option  (** if c then a [else b] *)
  | EFor of string * expr * expr * for_dir * expr
      (** for i = a to/downto b do ... done *)
  | EWhile of expr * expr
  | ESeq of expr * expr
  | EMatch of expr * (pattern * expr) list
  (* Records and variants *)
  | ERecord of string option * (string * expr) list
      (** Record literal with optional type name *)
  | EConstr of string * expr option  (** Constructor application *)
  | ETuple of expr list  (** (a, b, c) *)
  (* Special forms *)
  | EReturn of expr
  | ECreateArray of expr * type_expr * memspace
  | EGlobalRef of string  (** Reference to OCaml value by name *)
  | ENative of {
      gpu : Ppxlib.expression;  (** fun dev -> "cuda/opencl code" *)
      ocaml : Ppxlib.expression;  (** fun arg1 arg2 ... -> OCaml fallback *)
    }
      (** Native code with GPU string generator and OCaml fallback function.
          Usage: [%native (fun dev -> "code"), (fun x y -> ...)] x y The result
          is a function that takes the same args as the ocaml fallback. *)
  | EPragma of string list * expr  (** pragma "unroll" body *)
  (* Type annotation *)
  | ETyped of expr * type_expr
  (* Module access *)
  | EOpen of string list * expr  (** let open M.N in e *)
  (* BSP superstep constructs *)
  | ELetShared of string * type_expr * expr option * expr
      (** let%shared name : elem_type [= size] in body *)
  | ESuperstep of string * bool * expr * expr
      (** let%superstep [~divergent] name = body in cont *)

and for_dir = Upto | Downto

(** Kernel-local type declarations *)
type type_decl =
  | Type_record of {
      tdecl_name : string;
      tdecl_module : string option;
      tdecl_fields : (string * bool * type_expr) list;
          (** name, mutable, type *)
      tdecl_loc : loc;
    }
  | Type_variant of {
      tdecl_name : string;
      tdecl_module : string option;
      tdecl_constructors : (string * type_expr option) list;
      tdecl_loc : loc;
    }

(** Module-level items (constants or functions) within a kernel payload *)
type module_item =
  | MConst of string * type_expr * expr  (** let name : ty = expr *)
  | MFun of string * bool * param list * expr
      (** MFun(name, is_recursive, params, body) *)

(** A complete kernel definition *)
type kernel = {
  kern_name : string option;  (** None for anonymous kernels *)
  kern_types : type_decl list;  (** Type declarations visible in body *)
  kern_module_items : module_item list;
      (** Module-level items visible in body *)
  kern_external_item_count : int;
      (** Number of items in kern_module_items that are external (from
          [@sarek.module]). First N items are external, rest are inline. *)
  kern_params : param list;
  kern_body : expr;
  kern_loc : loc;
}

(** Pretty printing for debugging *)
let rec pp_type_expr fmt = function
  | TEVar s -> Format.fprintf fmt "'%s" s
  | TEConstr (name, []) -> Format.fprintf fmt "%s" name
  | TEConstr (name, args) ->
      Format.fprintf
        fmt
        "(%a) %s"
        (Format.pp_print_list
           ~pp_sep:(fun fmt () -> Format.fprintf fmt ", ")
           pp_type_expr)
        args
        name
  | TEArrow (a, b) ->
      Format.fprintf fmt "%a -> %a" pp_type_expr a pp_type_expr b
  | TETuple ts ->
      Format.fprintf
        fmt
        "(%a)"
        (Format.pp_print_list
           ~pp_sep:(fun fmt () -> Format.fprintf fmt " * ")
           pp_type_expr)
        ts

let pp_binop fmt = function
  | Add -> Format.fprintf fmt "+"
  | Sub -> Format.fprintf fmt "-"
  | Mul -> Format.fprintf fmt "*"
  | Div -> Format.fprintf fmt "/"
  | Mod -> Format.fprintf fmt "mod"
  | And -> Format.fprintf fmt "&&"
  | Or -> Format.fprintf fmt "||"
  | Eq -> Format.fprintf fmt "="
  | Ne -> Format.fprintf fmt "<>"
  | Lt -> Format.fprintf fmt "<"
  | Le -> Format.fprintf fmt "<="
  | Gt -> Format.fprintf fmt ">"
  | Ge -> Format.fprintf fmt ">="
  | Land -> Format.fprintf fmt "land"
  | Lor -> Format.fprintf fmt "lor"
  | Lxor -> Format.fprintf fmt "lxor"
  | Lsl -> Format.fprintf fmt "lsl"
  | Lsr -> Format.fprintf fmt "lsr"
  | Asr -> Format.fprintf fmt "asr"

let pp_unop fmt = function
  | Neg -> Format.fprintf fmt "-"
  | Not -> Format.fprintf fmt "not"
  | Lnot -> Format.fprintf fmt "lnot"

let pp_memspace fmt = function
  | Local -> Format.fprintf fmt "local"
  | Shared -> Format.fprintf fmt "shared"
  | Global -> Format.fprintf fmt "global"

let rec pp_pattern fmt pat =
  match pat.pat with
  | PAny -> Format.fprintf fmt "_"
  | PVar s -> Format.fprintf fmt "%s" s
  | PConstr (name, None) -> Format.fprintf fmt "%s" name
  | PConstr (name, Some p) -> Format.fprintf fmt "%s %a" name pp_pattern p
  | PTuple ps ->
      Format.fprintf
        fmt
        "(%a)"
        (Format.pp_print_list
           ~pp_sep:(fun fmt () -> Format.fprintf fmt ", ")
           pp_pattern)
        ps

let rec pp_expr fmt expr =
  match expr.e with
  | EUnit -> Format.fprintf fmt "()"
  | EBool b -> Format.fprintf fmt "%b" b
  | EInt i -> Format.fprintf fmt "%d" i
  | EInt32 i -> Format.fprintf fmt "%ldl" i
  | EInt64 i -> Format.fprintf fmt "%LdL" i
  | EFloat f -> Format.fprintf fmt "%f" f
  | EDouble f -> Format.fprintf fmt "%f" f
  | EVar s -> Format.fprintf fmt "%s" s
  | EVecGet (v, i) -> Format.fprintf fmt "%a.[%a]" pp_expr v pp_expr i
  | EVecSet (v, i, x) ->
      Format.fprintf fmt "%a.[%a] <- %a" pp_expr v pp_expr i pp_expr x
  | EArrGet (a, i) -> Format.fprintf fmt "%a.(%a)" pp_expr a pp_expr i
  | EArrSet (a, i, x) ->
      Format.fprintf fmt "%a.(%a) <- %a" pp_expr a pp_expr i pp_expr x
  | EFieldGet (r, f) -> Format.fprintf fmt "%a.%s" pp_expr r f
  | EFieldSet (r, f, x) ->
      Format.fprintf fmt "%a.%s <- %a" pp_expr r f pp_expr x
  | EBinop (op, a, b) ->
      Format.fprintf fmt "(%a %a %a)" pp_expr a pp_binop op pp_expr b
  | EUnop (op, a) -> Format.fprintf fmt "(%a %a)" pp_unop op pp_expr a
  | EApp (f, args) ->
      Format.fprintf
        fmt
        "(%a %a)"
        pp_expr
        f
        (Format.pp_print_list
           ~pp_sep:(fun fmt () -> Format.fprintf fmt " ")
           pp_expr)
        args
  | EAssign (name, value) -> Format.fprintf fmt "(%s := %a)" name pp_expr value
  | ELet (name, ty, value, body) ->
      Format.fprintf
        fmt
        "(let %s%a = %a in %a)"
        name
        (fun fmt -> function
          | None -> () | Some t -> Format.fprintf fmt " : %a" pp_type_expr t)
        ty
        pp_expr
        value
        pp_expr
        body
  | ELetRec (name, params, ret_ty, fn_body, cont) ->
      Format.fprintf
        fmt
        "(let rec %s %a%a = %a in %a)"
        name
        (Format.pp_print_list
           ~pp_sep:(fun fmt () -> Format.fprintf fmt " ")
           (fun fmt p ->
             Format.fprintf
               fmt
               "(%s : %a)"
               p.param_name
               pp_type_expr
               p.param_type))
        params
        (fun fmt -> function
          | None -> () | Some t -> Format.fprintf fmt " : %a" pp_type_expr t)
        ret_ty
        pp_expr
        fn_body
        pp_expr
        cont
  | ELetMut (name, ty, value, body) ->
      Format.fprintf
        fmt
        "(let mutable %s%a = %a in %a)"
        name
        (fun fmt -> function
          | None -> () | Some t -> Format.fprintf fmt " : %a" pp_type_expr t)
        ty
        pp_expr
        value
        pp_expr
        body
  | EIf (c, t, None) -> Format.fprintf fmt "(if %a then %a)" pp_expr c pp_expr t
  | EIf (c, t, Some e) ->
      Format.fprintf fmt "(if %a then %a else %a)" pp_expr c pp_expr t pp_expr e
  | EFor (var, lo, hi, dir, body) ->
      Format.fprintf
        fmt
        "(for %s = %a %s %a do %a done)"
        var
        pp_expr
        lo
        (match dir with Upto -> "to" | Downto -> "downto")
        pp_expr
        hi
        pp_expr
        body
  | EWhile (c, body) ->
      Format.fprintf fmt "(while %a do %a done)" pp_expr c pp_expr body
  | ESeq (a, b) -> Format.fprintf fmt "(%a; %a)" pp_expr a pp_expr b
  | EMatch (e, cases) ->
      Format.fprintf
        fmt
        "(match %a with %a)"
        pp_expr
        e
        (Format.pp_print_list
           ~pp_sep:(fun fmt () -> Format.fprintf fmt " | ")
           (fun fmt (p, e) ->
             Format.fprintf fmt "%a -> %a" pp_pattern p pp_expr e))
        cases
  | ERecord (name, fields) ->
      Format.fprintf
        fmt
        "{%a%a}"
        (fun fmt -> function
          | None -> () | Some n -> Format.fprintf fmt "%s with " n)
        name
        (Format.pp_print_list
           ~pp_sep:(fun fmt () -> Format.fprintf fmt "; ")
           (fun fmt (f, e) -> Format.fprintf fmt "%s = %a" f pp_expr e))
        fields
  | EConstr (name, None) -> Format.fprintf fmt "%s" name
  | EConstr (name, Some e) -> Format.fprintf fmt "(%s %a)" name pp_expr e
  | ETuple es ->
      Format.fprintf
        fmt
        "(%a)"
        (Format.pp_print_list
           ~pp_sep:(fun fmt () -> Format.fprintf fmt ", ")
           pp_expr)
        es
  | EReturn e -> Format.fprintf fmt "(return %a)" pp_expr e
  | ECreateArray (size, ty, mem) ->
      Format.fprintf
        fmt
        "(create_array %a : %a %a)"
        pp_expr
        size
        pp_type_expr
        ty
        pp_memspace
        mem
  | EGlobalRef name -> Format.fprintf fmt "@%s" name
  | ENative _ -> Format.fprintf fmt "[%%native ~gpu:... ~ocaml:...]"
  | EPragma (opts, body) ->
      Format.fprintf
        fmt
        "(pragma %a %a)"
        (Format.pp_print_list
           ~pp_sep:(fun fmt () -> Format.fprintf fmt " ")
           Format.pp_print_string)
        opts
        pp_expr
        body
  | ETyped (e, ty) -> Format.fprintf fmt "(%a : %a)" pp_expr e pp_type_expr ty
  | EOpen (path, e) ->
      Format.fprintf
        fmt
        "(let open %s in %a)"
        (String.concat "." path)
        pp_expr
        e
  | ELetShared (name, elem_ty, size_opt, body) ->
      Format.fprintf
        fmt
        "(let%%shared %s : %a%a in %a)"
        name
        pp_type_expr
        elem_ty
        (fun fmt -> function
          | None -> () | Some size -> Format.fprintf fmt " = %a" pp_expr size)
        size_opt
        pp_expr
        body
  | ESuperstep (name, divergent, step_body, cont) ->
      Format.fprintf
        fmt
        "(let%%superstep %s%s = %a in %a)"
        (if divergent then "~divergent " else "")
        name
        pp_expr
        step_body
        pp_expr
        cont

let pp_param fmt p =
  Format.fprintf fmt "(%s : %a)" p.param_name pp_type_expr p.param_type

let pp_kernel fmt k =
  Format.fprintf
    fmt
    "kernel %a (%a) = %a"
    (fun fmt -> function
      | None -> Format.fprintf fmt "<anon>"
      | Some n -> Format.fprintf fmt "%s" n)
    k.kern_name
    (Format.pp_print_list
       ~pp_sep:(fun fmt () -> Format.fprintf fmt " ")
       pp_param)
    k.kern_params
    pp_expr
    k.kern_body
