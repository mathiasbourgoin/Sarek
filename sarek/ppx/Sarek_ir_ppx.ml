(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Sarek PPX - GPU kernel DSL for OCaml
 *
 * This module defines the Sarek IR types for use within the PPX.
 * These types mirror Sarek.Sarek_ir (runtime) but are available at compile time.
 * The PPX lowers typed AST to these types, then quotes them to runtime code.
 ******************************************************************************)

(** Memory spaces *)
type memspace = Global | Shared | Local

(** Element types *)
type elttype =
  | TInt32
  | TInt64
  | TFloat32
  | TFloat64
  | TBool
  | TUnit
  | TRecord of string * (string * elttype) list
  | TVariant of string * (string * elttype list) list
  | TArray of elttype * memspace
  | TVec of elttype

(** Variables with type info *)
type var = {
  var_name : string;
  var_id : int;
  var_type : elttype;
  var_mutable : bool;
}

(** Constants *)
type const =
  | CInt32 of int32
  | CInt64 of int64
  | CFloat32 of float
  | CFloat64 of float
  | CBool of bool
  | CUnit

(** Binary operators *)
type binop =
  | Add
  | Sub
  | Mul
  | Div
  | Mod
  | Eq
  | Ne
  | Lt
  | Le
  | Gt
  | Ge
  | And
  | Or
  | Shl
  | Shr
  | BitAnd
  | BitOr
  | BitXor

(** Unary operators *)
type unop = Neg | Not | BitNot

(** Loop direction *)
type for_dir = Upto | Downto

(** Match pattern *)
type pattern = PConstr of string * string list | PWild

(** Expressions (pure, no side effects) *)
type expr =
  | EConst of const
  | EVar of var
  | EBinop of binop * expr * expr
  | EUnop of unop * expr
  | EArrayRead of string * expr
  | EArrayReadExpr of
      expr * expr (* base expr, index expr - for complex bases *)
  | ERecordField of expr * string
  | EIntrinsic of string list * string * expr list
  | ECast of elttype * expr
  | ETuple of expr list
  | EApp of expr * expr list
  | ERecord of string * (string * expr) list
  | EVariant of string * string * expr list
  | EArrayLen of string
  | EArrayCreate of elttype * expr * memspace  (** elem type, size, memspace *)
  | EIf of
      expr * expr * expr (* condition, then, else - for value-returning if *)
  | EMatch of
      expr
      * (pattern * expr) list (* scrutinee, cases - for value-returning match *)

(** L-values (assignable locations) *)
type lvalue =
  | LVar of var
  | LArrayElem of string * expr
  | LArrayElemExpr of
      expr * expr (* base expr, index expr - for complex bases *)
  | LRecordField of lvalue * string

(** Statements (imperative, side effects) *)
type stmt =
  | SAssign of lvalue * expr
  | SSeq of stmt list
  | SIf of expr * stmt * stmt option
  | SWhile of expr * stmt
  | SFor of var * expr * expr * for_dir * stmt
  | SMatch of expr * (pattern * stmt) list
  | SReturn of expr
  | SBarrier
  | SWarpBarrier
  | SExpr of expr
  | SEmpty
  | SLet of var * expr * stmt
  | SLetMut of var * expr * stmt
  | SPragma of string list * stmt
  | SMemFence
  | SBlock of stmt  (** Scoped block for C variable isolation *)
  | SNative of {
      gpu : Ppxlib.expression;  (** fun dev -> "cuda/opencl code" *)
      ocaml : Ppxlib.expression;  (** OCaml fallback for interpreter/native *)
    }  (** Inline native GPU code with OCaml fallback *)

(** Array info for parameters *)
type array_info = {arr_elttype : elttype; arr_memspace : memspace}

(** Declarations *)
type decl =
  | DParam of var * array_info option
  | DLocal of var * expr option
  | DShared of string * elttype * expr option

(** Helper function (device function called from kernel) *)
type helper_func = {
  hf_name : string;
  hf_params : var list;
  hf_ret_type : elttype;
  hf_body : stmt;
}

(** Kernel representation *)
type kernel = {
  kern_name : string;
  kern_params : decl list;
  kern_locals : decl list;
  kern_body : stmt;
  kern_types : (string * (string * elttype) list) list;
      (** Record type definitions: (type_name, [(field_name, field_type); ...])
      *)
  kern_variants : (string * (string * elttype list) list) list;
      (** Variant type definitions: (type_name,
          [(constructor_name, payload_types); ...]) *)
  kern_funcs : helper_func list;
      (** Helper functions defined in kernel scope *)
  kern_native_fn : unit option;
      (** Placeholder for native function - actual function added during quoting
      *)
}
