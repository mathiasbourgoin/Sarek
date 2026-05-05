(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Sarek_ir_types - Pure type definitions for GPU kernel IR

    This module contains only type definitions with no external dependencies.
    Used by spoc_framework for typed generate_source signature. *)

(** Runtime type identities with equality proofs.

    This uses generative extensible GADT constructors, so equality can be checked
    without unsafe casts. *)
module Type_id = struct
  type _ witness = ..

  module type ID = sig
    type t

    type _ witness += Id : t witness
  end

  type 'a t = (module ID with type t = 'a)

  type (_, _) eq = Refl : ('a, 'a) eq

  let create (type a) () : a t =
    let module M = struct
      type t = a

      type _ witness += Id : t witness
    end in
    (module M)

  let equal (type a b) (id_a : a t) (id_b : b t) : (a, b) eq option =
    let module A = (val id_a : ID with type t = a) in
    let module B = (val id_b : ID with type t = b) in
    match A.Id with B.Id -> Some Refl | _ -> None
end

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
      (** Record type: name and field list *)
  | TVariant of string * (string * elttype list) list
      (** Variant type: name and constructor list with arg types *)
  | TArray of elttype * memspace
      (** Array type with element type and memory space *)
  | TVec of elttype  (** Vector (GPU array parameter) *)

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
type pattern =
  | PConstr of string * string list (* Constructor name, bound vars *)
  | PWild

(** Expressions (pure, no side effects) *)
type expr =
  | EConst of const
  | EVar of var
  | EBinop of binop * expr * expr
  | EUnop of unop * expr
  | EArrayRead of string * expr  (** arr[idx] *)
  | EArrayReadExpr of expr * expr  (** base_expr[idx] for complex bases *)
  | ERecordField of expr * string  (** r.field *)
  | EIntrinsic of string list * string * expr list
      (** module path, name, args *)
  | ECast of elttype * expr
  | ETuple of expr list
  | EApp of expr * expr list
  | ERecord of string * (string * expr) list
      (** Record construction: type name, field values *)
  | EVariant of string * string * expr list
      (** Variant construction: type name, constructor, args *)
  | EArrayLen of string  (** Array length intrinsic *)
  | EArrayCreate of elttype * expr * memspace  (** elem type, size, memspace *)
  | EIf of expr * expr * expr  (** condition, then, else - value-returning if *)
  | EMatch of expr * (pattern * expr) list
      (** scrutinee, cases - value-returning match *)

(** L-values (assignable locations) *)
type lvalue =
  | LVar of var
  | LArrayElem of string * expr (* arr[idx] *)
  | LArrayElemExpr of expr * expr (* base_expr[idx] for complex bases *)
  | LRecordField of lvalue * string (* r.field *)

(** Statements (imperative, side effects) *)
type stmt =
  | SAssign of lvalue * expr
  | SSeq of stmt list
  | SIf of expr * stmt * stmt option
  | SWhile of expr * stmt
  | SFor of var * expr * expr * for_dir * stmt
  | SMatch of expr * (pattern * stmt) list
  | SReturn of expr
  | SBarrier  (** Block-level barrier (__syncthreads) *)
  | SWarpBarrier  (** Warp-level sync (__syncwarp) *)
  | SExpr of expr  (** Side-effecting expression *)
  | SEmpty
  | SLet of var * expr * stmt  (** Let binding: let v = e in body *)
  | SLetMut of var * expr * stmt  (** Mutable let: let v = ref e in body *)
  | SPragma of string list * stmt  (** Pragma hints wrapping a statement *)
  | SMemFence  (** Memory fence (threadfence) *)
  | SBlock of stmt
      (** Scoped block - creates a C scope for variable isolation *)
  | SNative of {
      gpu : framework:string -> string;  (** Generate GPU code for framework *)
      ocaml : ocaml_closure;  (** Typed OCaml fallback *)
    }  (** Inline native GPU code with OCaml fallback *)

(** Declarations *)
and decl =
  | DParam of
      var * array_info option (* kernel parameter, optional array info *)
  | DLocal of var * expr option (* local variable, optional init *)
  | DShared of
      string * elttype * expr option (* shared array: name, elem type, size *)

and array_info = {arr_elttype : elttype; arr_memspace : memspace}

(** Helper function (device function called from kernel) *)
and helper_func = {
  hf_name : string;
  hf_params : var list;
  hf_ret_type : elttype;
  hf_body : stmt;
}

(** Native argument type for kernel execution. Typed arguments without Obj.t -
    used by PPX-generated native functions. *)
and native_arg =
  | NA_Int32 of int32
  | NA_Int64 of int64
  | NA_Float32 of float
  | NA_Float64 of float
  | NA_Vec of native_vec

and native_vec =
  | NV : ('elt, 'underlying) native_vec_ops -> native_vec

and ('elt, 'underlying) native_vec_ops = {
      length : int;
      elem_size : int;
      type_name : string;
      type_id : 'elt Type_id.t;
      get_f32 : int -> float;
      set_f32 : int -> float -> unit;
      get_f64 : int -> float;
      set_f64 : int -> float -> unit;
      get_i32 : int -> int32;
      set_i32 : int -> int32 -> unit;
      get_i64 : int -> int64;
      set_i64 : int -> int64 -> unit;
      get_typed : int -> 'elt;
      set_typed : int -> 'elt -> unit;
      underlying_type_id : 'underlying Type_id.t;
      underlying : 'underlying;
    }

and ocaml_closure = {
  run :
    block:int * int * int -> grid:int * int * int -> native_arg array -> unit;
}

(** {2 Typed Helpers for Custom Types} *)

(** Get element from NA_Vec as a type checked custom value. *)
let vec_get_custom : type a. a Type_id.t -> native_arg -> int -> a =
 fun type_id arg i ->
  match arg with
  | NA_Vec (NV v) -> (
      match Type_id.equal type_id v.type_id with
      | Some Type_id.Refl -> v.get_typed i
      | None -> failwith "vec_get_custom: vector element type mismatch")
  | _ -> failwith "vec_get_custom: expected NA_Vec"

(** Set element in NA_Vec from a type checked custom value. *)
let vec_set_custom : type a. a Type_id.t -> native_arg -> int -> a -> unit =
 fun type_id arg i x ->
  match arg with
  | NA_Vec (NV v) -> (
      match Type_id.equal type_id v.type_id with
      | Some Type_id.Refl -> v.set_typed i x
      | None -> failwith "vec_set_custom: vector element type mismatch")
  | _ -> failwith "vec_set_custom: expected NA_Vec"

(** Get length from NA_Vec *)
let vec_length : native_arg -> int =
 fun arg ->
  match arg with
  | NA_Vec (NV v) -> v.length
  | _ -> failwith "vec_length: expected NA_Vec"

(** Get the checked underlying vector/buffer value. *)
let vec_as_vector : type a. a Type_id.t -> native_arg -> a =
 fun type_id arg ->
  match arg with
  | NA_Vec (NV v) -> (
      match Type_id.equal type_id v.underlying_type_id with
      | Some Type_id.Refl -> v.underlying
      | None -> failwith "vec_as_vector: underlying type mismatch")
  | _ -> failwith "vec_as_vector: expected NA_Vec"

(** Native function type for V2 execution. Uses typed native_arg. *)
type native_fn_t =
  | NativeFn of
      (parallel:bool ->
      block:int * int * int ->
      grid:int * int * int ->
      native_arg array ->
      unit)

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
  kern_native_fn : native_fn_t option;
      (** Optional pre-compiled native function for CPU execution *)
}
