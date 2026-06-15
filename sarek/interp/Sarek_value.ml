(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Sarek_value - Runtime value representation for interpreter
 *
 * Separated from Sarek_ir_interp to allow Sarek_type_helpers to depend on it
 * without creating a circular dependency.
 ******************************************************************************)

(** Type-safe runtime value representation for the interpreter. *)
type value =
  | VInt32 of int32
  | VInt64 of int64
  | VFloat32 of float
  | VFloat64 of float
  | VBool of bool
  | VUnit
  | VArray of value array
  | VRecord of string * value array  (** type_name, fields *)
  | VVariant of string * int * value list  (** type, tag, args *)

(** Get human-readable type name for a value *)
let value_type_name = function
  | VInt32 _ -> "int32"
  | VInt64 _ -> "int64"
  | VFloat32 _ -> "float32"
  | VFloat64 _ -> "float64"
  | VBool _ -> "bool"
  | VUnit -> "unit"
  | VArray _ -> "array"
  | VRecord (name, _) -> name
  | VVariant (name, _, _) -> name
