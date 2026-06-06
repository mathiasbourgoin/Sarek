(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Spoc_core_base_scalar — pure, Ops-independent scalar utilities
 *
 * Contains the scalar_kind GADT, element-size helpers, kind-name helpers,
 * and scalar Type_id singletons. None of these reference CUSTOM_OPS.
 * Extracted from Spoc_core_base.Make to keep Spoc_core_base.ml < 500 lines.
 ******************************************************************************)

(** {1 Scalar element kinds} *)

type (_, _) scalar_kind =
  | Float32 : (float, Bigarray.float32_elt) scalar_kind
  | Float64 : (float, Bigarray.float64_elt) scalar_kind
  | Int32 : (int32, Bigarray.int32_elt) scalar_kind
  | Int64 : (int64, Bigarray.int64_elt) scalar_kind
  | Char : (char, Bigarray.int8_unsigned_elt) scalar_kind
  | Complex32 : (Complex.t, Bigarray.complex32_elt) scalar_kind

(** {1 Kind helpers — pure} *)

let to_bigarray_kind : type a b. (a, b) scalar_kind -> (a, b) Bigarray.kind =
  function
  | Float32 -> Bigarray.Float32
  | Float64 -> Bigarray.Float64
  | Int32 -> Bigarray.Int32
  | Int64 -> Bigarray.Int64
  | Char -> Bigarray.Char
  | Complex32 -> Bigarray.Complex32

let bigarray_elem_size : type a b. (a, b) Bigarray.kind -> int = function
  | Bigarray.Float16 -> 2
  | Bigarray.Float32 -> 4
  | Bigarray.Float64 -> 8
  | Bigarray.Int8_signed -> 1
  | Bigarray.Int8_unsigned -> 1
  | Bigarray.Int16_signed -> 2
  | Bigarray.Int16_unsigned -> 2
  | Bigarray.Int32 -> 4
  | Bigarray.Int64 -> 8
  (* word-sized: 8 on 64-bit, 4 on 32-bit — matches Ctypes sizeof per platform *)
  | Bigarray.Int -> Sys.word_size / 8
  | Bigarray.Nativeint -> Sys.word_size / 8
  | Bigarray.Complex32 -> 8
  | Bigarray.Complex64 -> 16
  | Bigarray.Char -> 1

let scalar_elem_size : type a b. (a, b) scalar_kind -> int = function
  | Float32 -> 4
  | Float64 -> 8
  | Int32 -> 4
  | Int64 -> 8
  | Char -> 1
  | Complex32 -> 8

let scalar_kind_name : type a b. (a, b) scalar_kind -> string = function
  | Float32 -> "Float32"
  | Float64 -> "Float64"
  | Int32 -> "Int32"
  | Int64 -> "Int64"
  | Char -> "Char"
  | Complex32 -> "Complex32"

(** {1 Type-id singletons for scalar element types} *)

let float32_type_id : float Sarek_ir_types.Type_id.t =
  Sarek_ir_types.Type_id.create ()

let float64_type_id : float Sarek_ir_types.Type_id.t =
  Sarek_ir_types.Type_id.create ()

let int32_type_id : int32 Sarek_ir_types.Type_id.t =
  Sarek_ir_types.Type_id.create ()

let int64_type_id : int64 Sarek_ir_types.Type_id.t =
  Sarek_ir_types.Type_id.create ()

let char_type_id : char Sarek_ir_types.Type_id.t =
  Sarek_ir_types.Type_id.create ()

let complex32_type_id : Complex.t Sarek_ir_types.Type_id.t =
  Sarek_ir_types.Type_id.create ()

let scalar_type_id : type a b.
    (a, b) scalar_kind -> a Sarek_ir_types.Type_id.t = function
  | Float32 -> float32_type_id
  | Float64 -> float64_type_id
  | Int32 -> int32_type_id
  | Int64 -> int64_type_id
  | Char -> char_type_id
  | Complex32 -> complex32_type_id
