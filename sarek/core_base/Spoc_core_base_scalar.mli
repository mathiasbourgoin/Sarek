(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Spoc_core_base_scalar — pure, Ops-independent scalar utilities (interface)
 *
 * Extracted from Spoc_core_base.Make. Re-exported with constructors inside
 * Make so that Make(Ops).Float32 etc. still resolve.
 ******************************************************************************)

(** {1 Scalar element kinds} *)

(** GADT mapping OCaml element types to Bigarray element kinds. *)
type (_, _) scalar_kind =
  | Float32 : (float, Bigarray.float32_elt) scalar_kind
  | Float64 : (float, Bigarray.float64_elt) scalar_kind
  | Int32 : (int32, Bigarray.int32_elt) scalar_kind
  | Int64 : (int64, Bigarray.int64_elt) scalar_kind
  | Char : (char, Bigarray.int8_unsigned_elt) scalar_kind
  | Complex32 : (Complex.t, Bigarray.complex32_elt) scalar_kind

(** {1 Kind helpers — pure} *)

(** Map a scalar_kind to the corresponding Bigarray kind. *)
val to_bigarray_kind : ('a, 'b) scalar_kind -> ('a, 'b) Bigarray.kind

(** Byte size of a Bigarray element kind. *)
val bigarray_elem_size : ('a, 'b) Bigarray.kind -> int

(** Byte size of a scalar_kind element. *)
val scalar_elem_size : ('a, 'b) scalar_kind -> int

(** Human-readable name for a scalar_kind. *)
val scalar_kind_name : ('a, 'b) scalar_kind -> string

(** {1 Type-id singletons for scalar element types} *)

val float32_type_id : float Sarek_ir_types.Type_id.t

val float64_type_id : float Sarek_ir_types.Type_id.t

val int32_type_id : int32 Sarek_ir_types.Type_id.t

val int64_type_id : int64 Sarek_ir_types.Type_id.t

val char_type_id : char Sarek_ir_types.Type_id.t

val complex32_type_id : Complex.t Sarek_ir_types.Type_id.t

(** Map a scalar_kind to its element Type_id singleton. *)
val scalar_type_id : ('a, 'b) scalar_kind -> 'a Sarek_ir_types.Type_id.t
