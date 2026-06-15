(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Kirc Types
 *
 * Type definitions for Sarek kernels. Separated from Kirc to allow the V2
 * path to use these types without depending on SPOC.
 *
 * NOTE: These types are legacy and only used by the PPX's intermediate
 * representation. Modern V2 kernels use Kirc_kernel.t directly.
 ******************************************************************************)

type float64 = float

type float32 = float

type extension = ExFloat32 | ExFloat64

(** Stub type replacing Spoc.Vector.kind - V2 path doesn't use this *)
type ('a, 'b) vector_kind_stub = unit

(** Stub type replacing Spoc.Kernel.spoc_kernel - V2 path doesn't use this *)
type ('a, 'b) spoc_kernel_stub = unit

type ('a, 'b, 'c) kirc_kernel = {
  ml_kern : 'a;
  body_ir : Sarek_ir_types.kernel option;
  ret_val : ('b, 'c) vector_kind_stub;
  extensions : extension array;
}

type ('a, 'b, 'c, 'd) kirc_function = {
  fun_name : string;
  ml_fun : 'a;
  fun_ret : ('b, 'c) vector_kind_stub;
  fastflow_acc : 'd;
  fun_extensions : extension array;
}

type ('a, 'b, 'c, 'd, 'e) sarek_kernel =
  ('a, 'b) spoc_kernel_stub * ('c, 'd, 'e) kirc_kernel

(** Constructor registry for variant types *)
val constructors : string list ref

val register_constructor_string : string -> unit
