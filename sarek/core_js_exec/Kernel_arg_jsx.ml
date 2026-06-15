(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Kernel_arg_jsx — minimal jsoo shim for Kernel_arg.
 *
 * Provides the GADT matching Sarek_ir_interp's usage of Spoc_core.Kernel_arg.
 ******************************************************************************)

(** Kernel argument GADT — mirrors native Kernel_arg.t *)
type t =
  | Vec : ('a, 'b) Vector_jsx.t -> t
  | Int : int -> t
  | Int32 : int32 -> t
  | Int64 : int64 -> t
  | Float32 : float -> t
  | Float64 : float -> t
