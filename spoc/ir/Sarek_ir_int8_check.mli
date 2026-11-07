(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Where the 8-bit storage types may appear, and the refusals for everywhere
    else. Re-exported by {!Sarek_ir_int8}, which documents the semantics; split
    out so that each file stays short. *)

open Sarek_ir_types

val is_8bit : elttype -> bool

val byte_elt_of_type : elttype -> elttype option

val byte_arrays : kernel -> (string * elttype) list

val refusal : kernel -> string option
