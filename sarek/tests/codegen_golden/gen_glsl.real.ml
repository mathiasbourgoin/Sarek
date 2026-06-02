(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Thin adapter for GLSL generator in golden tests — uses pure sarek_codegen *)

open Sarek_ir_types
open Sarek_codegen

let reset_state () = Sarek_ir_glsl.current_variants := []

let generate_with_types ~types (k : kernel) =
  Sarek_ir_glsl.generate_with_types ~types k
