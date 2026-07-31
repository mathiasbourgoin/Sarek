(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Thin adapter for GLSL generator in golden tests — uses pure sarek_codegen *)

open Sarek_ir_types
open Sarek_codegen

(* Nothing to reset: since backlog-185 the GLSL emitter holds no module-level
   state — {!Sarek_ir_glsl.generate_with_types} builds a per-generation
   [Sarek_ir_glsl.state] value (including a fresh vec-param table) from the
   kernel, so no generation can observe a previous one's values. Kept as a
   [unit -> unit] so [test_codegen_golden.ml] compiles unchanged. *)
let reset_state () = ()

let generate_with_types ~types (k : kernel) =
  Sarek_ir_glsl.generate_with_types ~types k
