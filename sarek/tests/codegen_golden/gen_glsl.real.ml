(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Thin adapter for GLSL/Vulkan generator in golden tests *)

open Sarek_ir_types
open Sarek_vulkan

let reset_state () = Sarek_ir_glsl.current_variants := []

let generate_with_types ~types (k : kernel) =
  Sarek_ir_glsl.generate_with_types ~types k
