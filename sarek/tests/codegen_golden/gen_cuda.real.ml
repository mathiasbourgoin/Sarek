(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Thin adapter for CUDA generator in golden tests — uses pure sarek_codegen *)

open Sarek_ir_types
open Sarek_codegen

let reset_state () =
  Sarek_ir_cuda.current_framework := None ;
  Sarek_ir_cuda.current_variants := []

let generate_with_types ~types (k : kernel) =
  Sarek_ir_cuda.generate_with_types ~types k
