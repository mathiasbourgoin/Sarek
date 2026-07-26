(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Thin adapter for Metal generator in golden tests — uses pure sarek_codegen
*)

open Sarek_ir_types
open Sarek_codegen

let reset_state () =
  Sarek_ir_metal.current_framework := None ;
  Sarek_ir_metal.current_variants := []

let generate_with_types ~types (k : kernel) =
  Sarek_ir_metal.generate_with_types ~types k

(* Exposed so the contraction-pragma gate can check BOTH preamble sites.
   [Sarek_ir_metal] emits its header in two places - here and in
   [generate_with_types] - and a gate that only exercised the latter would let
   this one drift back to contracting silently. *)
let generate (k : kernel) = Sarek_ir_metal.generate k
