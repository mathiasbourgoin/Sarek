(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Thin adapter for OpenCL generator in golden tests *)

open Sarek_ir_types
open Sarek_opencl

let reset_state () =
  Sarek_ir_opencl.current_framework := None ;
  Sarek_ir_opencl.current_variants := []

let generate_with_types ~types (k : kernel) =
  Sarek_ir_opencl.generate_with_types ~types k
