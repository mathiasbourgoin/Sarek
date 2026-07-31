(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Thin adapter for OpenCL generator in golden tests — uses pure sarek_codegen
*)

open Sarek_ir_types
open Sarek_codegen

(* backlog-185 scaffolding: the OpenCL emitter's per-generation state is now a
   value threaded through it, so there is nothing left to reset between
   generations. Kept as a no-op because the shared golden harness calls it for
   every backend. *)
let reset_state () = ()

let generate_with_types ~types (k : kernel) =
  Sarek_ir_opencl.generate_with_types ~types k
