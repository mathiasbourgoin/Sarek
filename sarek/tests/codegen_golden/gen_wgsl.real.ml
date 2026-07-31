(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Thin adapter for WGSL generator in golden tests — uses pure sarek_codegen *)

open Sarek_ir_types
open Sarek_codegen

(* Nothing to reset: backlog-185 replaced this backend's module-level refs with
   a per-generation state value threaded through the emit functions, so one
   generation can no longer contaminate the next. Kept as [unit -> unit] so the
   shared golden harness compiles unchanged. *)
let reset_state () = ()

let generate_with_types ~types (k : kernel) =
  Sarek_ir_wgsl.generate_with_types ~types k
