(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Log_jsx — no-op logging shim for the jsoo Execute path.
 *
 * Execute uses Log.Execute category and Log.debugf. This module provides
 * the minimum API so that Execute compiles under jsoo without pulling in
 * the native Log module (which uses Sys.getenv and Printf.printf).
 ******************************************************************************)

(** Log components matching the native Log module — Execute uses Log.Execute *)
type component = Transfer | Kernel | Device | Memory | Execute | All

(** Execute category constant — matches native Log.Execute *)
let execute_category = Execute

(** Alias: Execute is the component Execute.ml references as Log.Execute *)
let _ = execute_category

(** No-op debug format function *)
let debugf _comp fmt = Printf.ifprintf stdout fmt
