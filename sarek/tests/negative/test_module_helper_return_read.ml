(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Negative test: A payload-level helper's declared result type must be READ (backlog-192).

   The third annotation site, and the one with no slot to put the type in:
   `Sarek_ast.MFun` carries no type. It is applied as an `ETyped` constraint on
   the body instead, which `Sarek_typer` unifies and then discards, so nothing new
   reaches the IR. Before the fix this file compiled at exit 0. *)

let k =
  [%kernel
    let widen (x : int32) : float32 = x in
    fun (src : int32 vector) (dst : int32 vector) ->
      let tid = thread_idx_x in
      dst.(tid) <- widen src.(tid)]
