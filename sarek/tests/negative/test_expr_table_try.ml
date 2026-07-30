(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Negative test: `try ... with` in a kernel must be refused BY NAME (backlog-192).

   One of two representatives for `Sarek_unsupported`'s tables, which replaced a
   flat "Unsupported expression" covering some twenty distinct expression forms.
   The tables have no wildcard arm, so a ppxlib constructor added later stops the
   build rather than reaching a user as silence -- that property is held by the
   compiler, not by this test. What this test holds is that the table is WIRED to
   the parser's final arm. *)

let k =
  [%kernel
    fun (src : int32 vector) (dst : int32 vector) ->
      let tid = thread_idx_x in
      dst.(tid) <- (try src.(tid) with _ -> 0l)]
