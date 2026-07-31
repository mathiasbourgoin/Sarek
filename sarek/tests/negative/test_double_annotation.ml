(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Negative test: A binding annotated in BOTH places must be REFUSED (backlog-192, found by the
   cross-runtime review).

   `let (x : t1) : t2 = e` is legal OCaml and puts an annotation in the pattern
   AND in `pvb_constraint` (checked with `ocamlc -stop-after parsing`).
   `binding_type` reads one per binding, so the other was discarded — without
   being checked against it, which is what makes it a drop rather than a
   preference. *)

let k =
  [%kernel
    fun (src : int32 vector) (dst : int32 vector) ->
      let tid = thread_idx_x in
      let (x : int32) : int32 = src.(tid) in
      dst.(tid) <- x]
