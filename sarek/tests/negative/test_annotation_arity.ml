(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Negative test: A whole-binding annotation with fewer arrows than the function has
   parameters must be REFUSED (backlog-192).

   `ELetRec`'s type slot is the RESULT type, so a whole-binding annotation has
   one arrow peeled per parameter before it goes in. `int32` has none to peel for
   a one-parameter function. The old code put the annotation in UNPEELED, so the
   one spelling that reached the slot (`let (f : a -> b) = fun x -> ...`)
   unified an arrow against the body's type. *)

let k =
  [%kernel
    fun (src : int32 vector) (dst : int32 vector) ->
      let tid = thread_idx_x in
      let f : int32 = fun (x : int32) -> x + 1l in
      dst.(tid) <- f src.(tid)]
