(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Negative test: A labelled argument in a function TYPE must be REFUSED (backlog-192).

   `Ptyp_arrow`'s label was read as `_`, so `step:int32 -> int32` and
   `int32 -> int32` parsed to the same `TEArrow` -- the label dropped from the
   type, while the corresponding parameter is refused outright. *)

let k =
  [%kernel
    fun (src : int32 vector) (dst : int32 vector) ->
      let tid = thread_idx_x in
      let f : step:int32 -> int32 = fun (x : int32) -> x in
      dst.(tid) <- f src.(tid)]
