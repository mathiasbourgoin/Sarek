(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Negative test: A functor application in a type path must be REFUSED (backlog-192).

   `parse_type`'s `flatten` returned `[]` for `Lapply`, so the type came out
   named "" -- and an unresolvable name becomes an empty placeholder record, not
   an error. *)

let k =
  [%kernel
    fun (src : int32 vector) (dst : int32 vector) ->
      let tid = thread_idx_x in
      let x : Fctor(Arg).t = src.(tid) in
      dst.(tid) <- x]
