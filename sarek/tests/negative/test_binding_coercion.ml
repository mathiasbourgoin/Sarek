(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Negative test: A coercion on a `let` binding must be REFUSED (backlog-192).

   `pvb_constraint`'s `Pvc_coercion` arm had no reader, so the coercion was
   dropped along with the type it mentions. *)

let k =
  [%kernel
    fun (src : int32 vector) (dst : int32 vector) ->
      let tid = thread_idx_x in
      let x :> int32 = src.(tid) in
      dst.(tid) <- x]
