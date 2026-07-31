(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Negative test: A result annotation with fewer arrows than the parameters collected after it
   must be REFUSED (backlog-192).

   `fun_return_type` keeps the last `Pexp_function` constraint on the descent and
   peels one arrow per parameter collected AFTER it, because a constraint sitting
   above further parameters describes a function type rather than the flattened
   result. Here `: int32` sits above one more parameter and has no arrow to peel,
   so which part of it is the result type is not answerable — and answering it
   anyway is how an annotation gets half-applied. *)

let k =
  [%kernel
    fun (src : int32 vector) (dst : int32 vector) ->
      let tid = thread_idx_x in
      let f (x : int32) : int32 = fun (y : int32) -> x + y in
      dst.(tid) <- f src.(tid) 1l]
