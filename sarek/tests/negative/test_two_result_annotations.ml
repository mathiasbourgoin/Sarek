(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Negative test: Two result annotations on one binding must be REFUSED (backlog-192, found by
   the cross-runtime review).

   `fun_return_type` walks every `Pexp_function` on the descent because
   `collect_fun_params` flattens them into one parameter list. When more than one
   of them carries a constraint, only one can reach the single result slot — and
   the first version of this branch's fix took the inner one silently, discarding
   the relationship the outer one states between the parameters and the result. *)

let k =
  [%kernel
    fun (src : int32 vector) (dst : int32 vector) ->
      let tid = thread_idx_x in
      let f (x : int32) : int32 -> int32 = fun (y : int32) : int32 -> x + y in
      dst.(tid) <- f src.(tid) 1l]
