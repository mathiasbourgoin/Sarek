(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Negative test: a NESTED `fun`'s declared result type must be READ
   (backlog-192, found by review of this branch's own first fix).

   Same positive-half shape as test_helper_return_read: the annotation
   contradicts the body, so the case compiles at exit 0 if the annotation is
   dropped and fails if it is read.

   The first version of this branch's `fun_return_type` read ONLY the outermost
   `Pexp_function`'s constraint slot, and said in a comment that a nested one
   "belongs to that inner function and not to the binding". That is false:
   `collect_fun_params` DESCENDS through `Pfunction_body` and merges the inner
   function's parameters into the binding's list, so after flattening there is no
   inner function left for the annotation to belong to. Measured at the time: the
   nested spelling below compiled at exit 0 while the flattened spelling of the
   same function (`let widen (x : int32) (y : int32) : float32 = x + y`) failed
   to unify -- same two-parameter helper, opposite verdict, the only difference
   being where the `: float32` was written.

   Every `Pexp_function` on the descent is now inspected, the last constraint
   wins, and it is peeled by the number of parameters collected after it. *)

let k =
  [%kernel
    fun (src : int32 vector) (dst : int32 vector) ->
      let tid = thread_idx_x in
      let widen (x : int32) = fun (y : int32) : float32 -> x + y in
      dst.(tid) <- widen src.(tid) 1l]
