(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Negative test: A whole-binding type annotation on a FUNCTION must be REFUSED (backlog-192,
   found by the cross-runtime review of this branch's own first fix).

   This file used to be test_annotation_arity, asserting a narrower refusal. The
   first version of `binding_result_type` PEELED one arrow per parameter off a
   whole-binding annotation and used the result — which silently discarded every
   DOMAIN the user had written: `let (f : float32 -> int32) = fun (x : int32) ->
   x` was accepted as an `int32 -> int32` helper with the declared `float32` read
   by nobody. The sweep's own first draft therefore introduced an instance of the
   defect class the sweep exists to close.

   Refusing costs nothing: a kernel function's parameters must already carry
   their own annotations (`extract_param_from_pattern` raises "Kernel parameters
   must have type annotations"), so the domain half of a whole-binding arrow is
   always redundant with them and was never checked against them. *)

let k =
  [%kernel
    fun (src : int32 vector) (dst : int32 vector) ->
      let tid = thread_idx_x in
      let f : int32 = fun (x : int32) -> x + 1l in
      dst.(tid) <- f src.(tid)]
