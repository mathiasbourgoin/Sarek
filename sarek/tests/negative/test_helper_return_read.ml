(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Negative test: A kernel-local helper's declared result type must be READ (backlog-192).

   Same positive-half shape as test_local_annotation_read: the annotation
   contradicts the body. `let f x : t = e` puts `t` in `Pexp_function`'s
   type_constraint slot, which `collect_fun_params` read as `_`, so the declared
   result type of every kernel helper was discarded. Before the fix this file
   compiled at exit 0. *)

let k =
  [%kernel
    fun (src : int32 vector) (dst : int32 vector) ->
      let tid = thread_idx_x in
      let widen (x : int32) : float32 = x in
      dst.(tid) <- widen src.(tid)]
