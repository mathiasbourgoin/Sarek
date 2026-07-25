(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Negative test (#57 slice 1 review, MF4a): the Eq/Ne arm of [infer_binop]
   skipped every numeric check, so `a.(i) = b.(i)` on a float16 vector COMPILED
   and emitted `a[tid] == b[tid]` on `__half`. Equality must be rejected too.

   Expected error:
     "float16 is a storage-only type and has no arithmetic: '='" *)

let k =
  [%kernel
    fun (out : int32 vector) (a : float16 vector) (b : float16 vector) ->
      let open Sarek_stdlib.Std in
      let tid = global_thread_id in
      if a.(tid) = b.(tid) then out.(tid) <- 1 else out.(tid) <- 0]
