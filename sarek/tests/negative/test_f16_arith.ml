(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Negative test (#57 slice 1 review, MF4b): f16 is a storage-only type, so
   arithmetic on an f16 element must be a COMPILE error with a comprehensible
   located message — not "expected int32, got float16", and not silently
   emitting `+` on `__half`.

   Expected error:
     "float16 is a storage-only type and has no arithmetic: '+' / '+.'" *)

let k =
  [%kernel
    fun (out : float16 vector) (a : float16 vector) (b : float16 vector) ->
      let open Sarek_stdlib.Std in
      let tid = global_thread_id in
      out.(tid) <- a.(tid) +. b.(tid)]
