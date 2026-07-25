(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Negative test (#57 slice 1 review, MF4a): ordered comparison on f16. This one
   DID reach [check_numeric] before the fix, but reported "expected int32, got
   float16", which names a type the user never asked for. It must now carry the
   f16-specific remedy.

   Expected error:
     "float16 is a storage-only type and has no arithmetic: '<'" *)

let k =
  [%kernel
    fun (out : int32 vector) (a : float16 vector) (b : float16 vector) ->
      let open Sarek_stdlib.Std in
      let tid = global_thread_id in
      if a.(tid) < b.(tid) then out.(tid) <- 1 else out.(tid) <- 0]
