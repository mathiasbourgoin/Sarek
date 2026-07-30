(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Negative test: A type annotation on a tuple-pattern `let` must be REFUSED (backlog-192).

   A tuple-pattern binding is desugared to a single-arm `EMatch`, which carries
   no type. The annotation therefore had nowhere to go; now that
   `binding_result_type` reads both annotation spellings, silently discarding it
   here would be the same defect this sweep is about. *)

let k =
  [%kernel
    fun (src : int32 vector) (dst : int32 vector) ->
      let tid = thread_idx_x in
      let (a, b) : int32 * int32 = (src.(tid), 1l) in
      dst.(tid) <- a + b]
