(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Negative test: A coercion in a function's return position must be REFUSED (backlog-192).

   `Pexp_function`'s type_constraint slot was read as `_`, so `Pcoerce` was
   dropped exactly like `Pconstraint` was. *)

let k =
  [%kernel
    fun (src : int32 vector) (dst : int32 vector) ->
      let tid = thread_idx_x in
      let f (x : int32) :> int32 = x in
      dst.(tid) <- f src.(tid)]
