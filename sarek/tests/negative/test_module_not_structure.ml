(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Negative test: A payload `let module M = <not a struct>` must be REFUSED (backlog-192).

   Only `Pmod_structure` was read; every other module expression contributed
   `([], [])`, so `let module M = N in` brought in NOTHING and said nothing. *)

let k =
  [%kernel
    let module M = Sarek_stdlib in
    fun (src : int32 vector) (dst : int32 vector) ->
      let tid = thread_idx_x in
      dst.(tid) <- src.(tid)]
