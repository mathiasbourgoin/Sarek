(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Negative test: An existential binder in a constructor pattern must be REFUSED
   (backlog-192).

   `Ppat_construct`'s payload is `(string loc list * pattern) option` and the
   binder list was read as `_`, so `Circle (type a) r` parsed as the plain
   `Circle r`. *)

type float32 = float

type shape = Circle of float32 | Square of float32 [@@sarek.type]

let k =
  [%kernel
    fun (src : float32 vector) (dst : float32 vector) ->
      let tid = thread_idx_x in
      let s = Circle src.(tid) in
      let got = match s with Circle (type a) r -> r | Square q -> q in
      dst.(tid) <- got]
