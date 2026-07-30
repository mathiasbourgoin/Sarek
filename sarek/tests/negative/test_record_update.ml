(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Negative test: A `{ r with f = e }` record update must be REFUSED (backlog-192).

   `Pexp_record`'s second component is the `with` base. It was bound as `_base`
   and never used, so this kernel parsed to exactly the same `ERecord` as
   `{ x = 1.0 }` -- with `y` NOT mentioned. The lowering emits a record literal
   as a struct initialiser, so `y` was left uninitialised and read on the device
   as whatever the memory held. No diagnostic anywhere: the same shape as the
   `when` guard of backlog-191, and the reason this file exists. *)

type float32 = float

type point = {x : float32; y : float32} [@@sarek.type]

let k =
  [%kernel
    fun (src : float32 vector) (dst : float32 vector) ->
      let tid = thread_idx_x in
      let p = {x = src.(tid); y = src.(tid)} in
      let q = {p with x = 1.0} in
      dst.(tid) <- q.x]
