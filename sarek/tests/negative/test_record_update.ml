(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Negative test: A `{ r with f = e }` record update must be REFUSED (backlog-192).

   `Pexp_record`'s second component is the `with` base. It was bound as `_base`
   and never used, so this kernel parsed to exactly the same `ERecord` as
   `{ x = 1.0 }` -- with `y` NOT mentioned.

   This case's red was PROVEN by removing the refusal and rebuilding, and what
   came back was not silence: OCaml's own

     Error: Some record fields are undefined: y

   on the `{ p with x = 1.0 }` line, because the PPX re-emits the record literal
   into the generated native fallback carrying the original location. So the
   pre-fix behaviour of a dropped `with` base is a diagnostic that names a
   missing field and never says the `with` was discarded -- NOT the silently
   wrong device code that backlog-191's `when` guard produced. An earlier
   revision of this header, of `record_update_msg` and of the commit that added
   them claimed the stronger thing; it was measured and it is false. *)

type float32 = float

type point = {x : float32; y : float32} [@@sarek.type]

let k =
  [%kernel
    fun (src : float32 vector) (dst : float32 vector) ->
      let tid = thread_idx_x in
      let p = {x = src.(tid); y = src.(tid)} in
      let q = {p with x = 1.0} in
      dst.(tid) <- q.x]
