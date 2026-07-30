(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Negative test: A parameterised type in a kernel payload must be REFUSED (backlog-192).

   `ptype_params` had no reader. A payload type declaration never reaches OCaml,
   so a field nobody reads here is a field nobody reads at all: this declared a
   `box` whose device layout depends on `'a`, and Sarek recorded it as if the
   parameter were not there. *)

let k =
  [%kernel
    let module M = struct
      type 'a box = {v : 'a}
    end in
    fun (src : int32 vector) (dst : int32 vector) ->
      let tid = thread_idx_x in
      dst.(tid) <- src.(tid)]
