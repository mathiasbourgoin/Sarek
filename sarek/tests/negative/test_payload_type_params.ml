(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Negative test: A parameterised type in a kernel payload must be REFUSED (backlog-192).

   `ptype_params` had no reader, so Sarek recorded this as if the parameter were
   not there. Measured by removing the refusal and rebuilding: the pre-fix
   behaviour for THIS shape (a field mentioning the parameter) is OCaml's

     Error: A type wildcard _ is not allowed in this type declaration.

   pinned to the whole `[%kernel]` payload -- an error that names nothing the
   user wrote, arising because the parameter-less re-emission turns `'a` into
   `_`. So it was a bad diagnostic in the wrong place, not silence. Whether a
   parameterised type NONE of whose fields mention the parameter was silent was
   not measured, and nothing here claims it either way. *)

let k =
  [%kernel
    let module M = struct
      type 'a box = {v : 'a}
    end in
    fun (src : int32 vector) (dst : int32 vector) ->
      let tid = thread_idx_x in
      dst.(tid) <- src.(tid)]
