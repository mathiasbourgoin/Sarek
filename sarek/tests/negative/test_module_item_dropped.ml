(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Negative test: A structure item a kernel module cannot hold must be REFUSED (backlog-192).

   The fold's last arm returned its accumulator UNCHANGED for everything that was
   not `Pstr_type` or `Pstr_value`, so an item written here was silently absent.
   `exception` is the representative: there are no exceptions on a device, so
   there was never any chance of honouring it. *)

let k =
  [%kernel
    let module M = struct
      exception Nope

      let bump : int32 = 2l
    end in
    fun (src : int32 vector) (dst : int32 vector) ->
      let tid = thread_idx_x in
      dst.(tid) <- src.(tid) + M.bump]
