(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Negative test: A GADT-style constructor declaration must be REFUSED (backlog-192).

   `pcd_res` and `pcd_vars` had no reader, so `Circle : int32 -> shape` was
   recorded exactly as if it had been written `Circle of int32`. For a
   parameterised type that is a different declaration, and the device
   representation is a tag plus one payload whose type comes from the
   declaration. *)

let k =
  [%kernel
    let module M = struct
      type shape = Circle : int32 -> shape
    end in
    fun (src : int32 vector) (dst : int32 vector) ->
      let tid = thread_idx_x in
      dst.(tid) <- src.(tid)]
