(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Negative test: A type alias in a kernel payload must be REFUSED (backlog-192).

   `ptype_manifest` had no reader, and `Ptype_abstract` fell into a catch-all
   that said "Unsupported type declaration in kernel payload" without saying
   which part was unsupported or what to do. *)

let k =
  [%kernel
    let module M = struct
      type idx = int32
    end in
    fun (src : int32 vector) (dst : int32 vector) ->
      let tid = thread_idx_x in
      dst.(tid) <- src.(tid)]
