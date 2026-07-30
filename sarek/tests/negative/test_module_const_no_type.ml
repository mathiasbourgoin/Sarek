(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Negative test: An unannotated constant in a kernel module must be REFUSED (backlog-192).

   The `let module` fold DROPPED it: `MConst` needs a type and there was none, so
   the binding was simply not registered and this kernel failed with an unbound
   variable pointing at `M.bump` -- the USE, not the declaration. The top-level
   payload fold already refused this; the two now agree. *)

let k =
  [%kernel
    let module M = struct
      let bump = 2l
    end in
    fun (src : int32 vector) (dst : int32 vector) ->
      let tid = thread_idx_x in
      dst.(tid) <- src.(tid) + M.bump]
