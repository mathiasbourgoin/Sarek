(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Negative test: A labelled argument at a call site must be REFUSED (backlog-192).

   `Pexp_apply` carries `(arg_label * expression) list` and the label was read
   as `_`. So `sub2 ~a:x ~b:1l` lowered to the positional `sub2 x 1l` -- which
   happens to be right here, and would be WRONG if the labels were written in
   the other order, silently computing `1l - x`. A labelled PARAMETER is already
   refused by `collect_fun_params`, so no kernel function can legitimately be
   called this way. *)

let k =
  [%kernel
    let sub2 (a : int32) (b : int32) : int32 = a - b in
    fun (src : int32 vector) (dst : int32 vector) ->
      let tid = thread_idx_x in
      dst.(tid) <- sub2 ~a:src.(tid) ~b:1l]
