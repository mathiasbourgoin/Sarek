(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Negative test: A wildcard type annotation must be REFUSED (backlog-192).

   `parse_type`'s last arm answered `TEConstr ("unknown", [])` for every
   core_type shape it did not list, and `Sarek_types.type_of_type_expr` maps an
   unrecognised constructor to `TRecord (name, [])` -- an EMPTY RECORD type. So
   `_` became a phantom type named "unknown" rather than an error. *)

let k =
  [%kernel
    fun (src : int32 vector) (dst : int32 vector) ->
      let tid = thread_idx_x in
      let x : _ = src.(tid) in
      dst.(tid) <- x]
