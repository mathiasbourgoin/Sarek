(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Negative test: `type nonrec` in a kernel payload must be REFUSED (backlog-192, found by the
   cross-runtime review).

   `Pstr_type`'s rec_flag had no reader, and an earlier revision of this branch
   excused that by claiming a kernel type cannot refer to another kernel type in
   its fields. That is false — nested records are supported and exercised by
   sarek/tests/e2e/test_nested_types.ml — so `nonrec` is meaningful here and was
   being ignored: Sarek resolves a field's type by NAME against the kernel's own
   types, which under `nonrec` is the wrong binding. *)

let k =
  [%kernel
    let module M = struct
      type nonrec box = {v : int32}
    end in
    fun (src : int32 vector) (dst : int32 vector) ->
      dst.(thread_idx_x) <- src.(thread_idx_x)]
