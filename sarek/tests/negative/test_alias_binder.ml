(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Negative test: An `as` alias in a binder must be REFUSED (backlog-192).

   `extract_name_from_pattern` answered the ALIAS name and threw the inner
   pattern away. For `(v as w)` that loses `v`; for `((a, b) as t)` it loses both
   `a` and `b`, and the kernel fails at the first use of one with an unbound
   variable pointing at the USE rather than at the alias. *)

let k =
  [%kernel
    fun (src : int32 vector) (dst : int32 vector) ->
      let tid = thread_idx_x in
      let (v as w) = src.(tid) in
      dst.(tid) <- w]
