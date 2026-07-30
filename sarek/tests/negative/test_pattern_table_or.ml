(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Negative test: An or-pattern in a kernel must be refused BY NAME (backlog-192).

   The pattern half of the same claim as test_expr_table_try: `parse_pattern`'s
   final arm said "Unsupported pattern" for every pattern form it did not build,
   and now consults `Sarek_unsupported.pattern_refusal`. *)

type float32 = float

type shape = Circle of float32 | Square of float32 [@@sarek.type]

let k =
  [%kernel
    fun (src : float32 vector) (dst : float32 vector) ->
      let tid = thread_idx_x in
      let s = Circle src.(tid) in
      let got = match s with Circle r | Square r -> r in
      dst.(tid) <- got]
