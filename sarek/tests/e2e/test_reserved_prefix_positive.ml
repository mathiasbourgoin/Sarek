(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Positive boundary test for the reserved-prefix policy.
 *
 * The policy rejects user identifiers beginning with exactly [sarek_]. It must
 * NOT over-reject names that merely resemble the prefix. This kernel binds a
 * helper [sarekX_foo] (no underscore after [sarek]), a param [mysarek_] (prefix
 * not at the start), and a local [_sarek_foo] (leading underscore) - all legal.
 * Compiling this file to a kernel IR is the proof; the reserved-prefix policy
 * runs at elaboration, so a false positive would fail the build here.
 ******************************************************************************)

[@@@warning "-33"]

module Std = Sarek_stdlib.Std

let boundary_kernel =
  [%kernel
    let sarekX_foo (x : int32) : int32 = x + x in
    fun (a : int32 vector) (r : int32 vector) (mysarek_ : int32) ->
      let open Std in
      let idx = global_idx_x in
      let _sarek_foo = sarekX_foo a.(idx) in
      r.(idx) <- _sarek_foo + mysarek_]

let () =
  let _, kirc = boundary_kernel in
  (match kirc.Sarek.Kirc_types.body_ir with
  | Some _ -> ()
  | None -> failwith "boundary kernel produced no IR") ;
  print_endline
    "test_reserved_prefix_positive: PASSED (boundary names accepted)"
