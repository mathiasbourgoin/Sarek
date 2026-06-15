(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Sarek PPX - Negative Test for Reserved Keywords
 *
 * This test should FAIL to compile with a clear error message about
 * 'double' being a reserved C/CUDA/OpenCL keyword.
 ******************************************************************************)

open Spoc
open Sarek

let test_kernel =
  [%kernel
    let double (x : int32) : int32 = x + x in
    fun (src : int32 vector) (dst : int32 vector) ->
      let open Std in
      let idx = global_idx_x in
      dst.(idx) <- double src.(idx)]

let () =
  let _, kirc = test_kernel in
  Kirc.print_ast kirc.Kirc.body
