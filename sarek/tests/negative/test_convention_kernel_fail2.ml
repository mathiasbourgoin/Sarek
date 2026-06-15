(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Test kernel that should FAIL type checking:
   - Computes float32 distance but tries to write to int32 vector *)
open Spoc
open Sarek_geometry

let () =
  (* This should fail because we're writing float32 to int32 vector *)
  let bad_kernel =
    [%kernel
      fun (points : Geometry_lib.point vector)
          (distances : int32 vector) (* int32 instead of float32! *)
          (n : int32) ->
        let tid = thread_idx_x + (block_dim_x * block_idx_x) in
        if tid < n then
          let p = points.(tid) in
          let x = p.x in
          let y = p.y in
          distances.(tid) <- sqrt ((x *. x) +. (y *. y))
      (* float32 result to int32 vector *)]
  in

  let _, kirc = bad_kernel in
  Sarek.Kirc.print_ast kirc.Sarek.Kirc.body ;
  print_endline "This should not print - test should have failed to compile"
