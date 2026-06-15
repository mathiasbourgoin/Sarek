(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Test kernel that should FAIL type checking:
   - Uses Geometry_lib.point but accesses non-existent field 'z' *)
open Spoc
open Sarek_geometry

let () =
  (* This should fail because Geometry_lib.point has no field 'z' *)
  let bad_kernel =
    [%kernel
      fun (points : Geometry_lib.point vector)
          (distances : float32 vector)
          (n : int32) ->
        let tid = thread_idx_x + (block_dim_x * block_idx_x) in
        if tid < n then
          let p = points.(tid) in
          let z = p.z in
          (* ERROR: field 'z' does not exist! *)
          distances.(tid) <- z]
  in

  let _, kirc = bad_kernel in
  Sarek.Kirc.print_ast kirc.Sarek.Kirc.body ;
  print_endline "This should not print - test should have failed to compile"
