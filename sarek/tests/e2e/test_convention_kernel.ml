(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Test kernel using external Geometry_lib types via convention *)
open Sarek_geometry

(* This kernel takes Geometry_lib.point vectors and computes distance to origin *)
let () =
  (* Version that should type correctly:
     - Uses Geometry_lib.point from external library
     - Accesses fields x, y
     - Computes sqrt(x^2 + y^2) *)
  let distance_to_origin_kernel =
    [%kernel
      fun (points : Geometry_lib.point vector)
          (distances : float32 vector)
          (n : int32) ->
        let tid = thread_idx_x + (block_dim_x * block_idx_x) in
        if tid < n then
          let p = points.(tid) in
          let x = p.x in
          let y = p.y in
          distances.(tid) <- sqrt ((x *. x) +. (y *. y))]
  in

  let _native, _kirc = distance_to_origin_kernel in
  print_endline "=== Distance to origin kernel IR ===" ;
  print_endline "=====================================" ;
  print_endline "Convention kernel test PASSED"
