(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Test that field accessors are generated and accessible from external module *)
open Sarek_geometry

let () =
  (* Test that accessors have correct types *)
  let p : Geometry_lib.point = {x = 1.0; y = 2.0} in
  let x : float = Geometry_lib.sarek_get_point_x p in
  let y : float = Geometry_lib.sarek_get_point_y p in
  Printf.printf "Convention test:\n" ;
  Printf.printf "  point.x = %f, point.y = %f\n" x y ;
  print_endline "Convention test PASSED"
