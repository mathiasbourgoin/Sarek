(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Unit tests for Execute - kernel execution dispatcher *)

open Sarek.Execute
open Spoc_core
open Spoc_framework
open Alcotest

type point = {x : float; y : float}

let point_type_id : point Sarek_ir_types.Type_id.t =
  Sarek_ir_types.Type_id.create ()

let point_vector_type_id : (point, unit) Vector.t Sarek_ir_types.Type_id.t =
  Sarek_ir_types.Type_id.create ()

let point_custom : point Vector.custom_type =
  let get ptr i =
    let off = i * 8 in
    {
      x = Vector.Custom_helpers.read_float32 ptr off;
      y = Vector.Custom_helpers.read_float32 ptr (off + 4);
    }
  in
  let set ptr i p =
    let off = i * 8 in
    Vector.Custom_helpers.write_float32 ptr off p.x ;
    Vector.Custom_helpers.write_float32 ptr (off + 4) p.y
  in
  {
    elem_size = 8;
    type_id = point_type_id;
    vector_type_id = point_vector_type_id;
    get;
    set;
    name = "point";
  }

(** {1 Tests for vector argument types} *)

let test_vec_arg_int () =
  let arg = Int 42 in
  ignore arg ;
  check bool "int arg created" true true

let test_vec_arg_int32 () =
  let arg = Int32 42l in
  ignore arg ;
  check bool "int32 arg created" true true

let test_vec_arg_float32 () =
  let arg = Float32 3.14 in
  ignore arg ;
  check bool "float32 arg created" true true

let test_vec_arg_float64 () =
  let arg = Float64 2.71828 in
  ignore arg ;
  check bool "float64 arg created" true true

(** {1 Tests for dimension helpers} *)

let test_dims1d () =
  let d = dims1d 1024 in
  check int "1d dims x" 1024 d.Framework_sig.x ;
  check int "1d dims y" 1 d.Framework_sig.y ;
  check int "1d dims z" 1 d.Framework_sig.z

let test_dims2d () =
  let d = dims2d 32 64 in
  check int "2d dims x" 32 d.Framework_sig.x ;
  check int "2d dims y" 64 d.Framework_sig.y ;
  check int "2d dims z" 1 d.Framework_sig.z

let test_dims3d () =
  let d = dims3d 16 32 8 in
  check int "3d dims x" 16 d.Framework_sig.x ;
  check int "3d dims y" 32 d.Framework_sig.y ;
  check int "3d dims z" 8 d.Framework_sig.z

(** {1 Tests for grid calculation} *)

let test_grid_for_size_exact () =
  (* Problem size 1024, block size 256 → grid size 4 *)
  let grid = grid_for_size ~problem_size:1024 ~block_size:256 in
  check int "grid exact division" 4 grid

let test_grid_for_size_remainder () =
  (* Problem size 1000, block size 256 → grid size 4 (rounds up) *)
  let grid = grid_for_size ~problem_size:1000 ~block_size:256 in
  check int "grid with remainder" 4 grid

let test_grid_for_size_small () =
  (* Problem size 100, block size 256 → grid size 1 *)
  let grid = grid_for_size ~problem_size:100 ~block_size:256 in
  check int "grid smaller than block" 1 grid

let test_grid_for_size_zero () =
  (* Problem size 0 → grid size 0 *)
  let grid = grid_for_size ~problem_size:0 ~block_size:256 in
  check int "grid zero size" 0 grid

(** {1 Tests for vector creation (integration)} *)

let test_create_int32_vector () =
  let v = Vector.create Vector.int32 10 in
  check int "vector length" 10 (Vector.length v) ;
  (* Set and get a value *)
  Vector.set v 5 42l ;
  check int32 "vector get/set" 42l (Vector.get v 5)

let test_create_float32_vector () =
  let v = Vector.create Vector.float32 8 in
  check int "vector length" 8 (Vector.length v) ;
  Vector.set v 3 3.14 ;
  check (float 0.001) "vector get/set" 3.14 (Vector.get v 3)

(** {1 Tests for vector wrapping} *)

let test_vec_wrapper () =
  let v = Vector.create Vector.int32 5 in
  let arg = Vec v in
  ignore arg ;
  check bool "vec wrapper created" true true

let test_multiple_args () =
  let v1 = Vector.create Vector.int32 10 in
  let v2 = Vector.create Vector.float32 20 in
  let args = [Vec v1; Int 42; Vec v2; Float32 3.14] in
  check int "arg list length" 4 (List.length args)

let test_custom_exec_vector_get_set () =
  let v = Vector.create_custom point_custom 2 in
  Vector.set v 0 {x = 1.5; y = 2.5} ;
  Vector.set v 1 {x = 0.0; y = 0.0} ;
  match vector_args_to_exec_array [Vec v] with
  | [|Framework_sig.EA_Vec (module V)|] -> (
      match V.get 0 with
      | Typed_value.TV_Composite (Typed_value.CV ((module C), p)) ->
          check string "custom type name" "point" C.name ;
          check int "custom byte size" 8 C.size ;
          check int "serialized length" 8 (Bytes.length (C.to_bytes p)) ;
          V.set 1 (Typed_value.TV_Composite (Typed_value.CV ((module C), p))) ;
          let got = Vector.get v 1 in
          check (float 0.001) "custom set x" 1.5 got.x ;
          check (float 0.001) "custom set y" 2.5 got.y
      | _ -> fail "expected composite typed value")
  | _ -> fail "expected single exec vector"

(** {1 Test suite} *)

let () =
  run
    "Execute"
    [
      ( "vector_arg_types",
        [
          test_case "int" `Quick test_vec_arg_int;
          test_case "int32" `Quick test_vec_arg_int32;
          test_case "float32" `Quick test_vec_arg_float32;
          test_case "float64" `Quick test_vec_arg_float64;
        ] );
      ( "dimension_helpers",
        [
          test_case "dims1d" `Quick test_dims1d;
          test_case "dims2d" `Quick test_dims2d;
          test_case "dims3d" `Quick test_dims3d;
        ] );
      ( "grid_calculation",
        [
          test_case "exact_division" `Quick test_grid_for_size_exact;
          test_case "with_remainder" `Quick test_grid_for_size_remainder;
          test_case "smaller_than_block" `Quick test_grid_for_size_small;
          test_case "zero_size" `Quick test_grid_for_size_zero;
        ] );
      ( "vector_operations",
        [
          test_case "create_int32" `Quick test_create_int32_vector;
          test_case "create_float32" `Quick test_create_float32_vector;
        ] );
      ( "vector_wrapping",
        [
          test_case "vec_wrapper" `Quick test_vec_wrapper;
          test_case "multiple_args" `Quick test_multiple_args;
          test_case
            "custom_exec_vector_get_set"
            `Quick
            test_custom_exec_vector_get_set;
        ] );
    ]
