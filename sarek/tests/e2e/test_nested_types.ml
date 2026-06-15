(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * E2E test for composable nested custom types with [@@sarek.type]
 *
 * Tests that nested types work correctly:
 *   type point = { x: float32; y: float32 } [@@sarek.type]
 *   type colored_point = { color: int32; point: point } [@@sarek.type]
 *
 * The PPX generates composable _custom accessors that can read/write
 * nested records by delegating to the inner type's accessor.
 ******************************************************************************)

module Vector = Spoc_core.Vector
module Device = Spoc_core.Device
module Transfer = Spoc_core.Transfer

[@@@warning "-32"]

(* Force backend registration *)
let () =
  Sarek_cuda.Cuda_plugin.init () ;
  Sarek_opencl.Opencl_plugin.init ()

type float32 = float

(* Base type: 2D point *)
type point = {x : float32; y : float32} [@@sarek.type]

(* Enum variant for color *)
type color = Red | Green | Blue [@@sarek.type]

(* Nested type: point with a color enum *)
type colored_point = {color : color; pt : point} [@@sarek.type]

(* Test 1: Basic point type (sanity check) *)
let test_point () =
  print_endline "=== Test: Basic point type ===" ;
  let n = 4 in
  let v = Vector.create_custom point_custom n in
  for i = 0 to n - 1 do
    Vector.set v i {x = float_of_int i; y = float_of_int (i * 2)}
  done ;
  let ok = ref true in
  for i = 0 to n - 1 do
    let p = Vector.get v i in
    let expected_x = float_of_int i in
    let expected_y = float_of_int (i * 2) in
    if
      abs_float (p.x -. expected_x) > 1e-6
      || abs_float (p.y -. expected_y) > 1e-6
    then (
      Printf.printf
        "  FAIL at %d: got {%.1f, %.1f} expected {%.1f, %.1f}\n"
        i
        p.x
        p.y
        expected_x
        expected_y ;
      ok := false)
  done ;
  if !ok then print_endline "PASSED" else print_endline "FAILED" ;
  !ok

(* Test 2: Nested colored_point type with enum color *)
let test_colored_point () =
  print_endline "=== Test: Nested colored_point type ===" ;
  let colors = [|Red; Green; Blue; Red|] in
  let n = 4 in
  let v = Vector.create_custom colored_point_custom n in
  for i = 0 to n - 1 do
    Vector.set
      v
      i
      {color = colors.(i); pt = {x = float_of_int i; y = float_of_int (i * 2)}}
  done ;
  let ok = ref true in
  for i = 0 to n - 1 do
    let cp = Vector.get v i in
    let expected_color = colors.(i) in
    let expected_x = float_of_int i in
    let expected_y = float_of_int (i * 2) in
    if
      cp.color <> expected_color
      || abs_float (cp.pt.x -. expected_x) > 1e-6
      || abs_float (cp.pt.y -. expected_y) > 1e-6
    then (
      let color_str c =
        match c with Red -> "Red" | Green -> "Green" | Blue -> "Blue"
      in
      Printf.printf
        "  FAIL at %d: got {color=%s, pt={%.1f, %.1f}} expected {color=%s, \
         pt={%.1f, %.1f}}\n"
        i
        (color_str cp.color)
        cp.pt.x
        cp.pt.y
        (color_str expected_color)
        expected_x
        expected_y ;
      ok := false)
  done ;
  if !ok then print_endline "PASSED" else print_endline "FAILED" ;
  !ok

(* Test 3: Mixed int32 and float32 fields *)
type mixed_record = {count : int32; value : float32; flags : int32}
[@@sarek.type]

let test_mixed_record () =
  print_endline "=== Test: Mixed int32/float32 record ===" ;
  let n = 4 in
  let v = Vector.create_custom mixed_record_custom n in
  for i = 0 to n - 1 do
    Vector.set
      v
      i
      {
        count = Int32.of_int i;
        value = float_of_int i *. 1.5;
        flags = Int32.of_int (i * 10);
      }
  done ;
  let ok = ref true in
  for i = 0 to n - 1 do
    let r = Vector.get v i in
    let expected_count = Int32.of_int i in
    let expected_value = float_of_int i *. 1.5 in
    let expected_flags = Int32.of_int (i * 10) in
    if
      r.count <> expected_count
      || abs_float (r.value -. expected_value) > 1e-6
      || r.flags <> expected_flags
    then (
      Printf.printf
        "  FAIL at %d: got {count=%ld, value=%.2f, flags=%ld}\n"
        i
        r.count
        r.value
        r.flags ;
      ok := false)
  done ;
  if !ok then print_endline "PASSED" else print_endline "FAILED" ;
  !ok

(* Test 4: Variant with composable nested type payloads *)
type maybe_colored_point =
  | Red_pt of point
  | Blue_pt of point
  | Any_pt of colored_point
  | No_point
[@@sarek.type]

let test_maybe_colored_point () =
  print_endline "=== Test: Variant with nested type payloads ===" ;
  let n = 4 in
  let v = Vector.create_custom maybe_colored_point_custom n in
  (* Set up test data *)
  Vector.set v 0 (Red_pt {x = 1.0; y = 2.0}) ;
  Vector.set v 1 (Blue_pt {x = 3.0; y = 4.0}) ;
  Vector.set v 2 (Any_pt {color = Green; pt = {x = 5.0; y = 6.0}}) ;
  Vector.set v 3 No_point ;
  let ok = ref true in
  (* Check element 0: Red_pt *)
  (match Vector.get v 0 with
  | Red_pt p when abs_float (p.x -. 1.0) < 1e-6 && abs_float (p.y -. 2.0) < 1e-6
    ->
      ()
  | other ->
      Printf.printf
        "  FAIL at 0: expected Red_pt {1.0, 2.0}, got %s\n"
        (match other with
        | Red_pt p -> Printf.sprintf "Red_pt {%.1f, %.1f}" p.x p.y
        | Blue_pt p -> Printf.sprintf "Blue_pt {%.1f, %.1f}" p.x p.y
        | Any_pt cp -> Printf.sprintf "Any_pt {pt={%.1f, %.1f}}" cp.pt.x cp.pt.y
        | No_point -> "No_point") ;
      ok := false) ;
  (* Check element 1: Blue_pt *)
  (match Vector.get v 1 with
  | Blue_pt p
    when abs_float (p.x -. 3.0) < 1e-6 && abs_float (p.y -. 4.0) < 1e-6 ->
      ()
  | _ ->
      print_endline "  FAIL at 1: expected Blue_pt {3.0, 4.0}" ;
      ok := false) ;
  (* Check element 2: Any_pt with nested colored_point *)
  (match Vector.get v 2 with
  | Any_pt cp
    when cp.color = Green
         && abs_float (cp.pt.x -. 5.0) < 1e-6
         && abs_float (cp.pt.y -. 6.0) < 1e-6 ->
      ()
  | _ ->
      print_endline "  FAIL at 2: expected Any_pt {Green, {5.0, 6.0}}" ;
      ok := false) ;
  (* Check element 3: No_point *)
  (match Vector.get v 3 with
  | No_point -> ()
  | _ ->
      print_endline "  FAIL at 3: expected No_point" ;
      ok := false) ;
  if !ok then print_endline "PASSED" else print_endline "FAILED" ;
  !ok

let () =
  print_endline "=== Nested/Composable Custom Types Test ===" ;
  print_endline "" ;
  let t1 = test_point () in
  print_endline "" ;
  let t2 = test_colored_point () in
  print_endline "" ;
  let t3 = test_mixed_record () in
  print_endline "" ;
  let t4 = test_maybe_colored_point () in
  print_endline "" ;
  print_endline "=== Summary ===" ;
  Printf.printf "Basic point: %s\n" (if t1 then "PASS" else "FAIL") ;
  Printf.printf "Nested colored_point: %s\n" (if t2 then "PASS" else "FAIL") ;
  Printf.printf "Mixed int32/float32: %s\n" (if t3 then "PASS" else "FAIL") ;
  Printf.printf "Variant with payloads: %s\n" (if t4 then "PASS" else "FAIL") ;
  if t1 && t2 && t3 && t4 then (
    print_endline "\nAll tests passed!" ;
    exit 0)
  else (
    print_endline "\nSome tests failed!" ;
    exit 1)
