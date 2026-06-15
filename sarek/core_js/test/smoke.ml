(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * smoke — spoc_core_js host Vector API smoke test.
 *
 * Exercises Float32, Float64, Int32, Int64 scalar vectors via the browser-safe
 * spoc_core_js API: create, set, get, to_array, length, kind, elem_size.
 * Prints "spoc_core_js smoke: PASS" and exits 0 on success, exits 1 on failure.
 *
 * Builds as:
 *   - smoke.bc     (bytecode, native-compat test runner)
 *   - smoke.bc.js  (js_of_ocaml, browser-path gate)
 ******************************************************************************)

module V = Spoc_core_js

let check_float32 () =
  let n = 6 in
  let vec = V.create V.float32 n in
  assert (vec.V.length = n) ;
  assert (V.kind_name V.float32 = "Float32") ;
  assert (V.elem_size V.float32 = 4) ;
  for i = 0 to n - 1 do
    V.set vec i (float_of_int (i * 10))
  done ;
  for i = 0 to n - 1 do
    assert (V.get vec i = float_of_int (i * 10))
  done ;
  let arr = V.to_array vec in
  assert (Array.length arr = n) ;
  Array.iteri (fun i v -> assert (v = float_of_int (i * 10))) arr

let check_float64 () =
  let n = 4 in
  let vec = V.create V.float64 n in
  assert (vec.V.length = n) ;
  assert (V.kind_name V.float64 = "Float64") ;
  assert (V.elem_size V.float64 = 8) ;
  let values = [|1.1; 2.2; 3.3; 4.4|] in
  Array.iteri (fun i v -> V.set vec i v) values ;
  Array.iteri (fun i v -> assert (V.get vec i = v)) values ;
  let arr = V.to_array vec in
  assert (Array.length arr = n) ;
  Array.iteri (fun i v -> assert (v = values.(i))) arr

let check_int32 () =
  let n = 5 in
  let vec = V.create V.int32 n in
  assert (vec.V.length = n) ;
  assert (V.kind_name V.int32 = "Int32") ;
  assert (V.elem_size V.int32 = 4) ;
  for i = 0 to n - 1 do
    V.set vec i (Int32.of_int (i + 100))
  done ;
  for i = 0 to n - 1 do
    assert (V.get vec i = Int32.of_int (i + 100))
  done ;
  let arr = V.to_array vec in
  assert (Array.length arr = n) ;
  Array.iteri (fun i v -> assert (v = Int32.of_int (i + 100))) arr

let check_int64 () =
  let n = 3 in
  let vec = V.create V.int64 n in
  assert (vec.V.length = n) ;
  assert (V.kind_name V.int64 = "Int64") ;
  assert (V.elem_size V.int64 = 8) ;
  let values = [|1000L; 2000L; 3000L|] in
  Array.iteri (fun i v -> V.set vec i v) values ;
  Array.iteri (fun i v -> assert (V.get vec i = v)) values ;
  let arr = V.to_array vec in
  assert (Array.length arr = n) ;
  Array.iteri (fun i v -> assert (v = values.(i))) arr

let () =
  try
    check_float32 () ;
    check_float64 () ;
    check_int32 () ;
    check_int64 () ;
    Printf.printf "spoc_core_js smoke: PASS\n%!"
  with e ->
    Printf.eprintf "spoc_core_js smoke: FAIL — %s\n%!" (Printexc.to_string e) ;
    exit 1
