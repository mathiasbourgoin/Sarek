(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Unit tests for Vector module
 *
 * Tests vector creation, element access, location tracking, and operations.
 * These tests run on CPU only (no GPU backend required).
 ******************************************************************************)

open Spoc_core.Vector

(** {1 Scalar Kind Tests} *)

let test_scalar_kinds () =
  (* Verify scalar kind constructors exist *)
  let _f32 : (float, Bigarray.float32_elt) scalar_kind = Float32 in
  let _f64 : (float, Bigarray.float64_elt) scalar_kind = Float64 in
  let _i32 : (int32, Bigarray.int32_elt) scalar_kind = Int32 in
  let _i64 : (int64, Bigarray.int64_elt) scalar_kind = Int64 in
  let _chr : (char, Bigarray.int8_unsigned_elt) scalar_kind = Char in
  let _cplx : (Complex.t, Bigarray.complex32_elt) scalar_kind = Complex32 in
  print_endline "  scalar_kinds: OK"

let test_kind_constructors () =
  let sk : (float, Bigarray.float32_elt) kind = Scalar Float32 in
  (match sk with Scalar Float32 -> ()) ;
  print_endline "  kind constructors: OK"

let test_elem_size () =
  assert (elem_size float32 = 4) ;
  assert (elem_size float64 = 8) ;
  assert (elem_size int32 = 4) ;
  assert (elem_size int64 = 8) ;
  print_endline "  elem_size: OK"

let test_kind_name () =
  assert (kind_name float32 = "Float32") ;
  assert (kind_name float64 = "Float64") ;
  assert (kind_name int32 = "Int32") ;
  assert (kind_name int64 = "Int64") ;
  print_endline "  kind_name: OK"

(** {1 Vector Creation Tests} *)

let test_create_float32 () =
  let v = create_float32 100 in
  assert (length v = 100) ;
  assert (kind_name (kind v) = "Float32") ;
  print_endline "  create_float32: OK"

let test_create_float64 () =
  let v = create_float64 50 in
  assert (length v = 50) ;
  assert (kind_name (kind v) = "Float64") ;
  print_endline "  create_float64: OK"

let test_create_int32 () =
  let v = create_int32 200 in
  assert (length v = 200) ;
  assert (kind_name (kind v) = "Int32") ;
  print_endline "  create_int32: OK"

let test_create_int64 () =
  let v = create_int64 75 in
  assert (length v = 75) ;
  assert (kind_name (kind v) = "Int64") ;
  print_endline "  create_int64: OK"

let test_create_generic () =
  let v = create float32 100 in
  assert (length v = 100) ;
  print_endline "  create generic: OK"

(** {1 Location Tests} *)

let test_initial_location () =
  let v = create_float32 10 in
  assert (location v = CPU) ;
  print_endline "  initial location (CPU): OK"

let test_location_predicates () =
  let v = create_float32 10 in
  assert (is_on_cpu v = true) ;
  assert (is_on_gpu v = false) ;
  assert (is_synced v = false) ;
  (* only CPU, not synced with GPU *)
  print_endline "  location predicates: OK"

let test_location_to_string () =
  assert (location_to_string CPU = "CPU") ;
  print_endline "  location_to_string: OK"

(** {1 Element Access Tests} *)

let test_get_set_float32 () =
  let v = create_float32 10 in
  set v 0 1.5 ;
  set v 5 2.5 ;
  set v 9 3.5 ;
  assert (get v 0 = 1.5) ;
  assert (get v 5 = 2.5) ;
  assert (get v 9 = 3.5) ;
  print_endline "  get/set float32: OK"

let test_get_set_int32 () =
  let v = create_int32 10 in
  set v 0 100l ;
  set v 5 200l ;
  assert (get v 0 = 100l) ;
  assert (get v 5 = 200l) ;
  print_endline "  get/set int32: OK"

let test_get_set_int64 () =
  let v = create_int64 10 in
  set v 0 1000L ;
  set v 5 2000L ;
  assert (get v 0 = 1000L) ;
  assert (get v 5 = 2000L) ;
  print_endline "  get/set int64: OK"

let test_bounds_check () =
  let v = create_float32 10 in
  let raised_low = ref false in
  let raised_high = ref false in
  (try set v (-1) 1.0 with Invalid_argument _ -> raised_low := true) ;
  (try set v 10 1.0 with Invalid_argument _ -> raised_high := true) ;
  assert !raised_low ;
  assert !raised_high ;
  print_endline "  bounds check: OK"

let test_unsafe_get_set () =
  let v = create_float32 10 in
  unsafe_set v 0 42.0 ;
  assert (unsafe_get v 0 = 42.0) ;
  print_endline "  unsafe_get/set: OK"

(** {1 Fill and Init Tests} *)

let test_fill () =
  let v = create_float32 5 in
  fill v 7.0 ;
  for i = 0 to 4 do
    assert (get v i = 7.0)
  done ;
  print_endline "  fill: OK"

let test_init () =
  let v = init float32 5 (fun i -> Float.of_int i) in
  assert (get v 0 = 0.0) ;
  assert (get v 1 = 1.0) ;
  assert (get v 4 = 4.0) ;
  print_endline "  init: OK"

(** {1 Copy Tests} *)

let test_copy () =
  let v1 = create_float32 5 in
  fill v1 3.0 ;
  let v2 = copy v1 in
  assert (length v2 = length v1) ;
  assert (get v2 0 = 3.0) ;
  (* Verify they're independent *)
  set v1 0 99.0 ;
  assert (get v2 0 = 3.0) ;
  print_endline "  copy: OK"

(** {1 Conversion Tests} *)

let test_of_bigarray () =
  let ba = Bigarray.Array1.create Bigarray.float32 Bigarray.c_layout 5 in
  Bigarray.Array1.fill ba 2.0 ;
  let v = of_bigarray Float32 ba in
  assert (length v = 5) ;
  assert (get v 0 = 2.0) ;
  print_endline "  of_bigarray: OK"

let test_to_bigarray () =
  let v = create_float32 5 in
  fill v 4.0 ;
  let ba = to_bigarray v in
  assert (Bigarray.Array1.dim ba = 5) ;
  assert (Bigarray.Array1.get ba 0 = 4.0) ;
  print_endline "  to_bigarray: OK"

let test_of_list () =
  let v = of_list float32 [1.0; 2.0; 3.0; 4.0; 5.0] in
  assert (length v = 5) ;
  assert (get v 0 = 1.0) ;
  assert (get v 4 = 5.0) ;
  print_endline "  of_list: OK"

let test_to_list () =
  let v = create_float32 3 in
  set v 0 1.0 ;
  set v 1 2.0 ;
  set v 2 3.0 ;
  let l = to_list v in
  assert (l = [1.0; 2.0; 3.0]) ;
  print_endline "  to_list: OK"

let test_of_array () =
  let v = of_array float32 [|10.0; 20.0; 30.0|] in
  assert (length v = 3) ;
  assert (get v 1 = 20.0) ;
  print_endline "  of_array: OK"

let test_to_array () =
  let v = create_float32 3 in
  set v 0 100.0 ;
  set v 1 200.0 ;
  set v 2 300.0 ;
  let arr = to_array v in
  assert (arr.(0) = 100.0) ;
  assert (arr.(2) = 300.0) ;
  print_endline "  to_array: OK"

(** {1 Iterator Tests} *)

let test_iter () =
  let v = of_list float32 [1.0; 2.0; 3.0] in
  let sum = ref 0.0 in
  iter (fun x -> sum := !sum +. x) v ;
  assert (!sum = 6.0) ;
  print_endline "  iter: OK"

let test_iteri () =
  let v = of_list float32 [10.0; 20.0; 30.0] in
  let idx_sum = ref 0 in
  iteri (fun i _ -> idx_sum := !idx_sum + i) v ;
  assert (!idx_sum = 3) ;
  (* 0 + 1 + 2 *)
  print_endline "  iteri: OK"

let test_map () =
  let v1 = of_list float32 [1.0; 2.0; 3.0] in
  let v2 = map (fun x -> x *. 2.0) float32 v1 in
  assert (get v2 0 = 2.0) ;
  assert (get v2 1 = 4.0) ;
  assert (get v2 2 = 6.0) ;
  print_endline "  map: OK"

let test_mapi () =
  let v1 = of_list float32 [10.0; 20.0; 30.0] in
  let v2 = mapi (fun i x -> x +. Float.of_int i) float32 v1 in
  assert (get v2 0 = 10.0) ;
  assert (get v2 1 = 21.0) ;
  assert (get v2 2 = 32.0) ;
  print_endline "  mapi: OK"

let test_map_inplace () =
  let v = of_list float32 [1.0; 2.0; 3.0] in
  map_inplace (fun x -> x *. 10.0) v ;
  assert (get v 0 = 10.0) ;
  assert (get v 1 = 20.0) ;
  assert (get v 2 = 30.0) ;
  print_endline "  map_inplace: OK"

(** {1 Fold Tests} *)

let test_fold_left () =
  let v = of_list float32 [1.0; 2.0; 3.0; 4.0] in
  let sum = fold_left (fun acc x -> acc +. x) 0.0 v in
  assert (sum = 10.0) ;
  print_endline "  fold_left: OK"

let test_fold_right () =
  let v = of_list float32 [1.0; 2.0; 3.0] in
  let result = fold_right (fun x acc -> x :: acc) v [] in
  assert (result = [1.0; 2.0; 3.0]) ;
  print_endline "  fold_right: OK"

(** {1 Predicate Tests} *)

let test_for_all () =
  let v = of_list float32 [2.0; 4.0; 6.0; 8.0] in
  assert (for_all (fun x -> x > 0.0) v = true) ;
  assert (for_all (fun x -> x > 5.0) v = false) ;
  print_endline "  for_all: OK"

let test_exists () =
  let v = of_list float32 [1.0; 2.0; 3.0; 4.0] in
  assert (exists (fun x -> x = 3.0) v = true) ;
  assert (exists (fun x -> x = 99.0) v = false) ;
  print_endline "  exists: OK"

let test_find () =
  let v = of_list float32 [1.0; 2.0; 3.0; 4.0] in
  let found = find (fun x -> x > 2.5) v in
  assert (found = Some 3.0) ;
  let not_found = find (fun x -> x > 100.0) v in
  assert (not_found = None) ;
  print_endline "  find: OK"

let test_find_index () =
  let v = of_list float32 [10.0; 20.0; 30.0; 40.0] in
  let found = find_index (fun x -> x = 30.0) v in
  assert (found = Some 2) ;
  let not_found = find_index (fun x -> x = 99.0) v in
  assert (not_found = None) ;
  print_endline "  find_index: OK"

(** {1 Aggregate Tests} *)

let test_sum () =
  let v = of_list float32 [1.0; 2.0; 3.0; 4.0; 5.0] in
  let s = sum ~zero:0.0 ~add:( +. ) v in
  assert (s = 15.0) ;
  print_endline "  sum: OK"

let test_min_elt () =
  let v = of_list float32 [3.0; 1.0; 4.0; 1.5; 9.0] in
  let m = min_elt ~compare:Float.compare v in
  assert (m = Some 1.0) ;
  print_endline "  min_elt: OK"

let test_max_elt () =
  let v = of_list float32 [3.0; 1.0; 4.0; 1.5; 9.0] in
  let m = max_elt ~compare:Float.compare v in
  assert (m = Some 9.0) ;
  print_endline "  max_elt: OK"

(** {1 Subvector Tests} *)

let test_sub_vector () =
  let v = of_list float32 [0.0; 1.0; 2.0; 3.0; 4.0; 5.0; 6.0; 7.0; 8.0; 9.0] in
  let sub = sub_vector v ~start:2 ~len:5 () in
  assert (length sub = 5) ;
  assert (get sub 0 = 2.0) ;
  assert (get sub 4 = 6.0) ;
  print_endline "  sub_vector: OK"

let test_is_sub () =
  let v = of_list float32 [0.0; 1.0; 2.0; 3.0; 4.0] in
  let sub = sub_vector v ~start:1 ~len:3 () in
  assert (is_sub v = false) ;
  assert (is_sub sub = true) ;
  print_endline "  is_sub: OK"

(** {1 Blit Tests} *)

let test_blit () =
  let src = of_list float32 [1.0; 2.0; 3.0; 4.0; 5.0] in
  let dst = create_float32 10 in
  fill dst 0.0 ;
  blit ~src ~src_off:1 ~dst ~dst_off:3 ~len:3 ;
  assert (get dst 3 = 2.0) ;
  assert (get dst 4 = 3.0) ;
  assert (get dst 5 = 4.0) ;
  print_endline "  blit: OK"

(** {1 to_string Tests} *)

let test_to_string () =
  let v = create_float32 10 in
  let s = to_string v in
  assert (String.length s > 0) ;
  print_endline "  to_string: OK"

(** {1 Auto-sync Tests} *)

let test_auto_sync_flag () =
  let v = create_float32 10 in
  assert (auto_sync v = true) ;
  set_auto_sync v false ;
  assert (auto_sync v = false) ;
  set_auto_sync v true ;
  assert (auto_sync v = true) ;
  print_endline "  auto_sync flag: OK"

(** {1 ID Tests} *)

let test_unique_ids () =
  let v1 = create_float32 10 in
  let v2 = create_float32 10 in
  let v3 = create_float32 10 in
  assert (id v1 <> id v2) ;
  assert (id v2 <> id v3) ;
  assert (id v1 <> id v3) ;
  print_endline "  unique IDs: OK"

(** {1 Main} *)

let () =
  print_endline "Vector module tests:" ;
  (* Kind tests *)
  test_scalar_kinds () ;
  test_kind_constructors () ;
  test_elem_size () ;
  test_kind_name () ;
  (* Creation tests *)
  test_create_float32 () ;
  test_create_float64 () ;
  test_create_int32 () ;
  test_create_int64 () ;
  test_create_generic () ;
  (* Location tests *)
  test_initial_location () ;
  test_location_predicates () ;
  test_location_to_string () ;
  (* Access tests *)
  test_get_set_float32 () ;
  test_get_set_int32 () ;
  test_get_set_int64 () ;
  test_bounds_check () ;
  test_unsafe_get_set () ;
  (* Fill/init tests *)
  test_fill () ;
  test_init () ;
  (* Copy test *)
  test_copy () ;
  (* Conversion tests *)
  test_of_bigarray () ;
  test_to_bigarray () ;
  test_of_list () ;
  test_to_list () ;
  test_of_array () ;
  test_to_array () ;
  (* Iterator tests *)
  test_iter () ;
  test_iteri () ;
  test_map () ;
  test_mapi () ;
  test_map_inplace () ;
  (* Fold tests *)
  test_fold_left () ;
  test_fold_right () ;
  (* Predicate tests *)
  test_for_all () ;
  test_exists () ;
  test_find () ;
  test_find_index () ;
  (* Aggregate tests *)
  test_sum () ;
  test_min_elt () ;
  test_max_elt () ;
  (* Subvector tests *)
  test_sub_vector () ;
  test_is_sub () ;
  (* Blit test *)
  test_blit () ;
  (* Utility tests *)
  test_to_string () ;
  test_auto_sync_flag () ;
  test_unique_ids () ;
  print_endline "All Vector module tests passed!"
