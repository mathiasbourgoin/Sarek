(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * SPIKE: test/assertion harness for numeric_vector_spike
 *
 * Runs under bytecode and js_of_ocaml — no Alcotest, no ctypes, no unix.
 * Exit code 0 = PASS, non-zero = FAIL.
 ******************************************************************************)

open Numeric_vector_spike

(* ------------------------------------------------------------------ helpers *)

let check_bool desc b =
  if not b then begin
    Printf.printf "FAIL: %s\n%!" desc ;
    exit 1
  end

let check_int desc expected actual =
  if expected <> actual then begin
    Printf.printf "FAIL: %s — expected %d, got %d\n%!" desc expected actual ;
    exit 1
  end

let check_float desc expected actual =
  let eps = 1e-6 in
  if abs_float (expected -. actual) > eps then begin
    Printf.printf "FAIL: %s — expected %f, got %f\n%!" desc expected actual ;
    exit 1
  end

let check_int32 desc expected actual =
  if Int32.compare expected actual <> 0 then begin
    Printf.printf "FAIL: %s — expected %ld, got %ld\n%!" desc expected actual ;
    exit 1
  end

(* ----------------------------------------------- elem_size_of_kind tests *)

let test_elem_sizes () =
  check_int "elem_size Float32" 4 (elem_size_of_kind Float32) ;
  check_int "elem_size Float64" 8 (elem_size_of_kind Float64) ;
  check_int "elem_size Int32" 4 (elem_size_of_kind Int32) ;
  check_int "elem_size Int64" 8 (elem_size_of_kind Int64) ;
  check_int "elem_size Int8u" 1 (elem_size_of_kind Int8u) ;
  check_int "elem_size Complex32" 8 (elem_size_of_kind Complex32)

(* ----------------------------------------------- Float32 vector tests *)

let test_float32 () =
  let v = create Float32 5 in
  check_int "Float32 length" 5 (length v) ;
  check_int "Float32 elem_size" 4 (elem_size v) ;
  set v 0 1.0 ;
  set v 1 2.0 ;
  set v 2 3.0 ;
  set v 3 4.0 ;
  set v 4 5.0 ;
  check_float "Float32 get 0" 1.0 (get v 0) ;
  check_float "Float32 get 4" 5.0 (get v 4) ;
  let arr = to_array v in
  check_int "Float32 to_array length" 5 (Array.length arr) ;
  check_float "Float32 to_array[2]" 3.0 arr.(2)

(* ----------------------------------------------- Float64 vector tests *)

let test_float64 () =
  let v = create Float64 3 in
  check_int "Float64 length" 3 (length v) ;
  check_int "Float64 elem_size" 8 (elem_size v) ;
  set v 0 1.5 ;
  set v 1 2.5 ;
  set v 2 3.5 ;
  check_float "Float64 get 1" 2.5 (get v 1) ;
  let arr = to_array v in
  check_float "Float64 to_array[0]" 1.5 arr.(0)

(* ----------------------------------------------- Int32 vector tests *)

let test_int32 () =
  let v = create Int32 4 in
  check_int "Int32 length" 4 (length v) ;
  check_int "Int32 elem_size" 4 (elem_size v) ;
  set v 0 10l ;
  set v 1 20l ;
  set v 2 30l ;
  set v 3 40l ;
  check_int32 "Int32 get 0" 10l (get v 0) ;
  check_int32 "Int32 get 3" 40l (get v 3) ;
  let arr = to_array v in
  check_int32 "Int32 to_array[1]" 20l arr.(1)

(* ----------------------------------------------- bounds check tests *)

let test_bounds () =
  let v = create Float32 3 in
  let got_exn = ref false in
  (try
     let _ = get v (-1) in
     ()
   with Invalid_argument _ -> got_exn := true) ;
  check_bool "Float32 get negative index raises" !got_exn ;
  got_exn := false ;
  (try
     let _ = get v 3 in
     ()
   with Invalid_argument _ -> got_exn := true) ;
  check_bool "Float32 get past-end raises" !got_exn ;
  got_exn := false ;
  (try set v 5 99.0 with Invalid_argument _ -> got_exn := true) ;
  check_bool "Float32 set past-end raises" !got_exn

(* ----------------------------------------------- injectable clock test *)

let test_injectable_clock () =
  (* Default is Sys.time — call it and verify we get a positive float *)
  let t0 = !now () in
  check_bool "default clock returns positive float" (t0 >= 0.0) ;
  (* Swap the clock for a constant stub *)
  let stub_time = 42.0 in
  let saved = !now in
  (now := fun () -> stub_time) ;
  let t1 = !now () in
  check_float "injected clock returns stub value" stub_time t1 ;
  now := saved

(* ----------------------------------------------- empty vector *)

let test_empty () =
  let v = create Float64 0 in
  check_int "empty length" 0 (length v) ;
  let arr = to_array v in
  check_int "empty to_array length" 0 (Array.length arr)

(* ----------------------------------------------- run all *)

let () =
  test_elem_sizes () ;
  test_float32 () ;
  test_float64 () ;
  test_int32 () ;
  test_bounds () ;
  test_injectable_clock () ;
  test_empty () ;
  print_endline "PASS"
