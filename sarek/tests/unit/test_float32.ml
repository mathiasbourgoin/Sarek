(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Unit tests for Sarek_float32 module
 ******************************************************************************)

module F32 = Sarek.Sarek_float32

let test_count = ref 0

let pass_count = ref 0

let fail_count = ref 0

let test name condition =
  incr test_count ;
  if condition then begin
    incr pass_count ;
    Printf.printf "  [PASS] %s\n" name
  end
  else begin
    incr fail_count ;
    Printf.printf "  [FAIL] %s\n" name
  end

let test_approx name ~expected ~got ~tolerance =
  let diff = abs_float (expected -. got) in
  let rel_diff =
    if abs_float expected > 1e-10 then diff /. abs_float expected else diff
  in
  test name (rel_diff < tolerance)

let () =
  Printf.printf "Testing Sarek_float32 module\n\n" ;

  (* Test constants *)
  Printf.printf "Constants:\n" ;
  test
    "max_float32 is approximately 3.4e38"
    (F32.max_float32 > 3.4e38 && F32.max_float32 < 3.5e38) ;
  test
    "min_positive_float32 is approximately 1.17e-38"
    (F32.min_positive_float32 > 1.0e-38 && F32.min_positive_float32 < 1.5e-38) ;
  test
    "epsilon_float32 is approximately 1.19e-7"
    (F32.epsilon_float32 > 1.0e-7 && F32.epsilon_float32 < 1.5e-7) ;

  (* Test to_float32 precision truncation *)
  Printf.printf "\nPrecision truncation:\n" ;
  (* Float32 has ~7 decimal digits precision, so values differing by ~1e-8 collapse *)
  let x = 1.00000001 in
  let y = 1.00000005 in
  test "float64 can distinguish 1.00000001 from 1.00000005" (x <> y) ;
  test
    "float32 cannot distinguish them (same after truncation)"
    (F32.to_float32 x = F32.to_float32 y) ;

  (* Test that values are properly truncated *)
  let pi64 = 3.14159265358979323846 in
  let pi32 = F32.to_float32 pi64 in
  test_approx
    "pi truncated to float32"
    ~expected:3.1415927
    ~got:pi32
    ~tolerance:1e-6 ;

  (* Test basic arithmetic *)
  Printf.printf "\nArithmetic:\n" ;
  test_approx
    "add 1.5 + 2.5"
    ~expected:4.0
    ~got:(F32.add 1.5 2.5)
    ~tolerance:1e-6 ;
  test_approx
    "sub 5.0 - 3.0"
    ~expected:2.0
    ~got:(F32.sub 5.0 3.0)
    ~tolerance:1e-6 ;
  test_approx
    "mul 2.0 * 3.0"
    ~expected:6.0
    ~got:(F32.mul 2.0 3.0)
    ~tolerance:1e-6 ;
  test_approx
    "div 10.0 / 4.0"
    ~expected:2.5
    ~got:(F32.div 10.0 4.0)
    ~tolerance:1e-6 ;
  test_approx "neg -5.0" ~expected:5.0 ~got:(F32.neg (-5.0)) ~tolerance:1e-6 ;

  (* Test exp overflow behavior *)
  Printf.printf "\nExp overflow:\n" ;
  test "exp(88.0) is finite" (F32.is_finite (F32.exp 88.0)) ;
  test "exp(89.0) overflows to infinity" (F32.exp 89.0 = infinity) ;
  test "exp(100.0) overflows to infinity" (F32.exp 100.0 = infinity) ;
  test_approx "exp(0.0) = 1.0" ~expected:1.0 ~got:(F32.exp 0.0) ~tolerance:1e-6 ;
  test_approx
    "exp(1.0) = e"
    ~expected:2.7182818
    ~got:(F32.exp 1.0)
    ~tolerance:1e-5 ;

  (* Test log *)
  Printf.printf "\nLog:\n" ;
  test_approx "log(1.0) = 0.0" ~expected:0.0 ~got:(F32.log 1.0) ~tolerance:1e-6 ;
  test_approx
    "log(e) = 1.0"
    ~expected:1.0
    ~got:(F32.log 2.7182818)
    ~tolerance:1e-5 ;
  test "log(0.0) = -infinity" (F32.log 0.0 = neg_infinity) ;
  test "log(-1.0) = -infinity" (F32.log (-1.0) = neg_infinity) ;

  (* Test log10 *)
  Printf.printf "\nLog10:\n" ;
  test_approx
    "log10(10.0) = 1.0"
    ~expected:1.0
    ~got:(F32.log10 10.0)
    ~tolerance:1e-6 ;
  test_approx
    "log10(100.0) = 2.0"
    ~expected:2.0
    ~got:(F32.log10 100.0)
    ~tolerance:1e-6 ;

  (* Test pow *)
  Printf.printf "\nPow:\n" ;
  test_approx
    "pow(2.0, 10.0) = 1024.0"
    ~expected:1024.0
    ~got:(F32.pow 2.0 10.0)
    ~tolerance:1e-4 ;
  test_approx
    "pow(3.0, 0.0) = 1.0"
    ~expected:1.0
    ~got:(F32.pow 3.0 0.0)
    ~tolerance:1e-6 ;
  test "pow(2.0, 200.0) overflows" (F32.pow 2.0 200.0 = infinity) ;

  (* Test sqrt *)
  Printf.printf "\nSqrt:\n" ;
  test_approx
    "sqrt(4.0) = 2.0"
    ~expected:2.0
    ~got:(F32.sqrt 4.0)
    ~tolerance:1e-6 ;
  test_approx
    "sqrt(2.0) = 1.414..."
    ~expected:1.4142135
    ~got:(F32.sqrt 2.0)
    ~tolerance:1e-5 ;
  test "sqrt(-1.0) is NaN" (F32.is_nan (F32.sqrt (-1.0))) ;

  (* Test trig functions *)
  Printf.printf "\nTrig:\n" ;
  test_approx "sin(0.0) = 0.0" ~expected:0.0 ~got:(F32.sin 0.0) ~tolerance:1e-6 ;
  test_approx "cos(0.0) = 1.0" ~expected:1.0 ~got:(F32.cos 0.0) ~tolerance:1e-6 ;
  let pi = 3.14159265 in
  test_approx
    "sin(pi/2) = 1.0"
    ~expected:1.0
    ~got:(F32.sin (pi /. 2.0))
    ~tolerance:1e-5 ;
  test_approx
    "cos(pi) = -1.0"
    ~expected:(-1.0)
    ~got:(F32.cos pi)
    ~tolerance:1e-5 ;

  (* Test min/max with NaN *)
  Printf.printf "\nMin/Max with NaN:\n" ;
  test_approx
    "min(2.0, 3.0) = 2.0"
    ~expected:2.0
    ~got:(F32.min 2.0 3.0)
    ~tolerance:1e-6 ;
  test_approx
    "max(2.0, 3.0) = 3.0"
    ~expected:3.0
    ~got:(F32.max 2.0 3.0)
    ~tolerance:1e-6 ;
  test_approx
    "min(nan, 3.0) = 3.0 (GPU semantics)"
    ~expected:3.0
    ~got:(F32.min nan 3.0)
    ~tolerance:1e-6 ;
  test_approx
    "max(nan, 3.0) = 3.0 (GPU semantics)"
    ~expected:3.0
    ~got:(F32.max nan 3.0)
    ~tolerance:1e-6 ;

  (* Test predicates *)
  Printf.printf "\nPredicates:\n" ;
  test "is_finite(1.0) = true" (F32.is_finite 1.0) ;
  test "is_finite(infinity) = false" (not (F32.is_finite infinity)) ;
  test "is_finite(nan) = false" (not (F32.is_finite nan)) ;
  test "is_nan(nan) = true" (F32.is_nan nan) ;
  test "is_nan(1.0) = false" (not (F32.is_nan 1.0)) ;
  test "is_inf(infinity) = true" (F32.is_inf infinity) ;
  test "is_inf(neg_infinity) = true" (F32.is_inf neg_infinity) ;
  test "is_inf(1.0) = false" (not (F32.is_inf 1.0)) ;

  (* Test overflow modes *)
  Printf.printf "\nOverflow modes:\n" ;
  F32.set_overflow_mode F32.Warn ;
  let _ = F32.exp 100.0 in
  (* Should print warning *)
  F32.set_overflow_mode F32.Silent ;
  test "overflow mode can be set" true ;

  F32.set_overflow_mode F32.Exception ;
  let overflow_exception_caught =
    try
      let _ = F32.exp 100.0 in
      false
    with F32.Float32_overflow _ -> true
  in
  test "Float32_overflow exception is raised" overflow_exception_caught ;
  F32.set_overflow_mode F32.Silent ;

  (* Test conversions *)
  Printf.printf "\nConversions:\n" ;
  test_approx
    "of_int32 42l = 42.0"
    ~expected:42.0
    ~got:(F32.of_int32 42l)
    ~tolerance:1e-6 ;
  test "to_int32 42.7 = 42l" (Int32.equal (F32.to_int32 42.7) 42l) ;
  test_approx
    "of_int 123 = 123.0"
    ~expected:123.0
    ~got:(F32.of_int 123)
    ~tolerance:1e-6 ;

  (* Summary *)
  Printf.printf "\n========================================\n" ;
  Printf.printf
    "Tests: %d, Passed: %d, Failed: %d\n"
    !test_count
    !pass_count
    !fail_count ;
  if !fail_count > 0 then exit 1 else Printf.printf "All tests passed!\n"
