(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Unit tests for Float64 module in Sarek_stdlib
 *
 * Tests the Float64 intrinsics that generate device code for float64 (double)
 * operations. Focuses on the OCaml-side implementations since GPU execution
 * is tested in E2E tests.
 ******************************************************************************)

module F64 = Sarek_float64.Float64

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
    if abs_float expected > 1e-15 then diff /. abs_float expected else diff
  in
  test name (rel_diff < tolerance)

let () =
  Printf.printf "Testing Float64 module (Sarek_stdlib)\n\n" ;

  (* Test arithmetic operators *)
  Printf.printf "Arithmetic operators:\n" ;
  test_approx
    "add_float64 1.5 + 2.5"
    ~expected:4.0
    ~got:(F64.add_float64 1.5 2.5)
    ~tolerance:1e-15 ;
  test_approx
    "sub_float64 5.0 - 3.0"
    ~expected:2.0
    ~got:(F64.sub_float64 5.0 3.0)
    ~tolerance:1e-15 ;
  test_approx
    "mul_float64 2.0 * 3.0"
    ~expected:6.0
    ~got:(F64.mul_float64 2.0 3.0)
    ~tolerance:1e-15 ;
  test_approx
    "div_float64 10.0 / 4.0"
    ~expected:2.5
    ~got:(F64.div_float64 10.0 4.0)
    ~tolerance:1e-15 ;

  (* Test conversion functions *)
  Printf.printf "\nConversion functions:\n" ;
  test_approx
    "of_int 42 = 42.0"
    ~expected:42.0
    ~got:(F64.of_int 42)
    ~tolerance:1e-15 ;
  test_approx
    "of_int (-123) = -123.0"
    ~expected:(-123.0)
    ~got:(F64.of_int (-123))
    ~tolerance:1e-15 ;
  test_approx
    "of_int 0 = 0.0"
    ~expected:0.0
    ~got:(F64.of_int 0)
    ~tolerance:1e-15 ;
  test "to_int 42.7 = 42" (F64.to_int 42.7 = 42) ;
  test "to_int (-3.9) = -3" (F64.to_int (-3.9) = -3) ;
  test "to_int 0.0 = 0" (F64.to_int 0.0 = 0) ;

  (* Test float32 conversions *)
  Printf.printf "\nFloat32 conversions:\n" ;
  test_approx
    "of_float32 3.14 = 3.14"
    ~expected:3.14
    ~got:(F64.of_float32 3.14)
    ~tolerance:1e-6 ;
  test_approx
    "to_float32 3.14159265358979 truncates"
    ~expected:3.14159265358979
    ~got:(F64.to_float32 3.14159265358979)
    ~tolerance:1e-6 ;

  (* Test trig functions *)
  Printf.printf "\nTrigonometric functions:\n" ;
  test_approx "sin(0.0) = 0.0" ~expected:0.0 ~got:(F64.sin 0.0) ~tolerance:1e-15 ;
  test_approx "cos(0.0) = 1.0" ~expected:1.0 ~got:(F64.cos 0.0) ~tolerance:1e-15 ;
  test_approx "tan(0.0) = 0.0" ~expected:0.0 ~got:(F64.tan 0.0) ~tolerance:1e-15 ;
  let pi = 3.14159265358979323846 in
  test_approx
    "sin(pi/2) = 1.0"
    ~expected:1.0
    ~got:(F64.sin (pi /. 2.0))
    ~tolerance:1e-10 ;
  test_approx
    "cos(pi) = -1.0"
    ~expected:(-1.0)
    ~got:(F64.cos pi)
    ~tolerance:1e-10 ;

  (* Test inverse trig *)
  Printf.printf "\nInverse trig functions:\n" ;
  test_approx
    "asin(0.0) = 0.0"
    ~expected:0.0
    ~got:(F64.asin 0.0)
    ~tolerance:1e-15 ;
  test_approx
    "acos(1.0) = 0.0"
    ~expected:0.0
    ~got:(F64.acos 1.0)
    ~tolerance:1e-15 ;
  test_approx
    "atan(0.0) = 0.0"
    ~expected:0.0
    ~got:(F64.atan 0.0)
    ~tolerance:1e-15 ;

  (* Test hyperbolic functions *)
  Printf.printf "\nHyperbolic functions:\n" ;
  test_approx
    "sinh(0.0) = 0.0"
    ~expected:0.0
    ~got:(F64.sinh 0.0)
    ~tolerance:1e-15 ;
  test_approx
    "cosh(0.0) = 1.0"
    ~expected:1.0
    ~got:(F64.cosh 0.0)
    ~tolerance:1e-15 ;
  test_approx
    "tanh(0.0) = 0.0"
    ~expected:0.0
    ~got:(F64.tanh 0.0)
    ~tolerance:1e-15 ;

  (* Test exp/log *)
  Printf.printf "\nExponential and logarithm:\n" ;
  test_approx "exp(0.0) = 1.0" ~expected:1.0 ~got:(F64.exp 0.0) ~tolerance:1e-15 ;
  test_approx
    "exp(1.0) = e"
    ~expected:2.718281828459045
    ~got:(F64.exp 1.0)
    ~tolerance:1e-10 ;
  test_approx "log(1.0) = 0.0" ~expected:0.0 ~got:(F64.log 1.0) ~tolerance:1e-15 ;
  test_approx
    "log(e) = 1.0"
    ~expected:1.0
    ~got:(F64.log 2.718281828459045)
    ~tolerance:1e-10 ;
  test_approx
    "log10(10.0) = 1.0"
    ~expected:1.0
    ~got:(F64.log10 10.0)
    ~tolerance:1e-15 ;
  test_approx
    "log10(100.0) = 2.0"
    ~expected:2.0
    ~got:(F64.log10 100.0)
    ~tolerance:1e-15 ;

  (* Test expm1/log1p for numerical stability *)
  Printf.printf "\nNumerical stability functions:\n" ;
  test_approx
    "expm1(0.0) = 0.0"
    ~expected:0.0
    ~got:(F64.expm1 0.0)
    ~tolerance:1e-15 ;
  test_approx
    "log1p(0.0) = 0.0"
    ~expected:0.0
    ~got:(F64.log1p 0.0)
    ~tolerance:1e-15 ;
  (* expm1 and log1p are accurate for small values *)
  test_approx
    "expm1(1e-15) accurate"
    ~expected:1e-15
    ~got:(F64.expm1 1e-15)
    ~tolerance:1e-25 ;

  (* Test sqrt *)
  Printf.printf "\nSquare root:\n" ;
  test_approx
    "sqrt(4.0) = 2.0"
    ~expected:2.0
    ~got:(F64.sqrt 4.0)
    ~tolerance:1e-15 ;
  test_approx
    "sqrt(2.0) = 1.41421356..."
    ~expected:1.4142135623730951
    ~got:(F64.sqrt 2.0)
    ~tolerance:1e-15 ;
  test_approx
    "rsqrt(4.0) = 0.5"
    ~expected:0.5
    ~got:(F64.rsqrt 4.0)
    ~tolerance:1e-15 ;

  (* Test ceil/floor *)
  Printf.printf "\nCeil and floor:\n" ;
  test_approx
    "ceil(2.3) = 3.0"
    ~expected:3.0
    ~got:(F64.ceil 2.3)
    ~tolerance:1e-15 ;
  test_approx
    "floor(2.7) = 2.0"
    ~expected:2.0
    ~got:(F64.floor 2.7)
    ~tolerance:1e-15 ;
  test_approx
    "ceil(-2.3) = -2.0"
    ~expected:(-2.0)
    ~got:(F64.ceil (-2.3))
    ~tolerance:1e-15 ;
  test_approx
    "floor(-2.7) = -3.0"
    ~expected:(-3.0)
    ~got:(F64.floor (-2.7))
    ~tolerance:1e-15 ;

  (* Test abs_float *)
  Printf.printf "\nAbsolute value:\n" ;
  test_approx
    "abs_float(5.0) = 5.0"
    ~expected:5.0
    ~got:(F64.abs_float 5.0)
    ~tolerance:1e-15 ;
  test_approx
    "abs_float(-5.0) = 5.0"
    ~expected:5.0
    ~got:(F64.abs_float (-5.0))
    ~tolerance:1e-15 ;

  (* Test pow *)
  Printf.printf "\nPower:\n" ;
  test_approx
    "pow(2.0, 10.0) = 1024.0"
    ~expected:1024.0
    ~got:(F64.pow 2.0 10.0)
    ~tolerance:1e-10 ;
  test_approx
    "pow(3.0, 0.0) = 1.0"
    ~expected:1.0
    ~got:(F64.pow 3.0 0.0)
    ~tolerance:1e-15 ;
  test_approx
    "pow(2.0, 0.5) = sqrt(2)"
    ~expected:1.4142135623730951
    ~got:(F64.pow 2.0 0.5)
    ~tolerance:1e-15 ;

  (* Test atan2 *)
  Printf.printf "\nAtan2:\n" ;
  test_approx
    "atan2(0.0, 1.0) = 0.0"
    ~expected:0.0
    ~got:(F64.atan2 0.0 1.0)
    ~tolerance:1e-15 ;
  test_approx
    "atan2(1.0, 0.0) = pi/2"
    ~expected:(pi /. 2.0)
    ~got:(F64.atan2 1.0 0.0)
    ~tolerance:1e-10 ;

  (* Test hypot *)
  Printf.printf "\nHypot:\n" ;
  test_approx
    "hypot(3.0, 4.0) = 5.0"
    ~expected:5.0
    ~got:(F64.hypot 3.0 4.0)
    ~tolerance:1e-15 ;

  (* Test copysign *)
  Printf.printf "\nCopysign:\n" ;
  test_approx
    "copysign(5.0, -1.0) = -5.0"
    ~expected:(-5.0)
    ~got:(F64.copysign 5.0 (-1.0))
    ~tolerance:1e-15 ;
  test_approx
    "copysign(-5.0, 1.0) = 5.0"
    ~expected:5.0
    ~got:(F64.copysign (-5.0) 1.0)
    ~tolerance:1e-15 ;

  (* Test precision - float64 should handle more digits than float32 *)
  Printf.printf "\nPrecision (float64 vs float32):\n" ;
  let x = 1.00000000001 in
  let y = 1.00000000005 in
  test "float64 can distinguish 1.00000000001 from 1.00000000005" (x <> y) ;
  let pi64 = 3.14159265358979323846 in
  test_approx
    "pi with float64 precision"
    ~expected:3.14159265358979323846
    ~got:pi64
    ~tolerance:1e-15 ;

  (* Summary *)
  Printf.printf "\n========================================\n" ;
  Printf.printf
    "Tests: %d, Passed: %d, Failed: %d\n"
    !test_count
    !pass_count
    !fail_count ;
  if !fail_count > 0 then exit 1 else Printf.printf "All tests passed!\n"
