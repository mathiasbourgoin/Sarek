(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Unit tests for Sarek_float32 module - float32 precision operations *)

open Alcotest
open Sarek_float32

(** Helper to check floats with tolerance *)
let check_float_approx msg expected actual =
  let tolerance = 1e-5 in
  let diff = abs_float (expected -. actual) in
  check bool msg true (diff < tolerance)

(** Test constants *)
let test_max_float32 () =
  check_float_approx "max_float32" 3.40282347e+38 max_float32

let test_min_positive_float32 () =
  check_float_approx "min_positive_float32" 1.17549435e-38 min_positive_float32

let test_epsilon_float32 () =
  check_float_approx "epsilon_float32" 1.19209290e-07 epsilon_float32

(** Test basic arithmetic *)
let test_add () =
  let result = add 1.5 2.5 in
  check_float_approx "1.5 + 2.5 = 4.0" 4.0 result

let test_sub () =
  let result = sub 5.0 3.0 in
  check_float_approx "5.0 - 3.0 = 2.0" 2.0 result

let test_mul () =
  let result = mul 2.0 3.0 in
  check_float_approx "2.0 * 3.0 = 6.0" 6.0 result

let test_div () =
  let result = div 6.0 2.0 in
  check_float_approx "6.0 / 2.0 = 3.0" 3.0 result

let test_neg () =
  let result = neg 5.0 in
  check_float_approx "-(5.0) = -5.0" (-5.0) result

(** Test comparison operations *)
let test_eq () =
  let result = 1.0 = 1.0 in
  check bool "1.0 = 1.0" true result

let test_neq () =
  let result = 1.0 <> 2.0 in
  check bool "1.0 <> 2.0" true result

let test_lt () =
  let result = 1.0 < 2.0 in
  check bool "1.0 < 2.0" true result

let test_gt () =
  let result = 2.0 > 1.0 in
  check bool "2.0 > 1.0" true result

let test_le () =
  let result = 1.0 <= 1.0 in
  check bool "1.0 <= 1.0" true result

let test_ge () =
  let result = 2.0 >= 2.0 in
  check bool "2.0 >= 2.0" true result

(** Test math intrinsics *)
let test_sqrt () =
  let result = sqrt 9.0 in
  check_float_approx "sqrt(9.0) = 3.0" 3.0 result

let test_sqrt_negative_is_nan () =
  let result = sqrt (-1.0) in
  check bool "sqrt(-1.0) is NaN" true (is_nan result)

let test_exp () =
  let result = exp 0.0 in
  check_float_approx "exp(0.0) = 1.0" 1.0 result

let test_log () =
  let result = log 1.0 in
  check_float_approx "log(1.0) = 0.0" 0.0 result

let test_log_negative_is_inf () =
  let result = log (-1.0) in
  check bool "log(-1.0) is -inf" true (result = neg_infinity)

let test_sin () =
  let result = sin 0.0 in
  check_float_approx "sin(0.0) = 0.0" 0.0 result

let test_cos () =
  let result = cos 0.0 in
  check_float_approx "cos(0.0) = 1.0" 1.0 result

let test_tan () =
  let result = tan 0.0 in
  check_float_approx "tan(0.0) = 0.0" 0.0 result

let test_floor () =
  let result = floor 2.7 in
  check_float_approx "floor(2.7) = 2.0" 2.0 result

let test_ceil () =
  let result = ceil 2.3 in
  check_float_approx "ceil(2.3) = 3.0" 3.0 result

let test_abs_positive () =
  let result = abs 5.0 in
  check_float_approx "abs(5.0) = 5.0" 5.0 result

let test_abs_negative () =
  let result = abs (-5.0) in
  check_float_approx "abs(-5.0) = 5.0" 5.0 result

(** Test min/max with NaN handling *)
let test_min () =
  let result = min 3.0 5.0 in
  check_float_approx "min(3.0, 5.0) = 3.0" 3.0 result

let test_min_with_nan () =
  let result = min nan 5.0 in
  check_float_approx "min(NaN, 5.0) = 5.0" 5.0 result

let test_max () =
  let result = max 3.0 5.0 in
  check_float_approx "max(3.0, 5.0) = 5.0" 5.0 result

let test_max_with_nan () =
  let result = max nan 5.0 in
  check_float_approx "max(NaN, 5.0) = 5.0" 5.0 result

(** Test clamp *)
let test_clamp_within_range () =
  let result = clamp 5.0 0.0 10.0 in
  check_float_approx "clamp(5.0, 0.0, 10.0) = 5.0" 5.0 result

let test_clamp_below_range () =
  let result = clamp (-1.0) 0.0 10.0 in
  check_float_approx "clamp(-1.0, 0.0, 10.0) = 0.0" 0.0 result

let test_clamp_above_range () =
  let result = clamp 15.0 0.0 10.0 in
  check_float_approx "clamp(15.0, 0.0, 10.0) = 10.0" 10.0 result

(** Test predicates *)
let test_is_finite_normal () =
  check bool "is_finite(5.0)" true (is_finite 5.0)

let test_is_finite_infinity () =
  check bool "not is_finite(infinity)" false (is_finite infinity)

let test_is_finite_nan () =
  check bool "not is_finite(NaN)" false (is_finite nan)

let test_is_nan_normal () =
  check bool "not is_nan(5.0)" false (is_nan 5.0)

let test_is_nan_nan () =
  check bool "is_nan(NaN)" true (is_nan nan)

let test_is_inf_normal () =
  check bool "not is_inf(5.0)" false (is_inf 5.0)

let test_is_inf_infinity () =
  check bool "is_inf(infinity)" true (is_inf infinity)

let test_is_inf_neg_infinity () =
  check bool "is_inf(neg_infinity)" true (is_inf neg_infinity)

(** Test conversion functions *)
let test_of_int32 () =
  let result = of_int32 42l in
  check_float_approx "of_int32(42l) = 42.0" 42.0 result

let test_to_int32 () =
  let result = to_int32 42.7 in
  check int32 "to_int32(42.7) = 42l" 42l result

let test_of_int () =
  let result = of_int 100 in
  check_float_approx "of_int(100) = 100.0" 100.0 result

let test_to_int () =
  let result = to_int 99.9 in
  check int "to_int(99.9) = 99" 99 result

(** Test FMA (fused multiply-add) *)
let test_fma () =
  let result = fma 2.0 3.0 4.0 in
  check_float_approx "fma(2.0, 3.0, 4.0) = 10.0" 10.0 result

(** Test rsqrt (reciprocal square root) *)
let test_rsqrt () =
  let result = rsqrt 4.0 in
  check_float_approx "rsqrt(4.0) = 0.5" 0.5 result

let test_rsqrt_negative_is_nan () =
  let result = rsqrt (-1.0) in
  check bool "rsqrt(-1.0) is NaN" true (is_nan result)

let test_rsqrt_zero_is_nan () =
  let result = rsqrt 0.0 in
  check bool "rsqrt(0.0) is NaN" true (is_nan result)

(** Test to_string *)
let test_to_string () =
  let s = to_string 3.14159 in
  check bool "to_string produces reasonable output" true (String.length s > 0)

(** Test precision truncation *)
let test_to_float32_truncates () =
  (* Double precision value that can't be exactly represented in float32 *)
  let double_precise = 1.23456789012345 in
  let f32 = to_float32 double_precise in
  (* After conversion, should have less precision *)
  check bool "to_float32 reduces precision" true (f32 <> double_precise)

(** Test overflow handling (Silent mode) *)
let test_overflow_silent () =
  set_overflow_mode Silent ;
  let result = exp 100.0 in
  check bool "exp(100.0) overflows to infinity in Silent mode" true
    (result = infinity)

let test_pow () =
  let result = pow 2.0 3.0 in
  check_float_approx "pow(2.0, 3.0) = 8.0" 8.0 result

let test_asin () =
  let result = asin 0.5 in
  check bool "asin(0.5) is reasonable" true (abs_float result < 2.0)

let test_acos () =
  let result = acos 0.5 in
  check bool "acos(0.5) is reasonable" true (abs_float result < 2.0)

let test_atan () =
  let result = atan 1.0 in
  check_float_approx "atan(1.0) ≈ π/4" 0.7853981 result

let test_atan2 () =
  let result = atan2 1.0 1.0 in
  check_float_approx "atan2(1.0, 1.0) ≈ π/4" 0.7853981 result

let test_sinh () =
  let result = sinh 0.0 in
  check_float_approx "sinh(0.0) = 0.0" 0.0 result

let test_cosh () =
  let result = cosh 0.0 in
  check_float_approx "cosh(0.0) = 1.0" 1.0 result

let test_tanh () =
  let result = tanh 0.0 in
  check_float_approx "tanh(0.0) = 0.0" 0.0 result

let test_log10 () =
  let result = log10 100.0 in
  check_float_approx "log10(100.0) = 2.0" 2.0 result

let () =
  run
    "Sarek_float32"
    [
      ( "constants",
        [
          test_case "max_float32" `Quick test_max_float32;
          test_case "min_positive_float32" `Quick test_min_positive_float32;
          test_case "epsilon_float32" `Quick test_epsilon_float32;
        ] );
      ( "arithmetic",
        [
          test_case "add" `Quick test_add;
          test_case "sub" `Quick test_sub;
          test_case "mul" `Quick test_mul;
          test_case "div" `Quick test_div;
          test_case "neg" `Quick test_neg;
        ] );
      ( "comparison",
        [
          test_case "eq" `Quick test_eq;
          test_case "neq" `Quick test_neq;
          test_case "lt" `Quick test_lt;
          test_case "gt" `Quick test_gt;
          test_case "le" `Quick test_le;
          test_case "ge" `Quick test_ge;
        ] );
      ( "math_intrinsics",
        [
          test_case "sqrt" `Quick test_sqrt;
          test_case "sqrt_negative" `Quick test_sqrt_negative_is_nan;
          test_case "exp" `Quick test_exp;
          test_case "log" `Quick test_log;
          test_case "log_negative" `Quick test_log_negative_is_inf;
          test_case "sin" `Quick test_sin;
          test_case "cos" `Quick test_cos;
          test_case "tan" `Quick test_tan;
          test_case "floor" `Quick test_floor;
          test_case "ceil" `Quick test_ceil;
          test_case "abs_positive" `Quick test_abs_positive;
          test_case "abs_negative" `Quick test_abs_negative;
          test_case "pow" `Quick test_pow;
          test_case "asin" `Quick test_asin;
          test_case "acos" `Quick test_acos;
          test_case "atan" `Quick test_atan;
          test_case "atan2" `Quick test_atan2;
          test_case "sinh" `Quick test_sinh;
          test_case "cosh" `Quick test_cosh;
          test_case "tanh" `Quick test_tanh;
          test_case "log10" `Quick test_log10;
        ] );
      ( "min_max",
        [
          test_case "min" `Quick test_min;
          test_case "min_with_nan" `Quick test_min_with_nan;
          test_case "max" `Quick test_max;
          test_case "max_with_nan" `Quick test_max_with_nan;
        ] );
      ( "clamp",
        [
          test_case "within_range" `Quick test_clamp_within_range;
          test_case "below_range" `Quick test_clamp_below_range;
          test_case "above_range" `Quick test_clamp_above_range;
        ] );
      ( "predicates",
        [
          test_case "is_finite_normal" `Quick test_is_finite_normal;
          test_case "is_finite_infinity" `Quick test_is_finite_infinity;
          test_case "is_finite_nan" `Quick test_is_finite_nan;
          test_case "is_nan_normal" `Quick test_is_nan_normal;
          test_case "is_nan_nan" `Quick test_is_nan_nan;
          test_case "is_inf_normal" `Quick test_is_inf_normal;
          test_case "is_inf_infinity" `Quick test_is_inf_infinity;
          test_case "is_inf_neg_infinity" `Quick test_is_inf_neg_infinity;
        ] );
      ( "conversion",
        [
          test_case "of_int32" `Quick test_of_int32;
          test_case "to_int32" `Quick test_to_int32;
          test_case "of_int" `Quick test_of_int;
          test_case "to_int" `Quick test_to_int;
        ] );
      ( "advanced",
        [
          test_case "fma" `Quick test_fma;
          test_case "rsqrt" `Quick test_rsqrt;
          test_case "rsqrt_negative" `Quick test_rsqrt_negative_is_nan;
          test_case "rsqrt_zero" `Quick test_rsqrt_zero_is_nan;
          test_case "to_string" `Quick test_to_string;
          test_case "truncation" `Quick test_to_float32_truncates;
          test_case "overflow_silent" `Quick test_overflow_silent;
        ] );
    ]
