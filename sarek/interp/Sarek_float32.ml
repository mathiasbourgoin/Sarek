(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Sarek Float32 - True 32-bit floating point operations
 *
 * Provides float32 semantics matching GPU behavior:
 * - All operations truncate to float32 precision
 * - Optional overflow/underflow detection
 * - Math intrinsics (exp, log, sin, cos, etc.) with float32 precision
 ******************************************************************************)

(** Float32 constants *)
let max_float32 = 3.40282347e+38

let min_positive_float32 = 1.17549435e-38

let epsilon_float32 = 1.19209290e-07

(** Maximum input for exp before overflow *)
let max_exp_input = 88.72283905 (* ln(max_float32) *)

(** Overflow detection mode *)
type overflow_mode =
  | Silent  (** Return infinity/-infinity silently (GPU behavior) *)
  | Warn  (** Print warning but continue *)
  | Exception  (** Raise exception *)

let overflow_mode = ref Silent

let set_overflow_mode mode = overflow_mode := mode

(** Exception for overflow detection *)
exception Float32_overflow of string

exception Float32_underflow of string

(** Truncate float64 to float32 precision by round-trip through Int32 bits *)
(* Note: C stubs could be added for better performance:
   external float32_of_float : float -> float = "caml_float32_of_float"
   external float_of_float32 : float -> float = "caml_float_of_float32"
*)

(* Pure OCaml implementation *)
let float32_of_float_impl x =
  (* Simulate float32 by clamping and reducing precision *)
  if x > max_float32 then infinity
  else if x < -.max_float32 then neg_infinity
  else if x <> 0. && abs_float x < min_positive_float32 then 0.
  else
    (* Round to float32 precision using Int32 bit representation *)
    Int32.float_of_bits (Int32.bits_of_float x)

let to_float32 = float32_of_float_impl

(** Alias for to_float32 - convert float64 to float32 *)
let of_float = float32_of_float_impl [@@warning "-32"]

(** Check for overflow and handle according to mode *)
let check_overflow name result =
  match !overflow_mode with
  | Silent -> result
  | Warn ->
      if result = infinity then
        Printf.eprintf "Warning: Float32 overflow in %s\n%!" name
      else if result = neg_infinity then
        Printf.eprintf "Warning: Float32 negative overflow in %s\n%!" name ;
      result
  | Exception ->
      if result = infinity then raise (Float32_overflow name)
      else if result = neg_infinity then
        raise (Float32_overflow (name ^ " (negative)"))
      else result

(** Check for underflow *)
let _check_underflow name x result =
  if x <> 0. && result = 0. then
    match !overflow_mode with
    | Silent -> result
    | Warn ->
        Printf.eprintf "Warning: Float32 underflow in %s\n%!" name ;
        result
    | Exception -> raise (Float32_underflow name)
  else result

(** Basic arithmetic with float32 precision *)
let add x y = to_float32 (x +. y) |> check_overflow "add"

let sub x y = to_float32 (x -. y) |> check_overflow "sub"

let mul x y = to_float32 (x *. y) |> check_overflow "mul"

let div x y = to_float32 (x /. y) |> check_overflow "div"

let neg x = to_float32 (-.x)

(** Comparison *)
let ( = ) x y = x = y

let ( <> ) x y = x <> y

let ( < ) x y = x < y

let ( > ) x y = x > y

let ( <= ) x y = x <= y

let ( >= ) x y = x >= y

(** Math intrinsics with float32 precision *)

let exp x =
  if x > max_exp_input then check_overflow "exp" infinity
  else to_float32 (Stdlib.exp x) |> check_overflow "exp"

let log x =
  if x <= 0. then check_overflow "log" neg_infinity
  else to_float32 (Stdlib.log x)

let log10 x =
  if x <= 0. then check_overflow "log10" neg_infinity
  else to_float32 (Stdlib.log10 x)

let pow x y =
  let result = to_float32 (x ** y) in
  check_overflow "pow" result

let sqrt x = if x < 0. then nan else to_float32 (Stdlib.sqrt x)

let rsqrt x = if x <= 0. then nan else to_float32 (1. /. Stdlib.sqrt x)

let sin x = to_float32 (Stdlib.sin x)

let cos x = to_float32 (Stdlib.cos x)

let tan x = to_float32 (Stdlib.tan x)

let asin x = to_float32 (Stdlib.asin x)

let acos x = to_float32 (Stdlib.acos x)

let atan x = to_float32 (Stdlib.atan x)

let atan2 y x = to_float32 (Stdlib.atan2 y x)

let sinh x = to_float32 (Stdlib.sinh x) |> check_overflow "sinh"

let cosh x = to_float32 (Stdlib.cosh x) |> check_overflow "cosh"

let tanh x = to_float32 (Stdlib.tanh x)

let floor x = to_float32 (Stdlib.floor x)

let ceil x = to_float32 (Stdlib.ceil x)

let abs x = to_float32 (Stdlib.abs_float x)

(* A single fused rounding (Float.fma), then one float32 rounding — not
   [(x *. y) +. z], whose intermediate product rounds independently and breaks
   exact TwoProd error extraction for the df64 (float-float) library. *)

(** FMA (fused multiply-add) - important for GPU accuracy *)
let fma x y z = to_float32 (Float.fma x y z) |> check_overflow "fma"

(** Min/max that handle NaN correctly (GPU semantics) *)
let min x y =
  if x <> x then y (* x is NaN *)
  else if y <> y then x (* y is NaN *)
  else if x < y then x
  else y

let max x y = if x <> x then y else if y <> y then x else if x > y then x else y

(** Clamp value to range *)
let clamp x lo hi = min (max x lo) hi

(** Check if value is finite *)
let is_finite x = x = x && x <> infinity && x <> neg_infinity

(** Check if value is NaN *)
let is_nan x = x <> x

(** Check if value is infinity *)
let is_inf x = x = infinity || x = neg_infinity

(** Convert from/to int32 *)
let of_int32 x = to_float32 (Int32.to_float x)

let to_int32 x = Int32.of_float x

(** Convert from/to int *)
let of_int x = to_float32 (float_of_int x)

let to_int x = int_of_float x

(** Pretty print with float32-appropriate precision *)
let to_string x = Printf.sprintf "%.7g" x

(******************************************************************************
 * Sarek stdlib surface mirror
 *
 * The native (cpu_kern) backend lowers an [IntrinsicRef (["Float32"], name)] to
 * the OCaml identifier [Sarek.Sarek_cpu_runtime.Float32.<name>] with the name
 * copied VERBATIM (sarek/ppx/Sarek_native_intrinsics.ml: map_stdlib_path, then
 * Sarek_native_helpers.evar_qualified). So every name declared by
 * sarek/Sarek_stdlib/Float32.ml must exist here under exactly that spelling,
 * or the kernel does not compile — the user sees "Unbound value
 * Sarek.Sarek_cpu_runtime.Float32.<name>" pointing into PPX-generated code.
 *
 * The seven names below were declared in the stdlib but missing here, so
 * Float32.abs_float / expm1 / log1p / hypot / copysign / fmod / minus were
 * unusable from any native kernel. sarek/tests/unit/test_intrinsic_surface.ml
 * reconciles the two surfaces so this cannot silently recur.
 *
 * The four *_float32 aliases mirror the stdlib's explicit arithmetic
 * intrinsics. They are reachable only when written as qualified calls
 * (Float32.add_float32 a b); infix `+.` is lowered structurally by the PPX and
 * never reaches this module.
 ******************************************************************************)

let abs_float x = abs x

let minus x y = sub x y

(* check_overflow is value-neutral in the default [Silent] mode -- it returns
   its argument untouched -- so these three report an infinite result in [Warn]
   and [Exception] mode without changing a single number the interpreter
   produces as the cross-backend oracle.

   Reachability was settled by an exhaustive sweep of all 2^32 binary32 inputs
   against C's expm1f/log1pf (clang 21, glibc, x86-64), replicating
   float32_of_float_impl exactly; see the PR for the full table.
     - expm1 overflows to +inf from x > 88.7228394, and that threshold is
       BIT-IDENTICAL to expm1f's: 0 spurious and 0 missed overflows over the
       whole domain, and only 2 inputs in 4.28e9 differ by 1 ulp. So the value
       was already right; only the reporting was missing (the Major finding).
     - log1p cannot overflow from any finite input -- its only infinity is the
       pole at x = -1. It is wired anyway because [log] and [log10] above route
       their pole through check_overflow too, so this is the house convention
       rather than a dead gate. NaN is unaffected: check_overflow tests only
       = infinity / = neg_infinity, and NaN compares false to both, so
       log1p x < -1. still returns NaN exactly as C's log1pf does.
     - hypot DOES overflow from finite inputs (first at hypot(6.11e37, 3.35e38)),
       with 0 spurious and 0 missed over a 1.33e8-pair boundary sweep. *)

let expm1 x = to_float32 (Stdlib.expm1 x) |> check_overflow "expm1"

let log1p x = to_float32 (Stdlib.log1p x) |> check_overflow "log1p"

let hypot x y = to_float32 (Stdlib.hypot x y) |> check_overflow "hypot"

let copysign x y = to_float32 (Stdlib.copysign x y)

(* C [fmod] semantics: sign of the dividend, magnitude < |divisor|.
   [Float.rem] is OCaml's [fmod]; the stdlib intrinsic uses the same. *)
let fmod x y = to_float32 (Float.rem x y)

let add_float32 x y = add x y

let sub_float32 x y = sub x y

let mul_float32 x y = mul x y

let div_float32 x y = div x y
