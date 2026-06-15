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

(** FMA (fused multiply-add) - important for GPU accuracy *)
let fma x y z = to_float32 ((x *. y) +. z) |> check_overflow "fma"

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
