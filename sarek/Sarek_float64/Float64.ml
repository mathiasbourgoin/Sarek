(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Sarek Float64 Standard Library
 *
 * Provides float64 (double precision) type and math functions for Sarek kernels.
 * Uses %sarek_intrinsic to define GPU intrinsics.
 *
 * Each intrinsic generates:
 * - func_device : device -> string  (the device code generator)
 * - func_device_ref : (device -> string) ref  (for extension chaining)
 * - func : host-side OCaml implementation
 * - Registry entry for JIT code generation
 ******************************************************************************)

(******************************************************************************
 * Type registration
 ******************************************************************************)

let%sarek_intrinsic float64 =
  {device = (fun _ -> "double"); ctype = Ctypes.double}

(******************************************************************************
 * Arithmetic operators
 ******************************************************************************)

let dev cuda opencl d = Sarek_registry.cuda_or_opencl d cuda opencl

let%sarek_intrinsic (add_float64 : float -> float -> float) =
  {device = dev "(%s + %s)" "(%s + %s)"; ocaml = ( +. )}

let%sarek_intrinsic (sub_float64 : float -> float -> float) =
  {device = dev "(%s - %s)" "(%s - %s)"; ocaml = ( -. )}

let%sarek_intrinsic (mul_float64 : float -> float -> float) =
  {device = dev "(%s * %s)" "(%s * %s)"; ocaml = ( *. )}

let%sarek_intrinsic (div_float64 : float -> float -> float) =
  {device = dev "(%s / %s)" "(%s / %s)"; ocaml = ( /. )}

(******************************************************************************
 * Unary math functions
 * Note: float64 uses same function names as float32 but without 'f' suffix
 ******************************************************************************)

let%sarek_intrinsic (sin : float -> float) =
  {device = dev "sin" "sin"; ocaml = Stdlib.sin}

let%sarek_intrinsic (cos : float -> float) =
  {device = dev "cos" "cos"; ocaml = Stdlib.cos}

let%sarek_intrinsic (tan : float -> float) =
  {device = dev "tan" "tan"; ocaml = Stdlib.tan}

let%sarek_intrinsic (asin : float -> float) =
  {device = dev "asin" "asin"; ocaml = Stdlib.asin}

let%sarek_intrinsic (acos : float -> float) =
  {device = dev "acos" "acos"; ocaml = Stdlib.acos}

let%sarek_intrinsic (atan : float -> float) =
  {device = dev "atan" "atan"; ocaml = Stdlib.atan}

let%sarek_intrinsic (sinh : float -> float) =
  {device = dev "sinh" "sinh"; ocaml = Stdlib.sinh}

let%sarek_intrinsic (cosh : float -> float) =
  {device = dev "cosh" "cosh"; ocaml = Stdlib.cosh}

let%sarek_intrinsic (tanh : float -> float) =
  {device = dev "tanh" "tanh"; ocaml = Stdlib.tanh}

let%sarek_intrinsic (exp : float -> float) =
  {device = dev "exp" "exp"; ocaml = Stdlib.exp}

let%sarek_intrinsic (log : float -> float) =
  {device = dev "log" "log"; ocaml = Stdlib.log}

let%sarek_intrinsic (log10 : float -> float) =
  {device = dev "log10" "log10"; ocaml = Stdlib.log10}

let%sarek_intrinsic (sqrt : float -> float) =
  {device = dev "sqrt" "sqrt"; ocaml = Stdlib.sqrt}

let%sarek_intrinsic (ceil : float -> float) =
  {device = dev "ceil" "ceil"; ocaml = Stdlib.ceil}

let%sarek_intrinsic (floor : float -> float) =
  {device = dev "floor" "floor"; ocaml = Stdlib.floor}

let%sarek_intrinsic (expm1 : float -> float) =
  {device = dev "expm1" "expm1"; ocaml = Stdlib.expm1}

let%sarek_intrinsic (log1p : float -> float) =
  {device = dev "log1p" "log1p"; ocaml = Stdlib.log1p}

let%sarek_intrinsic (abs_float : float -> float) =
  {device = dev "fabs" "fabs"; ocaml = Stdlib.abs_float}

let%sarek_intrinsic (rsqrt : float -> float) =
  {device = dev "rsqrt" "rsqrt"; ocaml = (fun x -> 1.0 /. Stdlib.sqrt x)}

(******************************************************************************
 * Binary math functions
 ******************************************************************************)

let%sarek_intrinsic (pow : float -> float -> float) =
  {device = dev "pow" "pow"; ocaml = Float.pow}

let%sarek_intrinsic (atan2 : float -> float -> float) =
  {device = dev "atan2" "atan2"; ocaml = Stdlib.atan2}

let%sarek_intrinsic (hypot : float -> float -> float) =
  {device = dev "hypot" "hypot"; ocaml = Stdlib.hypot}

let%sarek_intrinsic (copysign : float -> float -> float) =
  {device = dev "copysign" "copysign"; ocaml = Stdlib.copysign}

(******************************************************************************
 * Conversion functions
 *
 * These are GPU intrinsics that generate proper casts in device code.
 * Note: We use `float` in the type annotation to represent float64 on the
 * host side, since OCaml's float is always 64-bit.
 ******************************************************************************)

let%sarek_intrinsic (of_int : int -> float) =
  {device = dev "(double)(%s)" "(double)(%s)"; ocaml = Stdlib.float_of_int}

let%sarek_intrinsic (of_int32 : int32 -> float) =
  {device = dev "(double)(%s)" "(double)(%s)"; ocaml = Int32.to_float}

let%sarek_intrinsic (to_int : float -> int) =
  {device = dev "(int)(%s)" "(int)(%s)"; ocaml = Stdlib.int_of_float}

let%sarek_intrinsic (to_int32 : float -> int32) =
  {device = dev "(int)(%s)" "(int)(%s)"; ocaml = Int32.of_float}

let%sarek_intrinsic (of_float32 : float32 -> float) =
  {device = dev "(double)(%s)" "(double)(%s)"; ocaml = (fun x -> x)}

let%sarek_intrinsic (to_float32 : float -> float32) =
  {device = dev "(float)(%s)" "(float)(%s)"; ocaml = (fun x -> x)}

(******************************************************************************
 * Operator aliases
 *
 * These provide convenient infix operators for float64 arithmetic.
 * Use `let open Float64 in` to shadow the float32 operators from Std.
 ******************************************************************************)

let%sarek_intrinsic (( +. ) : float -> float -> float) =
  {device = dev "(%s + %s)" "(%s + %s)"; ocaml = ( +. )}

let%sarek_intrinsic (( -. ) : float -> float -> float) =
  {device = dev "(%s - %s)" "(%s - %s)"; ocaml = ( -. )}

let%sarek_intrinsic (( *. ) : float -> float -> float) =
  {device = dev "(%s * %s)" "(%s * %s)"; ocaml = ( *. )}

let%sarek_intrinsic (( /. ) : float -> float -> float) =
  {device = dev "(%s / %s)" "(%s / %s)"; ocaml = ( /. )}

let%sarek_intrinsic (( <= ) : float -> float -> bool) =
  {device = dev "(%s <= %s)" "(%s <= %s)"; ocaml = ( <= )}

let%sarek_intrinsic (( >= ) : float -> float -> bool) =
  {device = dev "(%s >= %s)" "(%s >= %s)"; ocaml = ( >= )}

let%sarek_intrinsic (( < ) : float -> float -> bool) =
  {device = dev "(%s < %s)" "(%s < %s)"; ocaml = ( < )}

let%sarek_intrinsic (( > ) : float -> float -> bool) =
  {device = dev "(%s > %s)" "(%s > %s)"; ocaml = ( > )}
