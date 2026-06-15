(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Sarek Float32 Standard Library
 *
 * Provides float32 type and math functions for use in Sarek kernels.
 * Uses %sarek_intrinsic to define GPU intrinsics.
 *
 * Each intrinsic generates:
 * - func_device : device -> string  (the device code generator)
 * - func_device_ref : (device -> string) ref  (for extension chaining)
 * - func : host-side OCaml implementation
 * - Registry entry for JIT code generation
 *
 * To extend for a new backend (e.g., FPGA), use %sarek_extend:
 *   let%sarek_extend Float32.sin = fun dev ->
 *     if is_fpga dev then "fpga_sin" else Float32.sin_device dev
 ******************************************************************************)

(******************************************************************************
 * Type registration
 *
 * Register float32 as a primitive type. This must happen before any functions
 * that use float32 are registered.
 ******************************************************************************)

let%sarek_intrinsic float32 =
  {device = (fun _ -> "float"); ctype = Ctypes.float}

(******************************************************************************
 * Arithmetic operators
 *
 * These are the fundamental float32 operations used by the kernel code generator.
 ******************************************************************************)

let dev cuda opencl d = Sarek_registry.cuda_or_opencl d cuda opencl

let%sarek_intrinsic (add_float32 : float32 -> float32 -> float32) =
  {device = dev "(%s + %s)" "(%s + %s)"; ocaml = ( +. )}

let%sarek_intrinsic (sub_float32 : float32 -> float32 -> float32) =
  {device = dev "(%s - %s)" "(%s - %s)"; ocaml = ( -. )}

let%sarek_intrinsic (mul_float32 : float32 -> float32 -> float32) =
  {device = dev "(%s * %s)" "(%s * %s)"; ocaml = ( *. )}

let%sarek_intrinsic (div_float32 : float32 -> float32 -> float32) =
  {device = dev "(%s / %s)" "(%s / %s)"; ocaml = ( /. )}

(******************************************************************************
 * Unary math functions
 ******************************************************************************)

let%sarek_intrinsic (sin : float32 -> float32) =
  {device = dev "sinf" "sin"; ocaml = Stdlib.sin}

let%sarek_intrinsic (cos : float32 -> float32) =
  {device = dev "cosf" "cos"; ocaml = Stdlib.cos}

let%sarek_intrinsic (tan : float32 -> float32) =
  {device = dev "tanf" "tan"; ocaml = Stdlib.tan}

let%sarek_intrinsic (asin : float32 -> float32) =
  {device = dev "asinf" "asin"; ocaml = Stdlib.asin}

let%sarek_intrinsic (acos : float32 -> float32) =
  {device = dev "acosf" "acos"; ocaml = Stdlib.acos}

let%sarek_intrinsic (atan : float32 -> float32) =
  {device = dev "atanf" "atan"; ocaml = Stdlib.atan}

let%sarek_intrinsic (sinh : float32 -> float32) =
  {device = dev "sinhf" "sinh"; ocaml = Stdlib.sinh}

let%sarek_intrinsic (cosh : float32 -> float32) =
  {device = dev "coshf" "cosh"; ocaml = Stdlib.cosh}

let%sarek_intrinsic (tanh : float32 -> float32) =
  {device = dev "tanhf" "tanh"; ocaml = Stdlib.tanh}

let%sarek_intrinsic (exp : float32 -> float32) =
  {device = dev "expf" "exp"; ocaml = Stdlib.exp}

let%sarek_intrinsic (log : float32 -> float32) =
  {device = dev "logf" "log"; ocaml = Stdlib.log}

let%sarek_intrinsic (log10 : float32 -> float32) =
  {device = dev "log10f" "log10"; ocaml = Stdlib.log10}

let%sarek_intrinsic (sqrt : float32 -> float32) =
  {device = dev "sqrtf" "sqrt"; ocaml = Stdlib.sqrt}

let%sarek_intrinsic (ceil : float32 -> float32) =
  {device = dev "ceilf" "ceil"; ocaml = Stdlib.ceil}

let%sarek_intrinsic (floor : float32 -> float32) =
  {device = dev "floorf" "floor"; ocaml = Stdlib.floor}

let%sarek_intrinsic (expm1 : float32 -> float32) =
  {device = dev "expm1f" "expm1"; ocaml = Stdlib.expm1}

let%sarek_intrinsic (log1p : float32 -> float32) =
  {device = dev "log1pf" "log1p"; ocaml = Stdlib.log1p}

let%sarek_intrinsic (abs_float : float32 -> float32) =
  {device = dev "fabsf" "fabs"; ocaml = Stdlib.abs_float}

let%sarek_intrinsic (rsqrt : float32 -> float32) =
  {device = dev "rsqrtf" "rsqrt"; ocaml = (fun x -> 1.0 /. Stdlib.sqrt x)}

(******************************************************************************
 * Binary math functions
 ******************************************************************************)

let%sarek_intrinsic (pow : float32 -> float32 -> float32) =
  {device = dev "powf" "pow"; ocaml = Float.pow}

let%sarek_intrinsic (atan2 : float32 -> float32 -> float32) =
  {device = dev "atan2f" "atan2"; ocaml = Stdlib.atan2}

let%sarek_intrinsic (hypot : float32 -> float32 -> float32) =
  {device = dev "hypotf" "hypot"; ocaml = Stdlib.hypot}

let%sarek_intrinsic (copysign : float32 -> float32 -> float32) =
  {device = dev "copysignf" "copysign"; ocaml = Stdlib.copysign}

(******************************************************************************
 * Short aliases for backward compatibility
 *
 * The old Math.Float32 module used short names like `add`, `minus`, `mul`, `div`.
 ******************************************************************************)

let%sarek_intrinsic (add : float32 -> float32 -> float32) =
  {device = dev "(%s + %s)" "(%s + %s)"; ocaml = ( +. )}

let%sarek_intrinsic (minus : float32 -> float32 -> float32) =
  {device = dev "(%s - %s)" "(%s - %s)"; ocaml = ( -. )}

let%sarek_intrinsic (mul : float32 -> float32 -> float32) =
  {device = dev "(%s * %s)" "(%s * %s)"; ocaml = ( *. )}

let%sarek_intrinsic (div : float32 -> float32 -> float32) =
  {device = dev "(%s / %s)" "(%s / %s)"; ocaml = ( /. )}

(******************************************************************************
 * Conversion functions
 *
 * These are GPU intrinsics that generate proper casts in device code.
 ******************************************************************************)

let%sarek_intrinsic (of_int : int -> float32) =
  {device = dev "(float)(%s)" "(float)(%s)"; ocaml = Stdlib.float_of_int}

let%sarek_intrinsic (to_int : float32 -> int) =
  {device = dev "(int)(%s)" "(int)(%s)"; ocaml = Stdlib.int_of_float}
