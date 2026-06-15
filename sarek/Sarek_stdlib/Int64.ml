(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Sarek Int64 Standard Library
 *
 * Provides int64 type and operations for Sarek kernels.
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

let%sarek_intrinsic int64 = {device = (fun _ -> "long"); ctype = Ctypes.int64_t}

(******************************************************************************
 * Arithmetic operators
 ******************************************************************************)

let dev cuda opencl d = Sarek_registry.cuda_or_opencl d cuda opencl

let%sarek_intrinsic (add_int64 : int64 -> int64 -> int64) =
  {device = dev "(%s + %s)" "(%s + %s)"; ocaml = Stdlib.Int64.add}

let%sarek_intrinsic (sub_int64 : int64 -> int64 -> int64) =
  {device = dev "(%s - %s)" "(%s - %s)"; ocaml = Stdlib.Int64.sub}

let%sarek_intrinsic (mul_int64 : int64 -> int64 -> int64) =
  {device = dev "(%s * %s)" "(%s * %s)"; ocaml = Stdlib.Int64.mul}

let%sarek_intrinsic (div_int64 : int64 -> int64 -> int64) =
  {device = dev "(%s / %s)" "(%s / %s)"; ocaml = Stdlib.Int64.div}

let%sarek_intrinsic (mod_int64 : int64 -> int64 -> int64) =
  {device = dev "(%s %% %s)" "(%s %% %s)"; ocaml = Stdlib.Int64.rem}

(******************************************************************************
 * Bitwise operators
 ******************************************************************************)

let%sarek_intrinsic (logand : int64 -> int64 -> int64) =
  {device = dev "(%s & %s)" "(%s & %s)"; ocaml = Stdlib.Int64.logand}

let%sarek_intrinsic (logor : int64 -> int64 -> int64) =
  {device = dev "(%s | %s)" "(%s | %s)"; ocaml = Stdlib.Int64.logor}

let%sarek_intrinsic (logxor : int64 -> int64 -> int64) =
  {device = dev "(%s ^ %s)" "(%s ^ %s)"; ocaml = Stdlib.Int64.logxor}

let%sarek_intrinsic (lognot : int64 -> int64) =
  {device = dev "(~%s)" "(~%s)"; ocaml = Stdlib.Int64.lognot}

let%sarek_intrinsic (shift_left : int64 -> int -> int64) =
  {device = dev "(%s << %s)" "(%s << %s)"; ocaml = Stdlib.Int64.shift_left}

let%sarek_intrinsic (shift_right : int64 -> int -> int64) =
  {device = dev "(%s >> %s)" "(%s >> %s)"; ocaml = Stdlib.Int64.shift_right}

let%sarek_intrinsic (shift_right_logical : int64 -> int -> int64) =
  {
    device = dev "((unsigned long)%s >> %s)" "((unsigned long)%s >> %s)";
    ocaml = Stdlib.Int64.shift_right_logical;
  }

(******************************************************************************
 * Comparison operators
 ******************************************************************************)

let%sarek_intrinsic (abs : int64 -> int64) =
  {device = dev "llabs(%s)" "labs(%s)"; ocaml = Stdlib.Int64.abs}

let%sarek_intrinsic (neg : int64 -> int64) =
  {device = dev "(-%s)" "(-%s)"; ocaml = Stdlib.Int64.neg}

let%sarek_intrinsic (min : int64 -> int64 -> int64) =
  {device = dev "min(%s, %s)" "min(%s, %s)"; ocaml = Stdlib.Int64.min}

let%sarek_intrinsic (max : int64 -> int64 -> int64) =
  {device = dev "max(%s, %s)" "max(%s, %s)"; ocaml = Stdlib.Int64.max}

(******************************************************************************
 * Conversion functions
 ******************************************************************************)

let of_int x = Stdlib.Int64.of_int x

let to_int x = Stdlib.Int64.to_int x

let of_int32 x = Stdlib.Int64.of_int32 x

let to_int32 x = Stdlib.Int64.to_int32 x

let of_float x = Stdlib.Int64.of_float x

let to_float x = Stdlib.Int64.to_float x

let zero = 0L

let one = 1L

let minus_one = -1L
