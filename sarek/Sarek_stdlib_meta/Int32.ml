(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Sarek Int32 Stdlib Meta - pure metadata registration (FFI-free)
 *
 * Registers the int32 type and all Int32 intrinsic function signatures
 * into Sarek_ppx_registry without any Ctypes or Spoc_core dependency.
 * Size is derived from the type name (int32 = 4 bytes).
 ******************************************************************************)

(******************************************************************************
 * Type registration (ctype omitted — size derived from name by PPX)
 ******************************************************************************)

let%sarek_intrinsic int32 = {device = (fun _ -> "int")}

(******************************************************************************
 * Arithmetic operators
 ******************************************************************************)

let dev cuda opencl d = Sarek_registry.cuda_or_opencl d cuda opencl

let%sarek_intrinsic (add_int32 : int32 -> int32 -> int32) =
  {device = dev "(%s + %s)" "(%s + %s)"; ocaml = Stdlib.Int32.add}

let%sarek_intrinsic (sub_int32 : int32 -> int32 -> int32) =
  {device = dev "(%s - %s)" "(%s - %s)"; ocaml = Stdlib.Int32.sub}

let%sarek_intrinsic (mul_int32 : int32 -> int32 -> int32) =
  {device = dev "(%s * %s)" "(%s * %s)"; ocaml = Stdlib.Int32.mul}

let%sarek_intrinsic (div_int32 : int32 -> int32 -> int32) =
  {device = dev "(%s / %s)" "(%s / %s)"; ocaml = Stdlib.Int32.div}

let%sarek_intrinsic (mod_int32 : int32 -> int32 -> int32) =
  {device = dev "(%s %% %s)" "(%s %% %s)"; ocaml = Stdlib.Int32.rem}

(******************************************************************************
 * Bitwise operators
 ******************************************************************************)

let%sarek_intrinsic (logand : int32 -> int32 -> int32) =
  {device = dev "(%s & %s)" "(%s & %s)"; ocaml = Stdlib.Int32.logand}

let%sarek_intrinsic (logor : int32 -> int32 -> int32) =
  {device = dev "(%s | %s)" "(%s | %s)"; ocaml = Stdlib.Int32.logor}

let%sarek_intrinsic (logxor : int32 -> int32 -> int32) =
  {device = dev "(%s ^ %s)" "(%s ^ %s)"; ocaml = Stdlib.Int32.logxor}

let%sarek_intrinsic (lognot : int32 -> int32) =
  {device = dev "(~%s)" "(~%s)"; ocaml = Stdlib.Int32.lognot}

let%sarek_intrinsic (shift_left : int32 -> int -> int32) =
  {device = dev "(%s << %s)" "(%s << %s)"; ocaml = Stdlib.Int32.shift_left}

let%sarek_intrinsic (shift_right : int32 -> int -> int32) =
  {device = dev "(%s >> %s)" "(%s >> %s)"; ocaml = Stdlib.Int32.shift_right}

let%sarek_intrinsic (shift_right_logical : int32 -> int -> int32) =
  {
    device = dev "((unsigned int)%s >> %s)" "((unsigned int)%s >> %s)";
    ocaml = Stdlib.Int32.shift_right_logical;
  }

(******************************************************************************
 * Comparison / arithmetic
 ******************************************************************************)

let%sarek_intrinsic (abs : int32 -> int32) =
  {device = dev "abs(%s)" "abs(%s)"; ocaml = Stdlib.Int32.abs}

let%sarek_intrinsic (neg : int32 -> int32) =
  {device = dev "(-%s)" "(-%s)"; ocaml = Stdlib.Int32.neg}

let%sarek_intrinsic (min : int32 -> int32 -> int32) =
  {device = dev "min(%s, %s)" "min(%s, %s)"; ocaml = Stdlib.Int32.min}

let%sarek_intrinsic (max : int32 -> int32 -> int32) =
  {device = dev "max(%s, %s)" "max(%s, %s)"; ocaml = Stdlib.Int32.max}

(******************************************************************************
 * Conversion functions (pure OCaml; no Spoc_core dependency)
 ******************************************************************************)

let of_int x = Stdlib.Int32.of_int x

let to_int x = Stdlib.Int32.to_int x

let of_float x = Stdlib.Int32.of_float x

let to_float x = Stdlib.Int32.to_float x

let zero = 0l

let one = 1l

let minus_one = -1l
