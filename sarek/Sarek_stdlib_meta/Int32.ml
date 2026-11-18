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

(* CUDA C, OpenCL C and MSL spellings of one intrinsic. The wrapping
   operations below compute through the unsigned type and come back to the
   signed one by a cast in CUDA C and by a bit reinterpretation in OpenCL C
   ([as_int], [as_long]) and MSL ([as_type]), as the operator emitters do
   (Sarek_ir_c_arith). OpenCL C's [abs] returns an unsigned type, hence the
   reinterpretation there too. *)
let dev3 cuda opencl metal d = if d = "Metal" then metal else dev cuda opencl d

(* [add], [sub], [mul], [neg] and [shift_left] wrap as in OCaml, with the same
   templates as Sarek_stdlib (Sarek_ir_c_arith); test_int32_bitcasts_codegen
   checks the two registrations agree. *)

let%sarek_intrinsic (add_int32 : int32 -> int32 -> int32) =
  {
    device =
      dev3
        "((int)((unsigned int)(%s) + (unsigned int)(%s)))"
        "as_int(((uint)(%s) + (uint)(%s)))"
        "as_type<int>(((uint)(%s) + (uint)(%s)))";
    ocaml = Stdlib.Int32.add;
  }

let%sarek_intrinsic (sub_int32 : int32 -> int32 -> int32) =
  {
    device =
      dev3
        "((int)((unsigned int)(%s) - (unsigned int)(%s)))"
        "as_int(((uint)(%s) - (uint)(%s)))"
        "as_type<int>(((uint)(%s) - (uint)(%s)))";
    ocaml = Stdlib.Int32.sub;
  }

let%sarek_intrinsic (mul_int32 : int32 -> int32 -> int32) =
  {
    device =
      dev3
        "((int)((unsigned int)(%s) * (unsigned int)(%s)))"
        "as_int(((uint)(%s) * (uint)(%s)))"
        "as_type<int>(((uint)(%s) * (uint)(%s)))";
    ocaml = Stdlib.Int32.mul;
  }

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
  {
    device =
      dev3
        "((int)((unsigned int)(%s) << (%s)))"
        "as_int(((uint)(%s) << (%s)))"
        "as_type<int>(((uint)(%s) << (%s)))";
    ocaml = Stdlib.Int32.shift_left;
  }

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
  {
    device = dev3 "abs(%s)" "as_int(abs(%s))" "abs(%s)";
    ocaml = Stdlib.Int32.abs;
  }

let%sarek_intrinsic (neg : int32 -> int32) =
  {
    device =
      dev3
        "((int)(-(unsigned int)(%s)))"
        "as_int((-(uint)(%s)))"
        "as_type<int>((-(uint)(%s)))";
    ocaml = Stdlib.Int32.neg;
  }

let%sarek_intrinsic (min : int32 -> int32 -> int32) =
  {device = dev "min(%s, %s)" "min(%s, %s)"; ocaml = Stdlib.Int32.min}

let%sarek_intrinsic (max : int32 -> int32 -> int32) =
  {device = dev "max(%s, %s)" "max(%s, %s)"; ocaml = Stdlib.Int32.max}

(******************************************************************************
 * Bit reinterpretation (float32 <-> int32)
 *
 * Same names, argument order and semantics as OCaml's
 * [Stdlib.Int32.float_of_bits] / [Stdlib.Int32.bits_of_float]: the 32 bits are
 * reused unchanged, there is no numeric conversion. A device moves the bits
 * through a register. Measured (sarek/tests/e2e/test_int32_bitcasts.ml,
 * 2026-10-10): an int32 -> float32 -> int32 round trip, both inside one
 * kernel and through a float32 device buffer across two launches, returned
 * every tested pattern unchanged - +-0, +-inf, denormals, max finite, a
 * quiet-NaN payload and a signalling NaN - on HIP, OpenCL, Vulkan and
 * CUDA/PTX (ZLUDA) on an RX 7900 XTX and a Ryzen 7950X iGPU, and on Metal and
 * OpenCL on an Apple M4. GLSL and WGSL themselves leave the value of a
 * non-finite bitcast unspecified, and WGSL may refuse a constant-folded
 * bitcast that yields NaN or inf at shader creation.
 *
 * NaN caveat (host side only): Native and the Interpreter carry a float32 as an
 * OCaml double, so [float_of_bits] widens to double. Quiet-NaN payloads survive
 * the f32 -> f64 -> f32 round trip; a signalling NaN is quieted by that
 * widening (bit 22 set), as it is by any host read of a float32 vector.
 *
 * Spelling: the C-family backends read the template below; MSL has neither
 * CUDA's [__int_as_float] nor OpenCL C's [as_float], so Metal gets its own
 * [as_type<T>] branch instead of the [cuda_or_opencl] Metal->OpenCL mapping.
 * GLSL (intBitsToFloat/floatBitsToInt), WGSL (bitcast<f32>/bitcast<i32>) and
 * PTX (mov.b32) lower these two names natively; [cuda_or_opencl] still refuses
 * the shading languages here.
 ******************************************************************************)

let bitcast_dev cuda opencl metal d =
  if d = "Metal" then metal else dev cuda opencl d

let%sarek_intrinsic (float_of_bits : int32 -> float32) =
  {
    device = bitcast_dev "__int_as_float(%s)" "as_float(%s)" "as_type<float>(%s)";
    ocaml = Stdlib.Int32.float_of_bits;
  }

let%sarek_intrinsic (bits_of_float : float32 -> int32) =
  {
    device = bitcast_dev "__float_as_int(%s)" "as_int(%s)" "as_type<int>(%s)";
    ocaml = Stdlib.Int32.bits_of_float;
  }

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
