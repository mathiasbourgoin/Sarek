(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Sarek Float16 - IEEE binary16 rounding for the interpreter / native paths
 *
 * The f16 sibling of {!Sarek_float32}, and deliberately much smaller.
 *
 * f16 in Sarek is a STORAGE type: values are stored as binary16, arithmetic is
 * performed in f32, and the result is rounded back to binary16 on store. So
 * unlike Sarek_float32 there is no f16 arithmetic or math-intrinsic surface
 * here -- the ONLY operation f16 needs is "narrow this f32 value to binary16",
 * which is what makes the interpreter agree with a GPU that stores through a
 * `__half` cell.
 ******************************************************************************)

(** Round a value to IEEE binary16 precision.

    Implemented by a round-trip through a 1-element [Bigarray.Float16] array
    rather than by hand-rolled bit twiddling. This is deliberate:
    [Bigarray.Array1.set] on a [Float16] array performs the platform's
    round-to-nearest-even narrowing, including the subnormal and
    overflow-to-infinity edge cases that a hand-written version gets wrong, and
    it is the SAME code path that an f16 [Vector.set] uses. Sharing the
    narrowing implementation with the storage path is what makes
    "[ECast (TFloat16, e)] then store" and "store" agree bit-for-bit.

    A fresh 1-element array is allocated per call so the function is reentrant
    (no shared scratch cell across domains). The interpreter is a reference
    oracle, not a throughput path, so this is the right trade. *)
let to_float16 (x : float) : float =
  let scratch = Bigarray.Array1.create Bigarray.Float16 Bigarray.c_layout 1 in
  Bigarray.Array1.unsafe_set scratch 0 x ;
  Bigarray.Array1.unsafe_get scratch 0

(** Alias reading naturally at conversion sites. *)
let of_float = to_float16

(** Largest finite binary16 value (65504.0). Values above this round to
    infinity. *)
let max_float16 = 65504.0

(** Smallest positive NORMAL binary16 value (2^-14). Smaller magnitudes are
    representable only as subnormals, down to [min_positive_subnormal]. *)
let min_positive_float16 = 6.103515625e-05

(** Smallest positive binary16 subnormal (2^-24). *)
let min_positive_subnormal = 5.960464477539063e-08

(** Unit roundoff: 2^-11, the gap between 1.0 and the next binary16 value is
    2^-10, so the relative rounding error is bounded by 2^-11. Tests comparing
    an f16 result against an f32/f64 reference should use this, not an f32
    epsilon. *)
let epsilon_float16 = 4.8828125e-04
