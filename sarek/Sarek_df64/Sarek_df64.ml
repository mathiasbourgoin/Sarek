(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Sarek Df64 - double-float (float-float) extended precision
 *
 * Emulates ~2x float32 precision using unevaluated pairs of float32
 * (a "double-float" number x = hi + lo with |lo| <= ulp(hi)/2), targeting
 * devices without native float64 support and consumer GPUs where the
 * f64:f32 throughput ratio (1/32, 1/64) makes emulation competitive.
 *
 * The arithmetic is written in pure Sarek: every [@sarek.module] function
 * below is compiled to device code by the Sarek PPX and can be called from
 * [%kernel] bodies on ALL backends (CUDA/PTX, OpenCL, Vulkan, Metal,
 * Native, Interpreter).
 *
 * Usage from another compilation unit:
 *   - dune:   add sarek.df64 to (libraries)
 *             and this source to (preprocessor_deps) of the kernel's stanza
 *   - source: let%sarek_include _ = "path/to/Sarek_df64.ml"
 *             [%kernel fun ... -> ... df64_add x y ...]
 *   Host code uses Sarek_df64.df64 / df64_custom for vectors and
 *   Sarek_df64.Host to fill/read them.
 *
 * PRECISION CONTRACT
 *   - Relative error of add/sub/mul/div/sqrt: ~2^-48 .. 2^-47 (48-49
 *     significant bits), i.e. roughly double the 24-bit float32 precision.
 *   - Exponent range is that of float32 (~1e-38 .. ~3e38 normalised);
 *     overflow/underflow follow float32, NOT IEEE-754 binary64.
 *   - Requires IEEE-exact float32 ops and a correctly rounded float32 fma
 *     (used for exact TwoProd error extraction). The interpreter emulates
 *     float32 rounding (incl. Float.fma-based fma) and meets the contract.
 *   - Also requires that the backend compiler NOT contract a multiply into a
 *     neighbouring add or subtract. That is not the default anywhere, so the
 *     transformations below are written to make it impossible rather than to
 *     ask for it; see "Contraction barrier".
 *   - Results are NOT bit-exact with IEEE-754 binary64 arithmetic.
 *
 * PER-BACKEND STATUS
 *
 * Precision is a property of the backend COMPILER AND THE DEVICE, not of this
 * source. Every line below names the hardware and toolchain it was measured
 * on; a backend measured on one vendor says nothing about the same backend on
 * another. An earlier version of this header claimed "OpenCL, CUDA/PTX,
 * Interpreter: full contract" without naming hardware - that had only ever
 * been run on an AMD box (CUDA/PTX through ZLUDA, AMD OpenCL), and it is why
 * a real-NVIDIA collapse to ~2^-24 went unnoticed. Do not restate a result
 * more broadly than it was measured.
 *
 * Measured by sarek/tests/e2e/test_df64.ml unless noted:
 *
 *   Interpreter, sequential and parallel (any host)
 *       full contract (~2^-47). Emulates float32 rounding, incl. Float.fma.
 *
 *   OpenCL on AMD (RX 7900 XTX and Raphael iGPU, Mesa rusticl/radeonsi)
 *       full contract (~2^-47).
 *
 *   CUDA/PTX on NVIDIA Pascal (GTX 1070, sm_61, CUDA 12.9, driver 580.119.02)
 *       full contract expected after the contraction barrier below; the fix
 *       is verified in SASS offline (ptxas 12.9 -arch=sm_61 -O3) but has NOT
 *       yet been re-run on the device. Before the fix: mul 5.92e-08,
 *       div 5.64e-08, sqrt 2.88e-08 - i.e. plain float32.
 *
 *   CUDA/PTX on AMD through ZLUDA (RX 7900 XTX)
 *       full contract (~2^-47). NOT evidence for real NVIDIA hardware.
 *
 *   OpenCL on NVIDIA Pascal (GTX 1070)
 *       UNVERIFIED after the fix. Before the fix it showed the same
 *       ~2^-24 collapse as CUDA/PTX on that GPU.
 *
 *   Vulkan on AMD (RX 7900 XTX and Raphael iGPU, Mesa RADV)
 *       add/sub/sqrt meet the contract - float locals carry the GLSL
 *       [precise] qualifier - but mul/div degrade to ~2^-24 because RADV's
 *       GLSL [fma] is not correctly rounded, so TwoProd's error term is lost.
 *       Unfixed, and distinct from the ptxas contraction bug below.
 *
 *   Vulkan on NVIDIA Pascal (GTX 1070)
 *       meets the contract (mul 9.07e-15) - NVIDIA's GLSL [fma] is exact and
 *       [precise] blocks contraction. Measured before the fix.
 *
 *   Native (OCaml host code)
 *       evaluates float32 at OCaml binary64 precision, so the error-free
 *       transformations cancel and results degrade to plain f32 storage
 *       precision (~2^-24). Harmless in practice - Native has real float64 -
 *       but do not use df64 for extra precision there.
 *
 *   Metal, WGSL: UNTESTED.
 *
 * References:
 *   - T.J. Dekker, "A floating-point technique for extending the available
 *     precision", Numer. Math. 18, 1971.
 *   - D.E. Knuth, The Art of Computer Programming, vol. 2, 4.2.2.
 *   - Y. Hida, X.S. Li, D.H. Bailey, "Library for double-double and
 *     quad-double arithmetic" (QD), LBNL, 2000.
 *   - A. Thall, "Extended-precision floating-point numbers for GPU
 *     computation", SIGGRAPH 2006.
 ******************************************************************************)

[@@@warning "-32"]

type float32 = float

(** Double-float value: [hi + lo] with [|lo| <= ulp(hi)/2] when normalised. *)
type df64 = {hi : float32; lo : float32} [@@sarek.type]

(* OCaml-level bindings so the [@sarek.module] bodies below also compile as
   plain OCaml (inside kernels the same names resolve to GPU intrinsics).
   NOTE: the plain-OCaml versions run at double precision and are NOT a
   faithful float32 reference; use {!Host} for that. *)
let fma = Float.fma

let float x = float_of_int (Int32.to_int x)

(******************************************************************************
 * Contraction barrier
 *
 * An error-free transformation is only error-free if the backend compiler
 * evaluates it as written. Floating-point CONTRACTION - fusing a multiply into
 * a neighbouring add or subtract - is enabled by default in PTX, CUDA C and
 * OpenCL C, and it silently destroys these transformations.
 *
 * WHAT WAS MEASURED (sm_61 / GTX 1070, ptxas 12.9 -arch=sm_61 -O3, offline).
 * The [quick_two_sum p.hi err] that closes [df64_mul] compiled to
 *
 *     FFMA R11, a_hi,  b_hi, err    ; "s"       = fl(a_hi*b_hi + err)
 *     FFMA R0,  a_hi, -b_hi, R11    ; "s - p_hi"
 *
 * Both operands were rebuilt from the EXACT product a_hi*b_hi instead of the
 * rounded [p.hi] that [two_prod] had just separated from its error term. Since
 * [err] already carries that rounding error, adding it back doubles it: the
 * result is (true product + p.lo), and the relative error jumps from ~2^-47 to
 * ~2^-24 - plain float32. Simulating that instruction sequence in float32
 * reproduces the hardware failure exactly: 5.92e-08 measured, 5.92e-08
 * simulated, against 7.28e-15 for the same code uncontracted.
 *
 * The [fma] inside [two_prod] is NOT the casualty - ptxas keeps it and the
 * error term is correct. It is the surrounding add/sub that get fused. This is
 * also why add/sub survived while mul/div/sqrt collapsed: contraction needs a
 * multiply to fuse, and [df64_add]'s inputs are loads, not products.
 *
 * THE FIX. Deny the compiler the multiply. [mul_rn] computes the same product
 * through [fma], which is already fused and therefore cannot be fused again,
 * so every add and subtract downstream of a [two_prod] has nothing to contract
 * with. One instruction changes (FMUL -> FFMA, same rate on NVIDIA); the SASS
 * instruction count for df64_mul and df64_sqrt is unchanged at sm_61.
 *
 * WHY IT IS NOT APPLIED MORE WIDELY. Rewriting [two_sum] and [quick_two_sum]
 * in terms of [fma] as well was tried and MEASURED TO REGRESS RADV/Vulkan
 * (add 5.33e-15 -> 1.15e-07, sub 6.51e-15 -> 1.06e-07, sqrt 1.08e-14 ->
 * 8.17e-08 on RX 7900 XTX, Mesa RADV): RADV's GLSL [fma] is not correctly
 * rounded, which is exactly the pre-existing Vulkan mul/div deviation recorded
 * in the header. Confining the barrier to [two_prod] keeps the new dependency
 * inside the one transformation that already required a correct [fma], so no
 * backend is newly exposed.
 ******************************************************************************)

(** [a * b], correctly rounded, and - unlike [a *. b] - not a multiply the
    backend can fuse into a later add or subtract. Value-identical to [a *. b]
    except that a zero result is always [+0.0] (fma adds [+0.0]); nothing below
    depends on that sign. See the comment block above before changing this. *)
let[@sarek.module] mul_rn (a : float32) (b : float32) : float32 = fma a b 0.0

(******************************************************************************
 * Error-free transformations (Dekker 1971, Knuth TAOCP v2 4.2.2)
 ******************************************************************************)

(** Knuth TwoSum: [a + b] as an exact [hi + lo] pair. Branch-free, 6 flops. *)
let[@sarek.module] two_sum (a : float32) (b : float32) : df64 =
  let s = a +. b in
  let bb = s -. a in
  let err = a -. (s -. bb) +. (b -. bb) in
  {hi = s; lo = err}

(** Dekker QuickTwoSum: exact [hi + lo] pair, 3 flops. Requires [|a| >= |b|] (or
    [a = 0]). *)
let[@sarek.module] quick_two_sum (a : float32) (b : float32) : df64 =
  let s = a +. b in
  let err = b -. (s -. a) in
  {hi = s; lo = err}

(** Split-free TwoProd via fma: [a * b] as an exact [hi + lo] pair, 3 flops. *)
let[@sarek.module] two_prod (a : float32) (b : float32) : df64 =
  let p = mul_rn a b in
  let err = fma a b (0.0 -. p) in
  {hi = p; lo = err}

(******************************************************************************
 * df64 arithmetic (Hida/Li/Bailey QD, Thall 2006)
 ******************************************************************************)

(** Negation, 2 flops. *)
let[@sarek.module] df64_neg (a : df64) : df64 =
  {hi = 0.0 -. a.hi; lo = 0.0 -. a.lo}

(** Addition (Knuth-robust ieee_add), ~20 flops. *)
let[@sarek.module] df64_add (a : df64) (b : df64) : df64 =
  let s = two_sum a.hi b.hi in
  let t = two_sum a.lo b.lo in
  let r = quick_two_sum s.hi (s.lo +. t.hi) in
  quick_two_sum r.hi (r.lo +. t.lo)

(** Subtraction, ~22 flops. *)
let[@sarek.module] df64_sub (a : df64) (b : df64) : df64 =
  df64_add a (df64_neg b)

(** df64 + float32, ~10 flops. *)
let[@sarek.module] df64_add_f32 (a : df64) (b : float32) : df64 =
  let s = two_sum a.hi b in
  quick_two_sum s.hi (s.lo +. a.lo)

(** Multiplication, ~8 flops (2 of them fma). *)
let[@sarek.module] df64_mul (a : df64) (b : df64) : df64 =
  let p = two_prod a.hi b.hi in
  let err = fma a.hi b.lo (fma a.lo b.hi p.lo) in
  quick_two_sum p.hi err

(** df64 * float32, ~7 flops. *)
let[@sarek.module] df64_mul_f32 (a : df64) (b : float32) : df64 =
  let p = two_prod a.hi b in
  quick_two_sum p.hi (fma a.lo b p.lo)

(** Division (long division, 3 quotient digits), ~70 flops incl. 3 divides. *)
let[@sarek.module] df64_div (a : df64) (b : df64) : df64 =
  let q1 = a.hi /. b.hi in
  let r1 = df64_sub a (df64_mul_f32 b q1) in
  let q2 = r1.hi /. b.hi in
  let r2 = df64_sub r1 (df64_mul_f32 b q2) in
  let q3 = r2.hi /. b.hi in
  let q = quick_two_sum q1 q2 in
  df64_add_f32 q q3

(** Square root (Karp/Newton correction step), ~10 flops + 1 sqrt. Domain:
    [a >= 0]; returns 0 for non-positive [a.hi]. *)
let[@sarek.module] df64_sqrt (a : df64) : df64 =
  if a.hi <= 0.0 then {hi = 0.0; lo = 0.0}
  else
    let y = sqrt a.hi in
    let s = two_prod y y in
    let e = a.hi -. s.hi -. s.lo +. a.lo in
    quick_two_sum y (e /. (y +. y))

(** Absolute value. *)
let[@sarek.module] df64_abs (a : df64) : df64 =
  if a.hi < 0.0 then df64_neg a else {hi = a.hi; lo = a.lo}

(******************************************************************************
 * Comparisons (hi first, then lo; operands must be normalised)
 ******************************************************************************)

let[@sarek.module] df64_eq (a : df64) (b : df64) : bool =
  a.hi = b.hi && a.lo = b.lo

let[@sarek.module] df64_lt (a : df64) (b : df64) : bool =
  a.hi < b.hi || (a.hi = b.hi && a.lo < b.lo)

let[@sarek.module] df64_le (a : df64) (b : df64) : bool =
  a.hi < b.hi || (a.hi = b.hi && a.lo <= b.lo)

(******************************************************************************
 * Conversions
 ******************************************************************************)

let[@sarek.module] df64_of_float32 (x : float32) : df64 = {hi = x; lo = 0.0}

let[@sarek.module] df64_to_float32 (a : df64) : float32 = a.hi +. a.lo

(* OCaml-level shims so df64_of_int32's body compiles both as Sarek (where
   `/`, `*`, `-` operate on int32 and plain integer literals are int32) and
   as plain OCaml (where the same expressions mix Int32.t and int). Local
   to this point of the file - only df64_of_int32 below uses them. *)
let ( / ) a b = Int32.div a (Int32.of_int b)

let ( * ) a b = Int32.mul a (Int32.of_int b)

let ( - ) = Int32.sub

(** Exact int32 -> df64 conversion via 16-bit limb split (int32 has up to 31
    significant bits, float32 only 24; a df64 holds ~48). *)
let[@sarek.module] df64_of_int32 (i : int32) : df64 =
  let h16 = i / 65536 in
  let hs = h16 * 65536 in
  let l16 = i - hs in
  let s = two_prod (float h16) 65536.0 in
  df64_add_f32 s (float l16)

(******************************************************************************
 * Host-side companions
 *
 * Plain OCaml with explicit float32 rounding: use these to fill and read
 * df64 vectors from the host, and as a bit-faithful reference for the
 * device algorithms above (each *_ref mirrors its device twin op-for-op).
 ******************************************************************************)

module Host = struct
  (** Round an OCaml float (binary64) to the nearest float32 value. *)
  let round_f32 (x : float) : float =
    Int32.float_of_bits (Int32.bits_of_float x)

  (** Dekker split of a binary64 value into a df64 pair at f32 precision: [hi]
      carries the leading 24 bits, [lo] the next 24. *)
  let encode (x : float) : df64 =
    let hi = round_f32 x in
    let lo = round_f32 (x -. hi) in
    {hi; lo}

  (** Back to binary64: exact since both halves are f32 values. *)
  let decode (a : df64) : float = a.hi +. a.lo

  (* float32-emulated scalar ops *)
  let ( +% ) a b = round_f32 (a +. b)

  let ( -% ) a b = round_f32 (a -. b)

  let ( *% ) a b = round_f32 (a *. b)

  let ( /% ) a b = round_f32 (a /. b)

  let fma_f32 a b c = round_f32 (Float.fma a b c)

  let sqrt_f32 a = round_f32 (sqrt a)

  let two_sum a b =
    let s = a +% b in
    let bb = s -% a in
    {hi = s; lo = a -% (s -% bb) +% (b -% bb)}

  let quick_two_sum a b =
    let s = a +% b in
    {hi = s; lo = b -% (s -% a)}

  (* [a *% b] rather than the device's [mul_rn a b]: the contraction barrier is
     a codegen concern, and OCaml never contracts. The two agree on every
     value except a zero product's sign. *)
  let two_prod a b =
    let p = a *% b in
    {hi = p; lo = fma_f32 a b (-.p)}

  let neg a = {hi = -.a.hi; lo = -.a.lo}

  let add a b =
    let s = two_sum a.hi b.hi in
    let t = two_sum a.lo b.lo in
    let r = quick_two_sum s.hi (s.lo +% t.hi) in
    quick_two_sum r.hi (r.lo +% t.lo)

  let sub a b = add a (neg b)

  let add_f32 a b =
    let s = two_sum a.hi b in
    quick_two_sum s.hi (s.lo +% a.lo)

  let mul a b =
    let p = two_prod a.hi b.hi in
    let err = fma_f32 a.hi b.lo (fma_f32 a.lo b.hi p.lo) in
    quick_two_sum p.hi err

  let mul_f32 a b =
    let p = two_prod a.hi b in
    quick_two_sum p.hi (fma_f32 a.lo b p.lo)

  let div a b =
    let q1 = a.hi /% b.hi in
    let r1 = sub a (mul_f32 b q1) in
    let q2 = r1.hi /% b.hi in
    let r2 = sub r1 (mul_f32 b q2) in
    let q3 = r2.hi /% b.hi in
    add_f32 (quick_two_sum q1 q2) q3

  let sqrt a =
    if a.hi <= 0.0 then {hi = 0.0; lo = 0.0}
    else
      let y = sqrt_f32 a.hi in
      let s = two_prod y y in
      let e = a.hi -% s.hi -% s.lo +% a.lo in
      quick_two_sum y (e /% (y +% y))
end
