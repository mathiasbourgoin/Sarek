(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Software f64 transcendentals — backend-shared (PTX and GLSL/Vulkan).

    PTX has no f64 transcendental instructions (sin/cos/ex2/lg2 exist only as
    f32 [.approx] ops), and GLSL core has no f64 overload for any transcendental
    builtin. This module provides polynomial implementations as pure IR
    [helper_func] bodies, built from operations every such backend can express:
    add/sub/mul/div, fma, floor, min, abs, copysign, and the [f64_bits]/
    [bits_f64] bitcasts. Because the bodies are pure [Sarek_ir_types] IR (no
    backend assembly), each emitter reuses them by routing
    [EIntrinsic (["Float64"], name, args)] through its own machinery:
    - PTX ([Sarek_ir_ptx_expr]) registers them into its helper table and inlines
      via the EApp machinery;
    - GLSL ([Sarek_ir_glsl]) emits the needed family as top-level functions
      (forward-declared) gated per-kernel, with [f64_bits]/[bits_f64] lowered to
      [doubleBitsToInt64]/[int64BitsToDouble] under [GL_ARB_gpu_shader_int64].
      Neither emits assembly-level polynomials.

    Numerics: precise tier, max relative error ≤ 1e-12 on the documented domains
    (measured, see sarek/tests/unit/test_f64_softmath.ml).

    Coefficient provenance: the sin/cos/atan/asin/acos/expm1/log1p kernel
    polynomials and the Cody-Waite π/2 and ln2 hi/lo splits are the published
    fdlibm 5.3 constants (Sun Microsystems, freely-redistributable license; the
    same values appear in Cephes and glibc). The exp and atanh-based log
    polynomials are plain Taylor coefficients — on the reduced domains (|r| ≤
    ln2/2 for exp, z = s² ≤ 0.0295 for log) truncation error is below ~1e-14
    relative, so minimax refinement is unnecessary at this tier.

    Documented domains (graceful degradation outside, no Payne-Hanek):
    - sin/cos/tan: |x| ≤ 1e6 (3-term Cody-Waite reduction);
    - exp: x ∈ [-708, 709] (2^n scaling via exponent-field construction; no
      gradual-underflow handling below -708);
    - log/log10: finite normal x > 0;
    - pow: x > 0 (log of a negative is NaN; integer-exponent negative bases are
      not handled);
    - sinh/cosh: |x| ≤ 708 (via exp(|x|)); tanh: all finite x (argument clamped,
      saturates to ±1);
    - asin/acos: [-1, 1]; atan: all finite x (large |x| through the −1/x
      branch);
    - atan2: finite (y, x), x = −0 treated as +0, inf/nan unspecified;
    - expm1: x ∈ [-708, 709]; log1p: finite x > −1 (both keep full relative
      precision near 0 — that is what they exist for). *)

open Sarek_ir_types

(** {1 IR-building DSL} *)

let f64 c = EConst (CFloat64 c)

let i32 n = EConst (CInt32 (Int32.of_int n))

let i64 n = EConst (CInt64 n)

let vid = ref 0

(** Fresh variable (ids only need to be distinct within one helper body). *)
let mkvar ty name =
  incr vid ;
  {var_name = name; var_id = !vid; var_type = ty; var_mutable = false}

let fvar = mkvar TFloat64

let ivar = mkvar TInt32

let lvar = mkvar TInt64

let ( +! ) a b = EBinop (Add, a, b)

let ( -! ) a b = EBinop (Sub, a, b)

let ( *! ) a b = EBinop (Mul, a, b)

let ( /! ) a b = EBinop (Div, a, b)

let ( >! ) a b = EBinop (Gt, a, b)

let ( <! ) a b = EBinop (Lt, a, b)

let ( =! ) a b = EBinop (Eq, a, b)

(* NOTE on precedence: [&!] and [|!] start with '&'/'|' so OCaml parses them
   at comparison level, BELOW [+!]/[*!] — [a +! b &! c] is [(a+b) & c]. *)
let ( &! ) a b = EBinop (BitAnd, a, b)

let ( |! ) a b = EBinop (BitOr, a, b)

let shr e n = EBinop (Shr, e, i32 n)

let shl e n = EBinop (Shl, e, i32 n)

let neg e = EUnop (Neg, e)

let fma a b c = EIntrinsic ([], "fma", [a; b; c])

let floor_ e = EIntrinsic ([], "floor", [e])

let abs_ e = EIntrinsic ([], "fabs", [e])

let min_ a b = EIntrinsic ([], "min", [a; b])

(* Native sqrt.rn.f64 — correctly rounded, no software emulation needed. *)
let sqrt_ e = EIntrinsic ([], "sqrt", [e])

(** [copysign_ mag sgn] = |mag| with [sgn]'s sign (OCaml argument order). *)
let copysign_ mag sgn = EIntrinsic ([], "copysign", [mag; sgn])

let bits e = EIntrinsic ([], "f64_bits", [e])

let unbits e = EIntrinsic ([], "bits_f64", [e])

let to_i32 e = ECast (TInt32, e)

let to_i64 e = ECast (TInt64, e)

let to_f64 e = ECast (TFloat64, e)

let let_ v e body = SLet (v, e, body)

(** Call another softmath helper (inlined by the emitter's EApp machinery). *)
let call name arg_exprs = EApp (EVar (mkvar TFloat64 name), arg_exprs)

(** Horner evaluation with fma: [coeffs] highest-degree first, [last] is the
    constant term: p = (…(c₀·x + c₁)·x + …)·x + last. *)
let horner x ~coeffs ~last =
  let p =
    match coeffs with
    | [] -> None
    | c0 :: rest ->
        Some (List.fold_left (fun p c -> fma p x (f64 c)) (f64 c0) rest)
  in
  match p with None -> f64 last | Some p -> fma p x (f64 last)

(** {1 Constants (fdlibm 5.3 unless noted)} *)

let ln2_hi = 6.93147180369123816490e-01

let ln2_lo = 1.90821492927058770002e-10

let log2_e = 1.44269504088896338700e+00

let log10_e = 4.34294481903251827651e-01

(* 2/π and the 3-term Cody-Waite split of π/2 (fdlibm __ieee754_rem_pio2:
   pio2_1, pio2_2, pio2_2t). pio2_1 and pio2_2 are 33-bit truncations so that
   j·pio2_1 and j·pio2_2 are EXACT for |j| < 2^20; pio2_3 (fdlibm's pio2_2t)
   is the full-precision tail. π/2 − (pio2_1+pio2_2+pio2_3) ≈ 2.4e-37. *)
let inv_pio2 = 6.36619772367581382433e-01

let pio2_1 = 1.57079632673412561417e+00

let pio2_2 = 6.07710050630396597660e-11

let pio2_3 = 2.02226624879595063154e-21

(* Mantissa-normalization pivot for log/log1p: nearest double to √2. *)
let sqrt2 = 1.41421356237309514547

(* fdlibm __kernel_sin: sin(r) = r + r·z·(S1 + z·(S2 + … z·S6)), z = r²,
   |r| ≤ π/4. Highest degree first for Horner. *)
let sin_coeffs =
  [
    1.58969099521155010221e-10 (* S6 *);
    -2.50507602534068634195e-08 (* S5 *);
    2.75573137070700676789e-06 (* S4 *);
    -1.98412698298579493134e-04 (* S3 *);
    8.33333333332248946124e-03 (* S2 *);
  ]

let sin_s1 = -1.66666666666666324348e-01

(* fdlibm __kernel_cos: cos(r) = 1 - z/2 + z²·(C1 + z·(C2 + … z·C6)). *)
let cos_coeffs =
  [
    -1.13596475577881948265e-11 (* C6 *);
    2.08757232129817482790e-09 (* C5 *);
    -2.75573143513906633035e-07 (* C4 *);
    2.48015872894767294178e-05 (* C3 *);
    -1.38888888888741095749e-03 (* C2 *);
  ]

let cos_c1 = 4.16666666666666019037e-02

(* fdlibm s_atan.c: atanhi/atanlo = atan(0.5), atan(1), atan(1.5), atan(inf)
   as hi/lo pairs, and the degree-11 odd-polynomial coefficients aT. *)
let atan_hi =
  [|
    4.63647609000806093515e-01;
    7.85398163397448278999e-01;
    9.82793723247329054082e-01;
    1.57079632679489655800e+00;
  |]

let atan_lo =
  [|
    2.26987774529616870924e-17;
    3.06161699786838301793e-17;
    1.39033110312309984516e-17;
    6.12323399573676603587e-17;
  |]

let atan_at =
  [|
    3.33333333333329318027e-01 (* aT0 *);
    -1.99999999998764832476e-01;
    1.42857142725034663711e-01;
    -1.11111104054623557880e-01;
    9.09088713343650656196e-02;
    -7.69187620504482999495e-02;
    6.66107313738753120669e-02;
    -5.83357013379057348645e-02;
    4.97687799461593236017e-02;
    -3.65315727442169155270e-02;
    1.62858201153657823623e-02 (* aT10 *);
  |]

(* fdlibm e_asin.c / e_acos.c: shared rational R(t) = p(t)/q(t) coefficients
   (pS0..pS5 / qS1..qS4) and the π/2, π/4 hi/lo splits. *)
let pio2_hi = 1.57079632679489655800e+00

let pio2_lo = 6.12323399573676603587e-17

let pio4_hi = 7.85398163397448278999e-01

let pi = 3.14159265358979311600e+00

let asin_ps0 = 1.66666666666666657415e-01

let asin_p_coeffs =
  [
    3.47933107596021167570e-05 (* pS5 *);
    7.91534994289814532176e-04;
    -4.00555345006794114027e-02;
    2.01212532134862925881e-01;
    -3.25565818622400915405e-01 (* pS1 *);
  ]

let asin_q_coeffs =
  [
    7.70381505559019352791e-02 (* qS4 *);
    -6.88283971605453293030e-01;
    2.02094576023350569471e+00;
    -2.40339491173441421878e+00 (* qS1 *);
  ]

(* fdlibm s_expm1.c: rational-approximation coefficients Q1..Q5. *)
let expm1_q1 = -3.33333333333331316428e-02

let expm1_q_coeffs =
  [
    -2.01099218183624371326e-07 (* Q5 *);
    4.00821782732936239552e-06;
    -7.93650757867487942473e-05;
    1.58730158725481460165e-03 (* Q2 *);
  ]

(* fdlibm s_log1p.c: polynomial coefficients Lp1..Lp7. *)
let log1p_lp1 = 6.666666666666735130e-01

let log1p_lp_coeffs =
  [
    1.479819860511658591e-01 (* Lp7 *);
    1.531383769920937332e-01;
    1.818357216161805012e-01;
    2.222219843214978396e-01;
    2.857142874366239149e-01;
    3.999999999940941908e-01 (* Lp2 *);
  ]

let inv_fact k =
  let rec fact i acc =
    if i = 0 then acc else fact (i - 1) (acc *. float_of_int i)
  in
  1.0 /. fact k 1.0

(** {1 Helper bodies} *)

(* Saturation bounds for the 2^n exponent-field construction: n must stay in
   [-1022, 1024] or (n + 1023) << 52 wraps into the sign/exponent bits and
   returns garbage instead of 0/inf (audit finding M3). log(max_float) =
   709.7827...; below -708 the true result is subnormal and this tier
   flushes to zero (documented: no gradual underflow). *)

(** exp(x): n = rint(x·log2 e) (as floor(·+0.5)); r = x − n·ln2 (hi/lo split, 2
    fma); degree-12 Taylor on r ∈ [−ln2/2, ln2/2] (truncation ≈ 1.7e-16
    relative); scale by 2^n via exponent-field construction (n+1023) << 52. *)
let exp_hi_cut = 709.782712893384

let exp_lo_cut = -708.0

let exp_body x =
  let nf = fvar "nf" in
  let r_hi = fvar "r_hi" in
  let r = fvar "r" in
  let p = fvar "p" in
  let n = ivar "n" in
  (* c12 … c3, then c2 = 1/2 as the Horner constant term. *)
  let coeffs = List.init 10 (fun i -> inv_fact (12 - i)) in
  let core =
    let_ nf (floor_ (fma (EVar x) (f64 log2_e) (f64 0.5)))
    @@ let_ r_hi (fma (EVar nf) (f64 (-.ln2_hi)) (EVar x))
    @@ let_ r (fma (EVar nf) (f64 (-.ln2_lo)) (EVar r_hi))
    (* exp(r) = 1 + r·(1 + r·(1/2 + r/6 + …)) *)
    @@ let_
         p
         (fma (horner (EVar r) ~coeffs ~last:(inv_fact 2)) (EVar r) (f64 1.0))
    @@ let_ n (to_i32 (EVar nf))
    @@ SReturn
         (fma (EVar p) (EVar r) (f64 1.0)
         *! unbits (shl (to_i64 (EVar n +! i32 1023)) 52))
  in
  SIf
    ( EVar x <! f64 exp_lo_cut,
      SReturn (f64 0.0),
      Some (SIf (EVar x >! f64 exp_hi_cut, SReturn (f64 infinity), Some core))
    )

(** log(x): exponent extract + mantissa normalized to [√2/2, √2), then
    log(m) = 2·atanh(s) with s = (m−1)/(m+1): odd Taylor to s¹⁵ (truncation
    ≤ 3.3e-14 of the atanh term); log(x) = k·ln2_hi + (log(m) + k·ln2_lo). *)
let log_body x =
  let b = lvar "b" in
  let k_raw = ivar "k_raw" in
  let m0 = fvar "m0" in
  let big = mkvar TBool "big" in
  let m = fvar "m" in
  let k = ivar "k" in
  let s = fvar "s" in
  let z = fvar "z" in
  let lm = fvar "lm" in
  let kf = fvar "kf" in
  let atanh_q =
    horner
      (EVar z)
      ~coeffs:[1. /. 15.; 1. /. 13.; 1. /. 11.; 1. /. 9.; 1. /. 7.; 1. /. 5.]
      ~last:(1. /. 3.)
  in
  let_ b (bits (EVar x))
  @@ let_ k_raw (to_i32 (shr (EVar b) 52 &! i64 0x7FFL))
  @@ let_
       m0
       (unbits (EVar b &! i64 0xFFFFFFFFFFFFFL |! i64 0x3FF0000000000000L))
  @@ let_ big (EVar m0 >! f64 sqrt2)
  @@ let_ m (EIf (EVar big, EVar m0 *! f64 0.5, EVar m0))
  @@ let_ k (EIf (EVar big, EVar k_raw -! i32 1022, EVar k_raw -! i32 1023))
  @@ let_ s ((EVar m -! f64 1.0) /! (EVar m +! f64 1.0))
  @@ let_ z (EVar s *! EVar s)
  (* log(m) = 2s + 2s·z·(1/3 + z/5 + …) *)
  @@ let_ lm (fma ((EVar s +! EVar s) *! EVar z) atanh_q (EVar s +! EVar s))
  @@ let_ kf (to_f64 (EVar k))
  @@ SReturn (fma (EVar kf) (f64 ln2_hi) (fma (EVar kf) (f64 ln2_lo) (EVar lm)))

(** sin/cos share one reduction: j = rint(x·2/π), 3-term Cody-Waite r, quadrant
    q = (n + shift) & 3 (shift 0 = sin, 1 = cos), then select between the fdlibm
    sin (odd, degree 13) and cos (even, degree 14) kernels. *)
let trig_body ~shift x =
  let j = fvar "j" in
  let r0 = fvar "r0" in
  let r1 = fvar "r1" in
  let r = fvar "r" in
  let n = ivar "n" in
  let q = ivar "q" in
  let z = fvar "z" in
  let sin_r = fvar "sin_r" in
  let cos_r = fvar "cos_r" in
  let_ j (floor_ (fma (EVar x) (f64 inv_pio2) (f64 0.5)))
  @@ let_ r0 (fma (EVar j) (f64 (-.pio2_1)) (EVar x))
  @@ let_ r1 (fma (EVar j) (f64 (-.pio2_2)) (EVar r0))
  @@ let_ r (fma (EVar j) (f64 (-.pio2_3)) (EVar r1))
  @@ let_ n (to_i32 (EVar j))
  @@ let_ q (EVar n +! i32 shift &! i32 3)
  @@ let_ z (EVar r *! EVar r)
  @@ let_
       sin_r
       (fma
          (EVar r *! EVar z)
          (horner (EVar z) ~coeffs:sin_coeffs ~last:sin_s1)
          (EVar r))
  @@ let_
       cos_r
       (fma
          (EVar z *! EVar z)
          (horner (EVar z) ~coeffs:cos_coeffs ~last:cos_c1)
          (fma (EVar z) (f64 (-0.5)) (f64 1.0)))
  @@ SReturn
       (EIf
          ( EVar q =! i32 0,
            EVar sin_r,
            EIf
              ( EVar q =! i32 1,
                EVar cos_r,
                EIf (EVar q =! i32 2, neg (EVar sin_r), neg (EVar cos_r)) ) ))

(** tan = sin/cos (each via its own identical reduction; div.rn is correctly
    rounded so the quotient adds ≤ 1 ulp). *)
let tan_body x =
  SReturn (call "__sarek_f64_sin" [EVar x] /! call "__sarek_f64_cos" [EVar x])

(** log10(x) = log(x)·log10(e). *)
let log10_body x = SReturn (call "__sarek_f64_log" [EVar x] *! f64 log10_e)

(** pow(x, y) = exp(y·log x) for x > 0 (documented domain). The unsplit product
    y·log x caps accuracy at ~|y·log x| ulps — fine for moderate exponents,
    degrades as |y·log x| grows. *)
let pow_body x y =
  SReturn (call "__sarek_f64_exp" [EVar y *! call "__sarek_f64_log" [EVar x]])

(** sinh(x) = sign(x)·(e^|x| − e^−|x|)/2, with an odd-Taylor branch for |x| <
    2⁻⁵ where the subtraction would cancel. *)
let sinh_body x =
  let a = fvar "a" in
  let z = fvar "z" in
  let t = fvar "t" in
  let taylor =
    fma
      (EVar x *! EVar z)
      (horner (EVar z) ~coeffs:[1. /. 5040.; 1. /. 120.] ~last:(1. /. 6.))
      (EVar x)
  in
  let_ a (abs_ (EVar x))
  @@ let_ z (EVar x *! EVar x)
  @@ SIf
       ( EVar a <! f64 0.03125,
         SReturn taylor,
         Some
           (let_ t (call "__sarek_f64_exp" [EVar a])
           @@ SReturn
                (copysign_
                   ((EVar t -! (f64 1.0 /! EVar t)) *! f64 0.5)
                   (EVar x))) )

(** cosh(x) = (e^|x| + e^−|x|)/2 (no cancellation; e^−|x| as 1/e^|x| costs ≤ 1
    ulp on a term that is ≤ the dominant one). *)
let cosh_body x =
  let t = fvar "t" in
  let_ t (call "__sarek_f64_exp" [abs_ (EVar x)])
  @@ SReturn ((EVar t +! (f64 1.0 /! EVar t)) *! f64 0.5)

(** tanh(x) = copysign(1 − 2/(e^(2|x|)+1), x); the argument of exp is clamped at
    40 (tanh is ±1 to double precision beyond |x| ≈ 19, and the clamp keeps exp
    finite), with an odd-Taylor branch for |x| < 2⁻⁵ against cancellation. *)
let tanh_body x =
  let a = fvar "a" in
  let z = fvar "z" in
  let t = fvar "t" in
  let taylor =
    fma
      (EVar x *! EVar z)
      (horner (EVar z) ~coeffs:[-17. /. 315.; 2. /. 15.] ~last:(-1. /. 3.))
      (EVar x)
  in
  let_ a (abs_ (EVar x))
  @@ let_ z (EVar x *! EVar x)
  @@ SIf
       ( EVar a <! f64 0.03125,
         SReturn taylor,
         Some
           (let_ t (call "__sarek_f64_exp" [min_ (EVar a +! EVar a) (f64 40.0)])
           @@ SReturn
                (copysign_
                   (f64 1.0 -! (f64 2.0 /! (EVar t +! f64 1.0)))
                   (EVar x))) )

(** atan(x): fdlibm 5.3 s_atan.c — |x| reduced against the 7/16, 11/16, 19/16,
    39/16 breakpoints to t (|t| ≲ 0.46), degree-11 odd polynomial split into
    even/odd Horner halves (aT), result hi − ((p − lo) − t) with the
    atan(0.5)/atan(1)/atan(1.5)/(π/2) hi/lo tables (hi = lo = 0 on the
    no-reduction branch). Large |x| flows through the −1/x branch, so fdlibm's
    2^66 cutoff is unnecessary. *)
let atan_body x =
  let a = fvar "a" in
  let t = fvar "t" in
  let hi = fvar "hi" in
  let lo = fvar "lo" in
  let z = fvar "z" in
  let w = fvar "w" in
  let p = fvar "p" in
  (* 5-way select on the reduction interval of |x| (selp chains — all branches
     are computed; every divisor below is bounded away from 0 on its lane). *)
  let pick v0 v1 v2 v3 v4 =
    let br lim y n = EIf (EVar a <! f64 lim, y, n) in
    br 0.4375 v0 (br 0.6875 v1 (br 1.1875 v2 (br 2.4375 v3 v4)))
  in
  let tbl c = pick (f64 0.0) (f64 c.(0)) (f64 c.(1)) (f64 c.(2)) (f64 c.(3)) in
  let even = [atan_at.(10); atan_at.(8); atan_at.(6); atan_at.(4); atan_at.(2)]
  and odd = [atan_at.(9); atan_at.(7); atan_at.(5); atan_at.(3)] in
  let_ a (abs_ (EVar x))
  @@ let_
       t
       (pick
          (EVar a)
          ((EVar a +! EVar a -! f64 1.0) /! (f64 2.0 +! EVar a))
          ((EVar a -! f64 1.0) /! (EVar a +! f64 1.0))
          ((EVar a -! f64 1.5) /! fma (f64 1.5) (EVar a) (f64 1.0))
          (f64 (-1.0) /! EVar a))
  @@ let_ hi (tbl atan_hi)
  @@ let_ lo (tbl atan_lo)
  @@ let_ z (EVar t *! EVar t)
  @@ let_ w (EVar z *! EVar z)
  (* p = t·(s1 + s2), s1 = z·E(w), s2 = w·O(w) (fdlibm's split evaluation). *)
  @@ let_
       p
       (EVar t
       *! ((EVar z *! horner (EVar w) ~coeffs:even ~last:atan_at.(0))
          +! (EVar w *! horner (EVar w) ~coeffs:odd ~last:atan_at.(1))))
  @@ SReturn (copysign_ (EVar hi -! (EVar p -! EVar lo -! EVar t)) (EVar x))

(** atan2(y, x): fdlibm 5.3 e_atan2.c quadrant logic over atan(y/x), collapsed
    to the single x < 0 fixup atan2(y,x) = atan(y/x) + copysign(π, y); for x > 0
    (and x = +0 via atan(±∞) = ±π/2) atan(y/x) is already the answer. Only the π
    hi part is added (the π lo tail is ≪ 1e-12 relative on results of magnitude
    ≥ π/2). x = −0 is treated as +0; inf/inf and nan are unspecified (outside
    the documented domain). *)
let atan2_body y x =
  let z = fvar "z" in
  let_ z (call "__sarek_f64_atan" [EVar y /! EVar x])
  @@ SReturn
       (EIf (EVar x <! f64 0.0, copysign_ (f64 pi) (EVar y) +! EVar z, EVar z))

(** asin(x): fdlibm 5.3 e_asin.c — |x| < 0.5: x + x·R(x²) with the rational R =
    p/q; 0.5 ≤ |x| ≤ 0.975: π/4-anchored form with the low-word-zeroed hi part
    of s = sqrt((1−|x|)/2) and its exact-square correction c; |x| > 0.975 (incl.
    1): π/2 − 2·(s + s·R). *)
let asin_body x =
  let a = fvar "a" in
  let t = fvar "t" in
  let p = fvar "p" in
  let q = fvar "q" in
  let s = fvar "s" in
  let ws = fvar "ws" in
  let c = fvar "c" in
  let p2 = fvar "p2" in
  let q2 = fvar "q2" in
  let pnum z = z *! horner z ~coeffs:asin_p_coeffs ~last:asin_ps0 in
  let qden z = horner z ~coeffs:asin_q_coeffs ~last:1.0 in
  let_ a (abs_ (EVar x))
  @@ SIf
       ( EVar a <! f64 0.5,
         let_ t (EVar x *! EVar x)
         @@ SReturn (fma (EVar x) (pnum (EVar t) /! qden (EVar t)) (EVar x)),
         Some
           (let_ t ((f64 1.0 -! EVar a) *! f64 0.5)
           @@ let_ p (pnum (EVar t))
           @@ let_ q (qden (EVar t))
           @@ let_ s (sqrt_ (EVar t))
           @@ SIf
                ( EVar a >! f64 0.975,
                  SReturn
                    (copysign_
                       (f64 pio2_hi
                       -! ((f64 2.0 *! fma (EVar s) (EVar p /! EVar q) (EVar s))
                          -! f64 pio2_lo))
                       (EVar x)),
                  Some
                    (let_ ws (unbits (bits (EVar s) &! i64 0xFFFFFFFF00000000L))
                    @@ let_
                         c
                         ((EVar t -! (EVar ws *! EVar ws))
                         /! (EVar s +! EVar ws))
                    @@ let_
                         p2
                         ((f64 2.0 *! EVar s *! (EVar p /! EVar q))
                         -! (f64 pio2_lo -! (EVar c +! EVar c)))
                    @@ let_ q2 (f64 pio4_hi -! (EVar ws +! EVar ws))
                    @@ SReturn
                         (copysign_
                            (f64 pio4_hi -! (EVar p2 -! EVar q2))
                            (EVar x))) )) )

(** acos(x): fdlibm 5.3 e_acos.c — |x| < 0.5: π/2 − (x − (π/2_lo − x·R(x²))); x
    ≤ −0.5: π − 2·(s + (R·s − π/2_lo)) on z = (1+x)/2, s = sqrt(z); x ≥ 0.5:
    2·(df + (R·s + c)) on z = (1−x)/2 with the low-word-zeroed df and its
    exact-square correction c (c := 0 at z = 0, i.e. x = 1, avoiding the 0/0
    fdlibm dodges with an early return). *)
let acos_body x =
  let z = fvar "z" in
  let s = fvar "s" in
  let r = fvar "r" in
  let df = fvar "df" in
  let c = fvar "c" in
  (* The shared fdlibm rational R(v) = p(v)/q(v). *)
  let rfun v =
    v
    *! horner v ~coeffs:asin_p_coeffs ~last:asin_ps0
    /! horner v ~coeffs:asin_q_coeffs ~last:1.0
  in
  SIf
    ( abs_ (EVar x) <! f64 0.5,
      let_ z (EVar x *! EVar x)
      @@ SReturn
           (f64 pio2_hi
           -! (EVar x -! (f64 pio2_lo -! (EVar x *! rfun (EVar z))))),
      Some
        (SIf
           ( EVar x <! f64 0.0,
             let_ z ((f64 1.0 +! EVar x) *! f64 0.5)
             @@ let_ s (sqrt_ (EVar z))
             @@ let_ r (rfun (EVar z))
             @@ SReturn
                  (f64 pi
                  -! (f64 2.0 *! (EVar s +! ((EVar r *! EVar s) -! f64 pio2_lo)))
                  ),
             Some
               (let_ z ((f64 1.0 -! EVar x) *! f64 0.5)
               @@ let_ s (sqrt_ (EVar z))
               @@ let_ df (unbits (bits (EVar s) &! i64 0xFFFFFFFF00000000L))
               @@ let_
                    c
                    (EIf
                       ( EVar z =! f64 0.0,
                         f64 0.0,
                         (EVar z -! (EVar df *! EVar df)) /! (EVar s +! EVar df)
                       ))
               @@ let_ r (rfun (EVar z))
               @@ SReturn
                    (f64 2.0 *! (EVar df +! fma (EVar r) (EVar s) (EVar c)))) ))
    )

(** expm1(x): fdlibm 5.3 s_expm1.c — k = rint(x/ln2) with the fma Cody-Waite
    residual c; rational correction e from Q1..Q5 on the reduced x. k = 0 keeps
    fdlibm's exact near-zero form x − (x·e − x²/2) (full relative precision —
    the reason expm1 exists); k ≠ 0 collapses fdlibm's k-cases to 2^k·(1 − (e′ −
    x)) − 1 via fma (result magnitude ≥ 0.41 there, so the collapsed form stays
    ≪ 1e-12 relative). Domain x ∈ [−708, 709]. *)
let expm1_body x =
  let nf = fvar "nf" in
  let hi = fvar "hi" in
  let lo = fvar "lo" in
  let xr = fvar "xr" in
  let c = fvar "c" in
  let hfx = fvar "hfx" in
  let hxs = fvar "hxs" in
  let r1 = fvar "r1" in
  let t = fvar "t" in
  let e = fvar "e" in
  let n = ivar "n" in
  let e2 = fvar "e2" in
  let core =
    let_ nf (floor_ (fma (EVar x) (f64 log2_e) (f64 0.5)))
    @@ let_ hi (fma (EVar nf) (f64 (-.ln2_hi)) (EVar x))
    @@ let_ lo (EVar nf *! f64 ln2_lo)
    @@ let_ xr (EVar hi -! EVar lo)
    @@ let_ c (EVar hi -! EVar xr -! EVar lo)
    @@ let_ hfx (f64 0.5 *! EVar xr)
    @@ let_ hxs (EVar xr *! EVar hfx)
    @@ let_
         r1
         (fma
            (EVar hxs)
            (horner (EVar hxs) ~coeffs:expm1_q_coeffs ~last:expm1_q1)
            (f64 1.0))
    @@ let_ t (fma (EVar r1) (neg (EVar hfx)) (f64 3.0))
    @@ let_
         e
         (EVar hxs
         *! ((EVar r1 -! EVar t) /! fma (neg (EVar xr)) (EVar t) (f64 6.0)))
    @@ let_ n (to_i32 (EVar nf))
    @@ SIf
         ( EVar n =! i32 0,
           SReturn (EVar xr -! ((EVar xr *! EVar e) -! EVar hxs)),
           Some
             (let_ e2 ((EVar xr *! (EVar e -! EVar c)) -! EVar c -! EVar hxs)
             @@ SReturn
                  (fma
                     (f64 1.0 -! (EVar e2 -! EVar xr))
                     (unbits (shl (to_i64 (EVar n +! i32 1023)) 52))
                     (f64 (-1.0)))) )
  in
  (* Same exponent-field saturation as exp (audit finding M3): below the
     cut expm1 = -1 to within 1 ulp; above, +inf. *)
  SIf
    ( EVar x <! f64 exp_lo_cut,
      SReturn (f64 (-1.0)),
      Some (SIf (EVar x >! f64 exp_hi_cut, SReturn (f64 infinity), Some core))
    )

(** log1p(x): fdlibm 5.3 s_log1p.c — u = 1 + x with the rounding correction
    c = (k > 0 ? 1 − (u − x) : x − (u − 1))/u, u's mantissa normalized to
    [√2/2, √2) exactly as __sarek_f64_log, f = m − 1, R(z) on z = s²,
    s = f/(2+f), result k·ln2_hi + (f − (hfsq − (s·(hfsq+R) + (k·ln2_lo+c)))).
    fdlibm's separate k = 0 fast path is subsumed: for u ∈ [√2/2, √2) the
    normalization yields k = 0, f = u − 1 (exact by Sterbenz) and c carries
    the 1 + x rounding residual, preserving full relative precision near 0
    (log1p's raison d'être). Domain x > −1. *)
let log1p_body x =
  let u = fvar "u" in
  let b = lvar "b" in
  let k_raw = ivar "k_raw" in
  let m0 = fvar "m0" in
  let big = mkvar TBool "big" in
  let m = fvar "m" in
  let k = ivar "k" in
  let c = fvar "c" in
  let f = fvar "f" in
  let s = fvar "s" in
  let z = fvar "z" in
  let r = fvar "r" in
  let hfsq = fvar "hfsq" in
  let kf = fvar "kf" in
  let_ u (f64 1.0 +! EVar x)
  @@ let_ b (bits (EVar u))
  @@ let_ k_raw (to_i32 (shr (EVar b) 52 &! i64 0x7FFL))
  @@ let_
       m0
       (unbits (EVar b &! i64 0xFFFFFFFFFFFFFL |! i64 0x3FF0000000000000L))
  @@ let_ big (EVar m0 >! f64 sqrt2)
  @@ let_ m (EIf (EVar big, EVar m0 *! f64 0.5, EVar m0))
  @@ let_ k (EIf (EVar big, EVar k_raw -! i32 1022, EVar k_raw -! i32 1023))
  @@ let_
       c
       (EIf
          ( EVar k >! i32 0,
            f64 1.0 -! (EVar u -! EVar x),
            EVar x -! (EVar u -! f64 1.0) )
       /! EVar u)
  @@ let_ f (EVar m -! f64 1.0)
  @@ let_ s (EVar f /! (f64 2.0 +! EVar f))
  @@ let_ z (EVar s *! EVar s)
  @@ let_ r (EVar z *! horner (EVar z) ~coeffs:log1p_lp_coeffs ~last:log1p_lp1)
  @@ let_ hfsq (f64 0.5 *! EVar f *! EVar f)
  @@ let_ kf (to_f64 (EVar k))
  @@ SReturn
       (fma
          (EVar kf)
          (f64 ln2_hi)
          (EVar f
          -! (EVar hfsq
             -! ((EVar s *! (EVar hfsq +! EVar r))
                +! fma (EVar kf) (f64 ln2_lo) (EVar c)))))

(** {1 Helper table} *)

let unary name body =
  let x =
    {var_name = "x"; var_id = 0; var_type = TFloat64; var_mutable = false}
  in
  {hf_name = name; hf_params = [x]; hf_ret_type = TFloat64; hf_body = body x}

let binary name body =
  let x =
    {var_name = "x"; var_id = 0; var_type = TFloat64; var_mutable = false}
  in
  let y =
    {var_name = "y"; var_id = 1; var_type = TFloat64; var_mutable = false}
  in
  {
    hf_name = name;
    hf_params = [x; y];
    hf_ret_type = TFloat64;
    hf_body = body x y;
  }

let helpers =
  lazy
    [
      unary "__sarek_f64_exp" exp_body;
      unary "__sarek_f64_log" log_body;
      unary "__sarek_f64_sin" (trig_body ~shift:0);
      unary "__sarek_f64_cos" (trig_body ~shift:1);
      unary "__sarek_f64_tan" tan_body;
      unary "__sarek_f64_log10" log10_body;
      unary "__sarek_f64_sinh" sinh_body;
      unary "__sarek_f64_cosh" cosh_body;
      unary "__sarek_f64_tanh" tanh_body;
      binary "__sarek_f64_pow" pow_body;
      unary "__sarek_f64_atan" atan_body;
      binary "__sarek_f64_atan2" atan2_body;
      unary "__sarek_f64_asin" asin_body;
      unary "__sarek_f64_acos" acos_body;
      unary "__sarek_f64_expm1" expm1_body;
      unary "__sarek_f64_log1p" log1p_body;
    ]

let helper_name = function
  | "exp" -> Some "__sarek_f64_exp"
  | "log" -> Some "__sarek_f64_log"
  | "sin" -> Some "__sarek_f64_sin"
  | "cos" -> Some "__sarek_f64_cos"
  | "tan" -> Some "__sarek_f64_tan"
  | "log10" -> Some "__sarek_f64_log10"
  | "sinh" -> Some "__sarek_f64_sinh"
  | "cosh" -> Some "__sarek_f64_cosh"
  | "tanh" -> Some "__sarek_f64_tanh"
  | "pow" -> Some "__sarek_f64_pow"
  | "atan" -> Some "__sarek_f64_atan"
  | "atan2" -> Some "__sarek_f64_atan2"
  | "asin" -> Some "__sarek_f64_asin"
  | "acos" -> Some "__sarek_f64_acos"
  | "expm1" -> Some "__sarek_f64_expm1"
  | "log1p" -> Some "__sarek_f64_log1p"
  | _ -> None

let register funcs =
  List.iter (fun hf -> Hashtbl.replace funcs hf.hf_name hf) (Lazy.force helpers)

let all_helpers () = Lazy.force helpers

(** {1 Body introspection}

    A backend that emits the helpers as separate functions (rather than inlining
    them, as PTX does) needs, per kernel, the transitive family of helpers and
    the machinery each one requires. These fold over the pure-IR bodies so the
    GLSL emitter does not duplicate an IR walker. *)

let helper_by_name name =
  List.find_opt (fun hf -> String.equal hf.hf_name name) (Lazy.force helpers)

(** Fold [f] over every subexpression of [e] (pre-order). *)
let rec fold_expr f acc e =
  let acc = f acc e in
  match e with
  | EConst _ | EVar _ | EArrayLen _ -> acc
  | EBinop (_, a, b) | EArrayReadExpr (a, b) ->
      fold_expr f (fold_expr f acc a) b
  | EUnop (_, a) | ERecordField (a, _) | ECast (_, a) | EArrayRead (_, a) ->
      fold_expr f acc a
  | ETuple es | EVariant (_, _, es) -> List.fold_left (fold_expr f) acc es
  | EApp (fn, args) -> List.fold_left (fold_expr f) (fold_expr f acc fn) args
  | EIntrinsic (_, _, args) -> List.fold_left (fold_expr f) acc args
  | ERecord (_, fields) ->
      List.fold_left (fun a (_, e) -> fold_expr f a e) acc fields
  | EArrayCreate (_, size, _) -> fold_expr f acc size
  | EIf (c, t, e) -> fold_expr f (fold_expr f (fold_expr f acc c) t) e
  | EMatch (s, cases) ->
      List.fold_left (fun a (_, e) -> fold_expr f a e) (fold_expr f acc s) cases

(** Fold [f] (expr visitor) over every expression reachable from statement [s].
*)
let rec fold_stmt_exprs f acc s =
  match s with
  | SEmpty | SBarrier | SWarpBarrier | SMemFence | SNative _ -> acc
  | SSeq stmts -> List.fold_left (fold_stmt_exprs f) acc stmts
  | SAssign (_, e) | SReturn e | SExpr e -> fold_expr f acc e
  | SIf (c, t, e) ->
      let acc = fold_stmt_exprs f (fold_expr f acc c) t in
      Option.fold ~none:acc ~some:(fold_stmt_exprs f acc) e
  | SWhile (c, body) -> fold_stmt_exprs f (fold_expr f acc c) body
  | SFor (_, lo, hi, _, body) ->
      fold_stmt_exprs f (fold_expr f (fold_expr f acc lo) hi) body
  | SMatch (s, cases) ->
      List.fold_left
        (fun a (_, st) -> fold_stmt_exprs f a st)
        (fold_expr f acc s)
        cases
  | SLet (_, e, body) | SLetMut (_, e, body) ->
      fold_stmt_exprs f (fold_expr f acc e) body
  | SPragma (_, body) | SBlock body -> fold_stmt_exprs f acc body

let is_softmath_helper_name name =
  let p = "__sarek_f64_" in
  String.length name >= String.length p
  && String.sub name 0 (String.length p) = p

(** Names of the sibling softmath helpers [hf] calls directly (deduplicated).
    Softmath cross-calls are [EApp (EVar {var_name = "__sarek_f64_*"}, _)]. *)
let callees hf =
  fold_stmt_exprs
    (fun acc e ->
      match e with
      | EApp (EVar v, _) when is_softmath_helper_name v.var_name ->
          v.var_name :: acc
      | _ -> acc)
    []
    hf.hf_body
  |> List.sort_uniq compare

(** Whether [hf]'s body needs 64-bit integer support — the [f64_bits]/[bits_f64]
    bitcasts or any [int64] value/cast/literal. The pure-polynomial helpers
    (sin, cos, tan, atan, atan2) need none; the exponent/mantissa-manipulating
    ones (exp, log, …) do. *)
let uses_int64 hf =
  fold_stmt_exprs
    (fun acc e ->
      acc
      ||
      match e with
      | EIntrinsic ([], ("f64_bits" | "bits_f64"), _) -> true
      | EConst (CInt64 _) -> true
      | ECast (TInt64, _) -> true
      | _ -> false)
    false
    hf.hf_body

(** Whether [hf]'s body calls the [copysign] intrinsic (which a
    separate-function backend must have already declared — GLSL's
    [sarek_copysign] helper). *)
let uses_copysign hf =
  fold_stmt_exprs
    (fun acc e ->
      acc || match e with EIntrinsic (_, "copysign", _) -> true | _ -> false)
    false
    hf.hf_body
