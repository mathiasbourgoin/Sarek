(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Software f64 transcendentals for the PTX backend.

    PTX has no f64 transcendental instructions (sin/cos/ex2/lg2 exist only as
    f32 [.approx] ops). This module provides polynomial implementations as pure
    IR [helper_func] bodies, built from natively-supported f64 operations only:
    add/sub/mul/div.rn, fma.rn, floor (cvt.rmi), min, abs, copysign, and the
    [f64_bits]/[bits_f64] bitcasts (mov.b64). The PTX emitter registers these
    helpers on demand and routes [EIntrinsic (["Float64"], name, args)] through
    the existing EApp inlining machinery — no emitted-assembly polynomials.

    Numerics: precise tier, max relative error ≤ 1e-12 on the documented domains
    (measured, see sarek/tests/unit/test_f64_softmath.ml).

    Coefficient provenance: the sin/cos kernel polynomials and the Cody-Waite
    π/2 and ln2 hi/lo splits are the published fdlibm 5.3 constants (Sun
    Microsystems, freely-redistributable license; the same values appear in
    Cephes and glibc). The exp and atanh-based log polynomials are plain Taylor
    coefficients — on the reduced domains (|r| ≤ ln2/2 for exp, z = s² ≤ 0.0295
    for log) truncation error is below ~1e-14 relative, so minimax refinement is
    unnecessary at this tier.

    Documented domains (graceful degradation outside, no Payne-Hanek):
    - sin/cos/tan: |x| ≤ 1e6 (3-term Cody-Waite reduction);
    - exp: x ∈ [-708, 709] (2^n scaling via exponent-field construction; no
      gradual-underflow handling below -708);
    - log/log10: finite normal x > 0;
    - pow: x > 0 (log of a negative is NaN; integer-exponent negative bases are
      not handled);
    - sinh/cosh: |x| ≤ 708 (via exp(|x|)); tanh: all finite x (argument clamped,
      saturates to ±1). *)

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

let inv_fact k =
  let rec fact i acc =
    if i = 0 then acc else fact (i - 1) (acc *. float_of_int i)
  in
  1.0 /. fact k 1.0

(** {1 Helper bodies} *)

(** exp(x): n = rint(x·log2 e) (as floor(·+0.5)); r = x − n·ln2 (hi/lo split, 2
    fma); degree-12 Taylor on r ∈ [−ln2/2, ln2/2] (truncation ≈ 1.7e-16
    relative); scale by 2^n via exponent-field construction (n+1023) << 52. *)
let exp_body x =
  let nf = fvar "nf" in
  let r_hi = fvar "r_hi" in
  let r = fvar "r" in
  let p = fvar "p" in
  let n = ivar "n" in
  (* c12 … c3, then c2 = 1/2 as the Horner constant term. *)
  let coeffs = List.init 10 (fun i -> inv_fact (12 - i)) in
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

(** log(x): exponent extract + mantissa normalized to [√2/2, √2), then
    log(m) = 2·atanh(s) with s = (m−1)/(m+1): odd Taylor to s¹⁵ (truncation
    ≤ 3.3e-14 of the atanh term); log(x) = k·ln2_hi + (log(m) + k·ln2_lo). *)
let log_body x =
  let b = lvar "b" in
  let k_raw = ivar "k_raw" in
  let m0 = fvar "m0" in
  let big = ivar "big" in
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
  let sqrt2 = 1.41421356237309514547 in
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
  | _ -> None

let register funcs =
  List.iter (fun hf -> Hashtbl.replace funcs hf.hf_name hf) (Lazy.force helpers)

let all_helpers () = Lazy.force helpers
