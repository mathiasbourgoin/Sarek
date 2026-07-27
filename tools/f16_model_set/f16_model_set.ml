(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * #62 slice 1 — the named f16 rounding models, as host functions.
 *
 * WHAT THIS IS.
 *
 * docs/design/f16-relaxed-accuracy.md §1.2 does not accept a device result
 * because it is close to the interpreter's. It accepts it only when it is
 * BIT-IDENTICAL to one member of a finite set of NAMED, CLOSED-FORM reference
 * semantics — §0.1 is the argument for why a tolerance cannot work here. This
 * module is that set, computed exactly on the host, for the two f16 expression
 * shapes this project has measured on a device.
 *
 * It is deliberately NOT [Test_helpers]. §7 slice 0 is the slice that promotes
 * [ref_discipline] / [ref_fused] into shared test machinery with a
 * [classify_f16_result] classifier; this module is slice 1's MEASUREMENT
 * instrument and stays out of the test libraries so that slice 0 is free to
 * design the classifier without inheriting anything from here.
 *
 * WHY THE ARITHMETIC IS NOT JUST OCaml FLOATS.
 *
 * Three of the models below round a value to binary16 in a SINGLE step from a
 * sum that is not exactly representable in binary64. Concretely: the exact
 * product [x * fl32(1.1)] for a binary16 [x] is an integer multiple of 2^-47
 * (fl32(1.1) = 9227469 * 2^-23, and x is an integer multiple of 2^-24), while
 * the addend 1000 reaches 2^9. An exact sum therefore spans up to 2^-47 .. 2^17
 * — 65 bits — and binary64 has 53. Evaluating [p +. 1000.0] in OCaml would
 * round FIRST and then the model would be rounding a rounded value, which is a
 * different function from the one being named. So every single-rounding model
 * goes through [two_sum], which represents the exact sum as an unevaluated
 * pair, and through [round_dd], which rounds that pair in one step.
 *
 * This matters in exactly the place it is hardest to notice: near a binary16
 * tie. A double-rounded model and a correctly-rounded one agree everywhere
 * except on the ties, and the ties are precisely where §1.3's counterexample
 * (x = -907.5) lives.
 ******************************************************************************)

(* ------------------------------------------------------------------------ *)
(* binary64 -> binary32, single correct rounding.                            *)
(* Valid only when the argument is EXACTLY representable in binary64; every   *)
(* use below satisfies that and says so.                                      *)
(* ------------------------------------------------------------------------ *)
let f32 x = Int32.float_of_bits (Int32.bits_of_float x)

(* ------------------------------------------------------------------------ *)
(* binary16 decoding. Exact: no rounding is involved, so this cannot itself   *)
(* be a source of disagreement.                                              *)
(* ------------------------------------------------------------------------ *)
let f16_decode b =
  let sign = if b land 0x8000 <> 0 then -1.0 else 1.0 in
  let e = (b lsr 10) land 0x1f and m = b land 0x3ff in
  if e = 31 then None
  else if e = 0 then Some (sign *. float_of_int m *. ldexp 1.0 (-24))
  else Some (sign *. float_of_int (1024 + m) *. ldexp 1.0 (e - 25))

let dec b =
  match f16_decode b with
  | Some v -> v
  | None ->
      if b land 0x3ff <> 0 then Float.nan
      else if b land 0x8000 <> 0 then Float.neg_infinity
      else Float.infinity

(* All finite binary16 bit patterns: 63488 of them, so every statement this
   module supports is exhaustive over the domain rather than sampled. *)
let finite_bits =
  let acc = ref [] in
  for b = 0xFFFF downto 0 do
    if f16_decode b <> None then acc := b :: !acc
  done ;
  Array.of_list !acc

(* ------------------------------------------------------------------------ *)
(* Exact unevaluated sums.                                                    *)
(* ------------------------------------------------------------------------ *)

(* Knuth's twoSum: [a +. b] is EXACTLY [s +. e], with no assumption about the
   relative magnitudes of [a] and [b]. Both outputs are binary64 values, so the
   pair is an exact representation of a sum that binary64 alone cannot hold. *)
let two_sum a b =
  let s = a +. b in
  let bb = s -. a in
  let err = a -. (s -. bb) +. (b -. bb) in
  (s, err)

(* Round the exact value [s + e] to a binary format with [prec] significand
   bits whose smallest subnormal is 2^[emin_sub], returning the result as a
   binary64 (exact, since [prec] <= 24 here).

   The residual [e] can only ever matter at an exact tie. |e| <= ulp64(s)/2,
   and the rescaling below puts the target's ulp at 1, so the rescaled residual
   is at most 2^-30 for prec = 24 and 2^-43 for prec = 11, while the rescaled
   [s] is a multiple of at least 2^-29 / 2^-42 respectively. A non-tie fraction
   is therefore at least one rescaled-ulp away from 0.5 and the residual cannot
   push it across; a tie is exactly 0.5 and the residual decides. That argument
   is the whole reason this function exists instead of [s +. e]. *)
let round_dd ~prec ~emin_sub (s, e) =
  if Float.is_nan s || Float.is_nan e then Float.nan
  else if s = 0.0 then e
  else if Float.abs s = Float.infinity then s
  else begin
    let neg = s < 0.0 in
    let a = Float.abs s in
    (* sign of the residual RELATIVE TO THE MAGNITUDE of s *)
    let sticky = if e = 0.0 then 0 else if s > 0.0 = (e > 0.0) then 1 else -1 in
    let ex = snd (Float.frexp a) - 1 in
    let ulp_exp = max (ex - prec + 1) emin_sub in
    let v = a /. ldexp 1.0 ulp_exp in
    let f = Float.floor v in
    let r = v -. f in
    let q =
      if r > 0.5 then f +. 1.0
      else if r < 0.5 then f
      else if sticky > 0 then f +. 1.0
      else if sticky < 0 then f
      else if Float.rem f 2.0 = 0.0 then f
      else f +. 1.0
    in
    let m = ldexp 1.0 prec in
    let q, ulp_exp = if q >= m then (q /. 2.0, ulp_exp + 1) else (q, ulp_exp) in
    let out = q *. ldexp 1.0 ulp_exp in
    if neg then -.out else out
  end

(* ------------------------------------------------------------------------ *)
(* binary16 encoding of an exactly-representable binary64.                    *)
(* Ported verbatim from sarek-vulkan/test/test_vulkan_f16_tripwire.ml so the   *)
(* 620 calibration below is comparing against the same function the shipped    *)
(* tripwires already calibrate.                                               *)
(* ------------------------------------------------------------------------ *)
let round_even v =
  let f = Float.floor v in
  let r = v -. f in
  if r > 0.5 then f +. 1.0
  else if r < 0.5 then f
  else if Float.rem f 2.0 = 0.0 then f
  else f +. 1.0

let f16_bits d =
  if Float.is_nan d then 0x7E00
  else
    let s = if d < 0.0 || (d = 0.0 && 1.0 /. d < 0.0) then 0x8000 else 0 in
    let a = Float.abs d in
    if a = Float.infinity then s lor 0x7C00
    else if a = 0.0 then s
    else
      let e = snd (Float.frexp a) - 1 in
      if e < -14 then s lor int_of_float (round_even (a /. ldexp 1.0 (-24)))
      else
        let q = round_even (a /. ldexp 1.0 (e - 10)) in
        let e, q = if q >= 2048.0 then (e + 1, 1024.0) else (e, q) in
        if e + 15 >= 31 then s lor 0x7C00
        else s lor ((e + 15) lsl 10) lor (int_of_float q - 1024)

(* Single-rounding narrowing of the exact sum [a + b] to binary16. *)
let f16_of_exact_sum a b =
  f16_bits (round_dd ~prec:11 ~emin_sub:(-24) (two_sum a b))

(* Single-rounding narrowing of the exact sum [a + b] to binary32. *)
let f32_of_exact_sum a b = round_dd ~prec:24 ~emin_sub:(-149) (two_sum a b)

(* Spacing of binary16 at [v]; the denominator §1.3's ceiling is measured in.
   Subnormals use the absolute 2^-24 spacing, exactly as §1.3 prescribes. *)
let ulp16 v =
  let a = Float.abs v in
  if a = 0.0 then ldexp 1.0 (-24)
  else
    let e = snd (Float.frexp a) - 1 in
    if e < -14 then ldexp 1.0 (-24) else ldexp 1.0 (e - 10)

(* ------------------------------------------------------------------------ *)
(* The constant, and the two exact intermediates every model is built from.   *)
(* ------------------------------------------------------------------------ *)

let c11 = f32 1.1

(* The EXACT product x * fl32(1.1). A binary16 significand (11 bits) times a
   binary32 one (24 bits) needs at most 35 bits, so this multiplication is
   exact in binary64 and is not a rounding. *)
let p_exact b = dec b *. c11

(* The same product rounded to binary32, i.e. what the DSL mandates the
   multiply produce. *)
let p32 b = f32 (p_exact b)

(* ------------------------------------------------------------------------ *)
(* SHAPE 1 — f16(x * 1.1). One narrowing.                                     *)
(* ------------------------------------------------------------------------ *)

let s1_strict b = f16_bits (p32 b)

let s1_fuse_mul b = f16_bits (p_exact b)

(* ------------------------------------------------------------------------ *)
(* SHAPE 2 — f16(f16(x * 1.1) + 1000). Two narrowings.                        *)
(*                                                                            *)
(* The models are named for WHICH mandated rounding they elide, because that   *)
(* is what §1.2 requires a member of the admissible set to be: a named,        *)
(* closed-form function, not "whatever the device did".                        *)
(*                                                                            *)
(* [dec m +. 1000.0] where m is a binary16 value is EXACT in binary64 (m's     *)
(* lowest bit is at worst 2^-24 and 1000's highest is 2^9, so 34 bits), which  *)
(* is why the two-rounding models can use plain [f32 (... +. 1000.0)] while    *)
(* the single-rounding ones cannot.                                           *)
(* ------------------------------------------------------------------------ *)

(* Every mandated rounding performed. The interpreter. *)
let s2_strict b =
  let m = f16_bits (p32 b) in
  f16_bits (f32 (dec m +. 1000.0))

(* S_fuse_mul_into_narrowing: the f32 multiply is absorbed into the f16
   narrowing that consumes it. The binary32 rounding of the product is elided;
   the intermediate binary16 value still exists. This is the model measured on
   hiprtc/gfx1100 and rusticl/radeonsi at 620/63488. *)
let s2_fuse_mul b =
  let m = f16_bits (p_exact b) in
  f16_bits (f32 (dec m +. 1000.0))

(* Everything absorbed into the final narrowing: one rounding for an expression
   the DSL says has four. This is the shape of a single v_fma_mixlo_f16 that
   takes x, 1.1 and 1000 as its three operands. *)
let s2_absorb_all b = f16_of_exact_sum (p_exact b) 1000.0

(* The multiply keeps its own binary32 rounding — what a honoured
   NoContraction on the OpFMul buys — but the intermediate binary16 narrowing
   and the binary32 add are both absorbed into the final narrowing. *)
let s2_absorb_add b = f16_of_exact_sum (p32 b) 1000.0

(* The intermediate binary16 narrowing is dropped outright and the add is still
   rounded to binary32: the IGC defect signature of fp-contraction-policy.md
   §11.4, and the model the volatile-SSBO RADV variant matched at 4774/63488. *)
let s2_drop_inner b = f16_bits (f32_of_exact_sum (p32 b) 1000.0)

(* The multiply is absorbed into the intermediate narrowing AND the add is
   absorbed into the final one. *)
let s2_fuse_mul_absorb_add b =
  f16_of_exact_sum (dec (f16_bits (p_exact b))) 1000.0

(* Only the add is absorbed into the final narrowing; the multiply and the
   intermediate narrowing are both as the DSL mandates. *)
let s2_strict_absorb_add b = f16_of_exact_sum (dec (f16_bits (p32 b))) 1000.0

(* ------------------------------------------------------------------------ *)
(* The model sets, as data.                                                   *)
(* ------------------------------------------------------------------------ *)

type model = {
  name : string;
  descr : string;
  result : int -> int;
      (** binary16 bit pattern produced for input bit pattern [b] *)
  at_inner_narrowing : (int -> float) option;
      (** The exact real value this model presents at the position where
          S_strict materialises its intermediate binary16 value — i.e. the value
          that reaches the rest of the expression in place of the mandated one.
          §1.3's ceiling is evaluated HERE and not on the final value. [None]
          means the model materialises nothing comparable there and the ceiling
          is NOT APPLICABLE, which §1.3 requires be reported as such rather than
          silently evaluated on the final value. *)
}

let shape1_name = "f16(x * 1.1)"

let shape1_models =
  [
    {
      name = "S_strict";
      descr = "every mandated rounding performed (the interpreter)";
      result = s1_strict;
      at_inner_narrowing = Some (fun b -> dec (s1_strict b));
    };
    {
      name = "S_fuse_mul_into_narrowing";
      descr = "the f32 multiply absorbed into the f32->f16 narrowing";
      result = s1_fuse_mul;
      at_inner_narrowing = Some (fun b -> dec (s1_fuse_mul b));
    };
  ]

let shape2_name = "f16(f16(x * 1.1) + 1000)"

let shape2_models =
  [
    {
      name = "S_strict";
      descr = "every mandated rounding performed (the interpreter)";
      result = s2_strict;
      at_inner_narrowing = Some (fun b -> dec (f16_bits (p32 b)));
    };
    {
      name = "S_fuse_mul_into_narrowing";
      descr = "f32 multiply absorbed into the intermediate narrowing";
      result = s2_fuse_mul;
      at_inner_narrowing = Some (fun b -> dec (f16_bits (p_exact b)));
    };
    {
      name = "S_absorb_all_into_final_narrowing";
      descr =
        "multiply, intermediate narrowing and add all absorbed: one rounding";
      result = s2_absorb_all;
      at_inner_narrowing = Some p_exact;
    };
    {
      name = "S_f32_mul_then_absorb_add";
      descr =
        "multiply rounded to f32; intermediate narrowing and add absorbed into \
         the final narrowing";
      result = s2_absorb_add;
      at_inner_narrowing = Some p32;
    };
    {
      name = "S_drop_intermediate_narrowing";
      descr =
        "intermediate binary16 narrowing dropped, add still rounded to f32 \
         (the IGC signature)";
      result = s2_drop_inner;
      at_inner_narrowing = Some p32;
    };
    {
      name = "S_fuse_mul_and_absorb_add";
      descr =
        "multiply absorbed into the intermediate narrowing, add absorbed into \
         the final one";
      result = s2_fuse_mul_absorb_add;
      at_inner_narrowing = Some (fun b -> dec (f16_bits (p_exact b)));
    };
    {
      name = "S_strict_mul_absorb_add";
      descr = "only the add absorbed into the final narrowing";
      result = s2_strict_absorb_add;
      at_inner_narrowing = Some (fun b -> dec (f16_bits (p32 b)));
    };
  ]

(* ------------------------------------------------------------------------ *)
(* CALIBRATION — run before any device number is printed.                     *)
(* ------------------------------------------------------------------------ *)

exception Calibration_failed of string

let failf fmt = Printf.ksprintf (fun s -> raise (Calibration_failed s)) fmt

(* 1. The encoder must round-trip the whole domain. *)
let check_round_trip () =
  Array.iter
    (fun b ->
      if f16_bits (dec b) <> b then
        failf
          "host binary16 rounding is wrong: re-encoding the exact value of \
           0x%04X gives 0x%04X"
          b
          (f16_bits (dec b)))
    finite_bits ;
  if Array.length finite_bits <> 63488 then
    failf
      "expected 63488 finite binary16 inputs, enumerated %d"
      (Array.length finite_bits)

(* 2. The two models §1.2 names must separate on exactly the independently
   measured 620 — the figure reproduced on hiprtc/gfx1100, rusticl/radeonsi,
   the OpenCL fusedctl control on Intel Arc and the Metal round-to-odd control
   on an M4. Reproducing a known positive is what licenses believing the rest. *)
let check_620 () =
  let n = ref 0 in
  Array.iter (fun b -> if s2_strict b <> s2_fuse_mul b then incr n) finite_bits ;
  if !n <> 620 then
    failf
      "CALIBRATION FAILED: S_strict and S_fuse_mul_into_narrowing separate on \
       %d of %d inputs on %s. It must be 620."
      !n
      (Array.length finite_bits)
      shape2_name

(* 3. Shape 1's two models must separate on exactly 2912, the figure RADV
   reports against the discipline on that shape. *)
let check_2912 () =
  let n = ref 0 in
  Array.iter (fun b -> if s1_strict b <> s1_fuse_mul b then incr n) finite_bits ;
  if !n <> 2912 then
    failf
      "CALIBRATION FAILED: S_strict and S_fuse_mul_into_narrowing separate on \
       %d of %d inputs on %s. It must be 2912."
      !n
      (Array.length finite_bits)
      shape1_name

(* 4. §1.3's counterexample, host-only. At x = -907.5 the two models differ by
   exactly 1 ulp AT THE INTERMEDIATE NARROWING and by 512 ulp on the FINAL
   value. A ceiling evaluated on the final value rejects a result the admitted
   model produces; this check pins that the models really do reproduce the
   case, so the ceiling numbers printed later are measuring something. *)
let check_907_5 () =
  let b = f16_bits (-907.5) in
  if dec b <> -907.5 then
    failf
      "x = -907.5 is not exactly representable in binary16 (got %.9g)"
      (dec b) ;
  let strict_inner = dec (f16_bits (p32 b)) in
  let fused_inner = dec (f16_bits (p_exact b)) in
  if strict_inner <> -998.0 || fused_inner <> -998.5 then
    failf
      "§1.3's counterexample does not reproduce: intermediate is %.9g strict / \
       %.9g fused, expected -998 / -998.5"
      strict_inner
      fused_inner ;
  let strict_final = dec (s2_strict b) and fused_final = dec (s2_fuse_mul b) in
  if strict_final <> 2.0 || fused_final <> 1.5 then
    failf
      "§1.3's counterexample does not reproduce: final is %.9g strict / %.9g \
       fused, expected 2.0 / 1.5"
      strict_final
      fused_final ;
  let inner_ulp =
    Float.abs (fused_inner -. strict_inner) /. ulp16 strict_inner
  in
  let final_ulp =
    Float.abs (fused_final -. strict_final) /. ulp16 fused_final
  in
  if inner_ulp <> 1.0 then
    failf "§1.3: expected 1 ulp at the narrowing, computed %.9g" inner_ulp ;
  if final_ulp <> 512.0 then
    failf "§1.3: expected 512 ulp on the final value, computed %.9g" final_ulp

let calibrate () =
  check_round_trip () ;
  check_620 () ;
  check_2912 () ;
  check_907_5 ()

(* ------------------------------------------------------------------------ *)
(* Reporting                                                                  *)
(* ------------------------------------------------------------------------ *)

(* The trap this guards: a model set whose members happen to coincide on the
   swept inputs reports "exact agreement" while discriminating nothing. Over
   the finite binary16 domain the separation is a fixed, checkable number, so
   it is printed and any zero is called out. *)
let separation_matrix models =
  Printf.printf "  pairwise model separation (inputs where the two differ):\n" ;
  let arr = Array.of_list models in
  let n = Array.length arr in
  let coincident = ref [] in
  for i = 0 to n - 1 do
    for j = i + 1 to n - 1 do
      let c = ref 0 in
      Array.iter
        (fun b -> if arr.(i).result b <> arr.(j).result b then incr c)
        finite_bits ;
      Printf.printf "    %-34s vs %-34s : %6d\n" arr.(i).name arr.(j).name !c ;
      if !c = 0 then coincident := (arr.(i).name, arr.(j).name) :: !coincident
    done
  done ;
  (match !coincident with
  | [] -> ()
  | l ->
      List.iter
        (fun (a, b) ->
          Printf.printf
            "    *** %s and %s COINCIDE on the whole domain: a device matching \
             one matches the other, and the pair discriminates nothing ***\n"
            a
            b)
        l) ;
  !coincident = []

(* Element-wise classification of a device result against the model set. *)
type classification = {
  per_model : (string * int) list;  (** model name -> disagreement count *)
  exact_matches : string list;  (** models agreeing on ALL 63488 *)
  unexplained : int;  (** inputs matching NO model *)
  first_unexplained : int;  (** index into [finite_bits], or -1 *)
}

let classify models device =
  let per_model =
    List.map
      (fun m ->
        let c = ref 0 in
        Array.iteri
          (fun i b -> if device.(i) <> m.result b then incr c)
          finite_bits ;
        (m.name, !c))
      models
  in
  let unexplained = ref 0 and first = ref (-1) in
  Array.iteri
    (fun i b ->
      if not (List.exists (fun m -> device.(i) = m.result b) models) then begin
        incr unexplained ;
        if !first < 0 then first := i
      end)
    finite_bits ;
  {
    per_model;
    exact_matches =
      List.filter_map (fun (n, c) -> if c = 0 then Some n else None) per_model;
    unexplained = !unexplained;
    first_unexplained = !first;
  }

let print_classification ~label c =
  Printf.printf "  %s\n" label ;
  List.iter
    (fun (n, d) ->
      Printf.printf
        "    %-34s : %6d / %d disagreements%s\n"
        n
        d
        (Array.length finite_bits)
        (if d = 0 then "   <== EXACT, element-wise" else ""))
    c.per_model ;
  if c.unexplained = 0 then
    Printf.printf
      "    every input matches at least one named model (0 unexplained)\n"
  else begin
    let b = finite_bits.(c.first_unexplained) in
    Printf.printf
      "    *** %d inputs match NO named model; first at x = %.9g (0x%04X) ***\n"
      c.unexplained
      (dec b)
      b
  end

(* §1.3's ceiling, evaluated AT THE NARROWING WHERE THE ROUNDING WAS ELIDED.
   Both denominators are reported because §1.3 requires a gate to name which
   value it measures the ulp against. The final-value figure is printed
   alongside solely to show what the pre-correction formulation would have
   produced. *)
let ceiling_report ~model ~strict =
  match (model.at_inner_narrowing, strict.at_inner_narrowing) with
  | None, _ | _, None ->
      Printf.printf
        "    %-34s : ceiling NOT APPLICABLE — the model materialises no value \
         at the elided narrowing\n"
        model.name
  | Some vm, Some vs ->
      let worst_vs = ref 0.0
      and worst_vm = ref 0.0
      and worst_final = ref 0.0
      and over = ref 0
      and worst_b = ref (-1) in
      Array.iter
        (fun b ->
          let a = vs b and c = vm b in
          if Float.is_finite a && Float.is_finite c then begin
            let d = Float.abs (c -. a) in
            let u_vs = d /. ulp16 a and u_vm = d /. ulp16 c in
            if u_vs > !worst_vs then begin
              worst_vs := u_vs ;
              worst_b := b
            end ;
            if u_vm > !worst_vm then worst_vm := u_vm ;
            if u_vs > 1.0 && u_vm > 1.0 then incr over
          end ;
          let fa = dec (strict.result b) and fc = dec (model.result b) in
          if Float.is_finite fa && Float.is_finite fc then begin
            let df = Float.abs (fc -. fa) in
            let u =
              df /. ulp16 (if Float.abs fa < Float.abs fc then fa else fc)
            in
            if u > !worst_final then worst_final := u
          end)
        finite_bits ;
      Printf.printf
        "    %-34s : at the elided narrowing, worst = %.6g ulp (denominator: \
         S_strict's value there) / %.6g ulp (denominator: the model's own \
         value there); %d inputs exceed 1 ulp under BOTH denominators. On the \
         FINAL value the same deviation reaches %.6g ulp.\n"
        model.name
        !worst_vs
        !worst_vm
        !over
        !worst_final
