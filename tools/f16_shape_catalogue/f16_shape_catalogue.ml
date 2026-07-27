(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * backlog-151 — the 20-shape f16 catalogue, and the GENERATIVE RULE as code.
 *
 * WHAT QUESTION THIS MODULE EXISTS TO ANSWER.
 *
 * docs/fp-contraction-policy.md §12.4 closes slice 1 with one candidate rule
 * that produces all five of its results:
 *
 *   "An f32->f16 narrowing absorbs the entire f32 expression tree feeding it,
 *    evaluating it exactly and rounding once — intermediate binary16
 *    narrowings included, hence elided — cut wherever SPIR-V NoContraction
 *    forbids a multiply-add contraction, at which cut a correctly-rounded
 *    binary32 value is materialised."
 *
 * and marks it **unverified as a general rule**. The stakes are structural,
 * not a coverage metric: if the rule holds, docs/design/f16-relaxed-accuracy.md
 * §1.2's model set is a handful of NAMED GENERATORS applied to whatever shape
 * the user wrote; if it fails, §1.2 degrades to a per-expression lookup table.
 *
 * SO THE MODELS HERE ARE GENERATORS, NOT ENTRIES.
 *
 * F16_model_set hand-writes seven closed forms for two specific shapes. This
 * module writes FIVE POLICIES, each a way of deciding which mandated roundings
 * a compiler elides, and applies them to a 20-shape expression catalogue. Five
 * names for twenty shapes is the whole point: a model set that grows with the
 * shape catalogue is a lookup table wearing a contract's clothing.
 *
 * The five policies are not new semantics. On the two shapes slice 1 measured
 * they must reproduce F16_model_set's hand-written functions BIT-FOR-BIT, and
 * [calibrate] refuses to let anything else be read until they do. That is the
 * hinge of this module's credibility: the generator is trusted on the other 18
 * shapes precisely because it is pinned to slice 1's four named models on the
 * two shapes that have device numbers.
 *
 * ONE SOURCE OF TRUTH FOR THE SHAPE.
 *
 * The device source and the host model are generated from the SAME expression
 * tree. A hand-written GLSL kernel next to a hand-written host reference can
 * drift apart silently and the sweep then measures the drift; here they cannot,
 * because there is only one tree.
 *
 * EXACT ARITHMETIC, AND WHY ORDINARY FLOATS WILL NOT DO.
 *
 * The absorbing policies round a value to binary16 in a SINGLE step from a sum
 * binary64 cannot hold: the exact product x * fl32(1.1) is a multiple of 2^-47
 * while the addend reaches 2^9, so the exact sum spans 65 bits against
 * binary64's 53. Evaluating it in OCaml floats would round first and the model
 * would be a DIFFERENT FUNCTION — differing exactly at the binary16 ties, which
 * is where §1.3's counterexample lives. So the evaluator below works on
 * Shewchuk floating-point EXPANSIONS (exact, unevaluated sums of non-
 * overlapping binary64 values) with a sticky flag for the one operation that
 * cannot be exact, division.
 ******************************************************************************)

module M = F16_model_set

(* ========================================================================= *)
(* 1. Exact arithmetic: Shewchuk expansions + a sticky residual              *)
(* ========================================================================= *)

(* An exact real value, represented as:
     - [terms]: non-overlapping binary64 components in INCREASING magnitude
       whose sum is exact. Interior zeros are eliminated, Shewchuk-style, but
       the LEADING component is always kept even when it is a zero — because
       binary16 has two zeros, [f16_bits] distinguishes them, and the HIP
       reference this catalogue mirrors propagates the sign of zero through
       every shape. Dropping it would make the models disagree with the device
       on exactly one input, x = -0, which is precisely the kind of
       one-input-wide difference this whole exercise exists to detect.
     - [st]: the sign of a residual too small to represent, strictly below the
       magnitude of the smallest term. Only division ever sets it.
   The value is  (sum terms) + (an unrepresented residual of sign [st]). *)
type ex = {terms : float list; st : int}

exception Not_exactly_computable of string

let nec fmt = Printf.ksprintf (fun s -> raise (Not_exactly_computable s)) fmt

let single v = {terms = [v]; st = 0}

let zero = single 0.0

(* Knuth's twoSum: [a +. b] is EXACTLY [s +. e], no magnitude assumption. *)
let two_sum a b =
  let s = a +. b in
  let bb = s -. a in
  let err = a -. (s -. bb) +. (b -. bb) in
  (s, err)

(* Dekker/FMA twoProd: [a *. b] is EXACTLY [p +. e]. Exact because [Float.fma]
   computes a*b with a single rounding, so the residual is representable. *)
let two_prod a b =
  let p = a *. b in
  let e = Float.fma a b (-.p) in
  (p, e)

(* Shewchuk grow_expansion_zeroelim: add a scalar to an expansion, exactly. *)
let grow e b =
  let rec go q acc = function
    | [] -> List.rev (q :: acc)
    | ei :: t ->
        let s, err = two_sum q ei in
        go s (if err <> 0.0 then err :: acc else acc) t
  in
  go b [] e

(* Exact sum of two expansions. *)
let expansion_sum e f = List.fold_left grow e f

(* Shewchuk scale_expansion_zeroelim: expansion times a scalar, exactly. *)
let scale e b =
  match e with
  | [] -> []
  | e0 :: rest ->
      let q0, h0 = two_prod e0 b in
      let acc = if h0 <> 0.0 then [h0] else [] in
      let rec go q acc = function
        | [] -> List.rev (q :: acc)
        | ei :: t ->
            let pi, ei' = two_prod ei b in
            let s1, e1 = two_sum q ei' in
            let acc = if e1 <> 0.0 then e1 :: acc else acc in
            let s2, e2 = two_sum pi s1 in
            let acc = if e2 <> 0.0 then e2 :: acc else acc in
            go s2 acc t
      in
      go q0 acc rest

let ex_neg a = {terms = List.map (fun v -> -.v) a.terms; st = -a.st}

let ex_add a b = {terms = expansion_sum a.terms b.terms; st = a.st + b.st}

let ex_sub a b = ex_add a (ex_neg b)

(* Every multiply in the 20-shape catalogue has at least one operand that is a
   single binary64 value (a binary16 input, an f32 constant, or an already-
   rounded intermediate). [scale] is then exact. A multiply of two genuine
   expansions is refused loudly rather than approximated. *)
let ex_mul a b =
  if a.st <> 0 || b.st <> 0 then
    nec "multiply of a value carrying an inexact residual" ;
  match (a.terms, b.terms) with
  | [], _ | _, [] -> zero
  | _, [s] -> {terms = scale a.terms s; st = 0}
  | [s], _ -> {terms = scale b.terms s; st = 0}
  | _ -> nec "multiply of two multi-term expansions"

(* Division is the one operation that cannot be exact. [q] is the binary64
   quotient — correctly rounded, so within 2^-53 relative of the truth — and
   the FMA residual gives the exact SIGN of what is missing. That is enough for
   a single correct rounding to binary32 or binary16: a target ulp is at least
   2^-24 relative, so a residual below 2^-53 relative can only ever matter when
   the quotient sits exactly on a target tie, and there the sign decides. *)
let ex_div a b =
  if a.st <> 0 || b.st <> 0 then nec "divide involving an inexact residual" ;
  match (a.terms, b.terms) with
  | [], _ -> zero
  | _, [] -> nec "division by zero"
  | [x], [y] ->
      let q = x /. y in
      if not (Float.is_finite q) then {terms = [q]; st = 0}
      else
        let r = Float.fma (-.q) y x in
        let st = if r = 0.0 then 0 else if r > 0.0 = (y > 0.0) then 1 else -1 in
        (* [q] is kept even when it is a zero: (-0)/3 is -0, and binary16 has
           two zeros. An earlier revision dropped it and the model then claimed
           +0 where both IEEE and the device say -0 — a one-input-wide error, on
           the exact input a coarse sweep is least likely to look at. *)
        {terms = [q]; st}
  | _ -> nec "division of a multi-term expansion"

(* sqrt is exact only when the argument is a perfect square of a binary64. In
   this catalogue it appears once, as sqrt(x*x) for a binary16 x — where x*x is
   exact in binary64 (22 significand bits) and its root is |x| exactly. Any
   other use is refused. *)
let ex_sqrt a =
  match (a.terms, a.st) with
  | [], 0 -> zero
  | [v], 0 when v >= 0.0 ->
      let r = sqrt v in
      if r *. r = v then single r else nec "sqrt is not exact on this argument"
  | _ -> nec "sqrt of an expansion or of a negative value"

(* floor is exact on a single-term expansion, which is what the catalogue's one
   floor sees (floor(x * 1.1), whose argument is 35 significand bits). *)
let ex_floor a =
  match (a.terms, a.st) with
  | [], 0 -> zero
  | [v], 0 -> single (Float.floor v)
  | _ -> nec "floor of a multi-term expansion or an inexact value"

(* The residual's sign: the sign of everything below the largest term. The
   components are non-overlapping and decreasing, so the sign of their sum is
   the sign of the largest of them; below those, [st]. *)
let residual_sign a =
  match List.rev a.terms with
  | [] | [_] -> a.st
  | _ :: second :: _ -> if second > 0.0 then 1 else -1

(* Round the exact value to [prec] significand bits with smallest subnormal
   2^[emin_sub], in ONE step. Delegates the rounding itself to
   F16_model_set.round_dd, so this module and slice 1's cannot round
   differently: only the sign of the residual is passed, which is all round_dd
   consults it for (the leading term already fixes which ulp interval the value
   is in — the rest is non-overlapping and hence below half an ulp64 of it). *)
let round_ex ~prec ~emin_sub a =
  match List.rev a.terms with
  | [] -> 0.0
  | hi :: rest ->
      let s = residual_sign a in
      (* A zero leading term means the value IS a zero (the components are
         decreasing, so nothing can be below a zero). Return it as it is:
         round_dd's [s = 0.0 -> e] branch would lose the sign, and binary16
         has two zeros. *)
      if hi = 0.0 && rest = [] && s = 0 then hi
      else
        let e = if s = 0 then 0.0 else float_of_int s *. ldexp 1.0 (-1074) in
        M.round_dd ~prec ~emin_sub (hi, e)

let to_f32 a = single (round_ex ~prec:24 ~emin_sub:(-149) a)

let to_f16_bits a = M.f16_bits (round_ex ~prec:11 ~emin_sub:(-24) a)

let to_f16_value a = M.dec (to_f16_bits a)

(* ========================================================================= *)
(* 2. The expression language                                                *)
(* ========================================================================= *)

type e =
  | X  (** the widened binary16 input, exact *)
  | K of float * string
      (** an f32 constant: exact value, and its source text *)
  | Add of e * e
  | Sub of e * e
  | Mul of e * e
  | Div of e * e
  | Fma of e * e * e
  | Sqrt of e
  | Floor of e
  | Sel of e * e  (** [x > 0.0 ? a : b] *)
  | Nar of e  (** an f32 -> f16 narrowing, widened back to f32 *)

let k11 = K (M.f32 1.1, "1.1")

let k09 = K (M.f32 0.9, "0.9")

let k1000 = K (1000.0, "1000.0")

let k3 = K (3.0, "3.0")

let k0 = K (0.0, "0.0")

(* ========================================================================= *)
(* 3. The five policies — the generative rule, and its neighbours            *)
(* ========================================================================= *)

(* Each policy is a decision about which mandated roundings a compiler elides.
   The naming is F16_model_set's and deliberately so: these are the SAME four
   named members of §1.2, restated as functions of an arbitrary shape rather
   than as closed forms for two of them. *)
type policy = {
  pname : string;
  pdescr : string;
  exact : bool;  (** f32 operations are not rounded — the absorbing policies *)
  inner_nar : bool;  (** intermediate binary16 narrowings are performed *)
  fuse_mul : bool;
      (** a multiply/fma DIRECTLY consumed by a narrowing is absorbed into it,
          and nothing else is — §1.2's [S_fuse_mul_into_narrowing], verbatim *)
  cut : bool;
      (** NoContraction: a multiply feeding an addition, and an explicit fma,
          materialise a correctly-rounded binary32 value *)
}

let p_strict =
  {
    pname = "S_strict";
    pdescr = "every mandated rounding performed (the interpreter)";
    exact = false;
    inner_nar = true;
    fuse_mul = false;
    cut = false;
  }

let p_fuse_mul =
  {
    pname = "S_fuse_mul_into_narrowing";
    pdescr = "every f32 multiply immediately consumed by a narrowing absorbed";
    exact = false;
    inner_nar = true;
    fuse_mul = true;
    cut = false;
  }

let p_absorb_all =
  {
    pname = "S_absorb_all_into_final_narrowing";
    pdescr =
      "the whole f32 tree, intermediate narrowings included, absorbed into the \
       final narrowing: ONE rounding";
    exact = true;
    inner_nar = false;
    fuse_mul = false;
    cut = false;
  }

let p_cut_mul =
  {
    pname = "S_f32_mul_then_absorb_add";
    pdescr =
      "the same, cut where NoContraction forbids a multiply-add: the multiply \
       keeps its own binary32 result";
    exact = true;
    inner_nar = false;
    fuse_mul = false;
    cut = true;
  }

let p_drop_inner =
  {
    pname = "S_drop_intermediate_narrowing";
    pdescr =
      "intermediate binary16 narrowings dropped; every f32 op still rounds to \
       f32 (the IGC signature)";
    exact = false;
    inner_nar = false;
    fuse_mul = false;
    cut = false;
  }

(* THE RULE UNDER TEST is exactly two of these five: [p_absorb_all] is the rule
   with nothing cut, [p_cut_mul] is the rule with the NoContraction cut applied.
   The other three are its neighbours, present so that a shape which fails the
   rule is not merely reported as "no match" but placed. *)
let all_policies = [p_strict; p_fuse_mul; p_absorb_all; p_cut_mul; p_drop_inner]

let rule_plain = p_absorb_all

let rule_precise = p_cut_mul

(* ------------------------------------------------------------------------ *)
(* THE CORRECTED RULE — absorption is LOCAL, not whole-tree.                 *)
(*                                                                           *)
(* The five policies above are slice 1's. Shapes A11, A12 and B4 refute the   *)
(* whole-tree reading of §12.4's rule, and the ISA says exactly why: the      *)
(* absorbing instruction is v_fma_mixlo_f16, which takes ONE multiply-add and *)
(* ONE conversion. It cannot reach past an operation that is not one of those *)
(* — a v_floor_f32 (A11) or a v_cndmask_b32 (A12) between the multiply and    *)
(* the narrowing leaves the multiply materialised as its own v_fma_mix_f32,   *)
(* and the result is S_strict.                                               *)
(*                                                                           *)
(* B4 refutes it harder: with two intermediate narrowings elided, ACO emits a *)
(* v_fma_mix_f32 contracting x*1.1+1000 into ONE binary32 rounding and only   *)
(* THEN absorbs the final multiply into the narrowing. That is two separate   *)
(* single-rounding events at two different precisions, which no whole-tree    *)
(* model can be.                                                             *)
(*                                                                           *)
(* So the corrected rule is stated as a LOCAL peephole and evaluated below:   *)
(*                                                                           *)
(*   Each f32->f16 narrowing absorbs the single f32 operation immediately     *)
(*   feeding it — a multiply, an add/sub, or an explicit fma — evaluating it  *)
(*   exactly from its operands and rounding once. An intermediate binary16    *)
(*   narrowing whose value is consumed only by f32 arithmetic is elided.      *)
(*   Independently, a multiply feeding an addition is contracted into a       *)
(*   single-rounded binary32 fma. EVERY OTHER f32 OPERATION KEEPS ITS OWN     *)
(*   CORRECTLY-ROUNDED BINARY32 RESULT. NoContraction removes the second      *)
(*   clause only; it does not reach the narrowing's own absorption, nor a     *)
(*   plain multiply, nor an explicit fma.                                     *)
(* ------------------------------------------------------------------------ *)

(* The rule is ONE semantics with three boolean knobs, not one rule per driver
   and certainly not one per shape. Each (driver, decoration) pair picks a
   setting; the settings measured on this workstation are [aco_vulkan_plain],
   [aco_vulkan_precise] and [aco_opencl] below.

   The knobs exist because rusticl and RADV — the SAME ACO backend behind two
   different front ends — measured differently, which slice 1 could not see
   with two shapes: on B1 rusticl keeps the intermediate narrowing where RADV
   elides it, and on A12 rusticl sinks the conversion into the arms of a select
   where RADV does not. Both facts are invisible unless the catalogue is swept. *)
type lrule = {
  lname : string;
  contract : bool;  (** a multiply feeding an add becomes one f32 fma *)
  elide_inner : bool;  (** intermediate binary16 narrowings are dropped *)
  sink_select : bool;  (** a narrowing sinks into the arms of a select *)
}

let aco_vulkan_plain =
  {
    lname = "R_local_absorb";
    contract = true;
    elide_inner = true;
    sink_select = false;
  }

let aco_vulkan_precise =
  {
    lname = "R_local_absorb_nocontract";
    contract = false;
    elide_inner = true;
    sink_select = false;
  }

let aco_opencl =
  {
    lname = "R_local_absorb_opencl";
    contract = false;
    elide_inner = false;
    sink_select = true;
  }

let local_rules = [aco_vulkan_plain; aco_vulkan_precise; aco_opencl]

(* An ELIDED intermediate narrowing is not there as far as the peephole is
   concerned, so "is this absorbable?" and "is this a multiply feeding an add?"
   must look through one — but only when the rule elides it. *)
let rec strip_nar r e =
  match e with Nar s when r.elide_inner -> strip_nar r s | e -> e

let absorbable r e =
  match strip_nar r e with
  | Mul _ | Add _ | Sub _ | Fma _ -> true
  | Sel _ -> r.sink_select
  | X | K _ | Div _ | Sqrt _ | Floor _ | Nar _ -> false

let is_mul r e = match strip_nar r e with Mul _ -> true | _ -> false

(* [unrounded] means "my consumer performs the rounding": either the narrowing
   that absorbs me, or the fma my multiply is contracted into. *)
let rec ev_local r x ~unrounded e =
  let rd v = if unrounded then v else to_f32 v in
  let sub s = ev_local r x ~unrounded:false s in
  let addend s =
    (* the operand of an addition: a multiply is contracted into the fma unless
       NoContraction is in force *)
    if r.contract && is_mul r s then ev_local r x ~unrounded:true s else sub s
  in
  match e with
  | X -> single x
  | K (v, _) -> single v
  | Nar s ->
      if r.elide_inner then ev_local r x ~unrounded s
      else
        (* the narrowing is performed, and absorbs the one op feeding it *)
        single (to_f16_value (ev_local r x ~unrounded:(absorbable r s) s))
  | Add (a, b) -> rd (ex_add (addend a) (addend b))
  | Sub (a, b) -> rd (ex_sub (addend a) (addend b))
  | Mul (a, b) -> rd (ex_mul (sub a) (sub b))
  | Div (a, b) -> rd (ex_div (sub a) (sub b))
  | Fma (a, b, c) -> rd (ex_add (ex_mul (sub a) (sub b)) (sub c))
  | Sqrt a -> rd (ex_sqrt (sub a))
  | Floor a -> rd (ex_floor (sub a))
  | Sel (a, b) ->
      let arm = if x > 0.0 then a else b in
      if unrounded && r.sink_select then
        (* the conversion is sunk into this arm and absorbs the arm's own top
           operation; that is one narrowing per arm, not one for the select *)
        ev_local r x ~unrounded:(absorbable r arm) arm
      else sub arm

let local_result r expr x_bits =
  let x = M.dec x_bits in
  match expr with
  | Nar s -> to_f16_bits (ev_local r x ~unrounded:(absorbable r s) s)
  | e -> to_f16_bits (ev_local r x ~unrounded:false e)

(* [local_model] is completed after the ceiling helpers below, because §1.3's
   ceiling needs the value the rule presents at the elided narrowing and that
   needs [deepest_nar_in]. *)

(* ========================================================================= *)
(* 4. The evaluator                                                          *)
(* ========================================================================= *)

let rec strip_elided pol e =
  match e with Nar s when not pol.inner_nar -> strip_elided pol s | e -> e

let is_mul_or_fma = function Mul _ | Fma _ -> true | _ -> false

(* [unrounded] suppresses the f32 rounding of the TOP operation only: it is how
   [fuse_mul] expresses "absorbed into the narrowing that consumes it". *)
let rec ev pol x ~unrounded e =
  let r32 v = if pol.exact || unrounded then v else to_f32 v in
  let sub s = ev pol x ~unrounded:false s in
  (* Under the NoContraction cut, an operand of an addition that is (through
     any elided narrowing) a multiply or an fma materialises its binary32
     value. This is the single place the rule's "cut" lives. *)
  let cut_operand s =
    let v = sub s in
    if pol.exact && pol.cut && is_mul_or_fma (strip_elided pol s) then to_f32 v
    else v
  in
  match e with
  | X -> single x
  | K (v, _) -> single v
  | Nar s ->
      if not pol.inner_nar then sub s
      else
        let inner_unrounded = pol.fuse_mul && is_mul_or_fma s in
        single (to_f16_value (ev pol x ~unrounded:inner_unrounded s))
  | Add (a, b) -> r32 (ex_add (cut_operand a) (cut_operand b))
  | Sub (a, b) -> r32 (ex_sub (cut_operand a) (cut_operand b))
  | Mul (a, b) -> r32 (ex_mul (sub a) (sub b))
  | Div (a, b) -> r32 (ex_div (sub a) (sub b))
  | Fma (a, b, c) ->
      (* An EXPLICIT fma is not a contraction: the author wrote the fused
         operation, so there is no multiply-add for NoContraction to forbid and
         the cut does not apply. That reading is the rule's own words ("cut
         wherever NoContraction forbids a multiply-add CONTRACTION"), and it is
         not a matter of taste — shape A8 separates it from the eager reading
         that also cuts explicit fmas, and RADV measured on the literal side.
         See docs/measurements/f16-shapes-2026-07-27/vulkan-radv-eager-cut.txt,
         where A8 `precise` reports S_absorb_all rather than the cut model. *)
      r32 (ex_add (ex_mul (sub a) (sub b)) (sub c))
  | Sqrt a -> r32 (ex_sqrt (sub a))
  | Floor a -> r32 (ex_floor (sub a))
  | Sel (a, b) -> if x > 0.0 then sub a else sub b

(* The kernel's result, as a binary16 bit pattern. The outermost narrowing
   always rounds — it is the one that writes memory. *)
let result pol expr x_bits =
  let x = M.dec x_bits in
  match expr with
  | Nar s ->
      let unrounded = pol.fuse_mul && is_mul_or_fma s in
      to_f16_bits (ev pol x ~unrounded s)
  | e -> to_f16_bits (ev pol x ~unrounded:false e)

(* §1.3's ceiling is evaluated AT THE NARROWING WHERE THE ROUNDING WAS ELIDED,
   and its derivation is "the elision of exactly ONE round-to-nearest step".
   That phrasing has a consequence nobody needed before: a shape with TWO
   intermediate narrowings has two candidate evaluation points, and at the
   OUTER of them the absorbing policies have elided two roundings, not one, so
   the derivation does not cover it and the measured figure is unbounded.
   Shape B4 is that shape and it is the first in the project.
   So the ceiling is evaluated at the INNERMOST intermediate narrowing, where
   exactly one elision separates the policies, and [inner_narrowing_count]
   below is reported so a >1 row is read as partial rather than as a clean
   pass. *)
let rec deepest_nar_in e =
  match e with
  | Nar s -> ( match deepest_nar_in s with Some d -> Some d | None -> Some e)
  | Add (a, b) | Sub (a, b) | Mul (a, b) | Div (a, b) | Sel (a, b) -> (
      match deepest_nar_in a with None -> deepest_nar_in b | s -> s)
  | Fma (a, b, c) -> (
      match deepest_nar_in a with
      | None -> (
          match deepest_nar_in b with None -> deepest_nar_in c | s -> s)
      | s -> s)
  | Sqrt a | Floor a -> deepest_nar_in a
  | X | K _ -> None

let rec count_nar e =
  match e with
  | Nar s -> 1 + count_nar s
  | Add (a, b) | Sub (a, b) | Mul (a, b) | Div (a, b) | Sel (a, b) ->
      count_nar a + count_nar b
  | Fma (a, b, c) -> count_nar a + count_nar b + count_nar c
  | Sqrt a | Floor a -> count_nar a
  | X | K _ -> 0

(* Intermediate narrowings only: the final one is not "elided" by anything. *)
let inner_narrowing_count expr =
  match expr with Nar s -> count_nar s | e -> count_nar e

let at_inner_narrowing pol expr =
  match match expr with Nar s -> deepest_nar_in s | _ -> None with
  | None -> None
  | Some (Nar s) ->
      Some
        (fun b ->
          let x = M.dec b in
          if pol.inner_nar then
            let inner_unrounded = pol.fuse_mul && is_mul_or_fma s in
            to_f16_value (ev pol x ~unrounded:inner_unrounded s)
          else
            (* the narrowing is elided: what reaches the rest of the expression
               is the unrounded f32-tree value at that position *)
            let v = ev pol x ~unrounded:false s in
            let v =
              if pol.exact && pol.cut && is_mul_or_fma s then to_f32 v else v
            in
            round_ex ~prec:53 ~emin_sub:(-1074) v)
  | Some _ -> None

(* The value the corrected local rule presents where S_strict materialises its
   innermost intermediate binary16 value. The rule elides that narrowing, so
   what reaches the rest of the expression is the subtree's value UNROUNDED —
   which is precisely the claim "this rounding was elided", and therefore the
   right thing for §1.3's ceiling to measure against. *)
let local_at_inner r expr =
  match match expr with Nar s -> deepest_nar_in s | _ -> None with
  | Some (Nar s) ->
      Some
        (fun b ->
          round_ex
            ~prec:53
            ~emin_sub:(-1074)
            (ev_local r (M.dec b) ~unrounded:true s))
  | _ -> None

let local_model r expr =
  {
    M.name = r.lname;
    M.descr =
      Printf.sprintf
        "the corrected LOCAL rule (contract=%b, elide_inner=%b, \
         sink_select=%b): each narrowing absorbs the one operation feeding it; \
         every other f32 op keeps its own binary32 result"
        r.contract
        r.elide_inner
        r.sink_select;
    M.result = local_result r expr;
    M.at_inner_narrowing =
      (match local_at_inner r expr with
      | Some f -> Some f
      | None -> Some (fun b -> M.dec (local_result r expr b)));
  }

(* ========================================================================= *)
(* 5. The 20 shapes                                                          *)
(* ========================================================================= *)

type shape = {
  id : string;
  descr : string;
  expr : e;
  discriminating_note : string;
      (** stated up front where something OTHER than the absorption rule governs
          the shape, so a mismatch is not silently read as evidence against the
          rule *)
}

let shapes =
  [
    {id = "A1"; descr = "narrow x"; expr = Nar X; discriminating_note = ""};
    {
      id = "A2";
      descr = "narrow (x *. 1.1)";
      expr = Nar (Mul (X, k11));
      discriminating_note = "";
    };
    {
      id = "A3";
      descr = "narrow (x +. 1000.)";
      expr = Nar (Add (X, k1000));
      discriminating_note = "";
    };
    {
      id = "A4";
      descr = "narrow (x -. 1000.)";
      expr = Nar (Sub (X, k1000));
      discriminating_note = "";
    };
    {
      id = "A5";
      descr = "narrow (x /. 3.)";
      expr = Nar (Div (X, k3));
      discriminating_note =
        "DIVISION: Vulkan/SPIR-V allow OpFDiv 2.5 ULP, so a mismatch here is \
         evidence about fdiv precision and NOT about the absorption rule";
    };
    {
      id = "A6";
      descr = "narrow (x *. 1.1 +. 1000.)";
      expr = Nar (Add (Mul (X, k11), k1000));
      discriminating_note = "";
    };
    {
      id = "A7";
      descr = "narrow ((x +. 1000.) *. 1.1)";
      expr = Nar (Mul (Add (X, k1000), k11));
      discriminating_note = "";
    };
    {
      id = "A8";
      descr = "narrow (fma x 1.1 1000.)";
      expr = Nar (Fma (X, k11, k1000));
      discriminating_note = "";
    };
    {
      id = "A9";
      descr = "narrow (sqrt (x *. x))";
      expr = Nar (Sqrt (Mul (X, X)));
      discriminating_note =
        "SQRT: exact here only because sqrt(x*x) = |x| for a binary16 x; the \
         device's sqrt precision is separately specified";
    };
    {
      id = "A10";
      descr = "narrow (0. -. x)";
      expr = Nar (Sub (k0, X));
      discriminating_note = "";
    };
    {
      id = "A11";
      descr = "narrow (floor (x *. 1.1))";
      expr = Nar (Floor (Mul (X, k11)));
      discriminating_note = "";
    };
    {
      id = "A12";
      descr = "narrow (if x>0. then x*.1.1 else x*.0.9)";
      expr = Nar (Sel (Mul (X, k11), Mul (X, k09)));
      discriminating_note = "";
    };
    {
      id = "A13";
      descr = "narrow (x *. x)";
      expr = Nar (Mul (X, X));
      discriminating_note = "";
    };
    {
      id = "A14";
      descr = "narrow (x *. 1.1 *. 1.1)";
      expr = Nar (Mul (Mul (X, k11), k11));
      discriminating_note = "";
    };
    {
      id = "A15";
      descr = "narrow (x *. 1.1 +. x /. 3.)";
      expr = Nar (Add (Mul (X, k11), Div (X, k3)));
      discriminating_note =
        "DIVISION: as A5 — an fdiv mismatch is not evidence about absorption";
    };
    {
      id = "B1";
      descr = "narrow (narrow (x *. 1.1) +. 1000.)";
      expr = Nar (Add (Nar (Mul (X, k11)), k1000));
      discriminating_note = "";
    };
    {
      id = "B2";
      descr = "narrow (narrow (x *. 1.1) *. 1.1)";
      expr = Nar (Mul (Nar (Mul (X, k11)), k11));
      discriminating_note = "";
    };
    {
      id = "B3";
      descr = "narrow (narrow (x +. 1000.) *. 1.1)";
      expr = Nar (Mul (Nar (Add (X, k1000)), k11));
      discriminating_note = "";
    };
    {
      id = "B4";
      descr = "narrow (narrow (narrow (x *. 1.1) +. 1000.) *. 1.1)";
      expr = Nar (Mul (Nar (Add (Nar (Mul (X, k11)), k1000)), k11));
      discriminating_note = "";
    };
    {
      id = "C1";
      descr = "out.(i) <- inp.(i)   (f16 -> f16 copy, no cast)";
      expr = X;
      discriminating_note =
        "NO NARROWING AT ALL: there is nothing for any policy to elide, so \
         every model coincides by construction";
    };
  ]

let shape_by_id id = List.find (fun s -> s.id = id) shapes

(* ========================================================================= *)
(* 6. Source emission — the SAME tree the models are evaluated from          *)
(* ========================================================================= *)

type dialect = Glsl | Opencl

(* Three-address form, one temporary per operation, so that the `precise`
   variant differs from the plain one by exactly one keyword per declaration —
   which is what Sarek_ir_glsl.gen_var_decl emits on every float local. *)
type emitter = {
  buf : Buffer.t;
  mutable n : int;
  dialect : dialect;
  qual : string;  (** "precise " or "" *)
  barrier : bool;  (** round-trip every temporary through the volatile store *)
}

let fresh em =
  em.n <- em.n + 1 ;
  Printf.sprintf "t%d" em.n

let line em s = Buffer.add_string em.buf ("  " ^ s ^ "\n")

(* The volatile round-trip. On GLSL it is the SSBO the shader writes; on OpenCL
   it is a volatile __local slot. Both are the constructions slice 1 measured
   to restore the discipline on their respective stacks. *)
let bar_f32 em v =
  if not em.barrier then v
  else
    let t = fresh em in
    (match em.dialect with
    | Glsl ->
        line em (Printf.sprintf "outb[i] = floatBitsToUint(%s);" v) ;
        line em (Printf.sprintf "float %s = uintBitsToFloat(outb[i]);" t)
    | Opencl ->
        line em (Printf.sprintf "s[l] = %s;" v) ;
        line em (Printf.sprintf "float %s = s[l];" t)) ;
    t

let decl_f32 em rhs =
  let t = fresh em in
  line em (Printf.sprintf "%sfloat %s = %s;" em.qual t rhs) ;
  bar_f32 em t

let rec emit em e =
  match e with
  | X -> "x"
  | K (_, lit) -> ( match em.dialect with Glsl -> lit | Opencl -> lit ^ "f")
  | Add (a, b) -> decl_f32 em (Printf.sprintf "%s + %s" (emit em a) (emit em b))
  | Sub (a, b) -> decl_f32 em (Printf.sprintf "%s - %s" (emit em a) (emit em b))
  | Mul (a, b) -> decl_f32 em (Printf.sprintf "%s * %s" (emit em a) (emit em b))
  | Div (a, b) -> decl_f32 em (Printf.sprintf "%s / %s" (emit em a) (emit em b))
  | Fma (a, b, c) ->
      let a = emit em a and b = emit em b and c = emit em c in
      decl_f32 em (Printf.sprintf "fma(%s, %s, %s)" a b c)
  | Sqrt a -> decl_f32 em (Printf.sprintf "sqrt(%s)" (emit em a))
  | Floor a -> decl_f32 em (Printf.sprintf "floor(%s)" (emit em a))
  | Sel (a, b) ->
      let a = emit em a and b = emit em b in
      decl_f32
        em
        (Printf.sprintf
           "(x > 0.0%s) ? %s : %s"
           (match em.dialect with Glsl -> "" | Opencl -> "f")
           a
           b)
  | Nar a ->
      let v = emit em a in
      let h = fresh em and t = fresh em in
      (match (em.dialect, em.barrier) with
      | Glsl, false ->
          line em (Printf.sprintf "float16_t %s = float16_t(%s);" h v) ;
          line em (Printf.sprintf "%sfloat %s = float(%s);" em.qual t h)
      | Glsl, true ->
          line em (Printf.sprintf "outb[i] = pack(float16_t(%s));" v) ;
          line
            em
            (Printf.sprintf "float16_t %s = unpackFloat2x16(outb[i]).x;" h) ;
          line em (Printf.sprintf "float %s = float(%s);" t h)
      | Opencl, false ->
          line em (Printf.sprintf "half %s = (half)(%s);" h v) ;
          line em (Printf.sprintf "%sfloat %s = (float)%s;" em.qual t h)
      | Opencl, true ->
          line em (Printf.sprintf "out[i] = as_ushort((half)(%s));" v) ;
          line em (Printf.sprintf "half %s = as_half(out[i]);" h) ;
          line em (Printf.sprintf "float %s = (float)%s;" t h)) ;
      t

let n_local = 256

let source ~dialect ~precise ~barrier shape =
  let em =
    {
      buf = Buffer.create 512;
      n = 0;
      dialect;
      qual = (if precise then "precise " else "");
      barrier;
    }
  in
  (* C1 has no cast: the f16 bit pattern is copied straight through. *)
  let body =
    match shape.expr with
    | X ->
        (match dialect with
        | Glsl -> line em "outb[i] = inb[i] & 0xFFFFu;"
        | Opencl -> line em "out[i] = in[i];") ;
        Buffer.contents em.buf
    | Nar s ->
        let v = emit em s in
        (match dialect with
        | Glsl -> line em (Printf.sprintf "outb[i] = pack(float16_t(%s));" v)
        | Opencl -> line em (Printf.sprintf "out[i] = as_ushort((half)(%s));" v)) ;
        Buffer.contents em.buf
    | _ -> assert false
  in
  match dialect with
  | Glsl ->
      Printf.sprintf
        {|#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
layout(local_size_x = %d) in;
layout(std430, binding = 0) volatile buffer Out { uint outb[]; };
layout(std430, binding = 1) readonly buffer In { uint inb[]; };
uint pack(float16_t r) {
  return packFloat2x16(f16vec2(r, float16_t(0.0))) & 0xFFFFu;
}
void main() {
  uint i = gl_GlobalInvocationID.x;
  float x = float(unpackFloat2x16(inb[i]).x);
%s}
|}
        n_local
        body
  | Opencl ->
      Printf.sprintf
        {|#pragma OPENCL EXTENSION cl_khr_fp16 : enable
/* named `midround` so slice 1's probe can compile it through its existing
   run_kernel, which looks that entry point up by name.

   `out` is VOLATILE, mirroring the GLSL side where the SSBO is declared
   volatile. Without it the barrier variant is not a barrier: the f16 round
   trip goes through `out[i]`, and a non-volatile store is forwardable, so the
   compiler removes exactly the narrowing the control exists to preserve. That
   was measured, not feared — shape B2's green control reported
   S_drop_intermediate_narrowing until this qualifier was added. */
__kernel void midround(
    __global volatile ushort *out, __global const ushort *in, int n) {
  __local volatile float s[%d];
  int i = get_global_id(0);
  int l = get_local_id(0);
  (void)s; (void)l;
  if (i < n) {
    float x = (float)as_half(in[i]);
%s  }
}
|}
        n_local
        body

(* ========================================================================= *)
(* 7. Model instantiation and reporting, reusing F16_model_set's classifier   *)
(* ========================================================================= *)

(* A policy applied to a shape becomes an [F16_model_set.model], so the whole
   of slice 1's element-wise classifier, separation matrix and §1.3 ceiling
   machinery is reused unchanged rather than reimplemented. *)
let model_of_policy shape pol =
  {
    M.name = pol.pname;
    M.descr = pol.pdescr;
    M.result = result pol shape.expr;
    M.at_inner_narrowing =
      (match at_inner_narrowing pol shape.expr with
      | Some f -> Some f
      | None -> Some (fun b -> M.dec (result pol shape.expr b)));
  }

(* Duplicate names are impossible (the five policies have five names), but two
   policies CAN coincide as functions on a given shape — on A2 the three
   absorbing policies are the same function, because there is no addition for
   the cut to bite on and no intermediate narrowing to drop. That is reported,
   not hidden: [F16_model_set.separation_matrix] prints it and a shape where
   ALL FIVE coincide measures nothing at all. *)
let models_of shape = List.map (model_of_policy shape) all_policies

(* The full set actually swept: slice 1's five, plus the two the corrected
   local rule generates. The corrected pair is NOT a per-shape addition — it is
   ONE semantics evaluated on whatever tree it is given, which is the whole
   question backlog-151 was asked. On A2 and B1 it collapses onto slice 1's
   members and [check_local_rule_matches_slice1] pins that. *)
let models_with_local shape =
  models_of shape
  @ [
      local_model aco_vulkan_plain shape.expr;
      local_model aco_vulkan_precise shape.expr;
      local_model aco_opencl shape.expr;
    ]

(* The number of DISTINCT functions the five policies induce on a shape. 1 means
   the shape cannot discriminate between any of them and a device agreeing with
   S_strict there is not evidence for anything. *)
let distinct_model_count shape =
  let ms = models_of shape in
  let sig_of m = Array.map m.M.result M.finite_bits in
  let seen = ref [] in
  List.iter
    (fun m ->
      let s = sig_of m in
      if not (List.exists (fun t -> t = s) !seen) then seen := s :: !seen)
    ms ;
  List.length !seen

(* ========================================================================= *)
(* 8. CALIBRATION — the generator is pinned to slice 1's hand-written models *)
(* ========================================================================= *)

exception Calibration_failed of string

let failf fmt = Printf.ksprintf (fun s -> raise (Calibration_failed s)) fmt

let agree_on_all name f g =
  let bad = ref (-1) in
  Array.iter (fun b -> if !bad < 0 && f b <> g b then bad := b) M.finite_bits ;
  if !bad >= 0 then
    failf
      "GENERATOR MISMATCH: %s disagrees with slice 1's hand-written model at x \
       = %.9g (0x%04X): generator 0x%04X, slice 1 0x%04X"
      name
      (M.dec !bad)
      !bad
      (f !bad)
      (g !bad)

(* THE HINGE. The five generic policies must reproduce F16_model_set's
   hand-written closed forms, bit-for-bit on all 63488 inputs, on the two shapes
   slice 1 measured on a device. If they do not, nothing this module says about
   the other 18 shapes is worth reading, because the generator is then not the
   thing slice 1 measured. *)
let check_generator_matches_slice1 () =
  let a2 = (shape_by_id "A2").expr and b1 = (shape_by_id "B1").expr in
  let m name ms = List.find (fun m -> m.M.name = name) ms in
  let s1 = M.shape1_models and s2 = M.shape2_models in
  agree_on_all "A2 / S_strict" (result p_strict a2) (m "S_strict" s1).M.result ;
  agree_on_all
    "A2 / S_fuse_mul_into_narrowing"
    (result p_fuse_mul a2)
    (m "S_fuse_mul_into_narrowing" s1).M.result ;
  agree_on_all "B1 / S_strict" (result p_strict b1) (m "S_strict" s2).M.result ;
  agree_on_all
    "B1 / S_fuse_mul_into_narrowing"
    (result p_fuse_mul b1)
    (m "S_fuse_mul_into_narrowing" s2).M.result ;
  agree_on_all
    "B1 / S_absorb_all_into_final_narrowing"
    (result p_absorb_all b1)
    (m "S_absorb_all_into_final_narrowing" s2).M.result ;
  agree_on_all
    "B1 / S_f32_mul_then_absorb_add"
    (result p_cut_mul b1)
    (m "S_f32_mul_then_absorb_add" s2).M.result ;
  agree_on_all
    "B1 / S_drop_intermediate_narrowing"
    (result p_drop_inner b1)
    (m "S_drop_intermediate_narrowing" s2).M.result

(* The four separation counts §12.2 records, recomputed from the GENERIC
   evaluator rather than from the closed forms. Reproducing a known positive is
   what licenses believing the 18 new columns. *)
let check_recorded_separations () =
  let count f g =
    let n = ref 0 in
    Array.iter (fun b -> if f b <> g b then incr n) M.finite_bits ;
    !n
  in
  let a2 = (shape_by_id "A2").expr and b1 = (shape_by_id "B1").expr in
  let expect what got want =
    if got <> want then
      failf "CALIBRATION FAILED: %s separates on %d, must be %d" what got want
  in
  expect
    "A2 S_strict vs S_fuse_mul_into_narrowing"
    (count (result p_strict a2) (result p_fuse_mul a2))
    2912 ;
  expect
    "B1 S_strict vs S_fuse_mul_into_narrowing"
    (count (result p_strict b1) (result p_fuse_mul b1))
    620 ;
  expect
    "B1 S_strict vs S_absorb_all_into_final_narrowing"
    (count (result p_strict b1) (result p_absorb_all b1))
    5075 ;
  expect
    "B1 S_strict vs S_f32_mul_then_absorb_add"
    (count (result p_strict b1) (result p_cut_mul b1))
    4776 ;
  expect
    "B1 S_strict vs S_drop_intermediate_narrowing"
    (count (result p_strict b1) (result p_drop_inner b1))
    4774

(* Every shape's every policy must be computable exactly. A shape whose model
   raises Not_exactly_computable would otherwise be silently absent from the
   sweep. *)
let check_all_shapes_computable () =
  List.iter
    (fun sh ->
      List.iter
        (fun pol ->
          try ignore (result pol sh.expr M.finite_bits.(31000))
          with Not_exactly_computable m ->
            failf "%s / %s is not exactly computable: %s" sh.id pol.pname m)
        all_policies)
    shapes

(* The corrected local rule must ALSO reduce to slice 1's named members on the
   two shapes slice 1 measured — otherwise it is a new semantics dressed up as
   a correction, and its agreement on the other 18 would say nothing about the
   contract slice 1 admitted. *)
let check_local_rule_matches_slice1 () =
  let a2 = (shape_by_id "A2").expr and b1 = (shape_by_id "B1").expr in
  let m name ms = (List.find (fun m -> m.M.name = name) ms).M.result in
  let s1 = M.shape1_models and s2 = M.shape2_models in
  agree_on_all
    "A2 / aco_vulkan_plain == S_fuse_mul_into_narrowing"
    (local_result aco_vulkan_plain a2)
    (m "S_fuse_mul_into_narrowing" s1) ;
  agree_on_all
    "A2 / aco_vulkan_precise == S_fuse_mul_into_narrowing"
    (local_result aco_vulkan_precise a2)
    (m "S_fuse_mul_into_narrowing" s1) ;
  agree_on_all
    "A2 / aco_opencl == S_fuse_mul_into_narrowing"
    (local_result aco_opencl a2)
    (m "S_fuse_mul_into_narrowing" s1) ;
  agree_on_all
    "B1 / aco_vulkan_plain == S_absorb_all_into_final_narrowing"
    (local_result aco_vulkan_plain b1)
    (m "S_absorb_all_into_final_narrowing" s2) ;
  agree_on_all
    "B1 / aco_vulkan_precise == S_f32_mul_then_absorb_add"
    (local_result aco_vulkan_precise b1)
    (m "S_f32_mul_then_absorb_add" s2) ;
  agree_on_all
    "B1 / aco_opencl == S_fuse_mul_into_narrowing"
    (local_result aco_opencl b1)
    (m "S_fuse_mul_into_narrowing" s2)

let calibrate () =
  M.calibrate () ;
  check_all_shapes_computable () ;
  check_generator_matches_slice1 () ;
  check_recorded_separations () ;
  check_local_rule_matches_slice1 ()
