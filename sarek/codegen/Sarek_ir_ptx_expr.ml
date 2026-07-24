(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** PTX expression emitter: emit_expr, emit_binop, emit_cast, emit_intrinsic.

    All functions emit PTX instructions into a {!Buffer.t} as a side-effect and
    return the PTX register name holding the result. *)

open Sarek_ir_types
open Sarek_ir_ptx_types
open Sarek_ir_ptx_mem

(** f32 bit patterns for the base-2 change-of-base constants used by the exp/log
    lowerings (PTX only provides base-2 ex2/lg2). *)
let f32_log2_e_bits = Int32.bits_of_float (Float.log2 (Float.exp 1.0))

let f32_ln_2_bits = Int32.bits_of_float (Float.log 2.0)

let f32_log10_2_bits = Int32.bits_of_float (Float.log10 2.0)

let f32_two_log2_e_bits = Int32.bits_of_float (2.0 *. Float.log2 (Float.exp 1.0))

let f32_one_bits = Int32.bits_of_float 1.0

let f32_half_bits = Int32.bits_of_float 0.5

(** An f64 register is "%fd<n>"; an f32 register is "%f<n>" (not "%fd<n>"). *)
let is_f64_reg r = String.length r >= 3 && r.[1] = 'f' && r.[2] = 'd'

let is_f32_reg r = String.length r >= 2 && r.[1] = 'f' && not (is_f64_reg r)

(** Whether an expression must NOT be evaluated speculatively — i.e. an [EIf]
    with such a branch must emit real control flow rather than the eager
    evaluate-both-branches [selp] path. Two reasons a subexpression qualifies:
    - it has an observable effect (a store/atomic, or a helper call whose body
      may store/atomic/barrier); running the not-taken branch is wrong;
    - it dereferences memory (an array read); the not-taken branch's index may
      be out of bounds (the classic [if i < n then a.(i) else d] guard), and an
      unconditional load can fault or read garbage. *)
let rec expr_needs_branch_guard (e : expr) : bool =
  let is_atomic name =
    String.length name >= 7 && String.sub name 0 7 = "atomic_"
  in
  (* Barriers/fences emit side-effecting, convergence-sensitive instructions
     (bar.sync, membar); they must never run on a not-taken branch. *)
  let is_barrier = function
    | "block_barrier" | "warp_barrier" | "memory_fence" -> true
    | _ -> false
  in
  match e with
  | EApp _ -> true
  | EIntrinsic (_, name, args) ->
      is_atomic name || is_barrier name
      || List.exists expr_needs_branch_guard args
  (* Array reads must not be evaluated speculatively (out-of-bounds guard). *)
  | EArrayRead _ | EArrayReadExpr _ -> true
  | EConst _ | EVar _ | EArrayLen _ -> false
  | EUnop (_, a) | ECast (_, a) -> expr_needs_branch_guard a
  | EBinop (_, a, b) -> expr_needs_branch_guard a || expr_needs_branch_guard b
  | EIf (c, t, f) ->
      expr_needs_branch_guard c || expr_needs_branch_guard t
      || expr_needs_branch_guard f
  | EArrayCreate (_, s, _) -> expr_needs_branch_guard s
  | EMatch _ | ERecord _ | ERecordField _ | ETuple _ | EVariant _ ->
      (* EMatch is always branch-based; aggregate constructions/projections
         may nest arbitrary subexpressions — conservative answer. *)
      true

(** {1 Match dispatch}

    Shared branch-chain machinery for EMatch (value position) and SMatch
    (statement position): tag-compare branch chain, arm-scoped payload bindings,
    exhaustiveness checking (FR-022, C-9). *)

(** A pattern that matches unconditionally: wildcard, or the variable pattern
    the lowering encodes as [PConstr ("", [x])]. *)
let pattern_is_catch_all = function
  | PWild | PConstr ("", [_]) -> true
  | _ -> false

(** Bind [pat]'s variables against the scrutinee binding [scrut] and return the
    shadowed entries, to be undone with {!restore_bindings} when the arm ends
    (arm-scoped bindings, FR-022). *)
let bind_pattern_vars (env : env) (scrut : binding) (pat : pattern) :
    (string * binding option) list =
  let save_bind name b =
    let prev = Hashtbl.find_opt env name in
    env_bind_binding env name b ;
    (name, prev)
  in
  match (pat, scrut) with
  | PWild, _ -> []
  | PConstr ("", [x]), _ -> [save_bind x scrut]
  | PConstr ("tuple", vars), Agg (ARecord fields) ->
      (* Tuple destructuring: positional components of the anonymous record
         (fields "_0", "_1", ... — FR-024). *)
      if List.length vars <> List.length fields then
        fail
          (Printf.sprintf
             "PTX codegen: tuple pattern binds %d variables but the value has \
              %d components"
             (List.length vars)
             (List.length fields))
      else List.map2 (fun v (_, b) -> save_bind v b) vars fields
  | PConstr (ctor, vars), Agg (AVariant {vname; ctors; _}) -> (
      match List.assoc_opt ctor ctors with
      | None ->
          fail
            (Printf.sprintf
               "PTX codegen: variant type '%s' has no constructor '%s'"
               vname
               ctor)
      | Some payload ->
          if List.length vars <> List.length payload then
            fail
              (Printf.sprintf
                 "PTX codegen: constructor '%s' of '%s' has %d payload \
                  argument(s), pattern binds %d"
                 ctor
                 vname
                 (List.length payload)
                 (List.length vars))
          else List.map2 save_bind vars payload)
  | PConstr (ctor, _), _ ->
      fail
        ("PTX codegen: constructor pattern '" ^ ctor
       ^ "' on a non-variant value; only variant values and tuples can be \
          destructured by match")

(** Undo the env mutations recorded by {!bind_pattern_vars}. *)
let restore_bindings (env : env) saved =
  List.iter
    (fun (name, prev) ->
      match prev with
      | Some b -> Hashtbl.replace env name b
      | None -> Hashtbl.remove env name)
    saved

(** Check that a variant match is exhaustive: a catch-all arm, or every
    constructor covered. Raises a precise error otherwise (C-9). *)
let check_match_exhaustive vname ctors (arms : (pattern * 'a) list) =
  let has_catch_all = List.exists (fun (p, _) -> pattern_is_catch_all p) arms in
  let covered = function
    | cn, _ ->
        List.exists (function PConstr (c, _), _ -> c = cn | _ -> false) arms
  in
  let missing = List.filter (fun c -> not (covered c)) ctors in
  if (not has_catch_all) && missing <> [] then
    fail
      (Printf.sprintf
         "PTX codegen: non-exhaustive match on '%s' (missing: %s); add the \
          missing constructor arm(s) or a wildcard arm"
         vname
         (String.concat ", " (List.map fst missing)))

(** Emit the branch chain of a match: per-arm [setp.eq] on the tag register +
    predicated bra; the last arm (or the first catch-all arm) is branched to
    unconditionally. [tag_of ctor] = the constructor's declaration index (= its
    position in the binding's [ctors] list, aligned with
    [Sarek_ir_layout.ctor_tag]). *)
let emit_match_dispatch buf alloc vname tag_reg ctors arms labels =
  let n = List.length arms in
  let tag_of ctor =
    let rec index i = function
      | [] ->
          fail
            (Printf.sprintf
               "PTX codegen: variant type '%s' has no constructor '%s'"
               vname
               ctor)
      | (cn, _) :: rest -> if cn = ctor then i else index (i + 1) rest
    in
    index 0 ctors
  in
  let rec dispatch i arms_labels =
    match arms_labels with
    | [] -> ()
    | ((pat, _), lbl) :: rest ->
        if i = n - 1 || pattern_is_catch_all pat then
          (* Last arm / catch-all: unconditional (C-9; exhaustiveness was
             checked statically, so falling through every test implies the
             last constructor). *)
          emit buf "bra %s;" lbl
        else begin
          (match pat with
          | PConstr (ctor, _) ->
              let p = new_pred alloc in
              emit buf "setp.eq.u32 %s, %s, %d;" p tag_reg (tag_of ctor) ;
              emit buf "@%s bra %s;" p lbl
          | PWild -> assert false (* catch-all handled above *)) ;
          dispatch (i + 1) rest
        end
  in
  dispatch 0 (List.combine arms labels)

(** [emit_match_arms buf alloc env scrut arms ~emit_arm] emits a full match on
    the scrutinee binding [scrut]. Variant scrutinees get a tag branch chain
    (never selp — FR-022); tuple/record/scalar scrutinees support exactly one
    destructuring arm. [emit_arm] emits an arm body (expression or statement)
    with the arm's pattern variables bound arm-scoped. *)
let emit_match_arms buf alloc (env : env) (scrut : binding)
    (arms : (pattern * 'a) list) ~(emit_arm : 'a -> unit) : unit =
  if arms = [] then fail "PTX codegen: match with no arms" ;
  let emit_one_arm (pat, arm) =
    let saved = bind_pattern_vars env scrut pat in
    emit_arm arm ;
    restore_bindings env saved
  in
  match scrut with
  | Agg (AVariant {vname; tag_reg; ctors}) ->
      check_match_exhaustive vname ctors arms ;
      let labels = List.map (fun _ -> new_label alloc) arms in
      let l_end = new_label alloc in
      emit_match_dispatch buf alloc vname tag_reg ctors arms labels ;
      let n = List.length arms in
      List.iteri
        (fun i ((_, _) as arm) ->
          emit_label buf (List.nth labels i) ;
          emit_one_arm arm ;
          if i < n - 1 then emit buf "bra %s;" l_end)
        arms ;
      emit_label buf l_end
  | _ -> (
      (* Non-variant scrutinee: single-arm destructuring only (tuples,
         variable/wildcard patterns). *)
      match arms with
      | [arm] -> emit_one_arm arm
      | _ ->
          fail
            "PTX codegen: match with multiple arms on a non-variant value; \
             only variant values dispatch on a tag (tuple/record destructuring \
             takes a single pattern)")

(** {1 Recursive-inline budget helpers}

    A recursive helper is inlinable only when its body root carries
    [pragma ["sarek.inline N"]]: N bounds the recursive re-entry depth. *)

(** Inline budget declared by [hf], parsed from an [SPragma] at its body root.
    Minimal re-implementation of the option parsing in
    sarek/ppx/Sarek_tailrec_pragma.parse_sarek_inline_pragma (the source of
    truth for the "sarek.inline N" string format) — the PPX is a separate
    library not linkable from the codegen. *)
let helper_inline_budget (hf : helper_func) : int option =
  (* A negative depth would never reach the [Some 0] exhaustion arm of
     [emit_app_recursive] (the budget decrements past zero), making the
     expansion non-terminating — reject it loudly instead of parsing it. *)
  let checked n_str =
    match int_of_string_opt n_str with
    | Some n when n < 0 ->
        unsupported
          ("pragma [\"sarek.inline " ^ n_str
         ^ "\"]: the inline depth must be >= 0")
    | v -> v
  in
  let parse = function
    | [opt] -> (
        match String.split_on_char ' ' opt with
        | ["sarek.inline"; n] -> checked n
        | _ -> None)
    | ["sarek.inline"; n] -> checked n
    | _ -> None
  in
  let rec root = function SBlock s -> root s | s -> s in
  match root hf.hf_body with SPragma (opts, _) -> parse opts | _ -> None

(** Write a typed zero into every scalar leaf of [b] — the result of a recursive
    call at inline-budget exhaustion (a dynamically-unreachable branch; see
    [emit_app_recursive]). *)
let rec zero_binding buf (b : binding) : unit =
  match b with
  | Scalar r ->
      let zero =
        match reg_class r with
        | RU32 | RU64 -> "0"
        | RF32 -> "0F00000000"
        | RF64 -> "0D0000000000000000"
      in
      emit buf "%s %s, %s;" (mov_op_of_class (reg_class r)) r zero
  | Agg (ARecord fields) ->
      List.iter (fun (_, fb) -> zero_binding buf fb) fields
  | Agg (AVariant {tag_reg; ctors; _}) ->
      emit buf "mov.u32 %s, 0;" tag_reg ;
      List.iter (fun (_, bs) -> List.iter (zero_binding buf) bs) ctors

(** Exact C-semantics fmod (audit finding M1): the single-pass
    [x - trunc(x/y)*y] formula is only exact while the quotient fits the
    mantissa; beyond 2^53 (f64) / 2^24 (f32) the rounded quotient carries
    absolute error >> 1 and the result can leave [0, |y|) or flip sign.
    This emits an iterative reduction instead:

    - outer loop: while |r| >= |y|, subtract trunc(r/y)*y via a single fma
      (each round shrinks |r| by ~2^-mantissa, so it terminates in <= ~20
      rounds for f64 / ~6 for f32; once |r| < |y| the quotient truncates
      to 0 and further rounds are exact no-ops);
    - overflow branch: if r/y overflows to inf (huge x with tiny/subnormal
      y), reduce against y * 2^k (k = 1022 f64 / 126 f32, exact scaling and
      still a multiple of y, so congruence mod y is preserved), retrying
      with a larger scale while the scaled quotient is still inf;
    - sign fix: the last fma can land one |y| past zero on the wrong side;
      one conditional +/-|y| restores the dividend's sign, and a final
      copysign fixes the sign of a +/-0 result;
    - domain guard: y = 0, y = NaN, |x| = inf and x = NaN all produce NaN
      (C fmod contract); x mod inf = x falls out naturally.

    Validated bit-exact against OCaml's [Float.rem] (C fmod) on 200k
    full-exponent-range fuzz cases including subnormals and the
    overflow-scaling path (see commit message). *)
let emit_float_fmod buf alloc ~is64 rx ry : string =
  let newf () = if is64 then new_f64 alloc else new_f32 alloc in
  let s = if is64 then "f64" else "f32" in
  let const b64 b32 =
    if is64 then Printf.sprintf "0D%016LX" b64 else Printf.sprintf "0F%08lX" b32
  in
  let c_inf = const 0x7FF0000000000000L 0x7F800000l in
  let c_nan = const 0x7FF8000000000000L 0x7FC00000l in
  (* 2^1022 / 2^126: largest power-of-two scale that keeps y * scale exact
     and finite for every y small enough to make x/y overflow. *)
  let c_scale = const 0x7FD0000000000000L 0x7E800000l in
  let c_zero = const 0L 0l in
  let rz = newf () in
  emit buf "mov.%s %s, %s;" s rz c_zero ;
  let rinf = newf () in
  emit buf "mov.%s %s, %s;" s rinf c_inf ;
  let rscale = newf () in
  emit buf "mov.%s %s, %s;" s rscale c_scale ;
  let ay = newf () in
  emit buf "abs.%s %s, %s;" s ay ry ;
  let ax = newf () in
  emit buf "abs.%s %s, %s;" s ax rx ;
  let r = newf () in
  emit buf "mov.%s %s, %s;" s r rx ;
  let l_outer = new_label alloc in
  let l_scale = new_label alloc in
  let l_scale_loop = new_label alloc in
  let l_fix = new_label alloc in
  (* Domain guard: ay > 0 rejects y = 0 and y = NaN; ax < inf rejects
     x = +/-inf and x = NaN. *)
  let p_y = new_pred alloc in
  emit buf "setp.gt.%s %s, %s, %s;" s p_y ay rz ;
  let p_x = new_pred alloc in
  emit buf "setp.lt.%s %s, %s, %s;" s p_x ax rinf ;
  let p_ok = new_pred alloc in
  emit buf "and.pred %s, %s, %s;" p_ok p_y p_x ;
  emit buf "@%s bra %s;" p_ok l_outer ;
  emit buf "mov.%s %s, %s;" s r c_nan ;
  emit buf "bra %s;" l_fix ;
  emit_label buf l_outer ;
  let ar = newf () in
  emit buf "abs.%s %s, %s;" s ar r ;
  let p_done = new_pred alloc in
  emit buf "setp.lt.%s %s, %s, %s;" s p_done ar ay ;
  emit buf "@%s bra %s;" p_done l_fix ;
  let q = newf () in
  emit buf "div.rn.%s %s, %s, %s;" s q r ry ;
  let aq = newf () in
  emit buf "abs.%s %s, %s;" s aq q ;
  let p_qinf = new_pred alloc in
  emit buf "setp.eq.%s %s, %s, %s;" s p_qinf aq rinf ;
  emit buf "@%s bra %s;" p_qinf l_scale ;
  let t = newf () in
  emit buf "cvt.rzi.%s.%s %s, %s;" s s t q ;
  let nt = newf () in
  emit buf "neg.%s %s, %s;" s nt t ;
  emit buf "fma.rn.%s %s, %s, %s, %s;" s r nt ry r ;
  emit buf "bra %s;" l_outer ;
  emit_label buf l_scale ;
  let ys = newf () in
  emit buf "mov.%s %s, %s;" s ys ry ;
  emit_label buf l_scale_loop ;
  emit buf "mul.%s %s, %s, %s;" s ys ys rscale ;
  let q2 = newf () in
  emit buf "div.rn.%s %s, %s, %s;" s q2 r ys ;
  let aq2 = newf () in
  emit buf "abs.%s %s, %s;" s aq2 q2 ;
  let p_q2inf = new_pred alloc in
  emit buf "setp.eq.%s %s, %s, %s;" s p_q2inf aq2 rinf ;
  emit buf "@%s bra %s;" p_q2inf l_scale_loop ;
  let t2 = newf () in
  emit buf "cvt.rzi.%s.%s %s, %s;" s s t2 q2 ;
  let nt2 = newf () in
  emit buf "neg.%s %s, %s;" s nt2 t2 ;
  emit buf "fma.rn.%s %s, %s, %s, %s;" s r nt2 ys r ;
  emit buf "bra %s;" l_outer ;
  emit_label buf l_fix ;
  let p_rn = new_pred alloc in
  emit buf "setp.lt.%s %s, %s, %s;" s p_rn r rz ;
  let p_xp = new_pred alloc in
  emit buf "setp.ge.%s %s, %s, %s;" s p_xp rx rz ;
  let p_c1 = new_pred alloc in
  emit buf "and.pred %s, %s, %s;" p_c1 p_rn p_xp ;
  let p_rp = new_pred alloc in
  emit buf "setp.gt.%s %s, %s, %s;" s p_rp r rz ;
  let p_xn = new_pred alloc in
  emit buf "setp.lt.%s %s, %s, %s;" s p_xn rx rz ;
  let p_c2 = new_pred alloc in
  emit buf "and.pred %s, %s, %s;" p_c2 p_rp p_xn ;
  let c = newf () in
  emit buf "selp.%s %s, %s, %s, %s;" s c ay rz p_c1 ;
  let nay = newf () in
  emit buf "neg.%s %s, %s;" s nay ay ;
  emit buf "selp.%s %s, %s, %s, %s;" s c nay c p_c2 ;
  emit buf "add.%s %s, %s, %s;" s r r c ;
  (* copysign d, a, b = |b| with a's sign: post-fix sign(r) already matches
     sign(x) for r <> 0, so this only normalizes the sign of a zero result
     (fmod(-x, y) must return -0 when the remainder is exact). *)
  let res = newf () in
  emit buf "copysign.%s %s, %s, %s;" s res rx r ;
  res

(** {1 Expression emitter}

    Returns the PTX register name holding the result. Emits instructions into
    [buf] as a side effect. *)
let rec emit_expr buf alloc (env : env) (expr : expr) : string =
  match expr with
  | EConst (CInt32 n) ->
      let r = new_u32 alloc in
      emit buf "mov.u32 %s, %ld;" r n ;
      r
  | EConst (CInt64 n) ->
      let r = new_u64 alloc in
      emit buf "mov.u64 %s, %Ld;" r n ;
      r
  | EConst (CFloat32 f) ->
      let r = new_f32 alloc in
      emit buf "mov.f32 %s, 0F%08lX;" r (Int32.bits_of_float f) ;
      r
  | EConst (CFloat64 f) ->
      let r = new_f64 alloc in
      emit buf "mov.f64 %s, 0D%016LX;" r (Int64.bits_of_float f) ;
      r
  | EConst (CBool true) ->
      let r = new_u32 alloc in
      emit buf "mov.u32 %s, 1;" r ;
      r
  | EConst (CBool false) ->
      let r = new_u32 alloc in
      emit buf "mov.u32 %s, 0;" r ;
      r
  | EConst CUnit ->
      let r = new_u32 alloc in
      emit buf "mov.u32 %s, 0;" r ;
      r
  | EVar v -> env_lookup env v.var_name
  | EBinop (op, e1, e2) -> emit_binop buf alloc env op e1 e2
  | EUnop (Neg, e) ->
      let r_src = emit_expr buf alloc env e in
      let is_f64 r = String.length r >= 3 && r.[1] = 'f' && r.[2] = 'd' in
      let is_f32 r = String.length r >= 2 && r.[1] = 'f' && not (is_f64 r) in
      let is_u64 r = String.length r >= 3 && r.[1] = 'r' && r.[2] = 'd' in
      if is_f64 r_src then (
        let r = new_f64 alloc in
        emit buf "neg.f64 %s, %s;" r r_src ;
        r)
      else if is_f32 r_src then (
        let r = new_f32 alloc in
        emit buf "neg.f32 %s, %s;" r r_src ;
        r)
      else if is_u64 r_src then (
        let r = new_u64 alloc in
        emit buf "neg.s64 %s, %s;" r r_src ;
        r)
      else
        let r = new_u32 alloc in
        emit buf "neg.s32 %s, %s;" r r_src ;
        r
  | EUnop (Not, e) ->
      (* Bools are u32 0/1 post-typer, but be class-aware so a 64-bit
         operand cannot produce invalid setp.eq.u32-on-%rd PTX (H2). *)
      let r_src = emit_expr buf alloc env e in
      let is_u64 r = String.length r >= 3 && r.[1] = 'r' && r.[2] = 'd' in
      let p = new_pred alloc in
      if is_u64 r_src then emit buf "setp.eq.s64 %s, %s, 0;" p r_src
      else emit buf "setp.eq.u32 %s, %s, 0;" p r_src ;
      let r = new_u32 alloc in
      emit buf "selp.u32 %s, 1, 0, %s;" r p ;
      r
  | EUnop (BitNot, e) ->
      let r_src = emit_expr buf alloc env e in
      let is_u64 r = String.length r >= 3 && r.[1] = 'r' && r.[2] = 'd' in
      if is_u64 r_src then (
        let r = new_u64 alloc in
        emit buf "not.b64 %s, %s;" r r_src ;
        r)
      else
        let r = new_u32 alloc in
        emit buf "not.b32 %s, %s;" r r_src ;
        r
  | EArrayRead (arr_name, idx_expr) ->
      let r_base = env_lookup env arr_name in
      let r_idx = emit_expr buf alloc env idx_expr in
      emit_array_read
        buf
        alloc
        r_base
        r_idx
        (infer_elt_type alloc arr_name)
        ~space:(arr_space_of alloc arr_name)
  | EArrayReadExpr (base_expr, idx_expr) ->
      let r_base = emit_expr buf alloc env base_expr in
      let r_idx = emit_expr buf alloc env idx_expr in
      let arr_name_opt =
        match base_expr with EVar v -> Some v.var_name | _ -> None
      in
      let elt_type =
        match arr_name_opt with
        | Some n -> infer_elt_type alloc n
        | None ->
            fail
              "EArrayReadExpr: cannot infer element type from non-variable \
               base expression"
      in
      let space =
        match arr_name_opt with Some n -> arr_space_of alloc n | None -> None
      in
      emit_array_read buf alloc r_base r_idx elt_type ~space
  | EIntrinsic (path, name, args) -> emit_intrinsic buf alloc env path name args
  | ECast (ty, e) ->
      let r_src = emit_expr buf alloc env e in
      emit_cast buf alloc r_src ty
  | EIf (_, then_e, else_e) as e
    when expr_needs_branch_guard then_e || expr_needs_branch_guard else_e -> (
      (* A branch with an effect or a (possibly out-of-bounds) array read must
         not be evaluated eagerly (the selp path below computes both); emit
         real control flow instead — emit_value owns the branch-based path. *)
      match emit_value buf alloc env e with
      | Scalar r -> r
      | Agg _ ->
          fail
            "PTX codegen: if-expression of record/variant type used in a \
             scalar context; bind it with let and read its scalar fields")
  | EIf (cond, then_e, else_e) ->
      let r_cond = emit_expr buf alloc env cond in
      let r_then = emit_expr buf alloc env then_e in
      let r_else = emit_expr buf alloc env else_e in
      let p = new_pred alloc in
      emit buf "setp.ne.u32 %s, %s, 0;" p r_cond ;
      let is_f64 r = String.length r >= 3 && r.[1] = 'f' && r.[2] = 'd' in
      let is_f32 r = String.length r >= 2 && r.[1] = 'f' && not (is_f64 r) in
      let is_u64 r = String.length r >= 3 && r.[1] = 'r' && r.[2] = 'd' in
      if is_f64 r_then then (
        let r = new_f64 alloc in
        emit buf "selp.f64 %s, %s, %s, %s;" r r_then r_else p ;
        r)
      else if is_f32 r_then then (
        let r = new_f32 alloc in
        emit buf "selp.f32 %s, %s, %s, %s;" r r_then r_else p ;
        r)
      else if is_u64 r_then then (
        let r = new_u64 alloc in
        emit buf "selp.u64 %s, %s, %s, %s;" r r_then r_else p ;
        r)
      else
        let r = new_u32 alloc in
        emit buf "selp.u32 %s, %s, %s, %s;" r r_then r_else p ;
        r
  | EArrayLen arr ->
      (* Bound by emit_params alongside the array pointer param. Only
         parameter arrays carry a length; local/shared arrays fall back to
         another backend. *)
      if Hashtbl.mem env (length_param_name arr) then
        env_lookup env (length_param_name arr)
      else
        unsupported
          (Printf.sprintf
             "EArrayLen on '%s' (only parameter arrays have a length)"
             arr)
  | EArrayCreate _ ->
      unsupported "EArrayCreate in expression position (use SLet)"
  | EMatch _ as e -> (
      match emit_value buf alloc env e with
      | Scalar r -> r
      | Agg _ ->
          fail
            "PTX codegen: match result is a record/variant used in a scalar \
             context; bind it with let and read its scalar fields")
  | ERecord (name, _) ->
      fail
        ("PTX codegen: record value of type '" ^ name
       ^ "' cannot be used in a scalar context; bind it with let and read its \
          scalar fields")
  | ERecordField (_, field) as e -> (
      (* A field projection is usually scalar; delegate to emit_value and
         reject only when the projected field is itself an aggregate. *)
      match emit_value buf alloc env e with
      | Scalar r -> r
      | Agg _ ->
          fail
            ("PTX codegen: field '" ^ field
           ^ "' is a nested record/variant and cannot be used in a scalar \
              context; bind it with let and read its scalar fields"))
  | ETuple _ ->
      fail
        "PTX codegen: tuple value used in a scalar context (e.g. stored into a \
         vector element); bind it with let and use its components \
         individually, or use a registered record type instead"
  | EApp (EVar f, args) -> (
      match emit_app buf alloc env f args with
      | Scalar r -> r
      | Agg _ ->
          fail
            ("PTX codegen: helper '" ^ f.var_name
           ^ "' returns a record/variant and cannot be used in a scalar \
              context; bind the result with let and use its fields"))
  | EApp _ -> unsupported "EApp with non-variable callee"
  | EVariant (tyname, _, _) ->
      fail
        ("PTX codegen: variant value of type '" ^ tyname
       ^ "' cannot be used in a scalar context; bind it with let and match on \
          it")

(** {1 Helper-function inlining}

    Inline the helper body at the call site. PTX .func would need a per-function
    register frame and .param ABI the single-pass emitter does not model;
    helpers are small and NVCC inlines them anyway. Recursive helpers are
    rejected and fall back to another backend, as before. *)

(** Bind one helper parameter in [callee_env] from the caller's evaluated
    argument. Scalar and aggregate values are leaf-wise copied so mutations
    inside the helper can never clobber the caller's registers; array params are
    base pointers, never written through LVar — bound directly. Returns array
    metadata to restore after the inline ([None] for non-arrays). *)
and bind_helper_param buf alloc env callee_env (p : var) (arg, arg_val) =
  (match (p.var_type, arg_val) with
  | (TVec _ | TArray _), Scalar r_arg -> env_bind callee_env p.var_name r_arg
  | (TVec _ | TArray _), Agg _ ->
      fail
        (Printf.sprintf
           "PTX codegen: array parameter '%s' bound to a record/variant value"
           p.var_name)
  | _, v -> env_bind_binding callee_env p.var_name (copy_binding buf alloc v)) ;
  (* Array params: register element type (from the param's own type) and
     propagate shared-ness / length binding from the caller when the argument
     is a plain array variable. Overwritten entries are saved for restore. *)
  match p.var_type with
  | TVec elt | TArray (elt, _) ->
      let prev_elt = Hashtbl.find_opt alloc.arr_elt_types p.var_name in
      let prev_ms = arr_space_of alloc p.var_name in
      Hashtbl.replace alloc.arr_elt_types p.var_name elt ;
      (match arg with
      | EVar a -> (
          (match arr_space_of alloc a.var_name with
          | Some space -> Hashtbl.replace alloc.arr_memspaces p.var_name space
          | None -> Hashtbl.remove alloc.arr_memspaces p.var_name) ;
          match Hashtbl.find_opt env (length_param_name a.var_name) with
          | Some len_binding ->
              env_bind_binding
                callee_env
                (length_param_name p.var_name)
                len_binding
          | None -> ())
      | _ -> Hashtbl.remove alloc.arr_memspaces p.var_name) ;
      Some (p.var_name, prev_elt, prev_ms)
  | _ -> None

(** Restore array metadata shadowed by helper parameter names. *)
and restore_helper_array_meta alloc saved =
  List.iter
    (function
      | None -> ()
      | Some (name, prev_elt, prev_ms) -> (
          (match prev_elt with
          | Some e -> Hashtbl.replace alloc.arr_elt_types name e
          | None -> Hashtbl.remove alloc.arr_elt_types name) ;
          match prev_ms with
          | Some space -> Hashtbl.replace alloc.arr_memspaces name space
          | None -> Hashtbl.remove alloc.arr_memspaces name))
    saved

(** Inline a helper call and return the binding holding its result. First entry
    seeds the helper's recursive-inline budget (from a [sarek.inline N] pragma
    at its body root, if any); recursive re-entry dispatches to
    [emit_app_recursive]. *)
and emit_app buf alloc (env : env) (f : var) (args : expr list) : binding =
  match Hashtbl.find_opt alloc.funcs f.var_name with
  | None -> unsupported ("EApp to unknown function '" ^ f.var_name ^ "'")
  | Some hf ->
      if List.length args <> List.length hf.hf_params then
        fail
          (Printf.sprintf
             "PTX codegen: helper '%s' called with %d args, expects %d"
             hf.hf_name
             (List.length args)
             (List.length hf.hf_params))
      else if List.mem hf.hf_name alloc.inline_stack then
        emit_app_recursive buf alloc env hf args
      else begin
        (match helper_inline_budget hf with
        | Some n -> Hashtbl.replace alloc.inline_budget hf.hf_name n
        | None -> ()) ;
        let res = emit_app_inline buf alloc env hf args in
        Hashtbl.remove alloc.inline_budget hf.hf_name ;
        res
      end

(** Recursive re-entry into a helper already on the inline stack. Allowed only
    for helpers carrying [pragma ["sarek.inline N"]]: each re-entry consumes one
    unit of the remaining budget (restored on exit, so sibling calls — e.g.
    fib's two — each see the same depth). At exhaustion the call's result is a
    typed zero: the pragma is the author's contract that N levels cover all
    runtime inputs, so the residual call site is dynamically unreachable and
    only needs to be well-formed PTX, never correct. *)
and emit_app_recursive buf alloc env hf args : binding =
  match Hashtbl.find_opt alloc.inline_budget hf.hf_name with
  | None ->
      unsupported
        ("EApp: recursive helper '" ^ hf.hf_name
       ^ "' (inlining supports non-recursive helpers only; annotate the helper \
          body with pragma [\"sarek.inline N\"] to enable depth-bounded \
          inlining)")
  | Some 0 -> (
      (* Evaluate the arguments even though the residual call is elided —
         keeps emitted PTX consistent with every budgeted level (argument
         side effects are never silently dropped). *)
      List.iter (fun a -> ignore (emit_value buf alloc env a)) args ;
      match hf.hf_ret_type with
      | TUnit ->
          let r = new_u32 alloc in
          emit buf "mov.u32 %s, 0;" r ;
          Scalar r
      | t ->
          let b = binding_of_elttype alloc t in
          zero_binding buf b ;
          b)
  | Some n ->
      Hashtbl.replace alloc.inline_budget hf.hf_name (n - 1) ;
      let res = emit_app_inline buf alloc env hf args in
      Hashtbl.replace alloc.inline_budget hf.hf_name n ;
      res

(** Emit the inline expansion of a call to [hf] — shared by first entry and
    budgeted recursive re-entry. *)
and emit_app_inline buf alloc env hf args : binding =
  (* Evaluate arguments in the caller's environment. *)
  let arg_vals = List.map (emit_value buf alloc env) args in
  (* Fresh environment for the helper body: only its parameters are in
     scope (helpers are module-level, no capture). *)
  let callee_env = make_env () in
  let saved =
    List.map2
      (bind_helper_param buf alloc env callee_env)
      hf.hf_params
      (List.combine args arg_vals)
  in
  let l_end = new_label alloc in
  let ret =
    match hf.hf_ret_type with
    | TUnit -> None
    | t -> Some (binding_of_elttype alloc t)
  in
  alloc.inline_stack <- hf.hf_name :: alloc.inline_stack ;
  alloc.inline_ret <- (ret, l_end) :: alloc.inline_ret ;
  !stmt_emitter buf alloc callee_env hf.hf_body ;
  alloc.inline_ret <- List.tl alloc.inline_ret ;
  alloc.inline_stack <- List.tl alloc.inline_stack ;
  emit_label buf l_end ;
  restore_helper_array_meta alloc saved ;
  match ret with
  | Some b -> b
  | None ->
      let r = new_u32 alloc in
      emit buf "mov.u32 %s, 0;" r ;
      Scalar r

(** {1 Value emitter (scalar or aggregate)}

    [emit_value] is the aggregate-aware entry point: scalar expressions delegate
    to {!emit_expr} (wrapped in [Scalar]); record construction and field
    projection build/select SROA register sets ([Agg]) without touching memory
    (FR-020). *)
and emit_value buf alloc (env : env) (e : expr) : binding =
  match e with
  | EArrayRead (arr_name, idx_expr) when elt_is_aggregate alloc arr_name ->
      (* Whole-aggregate element read: SROA register set materialized by one
         typed ld per leaf (FR-012). *)
      emit_agg_array_read buf alloc env arr_name idx_expr
  | EArrayReadExpr (EVar v, idx_expr) when elt_is_aggregate alloc v.var_name ->
      emit_agg_array_read buf alloc env v.var_name idx_expr
  | ERecord (_name, fields) ->
      (* Field order = declaration order as carried by the ERecord node. *)
      Agg
        (ARecord
           (List.map (fun (n, fe) -> (n, emit_value buf alloc env fe)) fields))
  | ERecordField (base, field) -> emit_record_field buf alloc env base field
  | EVar v -> env_lookup_binding env v.var_name
  | EApp (EVar f, args) -> emit_app buf alloc env f args
  | EVariant (tyname, ctor, args) -> emit_variant buf alloc env tyname ctor args
  | ETuple es ->
      (* Tuple = anonymous record with positional fields "_0", "_1", ...
         (FR-024); destructured by PConstr("tuple", vars) match patterns. *)
      Agg
        (ARecord
           (List.mapi
              (fun i e -> (Printf.sprintf "_%d" i, emit_value buf alloc env e))
              es))
  | EMatch (scrut_e, arms) -> (
      (* Branch-based match in value position (never selp — FR-022). The
         result binding is allocated by leaf-wise copying the first emitted
         arm's value; every other arm movs leaf-wise into it (post-typing arm
         result types are uniform). *)
      let scrut = emit_value buf alloc env scrut_e in
      let result = ref None in
      emit_match_arms buf alloc env scrut arms ~emit_arm:(fun arm_e ->
          let b = emit_value buf alloc env arm_e in
          match !result with
          | None -> result := Some (copy_binding buf alloc b)
          | Some dst -> mov_binding buf ~src:b ~dst) ;
      match !result with
      | Some b -> b
      | None -> fail "PTX codegen: match with no arms")
  | EIf (cond, then_e, else_e)
    when expr_needs_branch_guard then_e || expr_needs_branch_guard else_e ->
      (* Branch-based conditional, aggregate-capable: the then-value's binding
         is leaf-wise copied into the result, the else-value leaf-wise moved
         into it (for scalars this emits exactly the instructions of
         emit_expr's guarded EIf path). *)
      let r_cond = emit_expr buf alloc env cond in
      let p = new_pred alloc in
      emit buf "setp.ne.u32 %s, %s, 0;" p r_cond ;
      let l_else = new_label alloc in
      let l_merge = new_label alloc in
      emit buf "@!%s bra %s;" p l_else ;
      let b_then = emit_value buf alloc env then_e in
      let b_res = copy_binding buf alloc b_then in
      emit buf "bra %s;" l_merge ;
      emit_label buf l_else ;
      let b_else = emit_value buf alloc env else_e in
      mov_binding buf ~src:b_else ~dst:b_res ;
      emit_label buf l_merge ;
      b_res
  | _ -> Scalar (emit_expr buf alloc env e)

(** Element base address + whole-element SROA load of an aggregate-element array
    (used by [emit_value] for [v.(i)] reads of record/variant vectors). Stride
    and offsets come from Sarek_ir_layout (FR-001, FR-010). *)
and emit_agg_array_read buf alloc env arr_name idx_expr : binding =
  if is_soa alloc arr_name then
    (* SoA: one coalesced scalar load per leaf from its own base. *)
    let r_idx = emit_expr buf alloc env idx_expr in
    emit_soa_elem_load buf alloc r_idx arr_name
  else
    let elt = infer_elt_type alloc arr_name in
    let r_base = env_lookup env arr_name in
    let r_idx = emit_expr buf alloc env idx_expr in
    let r_addr =
      emit_agg_elem_addr
        buf
        alloc
        r_base
        r_idx
        ~stride:(elt_stride elt)
        ~space:(arr_space_of alloc arr_name)
        ~arr_name
    in
    emit_agg_elem_load buf alloc r_addr ~offset:0 elt

(** When [base.field] roots at an element of an aggregate-element array
    ([v.(i).f], possibly through nested projections [v.(i).f.g]), return the
    array name, index expression, and outermost-first field path. *)
and split_elem_field_read alloc base field :
    (string * expr * string list) option =
  let rec root e path =
    match e with
    | ERecordField (b, f) -> root b (f :: path)
    | EArrayRead (n, idx) -> Some (n, idx, path)
    | EArrayReadExpr (EVar v, idx) -> Some (v.var_name, idx, path)
    | _ -> None
  in
  match root base [field] with
  | Some (n, idx, path) when elt_is_aggregate alloc n -> Some (n, idx, path)
  | _ -> None

(** Field selection on a record value. On a local (SROA) record this is pure
    register selection (no instructions). On an element of an aggregate-element
    vector ([v.(i).field]) it is a single typed ld at
    [base + idx*stride + field_offset] (FR-011) — intercepted BEFORE base
    evaluation so the untouched fields are never loaded. *)
and emit_record_field buf alloc env base field : binding =
  match split_elem_field_read alloc base field with
  | Some (arr_name, idx_expr, path) when is_soa alloc arr_name ->
      (* SoA: one coalesced scalar load at the addressed leaf's own base. *)
      let r_idx = emit_expr buf alloc env idx_expr in
      emit_soa_field_load buf alloc r_idx arr_name path
  | Some (arr_name, idx_expr, path) ->
      let elt = infer_elt_type alloc arr_name in
      let offset, fty = agg_field_path elt path in
      let r_base = env_lookup env arr_name in
      let r_idx = emit_expr buf alloc env idx_expr in
      let r_addr =
        emit_agg_elem_addr
          buf
          alloc
          r_base
          r_idx
          ~stride:(elt_stride elt)
          ~space:(arr_space_of alloc arr_name)
          ~arr_name
      in
      emit_agg_elem_load buf alloc r_addr ~offset fty
  | None -> emit_local_record_field buf alloc env base field

(** Field selection on a local (SROA) record value: pure register selection, no
    instructions emitted for the projection itself. *)
and emit_local_record_field buf alloc env base field : binding =
  match emit_value buf alloc env base with
  | Agg (ARecord fields) -> (
      match List.assoc_opt field fields with
      | Some b -> b
      | None ->
          fail
            (Printf.sprintf
               "PTX codegen: record has no field '%s' (available: %s)"
               field
               (String.concat ", " (List.map fst fields))))
  | Agg (AVariant _) ->
      fail
        ("PTX codegen: field access '." ^ field
       ^ "' on a variant value; use match to inspect a variant")
  | Scalar _ ->
      fail
        ("PTX codegen: field access '." ^ field
       ^ "' on a non-record value; only record values (local or vector \
          elements of a record type) have fields")

(** Variant construction (FR-021): the tag register is a mov of the
    constructor's DECLARATION-INDEX constant (aligned with
    [Sarek_ir_layout.ctor_tag]); the constructed ctor's payload slots hold the
    evaluated arguments; every other constructor's slots are freshly-allocated,
    never-written registers (see [agg_value] in Sarek_ir_ptx_types). *)
and emit_variant buf alloc env tyname ctor args : binding =
  let decl =
    match Hashtbl.find_opt alloc.variant_decls tyname with
    | Some d -> d
    | None ->
        fail
          ("PTX codegen: variant type '" ^ tyname
         ^ "' has no registered declaration; declare it with [@@sarek.type] \
            (or a kernel-local Types module) so the kernel carries its \
            constructor list")
  in
  let tag =
    let rec index i = function
      | [] ->
          fail
            (Printf.sprintf
               "PTX codegen: variant type '%s' has no constructor '%s'"
               tyname
               ctor)
      | (cn, _) :: rest -> if cn = ctor then i else index (i + 1) rest
    in
    index 0 decl
  in
  let tag_reg = new_u32 alloc in
  emit buf "mov.u32 %s, %d;" tag_reg tag ;
  let ctors =
    List.map
      (fun (cn, tys) ->
        if cn = ctor then begin
          if List.length args <> List.length tys then
            fail
              (Printf.sprintf
                 "PTX codegen: constructor '%s' of '%s' expects %d \
                  argument(s), got %d"
                 ctor
                 tyname
                 (List.length tys)
                 (List.length args)) ;
          (cn, List.map (emit_value buf alloc env) args)
        end
        else (cn, List.map (binding_of_elttype alloc) tys))
      decl
  in
  Agg (AVariant {vname = tyname; tag_reg; ctors})

and emit_binop buf alloc env op e1 e2 : string =
  let r1 = emit_expr buf alloc env e1 in
  let r2 = emit_expr buf alloc env e2 in
  (* Infer type from first operand register name prefix.
     %r* -> u32, %rd* -> u64, %f* -> f32, %fd* -> f64 *)
  let is_f64 r = String.length r >= 3 && r.[1] = 'f' && r.[2] = 'd' in
  let is_f32 r = String.length r >= 2 && r.[1] = 'f' && not (is_f64 r) in
  let is_u64 r = String.length r >= 3 && r.[1] = 'r' && r.[2] = 'd' in
  match op with
  | Add ->
      if is_f64 r1 then (
        let r = new_f64 alloc in
        emit buf "add.f64 %s, %s, %s;" r r1 r2 ;
        r)
      else if is_f32 r1 then (
        let r = new_f32 alloc in
        emit buf "add.f32 %s, %s, %s;" r r1 r2 ;
        r)
      else if is_u64 r1 then (
        let r = new_u64 alloc in
        emit buf "add.u64 %s, %s, %s;" r r1 r2 ;
        r)
      else
        let r = new_u32 alloc in
        emit buf "add.u32 %s, %s, %s;" r r1 r2 ;
        r
  | Sub ->
      if is_f64 r1 then (
        let r = new_f64 alloc in
        emit buf "sub.f64 %s, %s, %s;" r r1 r2 ;
        r)
      else if is_f32 r1 then (
        let r = new_f32 alloc in
        emit buf "sub.f32 %s, %s, %s;" r r1 r2 ;
        r)
      else if is_u64 r1 then (
        let r = new_u64 alloc in
        emit buf "sub.u64 %s, %s, %s;" r r1 r2 ;
        r)
      else
        let r = new_u32 alloc in
        emit buf "sub.u32 %s, %s, %s;" r r1 r2 ;
        r
  | Mul ->
      if is_f64 r1 then (
        let r = new_f64 alloc in
        emit buf "mul.f64 %s, %s, %s;" r r1 r2 ;
        r)
      else if is_f32 r1 then (
        let r = new_f32 alloc in
        emit buf "mul.f32 %s, %s, %s;" r r1 r2 ;
        r)
      else if is_u64 r1 then (
        let r = new_u64 alloc in
        emit buf "mul.lo.u64 %s, %s, %s;" r r1 r2 ;
        r)
      else
        let r = new_u32 alloc in
        emit buf "mul.lo.u32 %s, %s, %s;" r r1 r2 ;
        r
  | Div ->
      if is_f64 r1 then (
        let r = new_f64 alloc in
        emit buf "div.rn.f64 %s, %s, %s;" r r1 r2 ;
        r)
      else if is_f32 r1 then (
        (* Correctly-rounded division (audit finding M2): div.approx.f32 is
           ~2 ulp and badly wrong at exponent extremes, which made PTX the
           least-accurate backend for ordinary /. and eroded Sarek_df64's
           error budget. Fast approximate division stays available to
           intrinsics that are already approximate (tan, tanh). *)
        let r = new_f32 alloc in
        emit buf "div.rn.f32 %s, %s, %s;" r r1 r2 ;
        r)
      else if is_u64 r1 then (
        (* Sarek int64 is signed: div.u64 on negative operands is silently
           wrong ((-7)/2 = huge). add/sub/mul are sign-agnostic in two's
           complement; div/rem are not (audit finding H1). *)
        let r = new_u64 alloc in
        emit buf "div.s64 %s, %s, %s;" r r1 r2 ;
        r)
      else
        let r = new_u32 alloc in
        emit buf "div.s32 %s, %s, %s;" r r1 r2 ;
        r
  | Mod ->
      (* Float Mod is C fmod (exact for all finite inputs, result sign
         follows the dividend). Lowered by emit_float_fmod's iterative
         reduction - see its doc comment (audit finding M1). *)
      if is_f64 r1 then emit_float_fmod buf alloc ~is64:true r1 r2
      else if is_f32 r1 then emit_float_fmod buf alloc ~is64:false r1 r2
      else if is_u64 r1 then (
        (* Signed rem, matching C's % (result sign follows the dividend),
           the interpreter's Int64.rem, and every C-family backend. *)
        let r = new_u64 alloc in
        emit buf "rem.s64 %s, %s, %s;" r r1 r2 ;
        r)
      else
        let r = new_u32 alloc in
        emit buf "rem.s32 %s, %s, %s;" r r1 r2 ;
        r
  | Eq | Ne | Lt | Le | Gt | Ge ->
      (* Comparison family, class-aware (audit finding H2): the old code
         fell through to setp.*.u32/s32 even on %rd (64-bit) operands,
         which is invalid PTX and fails at module load. Sarek ints are
         signed, so the integer forms are s32/s64 (sign matters for
         Lt/Le/Gt/Ge; for Eq/Ne it is irrelevant but harmless). *)
      let cmp =
        match op with
        | Eq -> "eq"
        | Ne -> "ne"
        | Lt -> "lt"
        | Le -> "le"
        | Gt -> "gt"
        | Ge -> "ge"
        | _ -> assert false
      in
      let ty =
        if is_f64 r1 then "f64"
        else if is_f32 r1 then "f32"
        else if is_u64 r1 then "s64"
        else if cmp = "eq" || cmp = "ne" then "u32"
        else "s32"
      in
      let p = new_pred alloc in
      emit buf "setp.%s.%s %s, %s, %s;" cmp ty p r1 r2 ;
      let r = new_u32 alloc in
      emit buf "selp.u32 %s, 1, 0, %s;" r p ;
      r
  | And ->
      let r = new_u32 alloc in
      emit buf "and.b32 %s, %s, %s;" r r1 r2 ;
      r
  | Or ->
      let r = new_u32 alloc in
      emit buf "or.b32 %s, %s, %s;" r r1 r2 ;
      r
  | Shl ->
      if is_u64 r1 then (
        (* PTX shift amounts are u32; narrow a 64-bit amount if needed. *)
        let amt = if is_u64 r2 then emit_cast buf alloc r2 TInt32 else r2 in
        let r = new_u64 alloc in
        emit buf "shl.b64 %s, %s, %s;" r r1 amt ;
        r)
      else
        let r = new_u32 alloc in
        emit buf "shl.b32 %s, %s, %s;" r r1 r2 ;
        r
  | Shr ->
      (* Arithmetic (sign-extending) shift: Ir.Shr is arithmetic on every
         backend (CUDA/OpenCL/Metal/GLSL/WGSL emit plain [>>] on a signed
         int type; the interpreter uses Int32.shift_right). [lsr] is lowered
         to a separate expression tree in Sarek_lower_ir.ml precisely
         because this node is arithmetic - see G phase 1 in
         briefs/fix-critical-semantics-evidence.md. Formal spec note:
         formal/codegen-ptx/theories/PtxTypes.v models Shr as a logical
         Nat.shiftr on U32; that model was written against the old (wrong)
         shr.u32 emission and is now out of sync with this fix. formal/ is
         out of scope for this task - flagged for the formal-verification
         owner. *)
      if is_u64 r1 then (
        (* 64-bit arithmetic shift (softmath exponent extraction); PTX shift
           amounts are u32, so a 64-bit amount is narrowed first. *)
        let amt = if is_u64 r2 then emit_cast buf alloc r2 TInt32 else r2 in
        let r = new_u64 alloc in
        emit buf "shr.s64 %s, %s, %s;" r r1 amt ;
        r)
      else
        let r = new_u32 alloc in
        emit buf "shr.s32 %s, %s, %s;" r r1 r2 ;
        r
  | BitAnd -> emit_bitwise buf alloc "and" r1 r2
  | BitOr -> emit_bitwise buf alloc "or" r1 r2
  | BitXor -> emit_bitwise buf alloc "xor" r1 r2

(** Bitwise and/or/xor at the width of the first operand (b64 when it is a
    64-bit register, b32 otherwise); a narrower second operand is widened. *)
and emit_bitwise buf alloc op r1 r2 : string =
  let is_u64 r = String.length r >= 3 && r.[1] = 'r' && r.[2] = 'd' in
  if is_u64 r1 then (
    let r2w = if is_u64 r2 then r2 else emit_cast buf alloc r2 TInt64 in
    let r = new_u64 alloc in
    emit buf "%s.b64 %s, %s, %s;" op r r1 r2w ;
    r)
  else
    let r = new_u32 alloc in
    emit buf "%s.b32 %s, %s, %s;" op r r1 r2 ;
    r

and emit_cast buf alloc r_src dst_ty : string =
  let is_f64 r = String.length r >= 3 && r.[1] = 'f' && r.[2] = 'd' in
  let is_f32 r = String.length r >= 2 && r.[1] = 'f' && not (is_f64 r) in
  let is_u64 r = String.length r >= 3 && r.[1] = 'r' && r.[2] = 'd' in
  match dst_ty with
  | TFloat32 ->
      if is_f32 r_src then r_src
      else
        let r = new_f32 alloc in
        let cvt =
          if is_f64 r_src then "cvt.rn.f32.f64"
          else if is_u64 r_src then "cvt.rn.f32.s64"
          else "cvt.rn.f32.s32"
        in
        emit buf "%s %s, %s;" cvt r r_src ;
        r
  | TFloat64 ->
      if is_f64 r_src then r_src
      else
        let r = new_f64 alloc in
        let cvt =
          if is_f32 r_src then "cvt.rn.f64.f32"
          else if is_u64 r_src then "cvt.rn.f64.s64"
          else "cvt.rn.f64.s32"
        in
        emit buf "%s %s, %s;" cvt r r_src ;
        r
  | TInt32 ->
      if (not (is_f32 r_src)) && (not (is_f64 r_src)) && not (is_u64 r_src) then
        r_src
      else
        let r = new_u32 alloc in
        let cvt =
          if is_f64 r_src then "cvt.rzi.s32.f64"
          else if is_f32 r_src then "cvt.rzi.s32.f32"
          else "cvt.u32.u64"
        in
        emit buf "%s %s, %s;" cvt r r_src ;
        r
  | TInt64 ->
      (* i32 -> i64 widens with sign extension (cvt.s64.s32); the reverse
         narrowing in the TInt32 arm (cvt.u32.u64) TRUNCATES to the low 32
         bits — values outside int32 range wrap, matching Int64.to_int32. *)
      if is_u64 r_src then r_src
      else
        let r = new_u64 alloc in
        let cvt =
          if is_f64 r_src then "cvt.rzi.s64.f64"
          else if is_f32 r_src then "cvt.rzi.s64.f32"
          else "cvt.s64.s32"
        in
        emit buf "%s %s, %s;" cvt r r_src ;
        r
  | TBool ->
      (* Bool lives in a u32 register as 0/1 (the setp/selp discipline). A
         cast to bool is C truth-testing: nonzero -> 1, zero -> 0. Sources
         that are already 0/1 pay one setp+selp normalization — the register
         class alone cannot prove a u32 holds only 0/1. Float sources use
         UNordered ne (setp.neu): NaN != 0.0 is true in C, so
         (bool)NaN = 1. *)
      let p = new_pred alloc in
      if is_f64 r_src then
        emit buf "setp.neu.f64 %s, %s, 0D0000000000000000;" p r_src
      else if is_f32 r_src then
        emit buf "setp.neu.f32 %s, %s, 0F00000000;" p r_src
      else if is_u64 r_src then emit buf "setp.ne.s64 %s, %s, 0;" p r_src
      else emit buf "setp.ne.u32 %s, %s, 0;" p r_src ;
      let r = new_u32 alloc in
      emit buf "selp.u32 %s, 1, 0, %s;" r p ;
      r
  | TUnit ->
      (* No meaningful value to produce; a cast to unit is a lowering bug —
         drop the expression (SExpr) instead of casting it. *)
      unsupported "ECast to unit: drop the expression instead of casting it"
  | TRecord _ | TVariant _ | TArray _ | TVec _ ->
      unsupported "ECast to an aggregate type: casts are scalar-only"

and emit_intrinsic buf alloc env path name args : string =
  match Sarek_ir_ptx_softmath.helper_name name with
  | Some hname when List.mem "Float64" path -> (
      (* Float64 transcendental: PTX has no f64 instruction for it, so route
         through the software implementation (polynomial helper_func bodies in
         Sarek_ir_ptx_softmath) via the existing EApp inline machinery. The
         "__sarek_f64_*" helper names are reserved. *)
      Sarek_ir_ptx_softmath.register alloc.funcs ;
      let f =
        {
          var_name = hname;
          var_id = -1;
          var_type = TFloat64;
          var_mutable = false;
        }
      in
      match emit_app buf alloc env f args with
      | Scalar r -> r
      | Agg _ ->
          fail
            "PTX codegen: internal error: softmath helper returned an aggregate"
      )
  | _ -> emit_intrinsic_native buf alloc env path name args

and emit_intrinsic_native buf alloc env path name args : string =
  (* Type conversions delegate to emit_cast; a unary helper for them. *)
  let unary_cast intr dst_ty =
    match args with
    | [a] -> emit_cast buf alloc (emit_expr buf alloc env a) dst_ty
    | _ -> unsupported (intr ^ " arity != 1")
  in
  (* An i64 register is "%rd<n>" (u64 class; distinct from f64 "%fd<n>"). *)
  let is_u64_reg r = String.length r >= 3 && r.[1] = 'r' && r.[2] = 'd' in
  (* The operand register class must agree with the atom.<op><ty> suffix; a
     mismatched register would be invalid PTX (module-load failure). *)
  let check_atom_operand intr result r =
    let ok, want =
      match result with
      | `F32 -> (is_f32_reg r, "f32")
      | `F64 -> (is_f64_reg r, "f64")
      | `U64 -> (is_u64_reg r, "int64")
      | `U32 ->
          ( (not (is_f32_reg r)) && (not (is_f64_reg r)) && not (is_u64_reg r),
            "int32" )
    in
    if not ok then
      unsupported
        (intr ^ ": operand " ^ r ^ " is not " ^ want ^ "; cast the value first")
  in
  let new_atom_result = function
    | `F32 -> new_f32 alloc
    | `F64 -> new_f64 alloc
    | `U64 -> new_u64 alloc
    | `U32 -> new_u32 alloc
  in
  (* Byte address of element [r_idx] of array [arr]: shared arrays use
     32-bit byte addressing, global arrays 64-bit. [elt_shift] is log2 of
     the element size (2 for 32-bit, 3 for 64-bit elements) — the stride
     must match the atom width or neighbouring elements alias. Returns the
     PTX space name and the address register. *)
  let atomic_addr ~intr ~global_only ~elt_shift arr r_idx =
    let r_base = env_lookup env arr.var_name in
    (* PTX has no atom.local — and per-thread memory needs no atomics. *)
    (match arr_space_of alloc arr.var_name with
    | Some SpaceLocal ->
        unsupported
          ("atomic operation on per-thread local array '" ^ arr.var_name
         ^ "' (atomics require shared or global memory)")
    | Some SpaceShared when global_only ->
        (* A *_global_* atomic on a shared array would use the 32-bit
           shared-window offset as a 64-bit global address — silent
           corruption (audit finding M5). *)
        unsupported
          (intr ^ ": '" ^ arr.var_name
         ^ "' is a shared array; use the non-global atomic form")
    | Some SpaceShared | None -> ()) ;
    (* The intrinsic's hardwired stride must match the array's element
       width: atomic_add_int32 on an int64/f64 vector would index with a
       4-byte stride into 8-byte elements — silent corruption of
       neighbouring elements (audit finding M5). *)
    (let elt = infer_elt_type alloc arr.var_name in
     let width_shift =
       match elt with
       | TInt32 | TFloat32 | TBool -> Some 2
       | TInt64 | TFloat64 -> Some 3
       | _ -> None
     in
     match width_shift with
     | Some w when w <> elt_shift ->
         unsupported
           (Printf.sprintf
              "%s: array '%s' has %d-byte elements but this atomic addresses \
               %d-byte elements; use the matching-width atomic"
              intr
              arr.var_name
              (1 lsl w)
              (1 lsl elt_shift))
     | Some _ -> ()
     | None ->
         unsupported
           (intr ^ ": atomics require a scalar int/float element type, but '"
          ^ arr.var_name ^ "' has an aggregate element type")) ;
    let is_shared =
      (not global_only) && arr_space_of alloc arr.var_name = Some SpaceShared
    in
    if is_shared then begin
      let r_off = new_u32 alloc in
      let r_addr = new_u32 alloc in
      emit buf "shl.b32 %s, %s, %d;" r_off r_idx elt_shift ;
      emit buf "add.u32 %s, %s, %s;" r_addr r_base r_off ;
      ("shared", r_addr)
    end
    else begin
      let r_idx64 = new_u64 alloc in
      let r_off = new_u64 alloc in
      let r_addr = new_u64 alloc in
      emit buf "cvt.u64.u32 %s, %s;" r_idx64 r_idx ;
      emit buf "shl.b64 %s, %s, %d;" r_off r_idx64 elt_shift ;
      emit buf "add.u64 %s, %s, %s;" r_addr r_base r_off ;
      ("global", r_addr)
    end
  in
  (* Atomic read-modify-write on one element of an array denoted by a plain
     variable; returns the old value. [op]/[ty] form the PTX suffix (e.g.
     add.s32, min.s32, and.b32, exch.b32, add.f32, add.u64, add.f64 — u64
     add is two's-complement, so it is also the int64 add). PTX has no
     atom.sub, so int32 "sub" is lowered to an add of the negated operand.
     [result] selects the old-value register class and the operand width
     check; [elt_shift] the addressing stride. *)
  let atomic_rmw intr ~global_only ~elt_shift ~op ~ty ~result =
    match args with
    | [EVar arr; idx_e; val_e] ->
        let r_idx = emit_expr buf alloc env idx_e in
        let r_val0 = emit_expr buf alloc env val_e in
        check_atom_operand intr result r_val0 ;
        let op, r_val =
          if op = "sub" then begin
            let r = new_u32 alloc in
            emit buf "neg.s32 %s, %s;" r r_val0 ;
            ("add", r)
          end
          else (op, r_val0)
        in
        let space, r_addr =
          atomic_addr ~intr ~global_only ~elt_shift arr r_idx
        in
        let r_old = new_atom_result result in
        emit buf "atom.%s.%s%s %s, [%s], %s;" space op ty r_old r_addr r_val ;
        r_old
    | _ -> unsupported (intr ^ ": expects (array-variable, index, value)")
  in
  (* Atomic compare-and-swap: atom.{shared,global}.cas.b{32,64}
     d, [addr], compare, value — stores value iff *addr == compare; returns
     the old value either way. *)
  let atomic_cas intr ~elt_shift ~ty ~result =
    match args with
    | [EVar arr; idx_e; cmp_e; val_e] ->
        let r_idx = emit_expr buf alloc env idx_e in
        let r_cmp = emit_expr buf alloc env cmp_e in
        let r_val = emit_expr buf alloc env val_e in
        check_atom_operand intr result r_cmp ;
        check_atom_operand intr result r_val ;
        let space, r_addr =
          atomic_addr ~intr ~global_only:false ~elt_shift arr r_idx
        in
        let r_old = new_atom_result result in
        emit
          buf
          "atom.%s.cas%s %s, [%s], %s, %s;"
          space
          ty
          r_old
          r_addr
          r_cmp
          r_val ;
        r_old
    | _ ->
        unsupported (intr ^ ": expects (array-variable, index, compare, value)")
  in
  (* atom.{shared,global}.{inc,dec}.u32 with limit 0xffffffff. Semantics
     note: PTX inc/dec WRAP at the limit operand —
       inc: d = (old >= limit) ? 0 : old + 1
       dec: d = (old == 0 || old > limit) ? limit : old - 1
     With limit = 0xffffffff both coincide exactly with add/sub of 1 modulo
     2^32, i.e. the interpreter's plain ±1 semantics; a smaller limit would
     turn them into wrapping ring-buffer counters, which no Sarek intrinsic
     exposes yet. *)
  let atomic_incdec intr ~global_only ~op =
    match args with
    | [EVar arr; idx_e] ->
        let r_idx = emit_expr buf alloc env idx_e in
        let space, r_addr =
          atomic_addr ~intr ~global_only ~elt_shift:2 arr r_idx
        in
        let r_lim = new_u32 alloc in
        emit buf "mov.u32 %s, 0xffffffff;" r_lim ;
        let r_old = new_u32 alloc in
        emit buf "atom.%s.%s.u32 %s, [%s], %s;" space op r_old r_addr r_lim ;
        r_old
    | _ -> unsupported (intr ^ ": expects (array-variable, index)")
  in
  (* Binary min/max: native PTX op. Both operands must share a register
     class — a mixed-width op would be invalid PTX. *)
  let binary_minmax intr op =
    match args with
    | [a; b] ->
        let ra = emit_expr buf alloc env a in
        let rb = emit_expr buf alloc env b in
        let mismatch () =
          unsupported
            (intr ^ ": operands " ^ ra ^ " and " ^ rb
           ^ " have different register classes; cast one operand first")
        in
        if is_f64_reg ra then (
          if not (is_f64_reg rb) then mismatch ()
          else
            let r = new_f64 alloc in
            emit buf "%s.f64 %s, %s, %s;" op r ra rb ;
            r)
        else if is_f32_reg ra then (
          if not (is_f32_reg rb) then mismatch ()
          else
            let r = new_f32 alloc in
            emit buf "%s.f32 %s, %s, %s;" op r ra rb ;
            r)
        else if is_f32_reg rb || is_f64_reg rb then mismatch ()
        else
          let is_u64 r = String.length r >= 3 && r.[1] = 'r' && r.[2] = 'd' in
          if is_u64 ra then (
            if not (is_u64 rb) then mismatch ()
            else
              let r = new_u64 alloc in
              emit buf "%s.s64 %s, %s, %s;" op r ra rb ;
              r)
          else if is_u64 rb then mismatch ()
          else
            let r = new_u32 alloc in
            emit buf "%s.s32 %s, %s, %s;" op r ra rb ;
            r
    | _ -> unsupported (intr ^ " arity != 2")
  in
  (* Unary same-type float rounding via cvt (rmi = floor, rpi = ceil). *)
  let unary_round intr cvt =
    match args with
    | [a] ->
        let r = emit_expr buf alloc env a in
        if is_f64_reg r then (
          let d = new_f64 alloc in
          emit buf "%s.f64.f64 %s, %s;" cvt d r ;
          d)
        else if is_f32_reg r then (
          let d = new_f32 alloc in
          emit buf "%s.f32.f32 %s, %s;" cvt d r ;
          d)
        else unsupported (intr ^ ": float operand required")
    | _ -> unsupported (intr ^ " arity != 1")
  in
  (* Emit the argument(s) of a math intrinsic (no width check). *)
  let unary_arg intr =
    match args with
    | [a] -> emit_expr buf alloc env a
    | _ -> unsupported (intr ^ " arity != 1")
  in
  let binary_args intr =
    match args with
    | [a; b] -> (emit_expr buf alloc env a, emit_expr buf alloc env b)
    | _ -> unsupported (intr ^ " arity != 2")
  in
  (* Clean rejection for a math intrinsic with no f64 lowering. Never emit an
     .f32-suffixed op on an f64 register: the mismatch is invalid PTX that
     only fails at module-load time (cuModuleLoadData). *)
  let no_f64 intr =
    unsupported
      (intr ^ ": no native f64 " ^ intr
     ^ " in PTX; compute in float32 or on the CPU")
  in
  (* Argument of a unary f32-only (.approx) lowering: f64 operands are
     rejected cleanly, never emitted with an .f32 suffix. *)
  let unary_f32_arg intr =
    let r = unary_arg intr in
    if is_f32_reg r then r
    else if is_f64_reg r then no_f64 intr
    else unsupported (intr ^ ": float operand required")
  in
  let binary_f32_args intr =
    let ra, rb = binary_args intr in
    if is_f32_reg ra && is_f32_reg rb then (ra, rb)
    else if is_f64_reg ra || is_f64_reg rb then no_f64 intr
    else unsupported (intr ^ ": float operands required")
  in
  (* Unary float op with (up to) one native instruction per width; [None]
     means the width has no lowering and is rejected cleanly. *)
  let unary_native intr ~f32_op ~f64_op =
    let r_arg = unary_arg intr in
    if is_f64_reg r_arg then
      match f64_op with
      | Some op ->
          let d = new_f64 alloc in
          emit buf "%s %s, %s;" op d r_arg ;
          d
      | None -> no_f64 intr
    else if is_f32_reg r_arg then
      match f32_op with
      | Some op ->
          let d = new_f32 alloc in
          emit buf "%s %s, %s;" op d r_arg ;
          d
      | None -> unsupported (intr ^ ": no f32 lowering")
    else unsupported (intr ^ ": float operand required")
  in
  (* f32 building blocks for the transcendental compositions below (all based
     on .approx ops — f32 precision only, caveats at each call site). *)
  let f32_op1 op a =
    let d = new_f32 alloc in
    emit buf "%s %s, %s;" op d a ;
    d
  in
  let f32_op2 op a b =
    let d = new_f32 alloc in
    emit buf "%s %s, %s, %s;" op d a b ;
    d
  in
  let f32_mul_const a bits =
    let d = new_f32 alloc in
    emit buf "mul.f32 %s, %s, 0F%08lX;" d a bits ;
    d
  in
  match name with
  | "thread_id_x" | "thread_idx_x" ->
      let r = new_u32 alloc in
      emit buf "mov.u32 %s, %%tid.x;" r ;
      r
  | "thread_id_y" | "thread_idx_y" ->
      let r = new_u32 alloc in
      emit buf "mov.u32 %s, %%tid.y;" r ;
      r
  | "thread_id_z" | "thread_idx_z" ->
      let r = new_u32 alloc in
      emit buf "mov.u32 %s, %%tid.z;" r ;
      r
  | "block_id_x" | "block_idx_x" ->
      let r = new_u32 alloc in
      emit buf "mov.u32 %s, %%ctaid.x;" r ;
      r
  | "block_id_y" | "block_idx_y" ->
      let r = new_u32 alloc in
      emit buf "mov.u32 %s, %%ctaid.y;" r ;
      r
  | "block_id_z" | "block_idx_z" ->
      let r = new_u32 alloc in
      emit buf "mov.u32 %s, %%ctaid.z;" r ;
      r
  | "block_dim_x" ->
      let r = new_u32 alloc in
      emit buf "mov.u32 %s, %%ntid.x;" r ;
      r
  | "block_dim_y" ->
      let r = new_u32 alloc in
      emit buf "mov.u32 %s, %%ntid.y;" r ;
      r
  | "block_dim_z" ->
      let r = new_u32 alloc in
      emit buf "mov.u32 %s, %%ntid.z;" r ;
      r
  | "grid_dim_x" ->
      let r = new_u32 alloc in
      emit buf "mov.u32 %s, %%nctaid.x;" r ;
      r
  | "grid_dim_y" ->
      let r = new_u32 alloc in
      emit buf "mov.u32 %s, %%nctaid.y;" r ;
      r
  | "grid_dim_z" ->
      let r = new_u32 alloc in
      emit buf "mov.u32 %s, %%nctaid.z;" r ;
      r
  | "global_thread_id" | "global_idx" | "global_idx_x" ->
      let r_tid = new_u32 alloc in
      emit buf "mov.u32 %s, %%tid.x;" r_tid ;
      let r_bid = new_u32 alloc in
      emit buf "mov.u32 %s, %%ctaid.x;" r_bid ;
      let r_bdim = new_u32 alloc in
      emit buf "mov.u32 %s, %%ntid.x;" r_bdim ;
      let r_off = new_u32 alloc in
      emit buf "mul.lo.u32 %s, %s, %s;" r_off r_bid r_bdim ;
      let r_gid = new_u32 alloc in
      emit buf "add.u32 %s, %s, %s;" r_gid r_tid r_off ;
      r_gid
  | "global_idx_y" ->
      let r_tid = new_u32 alloc in
      emit buf "mov.u32 %s, %%tid.y;" r_tid ;
      let r_bid = new_u32 alloc in
      emit buf "mov.u32 %s, %%ctaid.y;" r_bid ;
      let r_bdim = new_u32 alloc in
      emit buf "mov.u32 %s, %%ntid.y;" r_bdim ;
      let r_off = new_u32 alloc in
      emit buf "mul.lo.u32 %s, %s, %s;" r_off r_bid r_bdim ;
      let r_gid = new_u32 alloc in
      emit buf "add.u32 %s, %s, %s;" r_gid r_tid r_off ;
      r_gid
  | "global_size" ->
      let r_bdim = new_u32 alloc in
      emit buf "mov.u32 %s, %%ntid.x;" r_bdim ;
      let r_gdim = new_u32 alloc in
      emit buf "mov.u32 %s, %%nctaid.x;" r_gdim ;
      let r = new_u32 alloc in
      emit buf "mul.lo.u32 %s, %s, %s;" r r_bdim r_gdim ;
      r
  | "block_barrier" ->
      emit buf "bar.sync 0;" ;
      let r = new_u32 alloc in
      emit buf "mov.u32 %s, 0;" r ;
      r
  | "sin" -> f32_op1 "sin.approx.f32" (unary_f32_arg "sin")
  | "cos" -> f32_op1 "cos.approx.f32" (unary_f32_arg "cos")
  | "tan" ->
      (* tan = sin/cos, all .approx: f32 precision only. *)
      let r_arg = unary_f32_arg "tan" in
      let r_sin = f32_op1 "sin.approx.f32" r_arg in
      let r_cos = f32_op1 "cos.approx.f32" r_arg in
      f32_op2 "div.approx.f32" r_sin r_cos
  | "sqrt" ->
      unary_native
        "sqrt"
        ~f32_op:(Some "sqrt.approx.f32")
        ~f64_op:(Some "sqrt.rn.f64")
  | "exp" ->
      (* exp(x) = 2^(x·log2 e); PTX only has base-2 ex2 (.approx, f32). *)
      let r_arg = unary_f32_arg "exp" in
      f32_op1 "ex2.approx.f32" (f32_mul_const r_arg f32_log2_e_bits)
  | "log" ->
      (* log(x) = log2(x)·ln 2; PTX only has base-2 lg2 (.approx, f32). *)
      let r_arg = unary_f32_arg "log" in
      f32_mul_const (f32_op1 "lg2.approx.f32" r_arg) f32_ln_2_bits
  | "log10" ->
      (* log10(x) = log2(x)·log10 2; .approx, f32 precision only. *)
      let r_arg = unary_f32_arg "log10" in
      f32_mul_const (f32_op1 "lg2.approx.f32" r_arg) f32_log10_2_bits
  | "pow" ->
      (* pow(x,y) = 2^(y·log2 x) via lg2/ex2 (.approx, f32). Domain caveat:
         valid for x > 0 only — lg2 of a negative is NaN, so integer-exponent
         negative bases are not handled. *)
      let ra, rb = binary_f32_args "pow" in
      let r_lg = f32_op1 "lg2.approx.f32" ra in
      f32_op1 "ex2.approx.f32" (f32_op2 "mul.f32" rb r_lg)
  | "sinh" | "cosh" ->
      (* sinh/cosh x = (e^x ∓ e^-x)/2 via ex2(±x·log2 e); .approx, f32. *)
      let r_arg = unary_f32_arg name in
      let r_t = f32_mul_const r_arg f32_log2_e_bits in
      let r_pos = f32_op1 "ex2.approx.f32" r_t in
      let r_neg = f32_op1 "ex2.approx.f32" (f32_op1 "neg.f32" r_t) in
      let comb = if name = "sinh" then "sub.f32" else "add.f32" in
      f32_mul_const (f32_op2 comb r_pos r_neg) f32_half_bits
  | "tanh" ->
      (* tanh x = copysign(1 − 2/(e^(2|x|) + 1), x) via ex2(2|x|·log2 e).
         Using |x| keeps the exponential finite-or-+inf only: at overflow
         e^(2|x|) = +inf, 2/(inf+1) = 0 and the result saturates to ±1
         instead of the NaN the naive (e^2x−1)/(e^2x+1) form produces. *)
      let r_arg = unary_f32_arg "tanh" in
      let r_abs = f32_op1 "abs.f32" r_arg in
      let r_e2x =
        f32_op1 "ex2.approx.f32" (f32_mul_const r_abs f32_two_log2_e_bits)
      in
      let r_den = new_f32 alloc in
      emit buf "add.f32 %s, %s, 0F%08lX;" r_den r_e2x f32_one_bits ;
      let r_two = new_f32 alloc in
      emit buf "mov.f32 %s, 0F%08lX;" r_two (Int32.bits_of_float 2.0) ;
      let r_frac = f32_op2 "div.approx.f32" r_two r_den in
      let r_mag = new_f32 alloc in
      emit buf "sub.f32 %s, 0F%08lX, %s;" r_mag f32_one_bits r_frac ;
      let r = new_f32 alloc in
      emit buf "copysign.f32 %s, %s, %s;" r r_arg r_mag ;
      r
  | "asin" | "acos" | "atan" | "atan2" | "expm1" | "log1p" -> (
      (* f32 path: no native PTX instruction and no accurate ex2/lg2
         composition (and for expm1/log1p an exp/log composition would lose
         the near-zero precision they exist for). Widen to f64
         (cvt.rn.f64.f32), run the softmath f64 helper, round the result back
         (cvt.rn.f32.f64) — precision trivially exceeds f32 ulp. PERF: the
         f64 helper runs at the fp64 rate (1/16 of fp32 on most consumer
         GPUs); acceptable for these rare functions — a native-f32
         composition can come later if profiling demands. f64 callers reach
         the softmath helper directly via the ["Float64"] path. *)
      let hname =
        match Sarek_ir_ptx_softmath.helper_name name with
        | Some h -> h
        | None ->
            fail ("PTX codegen: internal error: no softmath helper for " ^ name)
      in
      Sarek_ir_ptx_softmath.register alloc.funcs ;
      let f =
        {
          var_name = hname;
          var_id = -1;
          var_type = TFloat64;
          var_mutable = false;
        }
      in
      let args64 = List.map (fun a -> ECast (TFloat64, a)) args in
      match emit_app buf alloc env f args64 with
      | Scalar r -> emit_cast buf alloc r TFloat32
      | Agg _ ->
          fail
            "PTX codegen: internal error: softmath helper returned an aggregate"
      )
  | "fabs" | "abs_float" ->
      unary_native name ~f32_op:(Some "abs.f32") ~f64_op:(Some "abs.f64")
  | "copysign" ->
      (* PTX: copysign d, a, b = |b| with a's sign. OCaml copysign x y = |x|
         with y's sign — the sign source (second Sarek argument) therefore
         goes in the FIRST PTX operand slot. *)
      let ra, rb = binary_args "copysign" in
      if is_f64_reg ra && is_f64_reg rb then (
        let d = new_f64 alloc in
        emit buf "copysign.f64 %s, %s, %s;" d rb ra ;
        d)
      else if is_f32_reg ra && is_f32_reg rb then (
        let d = new_f32 alloc in
        emit buf "copysign.f32 %s, %s, %s;" d rb ra ;
        d)
      else
        unsupported
          ("copysign: operands " ^ ra ^ ", " ^ rb
         ^ " must both be f32 or both f64; cast one operand first")
  | "hypot" ->
      (* hypot = sqrt(x² + y²) via mul + fma + sqrt. No overflow/underflow
         rescaling: exact enough for moderate magnitudes (f64 uses only
         correctly-rounded ops), wrong near FLT/DBL_MAX. *)
      let ra, rb = binary_args "hypot" in
      if is_f64_reg ra && is_f64_reg rb then (
        let r_xx = new_f64 alloc in
        emit buf "mul.f64 %s, %s, %s;" r_xx ra ra ;
        let r_sum = new_f64 alloc in
        emit buf "fma.rn.f64 %s, %s, %s, %s;" r_sum rb rb r_xx ;
        let d = new_f64 alloc in
        emit buf "sqrt.rn.f64 %s, %s;" d r_sum ;
        d)
      else if is_f32_reg ra && is_f32_reg rb then (
        let r_xx = f32_op2 "mul.f32" ra ra in
        let r_sum = new_f32 alloc in
        emit buf "fma.rn.f32 %s, %s, %s, %s;" r_sum rb rb r_xx ;
        f32_op1 "sqrt.rn.f32" r_sum)
      else
        unsupported
          ("hypot: operands " ^ ra ^ ", " ^ rb
         ^ " must both be f32 or both f64; cast one operand first")
  (* Bitcasts between f64 and int64 (mov.b64) — the exponent-field plumbing
     the softmath f64 transcendentals are built on. *)
  | "f64_bits" ->
      let r = unary_arg "f64_bits" in
      if is_f64_reg r then (
        let d = new_u64 alloc in
        emit buf "mov.b64 %s, %s;" d r ;
        d)
      else unsupported "f64_bits: f64 operand required"
  | "bits_f64" ->
      let r = unary_arg "bits_f64" in
      if String.length r >= 3 && r.[1] = 'r' && r.[2] = 'd' then (
        let d = new_f64 alloc in
        emit buf "mov.b64 %s, %s;" d r ;
        d)
      else unsupported "bits_f64: int64 operand required"
  | "fma" -> (
      match args with
      | [a; b; c] ->
          let ra = emit_expr buf alloc env a in
          let rb = emit_expr buf alloc env b in
          let rc = emit_expr buf alloc env c in
          if is_f64_reg ra && is_f64_reg rb && is_f64_reg rc then (
            let r = new_f64 alloc in
            emit buf "fma.rn.f64 %s, %s, %s, %s;" r ra rb rc ;
            r)
          else if is_f32_reg ra && is_f32_reg rb && is_f32_reg rc then (
            let r = new_f32 alloc in
            emit buf "fma.rn.f32 %s, %s, %s, %s;" r ra rb rc ;
            r)
          else
            unsupported
              ("fma: operands " ^ ra ^ ", " ^ rb ^ ", " ^ rc
             ^ " must all be f32 or all f64; cast the mismatched operand")
      | _ -> unsupported "fma arity != 3")
  (* Type conversions (Gpu.float / Float32.of_int / …). "of_int"/"to_int"
     are path-dependent: they exist in both the Float32 and Float64 stdlib
     modules. *)
  | "float" | "float_of_int" -> unary_cast name TFloat32
  | "float64" | "float64_of_int" -> unary_cast name TFloat64
  | "int_of_float" | "int_of_float64" -> unary_cast name TInt32
  | "of_int" ->
      if List.exists (fun p -> p = "Float64") path then unary_cast name TFloat64
      else unary_cast name TFloat32
  | "to_int" -> unary_cast name TInt32
  (* Atomics (int32 add; old value returned). *)
  (* Native math with a direct PTX op. *)
  | "min" -> binary_minmax name "min"
  | "max" -> binary_minmax name "max"
  | "floor" -> unary_round name "cvt.rmi"
  | "ceil" -> unary_round name "cvt.rpi"
  | "rsqrt" ->
      (* f64: rcp.rn∘sqrt.rn — rsqrt.approx.f64 exists but is low-precision
         (~1e-4); the two correctly-rounded ops give ~1-ulp-per-op accuracy.
         f32 keeps the fast rsqrt.approx.f32. *)
      let r = unary_arg "rsqrt" in
      if is_f64_reg r then (
        let r_sqrt = new_f64 alloc in
        emit buf "sqrt.rn.f64 %s, %s;" r_sqrt r ;
        let d = new_f64 alloc in
        emit buf "rcp.rn.f64 %s, %s;" d r_sqrt ;
        d)
      else if is_f32_reg r then (
        let d = new_f32 alloc in
        emit buf "rsqrt.approx.f32 %s, %s;" d r ;
        d)
      else unsupported "rsqrt: float operand required"
  (* Atomics (old value returned). Shared vs global is auto-detected from the
     array's memory space; the *_global_* names force the global path. *)
  | "atomic_add_int32" ->
      atomic_rmw
        name
        ~global_only:false
        ~elt_shift:2
        ~op:"add"
        ~ty:".s32"
        ~result:`U32
  | "atomic_add_global_int32" ->
      atomic_rmw
        name
        ~global_only:true
        ~elt_shift:2
        ~op:"add"
        ~ty:".s32"
        ~result:`U32
  | "atomic_sub_int32" ->
      atomic_rmw
        name
        ~global_only:false
        ~elt_shift:2
        ~op:"sub"
        ~ty:".s32"
        ~result:`U32
  | "atomic_min_int32" ->
      atomic_rmw
        name
        ~global_only:false
        ~elt_shift:2
        ~op:"min"
        ~ty:".s32"
        ~result:`U32
  | "atomic_max_int32" ->
      atomic_rmw
        name
        ~global_only:false
        ~elt_shift:2
        ~op:"max"
        ~ty:".s32"
        ~result:`U32
  | "atomic_and_int32" ->
      atomic_rmw
        name
        ~global_only:false
        ~elt_shift:2
        ~op:"and"
        ~ty:".b32"
        ~result:`U32
  | "atomic_or_int32" ->
      atomic_rmw
        name
        ~global_only:false
        ~elt_shift:2
        ~op:"or"
        ~ty:".b32"
        ~result:`U32
  | "atomic_xor_int32" ->
      atomic_rmw
        name
        ~global_only:false
        ~elt_shift:2
        ~op:"xor"
        ~ty:".b32"
        ~result:`U32
  | "atomic_exch_int32" ->
      atomic_rmw
        name
        ~global_only:false
        ~elt_shift:2
        ~op:"exch"
        ~ty:".b32"
        ~result:`U32
  (* exch generalized to 64-bit elements. No stdlib/ppx int64-exch name
     exists yet; the emitter accepts the conventional name ahead of it
     (snapshot-only coverage). *)
  | "atomic_exch_int64" ->
      atomic_rmw
        name
        ~global_only:false
        ~elt_shift:3
        ~op:"exch"
        ~ty:".b64"
        ~result:`U64
  (* atom.add.u64: PTX add has no .s64 form; u64 add is two's-complement,
     identical to signed int64 add. *)
  | "atomic_add_int64" ->
      atomic_rmw
        name
        ~global_only:false
        ~elt_shift:3
        ~op:"add"
        ~ty:".u64"
        ~result:`U64
  | "atomic_add_float32" ->
      atomic_rmw
        name
        ~global_only:false
        ~elt_shift:2
        ~op:"add"
        ~ty:".f32"
        ~result:`F32
  (* atom.add.f64 requires sm_60+; the default target is sm_86. ZLUDA
     support for f64 atomics is unverified — snapshot coverage only. *)
  | "atomic_add_float64" ->
      atomic_rmw
        name
        ~global_only:false
        ~elt_shift:3
        ~op:"add"
        ~ty:".f64"
        ~result:`F64
  | "atomic_cas_int32" -> atomic_cas name ~elt_shift:2 ~ty:".b32" ~result:`U32
  (* No stdlib/ppx int64-CAS name exists yet; the emitter accepts the
     conventional name ahead of it (snapshot-only coverage). *)
  | "atomic_cas_int64" -> atomic_cas name ~elt_shift:3 ~ty:".b64" ~result:`U64
  | "atomic_inc_int32" -> atomic_incdec name ~global_only:false ~op:"inc"
  | "atomic_inc_global_int32" -> atomic_incdec name ~global_only:true ~op:"inc"
  | "atomic_dec_int32" -> atomic_incdec name ~global_only:false ~op:"dec"
  | n -> unsupported ("intrinsic: " ^ n)
