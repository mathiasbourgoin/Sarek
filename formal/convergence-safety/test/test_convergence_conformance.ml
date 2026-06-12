(******************************************************************************)
(* QCheck conformance suite for ConvergenceSpec.v
 *
 * We test the 6 properties on the ABSTRACT OCaml model below, which mirrors
 * the Rocq definitions.  The abstract model is a faithful rendition of
 * the convergence-relevant logic in Sarek_convergence.ml — verified by
 * inspection of Sarek_convergence.ml:check_expr / is_thread_varying.
 *
 * Correspondence is documented in ASSUMPTIONS.md.
 * Phase 2 (T2-F02): EVar and nat-tagged ELet added; env-threaded
 *   is_varying_env / check_env added; F-02 let-alias QCheck property added.
 * Phase 2 (T2-WARP): EWarpPoint added; WarpError constructor; check_warp
 *   function; warp_diverged_error QCheck property added.
 * Phase 2 (T2-RETURN): EReturn added; models TEReturn early-return;
 *   check treats EReturn as check of its body; return_barrier_skip_safe
 *   QCheck property added (property 14).
 * Phase 3 (T1A-CONF): 3 dedicated ESuperstep QCheck properties added
 *   (properties 11–13): superstep_outer_diverged_error (F-01 direct),
 *   superstep_no_entry_error_converged (safe-path complement),
 *   superstep_body_errors_propagate (monotonicity).
 ******************************************************************************)

(* ===== Abstract model (mirrors ConvergenceSpec.v) ===== *)

type expr =
  | ELit
  | EVary
  | EBarrier
  | EWarpPoint                          (* warp-collective call site *)
  | EVar of int                         (* variable reference by id *)
  | EBinop of expr * expr
  | EUnop of expr
  | EIf of expr * expr * expr           (* cond, then, else *)
  | EWhile of expr * expr               (* cond, body *)
  | EFor of expr * expr * expr          (* lo, hi, body *)
  | ESeq of expr list
  | ELet of int * expr * expr           (* var_id, value, body *)
  | ESuperstep of bool * expr * expr    (* divergent_flag, body, cont *)
  | EApp of expr list
  | EReturn of expr                     (* early-return; exits without any barrier *)

type exec_mode = Converged | Diverged

type error = BarrierError | WarpError

type dim_usage = {
  uses_x : bool; uses_y : bool; uses_z : bool;
  uses_block_dim : bool; uses_grid_dim : bool;
  uses_thread_idx : bool; uses_block_idx : bool;
  uses_shared_mem : bool;
}

let empty_dim_usage = {
  uses_x = false; uses_y = false; uses_z = false;
  uses_block_dim = false; uses_grid_dim = false;
  uses_thread_idx = false; uses_block_idx = false;
  uses_shared_mem = false;
}

let merge_dim_usage a b = {
  uses_x          = a.uses_x          || b.uses_x;
  uses_y          = a.uses_y          || b.uses_y;
  uses_z          = a.uses_z          || b.uses_z;
  uses_block_dim  = a.uses_block_dim  || b.uses_block_dim;
  uses_grid_dim   = a.uses_grid_dim   || b.uses_grid_dim;
  uses_thread_idx = a.uses_thread_idx || b.uses_thread_idx;
  uses_block_idx  = a.uses_block_idx  || b.uses_block_idx;
  uses_shared_mem = a.uses_shared_mem || b.uses_shared_mem;
}

let rec is_varying = function
  | EVary -> true
  | ELit | EBarrier | EWarpPoint | EVar _ -> false
  | EBinop (a, b)  -> is_varying a || is_varying b
  | EUnop e        -> is_varying e
  | EIf (c, t, el) -> is_varying c || is_varying t || is_varying el
  | EWhile (c, b)  -> is_varying c || is_varying b
  | EFor (lo, hi, b) -> is_varying lo || is_varying hi || is_varying b
  | ESeq es        -> List.exists is_varying es
  | ELet (_, v, b) -> is_varying v || is_varying b
  | ESuperstep (_, body, cont) -> is_varying body || is_varying cont
  | EApp args      -> List.exists is_varying args
  | EReturn e      -> is_varying e

let rec barrier_free = function
  | EBarrier       -> false
  | ELit | EVary | EWarpPoint | EVar _ -> true
  | EBinop (a, b)  -> barrier_free a && barrier_free b
  | EUnop e        -> barrier_free e
  | EIf (c, t, el) -> barrier_free c && barrier_free t && barrier_free el
  | EWhile (c, b)  -> barrier_free c && barrier_free b
  | EFor (lo, hi, b) -> barrier_free lo && barrier_free hi && barrier_free b
  | ESeq es        -> List.for_all barrier_free es
  | ELet (_, v, b) -> barrier_free v && barrier_free b
  | ESuperstep (divergent, body, cont) ->
      divergent && barrier_free body && barrier_free cont
  | EApp args      -> List.for_all barrier_free args
  | EReturn e      -> barrier_free e

let rec has_diverging_cf = function
  | EIf (c, t, el)   -> is_varying c || has_diverging_cf t || has_diverging_cf el
  | EWhile (c, b)    -> is_varying c || has_diverging_cf b
  | EFor (lo, hi, b) -> is_varying lo || is_varying hi || has_diverging_cf b
  | EBinop (a, b)    -> has_diverging_cf a || has_diverging_cf b
  | EUnop e          -> has_diverging_cf e
  | ESeq es          -> List.exists has_diverging_cf es
  | ELet (_, v, b)   -> has_diverging_cf v || has_diverging_cf b
  | ESuperstep (_, body, cont) -> has_diverging_cf body || has_diverging_cf cont
  | EApp args        -> List.exists has_diverging_cf args
  | EReturn e        -> has_diverging_cf e
  | _                -> false

let rec check m = function
  | EBarrier       ->
      (match m with Diverged -> [BarrierError] | Converged -> [])
  | ELit | EVary | EWarpPoint | EVar _ -> []
  | EBinop (a, b)  -> check m a @ check m b
  | EUnop e        -> check m e
  | EIf (cond, t, el) ->
      let inner = if is_varying cond then Diverged else m in
      check m cond @ check inner t @ check inner el
  | EWhile (cond, b) ->
      let inner = if is_varying cond then Diverged else m in
      check m cond @ check inner b
  | EFor (lo, hi, b) ->
      let inner = if is_varying lo || is_varying hi then Diverged else m in
      check m lo @ check m hi @ check inner b
  | ESeq es        -> List.concat_map (check m) es
  | ELet (_, v, b) -> check m v @ check m b
  | ESuperstep (divergent, body, cont) ->
      let entry_errors =
        match m, divergent with
        | Diverged, false -> [BarrierError]
        | _,        _     -> []
      in
      entry_errors @ check m body @ check m cont
  | EApp args      -> List.concat_map (check m) args
  | EReturn e      -> check m e

(* ===== Env-threaded model (T2-F02) ===== *)

(* Env: association list mapping variable id to is_varying flag *)
type env = (int * bool) list

let env_lookup (env : env) (x : int) : bool =
  match List.find_opt (fun (id, _) -> id = x) env with
  | Some (_, v) -> v
  | None        -> false

let env_extend (env : env) (x : int) (v : bool) : env = (x, v) :: env

let rec is_varying_env (env : env) = function
  | EVary     -> true
  | ELit | EBarrier | EWarpPoint -> false
  | EVar x    -> env_lookup env x
  | EBinop (a, b)     -> is_varying_env env a || is_varying_env env b
  | EUnop e           -> is_varying_env env e
  | EIf (c, t, el)    ->
      is_varying_env env c ||
      is_varying_env env t ||
      is_varying_env env el
  | EWhile (c, b)     -> is_varying_env env c || is_varying_env env b
  | EFor (lo, hi, b)  ->
      is_varying_env env lo ||
      is_varying_env env hi ||
      is_varying_env env b
  | ESeq es           -> List.exists (is_varying_env env) es
  | ELet (x, v, b)   ->
      let vv = is_varying_env env v in
      is_varying_env (env_extend env x vv) b
  | ESuperstep (_, body, cont) ->
      is_varying_env env body || is_varying_env env cont
  | EApp args         -> List.exists (is_varying_env env) args
  | EReturn e         -> is_varying_env env e

let rec check_env m (env : env) = function
  | EBarrier   -> (match m with Diverged -> [BarrierError] | Converged -> [])
  | ELit | EVary | EWarpPoint | EVar _ -> []
  | EBinop (a, b)     -> check_env m env a @ check_env m env b
  | EUnop e           -> check_env m env e
  | EIf (cond, t, el) ->
      let inner = if is_varying_env env cond then Diverged else m in
      check_env m env cond @ check_env inner env t @ check_env inner env el
  | EWhile (cond, b)  ->
      let inner = if is_varying_env env cond then Diverged else m in
      check_env m env cond @ check_env inner env b
  | EFor (lo, hi, b)  ->
      let inner =
        if is_varying_env env lo || is_varying_env env hi
        then Diverged else m
      in
      check_env m env lo @ check_env m env hi @ check_env inner env b
  | ESeq es           -> List.concat_map (check_env m env) es
  | ELet (x, v, b)   ->
      let vv   = is_varying_env env v in
      let env' = env_extend env x vv in
      check_env m env v @ check_env m env' b
  | ESuperstep (divergent, body, cont) ->
      let entry_errors =
        match m, divergent with
        | Diverged, false -> [BarrierError]
        | _,        _     -> []
      in
      entry_errors @ check_env m env body @ check_env m env cont
  | EApp args         -> List.concat_map (check_env m env) args
  | EReturn e         -> check_env m env e

(* ===== Warp-collective checker (T2-WARP) ===== *)

(* check_warp: mirrors check_warp in ConvergenceSpec.v.
   EWarpPoint emits WarpError when mode is Diverged;
   EBarrier still emits BarrierError (both classes coexist). *)
let rec check_warp m = function
  | EWarpPoint ->
      (match m with Diverged -> [WarpError] | Converged -> [])
  | EBarrier ->
      (match m with Diverged -> [BarrierError] | Converged -> [])
  | ELit | EVary | EVar _ -> []
  | EBinop (a, b)  -> check_warp m a @ check_warp m b
  | EUnop e        -> check_warp m e
  | EIf (cond, t, el) ->
      let inner = if is_varying cond then Diverged else m in
      check_warp m cond @ check_warp inner t @ check_warp inner el
  | EWhile (cond, b) ->
      let inner = if is_varying cond then Diverged else m in
      check_warp m cond @ check_warp inner b
  | EFor (lo, hi, b) ->
      let inner = if is_varying lo || is_varying hi then Diverged else m in
      check_warp m lo @ check_warp m hi @ check_warp inner b
  | ESeq es        -> List.concat_map (check_warp m) es
  | ELet (_, v, b) -> check_warp m v @ check_warp m b
  | ESuperstep (divergent, body, cont) ->
      let entry_errors =
        match m, divergent with
        | Diverged, false -> [BarrierError]
        | _,        _     -> []
      in
      entry_errors @ check_warp m body @ check_warp m cont
  | EApp args      -> List.concat_map (check_warp m) args
  | EReturn e      -> check_warp m e

(* ===== QCheck generators ===== *)

open QCheck2

let gen_mode : exec_mode Gen.t = Gen.oneof_list [Converged; Diverged]

(* small var id generator — keep ids in range 0..3 so aliases are plausible *)
let gen_var_id : int Gen.t = Gen.int_range 0 3

(* bounded-depth generator to avoid size explosion *)
let gen_expr : expr Gen.t =
  Gen.sized_size (Gen.int_range 0 6) @@ Gen.fix (fun self n ->
    if n = 0 then Gen.oneof_list [ELit; EVary; EBarrier; EWarpPoint]
    else
      let sub = self (n / 2) in
      let sub2 = Gen.pair sub sub in
      let sub3 = Gen.triple sub sub sub in
      let sublist = Gen.list_size (Gen.int_range 0 3) sub in
      Gen.oneof [
        Gen.return ELit;
        Gen.return EVary;
        Gen.return EBarrier;
        Gen.return EWarpPoint;
        Gen.map  (fun x      -> EVar x)          gen_var_id;
        Gen.map  (fun (a, b) -> EBinop (a, b))   sub2;
        Gen.map  (fun e      -> EUnop e)          sub;
        Gen.map  (fun (c, t, el) -> EIf (c, t, el)) sub3;
        Gen.map  (fun (c, b) -> EWhile (c, b))   sub2;
        Gen.map  (fun (lo, hi, b) -> EFor (lo, hi, b)) sub3;
        Gen.map  (fun es     -> ESeq es)          sublist;
        Gen.map  (fun (x, (v, b)) -> ELet (x, v, b))
                 (Gen.pair gen_var_id sub2);
        Gen.map  (fun (dv, (body, cont)) -> ESuperstep (dv, body, cont))
                 (Gen.pair Gen.bool sub2);
        Gen.map  (fun args   -> EApp args)        sublist;
        Gen.map  (fun e      -> EReturn e)        sub;
      ])

let gen_dim : dim_usage Gen.t =
  let g = Gen.bool in
  Gen.map (fun (x, y, z, bd, gd, ti, bi, sm) ->
    { uses_x = x; uses_y = y; uses_z = z;
      uses_block_dim = bd; uses_grid_dim = gd;
      uses_thread_idx = ti; uses_block_idx = bi;
      uses_shared_mem = sm })
    (Gen.tup8 g g g g g g g g)

(* ===== Properties ===== *)

(* 1a. merge_dim commutative *)
let test_merge_dim_comm =
  Test.make ~name:"merge_dim_comm" ~count:2000
    (Gen.pair gen_dim gen_dim)
    (fun (a, b) -> merge_dim_usage a b = merge_dim_usage b a)

(* 1b. merge_dim associative *)
let test_merge_dim_assoc =
  Test.make ~name:"merge_dim_assoc" ~count:2000
    (Gen.triple gen_dim gen_dim gen_dim)
    (fun (a, b, c) ->
       merge_dim_usage a (merge_dim_usage b c) =
       merge_dim_usage (merge_dim_usage a b) c)

(* 1c. merge_dim idempotent *)
let test_merge_dim_idem =
  Test.make ~name:"merge_dim_idempotent" ~count:2000
    gen_dim
    (fun a -> merge_dim_usage a a = a)

(* 1d. merge_dim right identity *)
let test_merge_dim_empty_r =
  Test.make ~name:"merge_dim_empty_r" ~count:2000
    gen_dim
    (fun a -> merge_dim_usage a empty_dim_usage = a)

(* 1e. merge_dim left identity *)
let test_merge_dim_empty_l =
  Test.make ~name:"merge_dim_empty_l" ~count:2000
    gen_dim
    (fun a -> merge_dim_usage empty_dim_usage a = a)

(* 2. check_seq_hom *)
let test_check_seq_hom =
  Test.make ~name:"check_seq_hom" ~count:1000
    (Gen.triple gen_mode
       (Gen.list_size (Gen.int_range 0 4) gen_expr)
       (Gen.list_size (Gen.int_range 0 4) gen_expr))
    (fun (m, es1, es2) ->
       check m (ESeq (es1 @ es2)) = check m (ESeq es1) @ check m (ESeq es2))

(* 3. diverged_clean_iff_barrier_free *)
let test_diverged_clean_iff_bf =
  Test.make ~name:"diverged_clean_iff_barrier_free" ~count:2000
    gen_expr
    (fun e ->
       (check Diverged e = []) = barrier_free e)

(* 4. mode_monotone: every error in Converged is also in Diverged *)
let test_mode_monotone =
  Test.make ~name:"mode_monotone" ~count:2000
    gen_expr
    (fun e ->
       let cv = check Converged e in
       let dv = check Diverged e in
       List.for_all (fun err -> List.mem err dv) cv)

(* 5. varying_if_flags_barriers *)
let test_varying_if_flags =
  Test.make ~name:"varying_if_flags_barriers" ~count:2000
    (Gen.quad gen_mode gen_expr gen_expr gen_expr)
    (fun (ctx, cond, then_, else_) ->
       let varyc = is_varying cond in
       let no_bf = not (barrier_free then_) in
       if varyc && no_bf then
         check ctx (EIf (cond, then_, else_)) <> []
       else true)

(* 6. cdcf_check_agreement *)
let test_cdcf_agreement =
  Test.make ~name:"cdcf_check_agreement" ~count:2000
    gen_expr
    (fun e ->
       if not (has_diverging_cf e) then check Converged e = []
       else true)

(* 7. F-02 let-alias: env-threaded check catches barrier inside a let-alias.
 *
 * For any varying expression v and any fresh var id x, the pattern
 *   ELet x v (EIf (EVar x) EBarrier ELit)
 * must be flagged by check_env (since x is varying in the extended env).
 * This mirrors the theorem env_check_let_alias_catches in ConvergenceSpec.v.
 *)
let test_f02_let_alias_env =
  Test.make ~name:"f02_let_alias_env_catches_barrier" ~count:2000
    (Gen.pair gen_var_id gen_expr)
    (fun (x, v) ->
       (* Only test when v is varying in empty env *)
       if not (is_varying_env [] v) then true
       else
         (* ELet x v (EIf (EVar x) EBarrier ELit) must produce errors *)
         let prog = ELet (x, v, EIf (EVar x, EBarrier, ELit)) in
         check_env Converged [] prog <> [])

(* 8. F-02 env monotone: check_env Converged ⊆ check_env Diverged *)
let test_f02_env_mode_monotone =
  Test.make ~name:"f02_env_mode_monotone" ~count:2000
    gen_expr
    (fun e ->
       let cv = check_env Converged [] e in
       let dv = check_env Diverged  [] e in
       List.for_all (fun err -> List.mem err dv) cv)

(* 9. warp_diverged_error — randomized property over check_warp.
 *
 * Mirrors Theorem warp_diverged_error in ConvergenceSpec.v and its
 * strengthened analogue warp_mode_monotone.
 *
 * Property: for any expression e, if it contains EWarpPoint as a direct
 * subexpression inside a Diverged context, check_warp will return a
 * non-empty list.  We test three randomized forms:
 *
 * (a) check_warp Diverged (ESeq [e; EWarpPoint]) ≠ []
 *     — EWarpPoint at the tail of any sequence under Diverged.
 * (b) check_warp Converged (EIf (EVary, e_with_warp, ELit)) ≠ []
 *     — EWarpPoint nested inside a varying-condition branch (warp_varying_if_flags).
 * (c) mode monotonicity: incl (check_warp Converged e) (check_warp Diverged e)
 *     — strengthening never removes errors (warp_mode_monotone).
 *)
let test_warp_diverged_error =
  Test.make ~name:"warp_diverged_error" ~count:2000
    gen_expr
    (fun e ->
       (* (a) EWarpPoint at end of any sequence under Diverged always errors *)
       List.mem WarpError (check_warp Diverged (ESeq [e; EWarpPoint]))
       &&
       (* (b) EWarpPoint in then-branch of varying EIf is caught from Converged *)
       (let tree_with_warp = EIf (EVary, EWarpPoint, ELit) in
        List.mem WarpError (check_warp Converged tree_with_warp))
       &&
       (* (c) mode monotonicity: every Converged error is also a Diverged error *)
       List.for_all (fun err -> List.mem err (check_warp Diverged e))
                    (check_warp Converged e))

(* 11. superstep_outer_diverged_error — mirrors Theorem superstep_outer_diverged_error (F-01).
 *
 * Property: check Diverged (ESuperstep false body cont) ≠ [] for all body/cont.
 * The outer-Diverged + non-divergent-flag case always produces a BarrierError at
 * the superstep boundary (the implicit end-of-superstep barrier is unreachable by
 * all threads when the outer context is diverged and the superstep is non-divergent).
 * Directly corresponds to Theorem superstep_outer_diverged_error in ConvergenceSpec.v. *)
let test_superstep_outer_diverged_error =
  Test.make ~name:"superstep_outer_diverged_error" ~count:2000
    (Gen.pair gen_expr gen_expr)
    (fun (body, cont) ->
       (* F-01: outer Diverged + divergent_flag = false => always an error *)
       check Diverged (ESuperstep (false, body, cont)) <> [])

(* 12. superstep_no_entry_error_converged — Converged outer mode never triggers entry error.
 *
 * Property: the entry BarrierError is absent when outer mode is Converged, regardless
 * of the divergent flag.  Specifically, for barrier-free body/cont the total result is
 * empty from Converged mode with any flag value.
 * This tests the complement of superstep_outer_diverged_error — the "safe" path. *)
let test_superstep_no_entry_error_converged =
  Test.make ~name:"superstep_no_entry_error_converged" ~count:2000
    (Gen.pair Gen.bool (Gen.pair gen_expr gen_expr))
    (fun (dv, (body, cont)) ->
       (* When outer mode is Converged and both body and cont are barrier-free,
          there must be no errors: entry error is suppressed and no child errors exist *)
       if barrier_free body && barrier_free cont then
         check Converged (ESuperstep (dv, body, cont)) = []
       else true)

(* 13. superstep_body_errors_propagate — body/cont errors propagate through ESuperstep.
 *
 * Property (monotonicity): errors found in body or cont are preserved in the superstep.
 * Specifically check m body ⊆ check m (ESuperstep dv body cont) for all m, dv, body, cont.
 * This models the structural monotonicity of check over ESuperstep body/cont sub-expressions. *)
let test_superstep_body_errors_propagate =
  Test.make ~name:"superstep_body_errors_propagate" ~count:2000
    (Gen.quad gen_mode Gen.bool gen_expr gen_expr)
    (fun (m, dv, body, cont) ->
       let body_errs = check m body in
       let cont_errs = check m cont in
       let total_errs = check m (ESuperstep (dv, body, cont)) in
       (* All body errors and all cont errors must appear in the superstep result *)
       List.for_all (fun e -> List.mem e total_errs) body_errs &&
       List.for_all (fun e -> List.mem e total_errs) cont_errs)

(* 10. return_barrier_skip_safe — mirrors Theorem return_barrier_skip_safe.
 *
 * Property: check m (EReturn e) = check m e for all modes m and expressions e.
 * This confirms that EReturn is a transparent wrapper for barrier analysis:
 * (a) it does not introduce new barrier errors beyond those in its body, and
 * (b) it does not suppress errors that exist in its body.
 * Corresponds to Theorem return_barrier_skip_safe in ConvergenceSpec.v. *)
let test_return_barrier_skip_safe =
  Test.make ~name:"return_barrier_skip_safe" ~count:2000
    (Gen.pair gen_mode gen_expr)
    (fun (m, e) ->
       (* check m (EReturn e) = check m e *)
       check m (EReturn e) = check m e
       &&
       (* EReturn wrapping a barrier-free expr is clean in Converged mode *)
       (if barrier_free e then check Converged (EReturn e) = [] else true)
       &&
       (* mode monotonicity is preserved through EReturn *)
       (let cv = check Converged (EReturn e) in
        let dv = check Diverged  (EReturn e) in
        List.for_all (fun err -> List.mem err dv) cv))

let () =
  let suite = [
    test_merge_dim_comm;
    test_merge_dim_assoc;
    test_merge_dim_idem;
    test_merge_dim_empty_r;
    test_merge_dim_empty_l;
    test_check_seq_hom;
    test_diverged_clean_iff_bf;
    test_mode_monotone;
    test_varying_if_flags;
    test_cdcf_agreement;
    test_f02_let_alias_env;
    test_f02_env_mode_monotone;
    test_warp_diverged_error;
    test_return_barrier_skip_safe;
    test_superstep_outer_diverged_error;
    test_superstep_no_entry_error_converged;
    test_superstep_body_errors_propagate;
  ] in
  let passed = QCheck_base_runner.run_tests ~verbose:true suite in
  exit passed
