(******************************************************************************)
(* test_convergence_semantics.ml  (T3-S8)
 *
 * Differential conformance for the OPERATIONAL semantics (CMBT closure).
 * Where test_convergence_extraction.ml exercises the *static* checkers
 * (is_varying / barrier_free / check / check_warp), this suite exercises the
 * extracted operational evaluator ConvergenceModel.eval_concrete — the
 * fuel-indexed big-step evaluator with the abstract per-thread varying value
 * instantiated as the identity (vary_val := fun t -> t).
 *
 *   Rocq eval (ConvergenceSemantics.v) → coqc extraction → eval_concrete →
 *   this test.
 *
 * Four properties, each mirroring a PROVEN Rocq statement over random inputs:
 *
 *   (1) eval_fuel_monotone   — eval_concrete_fuel_monotone
 *       eval n t rho e = Some r  ⇒  eval (S n) t rho e = Some r.
 *
 *   (2) barrier_free_silent  — eval_concrete_barrier_free_silent
 *       barrier_free e ∧ superstep_free e ⇒ no EvBarrier in the trace.
 *
 *   (3) differential CMBT    — check_env_sound_core (T3-S4) instantiated
 *       e ∈ core_frag ∧ check_env Converged [] e = [] ⇒ the barrier
 *       projections (erase_warp) of two distinct threads' traces agree.
 *
 *   (4) F-04 hazard regression — hazard_not_barrier_safe (T3-S5)
 *       the F-04 hazard ESeq [EIf EVary (EReturn ELit) ELit; EBarrier]
 *       still produces DIFFERING barrier traces across threads, i.e. the
 *       checker's silence (check_env Converged [] hazard = []) is a real
 *       counterexample to barrier safety — NOT masked by the semantics.
 ******************************************************************************)

open QCheck2
module M = Convergence_model.ConvergenceModel

(* ===== Inline expr mirror + translation to the extracted type ===== *)

type expr =
  | ELit
  | EVary
  | EBarrier
  | EWarpPoint
  | EVar of int
  | EBinop of expr * expr
  | EUnop of expr
  | EIf of expr * expr * expr
  | EWhile of expr * expr
  | EFor of expr * expr * expr
  | ESeq of expr list
  | ELet of int * expr * expr
  | ESuperstep of bool * expr * expr
  | EApp of expr list
  | EReturn of expr

type exec_mode = Converged | Diverged

let rec to_extracted : expr -> M.expr = function
  | ELit -> ELit
  | EVary -> EVary
  | EBarrier -> EBarrier
  | EWarpPoint -> EWarpPoint
  | EVar x -> EVar x
  | EBinop (a, b) -> EBinop (to_extracted a, to_extracted b)
  | EUnop e -> EUnop (to_extracted e)
  | EIf (c, t, el) -> EIf (to_extracted c, to_extracted t, to_extracted el)
  | EWhile (c, b) -> EWhile (to_extracted c, to_extracted b)
  | EFor (lo, hi, b) -> EFor (to_extracted lo, to_extracted hi, to_extracted b)
  | ESeq es -> ESeq (List.map to_extracted es)
  | ELet (x, v, b) -> ELet (x, to_extracted v, to_extracted b)
  | ESuperstep (dv, body, cont) ->
      ESuperstep (dv, to_extracted body, to_extracted cont)
  | EApp args -> EApp (List.map to_extracted args)
  | EReturn e -> EReturn (to_extracted e)

(* ===== Inline static-analysis reference (mirrors ConvergenceSemantics.v) ===== *)

let rec barrier_free = function
  | EBarrier -> false
  | ELit | EVary | EWarpPoint | EVar _ -> true
  | EBinop (a, b) -> barrier_free a && barrier_free b
  | EUnop e -> barrier_free e
  | EIf (c, t, el) -> barrier_free c && barrier_free t && barrier_free el
  | EWhile (c, b) -> barrier_free c && barrier_free b
  | EFor (lo, hi, b) -> barrier_free lo && barrier_free hi && barrier_free b
  | ESeq es -> List.for_all barrier_free es
  | ELet (_, v, b) -> barrier_free v && barrier_free b
  | ESuperstep (dv, body, cont) -> dv && barrier_free body && barrier_free cont
  | EApp args -> List.for_all barrier_free args
  | EReturn e -> barrier_free e

(* superstep_free: no ESuperstep node anywhere (mirrors ConvergenceSemantics). *)
let rec superstep_free = function
  | ELit | EVary | EBarrier | EWarpPoint | EVar _ -> true
  | EBinop (a, b) -> superstep_free a && superstep_free b
  | EUnop e -> superstep_free e
  | EIf (c, t, el) -> superstep_free c && superstep_free t && superstep_free el
  | EWhile (c, b) -> superstep_free c && superstep_free b
  | EFor (lo, hi, b) ->
      superstep_free lo && superstep_free hi && superstep_free b
  | ESeq es -> List.for_all superstep_free es
  | ELet (_, v, b) -> superstep_free v && superstep_free b
  | ESuperstep (_, _, _) -> false
  | EApp args -> List.for_all superstep_free args
  | EReturn e -> superstep_free e

(* core_frag: the verified fragment (mirrors ConvergenceSemantics.core_frag).
   ESuperstep and EReturn are excluded. *)
let rec core_frag = function
  | ELit | EVary | EBarrier | EWarpPoint | EVar _ -> true
  | EBinop (a, b) -> core_frag a && core_frag b
  | EUnop e -> core_frag e
  | EIf (c, t, el) -> core_frag c && core_frag t && core_frag el
  | EWhile (c, b) -> core_frag c && core_frag b
  | EFor (lo, hi, b) -> core_frag lo && core_frag hi && core_frag b
  | ESeq es -> List.for_all core_frag es
  | ELet (_, v, b) -> core_frag v && core_frag b
  | ESuperstep (_, _, _) -> false
  | EApp args -> List.for_all core_frag args
  | EReturn _ -> false

(* Env-threaded varying analysis + checker (mirrors check_env). *)
type env = (int * bool) list

let env_lookup (env : env) (x : int) : bool =
  match List.find_opt (fun (id, _) -> id = x) env with
  | Some (_, v) -> v
  | None -> false

let env_extend (env : env) (x : int) (v : bool) : env = (x, v) :: env

let rec is_varying_env (env : env) = function
  | EVary -> true
  | ELit | EBarrier | EWarpPoint -> false
  | EVar x -> env_lookup env x
  | EBinop (a, b) -> is_varying_env env a || is_varying_env env b
  | EUnop e -> is_varying_env env e
  | EIf (c, t, el) ->
      is_varying_env env c || is_varying_env env t || is_varying_env env el
  | EWhile (c, b) -> is_varying_env env c || is_varying_env env b
  | EFor (lo, hi, b) ->
      is_varying_env env lo || is_varying_env env hi || is_varying_env env b
  | ESeq es -> List.exists (is_varying_env env) es
  | ELet (x, v, b) ->
      let vv = is_varying_env env v in
      is_varying_env (env_extend env x vv) b
  | ESuperstep (_, body, cont) ->
      is_varying_env env body || is_varying_env env cont
  | EApp args -> List.exists (is_varying_env env) args
  | EReturn e -> is_varying_env env e

let rec check_env m (env : env) = function
  | EBarrier -> ( match m with Diverged -> [`BarrierError] | Converged -> [])
  | ELit | EVary | EWarpPoint | EVar _ -> []
  | EBinop (a, b) -> check_env m env a @ check_env m env b
  | EUnop e -> check_env m env e
  | EIf (cond, t, el) ->
      let inner = if is_varying_env env cond then Diverged else m in
      check_env m env cond @ check_env inner env t @ check_env inner env el
  | EWhile (cond, b) ->
      let inner = if is_varying_env env cond then Diverged else m in
      check_env m env cond @ check_env inner env b
  | EFor (lo, hi, b) ->
      let inner =
        if is_varying_env env lo || is_varying_env env hi then Diverged else m
      in
      check_env m env lo @ check_env m env hi @ check_env inner env b
  | ESeq es -> List.concat_map (check_env m env) es
  | ELet (x, v, b) ->
      let vv = is_varying_env env v in
      let env' = env_extend env x vv in
      check_env m env v @ check_env m env' b
  | ESuperstep (divergent, body, cont) ->
      let entry_errors =
        match (m, divergent) with
        | Diverged, false -> [`BarrierError]
        | _, _ -> []
      in
      entry_errors @ check_env m env body @ check_env m env cont
  | EApp args -> List.concat_map (check_env m env) args
  | EReturn e -> check_env m env e

(* erase_warp: project a trace to its EvBarrier events (mirrors erase_warp). *)
let erase_warp (tr : M.event list) : M.event list =
  List.filter (function M.EvBarrier -> true | M.EvWarp -> false) tr

let has_barrier_event (tr : M.event list) : bool =
  List.exists (function M.EvBarrier -> true | M.EvWarp -> false) tr

(* ===== Generators ===== *)

let gen_var_id : int Gen.t = Gen.int_range 0 3

let gen_tid : int Gen.t = Gen.int_range 0 7

let gen_fuel : int Gen.t = Gen.int_range 0 30

(* Full-fragment generator (all constructors): used by the fuel-monotone
   property which holds for every expression. *)
let gen_expr : expr Gen.t =
  Gen.sized_size (Gen.int_range 0 5)
  @@ Gen.fix (fun self n ->
      if n = 0 then Gen.oneof_list [ELit; EVary; EBarrier; EWarpPoint]
      else
        let sub = self (n / 2) in
        let sub2 = Gen.pair sub sub in
        let sub3 = Gen.triple sub sub sub in
        let sublist = Gen.list_size (Gen.int_range 0 3) sub in
        Gen.oneof
          [
            Gen.return ELit;
            Gen.return EVary;
            Gen.return EBarrier;
            Gen.return EWarpPoint;
            Gen.map (fun x -> EVar x) gen_var_id;
            Gen.map (fun (a, b) -> EBinop (a, b)) sub2;
            Gen.map (fun e -> EUnop e) sub;
            Gen.map (fun (c, t, el) -> EIf (c, t, el)) sub3;
            Gen.map (fun (c, b) -> EWhile (c, b)) sub2;
            Gen.map (fun (lo, hi, b) -> EFor (lo, hi, b)) sub3;
            Gen.map (fun es -> ESeq es) sublist;
            Gen.map
              (fun (x, (v, b)) -> ELet (x, v, b))
              (Gen.pair gen_var_id sub2);
            Gen.map
              (fun (dv, (body, cont)) -> ESuperstep (dv, body, cont))
              (Gen.pair Gen.bool sub2);
            Gen.map (fun args -> EApp args) sublist;
            Gen.map (fun e -> EReturn e) sub;
          ])

(* Core-fragment generator: ESuperstep and EReturn excluded by construction, so
   every generated term satisfies core_frag = true. Used by the barrier-silence
   and differential properties. EWhile/EFor are excluded to avoid runs that
   simply exhaust fuel (None) — the verified properties are about COMPLETED
   evaluations, and these recursive forms rarely terminate within bounded fuel
   on random conditions; their barrier behaviour is covered by the Rocq proof. *)
let gen_core_expr : expr Gen.t =
  Gen.sized_size (Gen.int_range 0 5)
  @@ Gen.fix (fun self n ->
      if n = 0 then Gen.oneof_list [ELit; EVary; EBarrier; EWarpPoint]
      else
        let sub = self (n / 2) in
        let sub2 = Gen.pair sub sub in
        let sub3 = Gen.triple sub sub sub in
        let sublist = Gen.list_size (Gen.int_range 0 3) sub in
        Gen.oneof
          [
            Gen.return ELit;
            Gen.return EVary;
            Gen.return EBarrier;
            Gen.return EWarpPoint;
            Gen.map (fun x -> EVar x) gen_var_id;
            Gen.map (fun (a, b) -> EBinop (a, b)) sub2;
            Gen.map (fun e -> EUnop e) sub;
            Gen.map (fun (c, t, el) -> EIf (c, t, el)) sub3;
            Gen.map (fun es -> ESeq es) sublist;
            Gen.map
              (fun (x, (v, b)) -> ELet (x, v, b))
              (Gen.pair gen_var_id sub2);
            Gen.map (fun args -> EApp args) sublist;
          ])

(* ===== F-04 hazard (mirrors ConvergenceSemantics.hazard) ===== *)

let hazard : expr = ESeq [EIf (EVary, EReturn ELit, ELit); EBarrier]

(* ===== Properties ===== *)

let () =
  let big_fuel = 60 in
  let tests =
    [
      (* (1) Fuel monotonicity: eval n = Some r ⇒ eval (S n) = Some r. *)
      Test.make
        ~name:"sem:eval_fuel_monotone"
        ~count:3000
        (Gen.triple gen_fuel gen_tid gen_expr)
        (fun (n, t, e) ->
          let ex = to_extracted e in
          match M.eval_concrete n t [] ex with
          | None -> true (* vacuous: property only constrains success at n *)
          | Some r -> M.eval_concrete (n + 1) t [] ex = Some r);
      (* (2) Barrier silence: barrier_free ∧ superstep_free ⇒ no EvBarrier. *)
      Test.make
        ~name:"sem:barrier_free_silent"
        ~count:3000
        (Gen.pair gen_tid gen_expr)
        (fun (t, e) ->
          if barrier_free e && superstep_free e then
            match M.eval_concrete big_fuel t [] (to_extracted e) with
            | None -> true (* did not complete within fuel: nothing to check *)
            | Some (_, tr) -> not (has_barrier_event tr)
          else true (* precondition false: vacuously true *));
      (* (3) Differential CMBT: core_frag ∧ check_env Converged [] = [] ⇒
             two distinct threads' barrier projections agree. *)
      Test.make
        ~name:"sem:differential_barrier_safe"
        ~count:3000
        (Gen.triple gen_core_expr gen_tid gen_tid)
        (fun (e, t1, t2) ->
          if core_frag e && check_env Converged [] e = [] then begin
            let ex = to_extracted e in
            match
              ( M.eval_concrete big_fuel t1 [] ex,
                M.eval_concrete big_fuel t2 [] ex )
            with
            | Some (_, tr1), Some (_, tr2) -> erase_warp tr1 = erase_warp tr2
            | _ -> true (* one side did not complete: nothing to compare *)
          end
          else true);
      (* (4) F-04 hazard regression: the hazard is checker-clean yet its
             barrier traces DIFFER across threads — confirming the
             counterexample survives in the operational semantics. *)
      Test.make
        ~name:"sem:f04_hazard_counterexample"
        ~count:1
        (Gen.return ())
        (fun () ->
          (* checker is blind: Converged-mode env-checker reports no error *)
          let checker_blind = check_env Converged [] hazard = [] in
          let ex = to_extracted hazard in
          (* thread 0: vary_val 0 = 0 ⇒ EVary = 0 ⇒ e_else = ELit, falls
             through to the barrier ⇒ trace [EvBarrier].
             thread 1: vary_val 1 = 1 ⇒ nonzero ⇒ e_then = EReturn ELit,
             early-returns before the barrier ⇒ empty barrier trace.
             (eval_concrete uses vary_val = identity.) *)
          match (M.eval_concrete 6 0 [] ex, M.eval_concrete 6 1 [] ex) with
          | Some (_, tr0), Some (_, tr1) ->
              checker_blind && erase_warp tr0 <> erase_warp tr1
          | _ -> false);
    ]
  in
  exit (QCheck_base_runner.run_tests ~verbose:true tests)
