(******************************************************************************)
(* test_mutation.ml
 *
 * Mutation-testing harness: proves the QCheck semantics properties are
 * SENSITIVE to evaluator mutations and therefore non-tautological.
 *
 * Two mutants, each caught by a distinct property:
 *
 *   M1 trace-erasure  : strips all trace events.
 *   Caught by sem:f04_hazard_counterexample — the hazard property requires
 *   thread-0 and thread-1 traces to DIFFER; M1 makes both [] so they match.
 *
 *   M2 barrier-inject : appends EvBarrier to every trace.
 *   Caught by sem:barrier_free_silent — the silence property requires
 *   barrier-free programs to produce no EvBarrier; M2 injects one always.
 *
 * Test exits 0 iff both mutants are caught. Exits 1 if any mutant slips
 * through undetected (the properties would then be vacuous for that mutation).
 ******************************************************************************)

open QCheck2
module M = Convergence_model.ConvergenceModel

(* ── Minimal expr mirror (must match test_convergence_semantics.ml) ──────── *)

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
  | ESuperstep _ -> false
  | EApp args -> List.for_all superstep_free args
  | EReturn e -> superstep_free e

let erase_warp tr =
  List.filter (function M.EvBarrier -> true | M.EvWarp -> false) tr

(* F-04 hazard: ESeq [EIf EVary (EReturn ELit) ELit; EBarrier] *)
let hazard = to_extracted (ESeq [EIf (EVary, EReturn ELit, ELit); EBarrier])

(* ── Generators ────────────────────────────────────────────────────────── *)

let gen_tid = Gen.int_range 0 7

let gen_expr =
  Gen.sized_size (Gen.int_range 0 4)
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
            Gen.map (fun e -> EUnop e) sub;
            Gen.map (fun (a, b) -> EBinop (a, b)) sub2;
            Gen.map (fun (c, t, el) -> EIf (c, t, el)) sub3;
            Gen.map (fun es -> ESeq es) sublist;
            Gen.map (fun args -> EApp args) sublist;
            Gen.map (fun e -> EReturn e) sub;
          ])

(* ── Mutant evaluators ──────────────────────────────────────────────────── *)

(* M1: strip all trace events (barrier-blind evaluator) *)
let eval_m1 fuel t rho e =
  match M.eval_concrete fuel t rho e with
  | None -> None
  | Some (o, _tr) -> Some (o, [])

(* M2: inject EvBarrier into every completed trace *)
let eval_m2 fuel t rho e =
  match M.eval_concrete fuel t rho e with
  | None -> None
  | Some (o, tr) -> Some (o, tr @ [M.EvBarrier])

(* ── Mutation-sensitive test builders ──────────────────────────────────── *)

(* Rebuild the f04_hazard property with a custom evaluator. *)
let f04_with eval_fn =
  Test.make
    ~name:"mut:f04_hazard_counterexample_m1"
    ~count:1
    (Gen.return ())
    (fun () ->
      match (eval_fn 10 0 [] hazard, eval_fn 10 1 [] hazard) with
      | Some (_, tr0), Some (_, tr1) -> erase_warp tr0 <> erase_warp tr1
      | _ -> false)

(* Rebuild the barrier_free_silent property with a custom evaluator. *)
let barrier_silent_with eval_fn =
  Test.make
    ~name:"mut:barrier_free_silent_m2"
    ~count:3000
    (Gen.pair gen_tid gen_expr)
    (fun (t, e) ->
      if barrier_free e && superstep_free e then
        match eval_fn 60 t [] (to_extracted e) with
        | None -> true
        | Some (_, tr) ->
            not (List.exists (function M.EvBarrier -> true | _ -> false) tr)
      else true)

(* ── Assertion harness ──────────────────────────────────────────────────── *)

let assert_mutation_caught label test =
  let caught = ref false in
  (try QCheck2.Test.check_exn ~rand:(Random.State.make [|12345|]) test
   with _ -> caught := true) ;
  if not !caught then begin
    Printf.eprintf "MUTATION NOT CAUGHT: %s\n%!" label ;
    exit 1
  end
  else Printf.printf "MUTATION CAUGHT: %s\n%!" label

let () =
  (* M1 (trace erasure) must be caught by the F-04 hazard property. *)
  assert_mutation_caught
    "M1 (trace-erasure) vs sem:f04_hazard"
    (f04_with eval_m1) ;
  (* M2 (barrier injection) must be caught by the barrier-silence property. *)
  assert_mutation_caught
    "M2 (barrier-inject) vs sem:barrier_free_silent"
    (barrier_silent_with eval_m2)
