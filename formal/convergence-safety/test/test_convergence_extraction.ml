(******************************************************************************)
(* test_convergence_extraction.ml
 *
 * Extraction conformance suite.
 * Verifies that the OCaml extracted from ConvergenceSpec.v (ConvergenceModel)
 * agrees with the hand-written inline abstract model on all 6 semantic
 * functions.  This is CMBT links 2 (extraction) + 4 (test) combined:
 *
 *   Rocq spec → coqc extraction → ConvergenceModel.ml → this test
 *
 * The two models use structurally identical but distinct OCaml types.
 * to_extracted translates inline→extracted; normalize_errors does the reverse
 * for comparison.
 ******************************************************************************)

open QCheck2

(* ===== Inline abstract model (mirrors ConvergenceSpec.v) ===== *)

type expr =
  | ELit | EVary | EBarrier
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

type exec_mode = Converged | Diverged
type error = BarrierError

let rec is_varying_inline = function
  | EVary -> true | ELit | EBarrier | EVar _ -> false
  | EBinop (a, b)             -> is_varying_inline a || is_varying_inline b
  | EUnop e                   -> is_varying_inline e
  | EIf (c, t, el)            -> is_varying_inline c || is_varying_inline t || is_varying_inline el
  | EWhile (c, b)             -> is_varying_inline c || is_varying_inline b
  | EFor (lo, hi, b)          -> is_varying_inline lo || is_varying_inline hi || is_varying_inline b
  | ESeq es                   -> List.exists is_varying_inline es
  | ELet (_, v, b)            -> is_varying_inline v || is_varying_inline b
  | ESuperstep (_, body, cont) -> is_varying_inline body || is_varying_inline cont
  | EApp args                 -> List.exists is_varying_inline args

let rec barrier_free_inline = function
  | EBarrier -> false | ELit | EVary | EVar _ -> true
  | EBinop (a, b)             -> barrier_free_inline a && barrier_free_inline b
  | EUnop e                   -> barrier_free_inline e
  | EIf (c, t, el)            -> barrier_free_inline c && barrier_free_inline t && barrier_free_inline el
  | EWhile (c, b)             -> barrier_free_inline c && barrier_free_inline b
  | EFor (lo, hi, b)          -> barrier_free_inline lo && barrier_free_inline hi && barrier_free_inline b
  | ESeq es                   -> List.for_all barrier_free_inline es
  | ELet (_, v, b)            -> barrier_free_inline v && barrier_free_inline b
  | ESuperstep (dv, body, cont) ->
      dv && barrier_free_inline body && barrier_free_inline cont
  | EApp args                 -> List.for_all barrier_free_inline args

let rec has_diverging_cf_inline = function
  | EIf (c, t, el)    -> is_varying_inline c || has_diverging_cf_inline t || has_diverging_cf_inline el
  | EWhile (c, b)     -> is_varying_inline c || has_diverging_cf_inline b
  | EFor (lo, hi, b)  -> is_varying_inline lo || is_varying_inline hi || has_diverging_cf_inline b
  | EBinop (a, b)     -> has_diverging_cf_inline a || has_diverging_cf_inline b
  | EUnop e           -> has_diverging_cf_inline e
  | ESeq es           -> List.exists has_diverging_cf_inline es
  | ELet (_, v, b)    -> has_diverging_cf_inline v || has_diverging_cf_inline b
  | ESuperstep (_, body, cont) -> has_diverging_cf_inline body || has_diverging_cf_inline cont
  | EApp args         -> List.exists has_diverging_cf_inline args
  | _                 -> false

let rec check_inline m e =
  match e with
  | EBarrier -> (match m with Diverged -> [BarrierError] | Converged -> [])
  | ELit | EVary | EVar _ -> []
  | EBinop (a, b)    -> check_inline m a @ check_inline m b
  | EUnop e          -> check_inline m e
  | EIf (c, t, el)   ->
    let inner = if is_varying_inline c then Diverged else m in
    check_inline m c @ check_inline inner t @ check_inline inner el
  | EWhile (c, b)    ->
    let inner = if is_varying_inline c then Diverged else m in
    check_inline m c @ check_inline inner b
  | EFor (lo, hi, b) ->
    let inner = if is_varying_inline lo || is_varying_inline hi then Diverged else m in
    check_inline m lo @ check_inline m hi @ check_inline inner b
  | ESeq es          -> List.concat_map (check_inline m) es
  | ELet (_, v, b)   -> check_inline m v @ check_inline m b
  | ESuperstep (divergent, body, cont) ->
    let entry_errors =
      match m, divergent with
      | Diverged, false -> [BarrierError]
      | _,        _     -> []
    in
    entry_errors @ check_inline m body @ check_inline m cont
  | EApp args        -> List.concat_map (check_inline m) args

(* ===== Translation: inline types → extracted types ===== *)

module M = Convergence_model.ConvergenceModel

let rec to_extracted : expr -> M.expr = function
  | ELit              -> ELit
  | EVary             -> EVary
  | EBarrier          -> EBarrier
  | EVar x            -> EVar x
  | EBinop (a, b)     -> EBinop (to_extracted a, to_extracted b)
  | EUnop e           -> EUnop (to_extracted e)
  | EIf (c, t, el)    -> EIf (to_extracted c, to_extracted t, to_extracted el)
  | EWhile (c, b)     -> EWhile (to_extracted c, to_extracted b)
  | EFor (lo, hi, b)  -> EFor (to_extracted lo, to_extracted hi, to_extracted b)
  | ESeq es           -> ESeq (List.map to_extracted es)
  | ELet (x, v, b)    -> ELet (x, to_extracted v, to_extracted b)
  | ESuperstep (dv, body, cont) ->
      ESuperstep (dv, to_extracted body, to_extracted cont)
  | EApp args         -> EApp (List.map to_extracted args)

let to_extracted_mode : exec_mode -> M.exec_mode = function
  | Converged -> Converged
  | Diverged  -> Diverged

let normalize_errors (errs : M.error list) : error list =
  List.map (fun M.BarrierError -> BarrierError) errs

(* ===== Generator (same pattern as test_convergence_conformance) ===== *)

let gen_mode : exec_mode Gen.t = Gen.oneof_list [Converged; Diverged]

let gen_var_id : int Gen.t = Gen.int_range 0 3

let gen_expr : expr Gen.t =
  Gen.sized_size (Gen.int_range 0 6) @@ Gen.fix (fun self n ->
    if n = 0 then Gen.oneof_list [ELit; EVary; EBarrier]
    else
      let sub     = self (n / 2) in
      let sub2    = Gen.pair sub sub in
      let sub3    = Gen.triple sub sub sub in
      let sublist = Gen.list_size (Gen.int_range 0 3) sub in
      Gen.oneof [
        Gen.return ELit;
        Gen.return EVary;
        Gen.return EBarrier;
        Gen.map (fun x            -> EVar x)               gen_var_id;
        Gen.map (fun (a, b)       -> EBinop (a, b))        sub2;
        Gen.map (fun e            -> EUnop e)               sub;
        Gen.map (fun (c, t, el)   -> EIf (c, t, el))       sub3;
        Gen.map (fun (c, b)       -> EWhile (c, b))         sub2;
        Gen.map (fun (lo, hi, b)  -> EFor (lo, hi, b))      sub3;
        Gen.map (fun es           -> ESeq es)               sublist;
        Gen.map (fun (x, (v, b))  -> ELet (x, v, b))
                (Gen.pair gen_var_id sub2);
        Gen.map (fun (dv, (body, cont)) -> ESuperstep (dv, body, cont))
                (Gen.pair Gen.bool sub2);
        Gen.map (fun args         -> EApp args)             sublist;
      ])

(* ===== Properties: inline vs extracted must agree ===== *)

let () =
  let tests = [

    Test.make ~name:"extr:is_varying_agrees" ~count:2000 gen_expr (fun e ->
      let expected = is_varying_inline e in
      let got = M.is_varying (to_extracted e) in
      expected = got);

    Test.make ~name:"extr:barrier_free_agrees" ~count:2000 gen_expr (fun e ->
      let expected = barrier_free_inline e in
      let got = M.barrier_free (to_extracted e) in
      expected = got);

    Test.make ~name:"extr:has_diverging_cf_agrees" ~count:2000 gen_expr (fun e ->
      let expected = has_diverging_cf_inline e in
      let got = M.has_diverging_cf (to_extracted e) in
      expected = got);

    Test.make ~name:"extr:check_converged_agrees" ~count:1500 gen_expr (fun e ->
      let expected = check_inline Converged e in
      let got = normalize_errors (M.check Converged (to_extracted e)) in
      expected = got);

    Test.make ~name:"extr:check_diverged_agrees" ~count:1500 gen_expr (fun e ->
      let expected = check_inline Diverged e in
      let got = normalize_errors (M.check Diverged (to_extracted e)) in
      expected = got);

    Test.make ~name:"extr:check_any_mode_agrees"
      ~count:1500 (Gen.pair gen_expr gen_mode) (fun (e, m) ->
      let expected = check_inline m e in
      let got = normalize_errors (M.check (to_extracted_mode m) (to_extracted e)) in
      expected = got);

  ] in
  let _results = QCheck_base_runner.run_tests ~verbose:true tests in
  ()
