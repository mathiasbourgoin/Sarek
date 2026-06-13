(******************************************************************************)
(* Live CMBT tests: call the real Sarek_convergence.check_expr.               *)
(*                                                                            *)
(* Four tiers:                                                                *)
(*  - Positive baseline: direct-intrinsic patterns the checker catches        *)
(*  - F-02 regression: let-aliased thread-varying value (now fixed)           *)
(*  - F-01 regression: superstep inside diverged outer context (now fixed)    *)
(*  - Negative baseline: patterns that must remain clean (no false positives) *)
(******************************************************************************)

open Sarek_typed_ast
open Sarek_types
open Sarek_ast
open Sarek_env

(* ── helpers ────────────────────────────────────────────────────────────── *)

let mk ?(ty = t_unit) te = {te; ty; te_loc = dummy_loc}

let mk_int () = mk ~ty:t_int (TEInt 0)

let barrier_ref = CorePrimitiveRef "block_barrier"

let thread_idx_x_ref = CorePrimitiveRef "thread_idx_x"

let mk_barrier () =
  mk
    (TEIntrinsicFun
       (barrier_ref, Some Sarek_core_primitives.ConvergencePoint, []))

let mk_thread_idx_x () = mk ~ty:t_int (TEIntrinsicConst thread_idx_x_ref)

(* ── checker helpers ─────────────────────────────────────────────────────── *)

let check e = Sarek_convergence.check_expr Sarek_convergence.init_ctx e

let has_barrier_error errors =
  List.exists
    (function Sarek_error.Barrier_in_diverged_flow _ -> true | _ -> false)
    errors

(* ── baseline: patterns the checker catches correctly ───────────────────── *)

(* if thread_idx_x > 0 then barrier() — direct intrinsic, caught *)
let test_direct_varying_if_catches () =
  let cond = mk ~ty:t_bool (TEBinop (Gt, mk_thread_idx_x (), mk_int ())) in
  let prog = mk (TEIf (cond, mk_barrier (), None)) in
  let errors = check prog in
  if not (has_barrier_error errors) then
    failwith "BASELINE FAIL: direct varying if should be caught"

(* if 0 > 0 then barrier() — constant condition, correctly clean *)
let test_constant_if_is_clean () =
  let cond = mk ~ty:t_bool (TEBinop (Gt, mk_int (), mk_int ())) in
  let prog = mk (TEIf (cond, mk_barrier (), None)) in
  let errors = check prog in
  if has_barrier_error errors then
    failwith "BASELINE FAIL: constant-condition if should be clean"

(* barrier() at top level (converged) — correctly clean *)
let test_barrier_converged_clean () =
  let errors = check (mk_barrier ()) in
  if has_barrier_error errors then
    failwith "BASELINE FAIL: top-level barrier should be clean"

(* for i = thread_idx_x to 10 do barrier() done — varying lo, caught *)
let test_for_varying_lo_catches () =
  let prog =
    mk
      (TEFor
         ( "i",
           0,
           mk_thread_idx_x (),
           mk ~ty:t_int (TEInt 10),
           Upto,
           mk_barrier () ))
  in
  let errors = check prog in
  if not (has_barrier_error errors) then
    failwith "BASELINE FAIL: for with varying lo should be caught"

(* ── negative baseline: no false positives from let-bound variance ───────── *)

(* let tid = thread_idx_x in if tid < n then work() — NO barrier, must be clean *)
let test_let_varying_no_barrier_is_clean () =
  let n = 42 in
  let te_tid = mk ~ty:t_int (TEVar ("tid", n)) in
  let cond = mk ~ty:t_bool (TEBinop (Lt, te_tid, mk_int ())) in
  let body = mk (TEIf (cond, mk TEUnit, None)) in
  let prog = mk (TELet ("tid", n, mk_thread_idx_x (), body)) in
  let errors = check prog in
  if has_barrier_error errors then
    failwith
      "FALSE-POSITIVE: let tid=tid_x; if tid < n then () — no barrier, must be \
       clean"

(* non-divergent superstep with thread-varying if inside (no barrier) — must be clean *)
let test_superstep_varying_if_no_barrier_is_clean () =
  let n = 43 in
  let te_tid = mk ~ty:t_int (TEVar ("tid", n)) in
  let cond = mk ~ty:t_bool (TEBinop (Lt, te_tid, mk_int ())) in
  let body =
    mk (TELet ("tid", n, mk_thread_idx_x (), mk (TEIf (cond, mk TEUnit, None))))
  in
  let cont = mk TEUnit in
  let prog = mk (TESuperstep ("s", false, body, cont)) in
  let errors = check prog in
  if has_barrier_error errors then
    failwith
      "FALSE-POSITIVE: superstep { let tid=tid_x in if tid < n then () } — no \
       barrier inside, implicit barrier is safe, must be clean"

(* ── F-02 regression: let-aliased thread-varying value ─────────────────── *)

(* let x = thread_idx_x in if x > 0 then barrier() — must be caught *)
let test_f02_let_alias_if () =
  let x_id = 42 in
  let te_x = mk ~ty:t_int (TEVar ("x", x_id)) in
  let cond = mk ~ty:t_bool (TEBinop (Gt, te_x, mk_int ())) in
  let body = mk (TEIf (cond, mk_barrier (), None)) in
  let prog = mk (TELet ("x", x_id, mk_thread_idx_x (), body)) in
  let errors = check prog in
  if not (has_barrier_error errors) then
    failwith
      "F-02 FAIL: let x = thread_idx_x in if x > 0 then barrier() — checker \
       missed diverged barrier (expected BarrierError)"

(* let x = thread_idx_x in while x > 0 do barrier() done — must be caught *)
let test_f02_let_alias_while () =
  let x_id = 43 in
  let te_x = mk ~ty:t_int (TEVar ("x", x_id)) in
  let cond = mk ~ty:t_bool (TEBinop (Gt, te_x, mk_int ())) in
  let prog =
    mk
      (TELet ("x", x_id, mk_thread_idx_x (), mk (TEWhile (cond, mk_barrier ()))))
  in
  let errors = check prog in
  if not (has_barrier_error errors) then
    failwith
      "F-02 FAIL: let x = thread_idx_x in while x > 0 — checker missed \
       diverged barrier"

(* double alias: let x = thread_idx_x in let y = x in if y > 0 then barrier() *)
let test_f02_double_alias () =
  let x_id = 44 and y_id = 45 in
  let te_y = mk ~ty:t_int (TEVar ("y", y_id)) in
  let cond = mk ~ty:t_bool (TEBinop (Gt, te_y, mk_int ())) in
  let inner_if = mk (TEIf (cond, mk_barrier (), None)) in
  let inner_let =
    mk (TELet ("y", y_id, mk ~ty:t_int (TEVar ("x", x_id)), inner_if))
  in
  let prog = mk (TELet ("x", x_id, mk_thread_idx_x (), inner_let)) in
  let errors = check prog in
  if not (has_barrier_error errors) then
    failwith
      "F-02 FAIL: double alias let x=tid; let y=x; if y > 0 then barrier() — \
       checker missed diverged barrier through chain"

(* ── F-01 regression: TESuperstep outer-mode reset ─────────────────────── *)

(* if thread_idx_x > 0 then (superstep { ... } done) — implicit barrier under diverged outer *)
let test_f01_superstep_in_diverged_if () =
  let cond = mk ~ty:t_bool (TEBinop (Gt, mk_thread_idx_x (), mk_int ())) in
  let superstep_body = mk TEUnit in
  let cont = mk TEUnit in
  let superstep = mk (TESuperstep ("s", false, superstep_body, cont)) in
  let prog = mk (TEIf (cond, superstep, None)) in
  let errors = check prog in
  if not (has_barrier_error errors) then
    failwith
      "F-01 FAIL: superstep inside diverged if — implicit end-of-superstep \
       barrier not flagged"

(* ── run all ────────────────────────────────────────────────────────────── *)

let () =
  let tests =
    [
      ("baseline: direct varying if", test_direct_varying_if_catches);
      ("baseline: constant if is clean", test_constant_if_is_clean);
      ("baseline: barrier converged clean", test_barrier_converged_clean);
      ("baseline: for varying lo", test_for_varying_lo_catches);
      ("no-fp: let varying no barrier", test_let_varying_no_barrier_is_clean);
      ( "no-fp: superstep varying if no barrier",
        test_superstep_varying_if_no_barrier_is_clean );
      ("F-02: let alias if", test_f02_let_alias_if);
      ("F-02: let alias while", test_f02_let_alias_while);
      ("F-02: double alias", test_f02_double_alias);
      ("F-01: superstep in diverged if", test_f01_superstep_in_diverged_if);
    ]
  in
  let pass = ref 0 and fail = ref 0 in
  List.iter
    (fun (name, f) ->
      match f () with
      | () ->
          incr pass ;
          Printf.printf "  [PASS] %s\n%!" name
      | exception Failure msg ->
          incr fail ;
          Printf.printf "  [FAIL] %s: %s\n%!" name msg)
    tests ;
  Printf.printf "%d/%d PASS\n%!" !pass (List.length tests) ;
  if !fail > 0 then exit 1
