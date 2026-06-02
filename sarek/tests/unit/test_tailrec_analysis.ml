[@@@warning "-32-34"]

(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Unit tests for Sarek_tailrec_analysis
 *
 * Tests recursion detection and analysis functions.
 ******************************************************************************)

(* Force stdlib initialization *)
let () = Sarek_stdlib.force_init ()

open Sarek_ast
open Sarek_types
open Sarek_typed_ast
open Sarek_tailrec_analysis

(* Import for_dir type without shadowing other types *)
type for_dir = Sarek_ir_ppx.for_dir = Upto | Downto

let dummy_loc =
  {
    loc_file = "test";
    loc_line = 1;
    loc_col = 0;
    loc_end_line = 1;
    loc_end_col = 0;
  }

(* Helper to create typed expressions *)
let mk_texpr te ty = {te; ty; te_loc = dummy_loc}

let te_int i = mk_texpr (TEInt i) (TPrim TInt32)

let te_int32 i = mk_texpr (TEInt32 i) (TPrim TInt32)

let te_bool b = mk_texpr (TEBool b) t_bool

let te_var name id ty = mk_texpr (TEVar (name, id)) ty

let te_unit = mk_texpr TEUnit t_unit

let te_binop op a b =
  let ty =
    match op with Lt | Le | Gt | Ge | Eq | Ne | And | Or -> t_bool | _ -> a.ty
  in
  mk_texpr (TEBinop (op, a, b)) ty

let te_if c t e ty = mk_texpr (TEIf (c, t, e)) ty

let te_app fn args ty = mk_texpr (TEApp (fn, args)) ty

let te_seq exprs =
  let ty = match List.rev exprs with [] -> t_unit | last :: _ -> last.ty in
  mk_texpr (TESeq exprs) ty

let te_while c b = mk_texpr (TEWhile (c, b)) t_unit

let te_for v id lo hi dir body =
  mk_texpr (TEFor (v, id, lo, hi, dir, body)) t_unit

(* ===== Test: is_self_call ===== *)

let test_is_self_call_positive () =
  let fn = te_var "factorial" 0 (TFun ([TPrim TInt32], TPrim TInt32)) in
  let result = is_self_call "factorial" fn in
  Alcotest.(check bool) "detects self call" true result

let test_is_self_call_negative () =
  let fn = te_var "helper" 0 (TFun ([TPrim TInt32], TPrim TInt32)) in
  let result = is_self_call "factorial" fn in
  Alcotest.(check bool) "rejects non-self call" false result

let test_is_self_call_non_var () =
  let expr = te_int32 42l in
  let result = is_self_call "factorial" expr in
  Alcotest.(check bool) "rejects non-variable" false result

(* ===== Test: is_recursive_call ===== *)

let test_is_recursive_call_positive () =
  let fn = te_var "f" 0 (TFun ([TPrim TInt32], TPrim TInt32)) in
  let expr = te_app fn [te_int32 1l] (TPrim TInt32) in
  let result = is_recursive_call "f" expr in
  Alcotest.(check bool) "detects recursive call" true result

let test_is_recursive_call_negative () =
  let fn = te_var "g" 0 (TFun ([TPrim TInt32], TPrim TInt32)) in
  let expr = te_app fn [te_int32 1l] (TPrim TInt32) in
  let result = is_recursive_call "f" expr in
  Alcotest.(check bool) "rejects non-recursive call" false result

(* ===== Test: count_recursive_calls ===== *)

let test_count_no_recursion () =
  (* Simple expression: 1 + 2 *)
  let expr = te_binop Add (te_int32 1l) (te_int32 2l) in
  let count = count_recursive_calls "f" expr in
  Alcotest.(check int) "no recursive calls" 0 count

let test_count_one_call () =
  (* Expression: f(1) *)
  let fn = te_var "f" 0 (TFun ([TPrim TInt32], TPrim TInt32)) in
  let expr = te_app fn [te_int32 1l] (TPrim TInt32) in
  let count = count_recursive_calls "f" expr in
  Alcotest.(check int) "one recursive call" 1 count

let test_count_nested_calls () =
  (* Expression: f(f(1)) *)
  let fn = te_var "f" 0 (TFun ([TPrim TInt32], TPrim TInt32)) in
  let inner = te_app fn [te_int32 1l] (TPrim TInt32) in
  let expr = te_app fn [inner] (TPrim TInt32) in
  let count = count_recursive_calls "f" expr in
  Alcotest.(check int) "two recursive calls" 2 count

let test_count_in_if_branch () =
  (* if x > 0 then f(x-1) else 0 *)
  let fn = te_var "f" 0 (TFun ([TPrim TInt32], TPrim TInt32)) in
  let x = te_var "x" 1 (TPrim TInt32) in
  let cond = te_binop Gt x (te_int32 0l) in
  let then_ = te_app fn [te_binop Sub x (te_int32 1l)] (TPrim TInt32) in
  let else_ = te_int32 0l in
  let expr = te_if cond then_ (Some else_) (TPrim TInt32) in
  let count = count_recursive_calls "f" expr in
  Alcotest.(check int) "one recursive call in if" 1 count

let test_count_both_if_branches () =
  (* if x > 0 then f(x-1) else f(x+1) *)
  let fn = te_var "f" 0 (TFun ([TPrim TInt32], TPrim TInt32)) in
  let x = te_var "x" 1 (TPrim TInt32) in
  let cond = te_binop Gt x (te_int32 0l) in
  let then_ = te_app fn [te_binop Sub x (te_int32 1l)] (TPrim TInt32) in
  let else_ = te_app fn [te_binop Add x (te_int32 1l)] (TPrim TInt32) in
  let expr = te_if cond then_ (Some else_) (TPrim TInt32) in
  let count = count_recursive_calls "f" expr in
  Alcotest.(check int) "two recursive calls in if" 2 count

let test_count_in_sequence () =
  (* f(1); f(2); f(3) *)
  let fn = te_var "f" 0 (TFun ([TPrim TInt32], t_unit)) in
  let call1 = te_app fn [te_int32 1l] t_unit in
  let call2 = te_app fn [te_int32 2l] t_unit in
  let call3 = te_app fn [te_int32 3l] t_unit in
  let expr = te_seq [call1; call2; call3] in
  let count = count_recursive_calls "f" expr in
  Alcotest.(check int) "three recursive calls in sequence" 3 count

(* ===== Test: is_tail_recursive ===== *)

let test_tail_simple_base_case () =
  (* Simple base case: just return value (no recursion) *)
  let x = te_var "x" 1 (TPrim TInt32) in
  let is_tail = is_tail_recursive "f" x in
  Alcotest.(check bool) "base case is tail recursive" true is_tail

let test_tail_simple_recursive () =
  (* if x <= 0 then y else f(x-1, y+x) - tail recursive *)
  let fn = te_var "f" 0 (TFun ([TPrim TInt32; TPrim TInt32], TPrim TInt32)) in
  let x = te_var "x" 1 (TPrim TInt32) in
  let y = te_var "y" 2 (TPrim TInt32) in
  let cond = te_binop Le x (te_int32 0l) in
  let then_ = y in
  let else_ =
    te_app fn [te_binop Sub x (te_int32 1l); te_binop Add y x] (TPrim TInt32)
  in
  let expr = te_if cond then_ (Some else_) (TPrim TInt32) in
  let is_tail = is_tail_recursive "f" expr in
  Alcotest.(check bool) "simple tail recursion" true is_tail

let test_non_tail_in_binop () =
  (* f(x) + 1 - NOT tail recursive (recursive call not in tail position) *)
  let fn = te_var "f" 0 (TFun ([TPrim TInt32], TPrim TInt32)) in
  let x = te_var "x" 1 (TPrim TInt32) in
  let call = te_app fn [x] (TPrim TInt32) in
  let expr = te_binop Add call (te_int32 1l) in
  let is_tail = is_tail_recursive "f" expr in
  Alcotest.(check bool) "recursive call in binop is not tail" false is_tail

let test_non_tail_nested_call () =
  (* f(f(x)) - NOT tail recursive (inner call not in tail position) *)
  let fn = te_var "f" 0 (TFun ([TPrim TInt32], TPrim TInt32)) in
  let x = te_var "x" 1 (TPrim TInt32) in
  let inner = te_app fn [x] (TPrim TInt32) in
  let expr = te_app fn [inner] (TPrim TInt32) in
  let is_tail = is_tail_recursive "f" expr in
  Alcotest.(check bool) "nested recursive call is not tail" false is_tail

let test_tail_in_sequence_last () =
  (* let x = 1 in f(x) - tail recursive (call is last in sequence) *)
  let fn = te_var "f" 0 (TFun ([TPrim TInt32], TPrim TInt32)) in
  let call = te_app fn [te_int32 1l] (TPrim TInt32) in
  let expr = te_seq [te_unit; call] in
  let is_tail = is_tail_recursive "f" expr in
  Alcotest.(check bool)
    "recursive call as last in sequence is tail"
    true
    is_tail

let test_non_tail_in_sequence_middle () =
  (* f(x); return 1 - NOT tail recursive (call not in tail position) *)
  let fn = te_var "f" 0 (TFun ([TPrim TInt32], t_unit)) in
  let call = te_app fn [te_int32 1l] t_unit in
  let expr = te_seq [call; te_int32 1l] in
  let is_tail = is_tail_recursive "f" expr in
  Alcotest.(check bool)
    "recursive call not last in sequence is not tail"
    false
    is_tail

(* ===== Test: has_recursion_in_loops ===== *)

let test_no_recursion_in_loops () =
  (* for i = 0 to 10 do x := x + 1 done *)
  let body = te_unit in
  let expr = te_for "i" 1 (te_int32 0l) (te_int32 10l) Upto body in
  let result = has_recursion_in_loops "f" expr in
  Alcotest.(check bool) "no recursion in loop" false result

let test_recursion_in_for_loop () =
  (* for i = 0 to 10 do f(i) done *)
  let fn = te_var "f" 0 (TFun ([TPrim TInt32], t_unit)) in
  let i = te_var "i" 1 (TPrim TInt32) in
  let call = te_app fn [i] t_unit in
  let expr = te_for "i" 1 (te_int32 0l) (te_int32 10l) Upto call in
  let result = has_recursion_in_loops "f" expr in
  Alcotest.(check bool) "recursion in for loop" true result

let test_recursion_in_while_loop () =
  (* while x > 0 do f(x); x := x - 1 done *)
  let fn = te_var "f" 0 (TFun ([TPrim TInt32], t_unit)) in
  let x = te_var "x" 1 (TPrim TInt32) in
  let cond = te_binop Gt x (te_int32 0l) in
  let body = te_app fn [x] t_unit in
  let expr = te_while cond body in
  let result = has_recursion_in_loops "f" expr in
  Alcotest.(check bool) "recursion in while loop" true result

(* ===== Test: analyze_recursion ===== *)

let test_analyze_non_recursive () =
  (* Simple non-recursive function *)
  let expr = te_int32 42l in
  let info = analyze_recursion "f" expr in
  Alcotest.(check int) "call count is 0" 0 info.ri_call_count ;
  Alcotest.(check bool) "is not tail" false info.ri_is_tail ;
  Alcotest.(check bool) "no loops" false info.ri_in_loop ;
  Alcotest.(check string) "name matches" "f" info.ri_name

let test_analyze_tail_recursive () =
  (* if x <= 0 then 0 else f(x-1) *)
  let fn = te_var "f" 0 (TFun ([TPrim TInt32], TPrim TInt32)) in
  let x = te_var "x" 1 (TPrim TInt32) in
  let cond = te_binop Le x (te_int32 0l) in
  let then_ = te_int32 0l in
  let else_ = te_app fn [te_binop Sub x (te_int32 1l)] (TPrim TInt32) in
  let expr = te_if cond then_ (Some else_) (TPrim TInt32) in
  let info = analyze_recursion "f" expr in
  Alcotest.(check int) "call count is 1" 1 info.ri_call_count ;
  Alcotest.(check bool) "is tail recursive" true info.ri_is_tail ;
  Alcotest.(check bool) "no loops" false info.ri_in_loop

let test_analyze_non_tail_recursive () =
  (* f(x) + 1 *)
  let fn = te_var "f" 0 (TFun ([TPrim TInt32], TPrim TInt32)) in
  let x = te_var "x" 1 (TPrim TInt32) in
  let call = te_app fn [x] (TPrim TInt32) in
  let expr = te_binop Add call (te_int32 1l) in
  let info = analyze_recursion "f" expr in
  Alcotest.(check int) "call count is 1" 1 info.ri_call_count ;
  Alcotest.(check bool) "is not tail recursive" false info.ri_is_tail ;
  Alcotest.(check bool) "no loops" false info.ri_in_loop

let test_analyze_recursion_in_loop () =
  (* for i = 0 to 10 do f(i) done *)
  let fn = te_var "f" 0 (TFun ([TPrim TInt32], t_unit)) in
  let i = te_var "i" 1 (TPrim TInt32) in
  let call = te_app fn [i] t_unit in
  let expr = te_for "i" 1 (te_int32 0l) (te_int32 10l) Upto call in
  let info = analyze_recursion "f" expr in
  Alcotest.(check int) "call count is 1" 1 info.ri_call_count ;
  Alcotest.(check bool) "is not tail (in loop)" false info.ri_is_tail ;
  Alcotest.(check bool) "has recursion in loop" true info.ri_in_loop

(* ===== Test Suite ===== *)

let self_call_tests =
  [
    Alcotest.test_case "positive" `Quick test_is_self_call_positive;
    Alcotest.test_case "negative" `Quick test_is_self_call_negative;
    Alcotest.test_case "non-var" `Quick test_is_self_call_non_var;
  ]

let recursive_call_tests =
  [
    Alcotest.test_case "positive" `Quick test_is_recursive_call_positive;
    Alcotest.test_case "negative" `Quick test_is_recursive_call_negative;
  ]

let count_tests =
  [
    Alcotest.test_case "no recursion" `Quick test_count_no_recursion;
    Alcotest.test_case "one call" `Quick test_count_one_call;
    Alcotest.test_case "nested calls" `Quick test_count_nested_calls;
    Alcotest.test_case "if branch" `Quick test_count_in_if_branch;
    Alcotest.test_case "both if branches" `Quick test_count_both_if_branches;
    Alcotest.test_case "sequence" `Quick test_count_in_sequence;
  ]

let tail_tests =
  [
    Alcotest.test_case "base case" `Quick test_tail_simple_base_case;
    Alcotest.test_case "simple tail" `Quick test_tail_simple_recursive;
    Alcotest.test_case "non-tail binop" `Quick test_non_tail_in_binop;
    Alcotest.test_case "non-tail nested" `Quick test_non_tail_nested_call;
    Alcotest.test_case "tail in sequence" `Quick test_tail_in_sequence_last;
    Alcotest.test_case
      "non-tail in sequence"
      `Quick
      test_non_tail_in_sequence_middle;
  ]

let loop_tests =
  [
    Alcotest.test_case "no recursion" `Quick test_no_recursion_in_loops;
    Alcotest.test_case "for loop" `Quick test_recursion_in_for_loop;
    Alcotest.test_case "while loop" `Quick test_recursion_in_while_loop;
  ]

let analyze_tests =
  [
    Alcotest.test_case "non-recursive" `Quick test_analyze_non_recursive;
    Alcotest.test_case "tail recursive" `Quick test_analyze_tail_recursive;
    Alcotest.test_case
      "non-tail recursive"
      `Quick
      test_analyze_non_tail_recursive;
    Alcotest.test_case "recursion in loop" `Quick test_analyze_recursion_in_loop;
  ]

let () =
  Alcotest.run
    "Sarek_tailrec_analysis"
    [
      ("is_self_call", self_call_tests);
      ("is_recursive_call", recursive_call_tests);
      ("count_recursive_calls", count_tests);
      ("is_tail_recursive", tail_tests);
      ("has_recursion_in_loops", loop_tests);
      ("analyze_recursion", analyze_tests);
    ]
