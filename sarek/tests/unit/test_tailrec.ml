[@@@warning "-32-34"]

(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Unit tests for Sarek_tailrec
 *
 * Tests tail recursion detection and elimination.
 ******************************************************************************)

(* Force stdlib initialization *)
let () = Sarek_stdlib.force_init ()

open Sarek_ast
open Sarek_types
open Sarek_typed_ast
open Sarek_tailrec

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

(* ===== Test: count_recursive_calls ===== *)

let test_count_no_recursion () =
  (* Simple expression with no recursion: 1 + 2 *)
  let expr = te_binop Add (te_int32 1l) (te_int32 2l) in
  let count = count_recursive_calls "f" expr in
  Alcotest.(check int) "no recursive calls" 0 count

let test_count_one_call () =
  (* Expression: f(1) - single recursive call *)
  let fn = te_var "f" 0 (TFun ([TPrim TInt32], TPrim TInt32)) in
  let expr = te_app fn [te_int32 1l] (TPrim TInt32) in
  let count = count_recursive_calls "f" expr in
  Alcotest.(check int) "one recursive call" 1 count

let test_count_nested_calls () =
  (* Expression: f(f(1)) - two recursive calls *)
  let fn = te_var "f" 0 (TFun ([TPrim TInt32], TPrim TInt32)) in
  let inner = te_app fn [te_int32 1l] (TPrim TInt32) in
  let expr = te_app fn [inner] (TPrim TInt32) in
  let count = count_recursive_calls "f" expr in
  Alcotest.(check int) "two recursive calls" 2 count

let test_count_in_if () =
  (* Expression: if x > 0 then f(x-1) else 0 *)
  let fn = te_var "f" 0 (TFun ([TPrim TInt32], TPrim TInt32)) in
  let x = te_var "x" 1 (TPrim TInt32) in
  let cond = te_binop Gt x (te_int32 0l) in
  let then_ = te_app fn [te_binop Sub x (te_int32 1l)] (TPrim TInt32) in
  let else_ = te_int32 0l in
  let expr = te_if cond then_ (Some else_) (TPrim TInt32) in
  let count = count_recursive_calls "f" expr in
  Alcotest.(check int) "one recursive call in if" 1 count

(* ===== Test: is_tail_recursive ===== *)

let test_tail_simple () =
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

let test_non_tail_after_call () =
  (* 1 + f(x-1) - NOT tail recursive (operation after call) *)
  let fn = te_var "f" 0 (TFun ([TPrim TInt32], TPrim TInt32)) in
  let x = te_var "x" 1 (TPrim TInt32) in
  let call = te_app fn [te_binop Sub x (te_int32 1l)] (TPrim TInt32) in
  let expr = te_binop Add (te_int32 1l) call in
  let is_tail = is_tail_recursive "f" expr in
  Alcotest.(check bool) "not tail - operation after call" false is_tail

let test_non_tail_nested () =
  (* f(f(x)) - NOT tail recursive (recursive call as argument) *)
  let fn = te_var "f" 0 (TFun ([TPrim TInt32], TPrim TInt32)) in
  let x = te_var "x" 1 (TPrim TInt32) in
  let inner = te_app fn [x] (TPrim TInt32) in
  let expr = te_app fn [inner] (TPrim TInt32) in
  let is_tail = is_tail_recursive "f" expr in
  Alcotest.(check bool) "not tail - nested calls" false is_tail

(* ===== Test: analyze_recursion ===== *)

let test_analyze_non_recursive () =
  let expr = te_binop Add (te_int32 1l) (te_int32 2l) in
  let info = analyze_recursion "f" expr in
  Alcotest.(check int) "call count" 0 info.ri_call_count ;
  Alcotest.(check bool) "not tail (no calls)" false info.ri_is_tail ;
  Alcotest.(check bool) "not in loop" false info.ri_in_loop

let test_analyze_tail_recursive () =
  let fn = te_var "f" 0 (TFun ([TPrim TInt32], TPrim TInt32)) in
  let x = te_var "x" 1 (TPrim TInt32) in
  let cond = te_binop Le x (te_int32 0l) in
  let then_ = te_int32 1l in
  let else_ = te_app fn [te_binop Sub x (te_int32 1l)] (TPrim TInt32) in
  let expr = te_if cond then_ (Some else_) (TPrim TInt32) in
  let info = analyze_recursion "f" expr in
  Alcotest.(check int) "call count" 1 info.ri_call_count ;
  Alcotest.(check bool) "is tail" true info.ri_is_tail ;
  Alcotest.(check bool) "not in loop" false info.ri_in_loop

(* ===== Test: tail call in sequence (last position) ===== *)

let test_tail_in_sequence () =
  (* let _ = side_effect; f(x-1) - tail call at end of sequence *)
  let fn = te_var "f" 0 (TFun ([TPrim TInt32], TPrim TInt32)) in
  let x = te_var "x" 1 (TPrim TInt32) in
  let side_effect = te_unit in
  let call = te_app fn [te_binop Sub x (te_int32 1l)] (TPrim TInt32) in
  let expr = te_seq [side_effect; call] in
  let is_tail = is_tail_recursive "f" expr in
  Alcotest.(check bool) "tail call at end of sequence" true is_tail

let test_non_tail_in_sequence_middle () =
  (* f(x); 0 - NOT tail (call in middle of sequence) *)
  let fn = te_var "f" 0 (TFun ([TPrim TInt32], TPrim TInt32)) in
  let x = te_var "x" 1 (TPrim TInt32) in
  let call = te_app fn [x] (TPrim TInt32) in
  let expr = te_seq [call; te_int32 0l] in
  let is_tail = is_tail_recursive "f" expr in
  Alcotest.(check bool) "not tail - call in middle of sequence" false is_tail

(* ===== Test: multiple parameters ===== *)

let test_tail_multiple_params () =
  (* f(x-1, y+1, z) - tail call with multiple params *)
  let fn =
    te_var
      "f"
      0
      (TFun ([TPrim TInt32; TPrim TInt32; TPrim TInt32], TPrim TInt32))
  in
  let x = te_var "x" 1 (TPrim TInt32) in
  let y = te_var "y" 2 (TPrim TInt32) in
  let z = te_var "z" 3 (TPrim TInt32) in
  let cond = te_binop Le x (te_int32 0l) in
  let then_ = z in
  let else_ =
    te_app
      fn
      [te_binop Sub x (te_int32 1l); te_binop Add y (te_int32 1l); z]
      (TPrim TInt32)
  in
  let expr = te_if cond then_ (Some else_) (TPrim TInt32) in
  let is_tail = is_tail_recursive "f" expr in
  Alcotest.(check bool) "tail with multiple params" true is_tail

(* ===== Test: different function names ===== *)

let test_different_function_name () =
  (* Call to g is NOT a recursive call for f *)
  let g = te_var "g" 0 (TFun ([TPrim TInt32], TPrim TInt32)) in
  let x = te_var "x" 1 (TPrim TInt32) in
  let expr = te_app g [x] (TPrim TInt32) in
  let count = count_recursive_calls "f" expr in
  Alcotest.(check int) "no calls to f" 0 count

(* ===== Test: tail calls in both if branches ===== *)

let test_tail_in_both_branches () =
  (* if cond then f(x) else f(y) - tail in both branches *)
  let fn = te_var "f" 0 (TFun ([TPrim TInt32], TPrim TInt32)) in
  let x = te_var "x" 1 (TPrim TInt32) in
  let y = te_var "y" 2 (TPrim TInt32) in
  let cond = te_binop Gt x (te_int32 0l) in
  let then_ = te_app fn [x] (TPrim TInt32) in
  let else_ = te_app fn [y] (TPrim TInt32) in
  let expr = te_if cond then_ (Some else_) (TPrim TInt32) in
  let is_tail = is_tail_recursive "f" expr in
  let count = count_recursive_calls "f" expr in
  Alcotest.(check bool) "tail in both branches" true is_tail ;
  Alcotest.(check int) "two calls total" 2 count

(* ===== Test: accumulator pattern (factorial) ===== *)

let test_accumulator_pattern () =
  (* Classic factorial: if n <= 0 then acc else f(n-1, acc*n) *)
  let fn = te_var "f" 0 (TFun ([TPrim TInt32; TPrim TInt32], TPrim TInt32)) in
  let n = te_var "n" 1 (TPrim TInt32) in
  let acc = te_var "acc" 2 (TPrim TInt32) in
  let cond = te_binop Le n (te_int32 0l) in
  let then_ = acc in
  let else_ =
    te_app fn [te_binop Sub n (te_int32 1l); te_binop Mul acc n] (TPrim TInt32)
  in
  let expr = te_if cond then_ (Some else_) (TPrim TInt32) in
  let info = analyze_recursion "f" expr in
  Alcotest.(check bool) "factorial is tail recursive" true info.ri_is_tail ;
  Alcotest.(check int) "one recursive call" 1 info.ri_call_count

(* ===== Test suite ===== *)

let count_tests =
  [
    ("no recursion", `Quick, test_count_no_recursion);
    ("one call", `Quick, test_count_one_call);
    ("nested calls", `Quick, test_count_nested_calls);
    ("call in if", `Quick, test_count_in_if);
    ("different function name", `Quick, test_different_function_name);
  ]

let tail_tests =
  [
    ("simple tail", `Quick, test_tail_simple);
    ("non-tail after call", `Quick, test_non_tail_after_call);
    ("non-tail nested", `Quick, test_non_tail_nested);
    ("tail in sequence", `Quick, test_tail_in_sequence);
    ("non-tail in sequence middle", `Quick, test_non_tail_in_sequence_middle);
    ("tail multiple params", `Quick, test_tail_multiple_params);
    ("tail in both branches", `Quick, test_tail_in_both_branches);
  ]

let analyze_tests =
  [
    ("non-recursive", `Quick, test_analyze_non_recursive);
    ("tail-recursive", `Quick, test_analyze_tail_recursive);
    ("accumulator pattern", `Quick, test_accumulator_pattern);
  ]

let () =
  Alcotest.run
    "Sarek_tailrec"
    [
      ("count_recursive_calls", count_tests);
      ("is_tail_recursive", tail_tests);
      ("analyze_recursion", analyze_tests);
    ]
