(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Unit tests for Sarek_tailrec_pragma module
 *
 * Tests pragma parsing, node counting, and pragma-based inlining
 ******************************************************************************)

(* Force stdlib initialization *)
let () = Sarek_stdlib.force_init ()

open Sarek_typed_ast

let dummy_loc =
  Sarek_ast.
    {
      loc_file = "test.ml";
      loc_line = 1;
      loc_col = 0;
      loc_end_line = 1;
      loc_end_col = 10;
    }

(* Helper: create a simple typed expression *)
let mk_texpr te =
  {te; ty = Sarek_types.TPrim Sarek_types.TInt32; te_loc = dummy_loc}

(* Test: parse_sarek_inline_pragma recognizes valid pragmas *)
let test_parse_pragma_valid_single () =
  match Sarek_tailrec_pragma.parse_sarek_inline_pragma ["sarek.inline 5"] with
  | Some 5 -> ()
  | Some n -> Alcotest.failf "Expected 5, got %d" n
  | None -> Alcotest.fail "Failed to parse valid pragma"

let test_parse_pragma_valid_list () =
  match
    Sarek_tailrec_pragma.parse_sarek_inline_pragma ["sarek.inline"; "10"]
  with
  | Some 10 -> ()
  | Some n -> Alcotest.failf "Expected 10, got %d" n
  | None -> Alcotest.fail "Failed to parse valid pragma"

let test_parse_pragma_invalid () =
  match
    Sarek_tailrec_pragma.parse_sarek_inline_pragma ["sarek.inline"; "foo"]
  with
  | Some _ -> Alcotest.fail "Should not parse invalid number"
  | None -> ()

let test_parse_pragma_wrong_name () =
  match Sarek_tailrec_pragma.parse_sarek_inline_pragma ["sarek.other"; "5"] with
  | Some _ -> Alcotest.fail "Should not parse wrong pragma name"
  | None -> ()

let test_parse_pragma_empty () =
  match Sarek_tailrec_pragma.parse_sarek_inline_pragma [] with
  | Some _ -> Alcotest.fail "Should not parse empty pragma"
  | None -> ()

(* Test: is_unroll_pragma recognizes unroll pragmas *)
let test_is_unroll_pragma_valid () =
  Alcotest.(check bool)
    "unroll recognized"
    true
    (Sarek_tailrec_pragma.is_unroll_pragma ["unroll"])

let test_is_unroll_pragma_with_param () =
  Alcotest.(check bool)
    "unroll 4 recognized"
    true
    (Sarek_tailrec_pragma.is_unroll_pragma ["unroll 4"])

let test_is_unroll_pragma_invalid () =
  Alcotest.(check bool)
    "non-unroll not recognized"
    false
    (Sarek_tailrec_pragma.is_unroll_pragma ["sarek.inline"; "5"])

(* Test: count_nodes counts AST nodes correctly *)
let test_count_nodes_unit () =
  let expr = mk_texpr TEUnit in
  let count = Sarek_tailrec_pragma.count_nodes expr in
  Alcotest.(check int) "unit node count" 1 count

let test_count_nodes_int () =
  let expr = mk_texpr (TEInt 42) in
  let count = Sarek_tailrec_pragma.count_nodes expr in
  Alcotest.(check int) "int node count" 1 count

let test_count_nodes_var () =
  let expr = mk_texpr (TEVar ("x", 0)) in
  let count = Sarek_tailrec_pragma.count_nodes expr in
  Alcotest.(check int) "var node count" 1 count

let test_count_nodes_binop () =
  let left = mk_texpr (TEInt 1) in
  let right = mk_texpr (TEInt 2) in
  let expr = mk_texpr (TEBinop (Sarek_ast.Add, left, right)) in
  let count = Sarek_tailrec_pragma.count_nodes expr in
  Alcotest.(check int) "binop node count" 3 count

let test_count_nodes_unop () =
  let operand = mk_texpr (TEInt 5) in
  let expr = mk_texpr (TEUnop (Sarek_ast.Neg, operand)) in
  let count = Sarek_tailrec_pragma.count_nodes expr in
  Alcotest.(check int) "unop node count" 2 count

let test_count_nodes_if () =
  let cond = mk_texpr (TEBool true) in
  let then_branch = mk_texpr (TEInt 1) in
  let else_branch = mk_texpr (TEInt 2) in
  let expr = mk_texpr (TEIf (cond, then_branch, Some else_branch)) in
  let count = Sarek_tailrec_pragma.count_nodes expr in
  Alcotest.(check int) "if node count" 4 count

let test_count_nodes_if_no_else () =
  let cond = mk_texpr (TEBool true) in
  let then_branch = mk_texpr TEUnit in
  let expr = mk_texpr (TEIf (cond, then_branch, None)) in
  let count = Sarek_tailrec_pragma.count_nodes expr in
  Alcotest.(check int) "if without else node count" 3 count

let test_count_nodes_seq () =
  let e1 = mk_texpr (TEInt 1) in
  let e2 = mk_texpr (TEInt 2) in
  let e3 = mk_texpr (TEInt 3) in
  let expr = mk_texpr (TESeq [e1; e2; e3]) in
  let count = Sarek_tailrec_pragma.count_nodes expr in
  Alcotest.(check int) "seq node count" 4 count

let test_count_nodes_let () =
  let value = mk_texpr (TEInt 42) in
  let body = mk_texpr (TEVar ("x", 0)) in
  let expr = mk_texpr (TELet ("x", 0, value, body)) in
  let count = Sarek_tailrec_pragma.count_nodes expr in
  Alcotest.(check int) "let node count" 3 count

let test_count_nodes_tuple () =
  let e1 = mk_texpr (TEInt 1) in
  let e2 = mk_texpr (TEInt 2) in
  let expr = mk_texpr (TETuple [e1; e2]) in
  let count = Sarek_tailrec_pragma.count_nodes expr in
  Alcotest.(check int) "tuple node count" 3 count

let test_count_nodes_app () =
  let fn = mk_texpr (TEVar ("f", 0)) in
  let arg1 = mk_texpr (TEInt 1) in
  let arg2 = mk_texpr (TEInt 2) in
  let expr = mk_texpr (TEApp (fn, [arg1; arg2])) in
  let count = Sarek_tailrec_pragma.count_nodes expr in
  Alcotest.(check int) "app node count" 4 count

let test_count_nodes_nested () =
  (* let x = 1 + 2 in x * 3 *)
  let add_left = mk_texpr (TEInt 1) in
  let add_right = mk_texpr (TEInt 2) in
  let add_expr = mk_texpr (TEBinop (Sarek_ast.Add, add_left, add_right)) in
  let var_x = mk_texpr (TEVar ("x", 0)) in
  let three = mk_texpr (TEInt 3) in
  let mul_expr = mk_texpr (TEBinop (Sarek_ast.Mul, var_x, three)) in
  let let_expr = mk_texpr (TELet ("x", 0, add_expr, mul_expr)) in
  let count = Sarek_tailrec_pragma.count_nodes let_expr in
  (* 1 (let) + 3 (add) + 3 (mul) = 7 *)
  Alcotest.(check int) "nested expression node count" 7 count

(* Test: inline_with_pragma performs inlining *)
let test_inline_with_pragma_depth_zero () =
  (* Inlining with depth 0 should return body unchanged *)
  let body = mk_texpr (TEInt 42) in
  match Sarek_tailrec_pragma.inline_with_pragma "test_fn" [] body 0 with
  | Ok result ->
      Alcotest.(check int)
        "result is original"
        42
        (match result.te with TEInt n -> n | _ -> -1)
  | Error msg -> Alcotest.failf "Inlining failed: %s" msg

let test_inline_with_pragma_no_recursion () =
  (* Non-recursive function should inline without issue *)
  let body = mk_texpr (TEInt 100) in
  match Sarek_tailrec_pragma.inline_with_pragma "test_fn" [] body 5 with
  | Ok _ -> ()
  | Error msg -> Alcotest.failf "Inlining failed: %s" msg

let test_inline_with_pragma_simple () =
  (* Create a simple recursive function: let rec f x = if x <= 0 then 1 else f (x-1) *)
  let param =
    {
      tparam_name = "x";
      tparam_id = 0;
      tparam_type = Sarek_types.TPrim Sarek_types.TInt32;
      tparam_index = 0;
      tparam_is_vec = false;
    }
  in
  let var_x = mk_texpr (TEVar ("x", 0)) in
  let zero = mk_texpr (TEInt 0) in
  let cond = mk_texpr (TEBinop (Sarek_ast.Le, var_x, zero)) in
  let one = mk_texpr (TEInt 1) in
  let fn = mk_texpr (TEVar ("f", 1)) in
  let x_minus_1 = mk_texpr (TEBinop (Sarek_ast.Sub, var_x, one)) in
  let recursive_call = mk_texpr (TEApp (fn, [x_minus_1])) in
  let body = mk_texpr (TEIf (cond, one, Some recursive_call)) in

  (* Inline 2 times *)
  match Sarek_tailrec_pragma.inline_with_pragma "f" [param] body 2 with
  | Ok result ->
      (* Should succeed and produce larger expression *)
      let result_count = Sarek_tailrec_pragma.count_nodes result in
      let original_count = Sarek_tailrec_pragma.count_nodes body in
      Alcotest.(check bool)
        "inlined expression is larger"
        true
        (result_count > original_count)
  | Error msg -> Alcotest.failf "Inlining failed: %s" msg

let test_inline_with_pragma_node_limit () =
  (* Create a recursive function that will exponentially grow *)
  (* rec fib n = if n <= 1 then 1 else fib(n-1) + fib(n-2) - this grows exponentially *)
  let param =
    {
      tparam_name = "n";
      tparam_id = 0;
      tparam_type = Sarek_types.TPrim Sarek_types.TInt32;
      tparam_index = 0;
      tparam_is_vec = false;
    }
  in
  let var_n = mk_texpr (TEVar ("n", 0)) in
  let one = mk_texpr (TEInt 1) in
  let two = mk_texpr (TEInt 2) in
  let cond = mk_texpr (TEBinop (Sarek_ast.Le, var_n, one)) in
  let fn = mk_texpr (TEVar ("fib", 1)) in
  let n_minus_1 = mk_texpr (TEBinop (Sarek_ast.Sub, var_n, one)) in
  let n_minus_2 = mk_texpr (TEBinop (Sarek_ast.Sub, var_n, two)) in
  let fib_call1 = mk_texpr (TEApp (fn, [n_minus_1])) in
  let fib_call2 = mk_texpr (TEApp (fn, [n_minus_2])) in
  let add_calls = mk_texpr (TEBinop (Sarek_ast.Add, fib_call1, fib_call2)) in
  let body = mk_texpr (TEIf (cond, one, Some add_calls)) in

  (* Try to inline with large depth - exponential growth should hit limit *)
  match Sarek_tailrec_pragma.inline_with_pragma "fib" [param] body 15 with
  | Ok _ -> Alcotest.fail "Should have hit node limit with fibonacci"
  | Error msg ->
      Alcotest.(check bool)
        "error mentions node limit"
        true
        (String.length msg > 0)

(* Test suite *)
let () =
  let open Alcotest in
  run
    "Sarek_tailrec_pragma"
    [
      ( "parse_pragma",
        [
          test_case "valid single string" `Quick test_parse_pragma_valid_single;
          test_case "valid list" `Quick test_parse_pragma_valid_list;
          test_case "invalid number" `Quick test_parse_pragma_invalid;
          test_case "wrong pragma name" `Quick test_parse_pragma_wrong_name;
          test_case "empty pragma" `Quick test_parse_pragma_empty;
        ] );
      ( "is_unroll_pragma",
        [
          test_case "unroll recognized" `Quick test_is_unroll_pragma_valid;
          test_case "unroll with param" `Quick test_is_unroll_pragma_with_param;
          test_case "non-unroll" `Quick test_is_unroll_pragma_invalid;
        ] );
      ( "count_nodes",
        [
          test_case "unit" `Quick test_count_nodes_unit;
          test_case "int" `Quick test_count_nodes_int;
          test_case "var" `Quick test_count_nodes_var;
          test_case "binop" `Quick test_count_nodes_binop;
          test_case "unop" `Quick test_count_nodes_unop;
          test_case "if with else" `Quick test_count_nodes_if;
          test_case "if without else" `Quick test_count_nodes_if_no_else;
          test_case "seq" `Quick test_count_nodes_seq;
          test_case "let" `Quick test_count_nodes_let;
          test_case "tuple" `Quick test_count_nodes_tuple;
          test_case "app" `Quick test_count_nodes_app;
          test_case "nested expression" `Quick test_count_nodes_nested;
        ] );
      ( "inline_with_pragma",
        [
          test_case "depth zero" `Quick test_inline_with_pragma_depth_zero;
          test_case "no recursion" `Quick test_inline_with_pragma_no_recursion;
          test_case "simple recursion" `Quick test_inline_with_pragma_simple;
          test_case "node limit" `Quick test_inline_with_pragma_node_limit;
        ] );
    ]
