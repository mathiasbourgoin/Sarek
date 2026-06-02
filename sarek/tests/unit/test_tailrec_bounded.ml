(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Unit tests for Sarek_tailrec_bounded module
 *
 * Tests bounded recursion inlining (currently disabled in production)
 ******************************************************************************)

(* Force stdlib initialization *)
let () = Sarek_stdlib.force_init ()

let dummy_loc =
  Sarek_ast.
    {
      loc_file = "test.ml";
      loc_line = 1;
      loc_col = 0;
      loc_end_line = 1;
      loc_end_col = 10;
    }

(* Helper to create typed expression *)
let mk_texpr te ty = Sarek_typed_ast.{te; ty; te_loc = dummy_loc}

(* Test: inline_bounded_recursion with depth 0 returns unchanged *)
let test_inline_bounded_depth_zero () =
  let ty = Sarek_types.(TReg Int) in
  let body = mk_texpr (Sarek_typed_ast.TEInt 42) ty in
  let params = [] in
  let result =
    Sarek_tailrec_bounded.inline_bounded_recursion "f" params body 0 dummy_loc
  in
  (* Should return body unchanged for base case *)
  match result.Sarek_typed_ast.te with
  | TEInt 42 -> ()
  | _ -> Alcotest.fail "expected unchanged expression"

(* Test: inline_bounded_recursion with simple constant *)
let test_inline_bounded_constant () =
  let ty = Sarek_types.(TReg Int) in
  let body = mk_texpr (Sarek_typed_ast.TEInt 100) ty in
  let params = [] in
  let result =
    Sarek_tailrec_bounded.inline_bounded_recursion "f" params body 3 dummy_loc
  in
  match result.Sarek_typed_ast.te with
  | TEInt 100 -> ()
  | _ -> Alcotest.fail "expected constant"

(* Test: inline_bounded_recursion preserves non-recursive expressions *)
let test_inline_bounded_non_recursive () =
  let ty = Sarek_types.(TReg Int) in
  let var = mk_texpr (Sarek_typed_ast.TEVar ("x", 0)) ty in
  let params = [] in
  let result =
    Sarek_tailrec_bounded.inline_bounded_recursion "f" params var 5 dummy_loc
  in
  match result.Sarek_typed_ast.te with
  | TEVar ("x", 0) -> ()
  | _ -> Alcotest.fail "expected variable"

(* Test: inline_bounded_recursion handles if-expressions *)
let test_inline_bounded_if_expr () =
  let ty = Sarek_types.(TReg Int) in
  let cond = mk_texpr (Sarek_typed_ast.TEBool true) Sarek_types.(TPrim TBool) in
  let then_e = mk_texpr (Sarek_typed_ast.TEInt 1) ty in
  let else_e = mk_texpr (Sarek_typed_ast.TEInt 2) ty in
  let if_expr =
    mk_texpr (Sarek_typed_ast.TEIf (cond, then_e, Some else_e)) ty
  in
  let params = [] in
  let result =
    Sarek_tailrec_bounded.inline_bounded_recursion
      "f"
      params
      if_expr
      2
      dummy_loc
  in
  match result.Sarek_typed_ast.te with
  | TEIf _ -> ()
  | _ -> Alcotest.fail "expected if expression"

(* Test: inline_bounded_recursion handles sequences *)
let test_inline_bounded_seq () =
  let ty = Sarek_types.(TReg Int) in
  let e1 = mk_texpr (Sarek_typed_ast.TEInt 1) ty in
  let e2 = mk_texpr (Sarek_typed_ast.TEInt 2) ty in
  let seq = mk_texpr (Sarek_typed_ast.TESeq [e1; e2]) ty in
  let params = [] in
  let result =
    Sarek_tailrec_bounded.inline_bounded_recursion "f" params seq 3 dummy_loc
  in
  match result.Sarek_typed_ast.te with
  | TESeq es -> Alcotest.(check int) "two expressions" 2 (List.length es)
  | _ -> Alcotest.fail "expected sequence"

(* Test: inline_bounded_recursion handles binop *)
let test_inline_bounded_binop () =
  let ty = Sarek_types.(TReg Int) in
  let a = mk_texpr (Sarek_typed_ast.TEInt 1) ty in
  let b = mk_texpr (Sarek_typed_ast.TEInt 2) ty in
  let binop = mk_texpr (Sarek_typed_ast.TEBinop (Sarek_ast.Add, a, b)) ty in
  let params = [] in
  let result =
    Sarek_tailrec_bounded.inline_bounded_recursion "f" params binop 2 dummy_loc
  in
  match result.Sarek_typed_ast.te with
  | TEBinop (Sarek_ast.Add, _, _) -> ()
  | _ -> Alcotest.fail "expected binop"

(* Test: inline_bounded_recursion handles unop *)
let test_inline_bounded_unop () =
  let ty = Sarek_types.(TReg Int) in
  let e = mk_texpr (Sarek_typed_ast.TEInt 5) ty in
  let unop = mk_texpr (Sarek_typed_ast.TEUnop (Sarek_ast.Neg, e)) ty in
  let params = [] in
  let result =
    Sarek_tailrec_bounded.inline_bounded_recursion "f" params unop 2 dummy_loc
  in
  match result.Sarek_typed_ast.te with
  | TEUnop (Sarek_ast.Neg, _) -> ()
  | _ -> Alcotest.fail "expected unop"

(* Test suite *)
let () =
  Alcotest.run
    "Sarek_tailrec_bounded"
    [
      ( "inline_bounded_recursion",
        [
          Alcotest.test_case "depth zero" `Quick test_inline_bounded_depth_zero;
          Alcotest.test_case "constant" `Quick test_inline_bounded_constant;
          Alcotest.test_case
            "non-recursive"
            `Quick
            test_inline_bounded_non_recursive;
          Alcotest.test_case "if expression" `Quick test_inline_bounded_if_expr;
          Alcotest.test_case "sequence" `Quick test_inline_bounded_seq;
          Alcotest.test_case "binop" `Quick test_inline_bounded_binop;
          Alcotest.test_case "unop" `Quick test_inline_bounded_unop;
        ] );
    ]
