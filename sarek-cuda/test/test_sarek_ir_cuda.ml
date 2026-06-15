(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Unit tests for Sarek_ir_cuda code generation *)

open Sarek_cuda
open Sarek_ir_types

(** Helper to create a var record *)
let make_var name ty =
  {var_name = name; var_id = 0; var_type = ty; var_mutable = false}

(** Helper to generate code from an expression *)
let gen_expr_str e =
  let buf = Buffer.create 256 in
  Sarek_ir_cuda.gen_expr buf e ;
  Buffer.contents buf

(** Helper to generate code from a statement *)
let gen_stmt_str ?(indent = "") s =
  let buf = Buffer.create 256 in
  Sarek_ir_cuda.gen_stmt buf indent s ;
  Buffer.contents buf

(** Test basic expression generation *)
let test_gen_expr_basics () =
  (* Test integer literal *)
  let e1 = EConst (CInt32 42l) in
  Alcotest.(check string) "Integer literal" "42" (gen_expr_str e1) ;

  (* Test float literal *)
  let e2 = EConst (CFloat64 3.14) in
  let s2 = gen_expr_str e2 in
  Alcotest.(check bool)
    "Float literal contains dot"
    true
    (String.contains s2 '.') ;

  (* Test boolean literals *)
  let e3 = EConst (CBool true) in
  Alcotest.(check string) "Boolean true" "1" (gen_expr_str e3) ;
  let e4 = EConst (CBool false) in
  Alcotest.(check string) "Boolean false" "0" (gen_expr_str e4)

(** Test variable and binary operations *)
let test_gen_expr_operations () =
  (* Test variable *)
  let var = make_var "x" TInt32 in
  let e1 = EVar var in
  Alcotest.(check string) "Variable" "x" (gen_expr_str e1) ;

  (* Test binary addition *)
  let e2 = EBinop (Add, EConst (CInt32 1l), EConst (CInt32 2l)) in
  Alcotest.(check string) "Addition" "(1 + 2)" (gen_expr_str e2) ;

  (* Test binary multiplication *)
  let e3 = EBinop (Mul, EVar var, EConst (CInt32 3l)) in
  Alcotest.(check string) "Multiplication" "(x * 3)" (gen_expr_str e3)

(** Test statement generation - empty and sequence *)
let test_gen_stmt_basics () =
  (* Test empty statement *)
  let s1 = SEmpty in
  Alcotest.(check string) "Empty statement" "" (gen_stmt_str s1) ;

  (* Test sequence *)
  let s2 = SSeq [SEmpty; SEmpty] in
  Alcotest.(check string) "Sequence of empty" "" (gen_stmt_str s2)

(** Test assignment statement generation *)
let test_gen_stmt_assignment () =
  (* Simple variable assignment *)
  let var = make_var "result" TInt32 in
  let lv = LVar var in
  let e = EConst (CInt32 42l) in
  let s = SAssign (lv, e) in
  let code = gen_stmt_str s in
  Alcotest.(check bool) "Assignment contains =" true (String.contains code '=') ;
  Alcotest.(check bool)
    "Assignment contains result"
    true
    Str.(string_match (regexp ".*result.*") code 0) ;
  Alcotest.(check bool)
    "Assignment ends with semicolon"
    true
    (String.ends_with ~suffix:";\n" code)

(** Test if statement generation *)
let test_gen_stmt_if () =
  let cond = EConst (CBool true) in
  let then_ = SEmpty in

  (* If without else *)
  let s1 = SIf (cond, then_, None) in
  let code1 = gen_stmt_str s1 in
  Alcotest.(check bool) "If contains keyword" true (String.contains code1 'i') ;

  (* If with else *)
  let else_ = SEmpty in
  let s2 = SIf (cond, then_, Some else_) in
  let code2 = gen_stmt_str s2 in
  Alcotest.(check bool) "If-else contains else" true (String.contains code2 'e')

(** Test while loop generation *)
let test_gen_stmt_while () =
  let var = make_var "i" TInt32 in
  let cond = EBinop (Lt, EVar var, EConst (CInt32 10l)) in
  let body = SEmpty in
  let s = SWhile (cond, body) in
  let code = gen_stmt_str s in
  Alcotest.(check bool)
    "While contains keyword"
    true
    Str.(string_match (regexp ".*while.*") code 0) ;
  Alcotest.(check bool)
    "While contains condition"
    true
    Str.(string_match (regexp ".*i.*10.*") code 0)

(** Test for loop generation *)
let test_gen_stmt_for () =
  let var = make_var "i" TInt32 in
  let start = EConst (CInt32 0l) in
  let stop = EConst (CInt32 9l) in
  let body = SEmpty in

  (* Upto loop *)
  let s1 = SFor (var, start, stop, Upto, body) in
  let code1 = gen_stmt_str s1 in
  Alcotest.(check bool)
    "For-upto contains <="
    true
    Str.(string_match (regexp ".*<=.*") code1 0) ;
  Alcotest.(check bool)
    "For-upto contains ++"
    true
    Str.(string_match (regexp ".*\\+\\+.*") code1 0) ;

  (* Downto loop *)
  let s2 = SFor (var, start, stop, Downto, body) in
  let code2 = gen_stmt_str s2 in
  Alcotest.(check bool)
    "For-downto contains >="
    true
    Str.(string_match (regexp ".*>=.*") code2 0) ;
  Alcotest.(check bool)
    "For-downto contains --"
    true
    Str.(string_match (regexp ".*--.*") code2 0)

(** Test return statement *)
let test_gen_stmt_return () =
  let e = EConst (CInt32 0l) in
  let s = SReturn e in
  let code = gen_stmt_str s in
  Alcotest.(check bool)
    "Return contains keyword"
    true
    Str.(string_match (regexp ".*return.*") code 0) ;
  Alcotest.(check bool)
    "Return ends with semicolon"
    true
    (String.ends_with ~suffix:";\n" code)

(** Test barrier intrinsics *)
let test_gen_stmt_barriers () =
  (* Test SBarrier *)
  let s1 = SBarrier in
  let code1 = gen_stmt_str s1 in
  Alcotest.(check bool)
    "Barrier generates __syncthreads"
    true
    Str.(string_match (regexp ".*__syncthreads.*") code1 0) ;

  (* Test SWarpBarrier *)
  let s2 = SWarpBarrier in
  let code2 = gen_stmt_str s2 in
  Alcotest.(check bool)
    "Warp barrier generates __syncwarp"
    true
    Str.(string_match (regexp ".*__syncwarp.*") code2 0) ;

  (* Test SMemFence *)
  let s3 = SMemFence in
  let code3 = gen_stmt_str s3 in
  Alcotest.(check bool)
    "Memory fence generates __threadfence"
    true
    Str.(string_match (regexp ".*__threadfence.*") code3 0)

(** Test let binding *)
let test_gen_stmt_let () =
  let var = make_var "x" TInt32 in
  let e = EConst (CInt32 42l) in
  let body = SEmpty in
  let s = SLet (var, e, body) in
  let code = gen_stmt_str s in
  Alcotest.(check bool)
    "Let contains type"
    true
    Str.(string_match (regexp ".*int.*") code 0) ;
  Alcotest.(check bool)
    "Let contains variable name"
    true
    Str.(string_match (regexp ".*x.*") code 0) ;
  Alcotest.(check bool)
    "Let contains assignment"
    true
    (String.contains code '=')

(** Test mutable let binding *)
let test_gen_stmt_let_mut () =
  let var = make_var "y" TFloat64 in
  let e = EConst (CFloat64 3.14) in
  let body = SEmpty in
  let s = SLetMut (var, e, body) in
  let code = gen_stmt_str s in
  Alcotest.(check bool)
    "LetMut contains variable"
    true
    Str.(string_match (regexp ".*y.*") code 0)

(** Test block statement *)
let test_gen_stmt_block () =
  let body = SEmpty in
  let s = SBlock body in
  let code = gen_stmt_str s in
  Alcotest.(check bool)
    "Block has opening brace"
    true
    (String.contains code '{') ;
  Alcotest.(check bool)
    "Block has closing brace"
    true
    (String.contains code '}')

(** Test pragma statement *)
let test_gen_stmt_pragma () =
  let body = SEmpty in
  let s = SPragma (["unroll"], body) in
  let code = gen_stmt_str s in
  Alcotest.(check bool)
    "Pragma contains #pragma"
    true
    Str.(string_match (regexp ".*#pragma.*") code 0) ;
  Alcotest.(check bool)
    "Pragma contains unroll"
    true
    Str.(string_match (regexp ".*unroll.*") code 0)

(** Test suite *)
let () =
  Alcotest.run
    "Sarek_ir_cuda"
    [
      ( "expressions",
        [
          Alcotest.test_case "basic literals" `Quick test_gen_expr_basics;
          Alcotest.test_case "operations" `Quick test_gen_expr_operations;
        ] );
      ( "statements",
        [
          Alcotest.test_case "basics" `Quick test_gen_stmt_basics;
          Alcotest.test_case "assignment" `Quick test_gen_stmt_assignment;
          Alcotest.test_case "if statement" `Quick test_gen_stmt_if;
          Alcotest.test_case "while loop" `Quick test_gen_stmt_while;
          Alcotest.test_case "for loop" `Quick test_gen_stmt_for;
          Alcotest.test_case "return" `Quick test_gen_stmt_return;
          Alcotest.test_case "barriers" `Quick test_gen_stmt_barriers;
          Alcotest.test_case "let binding" `Quick test_gen_stmt_let;
          Alcotest.test_case "let mut" `Quick test_gen_stmt_let_mut;
          Alcotest.test_case "block" `Quick test_gen_stmt_block;
          Alcotest.test_case "pragma" `Quick test_gen_stmt_pragma;
        ] );
    ]
