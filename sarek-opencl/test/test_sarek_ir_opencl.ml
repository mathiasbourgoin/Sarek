(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Sarek_ir_opencl Tests - Verify OpenCL Code Generation
 ******************************************************************************)

open Sarek_opencl
open Sarek_ir_types

(** Helper: Create a variable record *)
let make_var name ty =
  {var_id = 0; var_name = name; var_type = ty; var_mutable = false}

(** Test basic expression generation *)
let test_basic_literals () =
  let buf = Buffer.create 64 in
  Sarek_ir_opencl.gen_expr buf (EConst (CInt32 42l)) ;
  Alcotest.(check string) "int32 literal" "42" (Buffer.contents buf) ;

  let buf = Buffer.create 64 in
  Sarek_ir_opencl.gen_expr buf (EConst (CInt64 42L)) ;
  Alcotest.(check string) "int64 literal" "42L" (Buffer.contents buf) ;

  let buf = Buffer.create 64 in
  Sarek_ir_opencl.gen_expr buf (EConst (CFloat32 3.14)) ;
  let result = Buffer.contents buf in
  Alcotest.(check bool)
    "float32 literal has 'f' suffix"
    true
    (String.length result > 0 && result.[String.length result - 1] = 'f') ;

  let buf = Buffer.create 64 in
  Sarek_ir_opencl.gen_expr buf (EConst (CBool true)) ;
  Alcotest.(check string) "bool true literal" "1" (Buffer.contents buf) ;

  let buf = Buffer.create 64 in
  Sarek_ir_opencl.gen_expr buf (EConst (CBool false)) ;
  Alcotest.(check string) "bool false literal" "0" (Buffer.contents buf)

(** Test operations *)
let test_operations () =
  let buf = Buffer.create 64 in
  let x = make_var "x" TInt32 in
  let y = make_var "y" TInt32 in
  Sarek_ir_opencl.gen_expr buf (EBinop (Add, EVar x, EVar y)) ;
  Alcotest.(check string) "addition" "(x + y)" (Buffer.contents buf) ;

  let buf = Buffer.create 64 in
  Sarek_ir_opencl.gen_expr buf (EBinop (Sub, EVar x, EVar y)) ;
  Alcotest.(check string) "subtraction" "(x - y)" (Buffer.contents buf) ;

  let buf = Buffer.create 64 in
  Sarek_ir_opencl.gen_expr buf (EBinop (Mul, EVar x, EVar y)) ;
  Alcotest.(check string) "multiplication" "(x * y)" (Buffer.contents buf)

(** Test basic statements *)
let test_basics () =
  let buf = Buffer.create 64 in
  Sarek_ir_opencl.gen_stmt buf "" SEmpty ;
  Alcotest.(check string) "empty statement" "" (Buffer.contents buf)

(** Test assignment *)
let test_assignment () =
  let buf = Buffer.create 64 in
  let x = make_var "x" TInt32 in
  Sarek_ir_opencl.gen_stmt buf "" (SAssign (LVar x, EConst (CInt32 42l))) ;
  Alcotest.(check string) "assignment" "x = 42;\n" (Buffer.contents buf)

(** Test if statement *)
let test_if_statement () =
  let buf = Buffer.create 64 in
  let x = make_var "x" TInt32 in
  Sarek_ir_opencl.gen_stmt
    buf
    ""
    (SIf
       ( EBinop (Gt, EVar x, EConst (CInt32 0l)),
         SAssign (LVar x, EConst (CInt32 1l)),
         None )) ;
  let result = Buffer.contents buf in
  Alcotest.(check bool)
    "if statement contains 'if'"
    true
    (Str.string_match (Str.regexp ".*if.*") result 0)

(** Test while loop *)
let test_while_loop () =
  let buf = Buffer.create 64 in
  let x = make_var "x" TInt32 in
  Sarek_ir_opencl.gen_stmt
    buf
    ""
    (SWhile
       ( EBinop (Gt, EVar x, EConst (CInt32 0l)),
         SAssign (LVar x, EBinop (Sub, EVar x, EConst (CInt32 1l))) )) ;
  let result = Buffer.contents buf in
  Alcotest.(check bool)
    "while loop contains 'while'"
    true
    (Str.string_match (Str.regexp ".*while.*") result 0)

(** Test for loop *)
let test_for_loop () =
  let buf = Buffer.create 128 in
  let i = make_var "i" TInt32 in
  Sarek_ir_opencl.gen_stmt
    buf
    ""
    (SFor (i, EConst (CInt32 0l), EConst (CInt32 10l), Upto, SEmpty)) ;
  let result = Buffer.contents buf in
  Alcotest.(check bool)
    "for loop contains 'for'"
    true
    (Str.string_match (Str.regexp ".*for.*") result 0) ;
  Alcotest.(check bool)
    "for loop has upto operator (<=)"
    true
    (Str.string_match (Str.regexp ".*<=.*") result 0)

(** Test return statement *)
let test_return () =
  let buf = Buffer.create 64 in
  Sarek_ir_opencl.gen_stmt buf "" (SReturn (EConst (CInt32 42l))) ;
  Alcotest.(check string)
    "return statement"
    "return 42;\n"
    (Buffer.contents buf)

(** Test barrier *)
let test_barriers () =
  let buf = Buffer.create 64 in
  Sarek_ir_opencl.gen_stmt buf "" SBarrier ;
  let result = Buffer.contents buf in
  Alcotest.(check bool)
    "barrier contains 'barrier'"
    true
    (Str.string_match (Str.regexp ".*barrier.*") result 0)

(** Test let binding *)
let test_let_binding () =
  let buf = Buffer.create 64 in
  let x = make_var "x" TInt32 in
  Sarek_ir_opencl.gen_stmt buf "" (SLet (x, EConst (CInt32 42l), SEmpty)) ;
  let result = Buffer.contents buf in
  Alcotest.(check bool)
    "let binding contains variable declaration"
    true
    (Str.string_match (Str.regexp ".*int.*x.*=.*42.*") result 0)

(** Test let mut *)
let test_let_mut () =
  let buf = Buffer.create 64 in
  let x = make_var "x" TInt32 in
  Sarek_ir_opencl.gen_stmt buf "" (SLetMut (x, EConst (CInt32 42l), SEmpty)) ;
  let result = Buffer.contents buf in
  Alcotest.(check bool)
    "let mut contains variable declaration"
    true
    (Str.string_match (Str.regexp ".*int.*x.*=.*42.*") result 0)

(** Test block *)
let test_block () =
  let buf = Buffer.create 64 in
  Sarek_ir_opencl.gen_stmt buf "" (SBlock SEmpty) ;
  let result = Buffer.contents buf in
  Alcotest.(check bool)
    "block contains opening brace"
    true
    (String.contains result '{') ;
  Alcotest.(check bool)
    "block contains closing brace"
    true
    (String.contains result '}')

(** Test pragma *)
let test_pragma () =
  let buf = Buffer.create 64 in
  Sarek_ir_opencl.gen_stmt buf "" (SPragma (["unroll"], SEmpty)) ;
  let result = Buffer.contents buf in
  Alcotest.(check bool)
    "pragma contains '#pragma'"
    true
    (Str.string_match (Str.regexp ".*#pragma.*") result 0) ;
  Alcotest.(check bool)
    "pragma contains 'unroll'"
    true
    (Str.string_match (Str.regexp ".*unroll.*") result 0)

(** Test suite *)
let () =
  Alcotest.run
    "Sarek_ir_opencl"
    [
      ( "expressions",
        [
          Alcotest.test_case "basic literals" `Quick test_basic_literals;
          Alcotest.test_case "operations" `Quick test_operations;
        ] );
      ( "statements",
        [
          Alcotest.test_case "basics" `Quick test_basics;
          Alcotest.test_case "assignment" `Quick test_assignment;
          Alcotest.test_case "if statement" `Quick test_if_statement;
          Alcotest.test_case "while loop" `Quick test_while_loop;
          Alcotest.test_case "for loop" `Quick test_for_loop;
          Alcotest.test_case "return" `Quick test_return;
          Alcotest.test_case "barriers" `Quick test_barriers;
          Alcotest.test_case "let binding" `Quick test_let_binding;
          Alcotest.test_case "let mut" `Quick test_let_mut;
          Alcotest.test_case "block" `Quick test_block;
          Alcotest.test_case "pragma" `Quick test_pragma;
        ] );
    ]
