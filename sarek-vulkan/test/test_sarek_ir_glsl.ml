(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Sarek_ir_glsl Tests - Verify GLSL Code Generation
 ******************************************************************************)

open Sarek_vulkan
open Sarek_ir_types

(** Helper: Create a variable record *)
let make_var name ty =
  {var_id = 0; var_name = name; var_type = ty; var_mutable = false}

(** Test basic expression generation *)
let test_basic_literals () =
  let buf = Buffer.create 64 in
  Sarek_ir_glsl.gen_expr buf (EConst (CInt32 42l)) ;
  Alcotest.(check string) "int32 literal" "42" (Buffer.contents buf) ;

  let buf = Buffer.create 64 in
  Sarek_ir_glsl.gen_expr buf (EConst (CInt64 42L)) ;
  Alcotest.(check string) "int64 literal" "42L" (Buffer.contents buf) ;

  let buf = Buffer.create 64 in
  Sarek_ir_glsl.gen_expr buf (EConst (CFloat32 3.14)) ;
  let result = Buffer.contents buf in
  Alcotest.(check bool)
    "float32 literal is numeric"
    true
    (String.length result > 0 && result.[0] >= '0' && result.[0] <= '9') ;

  let buf = Buffer.create 64 in
  Sarek_ir_glsl.gen_expr buf (EConst (CBool true)) ;
  Alcotest.(check string) "bool true literal" "true" (Buffer.contents buf) ;

  let buf = Buffer.create 64 in
  Sarek_ir_glsl.gen_expr buf (EConst (CBool false)) ;
  Alcotest.(check string) "bool false literal" "false" (Buffer.contents buf)

(** Test operations *)
let test_operations () =
  let buf = Buffer.create 64 in
  let x = make_var "x" TInt32 in
  let y = make_var "y" TInt32 in
  Sarek_ir_glsl.gen_expr buf (EBinop (Add, EVar x, EVar y)) ;
  Alcotest.(check string) "addition" "(x + y)" (Buffer.contents buf) ;

  let buf = Buffer.create 64 in
  Sarek_ir_glsl.gen_expr buf (EBinop (Sub, EVar x, EVar y)) ;
  Alcotest.(check string) "subtraction" "(x - y)" (Buffer.contents buf) ;

  let buf = Buffer.create 64 in
  Sarek_ir_glsl.gen_expr buf (EBinop (Mul, EVar x, EVar y)) ;
  Alcotest.(check string) "multiplication" "(x * y)" (Buffer.contents buf)

(** Test basic statements *)
let test_basics () =
  let buf = Buffer.create 64 in
  Sarek_ir_glsl.gen_stmt buf "" SEmpty ;
  Alcotest.(check string) "empty statement" "" (Buffer.contents buf)

(** Test assignment *)
let test_assignment () =
  let buf = Buffer.create 64 in
  let x = make_var "x" TInt32 in
  Sarek_ir_glsl.gen_stmt buf "" (SAssign (LVar x, EConst (CInt32 42l))) ;
  Alcotest.(check string) "assignment" "x = 42;\n" (Buffer.contents buf)

(** Test if statement *)
let test_if_statement () =
  let buf = Buffer.create 64 in
  let x = make_var "x" TInt32 in
  Sarek_ir_glsl.gen_stmt
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
  let i = make_var "i" TInt32 in
  Sarek_ir_glsl.gen_stmt
    buf
    ""
    (SWhile
       ( EBinop (Lt, EVar i, EConst (CInt32 10l)),
         SAssign (LVar i, EBinop (Add, EVar i, EConst (CInt32 1l))) )) ;
  let result = Buffer.contents buf in
  Alcotest.(check bool)
    "while loop contains 'while'"
    true
    (Str.string_match (Str.regexp ".*while.*") result 0)

(** Test for loop *)
let test_for_loop () =
  let buf = Buffer.create 64 in
  let i = make_var "i" TInt32 in
  Sarek_ir_glsl.gen_stmt
    buf
    ""
    (SFor (i, EConst (CInt32 0l), EConst (CInt32 10l), Upto, SEmpty)) ;
  let result = Buffer.contents buf in
  Alcotest.(check bool)
    "for loop contains 'for'"
    true
    (Str.string_match (Str.regexp ".*for.*") result 0) ;
  Alcotest.(check bool)
    "for loop uses <= for upto"
    true
    (Str.string_match (Str.regexp ".*<=.*") result 0)

(** Test barrier intrinsics *)
let test_barriers () =
  let buf = Buffer.create 64 in
  Sarek_ir_glsl.gen_stmt buf "  " SBarrier ;
  Alcotest.(check string)
    "barrier generates barrier()"
    "  barrier();\n"
    (Buffer.contents buf) ;

  let buf = Buffer.create 64 in
  Sarek_ir_glsl.gen_stmt buf "  " SWarpBarrier ;
  Alcotest.(check string)
    "warp barrier generates subgroupBarrier()"
    "  subgroupBarrier();\n"
    (Buffer.contents buf) ;

  let buf = Buffer.create 64 in
  Sarek_ir_glsl.gen_stmt buf "  " SMemFence ;
  Alcotest.(check string)
    "memfence generates memoryBarrier()"
    "  memoryBarrier();\n"
    (Buffer.contents buf)

(** Test thread intrinsics *)
let test_thread_intrinsics () =
  let result = Sarek_ir_glsl.glsl_thread_intrinsic "thread_idx_x" in
  Alcotest.(check bool)
    "thread_idx_x uses gl_LocalInvocationID.x"
    true
    (Str.string_match (Str.regexp ".*gl_LocalInvocationID.*x.*") result 0) ;

  let result = Sarek_ir_glsl.glsl_thread_intrinsic "global_idx_x" in
  Alcotest.(check bool)
    "global_idx_x uses gl_GlobalInvocationID.x"
    true
    (Str.string_match (Str.regexp ".*gl_GlobalInvocationID.*x.*") result 0)

(** Test atomic operations *)
let test_atomics () =
  let buf = Buffer.create 128 in
  let addr = make_var "counter" TInt32 in
  let value = EConst (CInt32 1l) in
  Sarek_ir_glsl.gen_intrinsic buf [] "atomic_add" [EVar addr; value] ;
  let result = Buffer.contents buf in
  Alcotest.(check bool)
    "atomic_add generates atomicAdd"
    true
    (Str.string_match (Str.regexp ".*atomicAdd.*") result 0)

(** Test type mapping *)
let test_type_mapping () =
  Alcotest.(check string)
    "int32 maps to int"
    "int"
    (Sarek_ir_glsl.glsl_type_of_elttype TInt32) ;
  Alcotest.(check string)
    "int64 maps to int64_t"
    "int64_t"
    (Sarek_ir_glsl.glsl_type_of_elttype TInt64) ;
  Alcotest.(check string)
    "float32 maps to float"
    "float"
    (Sarek_ir_glsl.glsl_type_of_elttype TFloat32) ;
  Alcotest.(check string)
    "float64 maps to double"
    "double"
    (Sarek_ir_glsl.glsl_type_of_elttype TFloat64) ;
  Alcotest.(check string)
    "bool maps to bool"
    "bool"
    (Sarek_ir_glsl.glsl_type_of_elttype TBool)

(** Test variable declaration helper *)
let test_var_decl () =
  let buf = Buffer.create 64 in
  let x = make_var "x" TInt32 in
  Sarek_ir_glsl.gen_var_decl buf "" x.var_name x.var_type (EConst (CInt32 42l)) ;
  Alcotest.(check string)
    "gen_var_decl produces type var = expr;"
    "int x = 42;\n"
    (Buffer.contents buf)

(** Test array declaration helper *)
let test_array_decl () =
  let buf = Buffer.create 64 in
  Sarek_ir_glsl.gen_array_decl buf "" "arr" TFloat32 (EConst (CInt32 256l)) ;
  Alcotest.(check string)
    "gen_array_decl produces type arr[size];"
    "float arr[256];\n"
    (Buffer.contents buf)

(** Test indent_nested helper *)
let test_indent_nested () =
  let nested = Sarek_ir_glsl.indent_nested "  " in
  Alcotest.(check string) "indent_nested adds two spaces" "    " nested

(** Test suite *)
let () =
  Alcotest.run
    "Sarek_ir_glsl"
    [
      ( "literals",
        [Alcotest.test_case "basic literals" `Quick test_basic_literals] );
      ("operations", [Alcotest.test_case "operations" `Quick test_operations]);
      ("basics", [Alcotest.test_case "basic statements" `Quick test_basics]);
      ("assignment", [Alcotest.test_case "assignment" `Quick test_assignment]);
      ("if", [Alcotest.test_case "if statement" `Quick test_if_statement]);
      ("while", [Alcotest.test_case "while loop" `Quick test_while_loop]);
      ("for", [Alcotest.test_case "for loop" `Quick test_for_loop]);
      ( "barriers",
        [Alcotest.test_case "barrier intrinsics" `Quick test_barriers] );
      ( "thread",
        [Alcotest.test_case "thread intrinsics" `Quick test_thread_intrinsics]
      );
      ("atomics", [Alcotest.test_case "atomic operations" `Quick test_atomics]);
      ("types", [Alcotest.test_case "type mapping" `Quick test_type_mapping]);
      ("var_decl", [Alcotest.test_case "var declaration" `Quick test_var_decl]);
      ( "array_decl",
        [Alcotest.test_case "array declaration" `Quick test_array_decl] );
      ("indent", [Alcotest.test_case "indent helper" `Quick test_indent_nested]);
    ]
