(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Unit tests for Sarek_ir_pp
 *
 * Tests pretty printing functions for IR types.
 ******************************************************************************)

open Sarek_ir_types
open Sarek_ir_pp

(** {1 string_of_elttype Tests} *)

let test_string_of_elttype_primitives () =
  assert (string_of_elttype TInt32 = "int32") ;
  assert (string_of_elttype TInt64 = "int64") ;
  assert (string_of_elttype TFloat32 = "float32") ;
  assert (string_of_elttype TFloat64 = "float64") ;
  assert (string_of_elttype TBool = "bool") ;
  assert (string_of_elttype TUnit = "unit") ;
  print_endline "  string_of_elttype primitives: OK"

let test_string_of_elttype_record () =
  let ty = TRecord ("point", [("x", TFloat32); ("y", TFloat32)]) in
  assert (string_of_elttype ty = "point") ;
  print_endline "  string_of_elttype record: OK"

let test_string_of_elttype_variant () =
  let ty = TVariant ("color", [("Red", []); ("Green", []); ("Blue", [])]) in
  assert (string_of_elttype ty = "color") ;
  print_endline "  string_of_elttype variant: OK"

let test_string_of_elttype_array () =
  let ty = TArray (TFloat32, Global) in
  assert (string_of_elttype ty = "global float32[]") ;
  let ty2 = TArray (TInt64, Shared) in
  assert (string_of_elttype ty2 = "shared int64[]") ;
  print_endline "  string_of_elttype array: OK"

let test_string_of_elttype_vec () =
  let ty = TVec TFloat64 in
  assert (string_of_elttype ty = "float64 vector") ;
  print_endline "  string_of_elttype vec: OK"

(** {1 string_of_memspace Tests} *)

let test_string_of_memspace () =
  assert (string_of_memspace Global = "global") ;
  assert (string_of_memspace Shared = "shared") ;
  assert (string_of_memspace Local = "local") ;
  print_endline "  string_of_memspace: OK"

(** {1 string_of_binop Tests} *)

let test_string_of_binop () =
  assert (string_of_binop Add = "+") ;
  assert (string_of_binop Sub = "-") ;
  assert (string_of_binop Mul = "*") ;
  assert (string_of_binop Div = "/") ;
  assert (string_of_binop Mod = "%") ;
  assert (string_of_binop Eq = "==") ;
  assert (string_of_binop Ne = "!=") ;
  assert (string_of_binop Lt = "<") ;
  assert (string_of_binop Le = "<=") ;
  assert (string_of_binop Gt = ">") ;
  assert (string_of_binop Ge = ">=") ;
  assert (string_of_binop And = "&&") ;
  assert (string_of_binop Or = "||") ;
  assert (string_of_binop Shl = "<<") ;
  assert (string_of_binop Shr = ">>") ;
  assert (string_of_binop BitAnd = "&") ;
  assert (string_of_binop BitOr = "|") ;
  assert (string_of_binop BitXor = "^") ;
  print_endline "  string_of_binop: OK"

(** {1 string_of_unop Tests} *)

let test_string_of_unop () =
  assert (string_of_unop Neg = "-") ;
  assert (string_of_unop Not = "!") ;
  assert (string_of_unop BitNot = "~") ;
  print_endline "  string_of_unop: OK"

(** {1 pp_expr Tests} *)

let pp_to_string pp x =
  let buf = Buffer.create 64 in
  let fmt = Format.formatter_of_buffer buf in
  pp fmt x ;
  Format.pp_print_flush fmt () ;
  Buffer.contents buf

let test_pp_expr_const () =
  let e = EConst (CInt32 42l) in
  let s = pp_to_string pp_expr e in
  assert (s = "42") ;

  let e2 = EConst (CFloat32 3.14) in
  let s2 = pp_to_string pp_expr e2 in
  assert (String.sub s2 0 4 = "3.14") ;

  let e3 = EConst (CBool true) in
  let s3 = pp_to_string pp_expr e3 in
  assert (s3 = "true") ;

  print_endline "  pp_expr const: OK"

let test_pp_expr_var () =
  let v : var =
    {var_name = "x"; var_id = 0; var_type = TFloat32; var_mutable = false}
  in
  let e = EVar v in
  let s = pp_to_string pp_expr e in
  assert (s = "x") ;
  print_endline "  pp_expr var: OK"

let test_pp_expr_binop () =
  let e = EBinop (Add, EConst (CInt32 1l), EConst (CInt32 2l)) in
  let s = pp_to_string pp_expr e in
  assert (s = "(1 + 2)") ;

  let e2 = EBinop (Mul, EConst (CFloat32 2.0), EConst (CFloat32 3.0)) in
  let s2 = pp_to_string pp_expr e2 in
  assert (String.sub s2 0 3 = "(2f") ;

  print_endline "  pp_expr binop: OK"

let test_pp_expr_unop () =
  let e = EUnop (Neg, EConst (CInt32 5l)) in
  let s = pp_to_string pp_expr e in
  assert (s = "(-5)") ;
  print_endline "  pp_expr unop: OK"

let test_pp_expr_array_read () =
  let e = EArrayRead ("arr", EConst (CInt32 10l)) in
  let s = pp_to_string pp_expr e in
  assert (s = "arr[10]") ;
  print_endline "  pp_expr array_read: OK"

let test_pp_expr_intrinsic () =
  let e = EIntrinsic (["Float32"], "sin", [EConst (CFloat32 0.5)]) in
  let s = pp_to_string pp_expr e in
  assert (String.length s > 0) ;
  assert (String.sub s 0 11 = "Float32.sin") ;
  print_endline "  pp_expr intrinsic: OK"

let test_pp_expr_record () =
  let e =
    ERecord
      ("point", [("x", EConst (CFloat32 1.0)); ("y", EConst (CFloat32 2.0))])
  in
  let s = pp_to_string pp_expr e in
  assert (String.sub s 0 6 = "point{") ;
  print_endline "  pp_expr record: OK"

let test_pp_expr_if () =
  let e = EIf (EConst (CBool true), EConst (CInt32 1l), EConst (CInt32 0l)) in
  let s = pp_to_string pp_expr e in
  assert (String.sub s 0 5 = "(true") ;
  print_endline "  pp_expr if: OK"

(** {1 pp_stmt Tests} *)

let test_pp_stmt_assign () =
  let v : var =
    {var_name = "x"; var_id = 0; var_type = TInt32; var_mutable = true}
  in
  let s = SAssign (LVar v, EConst (CInt32 42l)) in
  let str = pp_to_string pp_stmt s in
  assert (str = "x = 42;") ;
  print_endline "  pp_stmt assign: OK"

let test_pp_stmt_barrier () =
  let s = SBarrier in
  let str = pp_to_string pp_stmt s in
  assert (str = "__syncthreads();") ;
  print_endline "  pp_stmt barrier: OK"

let test_pp_stmt_warp_barrier () =
  let s = SWarpBarrier in
  let str = pp_to_string pp_stmt s in
  assert (str = "__syncwarp();") ;
  print_endline "  pp_stmt warp_barrier: OK"

let test_pp_stmt_mem_fence () =
  let s = SMemFence in
  let str = pp_to_string pp_stmt s in
  assert (str = "__threadfence();") ;
  print_endline "  pp_stmt mem_fence: OK"

let test_pp_stmt_return () =
  let s = SReturn (EConst (CInt32 0l)) in
  let str = pp_to_string pp_stmt s in
  assert (str = "return 0;") ;
  print_endline "  pp_stmt return: OK"

let test_pp_stmt_empty () =
  let s = SEmpty in
  let str = pp_to_string pp_stmt s in
  assert (str = "") ;
  print_endline "  pp_stmt empty: OK"

(** {1 pp_decl Tests} *)

let test_pp_decl_param () =
  let v : var =
    {var_name = "x"; var_id = 0; var_type = TInt32; var_mutable = false}
  in
  let d = DParam (v, None) in
  let s = pp_to_string pp_decl d in
  assert (s = "int32 x") ;
  print_endline "  pp_decl param: OK"

let test_pp_decl_param_array () =
  let v : var =
    {
      var_name = "arr";
      var_id = 0;
      var_type = TVec TFloat32;
      var_mutable = false;
    }
  in
  let d = DParam (v, Some {arr_elttype = TFloat32; arr_memspace = Global}) in
  let s = pp_to_string pp_decl d in
  assert (s = "global float32* arr") ;
  print_endline "  pp_decl param array: OK"

let test_pp_decl_local () =
  let v : var =
    {var_name = "sum"; var_id = 0; var_type = TFloat32; var_mutable = true}
  in
  let d = DLocal (v, Some (EConst (CFloat32 0.0))) in
  let s = pp_to_string pp_decl d in
  assert (String.sub s 0 11 = "float32 sum") ;
  print_endline "  pp_decl local: OK"

let test_pp_decl_shared () =
  let d = DShared ("cache", TFloat32, Some (EConst (CInt32 256l))) in
  let s = pp_to_string pp_decl d in
  assert (String.sub s 0 9 = "__shared_") ;
  print_endline "  pp_decl shared: OK"

(** {1 pp_kernel Tests} *)

let test_pp_kernel () =
  let input_var : var =
    {
      var_name = "input";
      var_id = 0;
      var_type = TVec TFloat32;
      var_mutable = false;
    }
  in
  let n_var : var =
    {var_name = "n"; var_id = 1; var_type = TInt32; var_mutable = false}
  in
  let k : kernel =
    {
      kern_name = "simple_kernel";
      kern_params =
        [
          DParam
            (input_var, Some {arr_elttype = TFloat32; arr_memspace = Global});
          DParam (n_var, None);
        ];
      kern_locals = [];
      kern_body = SReturn (EConst CUnit);
      kern_types = [];
      kern_variants = [];
      kern_funcs = [];
      kern_native_fn = None;
    }
  in
  let s = pp_to_string pp_kernel k in
  assert (String.length s > 0) ;
  assert (String.sub s 0 7 = "__kerne") ;
  print_endline "  pp_kernel: OK"

(** {1 pp_pattern Tests} *)

let test_pp_pattern () =
  let p1 = PConstr ("Some", ["x"]) in
  let s1 = pp_to_string pp_pattern p1 in
  assert (s1 = "Some(x)") ;

  let p2 = PConstr ("None", []) in
  let s2 = pp_to_string pp_pattern p2 in
  assert (s2 = "None") ;

  let p3 = PWild in
  let s3 = pp_to_string pp_pattern p3 in
  assert (s3 = "_") ;

  print_endline "  pp_pattern: OK"

(** {1 Main} *)

let () =
  print_endline "Sarek_ir_pp tests:" ;
  test_string_of_elttype_primitives () ;
  test_string_of_elttype_record () ;
  test_string_of_elttype_variant () ;
  test_string_of_elttype_array () ;
  test_string_of_elttype_vec () ;
  test_string_of_memspace () ;
  test_string_of_binop () ;
  test_string_of_unop () ;
  test_pp_expr_const () ;
  test_pp_expr_var () ;
  test_pp_expr_binop () ;
  test_pp_expr_unop () ;
  test_pp_expr_array_read () ;
  test_pp_expr_intrinsic () ;
  test_pp_expr_record () ;
  test_pp_expr_if () ;
  test_pp_stmt_assign () ;
  test_pp_stmt_barrier () ;
  test_pp_stmt_warp_barrier () ;
  test_pp_stmt_mem_fence () ;
  test_pp_stmt_return () ;
  test_pp_stmt_empty () ;
  test_pp_decl_param () ;
  test_pp_decl_param_array () ;
  test_pp_decl_local () ;
  test_pp_decl_shared () ;
  test_pp_kernel () ;
  test_pp_pattern () ;
  print_endline "All Sarek_ir_pp tests passed!"
