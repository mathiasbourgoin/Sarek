(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Unit tests for Sarek_ir_analysis
 *
 * Tests float64 detection functions across all IR constructs.
 ******************************************************************************)

open Sarek_ir_types
open Sarek_ir_analysis

(** {1 elttype_uses_float64 Tests} *)

let test_elttype_float64 () =
  assert (elttype_uses_float64 TFloat64 = true) ;
  assert (elttype_uses_float64 TFloat32 = false) ;
  assert (elttype_uses_float64 TInt32 = false) ;
  assert (elttype_uses_float64 TInt64 = false) ;
  assert (elttype_uses_float64 TBool = false) ;
  assert (elttype_uses_float64 TUnit = false) ;
  print_endline "  elttype_uses_float64 primitives: OK"

let test_elttype_record_float64 () =
  let rec_no_f64 = TRecord ("point2d", [("x", TFloat32); ("y", TFloat32)]) in
  assert (elttype_uses_float64 rec_no_f64 = false) ;

  let rec_with_f64 =
    TRecord ("point2d_f64", [("x", TFloat64); ("y", TFloat64)])
  in
  assert (elttype_uses_float64 rec_with_f64 = true) ;

  let rec_mixed = TRecord ("mixed", [("a", TInt32); ("b", TFloat64)]) in
  assert (elttype_uses_float64 rec_mixed = true) ;

  print_endline "  elttype_uses_float64 record: OK"

let test_elttype_variant_float64 () =
  let var_no_f64 =
    TVariant ("option_int", [("None", []); ("Some", [TInt32])])
  in
  assert (elttype_uses_float64 var_no_f64 = false) ;

  let var_with_f64 =
    TVariant ("number", [("Int", [TInt64]); ("Float", [TFloat64])])
  in
  assert (elttype_uses_float64 var_with_f64 = true) ;

  print_endline "  elttype_uses_float64 variant: OK"

let test_elttype_array_float64 () =
  let arr_no_f64 = TArray (TFloat32, Global) in
  assert (elttype_uses_float64 arr_no_f64 = false) ;

  let arr_with_f64 = TArray (TFloat64, Shared) in
  assert (elttype_uses_float64 arr_with_f64 = true) ;

  print_endline "  elttype_uses_float64 array: OK"

let test_elttype_vec_float64 () =
  let vec_no_f64 = TVec TFloat32 in
  assert (elttype_uses_float64 vec_no_f64 = false) ;

  let vec_with_f64 = TVec TFloat64 in
  assert (elttype_uses_float64 vec_with_f64 = true) ;

  print_endline "  elttype_uses_float64 vec: OK"

(** {1 const_uses_float64 Tests} *)

let test_const_float64 () =
  assert (const_uses_float64 (CFloat64 1.0) = true) ;
  assert (const_uses_float64 (CFloat32 1.0) = false) ;
  assert (const_uses_float64 (CInt32 1l) = false) ;
  assert (const_uses_float64 (CInt64 1L) = false) ;
  assert (const_uses_float64 (CBool true) = false) ;
  assert (const_uses_float64 CUnit = false) ;
  print_endline "  const_uses_float64: OK"

(** {1 expr_uses_float64 Tests} *)

let test_expr_const_float64 () =
  assert (expr_uses_float64 (EConst (CFloat64 1.0)) = true) ;
  assert (expr_uses_float64 (EConst (CFloat32 1.0)) = false) ;
  print_endline "  expr_uses_float64 const: OK"

let test_expr_var_float64 () =
  let v_f64 : var =
    {var_name = "x"; var_id = 0; var_type = TFloat64; var_mutable = false}
  in
  let v_f32 : var =
    {var_name = "y"; var_id = 1; var_type = TFloat32; var_mutable = false}
  in
  assert (expr_uses_float64 (EVar v_f64) = true) ;
  assert (expr_uses_float64 (EVar v_f32) = false) ;
  print_endline "  expr_uses_float64 var: OK"

let test_expr_binop_float64 () =
  let e_f64 = EConst (CFloat64 1.0) in
  let e_f32 = EConst (CFloat32 1.0) in

  assert (expr_uses_float64 (EBinop (Add, e_f64, e_f32)) = true) ;
  assert (expr_uses_float64 (EBinop (Add, e_f32, e_f64)) = true) ;
  assert (expr_uses_float64 (EBinop (Add, e_f32, e_f32)) = false) ;

  print_endline "  expr_uses_float64 binop: OK"

let test_expr_cast_float64 () =
  let e = EConst (CFloat32 1.0) in
  assert (expr_uses_float64 (ECast (TFloat64, e)) = true) ;
  assert (expr_uses_float64 (ECast (TFloat32, e)) = false) ;
  print_endline "  expr_uses_float64 cast: OK"

let test_expr_intrinsic_float64 () =
  let e_f64 = EConst (CFloat64 1.0) in
  let e_f32 = EConst (CFloat32 1.0) in

  assert (expr_uses_float64 (EIntrinsic (["Float64"], "sin", [e_f64])) = true) ;
  assert (expr_uses_float64 (EIntrinsic (["Float32"], "sin", [e_f32])) = false) ;

  print_endline "  expr_uses_float64 intrinsic: OK"

let test_expr_if_float64 () =
  let cond = EConst (CBool true) in
  let e_f64 = EConst (CFloat64 1.0) in
  let e_f32 = EConst (CFloat32 1.0) in

  assert (expr_uses_float64 (EIf (cond, e_f64, e_f32)) = true) ;
  assert (expr_uses_float64 (EIf (cond, e_f32, e_f64)) = true) ;
  assert (expr_uses_float64 (EIf (cond, e_f32, e_f32)) = false) ;

  print_endline "  expr_uses_float64 if: OK"

(** {1 stmt_uses_float64 Tests} *)

let test_stmt_assign_float64 () =
  let v : var =
    {var_name = "x"; var_id = 0; var_type = TFloat32; var_mutable = true}
  in
  let e_f64 = EConst (CFloat64 1.0) in
  let e_f32 = EConst (CFloat32 1.0) in

  assert (stmt_uses_float64 (SAssign (LVar v, e_f64)) = true) ;
  assert (stmt_uses_float64 (SAssign (LVar v, e_f32)) = false) ;

  print_endline "  stmt_uses_float64 assign: OK"

let test_stmt_seq_float64 () =
  let s_f64 = SExpr (EConst (CFloat64 1.0)) in
  let s_f32 = SExpr (EConst (CFloat32 1.0)) in

  assert (stmt_uses_float64 (SSeq [s_f64; s_f32]) = true) ;
  assert (stmt_uses_float64 (SSeq [s_f32; s_f32]) = false) ;

  print_endline "  stmt_uses_float64 seq: OK"

let test_stmt_if_float64 () =
  let cond = EConst (CBool true) in
  let s_f64 = SExpr (EConst (CFloat64 1.0)) in
  let s_f32 = SExpr (EConst (CFloat32 1.0)) in

  assert (stmt_uses_float64 (SIf (cond, s_f64, None)) = true) ;
  assert (stmt_uses_float64 (SIf (cond, s_f32, Some s_f64)) = true) ;
  assert (stmt_uses_float64 (SIf (cond, s_f32, Some s_f32)) = false) ;

  print_endline "  stmt_uses_float64 if: OK"

let test_stmt_for_float64 () =
  let v_f64 : var =
    {var_name = "i"; var_id = 0; var_type = TFloat64; var_mutable = true}
  in
  let v_i32 : var =
    {var_name = "i"; var_id = 0; var_type = TInt32; var_mutable = true}
  in
  let lo = EConst (CInt32 0l) in
  let hi = EConst (CInt32 10l) in

  assert (stmt_uses_float64 (SFor (v_f64, lo, hi, Upto, SEmpty)) = true) ;
  assert (stmt_uses_float64 (SFor (v_i32, lo, hi, Upto, SEmpty)) = false) ;

  print_endline "  stmt_uses_float64 for: OK"

let test_stmt_let_float64 () =
  let v_f64 : var =
    {var_name = "x"; var_id = 0; var_type = TFloat64; var_mutable = false}
  in
  let v_f32 : var =
    {var_name = "x"; var_id = 0; var_type = TFloat32; var_mutable = false}
  in
  let e_f32 = EConst (CFloat32 1.0) in

  assert (stmt_uses_float64 (SLet (v_f64, e_f32, SEmpty)) = true) ;
  assert (stmt_uses_float64 (SLet (v_f32, e_f32, SEmpty)) = false) ;

  print_endline "  stmt_uses_float64 let: OK"

let test_stmt_barrier_float64 () =
  assert (stmt_uses_float64 SBarrier = false) ;
  assert (stmt_uses_float64 SWarpBarrier = false) ;
  assert (stmt_uses_float64 SMemFence = false) ;
  assert (stmt_uses_float64 SEmpty = false) ;
  print_endline "  stmt_uses_float64 barrier/empty: OK"

(** {1 decl_uses_float64 Tests} *)

let test_decl_param_float64 () =
  let v_f64 : var =
    {var_name = "x"; var_id = 0; var_type = TFloat64; var_mutable = false}
  in
  let v_f32 : var =
    {var_name = "x"; var_id = 0; var_type = TFloat32; var_mutable = false}
  in

  assert (decl_uses_float64 (DParam (v_f64, None)) = true) ;
  assert (decl_uses_float64 (DParam (v_f32, None)) = false) ;

  let arr_info_f64 = Some {arr_elttype = TFloat64; arr_memspace = Global} in
  let arr_info_f32 = Some {arr_elttype = TFloat32; arr_memspace = Global} in

  assert (decl_uses_float64 (DParam (v_f32, arr_info_f64)) = true) ;
  assert (decl_uses_float64 (DParam (v_f32, arr_info_f32)) = false) ;

  print_endline "  decl_uses_float64 param: OK"

let test_decl_local_float64 () =
  let v_f64 : var =
    {var_name = "x"; var_id = 0; var_type = TFloat64; var_mutable = true}
  in
  let v_f32 : var =
    {var_name = "x"; var_id = 0; var_type = TFloat32; var_mutable = true}
  in
  let e_f64 = EConst (CFloat64 0.0) in
  let e_f32 = EConst (CFloat32 0.0) in

  assert (decl_uses_float64 (DLocal (v_f64, None)) = true) ;
  assert (decl_uses_float64 (DLocal (v_f32, Some e_f64)) = true) ;
  assert (decl_uses_float64 (DLocal (v_f32, Some e_f32)) = false) ;

  print_endline "  decl_uses_float64 local: OK"

let test_decl_shared_float64 () =
  assert (decl_uses_float64 (DShared ("cache", TFloat64, None)) = true) ;
  assert (decl_uses_float64 (DShared ("cache", TFloat32, None)) = false) ;
  print_endline "  decl_uses_float64 shared: OK"

(** {1 helper_uses_float64 Tests} *)

let test_helper_uses_float64 () =
  let param_f32 : var =
    {var_name = "x"; var_id = 0; var_type = TFloat32; var_mutable = false}
  in
  let param_f64 : var =
    {var_name = "x"; var_id = 0; var_type = TFloat64; var_mutable = false}
  in

  let hf_ret_f64 : helper_func =
    {
      hf_name = "f";
      hf_params = [param_f32];
      hf_ret_type = TFloat64;
      hf_body = SReturn (ECast (TFloat64, EVar param_f32));
    }
  in
  assert (helper_uses_float64 hf_ret_f64 = true) ;

  let hf_param_f64 : helper_func =
    {
      hf_name = "f";
      hf_params = [param_f64];
      hf_ret_type = TFloat32;
      hf_body = SReturn (ECast (TFloat32, EVar param_f64));
    }
  in
  assert (helper_uses_float64 hf_param_f64 = true) ;

  let hf_no_f64 : helper_func =
    {
      hf_name = "f";
      hf_params = [param_f32];
      hf_ret_type = TFloat32;
      hf_body = SReturn (EVar param_f32);
    }
  in
  assert (helper_uses_float64 hf_no_f64 = false) ;

  print_endline "  helper_uses_float64: OK"

(** {1 kernel_uses_float64 Tests} *)

let test_kernel_uses_float64_params () =
  let v_f64 : var =
    {var_name = "x"; var_id = 0; var_type = TVec TFloat64; var_mutable = false}
  in
  let v_f32 : var =
    {var_name = "x"; var_id = 0; var_type = TVec TFloat32; var_mutable = false}
  in

  let k_f64 : kernel =
    {
      kern_name = "test";
      kern_params =
        [DParam (v_f64, Some {arr_elttype = TFloat64; arr_memspace = Global})];
      kern_locals = [];
      kern_body = SEmpty;
      kern_types = [];
      kern_variants = [];
      kern_funcs = [];
      kern_native_fn = None;
    }
  in
  assert (kernel_uses_float64 k_f64 = true) ;

  let k_f32 : kernel =
    {
      kern_name = "test";
      kern_params =
        [DParam (v_f32, Some {arr_elttype = TFloat32; arr_memspace = Global})];
      kern_locals = [];
      kern_body = SEmpty;
      kern_types = [];
      kern_variants = [];
      kern_funcs = [];
      kern_native_fn = None;
    }
  in
  assert (kernel_uses_float64 k_f32 = false) ;

  print_endline "  kernel_uses_float64 params: OK"

let test_kernel_uses_float64_types () =
  let k_with_f64_type : kernel =
    {
      kern_name = "test";
      kern_params = [];
      kern_locals = [];
      kern_body = SEmpty;
      kern_types = [("point", [("x", TFloat64); ("y", TFloat64)])];
      kern_variants = [];
      kern_funcs = [];
      kern_native_fn = None;
    }
  in
  assert (kernel_uses_float64 k_with_f64_type = true) ;

  let k_with_f32_type : kernel =
    {
      kern_name = "test";
      kern_params = [];
      kern_locals = [];
      kern_body = SEmpty;
      kern_types = [("point", [("x", TFloat32); ("y", TFloat32)])];
      kern_variants = [];
      kern_funcs = [];
      kern_native_fn = None;
    }
  in
  assert (kernel_uses_float64 k_with_f32_type = false) ;

  print_endline "  kernel_uses_float64 types: OK"

let test_kernel_uses_float64_variants () =
  let k_with_f64_variant : kernel =
    {
      kern_name = "test";
      kern_params = [];
      kern_locals = [];
      kern_body = SEmpty;
      kern_types = [];
      kern_variants = [("number", [("Float", [TFloat64])])];
      kern_funcs = [];
      kern_native_fn = None;
    }
  in
  assert (kernel_uses_float64 k_with_f64_variant = true) ;

  print_endline "  kernel_uses_float64 variants: OK"

(** {1 Main} *)

let () =
  print_endline "Sarek_ir_analysis tests:" ;
  test_elttype_float64 () ;
  test_elttype_record_float64 () ;
  test_elttype_variant_float64 () ;
  test_elttype_array_float64 () ;
  test_elttype_vec_float64 () ;
  test_const_float64 () ;
  test_expr_const_float64 () ;
  test_expr_var_float64 () ;
  test_expr_binop_float64 () ;
  test_expr_cast_float64 () ;
  test_expr_intrinsic_float64 () ;
  test_expr_if_float64 () ;
  test_stmt_assign_float64 () ;
  test_stmt_seq_float64 () ;
  test_stmt_if_float64 () ;
  test_stmt_for_float64 () ;
  test_stmt_let_float64 () ;
  test_stmt_barrier_float64 () ;
  test_decl_param_float64 () ;
  test_decl_local_float64 () ;
  test_decl_shared_float64 () ;
  test_helper_uses_float64 () ;
  test_kernel_uses_float64_params () ;
  test_kernel_uses_float64_types () ;
  test_kernel_uses_float64_variants () ;
  print_endline "All Sarek_ir_analysis tests passed!"
