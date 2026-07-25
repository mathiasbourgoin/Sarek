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

(** {1 is_atomic_intrinsic_name / expr_uses_atomics Tests} *)

let test_is_atomic_intrinsic_name () =
  assert (is_atomic_intrinsic_name "atomic_add_int32" = true) ;
  assert (is_atomic_intrinsic_name "atomic_cas_int32" = true) ;
  assert (is_atomic_intrinsic_name "atomic_add_global_int32" = true) ;
  assert (is_atomic_intrinsic_name "thread_idx_x" = false) ;
  assert (is_atomic_intrinsic_name "block_barrier" = false) ;
  assert (is_atomic_intrinsic_name "" = false) ;
  print_endline "  is_atomic_intrinsic_name: OK"

let test_expr_uses_atomics () =
  let non_atomic = EIntrinsic (["Gpu"], "thread_idx_x", []) in
  assert (expr_uses_atomics non_atomic = false) ;
  let direct_atomic =
    EIntrinsic
      ( [],
        "atomic_add_int32",
        [
          EVar
            {
              var_name = "arr";
              var_id = 0;
              var_type = TInt32;
              var_mutable = true;
            };
          EConst (CInt32 0l);
          EConst (CInt32 1l);
        ] )
  in
  assert (expr_uses_atomics direct_atomic = true) ;
  (* Nested: atomic call buried inside an otherwise ordinary expression *)
  let nested_atomic = EBinop (Add, EConst (CInt32 1l), direct_atomic) in
  assert (expr_uses_atomics nested_atomic = true) ;
  print_endline "  expr_uses_atomics: OK"

(** {1 helper_uses_atomics / kernel_uses_atomics Tests} *)

let test_helper_uses_atomics () =
  let param : var =
    {var_name = "x"; var_id = 0; var_type = TInt32; var_mutable = false}
  in
  let hf_atomic : helper_func =
    {
      hf_name = "bump";
      hf_params = [param];
      hf_ret_type = TInt32;
      hf_body =
        SReturn
          (EIntrinsic ([], "atomic_add_int32", [EVar param; EConst (CInt32 1l)]));
    }
  in
  assert (helper_uses_atomics hf_atomic = true) ;

  let hf_no_atomic : helper_func =
    {
      hf_name = "identity";
      hf_params = [param];
      hf_ret_type = TInt32;
      hf_body = SReturn (EVar param);
    }
  in
  assert (helper_uses_atomics hf_no_atomic = false) ;
  print_endline "  helper_uses_atomics: OK"

let test_kernel_uses_atomics_direct () =
  let k_direct : kernel =
    {
      kern_name = "test";
      kern_params = [];
      kern_locals = [];
      kern_body =
        SExpr
          (EIntrinsic
             ([], "atomic_add_int32", [EConst (CInt32 0l); EConst (CInt32 1l)]));
      kern_types = [];
      kern_variants = [];
      kern_funcs = [];
      kern_native_fn = None;
    }
  in
  assert (kernel_uses_atomics k_direct = true) ;

  let k_none : kernel =
    {
      kern_name = "test";
      kern_params = [];
      kern_locals = [];
      kern_body = SEmpty;
      kern_types = [];
      kern_variants = [];
      kern_funcs = [];
      kern_native_fn = None;
    }
  in
  assert (kernel_uses_atomics k_none = false) ;
  print_endline "  kernel_uses_atomics direct: OK"

let test_kernel_uses_atomics_in_helper () =
  (* Load-bearing case: the atomic is only reachable through kern_funcs, not
     kern_body. A body-only walk would wrongly report false. *)
  let param : var =
    {var_name = "x"; var_id = 0; var_type = TInt32; var_mutable = false}
  in
  let hf_atomic : helper_func =
    {
      hf_name = "bump";
      hf_params = [param];
      hf_ret_type = TInt32;
      hf_body =
        SReturn
          (EIntrinsic ([], "atomic_add_int32", [EVar param; EConst (CInt32 1l)]));
    }
  in
  let k_helper_atomic : kernel =
    {
      kern_name = "test";
      kern_params = [];
      kern_locals = [];
      kern_body = SEmpty;
      kern_types = [];
      kern_variants = [];
      kern_funcs = [hf_atomic];
      kern_native_fn = None;
    }
  in
  assert (kernel_uses_atomics k_helper_atomic = true) ;
  print_endline "  kernel_uses_atomics via helper: OK"

(** {1 lvalue_uses_atomics / SAssign lvalue Tests (finding 3)} *)

let test_lvalue_uses_atomics () =
  let atomic_idx =
    EIntrinsic ([], "atomic_add_int32", [EConst (CInt32 0l); EConst (CInt32 1l)])
  in
  assert (
    lvalue_uses_atomics
      (LVar {var_name = "x"; var_id = 0; var_type = TInt32; var_mutable = true})
    = false) ;
  assert (lvalue_uses_atomics (LArrayElem ("arr", EConst (CInt32 0l))) = false) ;
  assert (lvalue_uses_atomics (LArrayElem ("arr", atomic_idx)) = true) ;
  assert (
    lvalue_uses_atomics (LArrayElemExpr (EConst (CInt32 0l), atomic_idx)) = true) ;
  assert (
    lvalue_uses_atomics (LArrayElemExpr (atomic_idx, EConst (CInt32 0l))) = true) ;
  assert (
    lvalue_uses_atomics (LRecordField (LArrayElem ("arr", atomic_idx), "field"))
    = true) ;
  print_endline "  lvalue_uses_atomics: OK"

(** Load-bearing case for finding 3: the only atomic in the kernel is inside an
    [SAssign]'s lvalue index expression (e.g. [arr.(atomic_add ...) <- 5]), not
    in the RHS. Pre-fix, [stmt_uses_atomics]'s [SAssign] case ignored the lvalue
    entirely and only walked the RHS, so this wrongly returned false. *)
let test_kernel_uses_atomics_in_assign_lvalue () =
  let atomic_idx =
    EIntrinsic ([], "atomic_add_int32", [EConst (CInt32 0l); EConst (CInt32 1l)])
  in
  let k_lvalue_atomic : kernel =
    {
      kern_name = "test";
      kern_params = [];
      kern_locals = [];
      kern_body = SAssign (LArrayElem ("arr", atomic_idx), EConst (CInt32 5l));
      kern_types = [];
      kern_variants = [];
      kern_funcs = [];
      kern_native_fn = None;
    }
  in
  assert (kernel_uses_atomics k_lvalue_atomic = true) ;
  print_endline "  kernel_uses_atomics via assign lvalue: OK"

(** {1 SNative conservative-atomics Tests (finding 4)} *)

let dummy_native_gpu ~framework:_ = ""

let dummy_native_ocaml : ocaml_closure =
  {run = (fun ~block:_ ~grid:_ _args -> ())}

let test_kernel_uses_atomics_snative () =
  (* SNative is opaque inline GPU code; stmt_uses_atomics must treat it
     conservatively as atomic-bearing rather than hard-coding "no atomics
     here". Pre-fix this returned false unconditionally. *)
  let k_native : kernel =
    {
      kern_name = "test";
      kern_params = [];
      kern_locals = [];
      kern_body = SNative {gpu = dummy_native_gpu; ocaml = dummy_native_ocaml};
      kern_types = [];
      kern_variants = [];
      kern_funcs = [];
      kern_native_fn = None;
    }
  in
  assert (kernel_uses_atomics k_native = true) ;
  print_endline "  kernel_uses_atomics via SNative (conservative): OK"

(** {1 expr_uses_int_mod / kernel_uses_int_mod Tests} *)

let test_expr_uses_int_mod () =
  let no_mod = EBinop (Add, EConst (CInt32 1l), EConst (CInt32 2l)) in
  assert (expr_uses_int_mod no_mod = false) ;
  let direct_mod = EBinop (Mod, EConst (CInt32 7l), EConst (CInt32 2l)) in
  assert (expr_uses_int_mod direct_mod = true) ;
  (* Nested: [mod] buried inside an otherwise ordinary expression. *)
  let nested_mod = EBinop (Add, EConst (CInt32 1l), direct_mod) in
  assert (expr_uses_int_mod nested_mod = true) ;
  print_endline "  expr_uses_int_mod: OK"

(** Regression pin (finding: [lvalue_uses_int_mod] must recurse through
    [LRecordField]). [LRecordField] wraps a NESTED lvalue whose array index can
    carry a [mod], e.g. [arr.(j mod n).field <- v]. A non-recursive arm returned
    false, so the GLSL backend would emit a [sarek_smod(...)] call from the
    lvalue index while [kernel_uses_int_mod] wrongly reported false and skipped
    emitting the helper — an undefined-function shader compile failure. *)
let test_lvalue_uses_int_mod () =
  let mod_idx = EBinop (Mod, EConst (CInt32 7l), EConst (CInt32 2l)) in
  assert (
    lvalue_uses_int_mod
      (LVar {var_name = "x"; var_id = 0; var_type = TInt32; var_mutable = true})
    = false) ;
  assert (lvalue_uses_int_mod (LArrayElem ("arr", EConst (CInt32 0l))) = false) ;
  assert (lvalue_uses_int_mod (LArrayElem ("arr", mod_idx)) = true) ;
  assert (
    lvalue_uses_int_mod (LArrayElemExpr (EConst (CInt32 0l), mod_idx)) = true) ;
  (* The load-bearing shape: [mod] only inside an LRecordField-wrapped index. *)
  assert (
    lvalue_uses_int_mod (LRecordField (LArrayElem ("arr", mod_idx), "field"))
    = true) ;
  print_endline "  lvalue_uses_int_mod: OK"

(** The only [mod] in the kernel is inside an [SAssign] lvalue whose index is
    wrapped in [LRecordField] ([arr.(j mod n).field <- v]) — the exact shape
    that a non-recursive [lvalue_uses_int_mod] would miss. *)
let test_kernel_uses_int_mod_in_record_field_lvalue () =
  let mod_idx = EBinop (Mod, EConst (CInt32 7l), EConst (CInt32 2l)) in
  let k_lvalue_mod : kernel =
    {
      kern_name = "test";
      kern_params = [];
      kern_locals = [];
      kern_body =
        SAssign
          ( LRecordField (LArrayElem ("arr", mod_idx), "field"),
            EConst (CInt32 5l) );
      kern_types = [];
      kern_variants = [];
      kern_funcs = [];
      kern_native_fn = None;
    }
  in
  assert (kernel_uses_int_mod k_lvalue_mod = true) ;
  let k_none : kernel =
    {
      kern_name = "test";
      kern_params = [];
      kern_locals = [];
      kern_body = SEmpty;
      kern_types = [];
      kern_variants = [];
      kern_funcs = [];
      kern_native_fn = None;
    }
  in
  assert (kernel_uses_int_mod k_none = false) ;
  print_endline "  kernel_uses_int_mod via record-field lvalue: OK"

(** {1 expr_uses_copysign / kernel_uses_copysign Tests} *)

let test_is_copysign_intrinsic_name () =
  assert (is_copysign_intrinsic_name "copysign" = true) ;
  assert (is_copysign_intrinsic_name "copysignf" = false) ;
  assert (is_copysign_intrinsic_name "sin" = false) ;
  assert (is_copysign_intrinsic_name "" = false) ;
  print_endline "  is_copysign_intrinsic_name: OK"

let test_expr_uses_copysign () =
  let non_copysign =
    EIntrinsic (["Float64"], "hypot", [EConst (CFloat64 1.0)])
  in
  assert (expr_uses_copysign non_copysign = false) ;
  let direct_copysign =
    EIntrinsic
      ( ["Float64"],
        "copysign",
        [EConst (CFloat64 1.0); EConst (CFloat64 (-2.0))] )
  in
  assert (expr_uses_copysign direct_copysign = true) ;
  (* Float32-qualified copysign is detected too (name-based, path-agnostic). *)
  let f32_copysign =
    EIntrinsic
      ( ["Float32"],
        "copysign",
        [EConst (CFloat32 1.0); EConst (CFloat32 (-2.0))] )
  in
  assert (expr_uses_copysign f32_copysign = true) ;
  (* Nested: copysign buried inside an otherwise ordinary expression. *)
  let nested = EBinop (Add, EConst (CFloat64 1.0), direct_copysign) in
  assert (expr_uses_copysign nested = true) ;
  print_endline "  expr_uses_copysign: OK"

(** Regression pin (round-3 LRecordField lesson): [lvalue_uses_copysign] must
    recurse through [LRecordField]. A non-recursive arm would return false when
    a copysign result only appears inside an [LRecordField]-wrapped index, so
    the GLSL backend would emit a [sarek_copysign(...)] call from the lvalue
    index while [kernel_uses_copysign] wrongly reported false and skipped
    emitting the helper — an undefined-function shader compile failure. *)
let test_lvalue_uses_copysign () =
  let cs_idx =
    EIntrinsic
      ( [],
        "int_of_float",
        [
          EIntrinsic
            ( ["Float64"],
              "copysign",
              [EConst (CFloat64 1.0); EConst (CFloat64 (-2.0))] );
        ] )
  in
  assert (
    lvalue_uses_copysign
      (LVar {var_name = "x"; var_id = 0; var_type = TInt32; var_mutable = true})
    = false) ;
  assert (lvalue_uses_copysign (LArrayElem ("arr", EConst (CInt32 0l))) = false) ;
  assert (lvalue_uses_copysign (LArrayElem ("arr", cs_idx)) = true) ;
  assert (
    lvalue_uses_copysign (LArrayElemExpr (EConst (CInt32 0l), cs_idx)) = true) ;
  (* The load-bearing shape: copysign only inside an LRecordField-wrapped index. *)
  assert (
    lvalue_uses_copysign (LRecordField (LArrayElem ("arr", cs_idx), "field"))
    = true) ;
  print_endline "  lvalue_uses_copysign: OK"

let test_kernel_uses_copysign_in_record_field_lvalue () =
  let cs_idx =
    EIntrinsic
      ( [],
        "int_of_float",
        [
          EIntrinsic
            ( ["Float64"],
              "copysign",
              [EConst (CFloat64 1.0); EConst (CFloat64 (-2.0))] );
        ] )
  in
  let k_lvalue_cs : kernel =
    {
      kern_name = "test";
      kern_params = [];
      kern_locals = [];
      kern_body =
        SAssign
          ( LRecordField (LArrayElem ("arr", cs_idx), "field"),
            EConst (CInt32 5l) );
      kern_types = [];
      kern_variants = [];
      kern_funcs = [];
      kern_native_fn = None;
    }
  in
  assert (kernel_uses_copysign k_lvalue_cs = true) ;
  let k_none : kernel =
    {
      kern_name = "test";
      kern_params = [];
      kern_locals = [];
      kern_body = SEmpty;
      kern_types = [];
      kern_variants = [];
      kern_funcs = [];
      kern_native_fn = None;
    }
  in
  assert (kernel_uses_copysign k_none = false) ;
  print_endline "  kernel_uses_copysign via record-field lvalue: OK"

(** Load-bearing case: the only copysign is reachable through [kern_funcs], not
    [kern_body]. A body-only walk would wrongly report false. *)
let test_kernel_uses_copysign_in_helper () =
  let param : var =
    {var_name = "x"; var_id = 0; var_type = TFloat64; var_mutable = false}
  in
  let hf_cs : helper_func =
    {
      hf_name = "f";
      hf_params = [param];
      hf_ret_type = TFloat64;
      hf_body =
        SReturn
          (EIntrinsic
             (["Float64"], "copysign", [EVar param; EConst (CFloat64 (-1.0))]));
    }
  in
  let k : kernel =
    {
      kern_name = "test";
      kern_params = [];
      kern_locals = [];
      kern_body = SEmpty;
      kern_types = [];
      kern_variants = [];
      kern_funcs = [hf_cs];
      kern_native_fn = None;
    }
  in
  assert (kernel_uses_copysign k = true) ;
  print_endline "  kernel_uses_copysign via helper: OK"

(** {1 kernel_uses_nonfinite_float64 Tests}

    A [CFloat64] whose value is ±inf or NaN cannot be spelled as a GLSL literal,
    so the detector flags any non-finite f64 constant anywhere in the kernel. A
    {e finite} f64 constant must NOT trip it, and [SNative] is treated as
    non-finite-free (native code carries its own literals). *)

let empty_kernel body : kernel =
  {
    kern_name = "test";
    kern_params = [];
    kern_locals = [];
    kern_body = body;
    kern_types = [];
    kern_variants = [];
    kern_funcs = [];
    kern_native_fn = None;
  }

(** {1 Float16 detection Tests}

    Mirrors the float64 detector tests above. Every positive case is paired with
    a negative one: the whole point of the detector is that a NON-f16 kernel
    must not trigger the CUDA/HIP [cuda_fp16.h] include, so a detector that
    over-reports would silently change every existing golden. *)

let test_elttype_float16 () =
  assert (elttype_uses_float16 TFloat16 = true) ;
  (* f16 must not be confused with the other float widths, in either
     direction. *)
  assert (elttype_uses_float16 TFloat32 = false) ;
  assert (elttype_uses_float16 TFloat64 = false) ;
  assert (elttype_uses_float64 TFloat16 = false) ;
  assert (elttype_uses_float16 TInt32 = false) ;
  assert (elttype_uses_float16 TInt64 = false) ;
  assert (elttype_uses_float16 TBool = false) ;
  assert (elttype_uses_float16 TUnit = false) ;
  print_endline "  elttype_uses_float16 primitives: OK"

let test_elttype_nested_float16 () =
  (* Records, variants, arrays and vectors are searched recursively. *)
  assert (
    elttype_uses_float16 (TRecord ("r", [("x", TFloat32); ("y", TFloat16)]))
    = true) ;
  assert (
    elttype_uses_float16 (TRecord ("r", [("x", TFloat32); ("y", TFloat64)]))
    = false) ;
  assert (
    elttype_uses_float16 (TVariant ("v", [("A", [TInt32]); ("B", [TFloat16])]))
    = true) ;
  assert (
    elttype_uses_float16 (TVariant ("v", [("A", [TInt32]); ("B", [TFloat32])]))
    = false) ;
  assert (elttype_uses_float16 (TArray (TFloat16, Shared)) = true) ;
  assert (elttype_uses_float16 (TArray (TFloat32, Shared)) = false) ;
  assert (elttype_uses_float16 (TVec TFloat16) = true) ;
  assert (elttype_uses_float16 (TVec TFloat32) = false) ;
  (* Doubly nested: vector of records containing an f16 field. *)
  assert (elttype_uses_float16 (TVec (TRecord ("r", [("h", TFloat16)]))) = true) ;
  print_endline "  elttype_uses_float16 nested: OK"

let test_expr_uses_float16 () =
  (* An f16 value enters an expression through a CAST -- f16 has no literal, so
     unlike float64 there is no constant case to detect. *)
  assert (expr_uses_float16 (ECast (TFloat16, EConst (CFloat32 1.5))) = true) ;
  assert (expr_uses_float16 (ECast (TFloat32, EConst (CFloat32 1.5))) = false) ;
  (* ... or through an f16-typed variable. *)
  let v_f16 : var =
    {var_name = "h"; var_id = 0; var_type = TFloat16; var_mutable = false}
  in
  let v_f32 : var =
    {var_name = "f"; var_id = 1; var_type = TFloat32; var_mutable = false}
  in
  assert (expr_uses_float16 (EVar v_f16) = true) ;
  assert (expr_uses_float16 (EVar v_f32) = false) ;
  (* ... or through an f16 array construction. *)
  assert (
    expr_uses_float16 (EArrayCreate (TFloat16, EConst (CInt32 4l), Shared))
    = true) ;
  assert (
    expr_uses_float16 (EArrayCreate (TFloat32, EConst (CInt32 4l), Shared))
    = false) ;
  (* Buried in a sub-expression: the traversal must still find it. *)
  assert (
    expr_uses_float16
      (EBinop
         ( Add,
           EConst (CFloat32 1.0),
           EBinop (Mul, EConst (CFloat32 2.0), ECast (TFloat16, EVar v_f32)) ))
    = true) ;
  (* A structurally identical f32-only expression must NOT trigger. *)
  assert (
    expr_uses_float16
      (EBinop
         ( Add,
           EConst (CFloat32 1.0),
           EBinop (Mul, EConst (CFloat32 2.0), ECast (TFloat32, EVar v_f32)) ))
    = false) ;
  print_endline "  expr_uses_float16: OK"

let test_stmt_decl_uses_float16 () =
  let v_f16 : var =
    {var_name = "h"; var_id = 0; var_type = TFloat16; var_mutable = false}
  in
  let v_f32 : var =
    {var_name = "f"; var_id = 1; var_type = TFloat32; var_mutable = false}
  in
  (* Binder types are inspected (the [ft] hook of the shared folder). *)
  assert (stmt_uses_float16 (SLet (v_f16, EConst (CFloat32 0.0), SEmpty)) = true) ;
  assert (
    stmt_uses_float16 (SLet (v_f32, EConst (CFloat32 0.0), SEmpty)) = false) ;
  assert (stmt_uses_float16 SEmpty = false) ;
  (* SNative is treated as f16-free, matching the float64 detector: inline
     device text is opaque and owns its own feature declaration. *)
  assert (
    stmt_uses_float16
      (SNative {gpu = dummy_native_gpu; ocaml = dummy_native_ocaml})
    = false) ;
  (* Declarations: the parameter type AND the array element type. *)
  assert (decl_uses_float16 (DParam (v_f16, None)) = true) ;
  assert (decl_uses_float16 (DParam (v_f32, None)) = false) ;
  assert (
    decl_uses_float16
      (DParam (v_f32, Some {arr_elttype = TFloat16; arr_memspace = Global}))
    = true) ;
  assert (
    decl_uses_float16
      (DParam (v_f32, Some {arr_elttype = TFloat32; arr_memspace = Global}))
    = false) ;
  assert (decl_uses_float16 (DShared ("s", TFloat16, None)) = true) ;
  assert (decl_uses_float16 (DShared ("s", TFloat32, None)) = false) ;
  print_endline "  stmt/decl_uses_float16: OK"

let test_kernel_uses_float16 () =
  let v_f16 : var =
    {var_name = "x"; var_id = 0; var_type = TVec TFloat16; var_mutable = false}
  in
  let v_f32 : var =
    {var_name = "x"; var_id = 0; var_type = TVec TFloat32; var_mutable = false}
  in
  let kern_with params body : kernel =
    {
      kern_name = "test";
      kern_params = params;
      kern_locals = [];
      kern_body = body;
      kern_types = [];
      kern_variants = [];
      kern_funcs = [];
      kern_native_fn = None;
    }
  in
  (* Positive: an f16 vector parameter. This is the case that must switch the
     CUDA/HIP fp16 include on. *)
  assert (
    kernel_uses_float16
      (kern_with
         [DParam (v_f16, Some {arr_elttype = TFloat16; arr_memspace = Global})]
         SEmpty)
    = true) ;
  (* Negative: the same kernel with an f32 vector. *)
  assert (
    kernel_uses_float16
      (kern_with
         [DParam (v_f32, Some {arr_elttype = TFloat32; arr_memspace = Global})]
         SEmpty)
    = false) ;
  (* Positive: f16 only in the BODY, via a cast -- no f16 in the signature. *)
  assert (
    kernel_uses_float16
      (kern_with [] (SExpr (ECast (TFloat16, EConst (CFloat32 1.0)))))
    = true) ;
  (* Positive: f16 only inside a helper function. *)
  let helper : helper_func =
    {hf_name = "h"; hf_params = []; hf_ret_type = TFloat16; hf_body = SEmpty}
  in
  let k_helper = {(kern_with [] SEmpty) with kern_funcs = [helper]} in
  assert (kernel_uses_float16 k_helper = true) ;
  assert (helper_uses_float16 helper = true) ;
  (* Positive: f16 only in a kernel record type declaration. *)
  let k_types =
    {(kern_with [] SEmpty) with kern_types = [("r", [("h", TFloat16)])]}
  in
  assert (kernel_uses_float16 k_types = true) ;
  (* Negative: an entirely f16-free kernel, including an f64 one -- f64 must not
     be mistaken for f16. *)
  let v_f64 : var =
    {var_name = "x"; var_id = 0; var_type = TVec TFloat64; var_mutable = false}
  in
  let k_f64 =
    kern_with
      [DParam (v_f64, Some {arr_elttype = TFloat64; arr_memspace = Global})]
      (SExpr (EConst (CFloat64 1.0)))
  in
  assert (kernel_uses_float16 k_f64 = false) ;
  assert (kernel_uses_float64 k_f64 = true) ;
  assert (kernel_uses_float16 (kern_with [] SEmpty) = false) ;
  print_endline "  kernel_uses_float16: OK"

(** {1 Parameterised feature API}

    The per-width [*_uses_float64] / [*_uses_float16] names above are thin
    aliases over ONE parameterised family. These assertions pin the family
    itself, so a future width (bf16) is covered by construction rather than by
    another 200 lines of copied assertions. *)

let feature_kern params body : kernel =
  {
    kern_name = "test";
    kern_params = params;
    kern_locals = [];
    kern_body = body;
    kern_types = [];
    kern_variants = [];
    kern_funcs = [];
    kern_native_fn = None;
  }

let test_feature_api_agrees_with_aliases () =
  let v_f16 : var =
    {var_name = "h"; var_id = 0; var_type = TFloat16; var_mutable = false}
  in
  let v_f64 : var =
    {var_name = "d"; var_id = 1; var_type = TFloat64; var_mutable = false}
  in
  let k16 = feature_kern [DParam (v_f16, None)] SEmpty in
  let k64 = feature_kern [DParam (v_f64, None)] SEmpty in
  let k32 = feature_kern [] SEmpty in
  (* The alias is exactly the parameterised call, at every level. *)
  assert (kernel_uses Float16 k16 = kernel_uses_float16 k16) ;
  assert (kernel_uses Float64 k64 = kernel_uses_float64 k64) ;
  assert (elttype_uses Float16 TFloat16 = elttype_uses_float16 TFloat16) ;
  assert (elttype_uses Float64 TFloat64 = elttype_uses_float64 TFloat64) ;
  assert (const_uses Float64 (CFloat64 1.0) = const_uses_float64 (CFloat64 1.0)) ;
  assert (
    expr_uses Float16 (ECast (TFloat16, EConst (CFloat32 1.0)))
    = expr_uses_float16 (ECast (TFloat16, EConst (CFloat32 1.0)))) ;
  assert (stmt_uses Float16 SEmpty = stmt_uses_float16 SEmpty) ;
  assert (
    decl_uses Float16 (DParam (v_f16, None))
    = decl_uses_float16 (DParam (v_f16, None))) ;
  (* Widths are orthogonal: neither detector sees the other's type. *)
  assert (kernel_uses Float64 k16 = false) ;
  assert (kernel_uses Float16 k64 = false) ;
  assert (kernel_uses Float16 k32 = false) ;
  assert (kernel_uses Float64 k32 = false) ;
  print_endline "  feature API agrees with the per-width aliases: OK"

let test_const_uses_is_width_specific () =
  (* The one deliberate per-width asymmetry: float64 has literals, f16 has none,
     so [const_uses Float16] is false for EVERY constant — by construction, not
     by a missing match arm. *)
  List.iter
    (fun c -> assert (const_uses Float16 c = false))
    [CFloat64 1.0; CFloat32 1.0; CInt32 1l; CInt64 1L; CBool true; CUnit] ;
  assert (const_uses Float64 (CFloat64 1.0) = true) ;
  assert (const_uses Float64 (CFloat32 1.0) = false) ;
  print_endline "  const_uses is width-specific: OK"

let test_kernel_requirements () =
  (* The set-valued form a future [Kernel.requirements] reduces to. *)
  let v_f16 : var =
    {var_name = "h"; var_id = 0; var_type = TVec TFloat16; var_mutable = false}
  in
  let v_f64 : var =
    {var_name = "d"; var_id = 1; var_type = TVec TFloat64; var_mutable = false}
  in
  assert (kernel_requirements (feature_kern [] SEmpty) = []) ;
  assert (
    kernel_requirements
      (feature_kern
         [DParam (v_f16, Some {arr_elttype = TFloat16; arr_memspace = Global})]
         SEmpty)
    = [Float16]) ;
  assert (
    kernel_requirements
      (feature_kern
         [DParam (v_f64, Some {arr_elttype = TFloat64; arr_memspace = Global})]
         SEmpty)
    = [Float64]) ;
  (* Both widths in one kernel, in declaration order of [all_features]. *)
  assert (
    kernel_requirements
      (feature_kern
         [
           DParam (v_f64, Some {arr_elttype = TFloat64; arr_memspace = Global});
           DParam (v_f16, Some {arr_elttype = TFloat16; arr_memspace = Global});
         ]
         SEmpty)
    = [Float64; Float16]) ;
  (* Every feature must be reachable from [all_features]; a new constructor that
     is not added there would silently never be required. *)
  assert (List.length all_features = 2) ;
  assert (List.map feature_name all_features = ["float64"; "float16"]) ;
  print_endline "  kernel_requirements: OK"

let test_kernel_uses_nonfinite_float64 () =
  (* Positive: +inf constant in the body. *)
  let k_inf = empty_kernel (SExpr (EConst (CFloat64 Float.infinity))) in
  assert (kernel_uses_nonfinite_float64 k_inf = true) ;
  (* Positive: NaN constant in the body. *)
  let k_nan = empty_kernel (SExpr (EConst (CFloat64 Float.nan))) in
  assert (kernel_uses_nonfinite_float64 k_nan = true) ;
  (* Positive: -inf buried inside an otherwise ordinary expression. *)
  let k_neg_inf =
    empty_kernel
      (SExpr
         (EBinop
            (Add, EConst (CFloat64 1.0), EConst (CFloat64 Float.neg_infinity))))
  in
  assert (kernel_uses_nonfinite_float64 k_neg_inf = true) ;
  (* Negative: an ordinary FINITE f64 constant must NOT trip the detector — the
     distinction from [kernel_uses_float64] is exactly finiteness. *)
  let k_finite = empty_kernel (SExpr (EConst (CFloat64 3.14))) in
  assert (kernel_uses_nonfinite_float64 k_finite = false) ;
  (* Negative: an f32 non-finite value is not an f64 constant. *)
  let k_f32_inf = empty_kernel (SExpr (EConst (CFloat32 Float.infinity))) in
  assert (kernel_uses_nonfinite_float64 k_f32_inf = false) ;
  (* Negative: empty kernel. *)
  assert (kernel_uses_nonfinite_float64 (empty_kernel SEmpty) = false) ;
  (* Negative: SNative is treated as non-finite-free (asymmetric vs atomics). *)
  let k_native =
    empty_kernel (SNative {gpu = dummy_native_gpu; ocaml = dummy_native_ocaml})
  in
  assert (kernel_uses_nonfinite_float64 k_native = false) ;
  print_endline "  kernel_uses_nonfinite_float64: OK"

(** {1 kernel_uses_intrinsic Tests}

    Generic named-intrinsic detector: [kernel_uses_intrinsic name k] matches an
    [EIntrinsic] by [name] only (module path ignored), so both the [Float32] and
    [Float64] spellings are found. [SNative] is conservatively assumed to
    reference the intrinsic. *)

let test_kernel_uses_intrinsic () =
  (* Positive: fmod invoked directly in the body (Float64 path). *)
  let k_fmod =
    empty_kernel
      (SExpr
         (EIntrinsic
            (["Float64"], "fmod", [EConst (CFloat64 7.0); EConst (CFloat64 2.0)])))
  in
  assert (kernel_uses_intrinsic "fmod" k_fmod = true) ;
  (* Path-agnostic: the Float32 spelling of the same name is detected too. *)
  let k_fmod_f32 =
    empty_kernel
      (SExpr
         (EIntrinsic
            (["Float32"], "fmod", [EConst (CFloat32 7.0); EConst (CFloat32 2.0)])))
  in
  assert (kernel_uses_intrinsic "fmod" k_fmod_f32 = true) ;
  (* Positive: reachable only through a helper function, not kern_body. *)
  let param : var =
    {var_name = "x"; var_id = 0; var_type = TFloat64; var_mutable = false}
  in
  let hf_fmod : helper_func =
    {
      hf_name = "f";
      hf_params = [param];
      hf_ret_type = TFloat64;
      hf_body =
        SReturn
          (EIntrinsic (["Float64"], "fmod", [EVar param; EConst (CFloat64 2.0)]));
    }
  in
  let k_helper = {(empty_kernel SEmpty) with kern_funcs = [hf_fmod]} in
  assert (kernel_uses_intrinsic "fmod" k_helper = true) ;
  (* Negative: kernel calls a DIFFERENT intrinsic — must not match "fmod". *)
  let k_sin =
    empty_kernel
      (SExpr (EIntrinsic (["Float64"], "sin", [EConst (CFloat64 1.0)])))
  in
  assert (kernel_uses_intrinsic "fmod" k_sin = false) ;
  (* Negative: no intrinsics at all. *)
  assert (kernel_uses_intrinsic "fmod" (empty_kernel SEmpty) = false) ;
  (* Conservative: SNative is assumed to reference the intrinsic. *)
  let k_native =
    empty_kernel (SNative {gpu = dummy_native_gpu; ocaml = dummy_native_ocaml})
  in
  assert (kernel_uses_intrinsic "fmod" k_native = true) ;
  print_endline "  kernel_uses_intrinsic: OK"

(** {1 kernel_float64_intrinsics Tests}

    STRING-LIST collector: gathers the names of every [EIntrinsic] whose [path]
    carries a ["Float64"] component, returning them via [List.sort_uniq compare]
    — i.e. deduplicated and sorted in ascending (OCaml polymorphic) order. The
    asserted lists below are pinned to that exact order. Float32-pathed
    intrinsics are excluded; [SNative] contributes nothing. *)

let test_kernel_float64_intrinsics () =
  (* Several f64 intrinsics (sin appears twice; cos, exp once), plus one
     Float32-pathed intrinsic that must be excluded. Expected result is the
     deduplicated, ascending-sorted name list: ["cos"; "exp"; "sin"]. *)
  let k_many =
    empty_kernel
      (SSeq
         [
           SExpr (EIntrinsic (["Float64"], "sin", [EConst (CFloat64 1.0)]));
           SExpr
             (EIntrinsic (["Math"; "Float64"], "cos", [EConst (CFloat64 1.0)]));
           SExpr (EIntrinsic (["Float64"], "exp", [EConst (CFloat64 1.0)]));
           (* duplicate name -> must be deduplicated *)
           SExpr (EIntrinsic (["Float64"], "sin", [EConst (CFloat64 2.0)]));
           (* Float32-pathed -> must be EXCLUDED from the f64 collector *)
           SExpr (EIntrinsic (["Float32"], "tan", [EConst (CFloat32 1.0)]));
         ])
  in
  assert (kernel_float64_intrinsics k_many = ["cos"; "exp"; "sin"]) ;
  (* Negative: a kernel using NO Float64-pathed intrinsic collects []. Here the
     only intrinsic is Float32-pathed. *)
  let k_f32_only =
    empty_kernel
      (SExpr (EIntrinsic (["Float32"], "sin", [EConst (CFloat32 1.0)])))
  in
  assert (kernel_float64_intrinsics k_f32_only = []) ;
  (* Negative: empty kernel collects []. *)
  assert (kernel_float64_intrinsics (empty_kernel SEmpty) = []) ;
  print_endline "  kernel_float64_intrinsics: OK"

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
  test_elttype_float16 () ;
  test_elttype_nested_float16 () ;
  test_expr_uses_float16 () ;
  test_stmt_decl_uses_float16 () ;
  test_kernel_uses_float16 () ;
  test_feature_api_agrees_with_aliases () ;
  test_const_uses_is_width_specific () ;
  test_kernel_requirements () ;
  test_is_atomic_intrinsic_name () ;
  test_expr_uses_atomics () ;
  test_helper_uses_atomics () ;
  test_kernel_uses_atomics_direct () ;
  test_kernel_uses_atomics_in_helper () ;
  test_lvalue_uses_atomics () ;
  test_kernel_uses_atomics_in_assign_lvalue () ;
  test_kernel_uses_atomics_snative () ;
  test_expr_uses_int_mod () ;
  test_lvalue_uses_int_mod () ;
  test_kernel_uses_int_mod_in_record_field_lvalue () ;
  test_is_copysign_intrinsic_name () ;
  test_expr_uses_copysign () ;
  test_lvalue_uses_copysign () ;
  test_kernel_uses_copysign_in_record_field_lvalue () ;
  test_kernel_uses_copysign_in_helper () ;
  test_kernel_uses_nonfinite_float64 () ;
  test_kernel_uses_intrinsic () ;
  test_kernel_float64_intrinsics () ;
  print_endline "All Sarek_ir_analysis tests passed!"
