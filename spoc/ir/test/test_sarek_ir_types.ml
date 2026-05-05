(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Unit tests for Sarek_ir_types
 *
 * Tests IR type definitions: elttypes, expressions, statements, kernels.
 ******************************************************************************)

open Sarek_ir_types

(** {1 Memspace Tests} *)

let test_memspace () =
  let g = Global in
  let s = Shared in
  let l = Local in
  assert (g <> s) ;
  assert (s <> l) ;
  assert (g <> l) ;
  print_endline "  memspace: OK"

(** {1 Elttype Tests} *)

let test_elttype_primitives () =
  let types = [TInt32; TInt64; TFloat32; TFloat64; TBool; TUnit] in
  assert (List.length types = 6) ;
  (* Ensure all are distinct *)
  let distinct = List.sort_uniq compare types in
  assert (List.length distinct = 6) ;
  print_endline "  elttype primitives: OK"

let test_elttype_record () =
  let rec_ty = TRecord ("point", [("x", TFloat32); ("y", TFloat32)]) in
  (match rec_ty with
  | TRecord (name, fields) ->
      assert (name = "point") ;
      assert (List.length fields = 2) ;
      assert (List.assoc "x" fields = TFloat32)
  | _ -> assert false) ;
  print_endline "  elttype record: OK"

let test_elttype_variant () =
  let var_ty = TVariant ("option_int", [("None", []); ("Some", [TInt32])]) in
  (match var_ty with
  | TVariant (name, constrs) ->
      assert (name = "option_int") ;
      assert (List.length constrs = 2) ;
      let none_args = List.assoc "None" constrs in
      assert (none_args = []) ;
      let some_args = List.assoc "Some" constrs in
      assert (some_args = [TInt32])
  | _ -> assert false) ;
  print_endline "  elttype variant: OK"

let test_elttype_array () =
  let arr = TArray (TFloat64, Shared) in
  (match arr with
  | TArray (elt, ms) ->
      assert (elt = TFloat64) ;
      assert (ms = Shared)
  | _ -> assert false) ;
  print_endline "  elttype array: OK"

let test_elttype_vec () =
  let vec = TVec TInt32 in
  (match vec with TVec elt -> assert (elt = TInt32) | _ -> assert false) ;
  print_endline "  elttype vec: OK"

(** {1 Var Tests} *)

let test_var () =
  let v : var =
    {var_name = "x"; var_id = 0; var_type = TFloat32; var_mutable = false}
  in
  assert (v.var_name = "x") ;
  assert (v.var_id = 0) ;
  assert (v.var_type = TFloat32) ;
  assert (not v.var_mutable) ;
  print_endline "  var: OK"

let test_var_mutable () =
  let v : var =
    {var_name = "counter"; var_id = 1; var_type = TInt32; var_mutable = true}
  in
  assert v.var_mutable ;
  print_endline "  var mutable: OK"

(** {1 Const Tests} *)

let test_const_int32 () =
  let c = CInt32 42l in
  (match c with CInt32 n -> assert (n = 42l) | _ -> assert false) ;
  print_endline "  const int32: OK"

let test_const_int64 () =
  let c = CInt64 123456789012345L in
  (match c with CInt64 n -> assert (n = 123456789012345L) | _ -> assert false) ;
  print_endline "  const int64: OK"

let test_const_float32 () =
  let c = CFloat32 3.14 in
  (match c with
  | CFloat32 f -> assert (abs_float (f -. 3.14) < 0.001)
  | _ -> assert false) ;
  print_endline "  const float32: OK"

let test_const_float64 () =
  let c = CFloat64 2.718281828459045 in
  (match c with
  | CFloat64 f -> assert (abs_float (f -. 2.718281828459045) < 1e-15)
  | _ -> assert false) ;
  print_endline "  const float64: OK"

let test_const_bool () =
  let ct = CBool true in
  let cf = CBool false in
  (match ct with CBool b -> assert b | _ -> assert false) ;
  (match cf with CBool b -> assert (not b) | _ -> assert false) ;
  print_endline "  const bool: OK"

let test_const_unit () =
  let c = CUnit in
  (match c with CUnit -> () | _ -> assert false) ;
  print_endline "  const unit: OK"

(** {1 Binop Tests} *)

let test_binop () =
  let ops =
    [
      Add;
      Sub;
      Mul;
      Div;
      Mod;
      Eq;
      Ne;
      Lt;
      Le;
      Gt;
      Ge;
      And;
      Or;
      Shl;
      Shr;
      BitAnd;
      BitOr;
      BitXor;
    ]
  in
  assert (List.length ops = 18) ;
  let distinct = List.sort_uniq compare ops in
  assert (List.length distinct = 18) ;
  print_endline "  binop: OK"

(** {1 Unop Tests} *)

let test_unop () =
  let ops = [Neg; Not; BitNot] in
  assert (List.length ops = 3) ;
  let distinct = List.sort_uniq compare ops in
  assert (List.length distinct = 3) ;
  print_endline "  unop: OK"

(** {1 Expr Tests} *)

let test_expr_const () =
  let e = EConst (CInt32 100l) in
  (match e with EConst (CInt32 n) -> assert (n = 100l) | _ -> assert false) ;
  print_endline "  expr const: OK"

let test_expr_var () =
  let v : var =
    {var_name = "x"; var_id = 0; var_type = TFloat32; var_mutable = false}
  in
  let e = EVar v in
  (match e with EVar v' -> assert (v'.var_name = "x") | _ -> assert false) ;
  print_endline "  expr var: OK"

let test_expr_binop () =
  let e1 = EConst (CInt32 10l) in
  let e2 = EConst (CInt32 20l) in
  let e = EBinop (Add, e1, e2) in
  (match e with EBinop (Add, _, _) -> () | _ -> assert false) ;
  print_endline "  expr binop: OK"

let test_expr_intrinsic () =
  let e = EIntrinsic (["Float32"], "sin", [EConst (CFloat32 0.5)]) in
  (match e with
  | EIntrinsic (path, name, args) ->
      assert (path = ["Float32"]) ;
      assert (name = "sin") ;
      assert (List.length args = 1)
  | _ -> assert false) ;
  print_endline "  expr intrinsic: OK"

let test_expr_record () =
  let e =
    ERecord
      ("point", [("x", EConst (CFloat32 1.0)); ("y", EConst (CFloat32 2.0))])
  in
  (match e with
  | ERecord (name, fields) ->
      assert (name = "point") ;
      assert (List.length fields = 2)
  | _ -> assert false) ;
  print_endline "  expr record: OK"

let test_expr_if () =
  let e = EIf (EConst (CBool true), EConst (CInt32 1l), EConst (CInt32 0l)) in
  (match e with
  | EIf (cond, then_, else_) -> (
      (match cond with EConst (CBool true) -> () | _ -> assert false) ;
      (match then_ with EConst (CInt32 1l) -> () | _ -> assert false) ;
      match else_ with EConst (CInt32 0l) -> () | _ -> assert false)
  | _ -> assert false) ;
  print_endline "  expr if: OK"

(** {1 Lvalue Tests} *)

let test_lvalue_var () =
  let v : var =
    {var_name = "x"; var_id = 0; var_type = TInt32; var_mutable = true}
  in
  let lv = LVar v in
  (match lv with LVar v' -> assert (v'.var_name = "x") | _ -> assert false) ;
  print_endline "  lvalue var: OK"

let test_lvalue_array () =
  let lv = LArrayElem ("arr", EConst (CInt32 5l)) in
  (match lv with
  | LArrayElem (name, idx) -> (
      assert (name = "arr") ;
      match idx with EConst (CInt32 5l) -> () | _ -> assert false)
  | _ -> assert false) ;
  print_endline "  lvalue array: OK"

(** {1 Stmt Tests} *)

let test_stmt_assign () =
  let v : var =
    {var_name = "x"; var_id = 0; var_type = TInt32; var_mutable = true}
  in
  let s = SAssign (LVar v, EConst (CInt32 42l)) in
  (match s with
  | SAssign (LVar v', EConst (CInt32 42l)) -> assert (v'.var_name = "x")
  | _ -> assert false) ;
  print_endline "  stmt assign: OK"

let test_stmt_seq () =
  let s = SSeq [SEmpty; SEmpty; SEmpty] in
  (match s with
  | SSeq stmts -> assert (List.length stmts = 3)
  | _ -> assert false) ;
  print_endline "  stmt seq: OK"

let test_stmt_if () =
  let s = SIf (EConst (CBool true), SEmpty, Some SEmpty) in
  (match s with SIf (_, _, Some _) -> () | _ -> assert false) ;
  print_endline "  stmt if: OK"

let test_stmt_for () =
  let v : var =
    {var_name = "i"; var_id = 0; var_type = TInt32; var_mutable = true}
  in
  let s = SFor (v, EConst (CInt32 0l), EConst (CInt32 10l), Upto, SEmpty) in
  (match s with
  | SFor (v', lo, hi, Upto, _) -> (
      assert (v'.var_name = "i") ;
      (match lo with EConst (CInt32 0l) -> () | _ -> assert false) ;
      match hi with EConst (CInt32 10l) -> () | _ -> assert false)
  | _ -> assert false) ;
  print_endline "  stmt for: OK"

let test_stmt_barrier () =
  let s = SBarrier in
  (match s with SBarrier -> () | _ -> assert false) ;
  print_endline "  stmt barrier: OK"

let test_stmt_let () =
  let v : var =
    {var_name = "tmp"; var_id = 0; var_type = TFloat32; var_mutable = false}
  in
  let s = SLet (v, EConst (CFloat32 1.0), SReturn (EVar v)) in
  (match s with
  | SLet (v', _, _) -> assert (v'.var_name = "tmp")
  | _ -> assert false) ;
  print_endline "  stmt let: OK"

let test_stmt_pragma () =
  let s = SPragma (["unroll"; "4"], SEmpty) in
  (match s with
  | SPragma (hints, _) ->
      assert (List.mem "unroll" hints) ;
      assert (List.mem "4" hints)
  | _ -> assert false) ;
  print_endline "  stmt pragma: OK"

(** {1 Decl Tests} *)

let test_decl_param () =
  let v : var =
    {
      var_name = "input";
      var_id = 0;
      var_type = TVec TFloat32;
      var_mutable = false;
    }
  in
  let arr_info = Some {arr_elttype = TFloat32; arr_memspace = Global} in
  let d = DParam (v, arr_info) in
  (match d with
  | DParam (v', Some ai) ->
      assert (v'.var_name = "input") ;
      assert (ai.arr_elttype = TFloat32) ;
      assert (ai.arr_memspace = Global)
  | _ -> assert false) ;
  print_endline "  decl param: OK"

let test_decl_local () =
  let v : var =
    {var_name = "sum"; var_id = 0; var_type = TFloat32; var_mutable = true}
  in
  let d = DLocal (v, Some (EConst (CFloat32 0.0))) in
  (match d with
  | DLocal (v', Some _) -> assert (v'.var_name = "sum")
  | _ -> assert false) ;
  print_endline "  decl local: OK"

let test_decl_shared () =
  let d = DShared ("cache", TFloat32, Some (EConst (CInt32 256l))) in
  (match d with
  | DShared (name, ty, Some _) ->
      assert (name = "cache") ;
      assert (ty = TFloat32)
  | _ -> assert false) ;
  print_endline "  decl shared: OK"

(** {1 Kernel Tests} *)

let test_kernel () =
  let input_var : var =
    {
      var_name = "input";
      var_id = 0;
      var_type = TVec TFloat32;
      var_mutable = false;
    }
  in
  let output_var : var =
    {
      var_name = "output";
      var_id = 1;
      var_type = TVec TFloat32;
      var_mutable = false;
    }
  in
  let k : kernel =
    {
      kern_name = "vector_add";
      kern_params =
        [
          DParam
            (input_var, Some {arr_elttype = TFloat32; arr_memspace = Global});
          DParam
            (output_var, Some {arr_elttype = TFloat32; arr_memspace = Global});
        ];
      kern_locals = [];
      kern_body = SEmpty;
      kern_types = [];
      kern_variants = [];
      kern_funcs = [];
      kern_native_fn = None;
    }
  in
  assert (k.kern_name = "vector_add") ;
  assert (List.length k.kern_params = 2) ;
  assert (k.kern_native_fn = None) ;
  print_endline "  kernel: OK"

(** {1 Helper Function Tests} *)

let test_helper_func () =
  let param : var =
    {var_name = "x"; var_id = 0; var_type = TFloat32; var_mutable = false}
  in
  let hf : helper_func =
    {
      hf_name = "square";
      hf_params = [param];
      hf_ret_type = TFloat32;
      hf_body = SReturn (EBinop (Mul, EVar param, EVar param));
    }
  in
  assert (hf.hf_name = "square") ;
  assert (List.length hf.hf_params = 1) ;
  assert (hf.hf_ret_type = TFloat32) ;
  print_endline "  helper_func: OK"

(** {1 native_arg Tests} *)

let test_native_arg () =
  let na_i32 = NA_Int32 100l in
  let na_i64 = NA_Int64 200L in
  let na_f32 = NA_Float32 1.5 in
  let na_f64 = NA_Float64 2.5 in
  (match na_i32 with NA_Int32 n -> assert (n = 100l) | _ -> assert false) ;
  (match na_i64 with NA_Int64 n -> assert (n = 200L) | _ -> assert false) ;
  (match na_f32 with NA_Float32 f -> assert (f = 1.5) | _ -> assert false) ;
  (match na_f64 with NA_Float64 f -> assert (f = 2.5) | _ -> assert false) ;
  print_endline "  native_arg: OK"

let test_vec_length () =
  (* Create a minimal NA_Vec for testing vec_length.
     Only the length field is used by vec_length, so we provide
     no-op implementations for unused fields. *)
  let na =
    NA_Vec
      (NV
      {
        length = 1024;
        elem_size = 4;
        type_name = "float32";
        type_id = Type_id.create ();
        get_f32 = (fun _ -> 0.0);
        set_f32 = (fun _ _ -> ());
        get_f64 = (fun _ -> 0.0);
        set_f64 = (fun _ _ -> ());
        get_i32 = (fun _ -> 0l);
        set_i32 = (fun _ _ -> ());
        get_i64 = (fun _ -> 0L);
        set_i64 = (fun _ _ -> ());
        get_typed = (fun _ -> 0.0);
        set_typed = (fun _ _ -> ());
        underlying_type_id = Type_id.create ();
        underlying = ();
      })
  in
  let len = vec_length na in
  assert (len = 1024) ;
  print_endline "  vec_length: OK"

(** {1 Pattern Tests} *)

let test_pattern () =
  let p1 = PConstr ("Some", ["x"]) in
  let p2 = PWild in
  (match p1 with
  | PConstr (name, vars) ->
      assert (name = "Some") ;
      assert (vars = ["x"])
  | _ -> assert false) ;
  (match p2 with PWild -> () | _ -> assert false) ;
  print_endline "  pattern: OK"

(** {1 Main} *)

let () =
  print_endline "Sarek_ir_types tests:" ;
  test_memspace () ;
  test_elttype_primitives () ;
  test_elttype_record () ;
  test_elttype_variant () ;
  test_elttype_array () ;
  test_elttype_vec () ;
  test_var () ;
  test_var_mutable () ;
  test_const_int32 () ;
  test_const_int64 () ;
  test_const_float32 () ;
  test_const_float64 () ;
  test_const_bool () ;
  test_const_unit () ;
  test_binop () ;
  test_unop () ;
  test_expr_const () ;
  test_expr_var () ;
  test_expr_binop () ;
  test_expr_intrinsic () ;
  test_expr_record () ;
  test_expr_if () ;
  test_lvalue_var () ;
  test_lvalue_array () ;
  test_stmt_assign () ;
  test_stmt_seq () ;
  test_stmt_if () ;
  test_stmt_for () ;
  test_stmt_barrier () ;
  test_stmt_let () ;
  test_stmt_pragma () ;
  test_decl_param () ;
  test_decl_local () ;
  test_decl_shared () ;
  test_kernel () ;
  test_helper_func () ;
  test_native_arg () ;
  test_vec_length () ;
  test_pattern () ;
  print_endline "All Sarek_ir_types tests passed!"
