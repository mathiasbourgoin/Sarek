(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Unit tests for Sarek_quote_ir module
 *
 * Tests IR quoting functions that convert Sarek_ir_ppx values to ppxlib
 * expressions constructing Sarek.Sarek_ir values
 ******************************************************************************)

(* Force stdlib initialization *)
let () = Sarek_stdlib.force_init ()

open Ppxlib
module Ir = Sarek_ir_ppx

let dummy_loc : Location.t = Location.none

(* Helper to check if expression contains specific identifier *)
let expr_contains_ident e name =
  let rec check expr =
    match expr.pexp_desc with
    | Pexp_ident {txt = Lident id; _} when id = name -> true
    | Pexp_ident {txt = Ldot (_, id); _} when id = name -> true
    | Pexp_apply (f, args) ->
        check f || List.exists (fun (_, arg) -> check arg) args
    | Pexp_construct ({txt = Lident id; _}, _) when id = name -> true
    | Pexp_construct ({txt = Ldot (_, id); _}, _) when id = name -> true
    | Pexp_construct (_, Some arg) -> check arg
    | Pexp_tuple exprs -> List.exists check exprs
    | Pexp_record (fields, _) -> List.exists (fun (_, e) -> check e) fields
    | Pexp_extension (_, payload) -> check_payload payload
    | _ -> false
  and check_payload = function
    | PStr [{pstr_desc = Pstr_eval (e, _); _}] -> check e
    | _ -> false
  in
  check e

(** {1 Memspace Tests} *)

let test_quote_memspace_global () =
  let e = Sarek_quote_ir.quote_memspace ~loc:dummy_loc Ir.Global in
  Alcotest.(check bool) "contains Global" true (expr_contains_ident e "Global")

let test_quote_memspace_shared () =
  let e = Sarek_quote_ir.quote_memspace ~loc:dummy_loc Ir.Shared in
  Alcotest.(check bool) "contains Shared" true (expr_contains_ident e "Shared")

let test_quote_memspace_local () =
  let e = Sarek_quote_ir.quote_memspace ~loc:dummy_loc Ir.Local in
  Alcotest.(check bool) "contains Local" true (expr_contains_ident e "Local")

(** {1 Elttype Tests} *)

let test_quote_elttype_int32 () =
  let e = Sarek_quote_ir.quote_elttype ~loc:dummy_loc Ir.TInt32 in
  Alcotest.(check bool) "contains TInt32" true (expr_contains_ident e "TInt32")

let test_quote_elttype_int64 () =
  let e = Sarek_quote_ir.quote_elttype ~loc:dummy_loc Ir.TInt64 in
  Alcotest.(check bool) "contains TInt64" true (expr_contains_ident e "TInt64")

let test_quote_elttype_float32 () =
  let e = Sarek_quote_ir.quote_elttype ~loc:dummy_loc Ir.TFloat32 in
  Alcotest.(check bool)
    "contains TFloat32"
    true
    (expr_contains_ident e "TFloat32")

let test_quote_elttype_float64 () =
  let e = Sarek_quote_ir.quote_elttype ~loc:dummy_loc Ir.TFloat64 in
  Alcotest.(check bool)
    "contains TFloat64"
    true
    (expr_contains_ident e "TFloat64")

let test_quote_elttype_bool () =
  let e = Sarek_quote_ir.quote_elttype ~loc:dummy_loc Ir.TBool in
  Alcotest.(check bool) "contains TBool" true (expr_contains_ident e "TBool")

let test_quote_elttype_unit () =
  let e = Sarek_quote_ir.quote_elttype ~loc:dummy_loc Ir.TUnit in
  Alcotest.(check bool) "contains TUnit" true (expr_contains_ident e "TUnit")

let test_quote_elttype_record () =
  let fields = [("x", Ir.TInt32); ("y", Ir.TFloat32)] in
  let e =
    Sarek_quote_ir.quote_elttype ~loc:dummy_loc (Ir.TRecord ("Point", fields))
  in
  Alcotest.(check bool)
    "contains TRecord"
    true
    (expr_contains_ident e "TRecord")

let test_quote_elttype_variant () =
  let constrs = [("Some", [Ir.TInt32]); ("None", [])] in
  let e =
    Sarek_quote_ir.quote_elttype
      ~loc:dummy_loc
      (Ir.TVariant ("Option", constrs))
  in
  Alcotest.(check bool)
    "contains TVariant"
    true
    (expr_contains_ident e "TVariant")

let test_quote_elttype_array () =
  let e =
    Sarek_quote_ir.quote_elttype
      ~loc:dummy_loc
      (Ir.TArray (Ir.TFloat32, Ir.Global))
  in
  Alcotest.(check bool) "contains TArray" true (expr_contains_ident e "TArray")

let test_quote_elttype_vec () =
  let e = Sarek_quote_ir.quote_elttype ~loc:dummy_loc (Ir.TVec Ir.TInt32) in
  Alcotest.(check bool) "contains TVec" true (expr_contains_ident e "TVec")

(** {1 Binop Tests} *)

let test_quote_binop_add () =
  let e = Sarek_quote_ir.quote_binop ~loc:dummy_loc Ir.Add in
  Alcotest.(check bool) "contains Add" true (expr_contains_ident e "Add")

let test_quote_binop_sub () =
  let e = Sarek_quote_ir.quote_binop ~loc:dummy_loc Ir.Sub in
  Alcotest.(check bool) "contains Sub" true (expr_contains_ident e "Sub")

let test_quote_binop_mul () =
  let e = Sarek_quote_ir.quote_binop ~loc:dummy_loc Ir.Mul in
  Alcotest.(check bool) "contains Mul" true (expr_contains_ident e "Mul")

let test_quote_binop_div () =
  let e = Sarek_quote_ir.quote_binop ~loc:dummy_loc Ir.Div in
  Alcotest.(check bool) "contains Div" true (expr_contains_ident e "Div")

let test_quote_binop_mod () =
  let e = Sarek_quote_ir.quote_binop ~loc:dummy_loc Ir.Mod in
  Alcotest.(check bool) "contains Mod" true (expr_contains_ident e "Mod")

let test_quote_binop_eq () =
  let e = Sarek_quote_ir.quote_binop ~loc:dummy_loc Ir.Eq in
  Alcotest.(check bool) "contains Eq" true (expr_contains_ident e "Eq")

let test_quote_binop_logical () =
  let e_and = Sarek_quote_ir.quote_binop ~loc:dummy_loc Ir.And in
  let e_or = Sarek_quote_ir.quote_binop ~loc:dummy_loc Ir.Or in
  Alcotest.(check bool) "And" true (expr_contains_ident e_and "And") ;
  Alcotest.(check bool) "Or" true (expr_contains_ident e_or "Or")

let test_quote_binop_bitwise () =
  let e_shl = Sarek_quote_ir.quote_binop ~loc:dummy_loc Ir.Shl in
  let e_shr = Sarek_quote_ir.quote_binop ~loc:dummy_loc Ir.Shr in
  let e_and = Sarek_quote_ir.quote_binop ~loc:dummy_loc Ir.BitAnd in
  let e_or = Sarek_quote_ir.quote_binop ~loc:dummy_loc Ir.BitOr in
  let e_xor = Sarek_quote_ir.quote_binop ~loc:dummy_loc Ir.BitXor in
  Alcotest.(check bool) "Shl" true (expr_contains_ident e_shl "Shl") ;
  Alcotest.(check bool) "Shr" true (expr_contains_ident e_shr "Shr") ;
  Alcotest.(check bool) "BitAnd" true (expr_contains_ident e_and "BitAnd") ;
  Alcotest.(check bool) "BitOr" true (expr_contains_ident e_or "BitOr") ;
  Alcotest.(check bool) "BitXor" true (expr_contains_ident e_xor "BitXor")

(** {1 Unop Tests} *)

let test_quote_unop_neg () =
  let e = Sarek_quote_ir.quote_unop ~loc:dummy_loc Ir.Neg in
  Alcotest.(check bool) "contains Neg" true (expr_contains_ident e "Neg")

let test_quote_unop_not () =
  let e = Sarek_quote_ir.quote_unop ~loc:dummy_loc Ir.Not in
  Alcotest.(check bool) "contains Not" true (expr_contains_ident e "Not")

let test_quote_unop_bitnot () =
  let e = Sarek_quote_ir.quote_unop ~loc:dummy_loc Ir.BitNot in
  Alcotest.(check bool) "contains BitNot" true (expr_contains_ident e "BitNot")

(** {1 Const Tests} *)

let test_quote_const_int32 () =
  let e = Sarek_quote_ir.quote_const ~loc:dummy_loc (Ir.CInt32 42l) in
  Alcotest.(check bool) "contains CInt32" true (expr_contains_ident e "CInt32")

let test_quote_const_int64 () =
  let e = Sarek_quote_ir.quote_const ~loc:dummy_loc (Ir.CInt64 100L) in
  Alcotest.(check bool) "contains CInt64" true (expr_contains_ident e "CInt64")

let test_quote_const_float32 () =
  let e = Sarek_quote_ir.quote_const ~loc:dummy_loc (Ir.CFloat32 3.14) in
  Alcotest.(check bool)
    "contains CFloat32"
    true
    (expr_contains_ident e "CFloat32")

let test_quote_const_float64 () =
  let e = Sarek_quote_ir.quote_const ~loc:dummy_loc (Ir.CFloat64 2.71) in
  Alcotest.(check bool)
    "contains CFloat64"
    true
    (expr_contains_ident e "CFloat64")

let test_quote_const_bool () =
  let e_true = Sarek_quote_ir.quote_const ~loc:dummy_loc (Ir.CBool true) in
  let e_false = Sarek_quote_ir.quote_const ~loc:dummy_loc (Ir.CBool false) in
  Alcotest.(check bool)
    "contains CBool true"
    true
    (expr_contains_ident e_true "CBool") ;
  Alcotest.(check bool)
    "contains CBool false"
    true
    (expr_contains_ident e_false "CBool")

let test_quote_const_unit () =
  let e = Sarek_quote_ir.quote_const ~loc:dummy_loc Ir.CUnit in
  Alcotest.(check bool) "contains CUnit" true (expr_contains_ident e "CUnit")

(** {1 Var Tests} *)

let test_quote_var () =
  let v =
    Ir.{var_name = "x"; var_id = 0; var_type = TInt32; var_mutable = false}
  in
  let e = Sarek_quote_ir.quote_var ~loc:dummy_loc v in
  (* Quoted vars are records - just check structure *)
  match e.pexp_desc with
  | Pexp_record _ -> ()
  | _ -> Alcotest.fail "expected record"

let test_quote_var_mutable () =
  let v =
    Ir.{var_name = "mut_x"; var_id = 1; var_type = TFloat32; var_mutable = true}
  in
  let e = Sarek_quote_ir.quote_var ~loc:dummy_loc v in
  match e.pexp_desc with
  | Pexp_record _ -> ()
  | _ -> Alcotest.fail "expected record"

(** {1 Pattern Tests} *)

let test_quote_pattern_constr () =
  let p = Ir.PConstr ("Some", ["x"]) in
  let e = Sarek_quote_ir.quote_pattern ~loc:dummy_loc p in
  Alcotest.(check bool)
    "contains PConstr"
    true
    (expr_contains_ident e "PConstr")

let test_quote_pattern_wild () =
  let p = Ir.PWild in
  let e = Sarek_quote_ir.quote_pattern ~loc:dummy_loc p in
  Alcotest.(check bool) "contains PWild" true (expr_contains_ident e "PWild")

(** {1 Expr Tests} *)

let test_quote_expr_const () =
  let e =
    Sarek_quote_ir.quote_expr ~loc:dummy_loc (Ir.EConst (Ir.CInt32 42l))
  in
  Alcotest.(check bool) "contains EConst" true (expr_contains_ident e "EConst")

let test_quote_expr_var () =
  let v =
    Ir.{var_name = "x"; var_id = 0; var_type = TInt32; var_mutable = false}
  in
  let e = Sarek_quote_ir.quote_expr ~loc:dummy_loc (Ir.EVar v) in
  Alcotest.(check bool) "contains EVar" true (expr_contains_ident e "EVar")

let test_quote_expr_binop () =
  let expr =
    Ir.EBinop (Ir.Add, Ir.EConst (Ir.CInt32 1l), Ir.EConst (Ir.CInt32 2l))
  in
  let e = Sarek_quote_ir.quote_expr ~loc:dummy_loc expr in
  Alcotest.(check bool) "contains EBinop" true (expr_contains_ident e "EBinop") ;
  Alcotest.(check bool) "contains Add" true (expr_contains_ident e "Add")

let test_quote_expr_unop () =
  let expr = Ir.EUnop (Ir.Neg, Ir.EConst (Ir.CInt32 5l)) in
  let e = Sarek_quote_ir.quote_expr ~loc:dummy_loc expr in
  Alcotest.(check bool) "contains EUnop" true (expr_contains_ident e "EUnop") ;
  Alcotest.(check bool) "contains Neg" true (expr_contains_ident e "Neg")

let test_quote_expr_array_read () =
  let expr = Ir.EArrayRead ("arr", Ir.EConst (Ir.CInt32 0l)) in
  let e = Sarek_quote_ir.quote_expr ~loc:dummy_loc expr in
  Alcotest.(check bool)
    "contains EArrayRead"
    true
    (expr_contains_ident e "EArrayRead")

let test_quote_expr_record_field () =
  let v =
    Ir.
      {
        var_name = "p";
        var_id = 0;
        var_type = TRecord ("Point", []);
        var_mutable = false;
      }
  in
  let expr = Ir.ERecordField (Ir.EVar v, "x") in
  let e = Sarek_quote_ir.quote_expr ~loc:dummy_loc expr in
  Alcotest.(check bool)
    "contains ERecordField"
    true
    (expr_contains_ident e "ERecordField")

let test_quote_expr_intrinsic () =
  let expr = Ir.EIntrinsic (["Std"], "global_idx_x", []) in
  let e = Sarek_quote_ir.quote_expr ~loc:dummy_loc expr in
  Alcotest.(check bool)
    "contains EIntrinsic"
    true
    (expr_contains_ident e "EIntrinsic")

let test_quote_expr_cast () =
  let expr = Ir.ECast (Ir.TFloat32, Ir.EConst (Ir.CInt32 42l)) in
  let e = Sarek_quote_ir.quote_expr ~loc:dummy_loc expr in
  Alcotest.(check bool) "contains ECast" true (expr_contains_ident e "ECast")

let test_quote_expr_tuple () =
  let expr = Ir.ETuple [Ir.EConst (Ir.CInt32 1l); Ir.EConst (Ir.CInt32 2l)] in
  let e = Sarek_quote_ir.quote_expr ~loc:dummy_loc expr in
  Alcotest.(check bool) "contains ETuple" true (expr_contains_ident e "ETuple")

let test_quote_expr_record () =
  let fields =
    [("x", Ir.EConst (Ir.CInt32 1l)); ("y", Ir.EConst (Ir.CInt32 2l))]
  in
  let expr = Ir.ERecord ("Point", fields) in
  let e = Sarek_quote_ir.quote_expr ~loc:dummy_loc expr in
  Alcotest.(check bool)
    "contains ERecord"
    true
    (expr_contains_ident e "ERecord")

let test_quote_expr_variant () =
  let expr = Ir.EVariant ("Option", "Some", [Ir.EConst (Ir.CInt32 42l)]) in
  let e = Sarek_quote_ir.quote_expr ~loc:dummy_loc expr in
  Alcotest.(check bool)
    "contains EVariant"
    true
    (expr_contains_ident e "EVariant")

let test_quote_expr_array_len () =
  let expr = Ir.EArrayLen "arr" in
  let e = Sarek_quote_ir.quote_expr ~loc:dummy_loc expr in
  Alcotest.(check bool)
    "contains EArrayLen"
    true
    (expr_contains_ident e "EArrayLen")

let test_quote_expr_array_create () =
  let expr =
    Ir.EArrayCreate (Ir.TInt32, Ir.EConst (Ir.CInt32 100l), Ir.Global)
  in
  let e = Sarek_quote_ir.quote_expr ~loc:dummy_loc expr in
  Alcotest.(check bool)
    "contains EArrayCreate"
    true
    (expr_contains_ident e "EArrayCreate")

let test_quote_expr_if () =
  let expr =
    Ir.EIf
      ( Ir.EConst (Ir.CBool true),
        Ir.EConst (Ir.CInt32 1l),
        Ir.EConst (Ir.CInt32 0l) )
  in
  let e = Sarek_quote_ir.quote_expr ~loc:dummy_loc expr in
  Alcotest.(check bool) "contains EIf" true (expr_contains_ident e "EIf")

let test_quote_expr_match () =
  let cases = [(Ir.PWild, Ir.EConst (Ir.CInt32 42l))] in
  let v =
    Ir.{var_name = "x"; var_id = 0; var_type = TInt32; var_mutable = false}
  in
  let expr = Ir.EMatch (Ir.EVar v, cases) in
  let e = Sarek_quote_ir.quote_expr ~loc:dummy_loc expr in
  Alcotest.(check bool) "contains EMatch" true (expr_contains_ident e "EMatch")

(** {1 Lvalue Tests} *)

let test_quote_lvalue_var () =
  let v =
    Ir.{var_name = "x"; var_id = 0; var_type = TInt32; var_mutable = true}
  in
  let lv = Ir.LVar v in
  let e = Sarek_quote_ir.quote_lvalue ~loc:dummy_loc lv in
  Alcotest.(check bool) "contains LVar" true (expr_contains_ident e "LVar")

let test_quote_lvalue_array_elem () =
  let lv = Ir.LArrayElem ("arr", Ir.EConst (Ir.CInt32 0l)) in
  let e = Sarek_quote_ir.quote_lvalue ~loc:dummy_loc lv in
  Alcotest.(check bool)
    "contains LArrayElem"
    true
    (expr_contains_ident e "LArrayElem")

let test_quote_lvalue_record_field () =
  let v =
    Ir.
      {
        var_name = "p";
        var_id = 0;
        var_type = TRecord ("Point", []);
        var_mutable = true;
      }
  in
  let lv = Ir.LRecordField (Ir.LVar v, "x") in
  let e = Sarek_quote_ir.quote_lvalue ~loc:dummy_loc lv in
  Alcotest.(check bool)
    "contains LRecordField"
    true
    (expr_contains_ident e "LRecordField")

(** {1 Stmt Tests} *)

let test_quote_stmt_assign () =
  let v =
    Ir.{var_name = "x"; var_id = 0; var_type = TInt32; var_mutable = true}
  in
  let stmt = Ir.SAssign (Ir.LVar v, Ir.EConst (Ir.CInt32 42l)) in
  let e = Sarek_quote_ir.quote_stmt ~loc:dummy_loc stmt in
  Alcotest.(check bool)
    "contains SAssign"
    true
    (expr_contains_ident e "SAssign")

let test_quote_stmt_seq () =
  let stmt = Ir.SSeq [] in
  let e = Sarek_quote_ir.quote_stmt ~loc:dummy_loc stmt in
  Alcotest.(check bool) "contains SSeq" true (expr_contains_ident e "SSeq")

let test_quote_stmt_if () =
  let stmt = Ir.SIf (Ir.EConst (Ir.CBool true), Ir.SEmpty, Some Ir.SEmpty) in
  let e = Sarek_quote_ir.quote_stmt ~loc:dummy_loc stmt in
  Alcotest.(check bool) "contains SIf" true (expr_contains_ident e "SIf")

let test_quote_stmt_while () =
  let stmt = Ir.SWhile (Ir.EConst (Ir.CBool true), Ir.SEmpty) in
  let e = Sarek_quote_ir.quote_stmt ~loc:dummy_loc stmt in
  Alcotest.(check bool) "contains SWhile" true (expr_contains_ident e "SWhile")

let test_quote_stmt_for () =
  let v =
    Ir.{var_name = "i"; var_id = 0; var_type = TInt32; var_mutable = false}
  in
  let stmt =
    Ir.SFor
      ( v,
        Ir.EConst (Ir.CInt32 0l),
        Ir.EConst (Ir.CInt32 10l),
        Ir.Upto,
        Ir.SEmpty )
  in
  let e = Sarek_quote_ir.quote_stmt ~loc:dummy_loc stmt in
  Alcotest.(check bool) "contains SFor" true (expr_contains_ident e "SFor") ;
  Alcotest.(check bool) "contains Upto" true (expr_contains_ident e "Upto")

let test_quote_stmt_match () =
  let v =
    Ir.{var_name = "x"; var_id = 0; var_type = TInt32; var_mutable = false}
  in
  let cases = [(Ir.PWild, Ir.SEmpty)] in
  let stmt = Ir.SMatch (Ir.EVar v, cases) in
  let e = Sarek_quote_ir.quote_stmt ~loc:dummy_loc stmt in
  Alcotest.(check bool) "contains SMatch" true (expr_contains_ident e "SMatch")

let test_quote_stmt_return () =
  let stmt = Ir.SReturn (Ir.EConst (Ir.CInt32 42l)) in
  let e = Sarek_quote_ir.quote_stmt ~loc:dummy_loc stmt in
  Alcotest.(check bool)
    "contains SReturn"
    true
    (expr_contains_ident e "SReturn")

let test_quote_stmt_barrier () =
  let stmt = Ir.SBarrier in
  let e = Sarek_quote_ir.quote_stmt ~loc:dummy_loc stmt in
  Alcotest.(check bool)
    "contains SBarrier"
    true
    (expr_contains_ident e "SBarrier")

let test_quote_stmt_warp_barrier () =
  let stmt = Ir.SWarpBarrier in
  let e = Sarek_quote_ir.quote_stmt ~loc:dummy_loc stmt in
  Alcotest.(check bool)
    "contains SWarpBarrier"
    true
    (expr_contains_ident e "SWarpBarrier")

let test_quote_stmt_expr () =
  let stmt = Ir.SExpr (Ir.EConst (Ir.CInt32 42l)) in
  let e = Sarek_quote_ir.quote_stmt ~loc:dummy_loc stmt in
  Alcotest.(check bool) "contains SExpr" true (expr_contains_ident e "SExpr")

let test_quote_stmt_empty () =
  let stmt = Ir.SEmpty in
  let e = Sarek_quote_ir.quote_stmt ~loc:dummy_loc stmt in
  Alcotest.(check bool) "contains SEmpty" true (expr_contains_ident e "SEmpty")

let test_quote_stmt_let () =
  let v =
    Ir.{var_name = "x"; var_id = 0; var_type = TInt32; var_mutable = false}
  in
  let stmt = Ir.SLet (v, Ir.EConst (Ir.CInt32 42l), Ir.SEmpty) in
  let e = Sarek_quote_ir.quote_stmt ~loc:dummy_loc stmt in
  Alcotest.(check bool) "contains SLet" true (expr_contains_ident e "SLet")

let test_quote_stmt_let_mut () =
  let v =
    Ir.{var_name = "x"; var_id = 0; var_type = TInt32; var_mutable = true}
  in
  let stmt = Ir.SLetMut (v, Ir.EConst (Ir.CInt32 42l), Ir.SEmpty) in
  let e = Sarek_quote_ir.quote_stmt ~loc:dummy_loc stmt in
  Alcotest.(check bool)
    "contains SLetMut"
    true
    (expr_contains_ident e "SLetMut")

let test_quote_stmt_pragma () =
  let stmt = Ir.SPragma (["unroll"], Ir.SEmpty) in
  let e = Sarek_quote_ir.quote_stmt ~loc:dummy_loc stmt in
  Alcotest.(check bool)
    "contains SPragma"
    true
    (expr_contains_ident e "SPragma")

let test_quote_stmt_mem_fence () =
  let stmt = Ir.SMemFence in
  let e = Sarek_quote_ir.quote_stmt ~loc:dummy_loc stmt in
  Alcotest.(check bool)
    "contains SMemFence"
    true
    (expr_contains_ident e "SMemFence")

let test_quote_stmt_block () =
  let stmt = Ir.SBlock Ir.SEmpty in
  let e = Sarek_quote_ir.quote_stmt ~loc:dummy_loc stmt in
  Alcotest.(check bool) "contains SBlock" true (expr_contains_ident e "SBlock")

(** {1 Decl Tests} *)

let test_quote_decl_param () =
  let v =
    Ir.{var_name = "x"; var_id = 0; var_type = TInt32; var_mutable = false}
  in
  let decl = Ir.DParam (v, None) in
  let e = Sarek_quote_ir.quote_decl ~loc:dummy_loc decl in
  Alcotest.(check bool) "contains DParam" true (expr_contains_ident e "DParam")

let test_quote_decl_param_with_array_info () =
  let v =
    Ir.
      {
        var_name = "arr";
        var_id = 0;
        var_type = TArray (TInt32, Global);
        var_mutable = false;
      }
  in
  let ai = Ir.{arr_elttype = TInt32; arr_memspace = Global} in
  let decl = Ir.DParam (v, Some ai) in
  let e = Sarek_quote_ir.quote_decl ~loc:dummy_loc decl in
  (* Check that the expression is a valid DParam constructor - quote_option wraps array_info in Some *)
  let s = Format.asprintf "%a" Pprintast.expression e in
  Alcotest.(check bool) "generates DParam expression" true (String.length s > 50)

let test_quote_decl_local () =
  let v =
    Ir.{var_name = "x"; var_id = 0; var_type = TInt32; var_mutable = true}
  in
  let decl = Ir.DLocal (v, Some (Ir.EConst (Ir.CInt32 0l))) in
  let e = Sarek_quote_ir.quote_decl ~loc:dummy_loc decl in
  Alcotest.(check bool) "contains DLocal" true (expr_contains_ident e "DLocal")

let test_quote_decl_shared () =
  let decl =
    Ir.DShared ("smem", Ir.TInt32, Some (Ir.EConst (Ir.CInt32 256l)))
  in
  let e = Sarek_quote_ir.quote_decl ~loc:dummy_loc decl in
  Alcotest.(check bool)
    "contains DShared"
    true
    (expr_contains_ident e "DShared")

(** {1 Kernel Tests} *)

let test_quote_kernel_simple () =
  let kern =
    Ir.
      {
        kern_name = "test_kernel";
        kern_params = [];
        kern_locals = [];
        kern_body = Ir.SEmpty;
        kern_types = [];
        kern_variants = [];
        kern_funcs = [];
        kern_native_fn = None;
      }
  in
  let e = Sarek_quote_ir.quote_kernel ~loc:dummy_loc kern in
  (* Just check it produces an expression *)
  Alcotest.(check bool)
    "produces expression"
    true
    (String.length (Format.asprintf "%a" Pprintast.expression e) > 0)

let test_quote_kernel_with_params () =
  let v =
    Ir.{var_name = "x"; var_id = 0; var_type = TInt32; var_mutable = false}
  in
  let kern =
    Ir.
      {
        kern_name = "kernel_with_params";
        kern_params = [Ir.DParam (v, None)];
        kern_locals = [];
        kern_body = Ir.SEmpty;
        kern_types = [];
        kern_variants = [];
        kern_funcs = [];
        kern_native_fn = None;
      }
  in
  let e = Sarek_quote_ir.quote_kernel ~loc:dummy_loc kern in
  Alcotest.(check bool)
    "produces expression"
    true
    (String.length (Format.asprintf "%a" Pprintast.expression e) > 0)

let test_quote_kernel_with_types () =
  let kern =
    Ir.
      {
        kern_name = "kernel_with_types";
        kern_params = [];
        kern_locals = [];
        kern_body = Ir.SEmpty;
        kern_types = [("Point", [("x", Ir.TInt32); ("y", Ir.TInt32)])];
        kern_variants = [];
        kern_funcs = [];
        kern_native_fn = None;
      }
  in
  let e = Sarek_quote_ir.quote_kernel ~loc:dummy_loc kern in
  Alcotest.(check bool)
    "produces expression"
    true
    (String.length (Format.asprintf "%a" Pprintast.expression e) > 0)

let test_quote_kernel_with_variants () =
  let kern =
    Ir.
      {
        kern_name = "kernel_with_variants";
        kern_params = [];
        kern_locals = [];
        kern_body = Ir.SEmpty;
        kern_types = [];
        kern_variants = [("Option", [("Some", [Ir.TInt32]); ("None", [])])];
        kern_funcs = [];
        kern_native_fn = None;
      }
  in
  let e = Sarek_quote_ir.quote_kernel ~loc:dummy_loc kern in
  Alcotest.(check bool)
    "produces expression"
    true
    (String.length (Format.asprintf "%a" Pprintast.expression e) > 0)

let test_quote_kernel_with_helper_funcs () =
  let v =
    Ir.{var_name = "x"; var_id = 0; var_type = TInt32; var_mutable = false}
  in
  let hf =
    Ir.
      {
        hf_name = "helper";
        hf_params = [v];
        hf_ret_type = Ir.TInt32;
        hf_body = Ir.SReturn (Ir.EVar v);
      }
  in
  let kern =
    Ir.
      {
        kern_name = "kernel_with_helpers";
        kern_params = [];
        kern_locals = [];
        kern_body = Ir.SEmpty;
        kern_types = [];
        kern_variants = [];
        kern_funcs = [hf];
        kern_native_fn = None;
      }
  in
  let e = Sarek_quote_ir.quote_kernel ~loc:dummy_loc kern in
  Alcotest.(check bool)
    "produces expression"
    true
    (String.length (Format.asprintf "%a" Pprintast.expression e) > 0)

(** {1 Test Suite} *)

let () =
  let open Alcotest in
  run
    "Sarek_quote_ir"
    [
      ( "memspace",
        [
          test_case "Global" `Quick test_quote_memspace_global;
          test_case "Shared" `Quick test_quote_memspace_shared;
          test_case "Local" `Quick test_quote_memspace_local;
        ] );
      ( "elttype",
        [
          test_case "TInt32" `Quick test_quote_elttype_int32;
          test_case "TInt64" `Quick test_quote_elttype_int64;
          test_case "TFloat32" `Quick test_quote_elttype_float32;
          test_case "TFloat64" `Quick test_quote_elttype_float64;
          test_case "TBool" `Quick test_quote_elttype_bool;
          test_case "TUnit" `Quick test_quote_elttype_unit;
          test_case "TRecord" `Quick test_quote_elttype_record;
          test_case "TVariant" `Quick test_quote_elttype_variant;
          test_case "TArray" `Quick test_quote_elttype_array;
          test_case "TVec" `Quick test_quote_elttype_vec;
        ] );
      ( "binop",
        [
          test_case "Add" `Quick test_quote_binop_add;
          test_case "Sub" `Quick test_quote_binop_sub;
          test_case "Mul" `Quick test_quote_binop_mul;
          test_case "Div" `Quick test_quote_binop_div;
          test_case "Mod" `Quick test_quote_binop_mod;
          test_case "Eq" `Quick test_quote_binop_eq;
          test_case "logical" `Quick test_quote_binop_logical;
          test_case "bitwise" `Quick test_quote_binop_bitwise;
        ] );
      ( "unop",
        [
          test_case "Neg" `Quick test_quote_unop_neg;
          test_case "Not" `Quick test_quote_unop_not;
          test_case "BitNot" `Quick test_quote_unop_bitnot;
        ] );
      ( "const",
        [
          test_case "CInt32" `Quick test_quote_const_int32;
          test_case "CInt64" `Quick test_quote_const_int64;
          test_case "CFloat32" `Quick test_quote_const_float32;
          test_case "CFloat64" `Quick test_quote_const_float64;
          test_case "CBool" `Quick test_quote_const_bool;
          test_case "CUnit" `Quick test_quote_const_unit;
        ] );
      ( "var",
        [
          test_case "basic" `Quick test_quote_var;
          test_case "mutable" `Quick test_quote_var_mutable;
        ] );
      ( "pattern",
        [
          test_case "PConstr" `Quick test_quote_pattern_constr;
          test_case "PWild" `Quick test_quote_pattern_wild;
        ] );
      ( "expr",
        [
          test_case "EConst" `Quick test_quote_expr_const;
          test_case "EVar" `Quick test_quote_expr_var;
          test_case "EBinop" `Quick test_quote_expr_binop;
          test_case "EUnop" `Quick test_quote_expr_unop;
          test_case "EArrayRead" `Quick test_quote_expr_array_read;
          test_case "ERecordField" `Quick test_quote_expr_record_field;
          test_case "EIntrinsic" `Quick test_quote_expr_intrinsic;
          test_case "ECast" `Quick test_quote_expr_cast;
          test_case "ETuple" `Quick test_quote_expr_tuple;
          test_case "ERecord" `Quick test_quote_expr_record;
          test_case "EVariant" `Quick test_quote_expr_variant;
          test_case "EArrayLen" `Quick test_quote_expr_array_len;
          test_case "EArrayCreate" `Quick test_quote_expr_array_create;
          test_case "EIf" `Quick test_quote_expr_if;
          test_case "EMatch" `Quick test_quote_expr_match;
        ] );
      ( "lvalue",
        [
          test_case "LVar" `Quick test_quote_lvalue_var;
          test_case "LArrayElem" `Quick test_quote_lvalue_array_elem;
          test_case "LRecordField" `Quick test_quote_lvalue_record_field;
        ] );
      ( "stmt",
        [
          test_case "SAssign" `Quick test_quote_stmt_assign;
          test_case "SSeq" `Quick test_quote_stmt_seq;
          test_case "SIf" `Quick test_quote_stmt_if;
          test_case "SWhile" `Quick test_quote_stmt_while;
          test_case "SFor" `Quick test_quote_stmt_for;
          test_case "SMatch" `Quick test_quote_stmt_match;
          test_case "SReturn" `Quick test_quote_stmt_return;
          test_case "SBarrier" `Quick test_quote_stmt_barrier;
          test_case "SWarpBarrier" `Quick test_quote_stmt_warp_barrier;
          test_case "SExpr" `Quick test_quote_stmt_expr;
          test_case "SEmpty" `Quick test_quote_stmt_empty;
          test_case "SLet" `Quick test_quote_stmt_let;
          test_case "SLetMut" `Quick test_quote_stmt_let_mut;
          test_case "SPragma" `Quick test_quote_stmt_pragma;
          test_case "SMemFence" `Quick test_quote_stmt_mem_fence;
          test_case "SBlock" `Quick test_quote_stmt_block;
        ] );
      ( "decl",
        [
          test_case "DParam" `Quick test_quote_decl_param;
          test_case
            "DParam with array_info"
            `Quick
            test_quote_decl_param_with_array_info;
          test_case "DLocal" `Quick test_quote_decl_local;
          test_case "DShared" `Quick test_quote_decl_shared;
        ] );
      ( "kernel",
        [
          test_case "simple" `Quick test_quote_kernel_simple;
          test_case "with params" `Quick test_quote_kernel_with_params;
          test_case "with types" `Quick test_quote_kernel_with_types;
          test_case "with variants" `Quick test_quote_kernel_with_variants;
          test_case
            "with helper funcs"
            `Quick
            test_quote_kernel_with_helper_funcs;
        ] );
    ]
