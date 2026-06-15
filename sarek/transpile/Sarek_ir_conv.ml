(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Sarek_ir_conv - Convert Sarek_ir_ppx.kernel -> Sarek_ir_types.kernel
 *
 * Both type hierarchies are structurally identical.  This converter exists
 * because Sarek_ir_ppx is defined in sarek_frontend (PPX compile-time types)
 * while Sarek_ir_types is in sarek_ir (runtime types used by the code
 * generators).  The transpile path needs to bridge between them.
 *
 * SNative nodes must never appear here: of_source rejects [%native] before
 * lowering.  The converter raises Invalid_argument if one is encountered.
 ******************************************************************************)

module P = Sarek_ir_ppx
module T = Sarek_ir_types

let conv_memspace : P.memspace -> T.memspace = function
  | P.Global -> T.Global
  | P.Shared -> T.Shared
  | P.Local -> T.Local

let rec conv_elttype : P.elttype -> T.elttype = function
  | P.TInt32 -> T.TInt32
  | P.TInt64 -> T.TInt64
  | P.TFloat32 -> T.TFloat32
  | P.TFloat64 -> T.TFloat64
  | P.TBool -> T.TBool
  | P.TUnit -> T.TUnit
  | P.TRecord (name, fields) ->
      T.TRecord (name, List.map (fun (f, et) -> (f, conv_elttype et)) fields)
  | P.TVariant (name, constrs) ->
      T.TVariant
        ( name,
          List.map
            (fun (cname, ets) -> (cname, List.map conv_elttype ets))
            constrs )
  | P.TArray (et, ms) -> T.TArray (conv_elttype et, conv_memspace ms)
  | P.TVec et -> T.TVec (conv_elttype et)

let conv_var (v : P.var) : T.var =
  {
    T.var_name = v.P.var_name;
    T.var_id = v.P.var_id;
    T.var_type = conv_elttype v.P.var_type;
    T.var_mutable = v.P.var_mutable;
  }

let conv_const : P.const -> T.const = function
  | P.CInt32 i -> T.CInt32 i
  | P.CInt64 i -> T.CInt64 i
  | P.CFloat32 f -> T.CFloat32 f
  | P.CFloat64 f -> T.CFloat64 f
  | P.CBool b -> T.CBool b
  | P.CUnit -> T.CUnit

let conv_binop : P.binop -> T.binop = function
  | P.Add -> T.Add
  | P.Sub -> T.Sub
  | P.Mul -> T.Mul
  | P.Div -> T.Div
  | P.Mod -> T.Mod
  | P.Eq -> T.Eq
  | P.Ne -> T.Ne
  | P.Lt -> T.Lt
  | P.Le -> T.Le
  | P.Gt -> T.Gt
  | P.Ge -> T.Ge
  | P.And -> T.And
  | P.Or -> T.Or
  | P.Shl -> T.Shl
  | P.Shr -> T.Shr
  | P.BitAnd -> T.BitAnd
  | P.BitOr -> T.BitOr
  | P.BitXor -> T.BitXor

let conv_unop : P.unop -> T.unop = function
  | P.Neg -> T.Neg
  | P.Not -> T.Not
  | P.BitNot -> T.BitNot

let conv_for_dir : P.for_dir -> T.for_dir = function
  | P.Upto -> T.Upto
  | P.Downto -> T.Downto

let conv_pattern : P.pattern -> T.pattern = function
  | P.PConstr (name, vars) -> T.PConstr (name, vars)
  | P.PWild -> T.PWild

let rec conv_expr : P.expr -> T.expr = function
  | P.EConst c -> T.EConst (conv_const c)
  | P.EVar v -> T.EVar (conv_var v)
  | P.EBinop (op, a, b) -> T.EBinop (conv_binop op, conv_expr a, conv_expr b)
  | P.EUnop (op, e) -> T.EUnop (conv_unop op, conv_expr e)
  | P.EArrayRead (name, idx) -> T.EArrayRead (name, conv_expr idx)
  | P.EArrayReadExpr (base, idx) ->
      T.EArrayReadExpr (conv_expr base, conv_expr idx)
  | P.ERecordField (e, field) -> T.ERecordField (conv_expr e, field)
  | P.EIntrinsic (path, name, args) ->
      T.EIntrinsic (path, name, List.map conv_expr args)
  | P.ECast (et, e) -> T.ECast (conv_elttype et, conv_expr e)
  | P.ETuple es -> T.ETuple (List.map conv_expr es)
  | P.EApp (f, args) -> T.EApp (conv_expr f, List.map conv_expr args)
  | P.ERecord (name, fields) ->
      T.ERecord (name, List.map (fun (f, e) -> (f, conv_expr e)) fields)
  | P.EVariant (type_name, cname, args) ->
      T.EVariant (type_name, cname, List.map conv_expr args)
  | P.EArrayLen name -> T.EArrayLen name
  | P.EArrayCreate (et, sz, ms) ->
      T.EArrayCreate (conv_elttype et, conv_expr sz, conv_memspace ms)
  | P.EIf (c, t, e) -> T.EIf (conv_expr c, conv_expr t, conv_expr e)
  | P.EMatch (scrutinee, cases) ->
      T.EMatch
        ( conv_expr scrutinee,
          List.map (fun (pat, body) -> (conv_pattern pat, conv_expr body)) cases
        )

and conv_lvalue : P.lvalue -> T.lvalue = function
  | P.LVar v -> T.LVar (conv_var v)
  | P.LArrayElem (name, idx) -> T.LArrayElem (name, conv_expr idx)
  | P.LArrayElemExpr (base, idx) ->
      T.LArrayElemExpr (conv_expr base, conv_expr idx)
  | P.LRecordField (lv, field) -> T.LRecordField (conv_lvalue lv, field)

and conv_stmt : P.stmt -> T.stmt = function
  | P.SAssign (lv, e) -> T.SAssign (conv_lvalue lv, conv_expr e)
  | P.SSeq stmts -> T.SSeq (List.map conv_stmt stmts)
  | P.SIf (c, t, e_opt) ->
      T.SIf (conv_expr c, conv_stmt t, Option.map conv_stmt e_opt)
  | P.SWhile (c, body) -> T.SWhile (conv_expr c, conv_stmt body)
  | P.SFor (v, start_, end_, dir, body) ->
      T.SFor
        ( conv_var v,
          conv_expr start_,
          conv_expr end_,
          conv_for_dir dir,
          conv_stmt body )
  | P.SMatch (e, cases) ->
      T.SMatch
        ( conv_expr e,
          List.map (fun (pat, body) -> (conv_pattern pat, conv_stmt body)) cases
        )
  | P.SReturn e -> T.SReturn (conv_expr e)
  | P.SBarrier -> T.SBarrier
  | P.SWarpBarrier -> T.SWarpBarrier
  | P.SExpr e -> T.SExpr (conv_expr e)
  | P.SEmpty -> T.SEmpty
  | P.SLet (v, e, body) -> T.SLet (conv_var v, conv_expr e, conv_stmt body)
  | P.SLetMut (v, e, body) -> T.SLetMut (conv_var v, conv_expr e, conv_stmt body)
  | P.SPragma (opts, body) -> T.SPragma (opts, conv_stmt body)
  | P.SMemFence -> T.SMemFence
  | P.SBlock body -> T.SBlock (conv_stmt body)
  | P.SNative _ ->
      (* [%native] is rejected before lowering in of_source — should never
         reach here. *)
      invalid_arg
        "Sarek_ir_conv: SNative reached converter; [%native] must be rejected \
         before lowering"

let conv_decl : P.decl -> T.decl = function
  | P.DParam (v, arr_opt) ->
      T.DParam
        ( conv_var v,
          Option.map
            (fun ai ->
              {
                T.arr_elttype = conv_elttype ai.P.arr_elttype;
                T.arr_memspace = conv_memspace ai.P.arr_memspace;
              })
            arr_opt )
  | P.DLocal (v, init_opt) ->
      T.DLocal (conv_var v, Option.map conv_expr init_opt)
  | P.DShared (name, et, sz_opt) ->
      T.DShared (name, conv_elttype et, Option.map conv_expr sz_opt)

let conv_helper_func (hf : P.helper_func) : T.helper_func =
  {
    T.hf_name = hf.P.hf_name;
    T.hf_params = List.map conv_var hf.P.hf_params;
    T.hf_ret_type = conv_elttype hf.P.hf_ret_type;
    T.hf_body = conv_stmt hf.P.hf_body;
  }

(** Convert a [Sarek_ir_ppx.kernel] to a [Sarek_ir_types.kernel]. The
    [kern_native_fn] field is set to [None] — transpiled kernels never carry a
    native CPU function. *)
let conv_kernel (k : P.kernel) : T.kernel =
  {
    T.kern_name = k.P.kern_name;
    T.kern_params = List.map conv_decl k.P.kern_params;
    T.kern_locals = List.map conv_decl k.P.kern_locals;
    T.kern_body = conv_stmt k.P.kern_body;
    T.kern_types =
      List.map
        (fun (tname, fields) ->
          (tname, List.map (fun (fname, et) -> (fname, conv_elttype et)) fields))
        k.P.kern_types;
    T.kern_variants =
      List.map
        (fun (tname, constrs) ->
          ( tname,
            List.map
              (fun (cname, ets) -> (cname, List.map conv_elttype ets))
              constrs ))
        k.P.kern_variants;
    T.kern_funcs = List.map conv_helper_func k.P.kern_funcs;
    T.kern_native_fn = None;
  }
