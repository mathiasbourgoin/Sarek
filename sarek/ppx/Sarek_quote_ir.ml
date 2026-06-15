(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Sarek PPX - GPU kernel DSL for OCaml
 *
 * This module quotes Sarek_ir_ppx values to OCaml AST expressions that
 * construct Sarek.Sarek_ir values at runtime.
 ******************************************************************************)

open Ppxlib
module Ir = Sarek_ir_ppx

(** Quote helpers *)
let quote_int ~loc i = Ast_builder.Default.eint ~loc i

let quote_int32 ~loc i =
  [%expr Int32.of_int [%e Ast_builder.Default.eint ~loc (Int32.to_int i)]]

let quote_int64 ~loc i =
  [%expr Int64.of_int [%e Ast_builder.Default.eint ~loc (Int64.to_int i)]]

let quote_float ~loc f = Ast_builder.Default.efloat ~loc (string_of_float f)

let quote_string ~loc s = Ast_builder.Default.estring ~loc s

let quote_bool ~loc b = if b then [%expr true] else [%expr false]

let quote_list ~loc quote_elem elems =
  List.fold_right
    (fun elem acc -> [%expr [%e quote_elem ~loc elem] :: [%e acc]])
    elems
    [%expr []]

let quote_option ~loc quote_elem = function
  | None -> [%expr None]
  | Some x -> [%expr Some [%e quote_elem ~loc x]]

(** Quote memspace *)
let quote_memspace ~loc (m : Ir.memspace) : expression =
  match m with
  | Ir.Global -> [%expr Sarek.Sarek_ir.Global]
  | Ir.Shared -> [%expr Sarek.Sarek_ir.Shared]
  | Ir.Local -> [%expr Sarek.Sarek_ir.Local]

(** Quote elttype *)
let rec quote_elttype ~loc (t : Ir.elttype) : expression =
  match t with
  | Ir.TInt32 -> [%expr Sarek.Sarek_ir.TInt32]
  | Ir.TInt64 -> [%expr Sarek.Sarek_ir.TInt64]
  | Ir.TFloat32 -> [%expr Sarek.Sarek_ir.TFloat32]
  | Ir.TFloat64 -> [%expr Sarek.Sarek_ir.TFloat64]
  | Ir.TBool -> [%expr Sarek.Sarek_ir.TBool]
  | Ir.TUnit -> [%expr Sarek.Sarek_ir.TUnit]
  | Ir.TRecord (name, fields) ->
      let fields_expr =
        quote_list
          ~loc
          (fun ~loc (n, ty) ->
            [%expr [%e quote_string ~loc n], [%e quote_elttype ~loc ty]])
          fields
      in
      [%expr
        Sarek.Sarek_ir.TRecord ([%e quote_string ~loc name], [%e fields_expr])]
  | Ir.TVariant (name, constrs) ->
      let constrs_expr =
        quote_list
          ~loc
          (fun ~loc (n, tys) ->
            [%expr
              [%e quote_string ~loc n], [%e quote_list ~loc quote_elttype tys]])
          constrs
      in
      [%expr
        Sarek.Sarek_ir.TVariant ([%e quote_string ~loc name], [%e constrs_expr])]
  | Ir.TArray (elt, mem) ->
      [%expr
        Sarek.Sarek_ir.TArray
          ([%e quote_elttype ~loc elt], [%e quote_memspace ~loc mem])]
  | Ir.TVec elt -> [%expr Sarek.Sarek_ir.TVec [%e quote_elttype ~loc elt]]

(** Quote var *)
let quote_var ~loc (v : Ir.var) : expression =
  [%expr
    {
      Sarek.Sarek_ir.var_name = [%e quote_string ~loc v.var_name];
      var_id = [%e quote_int ~loc v.var_id];
      var_type = [%e quote_elttype ~loc v.var_type];
      var_mutable = [%e quote_bool ~loc v.var_mutable];
    }]

(** Quote const *)
let quote_const ~loc (c : Ir.const) : expression =
  match c with
  | Ir.CInt32 i -> [%expr Sarek.Sarek_ir.CInt32 [%e quote_int32 ~loc i]]
  | Ir.CInt64 i -> [%expr Sarek.Sarek_ir.CInt64 [%e quote_int64 ~loc i]]
  | Ir.CFloat32 f -> [%expr Sarek.Sarek_ir.CFloat32 [%e quote_float ~loc f]]
  | Ir.CFloat64 f -> [%expr Sarek.Sarek_ir.CFloat64 [%e quote_float ~loc f]]
  | Ir.CBool b -> [%expr Sarek.Sarek_ir.CBool [%e quote_bool ~loc b]]
  | Ir.CUnit -> [%expr Sarek.Sarek_ir.CUnit]

(** Quote binop *)
let quote_binop ~loc (op : Ir.binop) : expression =
  match op with
  | Ir.Add -> [%expr Sarek.Sarek_ir.Add]
  | Ir.Sub -> [%expr Sarek.Sarek_ir.Sub]
  | Ir.Mul -> [%expr Sarek.Sarek_ir.Mul]
  | Ir.Div -> [%expr Sarek.Sarek_ir.Div]
  | Ir.Mod -> [%expr Sarek.Sarek_ir.Mod]
  | Ir.Eq -> [%expr Sarek.Sarek_ir.Eq]
  | Ir.Ne -> [%expr Sarek.Sarek_ir.Ne]
  | Ir.Lt -> [%expr Sarek.Sarek_ir.Lt]
  | Ir.Le -> [%expr Sarek.Sarek_ir.Le]
  | Ir.Gt -> [%expr Sarek.Sarek_ir.Gt]
  | Ir.Ge -> [%expr Sarek.Sarek_ir.Ge]
  | Ir.And -> [%expr Sarek.Sarek_ir.And]
  | Ir.Or -> [%expr Sarek.Sarek_ir.Or]
  | Ir.Shl -> [%expr Sarek.Sarek_ir.Shl]
  | Ir.Shr -> [%expr Sarek.Sarek_ir.Shr]
  | Ir.BitAnd -> [%expr Sarek.Sarek_ir.BitAnd]
  | Ir.BitOr -> [%expr Sarek.Sarek_ir.BitOr]
  | Ir.BitXor -> [%expr Sarek.Sarek_ir.BitXor]

(** Quote unop *)
let quote_unop ~loc (op : Ir.unop) : expression =
  match op with
  | Ir.Neg -> [%expr Sarek.Sarek_ir.Neg]
  | Ir.Not -> [%expr Sarek.Sarek_ir.Not]
  | Ir.BitNot -> [%expr Sarek.Sarek_ir.BitNot]

(** Quote for_dir *)
let quote_for_dir ~loc (d : Ir.for_dir) : expression =
  match d with
  | Ir.Upto -> [%expr Sarek.Sarek_ir.Upto]
  | Ir.Downto -> [%expr Sarek.Sarek_ir.Downto]

(** Quote pattern - must come before quote_expr since EMatch uses it *)
let quote_pattern ~loc (p : Ir.pattern) : expression =
  match p with
  | Ir.PConstr (name, vars) ->
      [%expr
        Sarek.Sarek_ir.PConstr
          ([%e quote_string ~loc name], [%e quote_list ~loc quote_string vars])]
  | Ir.PWild -> [%expr Sarek.Sarek_ir.PWild]

(** Quote expr *)
let rec quote_expr ~loc (e : Ir.expr) : expression =
  match e with
  | Ir.EConst c -> [%expr Sarek.Sarek_ir.EConst [%e quote_const ~loc c]]
  | Ir.EVar v -> [%expr Sarek.Sarek_ir.EVar [%e quote_var ~loc v]]
  | Ir.EBinop (op, a, b) ->
      [%expr
        Sarek.Sarek_ir.EBinop
          ( [%e quote_binop ~loc op],
            [%e quote_expr ~loc a],
            [%e quote_expr ~loc b] )]
  | Ir.EUnop (op, a) ->
      [%expr
        Sarek.Sarek_ir.EUnop ([%e quote_unop ~loc op], [%e quote_expr ~loc a])]
  | Ir.EArrayRead (name, idx) ->
      [%expr
        Sarek.Sarek_ir.EArrayRead
          ([%e quote_string ~loc name], [%e quote_expr ~loc idx])]
  | Ir.ERecordField (r, f) ->
      [%expr
        Sarek.Sarek_ir.ERecordField
          ([%e quote_expr ~loc r], [%e quote_string ~loc f])]
  | Ir.EIntrinsic (path, name, args) ->
      [%expr
        Sarek.Sarek_ir.EIntrinsic
          ( [%e quote_list ~loc quote_string path],
            [%e quote_string ~loc name],
            [%e quote_list ~loc quote_expr args] )]
  | Ir.ECast (ty, e) ->
      [%expr
        Sarek.Sarek_ir.ECast ([%e quote_elttype ~loc ty], [%e quote_expr ~loc e])]
  | Ir.ETuple es ->
      [%expr Sarek.Sarek_ir.ETuple [%e quote_list ~loc quote_expr es]]
  | Ir.EApp (f, args) ->
      [%expr
        Sarek.Sarek_ir.EApp
          ([%e quote_expr ~loc f], [%e quote_list ~loc quote_expr args])]
  | Ir.ERecord (name, fields) ->
      let fields_expr =
        quote_list
          ~loc
          (fun ~loc (n, e) ->
            [%expr [%e quote_string ~loc n], [%e quote_expr ~loc e]])
          fields
      in
      [%expr
        Sarek.Sarek_ir.ERecord ([%e quote_string ~loc name], [%e fields_expr])]
  | Ir.EVariant (ty, constr, args) ->
      [%expr
        Sarek.Sarek_ir.EVariant
          ( [%e quote_string ~loc ty],
            [%e quote_string ~loc constr],
            [%e quote_list ~loc quote_expr args] )]
  | Ir.EArrayLen name ->
      [%expr Sarek.Sarek_ir.EArrayLen [%e quote_string ~loc name]]
  | Ir.EArrayCreate (elem_ty, size, mem) ->
      [%expr
        Sarek.Sarek_ir.EArrayCreate
          ( [%e quote_elttype ~loc elem_ty],
            [%e quote_expr ~loc size],
            [%e quote_memspace ~loc mem] )]
  | Ir.EArrayReadExpr (base, idx) ->
      [%expr
        Sarek.Sarek_ir.EArrayReadExpr
          ([%e quote_expr ~loc base], [%e quote_expr ~loc idx])]
  | Ir.EIf (cond, then_, else_) ->
      [%expr
        Sarek.Sarek_ir.EIf
          ( [%e quote_expr ~loc cond],
            [%e quote_expr ~loc then_],
            [%e quote_expr ~loc else_] )]
  | Ir.EMatch (e, cases) ->
      let cases_expr =
        quote_list
          ~loc
          (fun ~loc (p, body) ->
            [%expr [%e quote_pattern ~loc p], [%e quote_expr ~loc body]])
          cases
      in
      [%expr Sarek.Sarek_ir.EMatch ([%e quote_expr ~loc e], [%e cases_expr])]

(** Quote lvalue *)
let rec quote_lvalue ~loc (lv : Ir.lvalue) : expression =
  match lv with
  | Ir.LVar v -> [%expr Sarek.Sarek_ir.LVar [%e quote_var ~loc v]]
  | Ir.LArrayElem (name, idx) ->
      [%expr
        Sarek.Sarek_ir.LArrayElem
          ([%e quote_string ~loc name], [%e quote_expr ~loc idx])]
  | Ir.LRecordField (lv, f) ->
      [%expr
        Sarek.Sarek_ir.LRecordField
          ([%e quote_lvalue ~loc lv], [%e quote_string ~loc f])]
  | Ir.LArrayElemExpr (base, idx) ->
      [%expr
        Sarek.Sarek_ir.LArrayElemExpr
          ([%e quote_expr ~loc base], [%e quote_expr ~loc idx])]

(** Quote stmt *)
let rec quote_stmt ~loc (s : Ir.stmt) : expression =
  match s with
  | Ir.SAssign (lv, e) ->
      [%expr
        Sarek.Sarek_ir.SAssign
          ([%e quote_lvalue ~loc lv], [%e quote_expr ~loc e])]
  | Ir.SSeq ss -> [%expr Sarek.Sarek_ir.SSeq [%e quote_list ~loc quote_stmt ss]]
  | Ir.SIf (c, t, e_opt) ->
      [%expr
        Sarek.Sarek_ir.SIf
          ( [%e quote_expr ~loc c],
            [%e quote_stmt ~loc t],
            [%e quote_option ~loc quote_stmt e_opt] )]
  | Ir.SWhile (c, b) ->
      [%expr
        Sarek.Sarek_ir.SWhile ([%e quote_expr ~loc c], [%e quote_stmt ~loc b])]
  | Ir.SFor (v, lo, hi, dir, body) ->
      [%expr
        Sarek.Sarek_ir.SFor
          ( [%e quote_var ~loc v],
            [%e quote_expr ~loc lo],
            [%e quote_expr ~loc hi],
            [%e quote_for_dir ~loc dir],
            [%e quote_stmt ~loc body] )]
  | Ir.SMatch (e, cases) ->
      let cases_expr =
        quote_list
          ~loc
          (fun ~loc (p, s) ->
            [%expr [%e quote_pattern ~loc p], [%e quote_stmt ~loc s]])
          cases
      in
      [%expr Sarek.Sarek_ir.SMatch ([%e quote_expr ~loc e], [%e cases_expr])]
  | Ir.SReturn e -> [%expr Sarek.Sarek_ir.SReturn [%e quote_expr ~loc e]]
  | Ir.SBarrier -> [%expr Sarek.Sarek_ir.SBarrier]
  | Ir.SWarpBarrier -> [%expr Sarek.Sarek_ir.SWarpBarrier]
  | Ir.SExpr e -> [%expr Sarek.Sarek_ir.SExpr [%e quote_expr ~loc e]]
  | Ir.SEmpty -> [%expr Sarek.Sarek_ir.SEmpty]
  | Ir.SLet (v, e, body) ->
      [%expr
        Sarek.Sarek_ir.SLet
          ( [%e quote_var ~loc v],
            [%e quote_expr ~loc e],
            [%e quote_stmt ~loc body] )]
  | Ir.SLetMut (v, e, body) ->
      [%expr
        Sarek.Sarek_ir.SLetMut
          ( [%e quote_var ~loc v],
            [%e quote_expr ~loc e],
            [%e quote_stmt ~loc body] )]
  | Ir.SPragma (opts, body) ->
      [%expr
        Sarek.Sarek_ir.SPragma
          ([%e quote_list ~loc quote_string opts], [%e quote_stmt ~loc body])]
  | Ir.SMemFence -> [%expr Sarek.Sarek_ir.SMemFence]
  | Ir.SBlock body -> [%expr Sarek.Sarek_ir.SBlock [%e quote_stmt ~loc body]]
  | Ir.SNative {gpu; ocaml} ->
      [%expr
        Sarek.Sarek_ir.SNative {gpu = [%e gpu]; ocaml = {run = [%e ocaml]}}]

(** Quote array_info *)
let quote_array_info ~loc (ai : Ir.array_info) : expression =
  [%expr
    {
      Sarek.Sarek_ir.arr_elttype = [%e quote_elttype ~loc ai.arr_elttype];
      arr_memspace = [%e quote_memspace ~loc ai.arr_memspace];
    }]

(** Quote decl *)
let quote_decl ~loc (d : Ir.decl) : expression =
  match d with
  | Ir.DParam (v, ai_opt) ->
      [%expr
        Sarek.Sarek_ir.DParam
          ([%e quote_var ~loc v], [%e quote_option ~loc quote_array_info ai_opt])]
  | Ir.DLocal (v, e_opt) ->
      [%expr
        Sarek.Sarek_ir.DLocal
          ([%e quote_var ~loc v], [%e quote_option ~loc quote_expr e_opt])]
  | Ir.DShared (name, elt, size_opt) ->
      [%expr
        Sarek.Sarek_ir.DShared
          ( [%e quote_string ~loc name],
            [%e quote_elttype ~loc elt],
            [%e quote_option ~loc quote_expr size_opt] )]

(** Quote a type definition (name, field list) *)
let quote_type_def ~loc (name, fields) : expression =
  let fields_expr =
    quote_list
      ~loc
      (fun ~loc (n, ty) ->
        [%expr [%e quote_string ~loc n], [%e quote_elttype ~loc ty]])
      fields
  in
  [%expr [%e quote_string ~loc name], [%e fields_expr]]

(** Quote a variant definition (name, constructors with payload types) *)
let quote_variant_def ~loc (name, constrs) : expression =
  let constrs_expr =
    quote_list
      ~loc
      (fun ~loc (cname, payload_types) ->
        let payloads = quote_list ~loc quote_elttype payload_types in
        [%expr [%e quote_string ~loc cname], [%e payloads]])
      constrs
  in
  [%expr [%e quote_string ~loc name], [%e constrs_expr]]

(** Quote helper_func *)
let quote_helper_func ~loc (hf : Ir.helper_func) : expression =
  [%expr
    {
      Sarek.Sarek_ir.hf_name = [%e quote_string ~loc hf.hf_name];
      hf_params = [%e quote_list ~loc quote_var hf.hf_params];
      hf_ret_type = [%e quote_elttype ~loc hf.hf_ret_type];
      hf_body = [%e quote_stmt ~loc hf.hf_body];
    }]

(** Quote kernel.
    @param native_fn_expr
      Optional expression that generates the native function (adapted from
      cpu_kern). If provided, it should have type: parallel:bool ->
      block:int*int*int -> grid:int*int*int -> Kernel_arg.t array -> unit *)
let quote_kernel ~loc ?(native_fn_expr : expression option) (k : Ir.kernel) :
    expression =
  let native_fn_field =
    match native_fn_expr with
    | Some e -> [%expr Some (Sarek.Sarek_ir.NativeFn [%e e])]
    | None -> [%expr None]
  in
  [%expr
    {
      Sarek.Sarek_ir.kern_name = [%e quote_string ~loc k.kern_name];
      kern_params = [%e quote_list ~loc quote_decl k.kern_params];
      kern_locals = [%e quote_list ~loc quote_decl k.kern_locals];
      kern_body = [%e quote_stmt ~loc k.kern_body];
      kern_types = [%e quote_list ~loc quote_type_def k.kern_types];
      kern_variants = [%e quote_list ~loc quote_variant_def k.kern_variants];
      kern_funcs = [%e quote_list ~loc quote_helper_func k.kern_funcs];
      kern_native_fn = [%e native_fn_field];
    }]
