(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Sarek PPX - GPU kernel DSL for OCaml
 *
 * This module quotes Kirc_Ast values back to OCaml AST, so the IR can be
 * embedded in the generated code.
 ******************************************************************************)

open Ppxlib
open Sarek_typed_ast
open Sarek_types

(** Helper to create an identifier expression *)
let evar ~loc name =
  Ast_builder.Default.pexp_ident ~loc {txt = Lident name; loc}

let evar_qualified ~loc path name =
  let lid =
    List.fold_left
      (fun acc m -> Ldot (acc, m))
      (Lident (List.hd path))
      (List.tl path @ [name])
  in
  Ast_builder.Default.pexp_ident ~loc {txt = lid; loc}

(** Quote an int as OCaml expression *)
let quote_int ~loc i = Ast_builder.Default.eint ~loc i

(** Quote an int32 as OCaml expression *)
let quote_int32 ~loc i =
  [%expr [%e Ast_builder.Default.eint ~loc (Int32.to_int i)]]

(** Quote a float as OCaml expression *)
let quote_float ~loc f = Ast_builder.Default.efloat ~loc (string_of_float f)

(** Quote a string as OCaml expression *)
let quote_string ~loc s = Ast_builder.Default.estring ~loc s

(** Quote a bool as OCaml expression *)
let quote_bool ~loc b = if b then [%expr true] else [%expr false]

(** Quote an elttype *)
let quote_elttype ~loc (e : Kirc_Ast.elttype) : expression =
  match e with
  | Kirc_Ast.EInt32 -> [%expr Sarek.Kirc_Ast.EInt32]
  | Kirc_Ast.EInt64 -> [%expr Sarek.Kirc_Ast.EInt64]
  | Kirc_Ast.EFloat32 -> [%expr Sarek.Kirc_Ast.EFloat32]
  | Kirc_Ast.EFloat64 -> [%expr Sarek.Kirc_Ast.EFloat64]

(** Quote a memspace *)
let quote_memspace ~loc (m : Kirc_Ast.memspace) : expression =
  match m with
  | Kirc_Ast.LocalSpace -> [%expr Sarek.Kirc_Ast.LocalSpace]
  | Kirc_Ast.Global -> [%expr Sarek.Kirc_Ast.Global]
  | Kirc_Ast.Shared -> [%expr Sarek.Kirc_Ast.Shared]

(** Quote a list *)
let quote_list ~loc quote_elem elems =
  List.fold_right
    (fun elem acc -> [%expr [%e quote_elem ~loc elem] :: [%e acc]])
    elems
    [%expr []]

(** Quote an array *)
let quote_array ~loc quote_elem elems =
  [%expr Array.of_list [%e quote_list ~loc quote_elem (Array.to_list elems)]]

(** Quote an option *)
let quote_option ~loc quote_elem = function
  | None -> [%expr None]
  | Some x -> [%expr Some [%e quote_elem ~loc x]]

(** Quote a case *)
let rec quote_case ~loc ((i, opt, body) : Kirc_Ast.case) : expression =
  let opt_expr =
    match opt with
    | None -> [%expr None]
    | Some (s1, s2, id, s3) ->
        [%expr
          Some
            ( [%e quote_string ~loc s1],
              [%e quote_string ~loc s2],
              [%e quote_int ~loc id],
              [%e quote_string ~loc s3] )]
  in
  [%expr [%e quote_int ~loc i], [%e opt_expr], [%e quote_k_ext ~loc body]]

(** Quote a Kirc_Ast.k_ext value to OCaml expression *)
and quote_k_ext ~loc (k : Kirc_Ast.k_ext) : expression =
  match k with
  | Kirc_Ast.Kern (params, body) ->
      [%expr
        Sarek.Kirc_Ast.Kern
          ([%e quote_k_ext ~loc params], [%e quote_k_ext ~loc body])]
  | Kirc_Ast.Block b -> [%expr Sarek.Kirc_Ast.Block [%e quote_k_ext ~loc b]]
  | Kirc_Ast.Params p -> [%expr Sarek.Kirc_Ast.Params [%e quote_k_ext ~loc p]]
  | Kirc_Ast.Plus (a, b) ->
      [%expr
        Sarek.Kirc_Ast.Plus ([%e quote_k_ext ~loc a], [%e quote_k_ext ~loc b])]
  | Kirc_Ast.Plusf (a, b) ->
      [%expr
        Sarek.Kirc_Ast.Plusf ([%e quote_k_ext ~loc a], [%e quote_k_ext ~loc b])]
  | Kirc_Ast.Min (a, b) ->
      [%expr
        Sarek.Kirc_Ast.Min ([%e quote_k_ext ~loc a], [%e quote_k_ext ~loc b])]
  | Kirc_Ast.Minf (a, b) ->
      [%expr
        Sarek.Kirc_Ast.Minf ([%e quote_k_ext ~loc a], [%e quote_k_ext ~loc b])]
  | Kirc_Ast.Mul (a, b) ->
      [%expr
        Sarek.Kirc_Ast.Mul ([%e quote_k_ext ~loc a], [%e quote_k_ext ~loc b])]
  | Kirc_Ast.Mulf (a, b) ->
      [%expr
        Sarek.Kirc_Ast.Mulf ([%e quote_k_ext ~loc a], [%e quote_k_ext ~loc b])]
  | Kirc_Ast.Div (a, b) ->
      [%expr
        Sarek.Kirc_Ast.Div ([%e quote_k_ext ~loc a], [%e quote_k_ext ~loc b])]
  | Kirc_Ast.Divf (a, b) ->
      [%expr
        Sarek.Kirc_Ast.Divf ([%e quote_k_ext ~loc a], [%e quote_k_ext ~loc b])]
  | Kirc_Ast.Mod (a, b) ->
      [%expr
        Sarek.Kirc_Ast.Mod ([%e quote_k_ext ~loc a], [%e quote_k_ext ~loc b])]
  | Kirc_Ast.Id s -> [%expr Sarek.Kirc_Ast.Id [%e quote_string ~loc s]]
  | Kirc_Ast.IdName s -> [%expr Sarek.Kirc_Ast.IdName [%e quote_string ~loc s]]
  | Kirc_Ast.GlobalFun (body, ret_type, name) ->
      [%expr
        Sarek.Kirc_Ast.GlobalFun
          ( [%e quote_k_ext ~loc body],
            [%e quote_string ~loc ret_type],
            [%e quote_string ~loc name] )]
  | Kirc_Ast.IntVar (id, name, is_mut) ->
      [%expr
        Sarek.Kirc_Ast.IntVar
          ( [%e quote_int ~loc id],
            [%e quote_string ~loc name],
            [%e quote_bool ~loc is_mut] )]
  | Kirc_Ast.FloatVar (id, name, is_mut) ->
      [%expr
        Sarek.Kirc_Ast.FloatVar
          ( [%e quote_int ~loc id],
            [%e quote_string ~loc name],
            [%e quote_bool ~loc is_mut] )]
  | Kirc_Ast.UnitVar (id, name, is_mut) ->
      [%expr
        Sarek.Kirc_Ast.UnitVar
          ( [%e quote_int ~loc id],
            [%e quote_string ~loc name],
            [%e quote_bool ~loc is_mut] )]
  | Kirc_Ast.CastDoubleVar (id, name) ->
      [%expr
        Sarek.Kirc_Ast.CastDoubleVar
          ([%e quote_int ~loc id], [%e quote_string ~loc name])]
  | Kirc_Ast.DoubleVar (id, name, is_mut) ->
      [%expr
        Sarek.Kirc_Ast.DoubleVar
          ( [%e quote_int ~loc id],
            [%e quote_string ~loc name],
            [%e quote_bool ~loc is_mut] )]
  | Kirc_Ast.BoolVar (id, name, is_mut) ->
      [%expr
        Sarek.Kirc_Ast.BoolVar
          ( [%e quote_int ~loc id],
            [%e quote_string ~loc name],
            [%e quote_bool ~loc is_mut] )]
  | Kirc_Ast.VecVar (elem, id, name) ->
      [%expr
        Sarek.Kirc_Ast.VecVar
          ( [%e quote_k_ext ~loc elem],
            [%e quote_int ~loc id],
            [%e quote_string ~loc name] )]
  | Kirc_Ast.Arr (name, size, elt, mem) ->
      [%expr
        Sarek.Kirc_Ast.Arr
          ( [%e quote_string ~loc name],
            [%e quote_k_ext ~loc size],
            [%e quote_elttype ~loc elt],
            [%e quote_memspace ~loc mem] )]
  | Kirc_Ast.Concat (a, b) ->
      [%expr
        Sarek.Kirc_Ast.Concat ([%e quote_k_ext ~loc a], [%e quote_k_ext ~loc b])]
  | Kirc_Ast.Constr (type_name, constr_name, args) ->
      [%expr
        Sarek.Kirc_Ast.Constr
          ( [%e quote_string ~loc type_name],
            [%e quote_string ~loc constr_name],
            [%e quote_list ~loc quote_k_ext args] )]
  | Kirc_Ast.Record (type_name, fields) ->
      [%expr
        Sarek.Kirc_Ast.Record
          ( [%e quote_string ~loc type_name],
            [%e quote_list ~loc quote_k_ext fields] )]
  | Kirc_Ast.RecGet (record, field) ->
      [%expr
        Sarek.Kirc_Ast.RecGet
          ([%e quote_k_ext ~loc record], [%e quote_string ~loc field])]
  | Kirc_Ast.RecSet (record, value) ->
      [%expr
        Sarek.Kirc_Ast.RecSet
          ([%e quote_k_ext ~loc record], [%e quote_k_ext ~loc value])]
  | Kirc_Ast.Empty -> [%expr Sarek.Kirc_Ast.Empty]
  | Kirc_Ast.Seq (a, b) ->
      [%expr
        Sarek.Kirc_Ast.Seq ([%e quote_k_ext ~loc a], [%e quote_k_ext ~loc b])]
  | Kirc_Ast.Return k -> [%expr Sarek.Kirc_Ast.Return [%e quote_k_ext ~loc k]]
  | Kirc_Ast.Set (a, b) ->
      [%expr
        Sarek.Kirc_Ast.Set ([%e quote_k_ext ~loc a], [%e quote_k_ext ~loc b])]
  | Kirc_Ast.Decl k -> [%expr Sarek.Kirc_Ast.Decl [%e quote_k_ext ~loc k]]
  | Kirc_Ast.SetV (a, b) ->
      [%expr
        Sarek.Kirc_Ast.SetV ([%e quote_k_ext ~loc a], [%e quote_k_ext ~loc b])]
  | Kirc_Ast.SetLocalVar (a, b, c) ->
      [%expr
        Sarek.Kirc_Ast.SetLocalVar
          ( [%e quote_k_ext ~loc a],
            [%e quote_k_ext ~loc b],
            [%e quote_k_ext ~loc c] )]
  | Kirc_Ast.Intrinsics (cuda, opencl) ->
      [%expr
        Sarek.Kirc_Ast.Intrinsics
          ([%e quote_string ~loc cuda], [%e quote_string ~loc opencl])]
  | Kirc_Ast.IntId (name, id) ->
      [%expr
        Sarek.Kirc_Ast.IntId
          ([%e quote_string ~loc name], [%e quote_int ~loc id])]
  | Kirc_Ast.Int i -> [%expr Sarek.Kirc_Ast.Int [%e quote_int ~loc i]]
  | Kirc_Ast.Float f -> [%expr Sarek.Kirc_Ast.Float [%e quote_float ~loc f]]
  | Kirc_Ast.Double f -> [%expr Sarek.Kirc_Ast.Double [%e quote_float ~loc f]]
  | Kirc_Ast.Custom (type_name, id, name) ->
      [%expr
        Sarek.Kirc_Ast.Custom
          ( [%e quote_string ~loc type_name],
            [%e quote_int ~loc id],
            [%e quote_string ~loc name] )]
  | Kirc_Ast.CustomVar (type_name, constr_name, var_name) ->
      [%expr
        Sarek.Kirc_Ast.CustomVar
          ( [%e quote_string ~loc type_name],
            [%e quote_string ~loc constr_name],
            [%e quote_string ~loc var_name] )]
  | Kirc_Ast.IntVecAcc (vec, idx) ->
      [%expr
        Sarek.Kirc_Ast.IntVecAcc
          ([%e quote_k_ext ~loc vec], [%e quote_k_ext ~loc idx])]
  | Kirc_Ast.Local (decl, body) ->
      [%expr
        Sarek.Kirc_Ast.Local
          ([%e quote_k_ext ~loc decl], [%e quote_k_ext ~loc body])]
  | Kirc_Ast.Acc (a, b) ->
      [%expr
        Sarek.Kirc_Ast.Acc ([%e quote_k_ext ~loc a], [%e quote_k_ext ~loc b])]
  | Kirc_Ast.Ife (cond, then_e, else_e) ->
      [%expr
        Sarek.Kirc_Ast.Ife
          ( [%e quote_k_ext ~loc cond],
            [%e quote_k_ext ~loc then_e],
            [%e quote_k_ext ~loc else_e] )]
  | Kirc_Ast.If (cond, then_e) ->
      [%expr
        Sarek.Kirc_Ast.If
          ([%e quote_k_ext ~loc cond], [%e quote_k_ext ~loc then_e])]
  | Kirc_Ast.Match (type_name, scrutinee, cases) ->
      [%expr
        Sarek.Kirc_Ast.Match
          ( [%e quote_string ~loc type_name],
            [%e quote_k_ext ~loc scrutinee],
            [%e quote_array ~loc quote_case cases] )]
  | Kirc_Ast.Or (a, b) ->
      [%expr
        Sarek.Kirc_Ast.Or ([%e quote_k_ext ~loc a], [%e quote_k_ext ~loc b])]
  | Kirc_Ast.And (a, b) ->
      [%expr
        Sarek.Kirc_Ast.And ([%e quote_k_ext ~loc a], [%e quote_k_ext ~loc b])]
  | Kirc_Ast.Not k -> [%expr Sarek.Kirc_Ast.Not [%e quote_k_ext ~loc k]]
  | Kirc_Ast.EqCustom (type_name, a, b) ->
      [%expr
        Sarek.Kirc_Ast.EqCustom
          ( [%e quote_string ~loc type_name],
            [%e quote_k_ext ~loc a],
            [%e quote_k_ext ~loc b] )]
  | Kirc_Ast.EqBool (a, b) ->
      [%expr
        Sarek.Kirc_Ast.EqBool ([%e quote_k_ext ~loc a], [%e quote_k_ext ~loc b])]
  | Kirc_Ast.LtBool (a, b) ->
      [%expr
        Sarek.Kirc_Ast.LtBool ([%e quote_k_ext ~loc a], [%e quote_k_ext ~loc b])]
  | Kirc_Ast.GtBool (a, b) ->
      [%expr
        Sarek.Kirc_Ast.GtBool ([%e quote_k_ext ~loc a], [%e quote_k_ext ~loc b])]
  | Kirc_Ast.LtEBool (a, b) ->
      [%expr
        Sarek.Kirc_Ast.LtEBool ([%e quote_k_ext ~loc a], [%e quote_k_ext ~loc b])]
  | Kirc_Ast.GtEBool (a, b) ->
      [%expr
        Sarek.Kirc_Ast.GtEBool ([%e quote_k_ext ~loc a], [%e quote_k_ext ~loc b])]
  | Kirc_Ast.DoLoop (var, lo, hi, body) ->
      [%expr
        Sarek.Kirc_Ast.DoLoop
          ( [%e quote_k_ext ~loc var],
            [%e quote_k_ext ~loc lo],
            [%e quote_k_ext ~loc hi],
            [%e quote_k_ext ~loc body] )]
  | Kirc_Ast.While (cond, body) ->
      [%expr
        Sarek.Kirc_Ast.While
          ([%e quote_k_ext ~loc cond], [%e quote_k_ext ~loc body])]
  | Kirc_Ast.App (fn, args) ->
      [%expr
        Sarek.Kirc_Ast.App
          ([%e quote_k_ext ~loc fn], [%e quote_array ~loc quote_k_ext args])]
  | Kirc_Ast.GInt _ ->
      (* This shouldn't be reached in PPX - we use GIntVar instead *)
      [%expr Sarek.Kirc_Ast.GInt (fun () -> 0l)]
  | Kirc_Ast.GFloat _ -> [%expr Sarek.Kirc_Ast.GFloat (fun () -> 0.0)]
  | Kirc_Ast.GFloat64 _ -> [%expr Sarek.Kirc_Ast.GFloat64 (fun () -> 0.0)]
  (* GIntVar, GFloatVar, GFloat64Var - generate closures that capture the variable *)
  (* These variables are refs, so we dereference them with ! *)
  | Kirc_Ast.GIntVar name ->
      let var_expr = evar ~loc name in
      [%expr Sarek.Kirc_Ast.GInt (fun () -> ![%e var_expr])]
  | Kirc_Ast.GFloatVar name ->
      let var_expr = evar ~loc name in
      [%expr Sarek.Kirc_Ast.GFloat (fun () -> ![%e var_expr])]
  | Kirc_Ast.GFloat64Var name ->
      let var_expr = evar ~loc name in
      [%expr Sarek.Kirc_Ast.GFloat64 (fun () -> ![%e var_expr])]
  | Kirc_Ast.Native _ ->
      (* This shouldn't be reached in PPX - we use NativeWithFallback instead *)
      [%expr Sarek.Kirc_Ast.Native (fun _framework -> "")]
  | Kirc_Ast.NativeWithFallback {gpu; ocaml} ->
      (* Native code with GPU expression and OCaml fallback function.
         The GPU expression is (fun dev -> "code").
         The ocaml expression is a function that will be applied to args.
         We now store it as a typed function (block/grid/native_args). *)
      [%expr
        Sarek.Kirc_Ast.NativeWithFallback {gpu = [%e gpu]; ocaml = [%e ocaml]}]
  | Kirc_Ast.Pragma (opts, body) ->
      [%expr
        Sarek.Kirc_Ast.Pragma
          ([%e quote_list ~loc quote_string opts], [%e quote_k_ext ~loc body])]
  | Kirc_Ast.Map (f, a, b) ->
      [%expr
        Sarek.Kirc_Ast.Map
          ( [%e quote_k_ext ~loc f],
            [%e quote_k_ext ~loc a],
            [%e quote_k_ext ~loc b] )]
  | Kirc_Ast.IntrinsicRef (path, name) ->
      let path_expr =
        Ast_builder.Default.elist
          ~loc
          (List.map (Ast_builder.Default.estring ~loc) path)
      in
      let name_expr = Ast_builder.Default.estring ~loc name in
      [%expr Sarek.Kirc_Ast.IntrinsicRef ([%e path_expr], [%e name_expr])]
  | Kirc_Ast.Unit -> [%expr Sarek.Kirc_Ast.Unit]

let core_type_of_typ ~loc (t : typ) : core_type option =
  match repr t with
  | TPrim TUnit -> Some [%type: unit]
  | TPrim (TBool | TInt32) -> Some [%type: int]
  | TReg Int64 -> Some [%type: int]
  | TReg (Float32 | Float64) -> Some [%type: float]
  | TVec elem -> (
      (* V2: Use Spoc_core.Vector.t instead of Spoc.Vector.vector *)
      match repr elem with
      | TPrim TInt32 -> Some [%type: (int32, _) Spoc_core.Vector.t]
      | TReg Int64 -> Some [%type: (int64, _) Spoc_core.Vector.t]
      | TReg Float32 -> Some [%type: (float, _) Spoc_core.Vector.t]
      | TReg Float64 -> Some [%type: (float, _) Spoc_core.Vector.t]
      | TPrim TBool -> Some [%type: (bool, _) Spoc_core.Vector.t]
      | TRecord _ | TVariant _ ->
          (* Don't add type constraint for custom vectors - let OCaml infer *)
          None
      | _ -> None)
  | TRecord _ | TVariant _ -> None
  | TArr _ | TFun _ | TTuple _ | TVar _ | TReg _ -> None

let kernel_ctor_name (t : typ) : string =
  match repr t with
  | TPrim (TBool | TInt32) -> "Int32"
  | TReg Int64 -> "Int64"
  | TReg Float32 -> "Float32"
  | TReg Float64 -> "Float64"
  | TVec elem -> (
      match repr elem with
      | TPrim (TBool | TInt32) -> "VInt32"
      | TReg Int64 -> "VInt64"
      | TReg Float32 -> "VFloat32"
      | TReg Float64 -> "VFloat64"
      | TRecord _ | TVariant _ -> "VCustom"
      | _ -> "Vector")
  | TRecord _ | TVariant _ -> "Custom"
  | _ -> "Vector"

let kernel_arg_expr ~loc:_ ty var =
  ignore ty ;
  (* V2-only path: pass arguments directly without SPOC constructors *)
  var

let kernel_arg_pat ~loc:_ ty var =
  ignore ty ;
  var

let build_kernel_args ~loc (params : tparam list) =
  let names = List.mapi (fun idx _ -> Printf.sprintf "spoc_arg%d" idx) params in
  let vars = List.map (fun name -> (name, evar ~loc name)) names in
  let args_pat =
    let pats =
      List.map
        (fun (name, _) -> Ast_builder.Default.ppat_var ~loc {txt = name; loc})
        vars
    in
    match pats with
    | [] -> [%pat? ()]
    | [p] -> p
    | _ -> Ast_builder.Default.ppat_tuple ~loc pats
  in
  let args_array_expr =
    let exprs =
      List.map2
        (fun p (_, v) -> kernel_arg_expr ~loc p.tparam_type v)
        params
        vars
    in
    Ast_builder.Default.pexp_array ~loc exprs
  in
  let list_to_args_pat =
    let pats =
      List.map2
        (fun p (name, _) ->
          kernel_arg_pat
            ~loc
            p.tparam_type
            (Ast_builder.Default.ppat_var ~loc {txt = name; loc}))
        params
        vars
    in
    Ast_builder.Default.ppat_array ~loc pats
  in
  let list_to_args_expr =
    (* Don't add type constraints here - let types be inferred from context.
       The args_pat already has user's type annotations which may use either
       Spoc.Vector.vector or Spoc_core.Vector.t depending on what's in scope. *)
    let exprs = List.map2 (fun _p (_, v) -> v) params vars in
    match exprs with
    | [] -> [%expr ()]
    | [e] -> e
    | _ -> Ast_builder.Default.pexp_tuple ~loc exprs
  in
  (args_pat, args_array_expr, list_to_args_pat, list_to_args_expr)

(******************************************************************************
 * Intrinsic Reference Collection
 *
 * Collect all intrinsic function references from a kernel and generate
 * a dummy expression that mentions each one. This ensures that if an
 * intrinsic function is missing from the stdlib, compilation will fail.
 ******************************************************************************)

module IntrinsicRefSet = Set.Make (struct
  type t = Sarek_env.intrinsic_ref

  let compare = compare
end)

(** Generate an OCaml expression for an intrinsic reference.

    Intrinsics are now defined in stdlib modules (Float32, Float64, Int32, etc.)
    via %sarek_intrinsic. We reference the function to ensure it exists at
    compile time. The module path enables extensibility: user libraries can
    define their own intrinsics and the PPX will reference them correctly.

    Examples:
    - IntrinsicRef (["Float32"], "sin") -> Float32.sin
    - IntrinsicRef (["Gpu"], "block_barrier") -> Gpu.block_barrier
    - CorePrimitiveRef "thread_idx_x" -> Gpu.thread_idx_x *)
let expr_of_intrinsic_ref ~loc (ref : Sarek_env.intrinsic_ref) : expression =
  match ref with
  | Sarek_env.IntrinsicRef (module_path, name) ->
      (* Build longident from module path: ["A"; "B"] + "f" -> A.B.f *)
      let lid =
        List.fold_left
          (fun acc m -> Ldot (acc, m))
          (Lident (List.hd module_path))
          (List.tl module_path @ [name])
      in
      Ast_builder.Default.pexp_ident ~loc {txt = lid; loc}
  | Sarek_env.CorePrimitiveRef name ->
      (* Core primitives are accessed via Gpu module *)
      let lid = Ldot (Lident "Gpu", name) in
      Ast_builder.Default.pexp_ident ~loc {txt = lid; loc}

(** Collect all intrinsic function refs from a typed expression *)
let rec collect_intrinsic_refs (te : texpr) : IntrinsicRefSet.t =
  match te.te with
  | TEUnit | TEBool _ | TEInt _ | TEInt32 _ | TEInt64 _ | TEFloat _ | TEDouble _
  | TEVar _ | TENative _ | TEGlobalRef _ ->
      IntrinsicRefSet.empty
  | TEVecGet (v, i) | TEVecSet (v, i, _) | TEArrGet (v, i) ->
      IntrinsicRefSet.union
        (collect_intrinsic_refs v)
        (collect_intrinsic_refs i)
  | TEArrSet (a, i, x) ->
      IntrinsicRefSet.union
        (collect_intrinsic_refs a)
        (IntrinsicRefSet.union
           (collect_intrinsic_refs i)
           (collect_intrinsic_refs x))
  | TEFieldGet (r, _, _) -> collect_intrinsic_refs r
  | TEFieldSet (r, _, _, x) ->
      IntrinsicRefSet.union
        (collect_intrinsic_refs r)
        (collect_intrinsic_refs x)
  | TEBinop (_, a, b) ->
      IntrinsicRefSet.union
        (collect_intrinsic_refs a)
        (collect_intrinsic_refs b)
  | TEUnop (_, a) -> collect_intrinsic_refs a
  | TEApp (f, args) ->
      List.fold_left
        (fun acc arg -> IntrinsicRefSet.union acc (collect_intrinsic_refs arg))
        (collect_intrinsic_refs f)
        args
  | TEAssign (_, _, value) -> collect_intrinsic_refs value
  | TELet (_, _, value, body) | TELetMut (_, _, value, body) ->
      IntrinsicRefSet.union
        (collect_intrinsic_refs value)
        (collect_intrinsic_refs body)
  | TEIf (cond, then_e, else_e) -> (
      let base =
        IntrinsicRefSet.union
          (collect_intrinsic_refs cond)
          (collect_intrinsic_refs then_e)
      in
      match else_e with
      | Some e -> IntrinsicRefSet.union base (collect_intrinsic_refs e)
      | None -> base)
  | TEFor (_, _, lo, hi, _, body) ->
      IntrinsicRefSet.union
        (collect_intrinsic_refs lo)
        (IntrinsicRefSet.union
           (collect_intrinsic_refs hi)
           (collect_intrinsic_refs body))
  | TEWhile (cond, body) ->
      IntrinsicRefSet.union
        (collect_intrinsic_refs cond)
        (collect_intrinsic_refs body)
  | TESeq exprs ->
      List.fold_left
        (fun acc e -> IntrinsicRefSet.union acc (collect_intrinsic_refs e))
        IntrinsicRefSet.empty
        exprs
  | TEMatch (scrutinee, cases) ->
      List.fold_left
        (fun acc (_, body) ->
          IntrinsicRefSet.union acc (collect_intrinsic_refs body))
        (collect_intrinsic_refs scrutinee)
        cases
  | TERecord (_, fields) ->
      List.fold_left
        (fun acc (_, e) -> IntrinsicRefSet.union acc (collect_intrinsic_refs e))
        IntrinsicRefSet.empty
        fields
  | TEConstr (_, _, arg) -> (
      match arg with
      | Some e -> collect_intrinsic_refs e
      | None -> IntrinsicRefSet.empty)
  | TETuple exprs ->
      List.fold_left
        (fun acc e -> IntrinsicRefSet.union acc (collect_intrinsic_refs e))
        IntrinsicRefSet.empty
        exprs
  | TEReturn e | TEPragma (_, e) -> collect_intrinsic_refs e
  | TECreateArray (size, _, _) -> collect_intrinsic_refs size
  | TELetShared (_, _, _, size_opt, body) ->
      let size_refs =
        match size_opt with
        | Some size -> collect_intrinsic_refs size
        | None -> IntrinsicRefSet.empty
      in
      IntrinsicRefSet.union size_refs (collect_intrinsic_refs body)
  | TESuperstep (_, _, step_body, cont) ->
      IntrinsicRefSet.union
        (collect_intrinsic_refs step_body)
        (collect_intrinsic_refs cont)
  | TELetRec (_, _, _, fn_body, cont) ->
      IntrinsicRefSet.union
        (collect_intrinsic_refs fn_body)
        (collect_intrinsic_refs cont)
  | TEOpen (_, body) -> collect_intrinsic_refs body
  | TEIntrinsicConst ref -> IntrinsicRefSet.singleton ref
  | TEIntrinsicFun (ref, _convergence, args) ->
      let base = IntrinsicRefSet.singleton ref in
      List.fold_left
        (fun acc arg -> IntrinsicRefSet.union acc (collect_intrinsic_refs arg))
        base
        args

(** Collect intrinsic refs from module items *)
let collect_from_module_items (items : tmodule_item list) : IntrinsicRefSet.t =
  List.fold_left
    (fun acc item ->
      match item with
      | TMConst (_, _, _, e) ->
          IntrinsicRefSet.union acc (collect_intrinsic_refs e)
      | TMFun (_, _, _, e) ->
          IntrinsicRefSet.union acc (collect_intrinsic_refs e))
    IntrinsicRefSet.empty
    items

(** Generate a dummy expression that references all intrinsic functions. This
    ensures compile-time checking that all intrinsics exist in their stdlib
    modules. *)
let generate_intrinsic_check ~loc (kernel : tkernel) : expression =
  let refs_from_body = collect_intrinsic_refs kernel.tkern_body in
  let refs_from_items = collect_from_module_items kernel.tkern_module_items in
  let all_refs = IntrinsicRefSet.union refs_from_body refs_from_items in
  let ref_list = IntrinsicRefSet.elements all_refs in
  let fn_exprs = List.map (expr_of_intrinsic_ref ~loc) ref_list in
  match fn_exprs with
  | [] -> [%expr ()]
  | exprs ->
      (* Generate: let _ = (fn1, fn2, fn3, ...) in () *)
      let tuple_expr =
        match exprs with
        | [e] -> e
        | es -> Ast_builder.Default.pexp_tuple ~loc es
      in
      [%expr
        let _ = [%e tuple_expr] in
        ()]

(** Quote a kernel to create a sarek_kernel expression *)
let quote_kernel ~loc ?(native_kernel : tkernel option)
    ~(ir_opt : Sarek_ir_ppx.kernel) ~(constructors : string list)
    (kernel : tkernel) : expression =
  (* Use native_kernel for CPU code generation if provided, otherwise use kernel.
     This allows passing the original kernel (before tailrec transformation)
     for native OCaml, since OCaml handles recursion natively. *)
  let kernel_for_native = Option.value native_kernel ~default:kernel in
  let args_pat, args_array_expr, list_to_args_pat, list_to_args_expr =
    build_kernel_args ~loc kernel.tkern_params
  in
  (* Suppress unused variable warnings for legacy args handling *)
  let _ = (args_pat, args_array_expr, list_to_args_pat, list_to_args_expr) in
  [%expr
    let () =
      List.iter
        Sarek.Kirc_types.register_constructor_string
        [%e
          Ast_builder.Default.elist
            ~loc
            (List.map (Ast_builder.Default.estring ~loc) constructors)]
    in
    let open Sarek.Kirc_types in
    let _intrinsic_check = [%e generate_intrinsic_check ~loc kernel] in
    (* Native function for host execution (uses Spoc_core.Vector) *)
    let native_fn =
      [%e
        Sarek_native_gen_kernel.gen_cpu_kern_native_wrapper
          ~loc
          kernel_for_native]
    in
    let body_ir_ir =
      Some
        [%e
          Sarek_quote_ir.quote_kernel
            ~loc
            ~native_fn_expr:[%expr native_fn]
            ir_opt]
    in
    let kirc_kernel =
      {
        Sarek.Kirc_types.ml_kern = (fun () -> ());
        Sarek.Kirc_types.body_ir = body_ir_ir;
        Sarek.Kirc_types.ret_val = ();
        Sarek.Kirc_types.extensions = [||];
      }
    in
    ((), kirc_kernel)]

(******************************************************************************
 * Sarek_ast Quoting
 *
 * Quote Sarek_ast types to OCaml expressions for registration of
 * [@sarek.module] items in the PPX registry.
 ******************************************************************************)

(** Quote a Sarek_ast.loc *)
let quote_sarek_loc ~loc (l : Sarek_ast.loc) : expression =
  [%expr
    {
      Sarek_ast.loc_file = [%e quote_string ~loc l.loc_file];
      loc_line = [%e quote_int ~loc l.loc_line];
      loc_col = [%e quote_int ~loc l.loc_col];
      loc_end_line = [%e quote_int ~loc l.loc_end_line];
      loc_end_col = [%e quote_int ~loc l.loc_end_col];
    }]

(** Quote a Sarek_ast.memspace *)
let quote_sarek_memspace ~loc (m : Sarek_ast.memspace) : expression =
  match m with
  | Sarek_ast.Local -> [%expr Sarek_ast.Local]
  | Sarek_ast.Shared -> [%expr Sarek_ast.Shared]
  | Sarek_ast.Global -> [%expr Sarek_ast.Global]

(** Quote a Sarek_ast.type_expr *)
let rec quote_sarek_type_expr ~loc (te : Sarek_ast.type_expr) : expression =
  match te with
  | Sarek_ast.TEVar s ->
      [%expr Sarek_ast.TEVar [%e quote_string ~loc s]]
  | Sarek_ast.TEConstr (name, args) ->
      [%expr
        Sarek_ast.TEConstr
          ( [%e quote_string ~loc name],
            [%e quote_list ~loc quote_sarek_type_expr args] )]
  | Sarek_ast.TEArrow (a, b) ->
      [%expr
        Sarek_ast.TEArrow
          ([%e quote_sarek_type_expr ~loc a], [%e quote_sarek_type_expr ~loc b])]
  | Sarek_ast.TETuple ts ->
      [%expr
        Sarek_ast.TETuple
          [%e quote_list ~loc quote_sarek_type_expr ts]]

(** Quote a Sarek_ast.binop *)
let quote_sarek_binop ~loc (op : Sarek_ast.binop) : expression =
  match op with
  | Sarek_ast.Add -> [%expr Sarek_ast.Add]
  | Sarek_ast.Sub -> [%expr Sarek_ast.Sub]
  | Sarek_ast.Mul -> [%expr Sarek_ast.Mul]
  | Sarek_ast.Div -> [%expr Sarek_ast.Div]
  | Sarek_ast.Mod -> [%expr Sarek_ast.Mod]
  | Sarek_ast.And -> [%expr Sarek_ast.And]
  | Sarek_ast.Or -> [%expr Sarek_ast.Or]
  | Sarek_ast.Eq -> [%expr Sarek_ast.Eq]
  | Sarek_ast.Ne -> [%expr Sarek_ast.Ne]
  | Sarek_ast.Lt -> [%expr Sarek_ast.Lt]
  | Sarek_ast.Le -> [%expr Sarek_ast.Le]
  | Sarek_ast.Gt -> [%expr Sarek_ast.Gt]
  | Sarek_ast.Ge -> [%expr Sarek_ast.Ge]
  | Sarek_ast.Land -> [%expr Sarek_ast.Land]
  | Sarek_ast.Lor -> [%expr Sarek_ast.Lor]
  | Sarek_ast.Lxor -> [%expr Sarek_ast.Lxor]
  | Sarek_ast.Lsl -> [%expr Sarek_ast.Lsl]
  | Sarek_ast.Lsr -> [%expr Sarek_ast.Lsr]
  | Sarek_ast.Asr -> [%expr Sarek_ast.Asr]

(** Quote a Sarek_ast.unop *)
let quote_sarek_unop ~loc (op : Sarek_ast.unop) : expression =
  match op with
  | Sarek_ast.Neg -> [%expr Sarek_ast.Neg]
  | Sarek_ast.Not -> [%expr Sarek_ast.Not]
  | Sarek_ast.Lnot -> [%expr Sarek_ast.Lnot]

(** Quote a Sarek_ast.for_dir *)
let quote_sarek_for_dir ~loc (d : Sarek_ast.for_dir) : expression =
  match d with
  | Sarek_ast.Upto -> [%expr Sarek_ast.Upto]
  | Sarek_ast.Downto -> [%expr Sarek_ast.Downto]

(** Quote a Sarek_ast.pattern *)
let rec quote_sarek_pattern ~loc (p : Sarek_ast.pattern) : expression =
  let desc =
    match p.pat with
    | Sarek_ast.PAny -> [%expr Sarek_ast.PAny]
    | Sarek_ast.PVar s ->
        [%expr Sarek_ast.PVar [%e quote_string ~loc s]]
    | Sarek_ast.PConstr (name, arg) ->
        [%expr
          Sarek_ast.PConstr
            ( [%e quote_string ~loc name],
              [%e quote_option ~loc quote_sarek_pattern arg] )]
    | Sarek_ast.PTuple ps ->
        [%expr
          Sarek_ast.PTuple
            [%e quote_list ~loc quote_sarek_pattern ps]]
  in
  [%expr
    {
      Sarek_ast.pat = [%e desc];
      pat_loc = [%e quote_sarek_loc ~loc p.pat_loc];
    }]

(** Quote a Sarek_ast.param *)
let quote_sarek_param ~loc (p : Sarek_ast.param) : expression =
  [%expr
    {
      Sarek_ast.param_name = [%e quote_string ~loc p.param_name];
      param_type = [%e quote_sarek_type_expr ~loc p.param_type];
      param_loc = [%e quote_sarek_loc ~loc p.param_loc];
    }]

(** Quote a Sarek_ast.expr - main recursive function *)
let rec quote_sarek_expr ~loc (e : Sarek_ast.expr) : expression =
  let desc =
    match e.e with
    | Sarek_ast.EUnit -> [%expr Sarek_ast.EUnit]
    | Sarek_ast.EBool b ->
        [%expr Sarek_ast.EBool [%e quote_bool ~loc b]]
    | Sarek_ast.EInt i ->
        [%expr Sarek_ast.EInt [%e quote_int ~loc i]]
    | Sarek_ast.EInt32 i ->
        let i_int = Int32.to_int i in
        [%expr Sarek_ast.EInt32 [%e quote_int32 ~loc i]]
        |> fun _ ->
        [%expr
          Sarek_ast.EInt32
            (Int32.of_int [%e quote_int ~loc i_int])]
    | Sarek_ast.EInt64 i ->
        let i_int = Int64.to_int i in
        [%expr
          Sarek_ast.EInt64
            (Int64.of_int [%e quote_int ~loc i_int])]
    | Sarek_ast.EFloat f ->
        [%expr Sarek_ast.EFloat [%e quote_float ~loc f]]
    | Sarek_ast.EDouble f ->
        [%expr Sarek_ast.EDouble [%e quote_float ~loc f]]
    | Sarek_ast.EVar s ->
        [%expr Sarek_ast.EVar [%e quote_string ~loc s]]
    | Sarek_ast.EVecGet (v, i) ->
        [%expr
          Sarek_ast.EVecGet
            ([%e quote_sarek_expr ~loc v], [%e quote_sarek_expr ~loc i])]
    | Sarek_ast.EVecSet (v, i, x) ->
        [%expr
          Sarek_ast.EVecSet
            ( [%e quote_sarek_expr ~loc v],
              [%e quote_sarek_expr ~loc i],
              [%e quote_sarek_expr ~loc x] )]
    | Sarek_ast.EArrGet (a, i) ->
        [%expr
          Sarek_ast.EArrGet
            ([%e quote_sarek_expr ~loc a], [%e quote_sarek_expr ~loc i])]
    | Sarek_ast.EArrSet (a, i, x) ->
        [%expr
          Sarek_ast.EArrSet
            ( [%e quote_sarek_expr ~loc a],
              [%e quote_sarek_expr ~loc i],
              [%e quote_sarek_expr ~loc x] )]
    | Sarek_ast.EFieldGet (e, f) ->
        [%expr
          Sarek_ast.EFieldGet
            ([%e quote_sarek_expr ~loc e], [%e quote_string ~loc f])]
    | Sarek_ast.EFieldSet (e, f, v) ->
        [%expr
          Sarek_ast.EFieldSet
            ( [%e quote_sarek_expr ~loc e],
              [%e quote_string ~loc f],
              [%e quote_sarek_expr ~loc v] )]
    | Sarek_ast.EBinop (op, a, b) ->
        [%expr
          Sarek_ast.EBinop
            ( [%e quote_sarek_binop ~loc op],
              [%e quote_sarek_expr ~loc a],
              [%e quote_sarek_expr ~loc b] )]
    | Sarek_ast.EUnop (op, e) ->
        [%expr
          Sarek_ast.EUnop
            ([%e quote_sarek_unop ~loc op], [%e quote_sarek_expr ~loc e])]
    | Sarek_ast.EApp (f, args) ->
        [%expr
          Sarek_ast.EApp
            ( [%e quote_sarek_expr ~loc f],
              [%e quote_list ~loc quote_sarek_expr args] )]
    | Sarek_ast.EAssign (name, e) ->
        [%expr
          Sarek_ast.EAssign
            ([%e quote_string ~loc name], [%e quote_sarek_expr ~loc e])]
    | Sarek_ast.ELet (name, ty, value, body) ->
        [%expr
          Sarek_ast.ELet
            ( [%e quote_string ~loc name],
              [%e quote_option ~loc quote_sarek_type_expr ty],
              [%e quote_sarek_expr ~loc value],
              [%e quote_sarek_expr ~loc body] )]
    | Sarek_ast.ELetRec (name, params, ret_ty, fn_body, cont) ->
        [%expr
          Sarek_ast.ELetRec
            ( [%e quote_string ~loc name],
              [%e quote_list ~loc quote_sarek_param params],
              [%e quote_option ~loc quote_sarek_type_expr ret_ty],
              [%e quote_sarek_expr ~loc fn_body],
              [%e quote_sarek_expr ~loc cont] )]
    | Sarek_ast.ELetMut (name, ty, value, body) ->
        [%expr
          Sarek_ast.ELetMut
            ( [%e quote_string ~loc name],
              [%e quote_option ~loc quote_sarek_type_expr ty],
              [%e quote_sarek_expr ~loc value],
              [%e quote_sarek_expr ~loc body] )]
    | Sarek_ast.EIf (cond, then_e, else_e) ->
        [%expr
          Sarek_ast.EIf
            ( [%e quote_sarek_expr ~loc cond],
              [%e quote_sarek_expr ~loc then_e],
              [%e quote_option ~loc quote_sarek_expr else_e] )]
    | Sarek_ast.EFor (var, lo, hi, dir, body) ->
        [%expr
          Sarek_ast.EFor
            ( [%e quote_string ~loc var],
              [%e quote_sarek_expr ~loc lo],
              [%e quote_sarek_expr ~loc hi],
              [%e quote_sarek_for_dir ~loc dir],
              [%e quote_sarek_expr ~loc body] )]
    | Sarek_ast.EWhile (cond, body) ->
        [%expr
          Sarek_ast.EWhile
            ([%e quote_sarek_expr ~loc cond], [%e quote_sarek_expr ~loc body])]
    | Sarek_ast.ESeq (a, b) ->
        [%expr
          Sarek_ast.ESeq
            ([%e quote_sarek_expr ~loc a], [%e quote_sarek_expr ~loc b])]
    | Sarek_ast.EMatch (scrutinee, cases) ->
        let quote_case ~loc (p, e) =
          [%expr [%e quote_sarek_pattern ~loc p], [%e quote_sarek_expr ~loc e]]
        in
        [%expr
          Sarek_ast.EMatch
            ( [%e quote_sarek_expr ~loc scrutinee],
              [%e quote_list ~loc quote_case cases] )]
    | Sarek_ast.ERecord (ty_name, fields) ->
        let quote_field ~loc (name, e) =
          [%expr [%e quote_string ~loc name], [%e quote_sarek_expr ~loc e]]
        in
        [%expr
          Sarek_ast.ERecord
            ( [%e quote_option ~loc quote_string ty_name],
              [%e quote_list ~loc quote_field fields] )]
    | Sarek_ast.EConstr (name, arg) ->
        [%expr
          Sarek_ast.EConstr
            ( [%e quote_string ~loc name],
              [%e quote_option ~loc quote_sarek_expr arg] )]
    | Sarek_ast.ETuple es ->
        [%expr
          Sarek_ast.ETuple
            [%e quote_list ~loc quote_sarek_expr es]]
    | Sarek_ast.EReturn e ->
        [%expr Sarek_ast.EReturn [%e quote_sarek_expr ~loc e]]
    | Sarek_ast.ECreateArray (size, ty, memspace) ->
        [%expr
          Sarek_ast.ECreateArray
            ( [%e quote_sarek_expr ~loc size],
              [%e quote_sarek_type_expr ~loc ty],
              [%e quote_sarek_memspace ~loc memspace] )]
    | Sarek_ast.EGlobalRef name ->
        [%expr Sarek_ast.EGlobalRef [%e quote_string ~loc name]]
    | Sarek_ast.ENative _ ->
        (* ENative contains Ppxlib expressions which can't be quoted at runtime *)
        failwith "Cannot quote ENative expressions"
    | Sarek_ast.EPragma (hints, body) ->
        [%expr
          Sarek_ast.EPragma
            ( [%e quote_list ~loc quote_string hints],
              [%e quote_sarek_expr ~loc body] )]
    | Sarek_ast.ETyped (e, ty) ->
        [%expr
          Sarek_ast.ETyped
            ([%e quote_sarek_expr ~loc e], [%e quote_sarek_type_expr ~loc ty])]
    | Sarek_ast.EOpen (path, body) ->
        [%expr
          Sarek_ast.EOpen
            ( [%e quote_list ~loc quote_string path],
              [%e quote_sarek_expr ~loc body] )]
    | Sarek_ast.ELetShared (name, ty, size, body) ->
        [%expr
          Sarek_ast.ELetShared
            ( [%e quote_string ~loc name],
              [%e quote_sarek_type_expr ~loc ty],
              [%e quote_option ~loc quote_sarek_expr size],
              [%e quote_sarek_expr ~loc body] )]
    | Sarek_ast.ESuperstep (name, divergent, step_body, cont) ->
        [%expr
          Sarek_ast.ESuperstep
            ( [%e quote_string ~loc name],
              [%e quote_bool ~loc divergent],
              [%e quote_sarek_expr ~loc step_body],
              [%e quote_sarek_expr ~loc cont] )]
  in
  [%expr
    {
      Sarek_ast.e = [%e desc];
      expr_loc = [%e quote_sarek_loc ~loc e.expr_loc];
    }]

(** Quote a Sarek_ast.module_item *)
let quote_sarek_module_item ~loc (item : Sarek_ast.module_item) : expression =
  match item with
  | Sarek_ast.MConst (name, ty, value) ->
      [%expr
        Sarek_ast.MConst
          ( [%e quote_string ~loc name],
            [%e quote_sarek_type_expr ~loc ty],
            [%e quote_sarek_expr ~loc value] )]
  | Sarek_ast.MFun (name, is_rec, params, body) ->
      [%expr
        Sarek_ast.MFun
          ( [%e quote_string ~loc name],
            [%e quote_bool ~loc is_rec],
            [%e quote_list ~loc quote_sarek_param params],
            [%e quote_sarek_expr ~loc body] )]

(** Quote a Sarek_ast.type_decl *)
let quote_sarek_type_decl ~loc (td : Sarek_ast.type_decl) : expression =
  match td with
  | Sarek_ast.Type_record {tdecl_name; tdecl_module; tdecl_fields; tdecl_loc} ->
      let quote_field ~loc (name, is_mut, ty) =
        [%expr
          [%e quote_string ~loc name],
          [%e quote_bool ~loc is_mut],
          [%e quote_sarek_type_expr ~loc ty]]
      in
      [%expr
        Sarek_ast.Type_record
          {
            tdecl_name = [%e quote_string ~loc tdecl_name];
            tdecl_module = [%e quote_option ~loc quote_string tdecl_module];
            tdecl_fields = [%e quote_list ~loc quote_field tdecl_fields];
            tdecl_loc = [%e quote_sarek_loc ~loc tdecl_loc];
          }]
  | Sarek_ast.Type_variant
      {tdecl_name; tdecl_module; tdecl_constructors; tdecl_loc} ->
      let quote_constr ~loc (name, ty_opt) =
        [%expr
          [%e quote_string ~loc name],
          [%e quote_option ~loc quote_sarek_type_expr ty_opt]]
      in
      [%expr
        Sarek_ast.Type_variant
          {
            tdecl_name = [%e quote_string ~loc tdecl_name];
            tdecl_module = [%e quote_option ~loc quote_string tdecl_module];
            tdecl_constructors =
              [%e quote_list ~loc quote_constr tdecl_constructors];
            tdecl_loc = [%e quote_sarek_loc ~loc tdecl_loc];
          }]
