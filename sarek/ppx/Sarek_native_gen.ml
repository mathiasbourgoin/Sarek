(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Sarek PPX - GPU kernel DSL for OCaml
 *
 * Native Code Generation - Main Module
 * ====================================
 *
 * Generates native OCaml code from Sarek's typed AST.
 * The generated code runs on CPU via Sarek_cpu_runtime.
 *
 * Organization:
 * - Expression generation (literals, operators, memory access, control flow)
 * - Module and type declaration generation  
 * - Kernel generation and wrapper functions
 *
 * See also:
 * - Sarek_native_helpers for utility functions and default values
 * - Sarek_native_intrinsics for type/intrinsic mapping
 ******************************************************************************)

open Ppxlib
open Sarek_typed_ast
open Sarek_types

(* Import helpers and intrinsics *)
open Sarek_native_helpers
open Sarek_native_intrinsics

(** {1 Expression Generation} *)

(** Mutable variable tracking. We track which variable IDs are mutable so that
    TEVar can dereference them. *)
module IntSet = Set.Make (Int)

(** Set of inline type names for first-class module approach *)
module StringSet = Set.Make (String)

(** Expression generation context *)
type gen_context = {
  mut_vars : IntSet.t;
      (** Variable IDs that are mutable (need dereferencing) *)
  inline_types : StringSet.t;
      (** Type names that use first-class module accessors *)
  current_module : string option;
      (** Current module name for same-module type detection *)
  gen_mode : gen_mode;
      (** Generation mode - affects how thread indices are accessed *)
  use_native_arg : bool;
      (** When true, vector params are accessor objects (v#get i, v#set i x)
          instead of Vector.t (Vector.get v i, Vector.kernel_set v i x) *)
}

(** Empty generation context *)
let empty_ctx =
  {
    mut_vars = IntSet.empty;
    inline_types = StringSet.empty;
    current_module = None;
    gen_mode = FullMode;
    use_native_arg = false;
  }

(** Check if a qualified type name is from the current module. For
    "Test_registered_variant.color", returns true if current_module is
    "Test_registered_variant". *)
let is_same_module ctx type_name =
  match ctx.current_module with
  | None -> false
  | Some cur_mod -> (
      match String.rindex_opt type_name '.' with
      | Some idx ->
          let type_mod = String.sub type_name 0 idx in
          String.equal type_mod cur_mod
      | None -> false)

(** The name of the first-class module variable *)
let types_module_var = "__types"

(** {1 First-Class Module Name Helpers} *)

(** Generate accessor function name for a field getter *)
let field_getter_name type_name field_name : string =
  Printf.sprintf "get_%s_%s" type_name field_name

(** Generate constructor function name for a record *)
let record_maker_name type_name : string = Printf.sprintf "make_%s" type_name

(** Generate constructor function name for a variant constructor *)
let variant_ctor_name type_name ctor_name : string =
  Printf.sprintf "make_%s_%s" type_name ctor_name

let gen_literal ~loc (te : texpr) : expression =
  match te.te with
  | TEUnit -> [%expr ()]
  | TEBool b -> if b then [%expr true] else [%expr false]
  | TEInt n -> (
      (* Check the type annotation to generate the correct literal type.
         In GPU kernels, integer literals compared with int32 should be int32. *)
      match repr te.ty with
      | TReg Int | TPrim TInt32 ->
          [%expr Int32.of_int [%e Ast_builder.Default.eint ~loc n]]
      | TReg Int64 -> [%expr Int64.of_int [%e Ast_builder.Default.eint ~loc n]]
      | _ ->
          (* Default to plain int *)
          Ast_builder.Default.eint ~loc n)
  | TEInt32 n ->
      (* Generate int32 literal using Int32.of_int *)
      [%expr Int32.of_int [%e Ast_builder.Default.eint ~loc (Int32.to_int n)]]
  | TEInt64 n ->
      (* Generate int64 literal using Int64.of_int *)
      [%expr Int64.of_int [%e Ast_builder.Default.eint ~loc (Int64.to_int n)]]
  | TEFloat f | TEDouble f ->
      Ast_builder.Default.efloat ~loc (string_of_float f)
  | _ -> failwith "gen_literal: not a literal expression"

(** Generate variable reference (local, module-level, qualified) *)
let gen_variable ~loc ~ctx (name : string) (id : int) : expression =
  let var_e =
    if String.contains name '.' then
      (* Qualified name - build a proper Ldot path.
         For stdlib modules like "Float32.of_float", we need to map
         to the runtime path. *)
      let parts = String.split_on_char '.' name in
      (* Split into module path and function name *)
      let module_path, func_name =
        match List.rev parts with
        | fn :: rest -> (List.rev rest, fn)
        | [] ->
            failwith
              (Printf.sprintf
                 "Internal error: String.split_on_char returned empty list for \
                  '%s'"
                 name)
      in
      (* Map stdlib module paths to runtime locations *)
      let mapped_path = map_stdlib_path module_path in
      evar_qualified ~loc mapped_path func_name
    else evar ~loc name
  in
  if IntSet.mem id ctx.mut_vars then
    (* Mutable variable - dereference the ref *)
    [%expr ![%e var_e]]
  else var_e

let custom_descriptor_expr ~loc type_name =
  let parts = String.split_on_char '.' type_name in
  match List.rev parts with
  | name :: modules -> evar_qualified ~loc (List.rev modules) (name ^ "_custom")
  | [] -> failwith "custom_descriptor_expr: empty type name"

let vector_type_id_expr ~loc elem_ty =
  match repr elem_ty with
  | TReg Float32 -> [%expr Spoc_core.Vector.float32_vector_type_id]
  | TReg Float64 -> [%expr Spoc_core.Vector.float64_vector_type_id]
  | TReg Int | TPrim TInt32 -> [%expr Spoc_core.Vector.int32_vector_type_id]
  | TReg Int64 -> [%expr Spoc_core.Vector.int64_vector_type_id]
  | TReg Char -> [%expr Spoc_core.Vector.char_vector_type_id]
  | TRecord (type_name, _) | TVariant (type_name, _) ->
      [%expr
        [%e custom_descriptor_expr ~loc type_name].Spoc_core.Vector.vector_type_id]
  | _ -> [%expr Sarek_ir_types.Type_id.create ()]

let custom_type_id_expr ~loc elem_ty =
  match repr elem_ty with
  | TRecord (type_name, _) | TVariant (type_name, _) ->
      [%expr [%e custom_descriptor_expr ~loc type_name].Spoc_core.Vector.type_id]
  | _ -> [%expr Sarek_ir_types.Type_id.create ()]

(** Generate memory access operations (vectors, arrays, record fields) *)
let gen_memory_access ~loc ~ctx ~gen_expr (te : texpr) : expression =
  match te.te with
  (* Vector/array access - V2 only *)
  | TEVecGet (vec, idx) ->
      let vec_e = gen_expr ~loc vec in
      let idx_e = gen_expr ~loc idx in
      if ctx.use_native_arg then
        (* Native arg mode: vectors are accessor objects with #get method *)
        [%expr [%e vec_e]#get (Int32.to_int [%e idx_e])]
      else [%expr Spoc_core.Vector.get [%e vec_e] (Int32.to_int [%e idx_e])]
  | TEVecSet (vec, idx, value) ->
      let vec_e = gen_expr ~loc vec in
      let idx_e = gen_expr ~loc idx in
      let val_e = gen_expr ~loc value in
      if ctx.use_native_arg then
        (* Native arg mode: vectors are accessor objects with #set method *)
        [%expr [%e vec_e]#set (Int32.to_int [%e idx_e]) [%e val_e]]
      else
        (* Use kernel_set: no bounds check, no location update race.
           Safe for parallel execution - kernel code ensures valid indices. *)
        [%expr
          Spoc_core.Vector.kernel_set
            [%e vec_e]
            (Int32.to_int [%e idx_e])
            [%e val_e]]
  (* Array access - for shared memory (regular OCaml arrays) *)
  | TEArrGet (arr, idx) ->
      let arr_e = gen_expr ~loc arr in
      let idx_e = gen_expr ~loc idx in
      [%expr [%e arr_e].(Int32.to_int [%e idx_e])]
  | TEArrSet (arr, idx, value) ->
      let arr_e = gen_expr ~loc arr in
      let idx_e = gen_expr ~loc idx in
      let val_e = gen_expr ~loc value in
      [%expr [%e arr_e].(Int32.to_int [%e idx_e]) <- [%e val_e]]
  (* Record field access - qualify field with module path from record type.
     For Geometry_lib.point, we need p.Geometry_lib.x, not p.x.
     For inline types using first-class modules, call __types.get_type_field.
     For same-module types, use unqualified field names. *)
  | TEFieldGet (record, field_name, _field_idx) -> (
      let rec_e = gen_expr ~loc record in
      match repr record.ty with
      | TRecord (type_name, _) -> (
          (* type_name may be "Module.type" or just "type" *)
          match String.rindex_opt type_name '.' with
          | Some _ when is_same_module ctx type_name ->
              (* Same-module type - use unqualified field access *)
              let field_lid = {txt = Lident field_name; loc} in
              Ast_builder.Default.pexp_field ~loc rec_e field_lid
          | Some idx ->
              (* External qualified type - use qualified field access *)
              let module_path = String.sub type_name 0 idx in
              let parts = String.split_on_char '.' module_path in
              let rec build_lid = function
                | [] -> Lident field_name
                | [m] -> Ldot (Lident m, field_name)
                | m :: rest -> Ldot (build_lid rest, m)
              in
              let field_lid = {txt = build_lid (List.rev parts); loc} in
              Ast_builder.Default.pexp_field ~loc rec_e field_lid
          | None ->
              (* Inline type - check if using first-class module approach *)
              if StringSet.mem type_name ctx.inline_types then
                (* Use accessor: __types#get_typename_fieldname record *)
                let fn_name = field_getter_name type_name field_name in
                let method_call =
                  Ast_builder.Default.pexp_send
                    ~loc
                    (evar ~loc types_module_var)
                    {txt = fn_name; loc}
                in
                Ast_builder.Default.pexp_apply
                  ~loc
                  method_call
                  [(Nolabel, rec_e)]
              else
                (* Direct field access *)
                let field_lid = {txt = Lident field_name; loc} in
                Ast_builder.Default.pexp_field ~loc rec_e field_lid)
      | _ ->
          (* Not a record type - shouldn't happen, but use unqualified name *)
          let field_lid = {txt = Lident field_name; loc} in
          Ast_builder.Default.pexp_field ~loc rec_e field_lid)
  | TEFieldSet (record, field_name, _field_idx, value) ->
      let rec_e = gen_expr ~loc record in
      let val_e = gen_expr ~loc value in
      (* Note: mutable record fields with first-class modules would need setters.
         For now we don't support mutable fields in inline types with FCM. *)
      let field_lid =
        match repr record.ty with
        | TRecord (type_name, _) -> (
            match String.rindex_opt type_name '.' with
            | Some _ when is_same_module ctx type_name ->
                (* Same-module type - use unqualified field access *)
                {txt = Lident field_name; loc}
            | Some idx ->
                (* External qualified type - use qualified field access *)
                let module_path = String.sub type_name 0 idx in
                let parts = String.split_on_char '.' module_path in
                let rec build_lid = function
                  | [] -> Lident field_name
                  | [m] -> Ldot (Lident m, field_name)
                  | m :: rest -> Ldot (build_lid rest, m)
                in
                {txt = build_lid (List.rev parts); loc}
            | None ->
                (* Inline type - use unqualified name *)
                {txt = Lident field_name; loc})
        | _ -> {txt = Lident field_name; loc}
      in
      Ast_builder.Default.pexp_setfield ~loc rec_e field_lid val_e
  | _ -> failwith "gen_memory_access: not a memory access expression"

(** Generate let bindings (let, let mut, assignment) *)
let gen_let_binding ~loc ~ctx ~gen_expr ~gen_expr_impl (te : texpr) : expression
    =
  match te.te with
  | TEAssign (name, _id, value) ->
      let val_e = gen_expr ~loc value in
      (* Mutable variables are stored as refs with the original name *)
      let var_e = evar ~loc name in
      [%expr [%e var_e] := [%e val_e]]
  | TELet (name, _id, value, body) ->
      let val_e = gen_expr ~loc value in
      let body_e = gen_expr ~loc body in
      let pat = Ast_builder.Default.ppat_var ~loc {txt = name; loc} in
      [%expr
        let [%p pat] = [%e val_e] in
        [%e body_e]]
  | TELetMut (name, id, value, body) ->
      let val_e = gen_expr ~loc value in
      (* Add this variable to the mutable set for the body *)
      let ctx' = {ctx with mut_vars = IntSet.add id ctx.mut_vars} in
      let body_e = gen_expr_impl ~loc ~ctx:ctx' body in
      let pat = Ast_builder.Default.ppat_var ~loc {txt = name; loc} in
      (* Create: let x = ref val in body
         Note: TEVar for mutable vars will dereference with !.
         TEAssign uses := which works on refs. *)
      [%expr
        let [%p pat] = ref [%e val_e] in
        [%e body_e]]
  | _ -> failwith "gen_let_binding: not a let binding expression"

(** Generate control flow (if, for, while) *)
let gen_control_flow ~loc ~gen_expr (te : texpr) : expression =
  match te.te with
  | TEIf (cond, then_e, else_e) ->
      let cond_e = gen_expr ~loc cond in
      let then_e' = gen_expr ~loc then_e in
      let else_e' =
        match else_e with Some e -> gen_expr ~loc e | None -> [%expr ()]
      in
      [%expr if [%e cond_e] then [%e then_e'] else [%e else_e']]
  (* For loop - OCaml for loops use int, but kernel expects int32.
     We use a temporary int variable for the loop, then shadow it with int32 in body. *)
  | TEFor (var_name_str, _var_id, lo, hi, dir, body) -> (
      let lo_e = gen_expr ~loc lo in
      let hi_e = gen_expr ~loc hi in
      let body_e = gen_expr ~loc body in
      (* Use a temporary name for the int loop variable *)
      let int_var_name = var_name_str ^ "__int" in
      let int_var_pat =
        Ast_builder.Default.ppat_var ~loc {txt = int_var_name; loc}
      in
      let int_var_e = evar ~loc int_var_name in
      (* The int32 variable that shadows the int one in the body, using original name *)
      let int32_var_pat =
        Ast_builder.Default.ppat_var ~loc {txt = var_name_str; loc}
      in
      (* Wrap body with int32 conversion *)
      let wrapped_body =
        [%expr
          let [%p int32_var_pat] = Int32.of_int [%e int_var_e] in
          [%e body_e]]
      in
      match dir with
      | Sarek_ast.Upto ->
          (* OCaml for loops are inclusive on both ends, just like Sarek.
             for i = 0 to k - 1l means iterate from 0 to k-1 inclusive. *)
          [%expr
            for
              [%p int_var_pat] = Int32.to_int [%e lo_e]
              to Int32.to_int [%e hi_e]
            do
              [%e wrapped_body]
            done]
      | Sarek_ast.Downto ->
          [%expr
            for
              [%p int_var_pat] = Int32.to_int [%e hi_e]
              downto Int32.to_int [%e lo_e]
            do
              [%e wrapped_body]
            done])
  | TEWhile (cond, body) ->
      let cond_e = gen_expr ~loc cond in
      let body_e = gen_expr ~loc body in
      [%expr
        while [%e cond_e] do
          [%e body_e]
        done]
  | _ -> failwith "gen_control_flow: not a control flow expression"

(** Generate data structures (records, variants, tuples, arrays) *)
let gen_data_structure ~loc ~ctx ~gen_expr (te : texpr) : expression =
  match te.te with
  (* Record construction - qualify field names with module path from type_name.
     For inline types with FCM, use __types.make_typename ~field1:v1 ~field2:v2.
     For same-module types, use unqualified field names. *)
  | TERecord (type_name, fields) -> (
      match String.rindex_opt type_name '.' with
      | Some _ when is_same_module ctx type_name ->
          (* Same-module type - use unqualified record construction *)
          let fields_e =
            List.map
              (fun (name, expr) ->
                ({txt = Lident name; loc}, gen_expr ~loc expr))
              fields
          in
          Ast_builder.Default.pexp_record ~loc fields_e None
      | Some idx ->
          (* External qualified type - use qualified field names *)
          let module_path = String.sub type_name 0 idx in
          let parts = String.split_on_char '.' module_path in
          let field_lid name =
            let rec build_lid = function
              | [] -> Lident name
              | [m] -> Ldot (Lident m, name)
              | m :: rest -> Ldot (build_lid rest, m)
            in
            {txt = build_lid (List.rev parts); loc}
          in
          let fields_e =
            List.map
              (fun (name, expr) -> (field_lid name, gen_expr ~loc expr))
              fields
          in
          Ast_builder.Default.pexp_record ~loc fields_e None
      | None ->
          (* Inline type - check if using first-class module approach *)
          if StringSet.mem type_name ctx.inline_types then
            (* Use maker: __types#make_typename ~field1:v1 ~field2:v2 *)
            let fn_name = record_maker_name type_name in
            let method_call =
              Ast_builder.Default.pexp_send
                ~loc
                (evar ~loc types_module_var)
                {txt = fn_name; loc}
            in
            (* Build labelled arguments *)
            let args =
              List.map
                (fun (name, expr) -> (Labelled name, gen_expr ~loc expr))
                fields
            in
            Ast_builder.Default.pexp_apply ~loc method_call args
          else
            (* Direct record construction *)
            let fields_e =
              List.map
                (fun (name, expr) ->
                  ({txt = Lident name; loc}, gen_expr ~loc expr))
                fields
            in
            Ast_builder.Default.pexp_record ~loc fields_e None)
  (* Variant construction - qualify constructor with module path.
     For inline types with FCM, use __types.make_typename_Ctor arg.
     For same-module types, use unqualified constructors. *)
  | TEConstr (type_name, constr_name, arg) -> (
      match String.rindex_opt type_name '.' with
      | Some _ when is_same_module ctx type_name ->
          (* Same-module type - use unqualified constructor *)
          let arg_e = Option.map (gen_expr ~loc) arg in
          Ast_builder.Default.pexp_construct
            ~loc
            {txt = Lident constr_name; loc}
            arg_e
      | Some idx ->
          (* External qualified type - use qualified constructor *)
          let module_path = String.sub type_name 0 idx in
          let parts = String.split_on_char '.' module_path in
          let rec build_lid = function
            | [] -> Lident constr_name
            | [m] -> Ldot (Lident m, constr_name)
            | m :: rest -> Ldot (build_lid rest, m)
          in
          let constr_lid = build_lid (List.rev parts) in
          let arg_e = Option.map (gen_expr ~loc) arg in
          Ast_builder.Default.pexp_construct ~loc {txt = constr_lid; loc} arg_e
      | None ->
          (* Inline type - check if using first-class module approach *)
          if StringSet.mem type_name ctx.inline_types then
            (* Use maker: __types#make_typename_Ctor arg or __types#make_typename_Ctor () *)
            let fn_name = variant_ctor_name type_name constr_name in
            let method_call =
              Ast_builder.Default.pexp_send
                ~loc
                (evar ~loc types_module_var)
                {txt = fn_name; loc}
            in
            let arg_e =
              match arg with Some a -> gen_expr ~loc a | None -> [%expr ()]
            in
            Ast_builder.Default.pexp_apply ~loc method_call [(Nolabel, arg_e)]
          else
            (* Direct constructor *)
            let arg_e = Option.map (gen_expr ~loc) arg in
            Ast_builder.Default.pexp_construct
              ~loc
              {txt = Lident constr_name; loc}
              arg_e)
  (* Tuple *)
  | TETuple exprs ->
      let exprs_e = List.map (gen_expr ~loc) exprs in
      Ast_builder.Default.pexp_tuple ~loc exprs_e
  (* Create local array - use regular OCaml arrays for native mode *)
  | TECreateArray (size, elem_ty, _memspace) ->
      let size_e = gen_expr ~loc size in
      let default_e = default_value_for_type ~loc elem_ty in
      [%expr Array.make [%e size_e] [%e default_e]]
  | _ -> failwith "gen_data_structure: not a data structure expression"

(** Generate special expressions (return, global ref, native, pragma, open) *)
let gen_special_expr ~loc ~gen_expr (te : texpr) : expression =
  match te.te with
  (* Return - just evaluate the expression *)
  | TEReturn e -> gen_expr ~loc e
  (* Global ref - reference to external value *)
  | TEGlobalRef (name, _typ) ->
      (* Dereference the ref *)
      let var_e = evar ~loc name in
      [%expr ![%e var_e]]
  (* Native code with OCaml fallback - use the OCaml expression directly *)
  | TENative {ocaml; _} ->
      (* The ocaml expression is a function that will be applied to arguments.
         Return it as-is; TEApp will handle the application. *)
      ocaml
  (* Pragma - just evaluate body (pragmas are hints for GPU) *)
  | TEPragma (_opts, body) -> gen_expr ~loc body
  (* Module open - generate let open M.N in body *)
  | TEOpen (path, body) ->
      let body_e = gen_expr ~loc body in
      (* Build the module path longident: M.N.O *)
      let mod_lid =
        match path with
        | [] -> failwith "empty module path in TEOpen"
        | [m] -> Lident m
        | m :: rest ->
            List.fold_left (fun acc p -> Ldot (acc, p)) (Lident m) rest
      in
      Ast_builder.Default.pexp_open
        ~loc
        (Ast_builder.Default.open_infos
           ~loc
           ~override:Fresh
           ~expr:(Ast_builder.Default.pmod_ident ~loc {txt = mod_lid; loc}))
        body_e
  | _ -> failwith "gen_special_expr: not a special expression"

(** Generate BSP parallel constructs (let%shared, let%superstep, let rec) *)
let gen_parallel_construct ~loc ~gen_expr (te : texpr) : expression =
  match te.te with
  (* BSP let%shared - allocate shared memory using OCaml arrays *)
  | TELetShared (name, _id, elem_ty, size_opt, body) ->
      (* Size needs to be int, but expressions may be int32 (like block_dim_x).
         Wrap in Int32.to_int for conversion. *)
      let size_e =
        match size_opt with
        | Some s ->
            let s_e = gen_expr ~loc s in
            [%expr Int32.to_int [%e s_e]]
        | None ->
            (* Default to block_dim_x - convert from int32 to int *)
            let state = evar ~loc state_var in
            [%expr Int32.to_int [%e state].Sarek.Sarek_cpu_runtime.block_dim_x]
      in
      let shared = evar ~loc shared_var in
      let body_e = gen_expr ~loc body in
      let pat = Ast_builder.Default.ppat_var ~loc {txt = name; loc} in
      let name_e = Ast_builder.Default.estring ~loc name in
      (* Use typed allocators for common types, generic for custom types *)
      let alloc_expr =
        match repr elem_ty with
        | TReg Float32 | TReg Float64 ->
            [%expr
              Sarek.Sarek_cpu_runtime.alloc_shared_float
                [%e shared]
                [%e name_e]
                [%e size_e]
                0.0]
        | TPrim TInt32 | TReg Int ->
            [%expr
              Sarek.Sarek_cpu_runtime.alloc_shared_int32
                [%e shared]
                [%e name_e]
                [%e size_e]
                0l]
        | TReg Int64 ->
            [%expr
              Sarek.Sarek_cpu_runtime.alloc_shared_int64
                [%e shared]
                [%e name_e]
                [%e size_e]
                0L]
        | TReg Char ->
            [%expr
              Sarek.Sarek_cpu_runtime.alloc_shared
                [%e shared]
                [%e name_e]
                [%e size_e]
                '\000']
        | _ ->
            (* For custom types, generate proper default value *)
            let default_val = default_value_for_type ~loc elem_ty in
            [%expr
              Sarek.Sarek_cpu_runtime.alloc_shared
                [%e shared]
                [%e name_e]
                [%e size_e]
                [%e default_val]]
      in
      [%expr
        let [%p pat] = [%e alloc_expr] in
        [%e body_e]]
  (* BSP let%superstep - synchronized block + barrier *)
  | TESuperstep (_name, _divergent, step_body, cont) ->
      let body_e = gen_expr ~loc step_body in
      let cont_e = gen_expr ~loc cont in
      let state = evar ~loc state_var in
      [%expr
        [%e body_e] ;
        [%e state].Sarek.Sarek_cpu_runtime.barrier () ;
        [%e cont_e]]
  (* Recursive let binding *)
  | TELetRec (name, _id, params, fn_body, cont) ->
      (* Generate: let rec name p1 p2 ... = body in cont *)
      let fn_body_e = gen_expr ~loc fn_body in
      let cont_e = gen_expr ~loc cont in
      (* Create function with parameters *)
      let fn_expr =
        List.fold_right
          (fun p acc ->
            let pvar =
              Ast_builder.Default.ppat_var ~loc {txt = p.tparam_name; loc}
            in
            Ast_builder.Default.pexp_fun ~loc Nolabel None pvar acc)
          params
          fn_body_e
      in
      let binding =
        Ast_builder.Default.value_binding
          ~loc
          ~pat:(Ast_builder.Default.ppat_var ~loc {txt = name; loc})
          ~expr:fn_expr
      in
      Ast_builder.Default.pexp_let ~loc Recursive [binding] cont_e
  | _ -> failwith "gen_parallel_construct: not a parallel construct"

(** Generate OCaml expression from typed Sarek expression.
    @param ctx Generation context with mutable vars and inline types *)
let rec gen_expr_impl ~loc:_ ~ctx (te : texpr) : expression =
  let loc = ppxlib_loc_of_sarek te.te_loc in
  (* Helper to recursively generate, passing ctx *)
  let gen_expr ~loc e = gen_expr_impl ~loc ~ctx e in
  match te.te with
  (* Literals *)
  | TEUnit | TEBool _ | TEInt _ | TEInt32 _ | TEInt64 _ | TEFloat _ | TEDouble _
    ->
      gen_literal ~loc te
  (* Variables *)
  | TEVar (name, id) -> gen_variable ~loc ~ctx name id
  (* Memory access *)
  | TEVecGet _ | TEVecSet _ | TEArrGet _ | TEArrSet _ | TEFieldGet _
  | TEFieldSet _ ->
      gen_memory_access ~loc ~ctx ~gen_expr te
  (* Control flow *)
  | TEIf _ | TEFor _ | TEWhile _ -> gen_control_flow ~loc ~gen_expr te
  (* Binary operations *)
  | TEBinop (op, a, b) ->
      let a_e = gen_expr ~loc a in
      let b_e = gen_expr ~loc b in
      gen_binop ~loc op a_e b_e a.ty
  (* Unary operations *)
  | TEUnop (op, a) ->
      let a_e = gen_expr ~loc a in
      gen_unop ~loc op a_e a.ty
  (* Function application *)
  | TEApp (fn, args) ->
      let fn_e = gen_expr ~loc fn in
      (* When use_native_arg is true, vector arguments need #underlying
         to get the actual Vector.t for functions that expect it *)
      let gen_arg arg =
        let arg_e = gen_expr ~loc arg in
        if ctx.use_native_arg then
          match repr arg.ty with
          | TVec _ -> [%expr [%e arg_e]#underlying]
          | _ -> arg_e
        else arg_e
      in
      let args_e = List.map gen_arg args in
      Ast_builder.Default.pexp_apply
        ~loc
        fn_e
        (List.map (fun a -> (Nolabel, a)) args_e)
  (* Let bindings and assignment *)
  | TEAssign _ | TELet _ | TELetMut _ ->
      gen_let_binding ~loc ~ctx ~gen_expr ~gen_expr_impl te
  (* Sequence *)
  | TESeq exprs -> (
      let exprs_e = List.map (gen_expr ~loc) exprs in
      match exprs_e with
      | [] -> [%expr ()]
      | [e] -> e
      | es ->
          List.fold_right
            (fun e acc ->
              [%expr
                [%e e] ;
                [%e acc]])
            (List.rev (List.tl (List.rev es)))
            (List.hd (List.rev es)))
  (* Match *)
  | TEMatch (scrutinee, cases) ->
      let scrut_e = gen_expr ~loc scrutinee in
      let cases_e =
        List.map
          (fun (pat, body) ->
            let pat_e = gen_pattern_impl ~loc ~ctx pat in
            let body_e = gen_expr ~loc body in
            Ast_builder.Default.case ~lhs:pat_e ~guard:None ~rhs:body_e)
          cases
      in
      Ast_builder.Default.pexp_match ~loc scrut_e cases_e
  (* Data structures *)
  | TERecord _ | TEConstr _ | TETuple _ | TECreateArray _ ->
      gen_data_structure ~loc ~ctx ~gen_expr te
  (* Special expressions *)
  | TEReturn _ | TEGlobalRef _ | TENative _ | TEPragma _ | TEOpen _ ->
      gen_special_expr ~loc ~gen_expr te
  (* Intrinsic constant - thread indices, etc. *)
  | TEIntrinsicConst ref -> gen_intrinsic_const ~loc ~gen_mode:ctx.gen_mode ref
  (* Intrinsic function - math functions, barriers, etc. *)
  | TEIntrinsicFun (ref, _convergence, args) ->
      (* When use_native_arg is true, vector arguments need #underlying *)
      let gen_arg arg =
        let arg_e = gen_expr ~loc arg in
        if ctx.use_native_arg then
          match repr arg.ty with
          | TVec _ -> [%expr [%e arg_e]#underlying]
          | _ -> arg_e
        else arg_e
      in
      let args_e = List.map gen_arg args in
      gen_intrinsic_fun ~loc ~gen_mode:ctx.gen_mode ref args_e
  (* Parallel constructs *)
  | TELetShared _ | TESuperstep _ | TELetRec _ ->
      gen_parallel_construct ~loc ~gen_expr te

(** Generate pattern from typed pattern. Takes context to detect same-module
    types that shouldn't be qualified. *)
and gen_pattern_impl ~loc:_ ~ctx (tpat : tpattern) : pattern =
  let loc = ppxlib_loc_of_sarek tpat.tpat_loc in
  match tpat.tpat with
  | TPAny -> Ast_builder.Default.ppat_any ~loc
  | TPVar (name, _id) ->
      (* Use original name for pattern variables *)
      Ast_builder.Default.ppat_var ~loc {txt = name; loc}
  | TPConstr (type_name, constr_name, arg) ->
      let arg_p = Option.map (gen_pattern_impl ~loc ~ctx) arg in
      (* Qualify constructor with module path from type_name if present.
         For "Geometry_lib.shape", we need Geometry_lib.Circle, not Circle.
         For inline types or same-module types, use unqualified name. *)
      let constr_lid =
        match String.rindex_opt type_name '.' with
        | Some idx ->
            (* Check if this is from the current module - use unqualified if so *)
            if is_same_module ctx type_name then Lident constr_name
            else
              let module_path = String.sub type_name 0 idx in
              let parts = String.split_on_char '.' module_path in
              let rec build_lid = function
                | [] -> Lident constr_name
                | [m] -> Ldot (Lident m, constr_name)
                | m :: rest -> Ldot (build_lid rest, m)
              in
              build_lid (List.rev parts)
        | None ->
            (* Inline type - use unqualified name *)
            Lident constr_name
      in
      Ast_builder.Default.ppat_construct ~loc {txt = constr_lid; loc} arg_p
  | TPTuple pats ->
      let pats_p = List.map (gen_pattern_impl ~loc ~ctx) pats in
      Ast_builder.Default.ppat_tuple ~loc pats_p

(** Generate binary operation *)
and gen_binop ~loc op a b ty : expression =
  let ty_repr = repr ty in
  match op with
  | Sarek_ast.Add -> (
      match ty_repr with
      | TReg Float32 | TReg Float64 -> [%expr [%e a] +. [%e b]]
      | TReg Int | TPrim TInt32 -> [%expr Int32.add [%e a] [%e b]]
      | TReg Int64 -> [%expr Int64.add [%e a] [%e b]]
      | _ -> [%expr [%e a] + [%e b]])
  | Sarek_ast.Sub -> (
      match ty_repr with
      | TReg Float32 | TReg Float64 -> [%expr [%e a] -. [%e b]]
      | TReg Int | TPrim TInt32 -> [%expr Int32.sub [%e a] [%e b]]
      | TReg Int64 -> [%expr Int64.sub [%e a] [%e b]]
      | _ -> [%expr [%e a] - [%e b]])
  | Sarek_ast.Mul -> (
      match ty_repr with
      | TReg Float32 | TReg Float64 -> [%expr [%e a] *. [%e b]]
      | TReg Int | TPrim TInt32 -> [%expr Int32.mul [%e a] [%e b]]
      | TReg Int64 -> [%expr Int64.mul [%e a] [%e b]]
      | _ -> [%expr [%e a] * [%e b]])
  | Sarek_ast.Div -> (
      match ty_repr with
      | TReg Float32 | TReg Float64 -> [%expr [%e a] /. [%e b]]
      | TReg Int | TPrim TInt32 -> [%expr Int32.div [%e a] [%e b]]
      | TReg Int64 -> [%expr Int64.div [%e a] [%e b]]
      | _ -> [%expr [%e a] / [%e b]])
  | Sarek_ast.Mod -> (
      match ty_repr with
      | TReg Int | TPrim TInt32 -> [%expr Int32.rem [%e a] [%e b]]
      | TReg Int64 -> [%expr Int64.rem [%e a] [%e b]]
      | _ -> [%expr [%e a] mod [%e b]])
  | Sarek_ast.And -> [%expr [%e a] && [%e b]]
  | Sarek_ast.Or -> [%expr [%e a] || [%e b]]
  (* Comparison operators work polymorphically in OCaml *)
  | Sarek_ast.Eq -> [%expr [%e a] = [%e b]]
  | Sarek_ast.Ne -> [%expr [%e a] <> [%e b]]
  | Sarek_ast.Lt -> [%expr [%e a] < [%e b]]
  | Sarek_ast.Gt -> [%expr [%e a] > [%e b]]
  | Sarek_ast.Le -> [%expr [%e a] <= [%e b]]
  | Sarek_ast.Ge -> [%expr [%e a] >= [%e b]]
  (* Bitwise operations *)
  | Sarek_ast.Land -> (
      match ty_repr with
      | TReg Int | TPrim TInt32 -> [%expr Int32.logand [%e a] [%e b]]
      | TReg Int64 -> [%expr Int64.logand [%e a] [%e b]]
      | _ -> [%expr [%e a] land [%e b]])
  | Sarek_ast.Lor -> (
      match ty_repr with
      | TReg Int | TPrim TInt32 -> [%expr Int32.logor [%e a] [%e b]]
      | TReg Int64 -> [%expr Int64.logor [%e a] [%e b]]
      | _ -> [%expr [%e a] lor [%e b]])
  | Sarek_ast.Lxor -> (
      match ty_repr with
      | TReg Int | TPrim TInt32 -> [%expr Int32.logxor [%e a] [%e b]]
      | TReg Int64 -> [%expr Int64.logxor [%e a] [%e b]]
      | _ -> [%expr [%e a] lxor [%e b]])
  | Sarek_ast.Lsl -> (
      match ty_repr with
      | TReg Int | TPrim TInt32 ->
          [%expr Int32.shift_left [%e a] (Int32.to_int [%e b])]
      | TReg Int64 -> [%expr Int64.shift_left [%e a] (Int64.to_int [%e b])]
      | _ -> [%expr [%e a] lsl [%e b]])
  | Sarek_ast.Lsr -> (
      match ty_repr with
      | TReg Int | TPrim TInt32 ->
          [%expr Int32.shift_right_logical [%e a] (Int32.to_int [%e b])]
      | TReg Int64 ->
          [%expr Int64.shift_right_logical [%e a] (Int64.to_int [%e b])]
      | _ -> [%expr [%e a] lsr [%e b]])
  | Sarek_ast.Asr -> (
      match ty_repr with
      | TReg Int | TPrim TInt32 ->
          [%expr Int32.shift_right [%e a] (Int32.to_int [%e b])]
      | TReg Int64 -> [%expr Int64.shift_right [%e a] (Int64.to_int [%e b])]
      | _ -> [%expr [%e a] asr [%e b]])

(** Generate unary operation *)
and gen_unop ~loc op a ty : expression =
  let ty_repr = repr ty in
  match op with
  | Sarek_ast.Neg -> (
      match ty_repr with
      | TReg Float32 | TReg Float64 -> [%expr -.[%e a]]
      | TReg Int | TPrim TInt32 -> [%expr Int32.neg [%e a]]
      | TReg Int64 -> [%expr Int64.neg [%e a]]
      | _ -> [%expr -[%e a]])
  | Sarek_ast.Not -> [%expr not [%e a]]
  | Sarek_ast.Lnot -> (
      match ty_repr with
      | TReg Int | TPrim TInt32 -> [%expr Int32.lognot [%e a]]
      | TReg Int64 -> [%expr Int64.lognot [%e a]]
      | _ -> [%expr lnot [%e a]])

(** Extract module name from a Sarek location (file path). For
    "/path/to/test_registered_variant.ml", returns "Test_registered_variant". *)
let module_name_of_sarek_loc (loc : Sarek_ast.loc) : string =
  let file = loc.loc_file in
  let base = Filename.(remove_extension (basename file)) in
  String.capitalize_ascii base

(** Top-level entry point for generating expressions. Starts with empty context.
*)
let gen_expr ~loc e : expression = gen_expr_impl ~loc ~ctx:empty_ctx e

(** Generate expression with inline types context for first-class modules. *)
let gen_expr_with_inline_types ~loc ~inline_type_names ~current_module e :
    expression =
  let ctx = {empty_ctx with inline_types = inline_type_names; current_module} in
  gen_expr_impl ~loc ~ctx e

(** {1 Module Item Generation} *)

(** Generate a module-level function (TMFun) as a let binding. *)
let gen_module_fun ~loc (name : string) (params : tparam list) (body : texpr) :
    pattern * expression =
  let fn_pat = Ast_builder.Default.ppat_var ~loc {txt = name; loc} in
  let body_e = gen_expr ~loc body in

  (* Build function from parameters - use parameter names with type constraints
     to help OCaml resolve record fields and other types *)
  let fn_e =
    List.fold_right
      (fun param acc ->
        let var_pat =
          Ast_builder.Default.ppat_var ~loc {txt = param.tparam_name; loc}
        in
        let ty = core_type_of_typ ~loc param.tparam_type in
        let param_pat = Ast_builder.Default.ppat_constraint ~loc var_pat ty in
        [%expr fun [%p param_pat] -> [%e acc]])
      params
      body_e
  in
  (fn_pat, fn_e)

(** Generate a module-level constant (TMConst) as a let binding. *)
let gen_module_const ~loc (name : string) (_id : int) (_typ : typ)
    (value : texpr) : pattern * expression =
  (* Use the constant name, not the ID *)
  let const_pat = Ast_builder.Default.ppat_var ~loc {txt = name; loc} in
  let value_e = gen_expr ~loc value in
  (const_pat, value_e)

(** {1 Type Declaration Generation} *)

(** Generate a type declaration for a record type *)
let gen_type_decl_record ~loc (name : string)
    (fields : (string * typ * bool) list) : structure_item =
  let field_decls =
    List.map
      (fun (fname, fty, is_mutable) ->
        let ty = core_type_of_typ ~loc fty in
        Ast_builder.Default.label_declaration
          ~loc
          ~name:{txt = fname; loc}
          ~mutable_:(if is_mutable then Mutable else Immutable)
          ~type_:ty)
      fields
  in
  Ast_builder.Default.pstr_type
    ~loc
    Recursive
    [
      Ast_builder.Default.type_declaration
        ~loc
        ~name:{txt = name; loc}
        ~params:[]
        ~cstrs:[]
        ~kind:(Ptype_record field_decls)
        ~private_:Public
        ~manifest:None;
    ]

(** Generate a type declaration for a variant type *)
let gen_type_decl_variant ~loc (name : string)
    (constrs : (string * typ option) list) : structure_item =
  let constr_decls =
    List.map
      (fun (cname, arg_opt) ->
        let args =
          match arg_opt with
          | None -> Pcstr_tuple []
          | Some ty -> Pcstr_tuple [core_type_of_typ ~loc ty]
        in
        Ast_builder.Default.constructor_declaration
          ~loc
          ~name:{txt = cname; loc}
          ~args
          ~res:None)
      constrs
  in
  Ast_builder.Default.pstr_type
    ~loc
    Recursive
    [
      Ast_builder.Default.type_declaration
        ~loc
        ~name:{txt = name; loc}
        ~params:[]
        ~cstrs:[]
        ~kind:(Ptype_variant constr_decls)
        ~private_:Public
        ~manifest:None;
    ]

(** Generate a structure item from a typed type declaration *)
let gen_type_decl_item ~loc (decl : ttype_decl) : structure_item =
  match decl with
  | TTypeRecord {tdecl_name; tdecl_fields; _} ->
      gen_type_decl_record ~loc tdecl_name tdecl_fields
  | TTypeVariant {tdecl_name; tdecl_constructors; _} ->
      gen_type_decl_variant ~loc tdecl_name tdecl_constructors

(** Wrap expression with module item bindings. *)
let wrap_module_items ~loc (items : tmodule_item list) (body : expression) :
    expression =
  List.fold_right
    (fun item acc ->
      match item with
      | TMFun (name, is_rec, params, item_body) ->
          let pat, expr = gen_module_fun ~loc name params item_body in
          if is_rec then
            (* Generate let rec for recursive functions *)
            [%expr
              let rec [%p pat] = [%e expr] in
              [%e acc]]
          else
            [%expr
              let [%p pat] = [%e expr] in
              [%e acc]]
      | TMConst (name, id, typ, value) ->
          let pat, expr = gen_module_const ~loc name id typ value in
          [%expr
            let [%p pat] = [%e expr] in
            [%e acc]])
    items
    body

(** {1 First-Class Module Approach for Inline Types}

    To avoid "type escapes its scope" errors, we use first-class modules to
    encapsulate inline types. The approach:

    1. Generate a module signature KERNEL_TYPES with abstract types and
    accessors 2. Generate a concrete module implementation with the actual types
    3. Transform the kernel body to use T.get_field and T.make_type accessors 4.
    Pass (module T : KERNEL_TYPES) as a parameter to the kernel

    This keeps the concrete types hidden behind an existential type. *)

(** Generate a concrete module implementation for inline types (types only).
    Example for record type point with fields x and y: struct type point =
    record with fields x: float and y: float end

    Note: We only generate type declarations here, not accessor functions. The
    accessor functions are generated as object methods by gen_types_object. *)
let gen_module_impl ~loc (decls : ttype_decl list) : module_expr =
  let struct_items =
    List.map
      (fun decl ->
        match decl with
        | TTypeRecord {tdecl_name; tdecl_fields; _} ->
            gen_type_decl_record ~loc tdecl_name tdecl_fields
        | TTypeVariant {tdecl_name; tdecl_constructors; _} ->
            gen_type_decl_variant ~loc tdecl_name tdecl_constructors)
      decls
  in
  Ast_builder.Default.pmod_structure ~loc struct_items

(** Get only inline type declarations (those without module prefix).
    External/registered types have qualified names like "Module.type". *)
let inline_type_decls (decls : ttype_decl list) : ttype_decl list =
  List.filter
    (fun decl ->
      let name =
        match decl with
        | TTypeRecord {tdecl_name; _} -> tdecl_name
        | TTypeVariant {tdecl_name; _} -> tdecl_name
      in
      (* Only include types without '.' - these are inline definitions *)
      not (String.contains name '.'))
    decls

(** Check if a kernel has inline types that need first-class module handling *)
let has_inline_types (kernel : tkernel) : bool =
  inline_type_decls kernel.tkern_type_decls <> []

(** {1 Kernel Generation} *)

(** Convert execution strategy to generation mode *)
let gen_mode_of_exec_strategy : Sarek_convergence.exec_strategy -> gen_mode =
  function
  | Sarek_convergence.Simple1D -> Simple1DMode
  | Sarek_convergence.Simple2D -> Simple2DMode
  | Sarek_convergence.Simple3D -> Simple3DMode
  | Sarek_convergence.FullState -> FullMode

(** Generate a type cast expression for extracting a kernel argument.

    For vectors: extract NA_Vec and create an accessor wrapper. For scalars:
    match on NA_Int32/NA_Float32/etc and extract the value.

    Uses typed native_arg for type safety. *)
let gen_arg_cast ~loc (param : tparam) (idx : int) : expression =
  let arr_access =
    [%expr Array.get __args [%e Ast_builder.Default.eint ~loc idx]]
  in
  match repr param.tparam_type with
  | TVec elem_ty -> (
      (* Generate accessor object based on element type.
         All accessor objects have an `underlying` method to get the actual
         Vector.t for passing to functions/intrinsics that need it. *)
      let vec_arg = [%expr [%e arr_access]] in
      match repr elem_ty with
      | TReg Float32 ->
          [%expr
            match [%e arr_access] with
            | Sarek_ir_types.NA_Vec (Sarek_ir_types.NV __v) ->
                object
                  method get i = __v.get_f32 i

                  method set i x = __v.set_f32 i x

                  method length = __v.length

                  method underlying =
                    Sarek_ir_types.vec_as_vector
                      [%e vector_type_id_expr ~loc elem_ty]
                      [%e vec_arg]
                end
            | _ -> failwith "Expected NA_Vec"]
      | TReg Float64 ->
          [%expr
            match [%e arr_access] with
            | Sarek_ir_types.NA_Vec (Sarek_ir_types.NV __v) ->
                object
                  method get i = __v.get_f64 i

                  method set i x = __v.set_f64 i x

                  method length = __v.length

                  method underlying =
                    Sarek_ir_types.vec_as_vector
                      [%e vector_type_id_expr ~loc elem_ty]
                      [%e vec_arg]
                end
            | _ -> failwith "Expected NA_Vec"]
      | TReg Int | TPrim TInt32 ->
          [%expr
            match [%e arr_access] with
            | Sarek_ir_types.NA_Vec (Sarek_ir_types.NV __v) ->
                object
                  method get i = __v.get_i32 i

                  method set i x = __v.set_i32 i x

                  method length = __v.length

                  method underlying =
                    Sarek_ir_types.vec_as_vector
                      [%e vector_type_id_expr ~loc elem_ty]
                      [%e vec_arg]
                end
            | _ -> failwith "Expected NA_Vec"]
      | TReg Int64 ->
          [%expr
            match [%e arr_access] with
            | Sarek_ir_types.NA_Vec (Sarek_ir_types.NV __v) ->
                object
                  method get i = __v.get_i64 i

                  method set i x = __v.set_i64 i x

                  method length = __v.length

                  method underlying =
                    Sarek_ir_types.vec_as_vector
                      [%e vector_type_id_expr ~loc elem_ty]
                      [%e vec_arg]
                end
            | _ -> failwith "Expected NA_Vec"]
      | _ ->
          (* Custom types (records, variants): use typed helper functions.
              The helpers encapsulate the type conversion internally. *)
          [%expr
            object
              method get i =
                Sarek_ir_types.vec_get_custom
                  [%e custom_type_id_expr ~loc elem_ty]
                  [%e vec_arg]
                  i

              method set i x =
                Sarek_ir_types.vec_set_custom
                  [%e custom_type_id_expr ~loc elem_ty]
                  [%e vec_arg]
                  i
                  x

              method length = Sarek_ir_types.vec_length [%e vec_arg]

              method underlying =
                Sarek_ir_types.vec_as_vector
                  [%e vector_type_id_expr ~loc elem_ty]
                  [%e vec_arg]
            end])
  | TReg Float32 ->
      [%expr
        match [%e arr_access] with
        | Sarek_ir_types.NA_Float32 v -> v
        | Sarek_ir_types.NA_Int32 n -> Int32.to_float n
        | _ -> failwith "Expected NA_Float32"]
  | TReg Float64 ->
      [%expr
        match [%e arr_access] with
        | Sarek_ir_types.NA_Float64 v -> v
        | Sarek_ir_types.NA_Float32 v -> v
        | _ -> failwith "Expected NA_Float64"]
  | TReg Int | TPrim TInt32 ->
      [%expr
        match [%e arr_access] with
        | Sarek_ir_types.NA_Int32 v -> v
        | _ -> failwith "Expected NA_Int32"]
  | TReg Int64 ->
      [%expr
        match [%e arr_access] with
        | Sarek_ir_types.NA_Int64 v -> v
        | Sarek_ir_types.NA_Int32 n -> Int64.of_int32 n
        | _ -> failwith "Expected NA_Int64"]
  | TPrim TBool ->
      [%expr
        match [%e arr_access] with
        | Sarek_ir_types.NA_Int32 n -> n <> 0l
        | _ -> failwith "Expected NA_Int32 for bool"]
  | _ ->
      (* Default - failwith for unsupported types *)
      [%expr failwith "Unsupported native_arg type"]

(** Generate an object expression with accessor methods for FCM. Example for
    type point with fields x and y: object method get_point_x r = r.x method
    get_point_y r = r.y method make_point ~x ~y = record with fields x and y end

    Using an object avoids needing to define a record type for the accessors. *)
let gen_types_object ~loc (decls : ttype_decl list) : expression =
  let methods =
    List.concat_map
      (fun decl ->
        match decl with
        | TTypeRecord {tdecl_name; tdecl_fields; _} ->
            (* Getters *)
            let getters =
              List.map
                (fun (fname, _fty, _is_mut) ->
                  let fn_name = field_getter_name tdecl_name fname in
                  let field_lid = {txt = Lident fname; loc} in
                  let fn_expr =
                    [%expr
                      fun __r ->
                        [%e
                          Ast_builder.Default.pexp_field
                            ~loc
                            [%expr __r]
                            field_lid]]
                  in
                  Ast_builder.Default.pcf_method
                    ~loc
                    ({txt = fn_name; loc}, Public, Cfk_concrete (Fresh, fn_expr)))
                tdecl_fields
            in
            (* Maker *)
            let maker =
              let fn_name = record_maker_name tdecl_name in
              let param_pats =
                List.map
                  (fun (fname, _fty, _) ->
                    ( Labelled fname,
                      Ast_builder.Default.ppat_var ~loc {txt = fname; loc} ))
                  tdecl_fields
              in
              let record_fields =
                List.map
                  (fun (fname, _, _) ->
                    ( {txt = Lident fname; loc},
                      Ast_builder.Default.pexp_ident
                        ~loc
                        {txt = Lident fname; loc} ))
                  tdecl_fields
              in
              let record_expr =
                Ast_builder.Default.pexp_record ~loc record_fields None
              in
              let fn_expr =
                List.fold_right
                  (fun (lbl, pat) body ->
                    Ast_builder.Default.pexp_fun ~loc lbl None pat body)
                  param_pats
                  record_expr
              in
              Ast_builder.Default.pcf_method
                ~loc
                ({txt = fn_name; loc}, Public, Cfk_concrete (Fresh, fn_expr))
            in
            getters @ [maker]
        | TTypeVariant {tdecl_name; tdecl_constructors; _} ->
            (* Constructor functions *)
            List.map
              (fun (cname, arg_opt) ->
                let fn_name = variant_ctor_name tdecl_name cname in
                let ctor_lid = {txt = Lident cname; loc} in
                let fn_expr =
                  match arg_opt with
                  | None ->
                      [%expr
                        fun () ->
                          [%e
                            Ast_builder.Default.pexp_construct
                              ~loc
                              ctor_lid
                              None]]
                  | Some _ ->
                      [%expr
                        fun __x ->
                          [%e
                            Ast_builder.Default.pexp_construct
                              ~loc
                              ctor_lid
                              (Some [%expr __x])]]
                in
                Ast_builder.Default.pcf_method
                  ~loc
                  ({txt = fn_name; loc}, Public, Cfk_concrete (Fresh, fn_expr)))
              tdecl_constructors)
      decls
  in
  Ast_builder.Default.pexp_object
    ~loc
    (Ast_builder.Default.class_structure
       ~self:(Ast_builder.Default.ppat_any ~loc)
       ~fields:methods)

(** Generate V2 cpu_kern - uses Spoc_core.Vector.get/set instead of Spoc.Mem.

    Generated signature: thread_state -> shared_mem -> args_tuple -> unit This
    matches the signature expected by run_parallel/run_sequential. *)
let gen_cpu_kern_native ~loc (kernel : tkernel) : expression =
  let use_fcm = has_inline_types kernel in
  let inline_type_names =
    if use_fcm then
      List.fold_left
        (fun acc decl ->
          let name =
            match decl with
            | TTypeRecord {tdecl_name; _} -> tdecl_name
            | TTypeVariant {tdecl_name; _} -> tdecl_name
          in
          if not (String.contains name '.') then StringSet.add name acc else acc)
        StringSet.empty
        kernel.tkern_type_decls
    else StringSet.empty
  in
  let current_module = Some (module_name_of_sarek_loc kernel.tkern_loc) in

  (* Use native_arg mode - vectors are accessor objects with typed helpers.
     Complex types use vec_get_custom/vec_set_custom from runtime. *)
  let body_e =
    if use_fcm then
      let ctx =
        {
          empty_ctx with
          inline_types = inline_type_names;
          current_module;
          use_native_arg = true;
        }
      in
      gen_expr_impl ~loc ~ctx kernel.tkern_body
    else
      let ctx = {empty_ctx with current_module; use_native_arg = true} in
      gen_expr_impl ~loc ~ctx kernel.tkern_body
  in

  (* Generate inline module items *)
  let inline_items =
    let all_items = kernel.tkern_module_items in
    let skip_count = kernel.tkern_external_item_count in
    let rec drop n lst =
      if n <= 0 then lst
      else match lst with [] -> [] | _ :: tl -> drop (n - 1) tl
    in
    drop skip_count all_items
  in
  let body_with_items = wrap_module_items ~loc inline_items body_e in

  (* Build parameter tuple pattern - no type constraints since vectors are
     accessor objects, not Vector.t *)
  let param_pats =
    List.map
      (fun p -> Ast_builder.Default.ppat_var ~loc {txt = p.tparam_name; loc})
      kernel.tkern_params
  in
  let params_pat =
    match param_pats with
    | [] -> [%pat? ()]
    | [p] -> p
    | ps -> Ast_builder.Default.ppat_tuple ~loc ps
  in

  let state_pat = Ast_builder.Default.ppat_var ~loc {txt = state_var; loc} in
  let shared_pat = Ast_builder.Default.ppat_var ~loc {txt = shared_var; loc} in

  let inner_fun =
    if use_fcm then
      let types_pat =
        Ast_builder.Default.ppat_var ~loc {txt = types_module_var; loc}
      in
      [%expr
        fun [%p types_pat]
            ([%p state_pat] : Sarek.Sarek_cpu_runtime.thread_state)
            ([%p shared_pat] : Sarek.Sarek_cpu_runtime.shared_mem)
            [%p params_pat] -> [%e body_with_items]]
    else
      [%expr
        fun ([%p state_pat] : Sarek.Sarek_cpu_runtime.thread_state)
            ([%p shared_pat] : Sarek.Sarek_cpu_runtime.shared_mem)
            [%p params_pat] -> [%e body_with_items]]
  in
  (* Add warning suppression attribute *)
  {
    inner_fun with
    pexp_attributes =
      [
        Ast_builder.Default.attribute
          ~loc
          ~name:{txt = "warning"; loc}
          ~payload:
            (PStr
               [
                 Ast_builder.Default.pstr_eval
                   ~loc
                   (Ast_builder.Default.estring ~loc "-27-32-33")
                   [];
               ]);
      ];
  }

(** Generate simple kernel for optimized threadpool execution using the modern
    vector path (Spoc_core.Vector). *)
let gen_simple_cpu_kern_native ~loc ~exec_strategy (kernel : tkernel) :
    expression =
  let use_fcm = has_inline_types kernel in
  let inline_type_names =
    if use_fcm then
      List.fold_left
        (fun acc decl ->
          let name =
            match decl with
            | TTypeRecord {tdecl_name; _} -> tdecl_name
            | TTypeVariant {tdecl_name; _} -> tdecl_name
          in
          if not (String.contains name '.') then StringSet.add name acc else acc)
        StringSet.empty
        kernel.tkern_type_decls
    else StringSet.empty
  in
  let current_module = Some (module_name_of_sarek_loc kernel.tkern_loc) in
  let gen_mode = gen_mode_of_exec_strategy exec_strategy in

  let ctx =
    {
      empty_ctx with
      current_module;
      inline_types = inline_type_names;
      gen_mode;
      use_native_arg = true;
    }
  in
  let body_e = gen_expr_impl ~loc ~ctx kernel.tkern_body in

  (* Generate inline module items *)
  let inline_items =
    let all_items = kernel.tkern_module_items in
    let skip_count = kernel.tkern_external_item_count in
    let rec drop n lst =
      if n <= 0 then lst
      else match lst with [] -> [] | _ :: tl -> drop (n - 1) tl
    in
    drop skip_count all_items
  in
  let body_with_items = wrap_module_items ~loc inline_items body_e in

  (* Build parameter tuple pattern - no type constraints since vectors are
     accessor objects, not Vector.t *)
  let param_pats =
    List.map
      (fun p -> Ast_builder.Default.ppat_var ~loc {txt = p.tparam_name; loc})
      kernel.tkern_params
  in
  let params_pat =
    match param_pats with
    | [] -> [%pat? ()]
    | [p] -> p
    | ps -> Ast_builder.Default.ppat_tuple ~loc ps
  in

  let gid_x_pat = Ast_builder.Default.ppat_var ~loc {txt = simple_gid_x; loc} in
  let gid_y_pat = Ast_builder.Default.ppat_var ~loc {txt = simple_gid_y; loc} in
  let gid_z_pat = Ast_builder.Default.ppat_var ~loc {txt = simple_gid_z; loc} in

  let inner_fun =
    if use_fcm then
      let types_pat =
        Ast_builder.Default.ppat_var ~loc {txt = types_module_var; loc}
      in
      match exec_strategy with
      | Sarek_convergence.Simple1D ->
          [%expr
            fun [%p types_pat] ([%p gid_x_pat] : int32) [%p params_pat] ->
              [%e body_with_items]]
      | Sarek_convergence.Simple2D ->
          [%expr
            fun [%p types_pat]
                ([%p gid_x_pat] : int32)
                ([%p gid_y_pat] : int32)
                [%p params_pat] -> [%e body_with_items]]
      | Sarek_convergence.Simple3D ->
          [%expr
            fun [%p types_pat]
                ([%p gid_x_pat] : int32)
                ([%p gid_y_pat] : int32)
                ([%p gid_z_pat] : int32)
                [%p params_pat] -> [%e body_with_items]]
      | Sarek_convergence.FullState ->
          failwith "gen_simple_cpu_kern_native called with FullState strategy"
    else
      match exec_strategy with
      | Sarek_convergence.Simple1D ->
          [%expr
            fun ([%p gid_x_pat] : int32) [%p params_pat] -> [%e body_with_items]]
      | Sarek_convergence.Simple2D ->
          [%expr
            fun ([%p gid_x_pat] : int32)
                ([%p gid_y_pat] : int32)
                [%p params_pat] -> [%e body_with_items]]
      | Sarek_convergence.Simple3D ->
          [%expr
            fun ([%p gid_x_pat] : int32)
                ([%p gid_y_pat] : int32)
                ([%p gid_z_pat] : int32)
                [%p params_pat] -> [%e body_with_items]]
      | Sarek_convergence.FullState ->
          failwith "gen_simple_cpu_kern_native called with FullState strategy"
  in
  (* Add warning suppression attribute *)
  {
    inner_fun with
    pexp_attributes =
      [
        Ast_builder.Default.attribute
          ~loc
          ~name:{txt = "warning"; loc}
          ~payload:
            (PStr
               [
                 Ast_builder.Default.pstr_eval
                   ~loc
                   (Ast_builder.Default.estring ~loc "-27-32-33")
                   [];
               ]);
      ];
  }

(** Generate the cpu_kern wrapper for use with native_fn_t.

    Generated function type: parallel:bool -> block:int*int*int ->
    grid:int*int*int -> native_arg array -> unit

    Uses typed native_arg accessors for type-safe vector access. Vectors are
    wrapped in accessor objects with get/set/length/underlying. *)
let gen_cpu_kern_native_wrapper ~loc (kernel : tkernel) : expression =
  let use_fcm = has_inline_types kernel in
  let native_kern = gen_cpu_kern_native ~loc kernel in

  (* Detect execution strategy for optimization *)
  let exec_strategy = Sarek_convergence.kernel_exec_strategy kernel in

  (* Detect barrier usage at compile time - passed to runtime *)
  let has_barriers = Sarek_convergence.kernel_uses_barriers kernel in
  let has_barriers_expr = Ast_builder.Default.ebool ~loc has_barriers in

  (* Generate argument extraction bindings - use parameter names *)
  let arg_bindings =
    List.mapi
      (fun i param ->
        let var_pat =
          Ast_builder.Default.ppat_var ~loc {txt = param.tparam_name; loc}
        in
        let cast_expr = gen_arg_cast ~loc param i in
        (var_pat, cast_expr))
      kernel.tkern_params
  in

  (* Build the args tuple expression - use parameter names *)
  let args_tuple =
    let arg_exprs =
      List.map (fun p -> evar ~loc p.tparam_name) kernel.tkern_params
    in
    match arg_exprs with
    | [] -> [%expr ()]
    | [e] -> e
    | es -> Ast_builder.Default.pexp_tuple ~loc es
  in

  (* Generate the simple kernel expression if needed *)
  let simple_kern_opt =
    match exec_strategy with
    | Sarek_convergence.Simple1D | Sarek_convergence.Simple2D
    | Sarek_convergence.Simple3D ->
        Some (gen_simple_cpu_kern_native ~loc ~exec_strategy kernel)
    | Sarek_convergence.FullState -> None
  in

  (* For simple kernels, generate optimized threadpool call *)
  let gen_simple_threadpool_call () =
    match exec_strategy with
    | Sarek_convergence.Simple1D ->
        if use_fcm then
          [%expr
            let bx, _, _ = block in
            let gx, _, _ = grid in
            Sarek.Sarek_cpu_runtime.run_1d_threadpool
              ~total_x:(bx * gx)
              (fun gid_x args -> __simple_kern __types_rec gid_x args)
              [%e args_tuple]]
        else
          [%expr
            let bx, _, _ = block in
            let gx, _, _ = grid in
            Sarek.Sarek_cpu_runtime.run_1d_threadpool
              ~total_x:(bx * gx)
              __simple_kern
              [%e args_tuple]]
    | Sarek_convergence.Simple2D ->
        if use_fcm then
          [%expr
            let bx, by, _ = block in
            let gx, gy, _ = grid in
            Sarek.Sarek_cpu_runtime.run_2d_threadpool
              ~width:(bx * gx)
              ~height:(by * gy)
              (fun gid_x gid_y args ->
                __simple_kern __types_rec gid_x gid_y args)
              [%e args_tuple]]
        else
          [%expr
            let bx, by, _ = block in
            let gx, gy, _ = grid in
            Sarek.Sarek_cpu_runtime.run_2d_threadpool
              ~width:(bx * gx)
              ~height:(by * gy)
              __simple_kern
              [%e args_tuple]]
    | Sarek_convergence.Simple3D ->
        if use_fcm then
          [%expr
            let bx, by, bz = block in
            let gx, gy, gz = grid in
            Sarek.Sarek_cpu_runtime.run_3d_threadpool
              ~width:(bx * gx)
              ~height:(by * gy)
              ~depth:(bz * gz)
              (fun gid_x gid_y gid_z args ->
                __simple_kern __types_rec gid_x gid_y gid_z args)
              [%e args_tuple]]
        else
          [%expr
            let bx, by, bz = block in
            let gx, gy, gz = grid in
            Sarek.Sarek_cpu_runtime.run_3d_threadpool
              ~width:(bx * gx)
              ~height:(by * gy)
              ~depth:(bz * gz)
              __simple_kern
              [%e args_tuple]]
    | Sarek_convergence.FullState ->
        failwith "gen_simple_threadpool_call called with FullState"
  in

  (* V2 uses parallel:bool - map to Sequential/Parallel/Threadpool:
     parallel=false -> Sequential
     parallel=true -> Threadpool for simple kernels, Parallel for complex *)
  let run_call =
    match exec_strategy with
    | Sarek_convergence.Simple1D | Sarek_convergence.Simple2D
    | Sarek_convergence.Simple3D ->
        (* Simple kernel - use optimized threadpool path when parallel *)
        let simple_call = gen_simple_threadpool_call () in
        if use_fcm then
          [%expr
            if __parallel then [%e simple_call]
            else
              Sarek.Sarek_cpu_runtime.run_sequential
                ~block
                ~grid
                (__native_kern __types_rec)
                [%e args_tuple]]
        else
          [%expr
            if __parallel then [%e simple_call]
            else
              Sarek.Sarek_cpu_runtime.run_sequential
                ~block
                ~grid
                __native_kern
                [%e args_tuple]]
    | Sarek_convergence.FullState ->
        (* Complex kernel - use threadpool with barrier support when parallel *)
        if use_fcm then
          [%expr
            if __parallel then
              Sarek.Sarek_cpu_runtime.run_threadpool
                ~has_barriers:[%e has_barriers_expr]
                ~block
                ~grid
                (__native_kern __types_rec)
                [%e args_tuple]
            else
              Sarek.Sarek_cpu_runtime.run_sequential
                ~block
                ~grid
                (__native_kern __types_rec)
                [%e args_tuple]]
        else
          [%expr
            if __parallel then
              Sarek.Sarek_cpu_runtime.run_threadpool
                ~has_barriers:[%e has_barriers_expr]
                ~block
                ~grid
                __native_kern
                [%e args_tuple]
            else
              Sarek.Sarek_cpu_runtime.run_sequential
                ~block
                ~grid
                __native_kern
                [%e args_tuple]]
  in

  (* Build the nested let bindings *)
  let body_with_bindings =
    List.fold_right
      (fun (pat, expr) body ->
        [%expr
          let [%p pat] = [%e expr] in
          [%e body]])
      arg_bindings
      run_call
  in

  if use_fcm then
    (* For FCM kernels, create the types object inside a local module *)
    let inline_decls = inline_type_decls kernel.tkern_type_decls in
    let types_object = gen_types_object ~loc inline_decls in
    let types_impl = gen_module_impl ~loc inline_decls in
    let inner_body =
      Ast_builder.Default.pexp_open
        ~loc
        (Ast_builder.Default.open_infos
           ~loc
           ~override:Fresh
           ~expr:
             (Ast_builder.Default.pmod_ident ~loc {txt = Lident "__Types"; loc}))
        (match simple_kern_opt with
        | Some simple_kern ->
            [%expr
              let __types_rec = [%e types_object] in
              let __native_kern = [%e native_kern] in
              let __simple_kern = [%e simple_kern] in
              [%e body_with_bindings]]
        | None ->
            [%expr
              let __types_rec = [%e types_object] in
              let __native_kern = [%e native_kern] in
              [%e body_with_bindings]])
    in
    let with_module =
      Ast_builder.Default.pexp_letmodule
        ~loc
        {txt = Some "__Types"; loc}
        types_impl
        inner_body
    in
    [%expr
      fun ~parallel:(__parallel : bool) ~block ~grid __args -> [%e with_module]]
  else
    match simple_kern_opt with
    | Some simple_kern ->
        [%expr
          fun ~parallel:(__parallel : bool) ~block ~grid __args ->
            let __native_kern = [%e native_kern] in
            let __simple_kern = [%e simple_kern] in
            [%e body_with_bindings]]
    | None ->
        [%expr
          fun ~parallel:(__parallel : bool) ~block ~grid __args ->
            let __native_kern = [%e native_kern] in
            [%e body_with_bindings]]
