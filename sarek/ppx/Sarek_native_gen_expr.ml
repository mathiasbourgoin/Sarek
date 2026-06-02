(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

open Ppxlib
open Sarek_typed_ast
open Sarek_types

(* Import helpers and intrinsics *)
open Sarek_native_helpers
open Sarek_native_gen_base

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
let gen_parallel_construct ?current_module ?inline_types ~loc ~gen_expr
    (te : texpr) : expression =
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
              Sarek.Sarek_cpu_runtime.alloc_shared_with_key
                [%e shared]
                Spoc_core.Vector.char_type_id
                [%e name_e]
                [%e size_e]
                '\000']
        | _ ->
            (* For custom types, generate proper default value *)
            let default_val = default_value_for_type ~loc elem_ty in
            let key_expr =
              match repr elem_ty with
              | TRecord (type_name, _) | TVariant (type_name, _) ->
                  if is_inline_type inline_types type_name then
                    inline_type_id_expr ~loc ~ids_var:types_module_var type_name
                  else
                    [%expr
                      [%e custom_descriptor_expr ?current_module ~loc type_name]
                        .Spoc_core.Vector.type_id]
              | _ -> [%expr failwith "unsupported shared memory type identity"]
            in
            [%expr
              Sarek.Sarek_cpu_runtime.alloc_shared_with_key
                [%e shared]
                [%e key_expr]
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
