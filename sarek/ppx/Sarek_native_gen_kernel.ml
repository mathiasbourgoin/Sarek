(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

open Ppxlib
open Sarek_typed_ast
open Sarek_types

(* Import helpers and intrinsics *)
open Sarek_native_helpers
open Sarek_native_intrinsics
open Sarek_native_gen_base
open Sarek_native_gen

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
let gen_arg_cast ?current_module ?inline_types
    ?(inline_ids_var = types_module_var) ~loc (param : tparam) (idx : int) :
    expression =
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
                      [%e
                        vector_type_id_expr
                          ?current_module
                          ?inline_types
                          ~inline_ids_var
                          ~loc
                          elem_ty]
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
                      [%e
                        vector_type_id_expr
                          ?current_module
                          ?inline_types
                          ~inline_ids_var
                          ~loc
                          elem_ty]
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
                      [%e
                        vector_type_id_expr
                          ?current_module
                          ?inline_types
                          ~inline_ids_var
                          ~loc
                          elem_ty]
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
                      [%e
                        vector_type_id_expr
                          ?current_module
                          ?inline_types
                          ~inline_ids_var
                          ~loc
                          elem_ty]
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
                  [%e
                    custom_type_id_expr
                      ?current_module
                      ?inline_types
                      ~inline_ids_var
                      ~loc
                      elem_ty]
                  [%e vec_arg]
                  i

              method set i x =
                Sarek_ir_types.vec_set_custom
                  [%e
                    custom_type_id_expr
                      ?current_module
                      ?inline_types
                      ~inline_ids_var
                      ~loc
                      elem_ty]
                  [%e vec_arg]
                  i
                  x

              method length = Sarek_ir_types.vec_length [%e vec_arg]

              method underlying =
                Sarek_ir_types.vec_as_vector
                  [%e
                    vector_type_id_expr
                      ?current_module
                      ?inline_types
                      ~inline_ids_var
                      ~loc
                      elem_ty]
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
  let id_bindings =
    List.concat_map
      (fun decl ->
        let type_name =
          match decl with
          | TTypeRecord {tdecl_name; _} -> tdecl_name
          | TTypeVariant {tdecl_name; _} -> tdecl_name
        in
        [
          ( inline_type_id_method type_name,
            "__" ^ inline_type_id_method type_name );
          ( inline_vector_type_id_method type_name,
            "__" ^ inline_vector_type_id_method type_name );
        ])
      decls
  in
  let id_methods =
    List.map
      (fun (method_name, binding_name) ->
        Ast_builder.Default.pcf_method
          ~loc
          ( {txt = method_name; loc},
            Public,
            Cfk_concrete (Fresh, evar ~loc binding_name) ))
      id_bindings
  in
  let methods =
    id_methods
    @ List.concat_map
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
                      ( {txt = fn_name; loc},
                        Public,
                        Cfk_concrete (Fresh, fn_expr) ))
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
  let object_expr =
    Ast_builder.Default.pexp_object
      ~loc
      (Ast_builder.Default.class_structure
         ~self:(Ast_builder.Default.ppat_any ~loc)
         ~fields:methods)
  in
  List.fold_right
    (fun (_method_name, binding_name) body ->
      [%expr
        let [%p Ast_builder.Default.pvar ~loc binding_name] =
          Sarek_ir_types.Type_id.create ()
        in
        [%e body]])
    id_bindings
    object_expr

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
  let native_kern = gen_cpu_kern_native ~loc kernel in

  (* Detect execution strategy for optimization *)
  let exec_strategy = Sarek_convergence.kernel_exec_strategy kernel in

  (* Detect barrier usage at compile time - passed to runtime *)
  let has_barriers = Sarek_convergence.kernel_uses_barriers kernel in
  let has_barriers_expr = Ast_builder.Default.ebool ~loc has_barriers in

  (* Generate argument extraction bindings - use parameter names *)
  let current_module = Some (module_name_of_sarek_loc kernel.tkern_loc) in
  let arg_bindings =
    List.mapi
      (fun i param ->
        let var_pat =
          Ast_builder.Default.ppat_var ~loc {txt = param.tparam_name; loc}
        in
        let cast_expr =
          gen_arg_cast
            ?current_module
            ~inline_types:inline_type_names
            ~inline_ids_var:"__types_rec"
            ~loc
            param
            i
        in
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
