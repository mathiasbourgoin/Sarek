(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** PTX statement emitter: emit_stmt and emit_assign.

    Translates Sarek IR statements to PTX instruction sequences. All emitters
    mutate [buf] and [alloc] as side effects. *)

open Sarek_ir_types
open Sarek_ir_ptx_types
open Sarek_ir_ptx_mem
open Sarek_ir_ptx_expr

(** {1 Statement emitter} *)

let rec emit_stmt buf alloc (env : env) (stmt : stmt) : unit =
  match stmt with
  | SEmpty -> ()
  | SSeq stmts -> List.iter (emit_stmt buf alloc env) stmts
  | SLet (v, EArrayCreate (elt, size_e, Shared), body) ->
      (* let%shared lowers to this shape (the PPX never emits DShared decls).
         Declare the array in .shared space and bind its base address. *)
      let n =
        match size_e with
        | EConst (CInt32 n) when Int32.compare n 0l > 0 -> Int32.to_int n
        | EConst (CInt32 _) ->
            fail
              (Printf.sprintf
                 "PTX codegen: shared array '%s': size must be positive"
                 v.var_name)
        | _ ->
            unsupported
              (Printf.sprintf
                 "shared array '%s' with non-literal size"
                 v.var_name)
      in
      if Hashtbl.mem alloc.arr_memspaces v.var_name then
        fail
          (Printf.sprintf
             "PTX codegen: duplicate shared array name '%s'"
             v.var_name) ;
      Buffer.add_string
        alloc.shared_decls
        (Printf.sprintf
           "    .shared .align %d .%s %s[%d];\n"
           (ptx_align_of_elttype elt)
           (ptx_btype_of_elttype elt)
           v.var_name
           n) ;
      let r = new_u32 alloc in
      env_bind env v.var_name r ;
      emit buf "mov.u32 %s, %s;" r v.var_name ;
      Hashtbl.replace alloc.arr_memspaces v.var_name () ;
      Hashtbl.replace alloc.arr_elt_types v.var_name elt ;
      emit_stmt buf alloc env body
  | SLet (v, EArrayCreate (_, _, ms), _) ->
      unsupported
        (Printf.sprintf
           "%s array creation for '%s' (only Shared is supported)"
           (match ms with
           | Sarek_ir_types.Local -> "Local"
           | Sarek_ir_types.Global -> "Global"
           | Sarek_ir_types.Shared -> assert false)
           v.var_name)
  | SLet (v, e, body) ->
      (* emit_value: scalar initializers behave exactly as before (Scalar of
         emit_expr's register); record/variant initializers bind their SROA
         register set. *)
      let b = emit_value buf alloc env e in
      env_bind_binding env v.var_name b ;
      emit_stmt buf alloc env body
  | SLetMut (v, e, body) ->
      (* Mutable binding: copy into fresh registers (leaf-wise for
         aggregates). Binding the initializer registers directly would alias
         them — `let mutable acc = y` followed by `acc <- …` would silently
         clobber y. *)
      let b_init = emit_value buf alloc env e in
      env_bind_binding env v.var_name (copy_binding buf alloc b_init) ;
      emit_stmt buf alloc env body
  | SAssign (lv, e) -> emit_assign buf alloc env lv e
  | SIf (cond, then_s, else_opt) -> (
      let r_cond = emit_expr buf alloc env cond in
      let p = new_pred alloc in
      emit buf "setp.ne.u32 %s, %s, 0;" p r_cond ;
      match else_opt with
      | None ->
          let l_skip = new_label alloc in
          emit buf "@!%s bra %s;" p l_skip ;
          emit_stmt buf alloc env then_s ;
          emit_label buf l_skip
      | Some else_s ->
          let l_else = new_label alloc in
          let l_merge = new_label alloc in
          emit buf "@!%s bra %s;" p l_else ;
          emit_stmt buf alloc env then_s ;
          emit buf "bra %s;" l_merge ;
          emit_label buf l_else ;
          emit_stmt buf alloc env else_s ;
          emit_label buf l_merge)
  | SFor (v, start_e, stop_e, dir, body) ->
      (* OCaml 'for i = a to b' is inclusive.
         Loop structure: init; header: bounds-check; body; incr; bra header *)
      let r_start = emit_expr buf alloc env start_e in
      let r_stop = emit_expr buf alloc env stop_e in
      let r_loop = new_u32 alloc in
      emit buf "mov.u32 %s, %s;" r_loop r_start ;
      env_bind env v.var_name r_loop ;
      let l_header = new_label alloc in
      let l_exit = new_label alloc in
      emit_label buf l_header ;
      let p = new_pred alloc in
      (match dir with
      | Upto -> emit buf "setp.gt.s32 %s, %s, %s;" p r_loop r_stop
      | Downto -> emit buf "setp.lt.s32 %s, %s, %s;" p r_loop r_stop) ;
      emit buf "@%s bra %s;" p l_exit ;
      emit_stmt buf alloc env body ;
      (match dir with
      | Upto -> emit buf "add.u32 %s, %s, 1;" r_loop r_loop
      | Downto -> emit buf "sub.u32 %s, %s, 1;" r_loop r_loop) ;
      emit buf "bra %s;" l_header ;
      emit_label buf l_exit
  | SWhile (cond, body) ->
      let l_header = new_label alloc in
      let l_exit = new_label alloc in
      emit_label buf l_header ;
      let r_cond = emit_expr buf alloc env cond in
      let p = new_pred alloc in
      emit buf "setp.eq.u32 %s, %s, 0;" p r_cond ;
      emit buf "@%s bra %s;" p l_exit ;
      emit_stmt buf alloc env body ;
      emit buf "bra %s;" l_header ;
      emit_label buf l_exit
  | SBarrier -> emit buf "bar.sync 0;"
  | SWarpBarrier -> emit buf "bar.warp.sync 0xffffffff;"
  | SMemFence -> emit buf "membar.gl;"
  | SReturn e -> (
      match alloc.inline_ret with
      | (r_ret, l_end) :: _ ->
          (* Inside an inlined helper body: write the result register (if the
             helper returns a value) and branch to the inline end label
             instead of returning from the kernel. *)
          let r_val = emit_expr buf alloc env e in
          (match r_ret with
          | None -> ()
          | Some r_dst ->
              let mov_op =
                if
                  String.length r_dst >= 3 && r_dst.[1] = 'r' && r_dst.[2] = 'd'
                then "mov.u64"
                else if
                  String.length r_dst >= 3 && r_dst.[1] = 'f' && r_dst.[2] = 'd'
                then "mov.f64"
                else if String.length r_dst >= 2 && r_dst.[1] = 'f' then
                  "mov.f32"
                else "mov.u32"
              in
              emit buf "%s %s, %s;" mov_op r_dst r_val) ;
          emit buf "bra %s;" l_end
      | [] ->
          ignore (emit_expr buf alloc env e) ;
          emit buf "ret;")
  | SExpr e -> ignore (emit_expr buf alloc env e)
  | SBlock inner -> emit_stmt buf alloc env inner
  | SPragma (_hints, body) ->
      (* PTX has no pragma equivalent; skip the hint and emit the body. *)
      emit_stmt buf alloc env body
  | SMatch _ -> unsupported "SMatch (requires variant lowering)"
  | SNative {gpu; _} ->
      (* Pass-through: caller must supply valid PTX as the gpu closure. *)
      let code = gpu ~framework:"PTX" in
      Buffer.add_string buf code ;
      if String.length code > 0 && code.[String.length code - 1] <> '\n' then
        Buffer.add_char buf '\n'

and emit_assign buf alloc (env : env) (lv : lvalue) (e : expr) : unit =
  match lv with
  | LVar v -> (
      match env_lookup_binding env v.var_name with
      | Scalar r_dst ->
          let r_val = emit_expr buf alloc env e in
          mov_scalar buf ~dst:r_dst ~src:r_val
      | Agg _ as dst ->
          let src = emit_value buf alloc env e in
          mov_binding buf ~src ~dst)
  | LArrayElem (arr_name, idx_expr) ->
      let r_base = env_lookup env arr_name in
      let r_val = emit_expr buf alloc env e in
      let r_idx = emit_expr buf alloc env idx_expr in
      emit_array_write
        buf
        alloc
        r_base
        r_idx
        r_val
        (infer_elt_type alloc arr_name)
        ~is_shared:(Hashtbl.mem alloc.arr_memspaces arr_name)
  | LArrayElemExpr (base_expr, idx_expr) ->
      let r_base = emit_expr buf alloc env base_expr in
      let r_val = emit_expr buf alloc env e in
      let r_idx = emit_expr buf alloc env idx_expr in
      let arr_name_opt =
        match base_expr with EVar v -> Some v.var_name | _ -> None
      in
      let elt_type =
        match arr_name_opt with
        | Some n -> infer_elt_type alloc n
        | None ->
            fail
              "LArrayElemExpr: cannot infer element type from non-variable \
               base expression"
      in
      let is_shared =
        match arr_name_opt with
        | Some n -> Hashtbl.mem alloc.arr_memspaces n
        | None -> false
      in
      emit_array_write buf alloc r_base r_idx r_val elt_type ~is_shared
  | LRecordField (root, field) -> (
      match resolve_local_field env root field with
      | Scalar r_dst ->
          let r_val = emit_expr buf alloc env e in
          mov_scalar buf ~dst:r_dst ~src:r_val
      | Agg _ as dst ->
          let src = emit_value buf alloc env e in
          mov_binding buf ~src ~dst)

(** Resolve the binding of [root.field] for a LOCAL record lvalue (root chain of
    LVar / nested LRecordField only). Assignments into fields of vector ELEMENTS
    (v.(i).field <- …) are a global-memory feature not yet supported at this
    stage. *)
and resolve_local_field (env : env) (root : lvalue) (field : string) : binding =
  let root_binding =
    match root with
    | LVar v -> env_lookup_binding env v.var_name
    | LRecordField (inner_root, inner_field) ->
        resolve_local_field env inner_root inner_field
    | LArrayElem _ | LArrayElemExpr _ ->
        unsupported
          (Printf.sprintf
             "LRecordField assignment on a vector/array element (v.(i).%s <- \
              …) — not yet supported at this stage; compute the record in a \
              local mutable binding and store its scalar fields individually"
             field)
  in
  match root_binding with
  | Agg (ARecord fields) -> (
      match List.assoc_opt field fields with
      | Some b -> b
      | None ->
          fail
            (Printf.sprintf
               "PTX codegen: record has no field '%s' (available: %s)"
               field
               (String.concat ", " (List.map fst fields))))
  | Agg (AVariant _) ->
      fail
        ("PTX codegen: field assignment '." ^ field
       ^ "' on a variant value; variants are immutable — rebuild the value \
          with its constructor instead")
  | Scalar _ ->
      fail
        ("PTX codegen: field assignment '." ^ field
       ^ "' on a non-record variable")

(* Install the statement emitter for EApp inlining (see stmt_emitter in
   Sarek_ir_ptx_types). *)
let () = Sarek_ir_ptx_types.stmt_emitter := emit_stmt
