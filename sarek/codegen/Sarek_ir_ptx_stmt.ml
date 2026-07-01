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
  | SLet (v, e, body) ->
      let r = emit_expr buf alloc env e in
      env_bind env v.var_name r ;
      emit_stmt buf alloc env body
  | SLetMut (v, e, body) ->
      let r = emit_expr buf alloc env e in
      env_bind env v.var_name r ;
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
  | SReturn e ->
      ignore (emit_expr buf alloc env e) ;
      emit buf "ret;"
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
  | LVar v ->
      let r_val = emit_expr buf alloc env e in
      let r_dst = env_lookup env v.var_name in
      let is_f64 r = String.length r >= 3 && r.[1] = 'f' && r.[2] = 'd' in
      let mov_op =
        if String.length r_dst >= 3 && r_dst.[1] = 'r' && r_dst.[2] = 'd' then
          "mov.u64"
        else if is_f64 r_dst then "mov.f64"
        else if String.length r_dst >= 2 && r_dst.[1] = 'f' then "mov.f32"
        else "mov.u32"
      in
      emit buf "%s %s, %s;" mov_op r_dst r_val
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
  | LRecordField _ ->
      unsupported "LRecordField assignment (requires struct layout)"
