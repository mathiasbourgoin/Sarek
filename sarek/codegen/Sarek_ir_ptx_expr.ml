(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** PTX expression emitter: emit_expr, emit_binop, emit_cast, emit_intrinsic.

    All functions emit PTX instructions into a {!Buffer.t} as a side-effect and
    return the PTX register name holding the result. *)

open Sarek_ir_types
open Sarek_ir_ptx_types
open Sarek_ir_ptx_mem

(** {1 Expression emitter}

    Returns the PTX register name holding the result. Emits instructions into
    [buf] as a side effect. *)
let rec emit_expr buf alloc (env : env) (expr : expr) : string =
  match expr with
  | EConst (CInt32 n) ->
      let r = new_u32 alloc in
      emit buf "mov.u32 %s, %ld;" r n ;
      r
  | EConst (CInt64 n) ->
      let r = new_u64 alloc in
      emit buf "mov.u64 %s, %Ld;" r n ;
      r
  | EConst (CFloat32 f) ->
      let r = new_f32 alloc in
      emit buf "mov.f32 %s, 0F%08lX;" r (Int32.bits_of_float f) ;
      r
  | EConst (CFloat64 f) ->
      let r = new_f64 alloc in
      emit buf "mov.f64 %s, 0D%016LX;" r (Int64.bits_of_float f) ;
      r
  | EConst (CBool true) ->
      let r = new_u32 alloc in
      emit buf "mov.u32 %s, 1;" r ;
      r
  | EConst (CBool false) ->
      let r = new_u32 alloc in
      emit buf "mov.u32 %s, 0;" r ;
      r
  | EConst CUnit ->
      let r = new_u32 alloc in
      emit buf "mov.u32 %s, 0;" r ;
      r
  | EVar v -> env_lookup env v.var_name
  | EBinop (op, e1, e2) -> emit_binop buf alloc env op e1 e2
  | EUnop (Neg, e) ->
      let r_src = emit_expr buf alloc env e in
      let is_f64 r = String.length r >= 3 && r.[1] = 'f' && r.[2] = 'd' in
      let is_f32 r = String.length r >= 2 && r.[1] = 'f' && not (is_f64 r) in
      let is_u64 r = String.length r >= 3 && r.[1] = 'r' && r.[2] = 'd' in
      if is_f64 r_src then (
        let r = new_f64 alloc in
        emit buf "neg.f64 %s, %s;" r r_src ;
        r)
      else if is_f32 r_src then (
        let r = new_f32 alloc in
        emit buf "neg.f32 %s, %s;" r r_src ;
        r)
      else if is_u64 r_src then (
        let r = new_u64 alloc in
        emit buf "neg.s64 %s, %s;" r r_src ;
        r)
      else
        let r = new_u32 alloc in
        emit buf "neg.s32 %s, %s;" r r_src ;
        r
  | EUnop (Not, e) ->
      let r_src = emit_expr buf alloc env e in
      let p = new_pred alloc in
      emit buf "setp.eq.u32 %s, %s, 0;" p r_src ;
      let r = new_u32 alloc in
      emit buf "selp.u32 %s, 1, 0, %s;" r p ;
      r
  | EUnop (BitNot, e) ->
      let r_src = emit_expr buf alloc env e in
      let r = new_u32 alloc in
      emit buf "not.b32 %s, %s;" r r_src ;
      r
  | EArrayRead (arr_name, idx_expr) ->
      let r_base = env_lookup env arr_name in
      let r_idx = emit_expr buf alloc env idx_expr in
      emit_array_read buf alloc r_base r_idx (infer_elt_type alloc arr_name)
  | EArrayReadExpr (base_expr, idx_expr) ->
      let r_base = emit_expr buf alloc env base_expr in
      let r_idx = emit_expr buf alloc env idx_expr in
      let elt_type =
        match base_expr with
        | EVar v -> infer_elt_type alloc v.var_name
        | _ -> TFloat32
      in
      emit_array_read buf alloc r_base r_idx elt_type
  | EIntrinsic (path, name, args) -> emit_intrinsic buf alloc env path name args
  | ECast (ty, e) ->
      let r_src = emit_expr buf alloc env e in
      emit_cast buf alloc r_src ty
  | EIf (cond, then_e, else_e) ->
      let r_cond = emit_expr buf alloc env cond in
      let r_then = emit_expr buf alloc env then_e in
      let r_else = emit_expr buf alloc env else_e in
      let p = new_pred alloc in
      emit buf "setp.ne.u32 %s, %s, 0;" p r_cond ;
      let is_f64 r = String.length r >= 3 && r.[1] = 'f' && r.[2] = 'd' in
      let is_f32 r = String.length r >= 2 && r.[1] = 'f' && not (is_f64 r) in
      let is_u64 r = String.length r >= 3 && r.[1] = 'r' && r.[2] = 'd' in
      if is_f64 r_then then (
        let r = new_f64 alloc in
        emit buf "selp.f64 %s, %s, %s, %s;" r r_then r_else p ;
        r)
      else if is_f32 r_then then (
        let r = new_f32 alloc in
        emit buf "selp.f32 %s, %s, %s, %s;" r r_then r_else p ;
        r)
      else if is_u64 r_then then (
        let r = new_u64 alloc in
        emit buf "selp.u64 %s, %s, %s, %s;" r r_then r_else p ;
        r)
      else
        let r = new_u32 alloc in
        emit buf "selp.u32 %s, %s, %s, %s;" r r_then r_else p ;
        r
  | EArrayLen _ ->
      unsupported "EArrayLen (needs (ptr,len) pair tracking in env)"
  | EArrayCreate _ ->
      unsupported "EArrayCreate in expression position (use SLet)"
  | EMatch _ -> unsupported "EMatch (requires variant lowering)"
  | ERecord _ -> unsupported "ERecord (requires struct layout)"
  | ERecordField _ -> unsupported "ERecordField (requires struct layout)"
  | ETuple _ -> unsupported "ETuple (no PTX equivalent)"
  | EApp _ ->
      unsupported "EApp (device function calls via .func not yet implemented)"
  | EVariant _ -> unsupported "EVariant (requires tagged-union lowering)"

and emit_binop buf alloc env op e1 e2 : string =
  let r1 = emit_expr buf alloc env e1 in
  let r2 = emit_expr buf alloc env e2 in
  (* Infer type from first operand register name prefix.
     %r* -> u32, %rd* -> u64, %f* -> f32, %fd* -> f64 *)
  let is_f64 r = String.length r >= 3 && r.[1] = 'f' && r.[2] = 'd' in
  let is_f32 r = String.length r >= 2 && r.[1] = 'f' && not (is_f64 r) in
  let is_u64 r = String.length r >= 3 && r.[1] = 'r' && r.[2] = 'd' in
  match op with
  | Add ->
      if is_f64 r1 then (
        let r = new_f64 alloc in
        emit buf "add.f64 %s, %s, %s;" r r1 r2 ;
        r)
      else if is_f32 r1 then (
        let r = new_f32 alloc in
        emit buf "add.f32 %s, %s, %s;" r r1 r2 ;
        r)
      else if is_u64 r1 then (
        let r = new_u64 alloc in
        emit buf "add.u64 %s, %s, %s;" r r1 r2 ;
        r)
      else
        let r = new_u32 alloc in
        emit buf "add.u32 %s, %s, %s;" r r1 r2 ;
        r
  | Sub ->
      if is_f64 r1 then (
        let r = new_f64 alloc in
        emit buf "sub.f64 %s, %s, %s;" r r1 r2 ;
        r)
      else if is_f32 r1 then (
        let r = new_f32 alloc in
        emit buf "sub.f32 %s, %s, %s;" r r1 r2 ;
        r)
      else if is_u64 r1 then (
        let r = new_u64 alloc in
        emit buf "sub.u64 %s, %s, %s;" r r1 r2 ;
        r)
      else
        let r = new_u32 alloc in
        emit buf "sub.u32 %s, %s, %s;" r r1 r2 ;
        r
  | Mul ->
      if is_f64 r1 then (
        let r = new_f64 alloc in
        emit buf "mul.f64 %s, %s, %s;" r r1 r2 ;
        r)
      else if is_f32 r1 then (
        let r = new_f32 alloc in
        emit buf "mul.f32 %s, %s, %s;" r r1 r2 ;
        r)
      else if is_u64 r1 then (
        let r = new_u64 alloc in
        emit buf "mul.lo.u64 %s, %s, %s;" r r1 r2 ;
        r)
      else
        let r = new_u32 alloc in
        emit buf "mul.lo.u32 %s, %s, %s;" r r1 r2 ;
        r
  | Div ->
      if is_f64 r1 then (
        let r = new_f64 alloc in
        emit buf "div.rn.f64 %s, %s, %s;" r r1 r2 ;
        r)
      else if is_f32 r1 then (
        let r = new_f32 alloc in
        emit buf "div.approx.f32 %s, %s, %s;" r r1 r2 ;
        r)
      else if is_u64 r1 then (
        let r = new_u64 alloc in
        emit buf "div.u64 %s, %s, %s;" r r1 r2 ;
        r)
      else
        let r = new_u32 alloc in
        emit buf "div.u32 %s, %s, %s;" r r1 r2 ;
        r
  | Mod ->
      if is_f32 r1 || is_f64 r1 then unsupported "Mod on float"
      else if is_u64 r1 then (
        let r = new_u64 alloc in
        emit buf "rem.u64 %s, %s, %s;" r r1 r2 ;
        r)
      else
        let r = new_u32 alloc in
        emit buf "rem.u32 %s, %s, %s;" r r1 r2 ;
        r
  | Eq ->
      let p = new_pred alloc in
      if is_f64 r1 then emit buf "setp.eq.f64 %s, %s, %s;" p r1 r2
      else if is_f32 r1 then emit buf "setp.eq.f32 %s, %s, %s;" p r1 r2
      else emit buf "setp.eq.u32 %s, %s, %s;" p r1 r2 ;
      let r = new_u32 alloc in
      emit buf "selp.u32 %s, 1, 0, %s;" r p ;
      r
  | Ne ->
      let p = new_pred alloc in
      if is_f64 r1 then emit buf "setp.ne.f64 %s, %s, %s;" p r1 r2
      else if is_f32 r1 then emit buf "setp.ne.f32 %s, %s, %s;" p r1 r2
      else emit buf "setp.ne.u32 %s, %s, %s;" p r1 r2 ;
      let r = new_u32 alloc in
      emit buf "selp.u32 %s, 1, 0, %s;" r p ;
      r
  | Lt ->
      let p = new_pred alloc in
      if is_f64 r1 then emit buf "setp.lt.f64 %s, %s, %s;" p r1 r2
      else if is_f32 r1 then emit buf "setp.lt.f32 %s, %s, %s;" p r1 r2
      else emit buf "setp.lt.s32 %s, %s, %s;" p r1 r2 ;
      let r = new_u32 alloc in
      emit buf "selp.u32 %s, 1, 0, %s;" r p ;
      r
  | Le ->
      let p = new_pred alloc in
      if is_f64 r1 then emit buf "setp.le.f64 %s, %s, %s;" p r1 r2
      else if is_f32 r1 then emit buf "setp.le.f32 %s, %s, %s;" p r1 r2
      else emit buf "setp.le.s32 %s, %s, %s;" p r1 r2 ;
      let r = new_u32 alloc in
      emit buf "selp.u32 %s, 1, 0, %s;" r p ;
      r
  | Gt ->
      let p = new_pred alloc in
      if is_f64 r1 then emit buf "setp.gt.f64 %s, %s, %s;" p r1 r2
      else if is_f32 r1 then emit buf "setp.gt.f32 %s, %s, %s;" p r1 r2
      else emit buf "setp.gt.s32 %s, %s, %s;" p r1 r2 ;
      let r = new_u32 alloc in
      emit buf "selp.u32 %s, 1, 0, %s;" r p ;
      r
  | Ge ->
      let p = new_pred alloc in
      if is_f64 r1 then emit buf "setp.ge.f64 %s, %s, %s;" p r1 r2
      else if is_f32 r1 then emit buf "setp.ge.f32 %s, %s, %s;" p r1 r2
      else emit buf "setp.ge.s32 %s, %s, %s;" p r1 r2 ;
      let r = new_u32 alloc in
      emit buf "selp.u32 %s, 1, 0, %s;" r p ;
      r
  | And ->
      let r = new_u32 alloc in
      emit buf "and.b32 %s, %s, %s;" r r1 r2 ;
      r
  | Or ->
      let r = new_u32 alloc in
      emit buf "or.b32 %s, %s, %s;" r r1 r2 ;
      r
  | Shl ->
      let r = new_u32 alloc in
      emit buf "shl.b32 %s, %s, %s;" r r1 r2 ;
      r
  | Shr ->
      let r = new_u32 alloc in
      emit buf "shr.u32 %s, %s, %s;" r r1 r2 ;
      r
  | BitAnd ->
      let r = new_u32 alloc in
      emit buf "and.b32 %s, %s, %s;" r r1 r2 ;
      r
  | BitOr ->
      let r = new_u32 alloc in
      emit buf "or.b32 %s, %s, %s;" r r1 r2 ;
      r
  | BitXor ->
      let r = new_u32 alloc in
      emit buf "xor.b32 %s, %s, %s;" r r1 r2 ;
      r

and emit_cast buf alloc r_src dst_ty : string =
  let is_f64 r = String.length r >= 3 && r.[1] = 'f' && r.[2] = 'd' in
  let is_f32 r = String.length r >= 2 && r.[1] = 'f' && not (is_f64 r) in
  let is_u64 r = String.length r >= 3 && r.[1] = 'r' && r.[2] = 'd' in
  match dst_ty with
  | TFloat32 ->
      if is_f32 r_src then r_src
      else
        let r = new_f32 alloc in
        let cvt =
          if is_f64 r_src then "cvt.rn.f32.f64"
          else if is_u64 r_src then "cvt.rn.f32.s64"
          else "cvt.rn.f32.s32"
        in
        emit buf "%s %s, %s;" cvt r r_src ;
        r
  | TFloat64 ->
      if is_f64 r_src then r_src
      else
        let r = new_f64 alloc in
        let cvt =
          if is_f32 r_src then "cvt.rn.f64.f32"
          else if is_u64 r_src then "cvt.rn.f64.s64"
          else "cvt.rn.f64.s32"
        in
        emit buf "%s %s, %s;" cvt r r_src ;
        r
  | TInt32 ->
      if (not (is_f32 r_src)) && (not (is_f64 r_src)) && not (is_u64 r_src) then
        r_src
      else
        let r = new_u32 alloc in
        let cvt =
          if is_f64 r_src then "cvt.rzi.s32.f64"
          else if is_f32 r_src then "cvt.rzi.s32.f32"
          else "cvt.u32.u64"
        in
        emit buf "%s %s, %s;" cvt r r_src ;
        r
  | TInt64 ->
      if is_u64 r_src then r_src
      else
        let r = new_u64 alloc in
        let cvt =
          if is_f64 r_src then "cvt.rzi.s64.f64"
          else if is_f32 r_src then "cvt.rzi.s64.f32"
          else "cvt.s64.s32"
        in
        emit buf "%s %s, %s;" cvt r r_src ;
        r
  | _ -> unsupported ("ECast to " ^ ptx_reg_type_of dst_ty)

and emit_intrinsic buf alloc env _path name args : string =
  match name with
  | "thread_id_x" | "thread_idx_x" ->
      let r = new_u32 alloc in
      emit buf "mov.u32 %s, %%tid.x;" r ;
      r
  | "thread_id_y" | "thread_idx_y" ->
      let r = new_u32 alloc in
      emit buf "mov.u32 %s, %%tid.y;" r ;
      r
  | "thread_id_z" | "thread_idx_z" ->
      let r = new_u32 alloc in
      emit buf "mov.u32 %s, %%tid.z;" r ;
      r
  | "block_id_x" | "block_idx_x" ->
      let r = new_u32 alloc in
      emit buf "mov.u32 %s, %%ctaid.x;" r ;
      r
  | "block_id_y" | "block_idx_y" ->
      let r = new_u32 alloc in
      emit buf "mov.u32 %s, %%ctaid.y;" r ;
      r
  | "block_id_z" | "block_idx_z" ->
      let r = new_u32 alloc in
      emit buf "mov.u32 %s, %%ctaid.z;" r ;
      r
  | "block_dim_x" ->
      let r = new_u32 alloc in
      emit buf "mov.u32 %s, %%ntid.x;" r ;
      r
  | "block_dim_y" ->
      let r = new_u32 alloc in
      emit buf "mov.u32 %s, %%ntid.y;" r ;
      r
  | "block_dim_z" ->
      let r = new_u32 alloc in
      emit buf "mov.u32 %s, %%ntid.z;" r ;
      r
  | "grid_dim_x" ->
      let r = new_u32 alloc in
      emit buf "mov.u32 %s, %%nctaid.x;" r ;
      r
  | "grid_dim_y" ->
      let r = new_u32 alloc in
      emit buf "mov.u32 %s, %%nctaid.y;" r ;
      r
  | "grid_dim_z" ->
      let r = new_u32 alloc in
      emit buf "mov.u32 %s, %%nctaid.z;" r ;
      r
  | "global_thread_id" | "global_idx" | "global_idx_x" ->
      let r_tid = new_u32 alloc in
      emit buf "mov.u32 %s, %%tid.x;" r_tid ;
      let r_bid = new_u32 alloc in
      emit buf "mov.u32 %s, %%ctaid.x;" r_bid ;
      let r_bdim = new_u32 alloc in
      emit buf "mov.u32 %s, %%ntid.x;" r_bdim ;
      let r_off = new_u32 alloc in
      emit buf "mul.lo.u32 %s, %s, %s;" r_off r_bid r_bdim ;
      let r_gid = new_u32 alloc in
      emit buf "add.u32 %s, %s, %s;" r_gid r_tid r_off ;
      r_gid
  | "global_idx_y" ->
      let r_tid = new_u32 alloc in
      emit buf "mov.u32 %s, %%tid.y;" r_tid ;
      let r_bid = new_u32 alloc in
      emit buf "mov.u32 %s, %%ctaid.y;" r_bid ;
      let r_bdim = new_u32 alloc in
      emit buf "mov.u32 %s, %%ntid.y;" r_bdim ;
      let r_off = new_u32 alloc in
      emit buf "mul.lo.u32 %s, %s, %s;" r_off r_bid r_bdim ;
      let r_gid = new_u32 alloc in
      emit buf "add.u32 %s, %s, %s;" r_gid r_tid r_off ;
      r_gid
  | "global_size" ->
      let r_bdim = new_u32 alloc in
      emit buf "mov.u32 %s, %%ntid.x;" r_bdim ;
      let r_gdim = new_u32 alloc in
      emit buf "mov.u32 %s, %%nctaid.x;" r_gdim ;
      let r = new_u32 alloc in
      emit buf "mul.lo.u32 %s, %s, %s;" r r_bdim r_gdim ;
      r
  | "block_barrier" ->
      emit buf "bar.sync 0;" ;
      let r = new_u32 alloc in
      emit buf "mov.u32 %s, 0;" r ;
      r
  | "sin" ->
      let r_arg =
        match args with
        | [a] -> emit_expr buf alloc env a
        | _ -> unsupported "sin arity != 1"
      in
      let r = new_f32 alloc in
      emit buf "sin.approx.f32 %s, %s;" r r_arg ;
      r
  | "cos" ->
      let r_arg =
        match args with
        | [a] -> emit_expr buf alloc env a
        | _ -> unsupported "cos arity != 1"
      in
      let r = new_f32 alloc in
      emit buf "cos.approx.f32 %s, %s;" r r_arg ;
      r
  | "sqrt" ->
      let r_arg =
        match args with
        | [a] -> emit_expr buf alloc env a
        | _ -> unsupported "sqrt arity != 1"
      in
      let r = new_f32 alloc in
      emit buf "sqrt.approx.f32 %s, %s;" r r_arg ;
      r
  | "fabs" ->
      let r_arg =
        match args with
        | [a] -> emit_expr buf alloc env a
        | _ -> unsupported "fabs arity != 1"
      in
      let r = new_f32 alloc in
      emit buf "abs.f32 %s, %s;" r r_arg ;
      r
  | "fma" -> (
      match args with
      | [a; b; c] ->
          let ra = emit_expr buf alloc env a in
          let rb = emit_expr buf alloc env b in
          let rc = emit_expr buf alloc env c in
          let r = new_f32 alloc in
          emit buf "fma.rn.f32 %s, %s, %s, %s;" r ra rb rc ;
          r
      | _ -> unsupported "fma arity != 3")
  | n -> unsupported ("intrinsic: " ^ n)
