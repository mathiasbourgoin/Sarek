(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Sarek_ir_ptx - PTX Code Generation from Sarek IR (SPIKE)
 *
 * Emits NVIDIA PTX virtual ISA directly from Sarek_ir_types.kernel.
 * This is a feasibility spike, not a production backend.
 *
 * Design constraints:
 * - SSA-style register allocation: one counter per PTX type
 * - No optimisation; every let-binding becomes a fresh register
 * - Covers the IR subset needed for vector_add and similar flat kernels
 * - Constructs that require design decisions are stubbed with explicit errors
 *
 * PTX subset targeted:
 *   .u32  - int32, thread IDs, array indices
 *   .u64  - pointer arithmetic (all global pointers are 64-bit)
 *   .f32  - float32
 *   .f64  - float64
 *   .pred - branch predicates
 *
 * Known gaps (documented in docs/plans/ptx-spike-findings.md):
 *   - Records / TRecord: layout would need a struct-to-offset mapping
 *   - Variants / TVariant: tagged-union lowering is non-trivial in PTX
 *   - EMatch / SMatch: depends on variant lowering
 *   - Helper functions / kern_funcs: .func directive; callable from kernel
 *   - EArrayLen: needs (ptr, len) pair tracking in env
 *   - EApp: device function calls via .func; not yet implemented
 ******************************************************************************)

open Sarek_ir_types

(** {1 Error handling} *)

exception Ptx_codegen_error of string

let fail msg = raise (Ptx_codegen_error msg)

let unsupported what = fail ("PTX codegen: unsupported construct: " ^ what)

(** {1 Register allocator} *)

(** Counter-based register allocator. Each PTX type has an independent counter
    so that register names stay readable (e.g. %r0, %f0, %rd0). *)
type reg_alloc = {
  mutable u32 : int;
  mutable u64 : int;
  mutable f32 : int;
  mutable f64 : int;
  mutable pred : int;
  mutable label : int;
}

let make_alloc () = {u32 = 0; u64 = 0; f32 = 0; f64 = 0; pred = 0; label = 0}

let new_u32 a =
  let n = a.u32 in
  a.u32 <- n + 1 ;
  Printf.sprintf "%%r%d" n

let new_u64 a =
  let n = a.u64 in
  a.u64 <- n + 1 ;
  Printf.sprintf "%%rd%d" n

let new_f32 a =
  let n = a.f32 in
  a.f32 <- n + 1 ;
  Printf.sprintf "%%f%d" n

let new_f64 a =
  let n = a.f64 in
  a.f64 <- n + 1 ;
  Printf.sprintf "%%fd%d" n

let new_pred a =
  let n = a.pred in
  a.pred <- n + 1 ;
  Printf.sprintf "%%p%d" n

let new_label a =
  let n = a.label in
  a.label <- n + 1 ;
  Printf.sprintf "L%d" n

(** {1 Type mapping} *)

let ptx_reg_type_of = function
  | TInt32 | TBool -> ".u32"
  | TInt64 -> ".u64"
  | TFloat32 -> ".f32"
  | TFloat64 -> ".f64"
  | TUnit -> ".u32"
  | TVec _ -> ".u64"
  | TArray _ -> ".u64"
  | TRecord _ -> unsupported "TRecord register type"
  | TVariant _ -> unsupported "TVariant register type"

let new_reg_for_type alloc = function
  | TInt32 | TBool | TUnit -> new_u32 alloc
  | TInt64 -> new_u64 alloc
  | TFloat32 -> new_f32 alloc
  | TFloat64 -> new_f64 alloc
  | TVec _ | TArray _ -> new_u64 alloc
  | TRecord _ -> unsupported "TRecord new_reg"
  | TVariant _ -> unsupported "TVariant new_reg"

(** {1 Environment: variable name -> PTX register name} *)

type env = (string, string) Hashtbl.t

let make_env () : env = Hashtbl.create 32

let env_bind (env : env) name reg = Hashtbl.replace env name reg

let env_lookup (env : env) name =
  match Hashtbl.find_opt env name with
  | Some r -> r
  | None -> fail ("PTX codegen: unbound variable: " ^ name)

(** {1 Emit helpers} *)

let emit buf fmt = Printf.bprintf buf ("    " ^^ fmt ^^ "\n")

let emit_label buf lbl = Printf.bprintf buf "%s:\n" lbl

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
      if String.length r_src >= 2 && r_src.[1] = 'f' then (
        let r = new_f32 alloc in
        emit buf "neg.f32 %s, %s;" r r_src ;
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
      (* Pointer + (idx * 4) address computation, then ld.global.f32.
         Element size is hardcoded to 4 bytes (float32/int32) for the spike.
         A full implementation would track element types in the env. *)
      let r_base = env_lookup env arr_name in
      let r_idx = emit_expr buf alloc env idx_expr in
      let r_idx64 = new_u64 alloc in
      emit buf "cvt.u64.u32 %s, %s;" r_idx64 r_idx ;
      let r_off = new_u64 alloc in
      emit buf "shl.b64 %s, %s, 2;" r_off r_idx64 ;
      let r_addr = new_u64 alloc in
      emit buf "add.u64 %s, %s, %s;" r_addr r_base r_off ;
      let r_val = new_f32 alloc in
      emit buf "ld.global.f32 %s, [%s];" r_val r_addr ;
      r_val
  | EArrayReadExpr (base_expr, idx_expr) ->
      let r_base = emit_expr buf alloc env base_expr in
      let r_idx = emit_expr buf alloc env idx_expr in
      let r_idx64 = new_u64 alloc in
      emit buf "cvt.u64.u32 %s, %s;" r_idx64 r_idx ;
      let r_off = new_u64 alloc in
      emit buf "shl.b64 %s, %s, 2;" r_off r_idx64 ;
      let r_addr = new_u64 alloc in
      emit buf "add.u64 %s, %s, %s;" r_addr r_base r_off ;
      let r_val = new_f32 alloc in
      emit buf "ld.global.f32 %s, [%s];" r_val r_addr ;
      r_val
  | EIntrinsic (path, name, args) -> emit_intrinsic buf alloc env path name args
  | ECast (ty, e) ->
      let r_src = emit_expr buf alloc env e in
      emit_cast buf alloc r_src ty
  | EIf (cond, then_e, else_e) ->
      (* Spike: branch-based value-if.  A full implementation would use selp
         for simple cases (no register liveness issues). *)
      let r_cond = emit_expr buf alloc env cond in
      let p = new_pred alloc in
      emit buf "setp.ne.u32 %s, %s, 0;" p r_cond ;
      let l_then = new_label alloc in
      let l_merge = new_label alloc in
      emit buf "@%s bra %s;" p l_then ;
      let _r_else = emit_expr buf alloc env else_e in
      emit buf "bra %s;" l_merge ;
      emit_label buf l_then ;
      let r_then = emit_expr buf alloc env then_e in
      emit_label buf l_merge ;
      r_then
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
  let is_float r = String.length r >= 2 && r.[1] = 'f' in
  let is_u64 r = String.length r >= 3 && r.[1] = 'r' && r.[2] = 'd' in
  match op with
  | Add ->
      if is_f64 r1 then (
        let r = new_f64 alloc in
        emit buf "add.f64 %s, %s, %s;" r r1 r2 ;
        r)
      else if is_float r1 then (
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
      else if is_float r1 then (
        let r = new_f32 alloc in
        emit buf "sub.f32 %s, %s, %s;" r r1 r2 ;
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
      else if is_float r1 then (
        let r = new_f32 alloc in
        emit buf "mul.f32 %s, %s, %s;" r r1 r2 ;
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
      else if is_float r1 then (
        let r = new_f32 alloc in
        emit buf "div.approx.f32 %s, %s, %s;" r r1 r2 ;
        r)
      else
        let r = new_u32 alloc in
        emit buf "div.u32 %s, %s, %s;" r r1 r2 ;
        r
  | Mod ->
      if is_float r1 then unsupported "Mod on float"
      else
        let r = new_u32 alloc in
        emit buf "rem.u32 %s, %s, %s;" r r1 r2 ;
        r
  | Eq ->
      let p = new_pred alloc in
      if is_float r1 then emit buf "setp.eq.f32 %s, %s, %s;" p r1 r2
      else emit buf "setp.eq.u32 %s, %s, %s;" p r1 r2 ;
      let r = new_u32 alloc in
      emit buf "selp.u32 %s, 1, 0, %s;" r p ;
      r
  | Ne ->
      let p = new_pred alloc in
      if is_float r1 then emit buf "setp.ne.f32 %s, %s, %s;" p r1 r2
      else emit buf "setp.ne.u32 %s, %s, %s;" p r1 r2 ;
      let r = new_u32 alloc in
      emit buf "selp.u32 %s, 1, 0, %s;" r p ;
      r
  | Lt ->
      let p = new_pred alloc in
      if is_float r1 then emit buf "setp.lt.f32 %s, %s, %s;" p r1 r2
      else emit buf "setp.lt.s32 %s, %s, %s;" p r1 r2 ;
      let r = new_u32 alloc in
      emit buf "selp.u32 %s, 1, 0, %s;" r p ;
      r
  | Le ->
      let p = new_pred alloc in
      if is_float r1 then emit buf "setp.le.f32 %s, %s, %s;" p r1 r2
      else emit buf "setp.le.s32 %s, %s, %s;" p r1 r2 ;
      let r = new_u32 alloc in
      emit buf "selp.u32 %s, 1, 0, %s;" r p ;
      r
  | Gt ->
      let p = new_pred alloc in
      if is_float r1 then emit buf "setp.gt.f32 %s, %s, %s;" p r1 r2
      else emit buf "setp.gt.s32 %s, %s, %s;" p r1 r2 ;
      let r = new_u32 alloc in
      emit buf "selp.u32 %s, 1, 0, %s;" r p ;
      r
  | Ge ->
      let p = new_pred alloc in
      if is_float r1 then emit buf "setp.ge.f32 %s, %s, %s;" p r1 r2
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
  match dst_ty with
  | TInt32 ->
      let r = new_u32 alloc in
      emit buf "cvt.s32.f32 %s, %s;" r r_src ;
      r
  | TFloat32 ->
      let r = new_f32 alloc in
      emit buf "cvt.rn.f32.s32 %s, %s;" r r_src ;
      r
  | TFloat64 ->
      let r = new_f64 alloc in
      emit buf "cvt.rn.f64.s32 %s, %s;" r r_src ;
      r
  | TInt64 ->
      let r = new_u64 alloc in
      emit buf "cvt.s64.s32 %s, %s;" r r_src ;
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
      (* Infer PTX type from register name prefix *)
      if String.length r_dst >= 3 && r_dst.[1] = 'r' && r_dst.[2] = 'd' then
        emit buf "mov.u64 %s, %s;" r_dst r_val
      else if String.length r_dst >= 2 && r_dst.[1] = 'f' then
        emit buf "mov.f32 %s, %s;" r_dst r_val
      else emit buf "mov.u32 %s, %s;" r_dst r_val
  | LArrayElem (arr_name, idx_expr) ->
      (* Compute byte address = base + idx * 4, then st.global.f32.
         Element size hardcoded to 4 for the spike; see EArrayRead note. *)
      let r_base = env_lookup env arr_name in
      let r_val = emit_expr buf alloc env e in
      let r_idx = emit_expr buf alloc env idx_expr in
      let r_idx64 = new_u64 alloc in
      emit buf "cvt.u64.u32 %s, %s;" r_idx64 r_idx ;
      let r_off = new_u64 alloc in
      emit buf "shl.b64 %s, %s, 2;" r_off r_idx64 ;
      let r_addr = new_u64 alloc in
      emit buf "add.u64 %s, %s, %s;" r_addr r_base r_off ;
      emit buf "st.global.f32 [%s], %s;" r_addr r_val
  | LArrayElemExpr (base_expr, idx_expr) ->
      let r_base = emit_expr buf alloc env base_expr in
      let r_val = emit_expr buf alloc env e in
      let r_idx = emit_expr buf alloc env idx_expr in
      let r_idx64 = new_u64 alloc in
      emit buf "cvt.u64.u32 %s, %s;" r_idx64 r_idx ;
      let r_off = new_u64 alloc in
      emit buf "shl.b64 %s, %s, 2;" r_off r_idx64 ;
      let r_addr = new_u64 alloc in
      emit buf "add.u64 %s, %s, %s;" r_addr r_base r_off ;
      emit buf "st.global.f32 [%s], %s;" r_addr r_val
  | LRecordField _ ->
      unsupported "LRecordField assignment (requires struct layout)"

(** {1 Parameter and local emitters} *)

(** Emit ld.param instructions for each kernel parameter, binding registers into
    [env]. Returns the formatted .param declaration block string to be embedded
    in the .entry header. *)
let emit_params buf alloc (env : env) (params : decl list) : string =
  let param_decls = Buffer.create 256 in
  let first = ref true in
  List.iter
    (fun decl ->
      match decl with
      | DParam (v, _arr_info) -> (
          if not !first then Buffer.add_string param_decls ",\n" ;
          first := false ;
          match v.var_type with
          | TVec _ | TArray _ ->
              Buffer.add_string
                param_decls
                (Printf.sprintf "    .param .u64 param_%s" v.var_name) ;
              let r = new_u64 alloc in
              env_bind env v.var_name r ;
              emit buf "ld.param.u64 %s, [param_%s];" r v.var_name
          | TInt32 | TBool ->
              Buffer.add_string
                param_decls
                (Printf.sprintf "    .param .u32 param_%s" v.var_name) ;
              let r = new_u32 alloc in
              env_bind env v.var_name r ;
              emit buf "ld.param.u32 %s, [param_%s];" r v.var_name
          | TInt64 ->
              Buffer.add_string
                param_decls
                (Printf.sprintf "    .param .u64 param_%s" v.var_name) ;
              let r = new_u64 alloc in
              env_bind env v.var_name r ;
              emit buf "ld.param.u64 %s, [param_%s];" r v.var_name
          | TFloat32 ->
              Buffer.add_string
                param_decls
                (Printf.sprintf "    .param .f32 param_%s" v.var_name) ;
              let r = new_f32 alloc in
              env_bind env v.var_name r ;
              emit buf "ld.param.f32 %s, [param_%s];" r v.var_name
          | TFloat64 ->
              Buffer.add_string
                param_decls
                (Printf.sprintf "    .param .f64 param_%s" v.var_name) ;
              let r = new_f64 alloc in
              env_bind env v.var_name r ;
              emit buf "ld.param.f64 %s, [param_%s];" r v.var_name
          | TUnit ->
              Buffer.add_string
                param_decls
                (Printf.sprintf "    .param .u32 param_%s" v.var_name) ;
              let r = new_u32 alloc in
              env_bind env v.var_name r ;
              emit buf "ld.param.u32 %s, [param_%s];" r v.var_name
          | TRecord _ | TVariant _ -> unsupported "DParam with custom type")
      | DLocal _ | DShared _ -> ())
    params ;
  Buffer.contents param_decls

let emit_locals buf alloc (env : env) (locals : decl list) : unit =
  List.iter
    (fun decl ->
      match decl with
      | DLocal (v, init_opt) -> (
          let r = new_reg_for_type alloc v.var_type in
          env_bind env v.var_name r ;
          match init_opt with
          | None -> ()
          | Some e ->
              let r_init = emit_expr buf alloc env e in
              emit buf "mov.u32 %s, %s;" r r_init)
      | DShared (name, _elt, _size_opt) ->
          (* Shared memory requires .shared address-space allocation and
             cvta.to.global for pointer arithmetic.  Logged as a design gap;
             the pointer register is reserved but not valid for load/store. *)
          let r_ptr = new_u64 alloc in
          env_bind env name r_ptr ;
          emit
            buf
            "// shared array '%s' -> %%rd%d (lowering pending)"
            name
            (alloc.u64 - 1)
      | DParam _ -> ())
    locals

(** {1 Register block declaration} *)

(** Emit .reg declarations based on the allocator high-water marks. Must be
    called AFTER all emit_* calls complete. *)
let emit_reg_decls buf alloc =
  if alloc.u32 > 0 then Printf.bprintf buf "    .reg .u32 %%r<%d>;\n" alloc.u32 ;
  if alloc.u64 > 0 then Printf.bprintf buf "    .reg .u64 %%rd<%d>;\n" alloc.u64 ;
  if alloc.f32 > 0 then Printf.bprintf buf "    .reg .f32 %%f<%d>;\n" alloc.f32 ;
  if alloc.f64 > 0 then Printf.bprintf buf "    .reg .f64 %%fd<%d>;\n" alloc.f64 ;
  if alloc.pred > 0 then
    Printf.bprintf buf "    .reg .pred %%p<%d>;\n" alloc.pred

(** {1 Top-level kernel generator} *)

(** Generate the PTX file header.
    @param sm_target
      SM architecture string, e.g. ["sm_86"] for Ampere or ["sm_61"] for Pascal.
      Defaults to ["sm_86"] (RTX 30xx / A100+).
    @param ptx_version PTX language version. Defaults to ["8.0"] (CUDA 11.8+).
*)
let make_ptx_header ?(sm_target = "sm_86") ?(ptx_version = "8.0") () =
  Printf.sprintf
    ".version %s\n.target %s\n.address_size 64\n"
    ptx_version
    sm_target

(** Generate PTX for a single kernel. Three-phase: (1) emit body to count
    registers, (2) build header with correct register counts, (3) concatenate.
    @param sm_target Override the default [sm_86] target for older hardware. *)
let generate ?(sm_target = "sm_86") (k : kernel) : string =
  let alloc = make_alloc () in
  let env = make_env () in
  let body_buf = Buffer.create 2048 in
  let param_str = emit_params body_buf alloc env k.kern_params in
  emit_locals body_buf alloc env k.kern_locals ;
  emit_stmt body_buf alloc env k.kern_body ;
  Buffer.add_string body_buf "    ret;\n" ;
  let out = Buffer.create 4096 in
  Buffer.add_string out (make_ptx_header ~sm_target ()) ;
  Buffer.add_char out '\n' ;
  Printf.bprintf out ".entry %s(\n" k.kern_name ;
  Buffer.add_string out param_str ;
  Buffer.add_string out "\n)\n{\n" ;
  emit_reg_decls out alloc ;
  Buffer.add_char out '\n' ;
  Buffer.add_buffer out body_buf ;
  Buffer.add_string out "}\n" ;
  Buffer.contents out

(** Same interface as [Sarek_ir_cuda.generate_with_types]. Record and variant
    type definitions are not representable as PTX struct types; this is a
    documented design gap in ptx-spike-findings.md. *)
let generate_with_types ~types:_ (k : kernel) : string = generate k

(** {1 Spike demo: vector_add}

    Constructs the vector_add IR and calls [generate], demonstrating that the
    emitter produces structurally correct PTX for the simplest Sarek kernel
    pattern. *)
let demo_vector_add_ptx () : string =
  let make_var name ty =
    {var_name = name; var_id = 0; var_type = ty; var_mutable = false}
  in
  let a = make_var "a" (TVec TFloat32) in
  let b = make_var "b" (TVec TFloat32) in
  let c = make_var "c" (TVec TFloat32) in
  let n = make_var "n" TInt32 in
  let tid = make_var "tid" TInt32 in
  let body =
    SLet
      ( tid,
        EIntrinsic ([], "global_thread_id", []),
        SIf
          ( EBinop (Lt, EVar tid, EVar n),
            SAssign
              ( LArrayElem ("c", EVar tid),
                EBinop
                  (Add, EArrayRead ("a", EVar tid), EArrayRead ("b", EVar tid))
              ),
            None ) )
  in
  let k =
    {
      kern_name = "vector_add";
      kern_params =
        [
          DParam (a, Some {arr_elttype = TFloat32; arr_memspace = Global});
          DParam (b, Some {arr_elttype = TFloat32; arr_memspace = Global});
          DParam (c, Some {arr_elttype = TFloat32; arr_memspace = Global});
          DParam (n, None);
        ];
      kern_locals = [];
      kern_body = body;
      kern_types = [];
      kern_variants = [];
      kern_funcs = [];
      kern_native_fn = None;
    }
  in
  generate k
