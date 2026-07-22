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

(** f32 bit patterns for the base-2 change-of-base constants used by the exp/log
    lowerings (PTX only provides base-2 ex2/lg2). *)
let f32_log2_e_bits = Int32.bits_of_float (Float.log2 (Float.exp 1.0))

let f32_ln_2_bits = Int32.bits_of_float (Float.log 2.0)

(** An f64 register is "%fd<n>"; an f32 register is "%f<n>" (not "%fd<n>"). *)
let is_f64_reg r = String.length r >= 3 && r.[1] = 'f' && r.[2] = 'd'

let is_f32_reg r = String.length r >= 2 && r.[1] = 'f' && not (is_f64_reg r)

(** Whether an expression must NOT be evaluated speculatively — i.e. an [EIf]
    with such a branch must emit real control flow rather than the eager
    evaluate-both-branches [selp] path. Two reasons a subexpression qualifies:
    - it has an observable effect (a store/atomic, or a helper call whose body
      may store/atomic/barrier); running the not-taken branch is wrong;
    - it dereferences memory (an array read); the not-taken branch's index may
      be out of bounds (the classic [if i < n then a.(i) else d] guard), and an
      unconditional load can fault or read garbage. *)
let rec expr_needs_branch_guard (e : expr) : bool =
  let is_atomic name =
    String.length name >= 7 && String.sub name 0 7 = "atomic_"
  in
  (* Barriers/fences emit side-effecting, convergence-sensitive instructions
     (bar.sync, membar); they must never run on a not-taken branch. *)
  let is_barrier = function
    | "block_barrier" | "warp_barrier" | "memory_fence" -> true
    | _ -> false
  in
  match e with
  | EApp _ -> true
  | EIntrinsic (_, name, args) ->
      is_atomic name || is_barrier name
      || List.exists expr_needs_branch_guard args
  (* Array reads must not be evaluated speculatively (out-of-bounds guard). *)
  | EArrayRead _ | EArrayReadExpr _ -> true
  | EConst _ | EVar _ | EArrayLen _ -> false
  | EUnop (_, a) | ECast (_, a) -> expr_needs_branch_guard a
  | EBinop (_, a, b) -> expr_needs_branch_guard a || expr_needs_branch_guard b
  | EIf (c, t, f) ->
      expr_needs_branch_guard c || expr_needs_branch_guard t
      || expr_needs_branch_guard f
  | EArrayCreate (_, s, _) -> expr_needs_branch_guard s
  | EMatch _ | ERecord _ | ERecordField _ | ETuple _ | EVariant _ ->
      (* Rejected later by the emitter; conservative answer is irrelevant. *)
      true

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
      emit_array_read
        buf
        alloc
        r_base
        r_idx
        (infer_elt_type alloc arr_name)
        ~is_shared:(Hashtbl.mem alloc.arr_memspaces arr_name)
  | EArrayReadExpr (base_expr, idx_expr) ->
      let r_base = emit_expr buf alloc env base_expr in
      let r_idx = emit_expr buf alloc env idx_expr in
      let arr_name_opt =
        match base_expr with EVar v -> Some v.var_name | _ -> None
      in
      let elt_type =
        match arr_name_opt with
        | Some n -> infer_elt_type alloc n
        | None ->
            fail
              "EArrayReadExpr: cannot infer element type from non-variable \
               base expression"
      in
      let is_shared =
        match arr_name_opt with
        | Some n -> Hashtbl.mem alloc.arr_memspaces n
        | None -> false
      in
      emit_array_read buf alloc r_base r_idx elt_type ~is_shared
  | EIntrinsic (path, name, args) -> emit_intrinsic buf alloc env path name args
  | ECast (ty, e) ->
      let r_src = emit_expr buf alloc env e in
      emit_cast buf alloc r_src ty
  | EIf (_, then_e, else_e) as e
    when expr_needs_branch_guard then_e || expr_needs_branch_guard else_e -> (
      (* A branch with an effect or a (possibly out-of-bounds) array read must
         not be evaluated eagerly (the selp path below computes both); emit
         real control flow instead — emit_value owns the branch-based path. *)
      match emit_value buf alloc env e with
      | Scalar r -> r
      | Agg _ ->
          fail
            "PTX codegen: if-expression of record/variant type used in a \
             scalar context; bind it with let and read its scalar fields")
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
  | EArrayLen arr ->
      (* Bound by emit_params alongside the array pointer param. Only
         parameter arrays carry a length; local/shared arrays fall back to
         another backend. *)
      if Hashtbl.mem env (length_param_name arr) then
        env_lookup env (length_param_name arr)
      else
        unsupported
          (Printf.sprintf
             "EArrayLen on '%s' (only parameter arrays have a length)"
             arr)
  | EArrayCreate _ ->
      unsupported "EArrayCreate in expression position (use SLet)"
  | EMatch _ -> unsupported "EMatch (requires variant lowering)"
  | ERecord (name, _) ->
      fail
        ("PTX codegen: record value of type '" ^ name
       ^ "' cannot be used in a scalar context; bind it with let and read its \
          scalar fields")
  | ERecordField (_, field) as e -> (
      (* A field projection is usually scalar; delegate to emit_value and
         reject only when the projected field is itself an aggregate. *)
      match emit_value buf alloc env e with
      | Scalar r -> r
      | Agg _ ->
          fail
            ("PTX codegen: field '" ^ field
           ^ "' is a nested record/variant and cannot be used in a scalar \
              context; bind it with let and read its scalar fields"))
  | ETuple _ ->
      fail
        "PTX codegen: tuple value used in a scalar context (e.g. stored into a \
         vector element); bind it with let and use its components \
         individually, or use a registered record type instead"
  | EApp (EVar f, args) -> (
      match emit_app buf alloc env f args with
      | Scalar r -> r
      | Agg _ ->
          fail
            ("PTX codegen: helper '" ^ f.var_name
           ^ "' returns a record/variant and cannot be used in a scalar \
              context; bind the result with let and use its fields"))
  | EApp _ -> unsupported "EApp with non-variable callee"
  | EVariant _ -> unsupported "EVariant (requires tagged-union lowering)"

(** {1 Helper-function inlining}

    Inline the helper body at the call site. PTX .func would need a per-function
    register frame and .param ABI the single-pass emitter does not model;
    helpers are small and NVCC inlines them anyway. Recursive helpers are
    rejected and fall back to another backend, as before. *)

(** Bind one helper parameter in [callee_env] from the caller's evaluated
    argument. Scalar and aggregate values are leaf-wise copied so mutations
    inside the helper can never clobber the caller's registers; array params are
    base pointers, never written through LVar — bound directly. Returns array
    metadata to restore after the inline ([None] for non-arrays). *)
and bind_helper_param buf alloc env callee_env (p : var) (arg, arg_val) =
  (match (p.var_type, arg_val) with
  | (TVec _ | TArray _), Scalar r_arg -> env_bind callee_env p.var_name r_arg
  | (TVec _ | TArray _), Agg _ ->
      fail
        (Printf.sprintf
           "PTX codegen: array parameter '%s' bound to a record/variant value"
           p.var_name)
  | _, v -> env_bind_binding callee_env p.var_name (copy_binding buf alloc v)) ;
  (* Array params: register element type (from the param's own type) and
     propagate shared-ness / length binding from the caller when the argument
     is a plain array variable. Overwritten entries are saved for restore. *)
  match p.var_type with
  | TVec elt | TArray (elt, _) ->
      let prev_elt = Hashtbl.find_opt alloc.arr_elt_types p.var_name in
      let prev_ms = Hashtbl.mem alloc.arr_memspaces p.var_name in
      Hashtbl.replace alloc.arr_elt_types p.var_name elt ;
      (match arg with
      | EVar a -> (
          if Hashtbl.mem alloc.arr_memspaces a.var_name then
            Hashtbl.replace alloc.arr_memspaces p.var_name ()
          else Hashtbl.remove alloc.arr_memspaces p.var_name ;
          match Hashtbl.find_opt env (length_param_name a.var_name) with
          | Some len_binding ->
              env_bind_binding
                callee_env
                (length_param_name p.var_name)
                len_binding
          | None -> ())
      | _ -> Hashtbl.remove alloc.arr_memspaces p.var_name) ;
      Some (p.var_name, prev_elt, prev_ms)
  | _ -> None

(** Restore array metadata shadowed by helper parameter names. *)
and restore_helper_array_meta alloc saved =
  List.iter
    (function
      | None -> ()
      | Some (name, prev_elt, prev_ms) ->
          (match prev_elt with
          | Some e -> Hashtbl.replace alloc.arr_elt_types name e
          | None -> Hashtbl.remove alloc.arr_elt_types name) ;
          if prev_ms then Hashtbl.replace alloc.arr_memspaces name ()
          else Hashtbl.remove alloc.arr_memspaces name)
    saved

(** Inline a helper call and return the binding holding its result. Aggregate
    returns are pre-allocated from [hf_ret_type] (FR-023); SReturn inside the
    inlined body movs leaf-wise into that binding. *)
and emit_app buf alloc (env : env) (f : var) (args : expr list) : binding =
  match Hashtbl.find_opt alloc.funcs f.var_name with
  | None -> unsupported ("EApp to unknown function '" ^ f.var_name ^ "'")
  | Some hf ->
      if List.mem hf.hf_name alloc.inline_stack then
        unsupported
          ("EApp: recursive helper '" ^ hf.hf_name
         ^ "' (inlining supports non-recursive helpers only)")
      else if List.length args <> List.length hf.hf_params then
        fail
          (Printf.sprintf
             "PTX codegen: helper '%s' called with %d args, expects %d"
             hf.hf_name
             (List.length args)
             (List.length hf.hf_params))
      else begin
        (* Evaluate arguments in the caller's environment. *)
        let arg_vals = List.map (emit_value buf alloc env) args in
        (* Fresh environment for the helper body: only its parameters are in
           scope (helpers are module-level, no capture). *)
        let callee_env = make_env () in
        let saved =
          List.map2
            (bind_helper_param buf alloc env callee_env)
            hf.hf_params
            (List.combine args arg_vals)
        in
        let l_end = new_label alloc in
        let ret =
          match hf.hf_ret_type with
          | TUnit -> None
          | t -> Some (binding_of_elttype alloc t)
        in
        alloc.inline_stack <- hf.hf_name :: alloc.inline_stack ;
        alloc.inline_ret <- (ret, l_end) :: alloc.inline_ret ;
        !stmt_emitter buf alloc callee_env hf.hf_body ;
        alloc.inline_ret <- List.tl alloc.inline_ret ;
        alloc.inline_stack <- List.tl alloc.inline_stack ;
        emit_label buf l_end ;
        restore_helper_array_meta alloc saved ;
        match ret with
        | Some b -> b
        | None ->
            let r = new_u32 alloc in
            emit buf "mov.u32 %s, 0;" r ;
            Scalar r
      end

(** {1 Value emitter (scalar or aggregate)}

    [emit_value] is the aggregate-aware entry point: scalar expressions delegate
    to {!emit_expr} (wrapped in [Scalar]); record construction and field
    projection build/select SROA register sets ([Agg]) without touching memory
    (FR-020). *)
and emit_value buf alloc (env : env) (e : expr) : binding =
  match e with
  | ERecord (_name, fields) ->
      (* Field order = declaration order as carried by the ERecord node. *)
      Agg
        (ARecord
           (List.map (fun (n, fe) -> (n, emit_value buf alloc env fe)) fields))
  | ERecordField (base, field) -> emit_record_field buf alloc env base field
  | EVar v -> env_lookup_binding env v.var_name
  | EApp (EVar f, args) -> emit_app buf alloc env f args
  | EIf (cond, then_e, else_e)
    when expr_needs_branch_guard then_e || expr_needs_branch_guard else_e ->
      (* Branch-based conditional, aggregate-capable: the then-value's binding
         is leaf-wise copied into the result, the else-value leaf-wise moved
         into it (for scalars this emits exactly the instructions of
         emit_expr's guarded EIf path). *)
      let r_cond = emit_expr buf alloc env cond in
      let p = new_pred alloc in
      emit buf "setp.ne.u32 %s, %s, 0;" p r_cond ;
      let l_else = new_label alloc in
      let l_merge = new_label alloc in
      emit buf "@!%s bra %s;" p l_else ;
      let b_then = emit_value buf alloc env then_e in
      let b_res = copy_binding buf alloc b_then in
      emit buf "bra %s;" l_merge ;
      emit_label buf l_else ;
      let b_else = emit_value buf alloc env else_e in
      mov_binding buf ~src:b_else ~dst:b_res ;
      emit_label buf l_merge ;
      b_res
  | _ -> Scalar (emit_expr buf alloc env e)

(** Field selection on a local (SROA) record value: pure register selection, no
    instructions emitted for the projection itself. *)
and emit_record_field buf alloc env base field : binding =
  match emit_value buf alloc env base with
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
        ("PTX codegen: field access '." ^ field
       ^ "' on a variant value; use match to inspect a variant")
  | Scalar _ ->
      fail
        ("PTX codegen: field access '." ^ field
       ^ "' on a non-record value (records in vector elements are not yet \
          supported at this stage; build the record locally)")

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
      (* Arithmetic (sign-extending) shift: Ir.Shr is arithmetic on every
         backend (CUDA/OpenCL/Metal/GLSL/WGSL emit plain [>>] on a signed
         int type; the interpreter uses Int32.shift_right). [lsr] is lowered
         to a separate expression tree in Sarek_lower_ir.ml precisely
         because this node is arithmetic - see G phase 1 in
         briefs/fix-critical-semantics-evidence.md. Formal spec note:
         formal/codegen-ptx/theories/PtxTypes.v models Shr as a logical
         Nat.shiftr on U32; that model was written against the old (wrong)
         shr.u32 emission and is now out of sync with this fix. formal/ is
         out of scope for this task - flagged for the formal-verification
         owner. *)
      let r = new_u32 alloc in
      emit buf "shr.s32 %s, %s, %s;" r r1 r2 ;
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

and emit_intrinsic buf alloc env path name args : string =
  (* Type conversions delegate to emit_cast; a unary helper for them. *)
  let unary_cast intr dst_ty =
    match args with
    | [a] -> emit_cast buf alloc (emit_expr buf alloc env a) dst_ty
    | _ -> unsupported (intr ^ " arity != 1")
  in
  (* atom.{shared,global}.add.s32 on an int32 array denoted by a plain
     variable; returns the old value. *)
  (* Atomic read-modify-write on a 4-byte element of an int32/float32 array
     denoted by a plain variable; returns the old value. [op]/[ty] form the
     PTX suffix (e.g. add.s32, min.s32, and.b32, exch.b32, add.f32). PTX has
     no atom.sub, so "sub" is lowered to an add of the negated operand.
     [result] selects the old-value register class. *)
  let atomic_rmw intr ~global_only ~op ~ty ~result =
    match args with
    | [EVar arr; idx_e; val_e] ->
        let r_base = env_lookup env arr.var_name in
        let r_idx = emit_expr buf alloc env idx_e in
        let r_val0 = emit_expr buf alloc env val_e in
        let op, r_val =
          if op = "sub" then begin
            let r = new_u32 alloc in
            emit buf "neg.s32 %s, %s;" r r_val0 ;
            ("add", r)
          end
          else (op, r_val0)
        in
        let is_shared =
          (not global_only) && Hashtbl.mem alloc.arr_memspaces arr.var_name
        in
        let r_old =
          match result with `F32 -> new_f32 alloc | `U32 -> new_u32 alloc
        in
        if is_shared then begin
          let r_off = new_u32 alloc in
          let r_addr = new_u32 alloc in
          emit buf "shl.b32 %s, %s, 2;" r_off r_idx ;
          emit buf "add.u32 %s, %s, %s;" r_addr r_base r_off ;
          emit buf "atom.shared.%s%s %s, [%s], %s;" op ty r_old r_addr r_val
        end
        else begin
          let r_idx64 = new_u64 alloc in
          let r_off = new_u64 alloc in
          let r_addr = new_u64 alloc in
          emit buf "cvt.u64.u32 %s, %s;" r_idx64 r_idx ;
          emit buf "shl.b64 %s, %s, 2;" r_off r_idx64 ;
          emit buf "add.u64 %s, %s, %s;" r_addr r_base r_off ;
          emit buf "atom.global.%s%s %s, [%s], %s;" op ty r_old r_addr r_val
        end ;
        r_old
    | _ -> unsupported (intr ^ ": expects (array-variable, index, value)")
  in
  (* Binary min/max: native PTX op, typed by the first operand's register. *)
  let binary_minmax intr op =
    match args with
    | [a; b] ->
        let ra = emit_expr buf alloc env a in
        let rb = emit_expr buf alloc env b in
        if is_f64_reg ra then (
          let r = new_f64 alloc in
          emit buf "%s.f64 %s, %s, %s;" op r ra rb ;
          r)
        else if is_f32_reg ra then (
          let r = new_f32 alloc in
          emit buf "%s.f32 %s, %s, %s;" op r ra rb ;
          r)
        else
          let r = new_u32 alloc in
          emit buf "%s.s32 %s, %s, %s;" op r ra rb ;
          r
    | _ -> unsupported (intr ^ " arity != 2")
  in
  (* Unary same-type float rounding via cvt (rmi = floor, rpi = ceil). *)
  let unary_round intr cvt =
    match args with
    | [a] ->
        let r = emit_expr buf alloc env a in
        if is_f64_reg r then (
          let d = new_f64 alloc in
          emit buf "%s.f64.f64 %s, %s;" cvt d r ;
          d)
        else if is_f32_reg r then (
          let d = new_f32 alloc in
          emit buf "%s.f32.f32 %s, %s;" cvt d r ;
          d)
        else unsupported (intr ^ ": float operand required")
    | _ -> unsupported (intr ^ " arity != 1")
  in
  (* Emit the single argument of a unary f32 intrinsic, rejecting f64
     operands (the .approx PTX ops used below are f32-only; an f64 operand
     would emit invalid PTX that only fails at module-load time). *)
  let unary_f32_arg intr args =
    match args with
    | [a] ->
        let r = emit_expr buf alloc env a in
        if is_f32_reg r then r
        else unsupported (intr ^ ": f32 only (operand " ^ r ^ " not lowered)")
    | _ -> unsupported (intr ^ " arity != 1")
  in
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
  | "exp" ->
      (* exp(x) = 2^(x * log2 e); PTX only has base-2 ex2, f32 only *)
      let r_arg = unary_f32_arg "exp" args in
      let r_scaled = new_f32 alloc in
      emit buf "mul.f32 %s, %s, 0F%08lX;" r_scaled r_arg f32_log2_e_bits ;
      let r = new_f32 alloc in
      emit buf "ex2.approx.f32 %s, %s;" r r_scaled ;
      r
  | "log" ->
      (* log(x) = log2(x) * ln 2; PTX only has base-2 lg2, f32 only *)
      let r_arg = unary_f32_arg "log" args in
      let r_lg2 = new_f32 alloc in
      emit buf "lg2.approx.f32 %s, %s;" r_lg2 r_arg ;
      let r = new_f32 alloc in
      emit buf "mul.f32 %s, %s, 0F%08lX;" r r_lg2 f32_ln_2_bits ;
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
  (* Type conversions (Gpu.float / Float32.of_int / …). "of_int"/"to_int"
     are path-dependent: they exist in both the Float32 and Float64 stdlib
     modules. *)
  | "float" | "float_of_int" -> unary_cast name TFloat32
  | "float64" | "float64_of_int" -> unary_cast name TFloat64
  | "int_of_float" | "int_of_float64" -> unary_cast name TInt32
  | "of_int" ->
      if List.exists (fun p -> p = "Float64") path then unary_cast name TFloat64
      else unary_cast name TFloat32
  | "to_int" -> unary_cast name TInt32
  (* Atomics (int32 add; old value returned). *)
  (* Native math with a direct PTX op. *)
  | "min" -> binary_minmax name "min"
  | "max" -> binary_minmax name "max"
  | "floor" -> unary_round name "cvt.rmi"
  | "ceil" -> unary_round name "cvt.rpi"
  | "rsqrt" -> (
      match args with
      | [a] ->
          let r = emit_expr buf alloc env a in
          if is_f64_reg r then (
            let d = new_f64 alloc in
            emit buf "rsqrt.approx.f64 %s, %s;" d r ;
            d)
          else if is_f32_reg r then (
            let d = new_f32 alloc in
            emit buf "rsqrt.approx.f32 %s, %s;" d r ;
            d)
          else unsupported "rsqrt: float operand required"
      | _ -> unsupported "rsqrt arity != 1")
  (* Atomics (old value returned). Shared vs global is auto-detected from the
     array's memory space; the *_global_* names force the global path. *)
  | "atomic_add_int32" ->
      atomic_rmw name ~global_only:false ~op:"add" ~ty:".s32" ~result:`U32
  | "atomic_add_global_int32" ->
      atomic_rmw name ~global_only:true ~op:"add" ~ty:".s32" ~result:`U32
  | "atomic_sub_int32" ->
      atomic_rmw name ~global_only:false ~op:"sub" ~ty:".s32" ~result:`U32
  | "atomic_min_int32" ->
      atomic_rmw name ~global_only:false ~op:"min" ~ty:".s32" ~result:`U32
  | "atomic_max_int32" ->
      atomic_rmw name ~global_only:false ~op:"max" ~ty:".s32" ~result:`U32
  | "atomic_and_int32" ->
      atomic_rmw name ~global_only:false ~op:"and" ~ty:".b32" ~result:`U32
  | "atomic_or_int32" ->
      atomic_rmw name ~global_only:false ~op:"or" ~ty:".b32" ~result:`U32
  | "atomic_xor_int32" ->
      atomic_rmw name ~global_only:false ~op:"xor" ~ty:".b32" ~result:`U32
  | "atomic_exch_int32" ->
      atomic_rmw name ~global_only:false ~op:"exch" ~ty:".b32" ~result:`U32
  | "atomic_add_float32" ->
      atomic_rmw name ~global_only:false ~op:"add" ~ty:".f32" ~result:`F32
  | n -> unsupported ("intrinsic: " ^ n)
