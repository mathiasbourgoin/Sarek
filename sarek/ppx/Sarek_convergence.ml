(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Sarek PPX - GPU kernel DSL for OCaml
 *
 * Convergence analysis for barrier safety.
 *
 * GPU barriers require ALL threads in a workgroup to reach the barrier.
 * If threads diverge (take different branches) and only some reach the
 * barrier, the GPU deadlocks.
 *
 * This pass tracks execution mode (Converged vs Diverged) and rejects
 * barrier calls when mode is Diverged.
 *
 * Example of rejected code:
 *   if thread_idx_x > 16 then
 *     block_barrier ()   (* Error: only threads 17-31 reach this! *)
 *
 * The rules are specified in specs/convergence-semantics.md. Supersteps
 * (section 2): a superstep reached in diverged flow is an error with or
 * without [@divergent], which is inert; the body and the continuation are
 * checked in the outer mode. Let-bound variance is tracked through
 * [varying_vars]; the strict pass also follows helper parameters, helper
 * results and mutables assigned in divergence (section 3); array variance is not
 * tracked yet.
 ******************************************************************************)

open Sarek_typed_ast
open Sarek_env
module StringSet = Set.Make (String)

(** Execution mode for convergence analysis *)
type exec_mode =
  | Converged  (** All threads in workgroup execute this code *)
  | Diverged  (** Threads may have taken different branches *)

(** The synchronisation point a helper function contains: what a call to it
    reaches. *)
type sync_kind =
  | Barrier_sync  (** a block barrier or a superstep *)
  | Warp_sync of string  (** a warp collective, by name *)

(** A helper function in scope: what a call to it can be analysed against. *)
type helper_def = {
  hd_self : string;  (** the name it was defined under *)
  hd_params : string list;
  hd_body : texpr;
  hd_defs : (string * helper_def) list;
      (** The helpers in scope at its definition (itself excluded): a call
          resolves names lexically, as the generated code does, not in the
          caller's scope. *)
  hd_captured : StringSet.t;
      (** Thread-varying variables a nested helper captures from its definition
          site; empty for a module-level function. *)
}

(** How variance is computed. The [legacy] analysis is the first, cheap pass.
    The [strict] analysis is applied to every synchronisation point (warp
    collectives, whose missed divergence is a silent wrong answer, and barriers,
    whose missed divergence is a deadlock or undefined behaviour): it also
    follows the value of an [if]/[match], a helper result that reads the thread,
    and a mutable assigned under thread-varying control flow
    (specs/convergence-semantics.md section 3). *)
type variance = {
  strict : bool;
  defs : (string * helper_def) list;  (** helpers in scope, innermost first *)
  visiting : string list;  (** helpers being unfolded: recursion guard *)
}

let legacy_variance = {strict = false; defs = []; visiting = []}

(** Memo of [assigned_varying], per mutable name: the scan of an inner [let mut]
    body is repeated by every enclosing [let mut], which is exponential in their
    nesting. An entry is valid for one body (physical equality), one variance
    environment and one set of visible helpers. Cleared at each kernel check. *)
type assigned_entry = {
  ae_body : texpr;
  ae_env : StringSet.t;
  ae_vi : variance;
  ae_result : bool;
}

let assigned_memo : (string, assigned_entry list) Hashtbl.t = Hashtbl.create 64

let find_assigned vi env name body =
  List.find_opt
    (fun e ->
      e.ae_body == body
      && StringSet.equal e.ae_env env
      && e.ae_vi.defs == vi.defs
      && e.ae_vi.visiting = vi.visiting
      && e.ae_vi.strict = vi.strict)
    (Option.value ~default:[] (Hashtbl.find_opt assigned_memo name))

(** Analysis context *)
type ctx = {
  mode : exec_mode;
  varying_vars : StringSet.t;
      (** Variables known to carry thread-varying values at this point *)
  sync_helpers : (string * sync_kind) list;
      (** Helper functions in scope that contain a synchronisation point,
          transitively (specs/convergence-semantics.md, rule H-2): a call to one
          in diverged flow is an error at the call site. *)
  variance : variance;
  call_stack : string list;
      (** Helpers whose body is being re-checked at a call (strict analysis) *)
}

(** Initial context - start converged *)
let init_ctx =
  {
    mode = Converged;
    varying_vars = StringSet.empty;
    sync_helpers = [];
    variance = legacy_variance;
    call_stack = [];
  }

(** Enter diverged mode, preserving known-varying variable set *)
let diverge ctx = {ctx with mode = Diverged}

(** Check if an intrinsic ref is thread-varying using core primitives *)
let is_thread_varying_intrinsic (ref : intrinsic_ref) : bool =
  match ref with
  | IntrinsicRef (_, base_name) ->
      Sarek_core_primitives.is_thread_varying base_name
  | CorePrimitiveRef name -> Sarek_core_primitives.is_thread_varying name

(** Check if an intrinsic ref is a barrier/convergence point *)
let is_barrier_ref (ref : intrinsic_ref) : bool =
  match ref with
  | IntrinsicRef (_, base_name) ->
      Sarek_core_primitives.is_convergence_point base_name
  | CorePrimitiveRef name -> Sarek_core_primitives.is_convergence_point name

(** True if texpr e contains a TEReturn reachable on some execution path *)
let rec has_reachable_return (e : texpr) : bool =
  match e.te with
  | TEReturn _ -> true
  | TEIf (_, t, else_opt) -> (
      has_reachable_return t
      || match else_opt with Some e -> has_reachable_return e | None -> false)
  | TESeq es -> List.exists has_reachable_return es
  | TELet (_, _, v, b) | TELetMut (_, _, v, b) ->
      has_reachable_return v || has_reachable_return b
  | TEWhile (_, b) -> has_reachable_return b
  | TEFor (_, _, _, _, _, b) -> has_reachable_return b
  | TEMatch (_, cases) ->
      List.exists (fun (_, e) -> has_reachable_return e) cases
  | TESuperstep (_, _, body, cont) ->
      has_reachable_return body || has_reachable_return cont
  | TEOpen (_, body) -> has_reachable_return body
  | TELetRec (_, _, _, fn_body, cont) ->
      has_reachable_return fn_body || has_reachable_return cont
  | _ -> false

(** The direct sub-expressions of [te], in evaluation order. *)
let sub_exprs (te : texpr) : texpr list =
  match te.te with
  | TEUnit | TEBool _ | TEInt _ | TEInt32 _ | TEInt64 _ | TEFloat _ | TEDouble _
  | TEVar _ | TEGlobalRef _ | TENative _ | TEIntrinsicConst _ ->
      []
  | TEVecGet (a, b) | TEArrGet (a, b) | TEBinop (_, a, b) | TEWhile (a, b) ->
      [a; b]
  | TEVecSet (a, b, c) | TEArrSet (a, b, c) -> [a; b; c]
  | TEFieldGet (e, _, _)
  | TEUnop (_, e)
  | TEAssign (_, _, e)
  | TEReturn e
  | TEPragma (_, e)
  | TEOpen (_, e)
  | TECreateArray (e, _, _) ->
      [e]
  | TEFieldSet (e, _, _, x) -> [e; x]
  | TEApp (f, args) -> f :: args
  | TELet (_, _, v, b) | TELetMut (_, _, v, b) -> [v; b]
  | TELetRec (_, _, _, fn_body, cont) -> [fn_body; cont]
  | TEIf (c, t, e) -> c :: t :: Option.to_list e
  | TEFor (_, _, lo, hi, _, b) -> [lo; hi; b]
  | TESeq es | TETuple es -> es
  | TEMatch (s, cases) -> s :: List.map snd cases
  | TERecord (_, fields) -> List.map snd fields
  | TEConstr (_, _, arg) -> Option.to_list arg
  | TELetShared (_, _, _, size, body) -> Option.to_list size @ [body]
  | TESuperstep (_, _, body, cont) -> [body; cont]
  | TEIntrinsicFun (_, _, args) -> args

(** Every variable name occurring in [e]. *)
let rec vars_of acc (e : texpr) : StringSet.t =
  let acc = match e.te with TEVar (n, _) -> StringSet.add n acc | _ -> acc in
  List.fold_left vars_of acc (sub_exprs e)

(** The variables whose membership in the environment {!assigned_varying_scan}
    can test for the mutable [name] over [body]. The scan asks whether an
    expression varies in exactly these places: the value assigned to [name] (and
    to a mutable bound inside [body], through the recursive analysis of that
    one), the conditions of [if] and [while], the bounds of [for], the scrutinee
    of [match], the value of [let] and [let mut] (also read by
    [has_varying_return_with] on every statement of a sequence). The values
    assigned to a mutable bound outside [body], and every other operand (for
    instance what is stored into an array), are never asked about, so a name
    that occurs only there does not matter. *)
let relevant_reads name (body : texpr) : StringSet.t =
  let rec bound acc (e : texpr) =
    let acc =
      match e.te with TELetMut (n, _, _, _) -> StringSet.add n acc | _ -> acc
    in
    List.fold_left bound acc (sub_exprs e)
  in
  let inner = bound StringSet.empty body in
  let rec go acc (e : texpr) =
    let acc =
      match e.te with
      | TEAssign (n, _, v) when n = name || StringSet.mem n inner ->
          vars_of acc v
      | TEIf (c, _, _) | TEWhile (c, _) | TEMatch (c, _) -> vars_of acc c
      | TEFor (_, _, lo, hi, _, _) -> vars_of (vars_of acc lo) hi
      | TELet (_, _, v, _) | TELetMut (_, _, v, _) -> vars_of acc v
      | _ -> acc
    in
    List.fold_left go acc (sub_exprs e)
  in
  go StringSet.empty body

let helper_name (f : texpr) : string option =
  match f.te with
  | TEVar (name, _) | TEGlobalRef (name, _) -> Some name
  | _ -> None

(** The helpers in scope inside the body of [def]: those of its definition site
    plus itself. *)
let body_defs (def : helper_def) =
  (def.hd_self, def) :: List.remove_assoc def.hd_self def.hd_defs

(** The environment a helper body is evaluated in at a call: what it captured,
    minus its own parameters, plus the parameters whose argument is varying. *)
let helper_call_env (def : helper_def) (varying_args : bool list) : StringSet.t
    =
  let base =
    List.fold_left
      (fun s p -> StringSet.remove p s)
      def.hd_captured
      def.hd_params
  in
  let rec bind env params flags =
    match (params, flags) with
    | p :: ps, true :: fs -> bind (StringSet.add p env) ps fs
    | _ :: ps, false :: fs -> bind env ps fs
    | _ -> env
  in
  bind base def.hd_params varying_args

(** Check if an expression's value varies per-thread.

    Thread-varying (different per thread):
    - thread_idx_x, thread_idx_y, thread_idx_z
    - global_thread_id
    - Expressions derived from above
    - Array accesses with thread-varying index

    Uniform (same for all threads in workgroup):
    - Constants and literals
    - block_idx_x/y/z (same for whole block)
    - block_dim_x/y/z, grid_dim_x/y/z
    - Expressions derived only from uniform values

    With [vi.strict] the value of control flow, helper results and mutables
    assigned in divergence are followed too (see {!variance}). *)
let rec varying (vi : variance) (env : StringSet.t) (te : texpr) : bool =
  let v = varying vi env in
  match te.te with
  (* Literals are uniform *)
  | TEUnit | TEBool _ | TEInt _ | TEInt32 _ | TEInt64 _ | TEFloat _ | TEDouble _
    ->
      false
  (* Variable: check static intrinsics table AND locally-inferred varying set *)
  | TEVar (name, _) ->
      Sarek_core_primitives.is_thread_varying name || StringSet.mem name env
  (* Intrinsic constants - check if thread-varying *)
  | TEIntrinsicConst ref -> is_thread_varying_intrinsic ref
  (* Binary ops: varying if either operand varies *)
  | TEBinop (_, e1, e2) -> v e1 || v e2
  (* Unary ops: varying if operand varies *)
  | TEUnop (_, e) -> v e
  (* Array/vector access: varying if index varies *)
  | TEVecGet (_, idx) | TEArrGet (_, idx) -> v idx
  (* Field access: varying if record varies *)
  | TEFieldGet (e, _, _) -> v e
  (* Application: varying if any argument varies; strictly, also if the helper's
     own result reads the thread *)
  | TEApp (f, args) ->
      List.exists v args || (vi.strict && helper_result_varying vi env f args)
  (* Intrinsic function: varying if any argument varies *)
  | TEIntrinsicFun (ref, _, args) ->
      (vi.strict && is_thread_varying_intrinsic ref) || List.exists v args
  (* Tuple: varying if any element varies *)
  | TETuple es -> List.exists v es
  (* Let binding: propagate variance of bound value into the body *)
  | TELet (name, _, value, body) ->
      let vv = v value in
      let env' = if vv then StringSet.add name env else env in
      (* Strict: the value of a [let] is that of its body, the bound value
         reaches it only through the name. *)
      ((not vi.strict) && vv) || varying (alias_def vi name value) env' body
  | TELetMut (name, _, value, body) ->
      let vv = v value in
      let env' =
        if vv || (vi.strict && assigned_varying vi env name body) then
          StringSet.add name env
        else env
      in
      ((not vi.strict) && vv) || varying vi env' body
  | TESeq es -> List.exists v es
  (* BSP constructs: check body/continuation *)
  | TELetShared (_, _, _, size_opt, body) ->
      (match size_opt with Some size -> v size | None -> false) || v body
  | TESuperstep (_, _, step_body, cont) -> v step_body || v cont
  | TEOpen (_, body) -> v body
  (* The value of control flow, only under the strict analysis *)
  | TEIf (cond, t, e) ->
      vi.strict
      && (v cond || v t || match e with Some e -> v e | None -> false)
  | TEMatch (scrut, cases) ->
      vi.strict && (v scrut || List.exists (fun (_, e) -> v e) cases)
  | TERecord (_, fields) -> vi.strict && List.exists (fun (_, e) -> v e) fields
  | TEConstr (_, _, arg) -> (
      vi.strict && match arg with Some e -> v e | None -> false)
  | TEPragma (_, e) -> vi.strict && v e
  | TELetRec (name, _, params, fn_body, cont) ->
      vi.strict
      &&
      let vi' = add_def vi env name params fn_body in
      varying vi' env cont
  (* Default: assume uniform (minimize false positives) *)
  | TEFor _ | TEWhile _ | TEReturn _ | TECreateArray _ | TEGlobalRef _
  | TENative _ | TEVecSet _ | TEArrSet _ | TEFieldSet _ | TEAssign _ ->
      false

(** [let g = f]: [g] is another name for the helper [f] (strict analysis). *)
and alias_def vi name (value : texpr) =
  if not vi.strict then vi
  else
    match helper_name value with
    | Some f when f <> name -> (
        match List.assoc_opt f vi.defs with
        | Some def ->
            {vi with defs = (name, def) :: List.remove_assoc name vi.defs}
        | None -> vi)
    | _ -> vi

(** Register a nested helper. *)
and add_def vi env name (params : tparam list) body =
  let def =
    {
      hd_self = name;
      hd_params = List.map (fun p -> p.tparam_name) params;
      hd_body = body;
      hd_defs = List.remove_assoc name vi.defs;
      hd_captured = env;
    }
  in
  {vi with defs = (name, def) :: List.remove_assoc name vi.defs}

(** Strict analysis: does a call to a known helper return a thread-dependent
    value, given which arguments vary? The body is unfolded once per call (a
    recursion guard stops cycles). *)
and helper_result_varying vi env (f : texpr) (args : texpr list) : bool =
  match helper_name f with
  | Some name when not (List.mem name vi.visiting) -> (
      match List.assoc_opt name vi.defs with
      | Some def ->
          let flags = List.map (varying vi env) args in
          let vi' =
            {vi with visiting = name :: vi.visiting; defs = body_defs def}
          in
          varying vi' (helper_call_env def flags) def.hd_body
      | None -> false)
  | _ -> false

(** Strict analysis: is the mutable [name] assigned, anywhere in [body], a
    thread-varying value or under thread-varying control flow? Then its value
    after the assignment differs per thread. *)
and assigned_varying vi env name (body : texpr) : bool =
  (* The answer depends on [env] only through the names the scan can test, so
     restrict it to them before it keys the memo. Without that, the [env] of an
     inner mutable differs by which enclosing mutables are being analysed, and
     each mutable was analysed once per subset of them (#558). *)
  let env = StringSet.inter env (relevant_reads name body) in
  match find_assigned vi env name body with
  | Some e -> e.ae_result
  | None ->
      let result = assigned_varying_scan vi env name body in
      Hashtbl.replace
        assigned_memo
        name
        ({ae_body = body; ae_env = env; ae_vi = vi; ae_result = result}
        :: Option.value ~default:[] (Hashtbl.find_opt assigned_memo name)) ;
      result

and assigned_varying_scan vi env name (body : texpr) : bool =
  let rec scan mode env (e : texpr) : bool =
    match e.te with
    | TEAssign (n, _, value) ->
        (n = name && (mode = Diverged || varying vi env value))
        || scan mode env value
    | TEIf (cond, t, el) -> (
        let inner = if varying vi env cond then Diverged else mode in
        scan mode env cond || scan inner env t
        || match el with Some e -> scan inner env e | None -> false)
    | TEWhile (cond, b) ->
        let inner = if varying vi env cond then Diverged else mode in
        scan mode env cond || scan inner env b
    | TEFor (_, _, lo, hi, _, b) ->
        let inner =
          if varying vi env lo || varying vi env hi then Diverged else mode
        in
        scan mode env lo || scan mode env hi || scan inner env b
    | TEMatch (s, cases) ->
        let inner = if varying vi env s then Diverged else mode in
        scan mode env s || List.exists (fun (_, e) -> scan inner env e) cases
    | TESeq es ->
        let rec go mode = function
          | [] -> false
          | e :: rest ->
              scan mode env e
              || go
                   (if has_varying_return_with vi env e then Diverged else mode)
                   rest
        in
        go mode es
    | TELet (n, _, value, b) ->
        scan mode env value
        || scan
             mode
             (if varying vi env value then StringSet.add n env else env)
             b
    | TELetMut (n, _, value, b) ->
        scan mode env value
        || n <> name
           && scan
                mode
                (if varying vi env value || assigned_varying vi env n b then
                   StringSet.add n env
                 else env)
                b
    | _ -> List.exists (scan mode env) (sub_exprs e)
  in
  scan Converged env body

(** True if [e] contains a return reachable under a thread-varying condition
    (some threads exit early). *)
and has_varying_return_with vi (env : StringSet.t) (e : texpr) : bool =
  match e.te with
  | TEIf (cond, t, else_opt) -> (
      if varying vi env cond then
        has_reachable_return t
        ||
        match else_opt with
        | Some e -> has_reachable_return e
        | None -> false
      else
        has_varying_return_with vi env t
        ||
        match else_opt with
        | Some e -> has_varying_return_with vi env e
        | None -> false)
  | TESeq es -> List.exists (has_varying_return_with vi env) es
  | TELet (name, _, v, b) | TELetMut (name, _, v, b) ->
      let env' = if varying vi env v then StringSet.add name env else env in
      has_varying_return_with vi env v || has_varying_return_with vi env' b
  | TEMatch (scrut, cases) ->
      if varying vi env scrut then
        List.exists (fun (_, e) -> has_reachable_return e) cases
      else List.exists (fun (_, e) -> has_varying_return_with vi env e) cases
  | TEWhile (cond, b) ->
      has_varying_return_with vi env cond || has_varying_return_with vi env b
  | TEFor (_, _, lo, hi, _, b) ->
      has_varying_return_with vi env lo
      || has_varying_return_with vi env hi
      || has_varying_return_with vi env b
  | _ -> false

let is_thread_varying_env = varying legacy_variance

(** True if e contains a TEReturn that is reachable under a thread-varying
    condition. Such a return causes SOME threads (those that take the varying
    branch) to exit early, while other threads continue, creating an implicit
    divergence point at every subsequent barrier in the enclosing sequence. *)
let has_varying_return = has_varying_return_with legacy_variance

(** Backward-compatible wrapper using an empty environment *)
let is_thread_varying = is_thread_varying_env StringSet.empty

(** Check if an intrinsic reference is a barrier *)
let is_barrier_intrinsic (ref : intrinsic_ref) : bool = is_barrier_ref ref

(** Check if an intrinsic ref requires warp convergence *)
let is_warp_convergence_ref (ref : intrinsic_ref) : bool =
  match ref with
  | IntrinsicRef (_, base_name) ->
      Sarek_core_primitives.is_warp_convergence_point base_name
  | CorePrimitiveRef name ->
      Sarek_core_primitives.is_warp_convergence_point name

(** The first synchronisation point [te] contains: a barrier, a superstep, a
    warp collective, or a call to a helper of [helpers] (which contains one). *)
let rec first_sync ~(helpers : (string * sync_kind) list) (te : texpr) :
    sync_kind option =
  match te.te with
  | TESuperstep _ -> Some Barrier_sync
  | TEIntrinsicFun (_, Some Sarek_core_primitives.ConvergencePoint, _) ->
      Some Barrier_sync
  | TEIntrinsicFun (ref, Some Sarek_core_primitives.WarpConvergence, _) ->
      Some (Warp_sync (Sarek_env.intrinsic_ref_display_name ref))
  | TEApp ({te = TEVar (name, _) | TEGlobalRef (name, _); _}, _)
    when List.mem_assoc name helpers ->
      List.assoc_opt name helpers
  | _ -> List.find_map (first_sync ~helpers) (sub_exprs te)

(** The error a call reaching [kind] gives in diverged flow. *)
let sync_error (kind : sync_kind) loc : Sarek_error.error =
  match kind with
  | Barrier_sync -> Sarek_error.Barrier_in_diverged_flow loc
  | Warp_sync name -> Sarek_error.Warp_collective_in_diverged_flow (name, loc)

let is_sync_error : Sarek_error.error -> bool = function
  | Sarek_error.Warp_collective_in_diverged_flow _
  | Sarek_error.Barrier_in_diverged_flow _ ->
      true
  | _ -> false

(** Collect errors from convergence analysis *)
let rec check_expr ctx (te : texpr) : Sarek_error.error list =
  match te.te with
  (* Intrinsic function calls - check convergence requirement from AST *)
  | TEIntrinsicFun (ref, convergence, args) ->
      let arg_errors = List.concat_map (check_expr ctx) args in
      if ctx.mode = Diverged then
        match convergence with
        | Some Sarek_core_primitives.ConvergencePoint ->
            Sarek_error.Barrier_in_diverged_flow te.te_loc :: arg_errors
        | Some Sarek_core_primitives.WarpConvergence ->
            let name = Sarek_env.intrinsic_ref_display_name ref in
            Sarek_error.Warp_collective_in_diverged_flow (name, te.te_loc)
            :: arg_errors
        | Some Sarek_core_primitives.NoEffect | None -> arg_errors
      else arg_errors
  (* If expression - diverge if condition is thread-varying *)
  | TEIf (cond, then_e, else_opt) ->
      let cond_errors = check_expr ctx cond in
      let inner_ctx =
        if varying ctx.variance ctx.varying_vars cond then diverge ctx else ctx
      in
      let then_errors = check_expr inner_ctx then_e in
      let else_errors =
        match else_opt with
        | None -> []
        | Some else_e -> check_expr inner_ctx else_e
      in
      cond_errors @ then_errors @ else_errors
  (* While loop - diverge if condition is thread-varying *)
  | TEWhile (cond, body) ->
      let cond_varying = varying ctx.variance ctx.varying_vars cond in
      (* A thread-varying condition is re-evaluated at every iteration by the
         threads still looping only, so a synchronisation point inside it runs in
         diverged flow (strict analysis only). *)
      let cond_ctx =
        if ctx.variance.strict && cond_varying then diverge ctx else ctx
      in
      let cond_errors = check_expr cond_ctx cond in
      let inner_ctx = if cond_varying then diverge ctx else ctx in
      let body_errors = check_expr inner_ctx body in
      cond_errors @ body_errors
  (* For loop - check if bounds are thread-varying *)
  | TEFor (_, _, lo, hi, _, body) ->
      let lo_errors = check_expr ctx lo in
      let hi_errors = check_expr ctx hi in
      let inner_ctx =
        if
          varying ctx.variance ctx.varying_vars lo
          || varying ctx.variance ctx.varying_vars hi
        then diverge ctx
        else ctx
      in
      let body_errors = check_expr inner_ctx body in
      lo_errors @ hi_errors @ body_errors
  (* Match - diverge if scrutinee is thread-varying *)
  | TEMatch (scrut, cases) ->
      let scrut_errors = check_expr ctx scrut in
      let inner_ctx =
        if varying ctx.variance ctx.varying_vars scrut then diverge ctx else ctx
      in
      let case_errors =
        List.concat_map (fun (_, e) -> check_expr inner_ctx e) cases
      in
      scrut_errors @ case_errors
  (* Recursively check subexpressions - these don't change mode *)
  | TEUnit | TEBool _ | TEInt _ | TEInt32 _ | TEInt64 _ | TEFloat _ | TEDouble _
  | TEVar _ | TEGlobalRef _ | TENative _ | TEIntrinsicConst _ ->
      []
  | TEVecGet (v, i) -> check_expr ctx v @ check_expr ctx i
  | TEVecSet (v, i, x) -> check_expr ctx v @ check_expr ctx i @ check_expr ctx x
  | TEArrGet (a, i) -> check_expr ctx a @ check_expr ctx i
  | TEArrSet (a, i, x) -> check_expr ctx a @ check_expr ctx i @ check_expr ctx x
  | TEFieldGet (e, _, _) -> check_expr ctx e
  | TEFieldSet (e, _, _, x) -> check_expr ctx e @ check_expr ctx x
  | TEBinop ((Sarek_ast.And | Sarek_ast.Or), a, b)
    when ctx.variance.strict && varying ctx.variance ctx.varying_vars a ->
      (* The right operand is evaluated only by some threads (short-circuit). *)
      check_expr ctx a @ check_expr (diverge ctx) b
  | TEBinop (_, a, b) -> check_expr ctx a @ check_expr ctx b
  | TEUnop (_, e) -> check_expr ctx e
  | TEApp (f, args) ->
      (* H-2: a call reached in diverged flow to a helper that contains a
         synchronisation point (transitively) is rejected at the call. *)
      let call_errors =
        match f.te with
        | (TEVar (name, _) | TEGlobalRef (name, _)) when ctx.mode = Diverged
          -> (
            match List.assoc_opt name ctx.sync_helpers with
            | Some kind -> [sync_error kind te.te_loc]
            | None -> [])
        | _ -> []
      in
      call_errors
      @ check_helper_call ctx f args
      @ check_expr ctx f
      @ List.concat_map (check_expr ctx) args
  | TEAssign (_, _, e) -> check_expr ctx e
  | TELet (name, _, value, body) ->
      let value_errors = check_expr ctx value in
      let ctx' =
        if varying ctx.variance ctx.varying_vars value then
          {ctx with varying_vars = StringSet.add name ctx.varying_vars}
        else ctx
      in
      let ctx' = {ctx' with variance = alias_def ctx'.variance name value} in
      (* [let g = helper]: the alias is as synchronising as the helper. *)
      let ctx' =
        match
          Option.bind (helper_name value) (fun h ->
              List.assoc_opt h ctx'.sync_helpers)
        with
        | Some kind ->
            {ctx' with sync_helpers = (name, kind) :: ctx'.sync_helpers}
        | None -> ctx'
      in
      value_errors @ check_expr ctx' body
  | TELetMut (name, _, value, body) ->
      let value_errors = check_expr ctx value in
      let is_varying =
        varying ctx.variance ctx.varying_vars value
        || ctx.variance.strict
           && assigned_varying ctx.variance ctx.varying_vars name body
      in
      let ctx' =
        if is_varying then
          {ctx with varying_vars = StringSet.add name ctx.varying_vars}
        else ctx
      in
      value_errors @ check_expr ctx' body
  | TESeq es ->
      (* F-04 fix: track whether a preceding statement can produce a varying
         early return. If so, some threads have exited; subsequent barriers are
         reached only by threads that fell through, making them divergence
         points. *)
      let rec check_seq ctx = function
        | [] -> []
        | e :: rest ->
            let errors = check_expr ctx e in
            let ctx' =
              if has_varying_return_with ctx.variance ctx.varying_vars e then
                diverge ctx
              else ctx
            in
            errors @ check_seq ctx' rest
      in
      check_seq ctx es
  | TERecord (_, fields) ->
      List.concat_map (fun (_, e) -> check_expr ctx e) fields
  | TEConstr (_, _, arg) -> (
      match arg with None -> [] | Some e -> check_expr ctx e)
  | TETuple es -> List.concat_map (check_expr ctx) es
  | TEReturn e -> check_expr ctx e
  | TECreateArray (size, _, _) -> check_expr ctx size
  | TEPragma (_, body) -> check_expr ctx body
  (* BSP constructs *)
  | TELetShared (_, _, _, size_opt, body) ->
      let size_errors =
        match size_opt with None -> [] | Some size -> check_expr ctx size
      in
      size_errors @ check_expr ctx body
  | TESuperstep (_, _divergent, step_body, cont) ->
      (* specs/convergence-semantics.md §2. The body runs, then an implicit
         block barrier, then the continuation.
         SS-1: a superstep reached in Diverged mode is an error, with or
         without [@divergent] (SS-4: the attribute is inert; skipping this
         check was unsound, C-7).
         SS-2/SS-3: body and continuation are checked in the outer mode.
         Structured control flow inside the body reconverges at its own end,
         before the implicit barrier, so a varying [if] with no
         synchronisation point in its branches is accepted.
         SS-5: the known-varying variable set is kept. *)
      let entry_errors =
        if ctx.mode = Diverged then
          [Sarek_error.Barrier_in_diverged_flow te.te_loc]
        else []
      in
      entry_errors @ check_expr ctx step_body @ check_expr ctx cont
  | TEOpen (_, body) -> check_expr ctx body
  | TELetRec (name, _, params, fn_body, cont) ->
      let ctx' =
        match first_sync ~helpers:ctx.sync_helpers fn_body with
        | Some kind ->
            {ctx with sync_helpers = (name, kind) :: ctx.sync_helpers}
        | None ->
            (* a later helper of the same name shadows an earlier one *)
            {ctx with sync_helpers = List.remove_assoc name ctx.sync_helpers}
      in
      let ctx' =
        {
          ctx' with
          variance = add_def ctx.variance ctx.varying_vars name params fn_body;
        }
      in
      check_expr ctx fn_body @ check_expr ctx' cont

(** Strict analysis: a call to a helper whose body is checked with the
    parameters that the call makes thread-varying. The definition itself is
    checked with uniform parameters, so a shuffle under [if param > k] is only
    seen here. Only synchronisation errors are reported; a helper already being
    re-checked is not entered again. *)
and check_helper_call ctx (f : texpr) (args : texpr list) :
    Sarek_error.error list =
  let strict_call =
    ctx.variance.strict && ctx.mode = Converged
    && List.length ctx.call_stack < 8
  in
  match (strict_call, helper_name f) with
  | true, Some name when not (List.mem name ctx.call_stack) -> (
      match List.assoc_opt name ctx.variance.defs with
      | None -> []
      | Some def ->
          let flags = List.map (varying ctx.variance ctx.varying_vars) args in
          if not (List.mem true flags) then []
          else
            let ctx_body =
              {
                ctx with
                varying_vars = helper_call_env def flags;
                variance = {ctx.variance with defs = body_defs def};
                call_stack = name :: ctx.call_stack;
              }
            in
            List.filter is_sync_error (check_expr ctx_body def.hd_body))
  | _ -> []

(** Check a module item *)
let check_module_item ctx (item : tmodule_item) : Sarek_error.error list =
  match item with
  | TMConst (_, _, _, e) -> check_expr ctx e
  | TMFun (_, _, _, e) -> check_expr ctx e

(** Run [check] over the module items and the body of [kernel] with [ctx]. *)
let check_kernel_with ctx (kernel : tkernel) : Sarek_error.error list =
  let item_errors =
    List.concat_map (check_module_item ctx) kernel.tkern_module_items
  in
  item_errors @ check_expr ctx kernel.tkern_body

(** The synchronisation errors only the strict analysis sees in [legacy]'s blind
    spots: [strict] errors that [legacy] did not already report. *)
let add_strict_sync_errors ~legacy ~strict =
  let extra =
    List.fold_left
      (fun acc e ->
        if is_sync_error e && not (List.mem e legacy || List.mem e acc) then
          e :: acc
        else acc)
      []
      strict
  in
  legacy @ List.rev extra

(** Does [te] contain a synchronisation point (barrier, superstep or warp
    collective)? The strict analysis is skipped for a kernel without one (it
    only reports synchronisation errors). *)
let rec has_sync_point (te : texpr) : bool =
  match te.te with
  | TESuperstep _
  | TEIntrinsicFun
      ( _,
        Some
          ( Sarek_core_primitives.ConvergencePoint
          | Sarek_core_primitives.WarpConvergence ),
        _ ) ->
      true
  | _ -> List.exists has_sync_point (sub_exprs te)

(** Check an expression as a whole kernel body: the legacy analysis, then the
    strict analysis for synchronisation points. *)
let check_toplevel_expr (te : texpr) : Sarek_error.error list =
  Hashtbl.reset assigned_memo ;
  add_strict_sync_errors
    ~legacy:(check_expr init_ctx te)
    ~strict:
      (check_expr
         {init_ctx with variance = {legacy_variance with strict = true}}
         te)

(** Check a kernel for convergence safety *)
let check_kernel (kernel : tkernel) : (unit, Sarek_error.error list) result =
  Hashtbl.reset assigned_memo ;
  (* Module functions that contain a synchronisation point, in definition order
     so a later one sees an earlier one. *)
  let sync_helpers =
    List.fold_left
      (fun helpers (item : tmodule_item) ->
        match item with
        | TMFun (name, _, _, body) -> (
            match first_sync ~helpers body with
            | Some kind -> (name, kind) :: helpers
            | None -> helpers)
        | TMConst _ -> helpers)
      []
      kernel.tkern_module_items
  in
  (* Module functions as strict-analysis helpers; they capture nothing. *)
  let defs =
    List.fold_left
      (fun scope (item : tmodule_item) ->
        match item with
        | TMFun (name, _, params, body) ->
            let def =
              {
                hd_self = name;
                hd_params = List.map (fun p -> p.tparam_name) params;
                hd_body = body;
                hd_defs = scope;
                hd_captured = StringSet.empty;
              }
            in
            (name, def) :: List.remove_assoc name scope
        | TMConst _ -> scope)
      []
      kernel.tkern_module_items
  in
  let ctx = {init_ctx with sync_helpers} in
  let strict_ctx = {ctx with variance = {strict = true; defs; visiting = []}} in
  let all_errors =
    add_strict_sync_errors
      ~legacy:(check_kernel_with ctx kernel)
      ~strict:
        (if
           has_sync_point kernel.tkern_body
           || List.exists
                (function
                  | TMFun (_, _, _, b) | TMConst (_, _, _, b) ->
                      has_sync_point b)
                kernel.tkern_module_items
         then check_kernel_with strict_ctx kernel
         else [])
  in
  if all_errors = [] then Ok () else Error all_errors

(** Check if a kernel uses any barriers (explicit or implicit). Used for
    compile-time optimization of native kernel execution. *)
let rec expr_uses_barriers (te : texpr) : bool =
  match te.te with
  (* Superstep has implicit barrier *)
  | TESuperstep _ -> true
  (* Intrinsic function - check if it's a convergence point *)
  | TEIntrinsicFun (_, Some Sarek_core_primitives.ConvergencePoint, _) -> true
  (* A warp collective is simulated on the CPU by exchanging through the block's
     shared memory between two barriers (Sarek_cpu_runtime.Warp), so a
     kernel that calls one is a barrier kernel: it needs the full thread state
     and the BSP runner, not the simple loops. *)
  | TEIntrinsicFun (_, Some Sarek_core_primitives.WarpConvergence, _) -> true
  | TEIntrinsicFun (_, _, args) -> List.exists expr_uses_barriers args
  (* Recursively check subexpressions *)
  | TEIf (cond, then_e, else_opt) ->
      expr_uses_barriers cond || expr_uses_barriers then_e
      || Option.is_some
           (Option.bind else_opt (fun e ->
                if expr_uses_barriers e then Some () else None))
  | TEWhile (cond, body) -> expr_uses_barriers cond || expr_uses_barriers body
  | TEFor (_, _, lo, hi, _, body) ->
      expr_uses_barriers lo || expr_uses_barriers hi || expr_uses_barriers body
  | TEMatch (scrut, cases) ->
      expr_uses_barriers scrut
      || List.exists (fun (_, e) -> expr_uses_barriers e) cases
  | TELet (_, _, v, b) | TELetMut (_, _, v, b) ->
      expr_uses_barriers v || expr_uses_barriers b
  | TESeq es -> List.exists expr_uses_barriers es
  | TEBinop (_, a, b) -> expr_uses_barriers a || expr_uses_barriers b
  | TEUnop (_, e) -> expr_uses_barriers e
  | TEApp (f, args) ->
      expr_uses_barriers f || List.exists expr_uses_barriers args
  | TEVecGet (v, i) -> expr_uses_barriers v || expr_uses_barriers i
  | TEVecSet (v, i, x) ->
      expr_uses_barriers v || expr_uses_barriers i || expr_uses_barriers x
  | TEArrGet (a, i) -> expr_uses_barriers a || expr_uses_barriers i
  | TEArrSet (a, i, x) ->
      expr_uses_barriers a || expr_uses_barriers i || expr_uses_barriers x
  | TEFieldGet (e, _, _) -> expr_uses_barriers e
  | TEFieldSet (e, _, _, x) -> expr_uses_barriers e || expr_uses_barriers x
  | TERecord (_, fields) ->
      List.exists (fun (_, e) -> expr_uses_barriers e) fields
  | TETuple es -> List.exists expr_uses_barriers es
  | TEReturn e -> expr_uses_barriers e
  | TEPragma (_, body) -> expr_uses_barriers body
  | TEAssign (_, _, e) -> expr_uses_barriers e
  | TECreateArray (size, _, _) -> expr_uses_barriers size
  | TEConstr (_, _, arg) ->
      Option.is_some
        (Option.bind arg (fun e ->
             if expr_uses_barriers e then Some () else None))
  | TELetShared (_, _, _, size_opt, body) ->
      (match size_opt with None -> false | Some s -> expr_uses_barriers s)
      || expr_uses_barriers body
  | TEOpen (_, body) -> expr_uses_barriers body
  | TELetRec (_, _, _, fn_body, cont) ->
      expr_uses_barriers fn_body || expr_uses_barriers cont
  (* Terminal cases - no barriers *)
  | TEUnit | TEBool _ | TEInt _ | TEInt32 _ | TEInt64 _ | TEFloat _ | TEDouble _
  | TEVar _ | TEGlobalRef _ | TENative _ | TEIntrinsicConst _ ->
      false

(** Check if a kernel uses barriers *)
let kernel_uses_barriers (kernel : tkernel) : bool =
  (* Check module items *)
  let item_uses =
    List.exists
      (fun (item : tmodule_item) ->
        match item with
        | TMConst (_, _, _, e) -> expr_uses_barriers e
        | TMFun (_, _, _, e) -> expr_uses_barriers e)
      kernel.tkern_module_items
  in
  item_uses || expr_uses_barriers kernel.tkern_body

(** {1 Kernel Dimensionality Analysis}

    Detect which thread/block dimensions a kernel uses to enable optimized
    native CPU execution. Simple kernels that only use global_idx_x can be run
    with a simple parallel for loop without the thread_state overhead. *)

(** Dimension usage record *)
type dim_usage = {
  uses_x : bool;
  uses_y : bool;
  uses_z : bool;
  uses_block_dim : bool;  (** Uses block_dim_x/y/z *)
  uses_grid_dim : bool;  (** Uses grid_dim_x/y/z *)
  uses_thread_idx : bool;  (** Uses thread_idx_x/y/z directly *)
  uses_block_idx : bool;  (** Uses block_idx_x/y/z directly *)
  uses_shared_mem : bool;  (** Uses shared memory *)
}

let empty_dim_usage =
  {
    uses_x = false;
    uses_y = false;
    uses_z = false;
    uses_block_dim = false;
    uses_grid_dim = false;
    uses_thread_idx = false;
    uses_block_idx = false;
    uses_shared_mem = false;
  }

let merge_dim_usage a b =
  {
    uses_x = a.uses_x || b.uses_x;
    uses_y = a.uses_y || b.uses_y;
    uses_z = a.uses_z || b.uses_z;
    uses_block_dim = a.uses_block_dim || b.uses_block_dim;
    uses_grid_dim = a.uses_grid_dim || b.uses_grid_dim;
    uses_thread_idx = a.uses_thread_idx || b.uses_thread_idx;
    uses_block_idx = a.uses_block_idx || b.uses_block_idx;
    uses_shared_mem = a.uses_shared_mem || b.uses_shared_mem;
  }

(** Check if an intrinsic name affects dimension usage *)
let dim_usage_of_name name =
  match name with
  | "global_idx" | "global_idx_x" | "global_size" | "global_size_x"
  | "global_thread_id" ->
      {empty_dim_usage with uses_x = true}
  | "global_idx_y" | "global_size_y" ->
      {empty_dim_usage with uses_x = true; uses_y = true}
  | "global_idx_z" | "global_size_z" ->
      {empty_dim_usage with uses_x = true; uses_y = true; uses_z = true}
  | "thread_idx_x" ->
      {empty_dim_usage with uses_x = true; uses_thread_idx = true}
  | "thread_idx_y" ->
      {empty_dim_usage with uses_y = true; uses_thread_idx = true}
  | "thread_idx_z" ->
      {empty_dim_usage with uses_z = true; uses_thread_idx = true}
  | "block_idx_x" -> {empty_dim_usage with uses_x = true; uses_block_idx = true}
  | "block_idx_y" -> {empty_dim_usage with uses_y = true; uses_block_idx = true}
  | "block_idx_z" -> {empty_dim_usage with uses_z = true; uses_block_idx = true}
  (* lane_id and warp_size are read from the full thread state (the simulated
     warp width and the linear thread id), never from the simple loops. *)
  | "lane_id" | "warp_size" ->
      {
        empty_dim_usage with
        uses_x = true;
        uses_y = true;
        uses_z = true;
        uses_thread_idx = true;
        uses_block_dim = true;
      }
  | "block_dim_x" -> {empty_dim_usage with uses_x = true; uses_block_dim = true}
  | "block_dim_y" -> {empty_dim_usage with uses_y = true; uses_block_dim = true}
  | "block_dim_z" -> {empty_dim_usage with uses_z = true; uses_block_dim = true}
  | "grid_dim_x" -> {empty_dim_usage with uses_x = true; uses_grid_dim = true}
  | "grid_dim_y" -> {empty_dim_usage with uses_y = true; uses_grid_dim = true}
  | "grid_dim_z" -> {empty_dim_usage with uses_z = true; uses_grid_dim = true}
  | _ -> empty_dim_usage

(** Get dimension usage from an intrinsic reference *)
let dim_usage_of_intrinsic_ref (ref : intrinsic_ref) =
  match ref with
  | IntrinsicRef (_, name) -> dim_usage_of_name name
  | CorePrimitiveRef name -> dim_usage_of_name name

(** Analyze expression for dimension usage *)
let rec expr_dim_usage (te : texpr) : dim_usage =
  match te.te with
  (* Variable - might be a thread index *)
  | TEVar (name, _) -> dim_usage_of_name name
  (* Intrinsic constant - check for thread/block/grid *)
  | TEIntrinsicConst ref -> dim_usage_of_intrinsic_ref ref
  (* Intrinsic function - check name and args *)
  | TEIntrinsicFun (ref, _, args) ->
      let ref_usage = dim_usage_of_intrinsic_ref ref in
      let args_usage =
        List.fold_left
          merge_dim_usage
          empty_dim_usage
          (List.map expr_dim_usage args)
      in
      merge_dim_usage ref_usage args_usage
  (* Shared memory allocation *)
  | TELetShared (_, _, _, size_opt, body) ->
      let size_usage =
        match size_opt with
        | None -> empty_dim_usage
        | Some e -> expr_dim_usage e
      in
      let body_usage = expr_dim_usage body in
      merge_dim_usage {size_usage with uses_shared_mem = true} body_usage
  (* Recursive cases *)
  | TEBinop (_, a, b) -> merge_dim_usage (expr_dim_usage a) (expr_dim_usage b)
  | TEUnop (_, e) -> expr_dim_usage e
  | TEIf (c, t, e_opt) -> (
      let usage = merge_dim_usage (expr_dim_usage c) (expr_dim_usage t) in
      match e_opt with
      | None -> usage
      | Some e -> merge_dim_usage usage (expr_dim_usage e))
  | TEWhile (c, b) -> merge_dim_usage (expr_dim_usage c) (expr_dim_usage b)
  | TEFor (_, _, lo, hi, _, body) ->
      merge_dim_usage
        (expr_dim_usage lo)
        (merge_dim_usage (expr_dim_usage hi) (expr_dim_usage body))
  | TELet (_, _, v, b) | TELetMut (_, _, v, b) ->
      merge_dim_usage (expr_dim_usage v) (expr_dim_usage b)
  | TESeq es ->
      List.fold_left
        merge_dim_usage
        empty_dim_usage
        (List.map expr_dim_usage es)
  | TEApp (f, args) ->
      List.fold_left
        merge_dim_usage
        (expr_dim_usage f)
        (List.map expr_dim_usage args)
  | TEVecGet (v, i) | TEArrGet (v, i) ->
      merge_dim_usage (expr_dim_usage v) (expr_dim_usage i)
  | TEVecSet (v, i, x) | TEArrSet (v, i, x) ->
      merge_dim_usage
        (expr_dim_usage v)
        (merge_dim_usage (expr_dim_usage i) (expr_dim_usage x))
  | TEFieldGet (e, _, _) -> expr_dim_usage e
  | TEFieldSet (e, _, _, x) ->
      merge_dim_usage (expr_dim_usage e) (expr_dim_usage x)
  | TEAssign (_, _, e) -> expr_dim_usage e
  | TERecord (_, fields) ->
      List.fold_left
        merge_dim_usage
        empty_dim_usage
        (List.map (fun (_, e) -> expr_dim_usage e) fields)
  | TETuple es ->
      List.fold_left
        merge_dim_usage
        empty_dim_usage
        (List.map expr_dim_usage es)
  | TEMatch (s, cases) ->
      List.fold_left
        merge_dim_usage
        (expr_dim_usage s)
        (List.map (fun (_, e) -> expr_dim_usage e) cases)
  | TEConstr (_, _, arg) -> (
      match arg with None -> empty_dim_usage | Some e -> expr_dim_usage e)
  | TEReturn e -> expr_dim_usage e
  | TECreateArray (size, _, _) -> expr_dim_usage size
  | TEPragma (_, body) -> expr_dim_usage body
  | TESuperstep (_, _, step, cont) ->
      merge_dim_usage (expr_dim_usage step) (expr_dim_usage cont)
  | TEOpen (_, body) -> expr_dim_usage body
  | TELetRec (_, _, _, fn_body, cont) ->
      merge_dim_usage (expr_dim_usage fn_body) (expr_dim_usage cont)
  (* Terminal cases - no dimension usage *)
  | TEUnit | TEBool _ | TEInt _ | TEInt32 _ | TEInt64 _ | TEFloat _ | TEDouble _
  | TEGlobalRef _ | TENative _ ->
      empty_dim_usage

(** Kernel execution strategy based on dimension analysis *)
type exec_strategy =
  | Simple1D  (** Only uses global_idx_x - can use simple parallel for *)
  | Simple2D  (** Only uses global_idx_x/y - can use nested loops *)
  | Simple3D  (** Uses all three dimensions with simple loops *)
  | FullState
      (** Uses block/thread indices or shared memory - needs full state *)

(** Determine the optimal execution strategy for a kernel *)
let kernel_exec_strategy (kernel : tkernel) : exec_strategy =
  (* First check if kernel uses barriers - if so, need full state for barrier function *)
  if kernel_uses_barriers kernel then FullState
  else begin
    (* Analyze module items *)
    let items_usage =
      List.fold_left
        merge_dim_usage
        empty_dim_usage
        (List.map
           (fun (item : tmodule_item) ->
             match item with
             | TMConst (_, _, _, e) -> expr_dim_usage e
             | TMFun (_, _, _, e) -> expr_dim_usage e)
           kernel.tkern_module_items)
    in
    (* Analyze kernel body *)
    let body_usage = expr_dim_usage kernel.tkern_body in
    let usage = merge_dim_usage items_usage body_usage in

    (* If uses block/thread indices directly, or shared memory, need full state *)
    if
      usage.uses_thread_idx || usage.uses_block_idx || usage.uses_block_dim
      || usage.uses_grid_dim || usage.uses_shared_mem
    then FullState
      (* Otherwise, can use simplified loops based on dimensions used *)
    else if usage.uses_z then Simple3D
    else if usage.uses_y then Simple2D
    else if usage.uses_x then Simple1D
    else
      (* No dimensions used at all - still need to run at least once per global thread *)
      Simple1D
  end
