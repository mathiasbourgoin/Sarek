(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Sarek IR - Vector-parameter helper inlining (GLSL / WGSL)
 *
 * On GLSL and WGSL a storage buffer cannot be passed as a function argument, so
 * a helper function taking a [TVec] parameter cannot be emitted as a real
 * device function: the earlier code stripped the vector parameter from the
 * signature and left the body referencing an identifier that no longer existed
 * (GLSL), while the WGSL call site still forwarded the stripped buffer argument
 * (arity mismatch). Both produced invalid shaders that no pass validated.
 *
 * This pass inlines every helper that takes a vector parameter at its call
 * sites, then drops those helpers from the kernel. It is the IR-level analogue
 * of the PTX EApp inliner (Sarek_ir_ptx_expr: bind params, splice the body,
 * recursion guard). PTX can inline directly inside its expression emitter
 * because that emitter is register/statement-based and returns a binding; the
 * GLSL/WGSL emitters are purely textual (gen_expr cannot emit statements
 * mid-expression), so the faithful analogue is an IR-to-IR splice performed
 * before textual generation. Only GLSL and WGSL use this pass — CUDA, OpenCL,
 * Metal pass buffers as real pointer parameters, and PTX inlines in its own
 * emitter.
 *
 * Scope / limitations (rejected with a located [Unsupported_construct]):
 * - a vector argument must be a plain vector variable (a global buffer or an
 *   enclosing vector parameter); an arbitrary expression cannot alias a buffer;
 * - the helper body's [return]s must be in tail position (the shape the
 *   tail-recursion elimination pass produces); a mid-body early return cannot
 *   be spliced into straight-line code and is refused;
 * - a residual self / mutually-recursive call to a vector-parameter helper is
 *   refused (depth-bounded non-tail recursion on these backends is out of
 *   scope; tail recursion is already a loop by the time it reaches codegen);
 * - a vector-parameter helper returning an aggregate (record/variant) is
 *   refused (no default initializer synthesised for the result temporary).
 ******************************************************************************)

open Sarek_ir_types

(** Does [hf] take at least one vector parameter? Only such helpers are inlined;
    scalar-only helpers stay real device functions on both backends. *)
let has_vec_param (hf : helper_func) =
  List.exists
    (fun (v : var) -> match v.var_type with TVec _ -> true | _ -> false)
    hf.hf_params

(** Where the result of an inlined call must go. *)
type sink =
  | Assign of lvalue  (** store the returned value into [lvalue] *)
  | Return  (** [return] the value from the enclosing function *)
  | Discard  (** unit-typed call kept only for its side effects *)

(** Inliner state: the vector-parameter helpers keyed by name, the backend name
    for error messages, and the stack of helpers currently being inlined (the
    recursion guard). *)
type ctx = {
  vec_helpers : (string, helper_func) Hashtbl.t;
  backend : string;
  mutable stack : string list;
  mutable temp_counter : int;
}

let fail ctx construct reason =
  Sarek_backend_error.Backend_error.raise_error
    (Sarek_backend_error.Backend_error.unsupported_construct
       ~backend:ctx.backend
       construct
       reason)

let fresh_temp_name ctx =
  ctx.temp_counter <- ctx.temp_counter + 1 ;
  Printf.sprintf "_sarek_inl_%d" ctx.temp_counter

(** Fresh name for a helper local that must be alpha-renamed to avoid capturing
    a substituted-in buffer reference. [sarek_]-prefixed (a reserved,
    collision-proof namespace on every backend). *)
let fresh_local_name ctx =
  ctx.temp_counter <- ctx.temp_counter + 1 ;
  Printf.sprintf "sarek_inl_local_%d" ctx.temp_counter

(** Is [e] a call to a vector-parameter helper? Returns its name + args. *)
let as_vec_call ctx = function
  | EApp (EVar f, args) when Hashtbl.mem ctx.vec_helpers f.var_name ->
      Some (f.var_name, args)
  | _ -> None

(* ------------------------------------------------------------------ *)
(* Vector-parameter name substitution                                  *)
(* ------------------------------------------------------------------ *)

(** [subst] maps a name to its replacement (identity if absent). Two uses:
    - vector-parameter substitution: maps a helper's vector-parameter name to
      the call site's buffer name, so every buffer access ([EArrayRead],
      [EArrayLen], [LArrayElem], and any bare [EVar]) reads the caller's global
      buffer directly. Here [~rename_binders:false] — the vector parameter has
      no binder in the body, and a shadowing local of the same name must NOT be
      touched.
    - alpha-renaming: maps a helper local that would collide with a
      substituted-in buffer name to a fresh name. Here [~rename_binders:true] so
      the binder occurrences ([SLet]/[SLetMut]/[SFor] variables and match
      pattern bindings) are rewritten too, not only the uses. *)
let sub subst name =
  match List.assoc_opt name subst with Some n -> n | None -> name

let sub_pattern ~rename_binders subst = function
  | PConstr (cname, bindings) when rename_binders ->
      PConstr (cname, List.map (sub subst) bindings)
  | p -> p

let rec subst_expr ~rename_binders subst e =
  let se = subst_expr ~rename_binders subst in
  match e with
  | EConst _ -> e
  | EVar v -> EVar {v with var_name = sub subst v.var_name}
  | EBinop (op, a, b) -> EBinop (op, se a, se b)
  | EUnop (op, a) -> EUnop (op, se a)
  | EArrayRead (name, idx) -> EArrayRead (sub subst name, se idx)
  | EArrayReadExpr (base, idx) -> EArrayReadExpr (se base, se idx)
  | ERecordField (a, f) -> ERecordField (se a, f)
  | EIntrinsic (path, name, args) -> EIntrinsic (path, name, List.map se args)
  | ECast (ty, a) -> ECast (ty, se a)
  | ETuple es -> ETuple (List.map se es)
  | EApp (fn, args) -> EApp (se fn, List.map se args)
  | ERecord (name, fields) ->
      ERecord (name, List.map (fun (f, x) -> (f, se x)) fields)
  | EVariant (t, c, args) -> EVariant (t, c, List.map se args)
  | EArrayLen name -> EArrayLen (sub subst name)
  | EArrayCreate (ty, sz, ms) -> EArrayCreate (ty, se sz, ms)
  | EIf (c, t, e2) -> EIf (se c, se t, se e2)
  | EMatch (s, cases) ->
      EMatch
        ( se s,
          List.map
            (fun (p, b) -> (sub_pattern ~rename_binders subst p, se b))
            cases )

and subst_lvalue ~rename_binders subst lv =
  let se = subst_expr ~rename_binders subst in
  match lv with
  | LVar v -> LVar {v with var_name = sub subst v.var_name}
  | LArrayElem (name, idx) -> LArrayElem (sub subst name, se idx)
  | LArrayElemExpr (base, idx) -> LArrayElemExpr (se base, se idx)
  | LRecordField (lv, f) ->
      LRecordField (subst_lvalue ~rename_binders subst lv, f)

(** Rewrite a binder's variable, renaming it only when [~rename_binders]. *)
let sub_binder ~rename_binders subst (v : var) =
  if rename_binders then {v with var_name = sub subst v.var_name} else v

let rec subst_stmt ~rename_binders subst s =
  let se = subst_expr ~rename_binders subst in
  let ss = subst_stmt ~rename_binders subst in
  let sb v = sub_binder ~rename_binders subst v in
  match s with
  | SAssign (lv, e) -> SAssign (subst_lvalue ~rename_binders subst lv, se e)
  | SSeq stmts -> SSeq (List.map ss stmts)
  | SIf (c, t, e) -> SIf (se c, ss t, Option.map ss e)
  | SWhile (c, b) -> SWhile (se c, ss b)
  | SFor (v, lo, hi, dir, b) -> SFor (sb v, se lo, se hi, dir, ss b)
  | SMatch (e, cases) ->
      SMatch
        ( se e,
          List.map
            (fun (p, b) -> (sub_pattern ~rename_binders subst p, ss b))
            cases )
  | SReturn e -> SReturn (se e)
  | (SBarrier | SWarpBarrier | SMemFence | SEmpty | SNative _) as s -> s
  | SExpr e -> SExpr (se e)
  | SLet (v, e, b) -> SLet (sb v, se e, ss b)
  | SLetMut (v, e, b) -> SLetMut (sb v, se e, ss b)
  | SPragma (h, b) -> SPragma (h, ss b)
  | SBlock b -> SBlock (ss b)

(* ------------------------------------------------------------------ *)
(* Binder collection (for alpha-renaming)                              *)
(* ------------------------------------------------------------------ *)

(** All local binder names introduced anywhere in [s]: [SLet]/[SLetMut]/[SFor]
    variables and match-pattern bindings. Used to detect names that would
    capture a substituted-in buffer reference. *)
let collect_binders s =
  let acc = ref [] in
  let add n = acc := n :: !acc in
  let pat = function
    | PConstr (_, bindings) -> List.iter add bindings
    | PWild -> ()
  in
  let rec go = function
    | SLet (v, _, b) | SLetMut (v, _, b) ->
        add v.var_name ;
        go b
    | SFor (v, _, _, _, b) ->
        add v.var_name ;
        go b
    | SSeq stmts -> List.iter go stmts
    | SIf (_, t, e) ->
        go t ;
        Option.iter go e
    | SWhile (_, b) | SPragma (_, b) | SBlock b -> go b
    | SMatch (_, cases) ->
        List.iter
          (fun (p, b) ->
            pat p ;
            go b)
          cases
    | SAssign _ | SReturn _ | SExpr _ | SBarrier | SWarpBarrier | SMemFence
    | SEmpty | SNative _ ->
        ()
  in
  go s ;
  !acc

(* ------------------------------------------------------------------ *)
(* Return rewriting (tail-position only)                               *)
(* ------------------------------------------------------------------ *)

(** Replace each (tail-position) [SReturn e] in the spliced helper body with the
    action dictated by [sink]. A [return] found in a non-tail position (e.g. the
    middle of a sequence) is refused — it would fall through into the code after
    the splice, changing semantics. *)
let rec rewrite_returns ctx sink s =
  let recurse = rewrite_returns ctx sink in
  match s with
  | SReturn e -> (
      match sink with
      | Assign lv -> SAssign (lv, e)
      | Return -> SReturn e
      | Discard -> ( match e with EConst CUnit -> SEmpty | _ -> SExpr e))
  | SSeq [] -> SSeq []
  | SSeq ss ->
      let rev = List.rev ss in
      let last = List.hd rev in
      let init = List.rev (List.tl rev) in
      List.iter (assert_no_return ctx) init ;
      SSeq (init @ [recurse last])
  | SIf (c, t, e) -> SIf (c, recurse t, Option.map recurse e)
  | SMatch (e, cases) ->
      SMatch (e, List.map (fun (p, b) -> (p, recurse b)) cases)
  | SLet (v, e, b) -> SLet (v, e, recurse b)
  | SLetMut (v, e, b) -> SLetMut (v, e, recurse b)
  | SPragma (h, b) -> SPragma (h, recurse b)
  | SBlock b -> SBlock (recurse b)
  | SWhile _ | SFor _ ->
      (* A tail-position loop cannot carry the function's return value. *)
      assert_no_return ctx s ;
      s
  | ( SAssign _ | SExpr _ | SBarrier | SWarpBarrier | SMemFence | SEmpty
    | SNative _ ) as s ->
      s

(** Verify [s] contains no [SReturn] anywhere (used for non-tail sub-statements
    of an inlined body). *)
and assert_no_return ctx s =
  let rec chk = function
    | SReturn _ ->
        fail
          ctx
          "recursion+vector helper"
          "helper has an early (non-tail) return that cannot be inlined on \
           this backend; rewrite it so every return is in tail position"
    | SSeq ss -> List.iter chk ss
    | SIf (_, t, e) ->
        chk t ;
        Option.iter chk e
    | SWhile (_, b) | SFor (_, _, _, _, b) | SBlock b | SPragma (_, b) -> chk b
    | SMatch (_, cases) -> List.iter (fun (_, b) -> chk b) cases
    | SLet (_, _, b) | SLetMut (_, _, b) -> chk b
    | SAssign _ | SExpr _ | SBarrier | SWarpBarrier | SMemFence | SEmpty
    | SNative _ ->
        ()
  in
  chk s

(* ------------------------------------------------------------------ *)
(* Default initializer for a result temporary                          *)
(* ------------------------------------------------------------------ *)

let default_expr ctx = function
  | TInt32 -> EConst (CInt32 0l)
  | TInt64 -> EConst (CInt64 0L)
  | TFloat32 -> EConst (CFloat32 0.0)
  | TFloat64 -> EConst (CFloat64 0.0)
  | TBool -> EConst (CBool false)
  | TUnit -> EConst CUnit
  | (TRecord _ | TVariant _ | TArray _ | TVec _) as t ->
      fail
        ctx
        "recursion+vector helper"
        (Printf.sprintf
           "cannot inline a vector-parameter helper returning %s (no default \
            initializer for the result temporary)"
           (match t with
           | TRecord (n, _) -> "record " ^ n
           | TVariant (n, _) -> "variant " ^ n
           | TArray _ -> "an array"
           | _ -> "a vector"))

(* ------------------------------------------------------------------ *)
(* Core: splice one call                                               *)
(* ------------------------------------------------------------------ *)

(** Build the statement that computes [f args] and routes the result to [sink].
    Scalar parameters are bound with [SLet]; vector parameters are substituted
    by name; the whole body is wrapped in an [SBlock] so its locals never leak
    into (or collide across) call sites. The spliced body is itself run through
    [inline_stmt] so a vector helper that calls another vector helper is fully
    resolved. *)
let rec splice_call ctx sink (fname : string) (args : expr list) : stmt =
  if List.mem fname ctx.stack then
    fail
      ctx
      "recursion+vector helper"
      (Printf.sprintf
         "helper '%s' is (mutually) recursive through a vector parameter; \
          depth-bounded non-tail recursion is not supported on this backend"
         fname) ;
  let hf = Hashtbl.find ctx.vec_helpers fname in
  if List.length args <> List.length hf.hf_params then
    fail
      ctx
      "recursion+vector helper"
      (Printf.sprintf
         "helper '%s' called with %d arguments, expects %d"
         fname
         (List.length args)
         (List.length hf.hf_params)) ;
  (* Partition parameters into vector substitutions and scalar bindings. *)
  let subst, scalar_binds =
    List.fold_right2
      (fun (p : var) arg (subst, binds) ->
        match p.var_type with
        | TVec _ -> (
            match arg with
            | EVar buf -> ((p.var_name, buf.var_name) :: subst, binds)
            | _ ->
                fail
                  ctx
                  "recursion+vector helper"
                  (Printf.sprintf
                     "vector argument to helper '%s' (parameter '%s') must be \
                      a vector variable, not an arbitrary expression"
                     fname
                     p.var_name))
        | _ -> (subst, (p, arg) :: binds))
      hf.hf_params
      args
      ([], [])
  in
  (* Alpha-capture avoidance. A helper local (or scalar-parameter) whose name
     collides with a name involved in the buffer substitution — either the
     buffer we substitute IN (a value) or the vector parameter we substitute
     AWAY (a key) — would, once spliced into the caller's block, shadow the
     global buffer and make substituted [buf[i]] accesses read the local
     instead. Rename every such colliding binder to a fresh [sarek_]-prefixed
     name (binder AND uses) BEFORE the buffer substitution runs. *)
  let dangerous = List.map fst subst @ List.map snd subst in
  let scalar_param_names = List.map (fun (p, _) -> p.var_name) scalar_binds in
  let rename =
    List.sort_uniq compare (collect_binders hf.hf_body @ scalar_param_names)
    |> List.filter_map (fun name ->
        if List.mem name dangerous then Some (name, fresh_local_name ctx)
        else None)
  in
  let body = subst_stmt ~rename_binders:true rename hf.hf_body in
  let scalar_binds =
    List.map
      (fun (p, arg) -> ({p with var_name = sub rename p.var_name}, arg))
      scalar_binds
  in
  (* Substitute buffer names, then route returns to the sink. *)
  let body = subst_stmt ~rename_binders:false subst body in
  let body = rewrite_returns ctx sink body in
  (* Bind scalar parameters as lets wrapping the body. *)
  let bound =
    List.fold_right (fun (p, arg) acc -> SLet (p, arg, acc)) scalar_binds body
  in
  (* Recursively inline nested vector-helper calls inside this body. *)
  ctx.stack <- fname :: ctx.stack ;
  let bound = inline_stmt ctx bound in
  ctx.stack <- List.tl ctx.stack ;
  SBlock bound

(* ------------------------------------------------------------------ *)
(* Statement / expression rewriting driving the splice                 *)
(* ------------------------------------------------------------------ *)

(** Lift every vector-helper call nested inside expression [e] into a preceding
    result temporary, so it can be spliced as a statement. Returns the list of
    (temp var, helper name, args) hoisted (in evaluation order) and the
    rewritten expression referring to the temporaries. Direct-position calls
    (handled by [inline_stmt]) are not reached here. *)
and hoist_expr ctx e =
  let hoisted = ref [] in
  let rec go e =
    match as_vec_call ctx e with
    | Some (fname, args) ->
        let args = List.map go args in
        let hf = Hashtbl.find ctx.vec_helpers fname in
        let tmp =
          {
            var_name = fresh_temp_name ctx;
            var_id = 0;
            var_type = hf.hf_ret_type;
            var_mutable = true;
          }
        in
        hoisted := (tmp, fname, args) :: !hoisted ;
        EVar tmp
    | None -> (
        match e with
        | EConst _ | EVar _ | EArrayLen _ -> e
        | EBinop (op, a, b) -> EBinop (op, go a, go b)
        | EUnop (op, a) -> EUnop (op, go a)
        | EArrayRead (n, i) -> EArrayRead (n, go i)
        | EArrayReadExpr (b, i) -> EArrayReadExpr (go b, go i)
        | ERecordField (a, f) -> ERecordField (go a, f)
        | EIntrinsic (p, n, args) -> EIntrinsic (p, n, List.map go args)
        | ECast (ty, a) -> ECast (ty, go a)
        | ETuple es -> ETuple (List.map go es)
        | EApp (fn, args) -> EApp (fn, List.map go args)
        | ERecord (n, fs) -> ERecord (n, List.map (fun (f, x) -> (f, go x)) fs)
        | EVariant (t, c, args) -> EVariant (t, c, List.map go args)
        | EArrayCreate (ty, sz, ms) -> EArrayCreate (ty, go sz, ms)
        | EIf (c, t, e2) -> EIf (go c, go t, go e2)
        | EMatch (s, cases) ->
            EMatch (go s, List.map (fun (p, b) -> (p, go b)) cases))
  in
  let e' = go e in
  (List.rev !hoisted, e')

(** Wrap [core] (a statement using the hoisted temporaries) with a mutable
    result temporary and an inlined splice for each hoisted call, in order. *)
and with_hoisted ctx hoisted core =
  List.fold_right
    (fun (tmp, fname, args) acc ->
      SLetMut
        ( tmp,
          default_expr ctx tmp.var_type,
          SSeq [splice_call ctx (Assign (LVar tmp)) fname args; acc] ))
    hoisted
    core

(** Rewrite one statement, inlining vector-helper calls. Direct-position calls
    (the whole RHS of an assignment / let / return / expression statement) are
    spliced without a temporary; calls nested inside other expressions are
    hoisted first. *)
and inline_stmt ctx s : stmt =
  match s with
  (* Direct-position calls: splice straight into the sink. *)
  | SAssign (lv, e) when as_vec_call ctx e <> None ->
      let fname, args = Option.get (as_vec_call ctx e) in
      splice_call ctx (Assign lv) fname args
  | SReturn e when as_vec_call ctx e <> None ->
      let fname, args = Option.get (as_vec_call ctx e) in
      splice_call ctx Return fname args
  | SExpr e when as_vec_call ctx e <> None ->
      let fname, args = Option.get (as_vec_call ctx e) in
      splice_call ctx Discard fname args
  | SLet (v, e, body) when as_vec_call ctx e <> None ->
      let fname, args = Option.get (as_vec_call ctx e) in
      SLetMut
        ( v,
          default_expr ctx v.var_type,
          SSeq
            [splice_call ctx (Assign (LVar v)) fname args; inline_stmt ctx body]
        )
  | SLetMut (v, e, body) when as_vec_call ctx e <> None ->
      let fname, args = Option.get (as_vec_call ctx e) in
      SLetMut
        ( v,
          default_expr ctx v.var_type,
          SSeq
            [splice_call ctx (Assign (LVar v)) fname args; inline_stmt ctx body]
        )
  (* Otherwise: hoist nested calls out of the statement's expressions, then
     recurse structurally. *)
  | SAssign (lv, e) ->
      let h, e' = hoist_expr ctx e in
      with_hoisted ctx h (SAssign (lv, e'))
  | SReturn e ->
      let h, e' = hoist_expr ctx e in
      with_hoisted ctx h (SReturn e')
  | SExpr e ->
      let h, e' = hoist_expr ctx e in
      with_hoisted ctx h (SExpr e')
  | SLet (v, e, body) ->
      let h, e' = hoist_expr ctx e in
      with_hoisted ctx h (SLet (v, e', inline_stmt ctx body))
  | SLetMut (v, e, body) ->
      let h, e' = hoist_expr ctx e in
      with_hoisted ctx h (SLetMut (v, e', inline_stmt ctx body))
  | SIf (c, t, e) ->
      let h, c' = hoist_expr ctx c in
      with_hoisted
        ctx
        h
        (SIf (c', inline_stmt ctx t, Option.map (inline_stmt ctx) e))
  | SWhile (c, b) ->
      let h, c' = hoist_expr ctx c in
      (* A hoisted call in the condition is evaluated once before the loop; a
         call in the loop condition proper is uncommon and would need
         re-evaluation each iteration — refuse rather than change semantics. *)
      if h <> [] then
        fail
          ctx
          "recursion+vector helper"
          "a vector-helper call in a while-condition cannot be inlined safely" ;
      SWhile (c', inline_stmt ctx b)
  | SFor (v, lo, hi, dir, b) ->
      let h1, lo' = hoist_expr ctx lo in
      let h2, hi' = hoist_expr ctx hi in
      with_hoisted ctx (h1 @ h2) (SFor (v, lo', hi', dir, inline_stmt ctx b))
  | SMatch (e, cases) ->
      let h, e' = hoist_expr ctx e in
      with_hoisted
        ctx
        h
        (SMatch (e', List.map (fun (p, b) -> (p, inline_stmt ctx b)) cases))
  | SSeq ss -> SSeq (List.map (inline_stmt ctx) ss)
  | SBlock b -> SBlock (inline_stmt ctx b)
  | SPragma (hints, b) -> SPragma (hints, inline_stmt ctx b)
  | (SBarrier | SWarpBarrier | SMemFence | SEmpty | SNative _) as s -> s

(* ------------------------------------------------------------------ *)
(* Entry point                                                         *)
(* ------------------------------------------------------------------ *)

(** Inline all vector-parameter helpers into the kernel body and into the
    remaining (scalar-only) helper bodies, then drop the inlined helpers. A
    no-op when the kernel has no vector-parameter helper. *)
let inline_vec_helpers ~backend (k : kernel) : kernel =
  let vec_helpers = Hashtbl.create 8 in
  List.iter
    (fun hf ->
      if has_vec_param hf then Hashtbl.replace vec_helpers hf.hf_name hf)
    k.kern_funcs ;
  if Hashtbl.length vec_helpers = 0 then k
  else begin
    let ctx = {vec_helpers; backend; stack = []; temp_counter = 0} in
    let kern_body = inline_stmt ctx k.kern_body in
    (* Keep only scalar-only helpers; inline vector-helper calls inside them. *)
    let kern_funcs =
      List.filter_map
        (fun hf ->
          if has_vec_param hf then None
          else Some {hf with hf_body = inline_stmt ctx hf.hf_body})
        k.kern_funcs
    in
    {k with kern_body; kern_funcs}
  end
