(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Capture-avoiding uniquification of every binder in a kernel IR — the "binder
    canary" half of the OpenCL validation gate (#128).

    {1 Why}

    A compile gate on generated OpenCL C catches the shape where a dropped
    binder leaves an identifier with no declaration:

    {v input.cl:30:52: error: use of undeclared identifier 'q' v}

    It provably cannot catch the shape where the dropped binder's name happens
    to match something else in scope. Binder names come from user source, so any
    colliding in-scope name — a parameter, a [let], a loop index — turns that
    build error into VALID OpenCL C that computes the wrong answer with no
    diagnostic at any layer. Measured on an RX 7900 XTX (rusticl/radeonsi): a
    dropped [EMatch] payload binder [r] resolved to an enclosing local [r], the
    program built clean, and every one of 1024 elements was wrong.

    {1 What this does}

    Rename every BINDING occurrence in the kernel to a globally unique name
    ([sk<n>_<original>]) and rewrite each reference to its innermost binder.
    This is a semantics-preserving α-conversion — the generated source must
    still compile and behave identically. But collisions are now impossible by
    construction: two binders that used to share the name [r] become [sk3_r] and
    [sk7_r]. A binder the emitter drops therefore always leaves an identifier
    that nothing declares, converting the whole class into the one shape the
    compile gate does catch.

    So: generate once from the kernel as written (checks what we ship), and once
    from its uniquified twin (checks that no binder was silently dropped). Both
    go through the same compiler. *)

open Sarek_ir_types
module SMap = Map.Make (String)

type env = string SMap.t

let counter = ref 0

let reset () = counter := 0

(** Fresh name for a binding occurrence. The [sk<n>_] prefix is unique across
    the whole kernel and is a valid C identifier, so a dropped binder surfaces
    as [use of undeclared identifier 'sk7_r'] — unmistakable in the compiler
    diagnostic and impossible to satisfy by accident. *)
let fresh (base : string) : string =
  incr counter ;
  Printf.sprintf "sk%d_%s" !counter base

(** References to names this pass never bound (kernel-external arrays, if any)
    are left alone: renaming them would create a false positive, and leaving
    them cannot create one — they were already resolving to whatever they
    resolved to before. *)
let look (env : env) (n : string) : string =
  match SMap.find_opt n env with Some m -> m | None -> n

let rename_var env (v : var) : var = {v with var_name = look env v.var_name}

let bind env (v : var) : env * var =
  let n = fresh v.var_name in
  (SMap.add v.var_name n env, {v with var_name = n})

let rec uexpr (env : env) (e : expr) : expr =
  let r = uexpr env in
  match e with
  | EConst _ -> e
  | EVar v -> EVar (rename_var env v)
  | EBinop (op, a, b) -> EBinop (op, r a, r b)
  | EUnop (op, a) -> EUnop (op, r a)
  | EArrayRead (arr, i) -> EArrayRead (look env arr, r i)
  | EArrayReadExpr (b, i) -> EArrayReadExpr (r b, r i)
  | ERecordField (a, f) -> ERecordField (r a, f)
  | EIntrinsic (p, n, args) -> EIntrinsic (p, n, List.map r args)
  | ECast (t, a) -> ECast (t, r a)
  | ETuple es -> ETuple (List.map r es)
  (* An [EApp] head names a helper function, not a local binder: helpers are
     renamed as a group by [uniquify_kernel] so call and definition stay in
     step. *)
  | EApp (fn, args) -> EApp (uexpr env fn, List.map r args)
  | ERecord (n, fs) -> ERecord (n, List.map (fun (f, a) -> (f, r a)) fs)
  | EVariant (n, c, args) -> EVariant (n, c, List.map r args)
  | EArrayLen a -> EArrayLen (look env a)
  | EArrayCreate (t, sz, ms) -> EArrayCreate (t, r sz, ms)
  | EIf (c, a, b) -> EIf (r c, r a, r b)
  | EMatch (s, cases) ->
      EMatch (r s, List.map (fun (p, a) -> ucase_expr env p a) cases)

(** A [PConstr] payload binds its variables in the arm only. This is exactly the
    binder class the #75 defect dropped, so it is the one that most needs a name
    no enclosing scope can supply. *)
and ucase_expr env p a =
  match p with
  | PWild -> (PWild, uexpr env a)
  | PConstr (c, vars) ->
      let env, vars =
        List.fold_left
          (fun (env, acc) v ->
            let n = fresh v in
            (SMap.add v n env, n :: acc))
          (env, [])
          vars
      in
      (PConstr (c, List.rev vars), uexpr env a)

let rec ulvalue env (lv : lvalue) : lvalue =
  match lv with
  | LVar v -> LVar (rename_var env v)
  | LArrayElem (a, i) -> LArrayElem (look env a, uexpr env i)
  | LArrayElemExpr (b, i) -> LArrayElemExpr (uexpr env b, uexpr env i)
  | LRecordField (b, f) -> LRecordField (ulvalue env b, f)

let rec ustmt (env : env) (s : stmt) : stmt =
  let e = uexpr env and r = ustmt env in
  match s with
  | SBarrier | SWarpBarrier | SEmpty | SMemFence -> s
  (* [SNative] carries an opaque source-generating closure whose identifiers
     this pass cannot see, so renaming around it would be unsound. Kernels
     containing one are refused by [uniquify_kernel] instead. *)
  | SNative _ -> s
  | SAssign (lv, x) -> SAssign (ulvalue env lv, e x)
  | SSeq ss -> SSeq (List.map r ss)
  | SIf (c, t, el) -> SIf (e c, r t, Option.map r el)
  | SWhile (c, b) -> SWhile (e c, r b)
  | SFor (v, lo, hi, d, b) ->
      (* Bounds are evaluated in the enclosing scope; the index binds the body. *)
      let lo = e lo and hi = e hi in
      let env', v' = bind env v in
      SFor (v', lo, hi, d, ustmt env' b)
  | SMatch (sc, cases) ->
      SMatch (e sc, List.map (fun (p, b) -> ucase_stmt env p b) cases)
  | SReturn x -> SReturn (e x)
  | SExpr x -> SExpr (e x)
  | SLet (v, x, b) ->
      let x = e x in
      let env', v' = bind env v in
      SLet (v', x, ustmt env' b)
  | SLetMut (v, x, b) ->
      let x = e x in
      let env', v' = bind env v in
      SLetMut (v', x, ustmt env' b)
  | SPragma (h, b) -> SPragma (h, r b)
  | SBlock b -> SBlock (r b)
  | SCoopmat op ->
      (* Mirrors [Sarek_ir_codegen.rename_shadowing_locals]: fragment names are
         not [var]s, are never in [env], and must not be looked up there — only
         the index and stride expressions can mention a renamed variable. *)
      SCoopmat
        (match op with
        | CM_decl _ -> op
        | CM_load req ->
            CM_load {req with index = e req.index; stride = e req.stride}
        | CM_store req ->
            CM_store {req with index = e req.index; stride = e req.stride}
        | CM_muladd _ -> op)

and ucase_stmt env p b =
  match p with
  | PWild -> (PWild, ustmt env b)
  | PConstr (c, vars) ->
      let env, vars =
        List.fold_left
          (fun (env, acc) v ->
            let n = fresh v in
            (SMap.add v n env, n :: acc))
          (env, [])
          vars
      in
      (PConstr (c, List.rev vars), ustmt env b)

let bind_decl (env : env) (d : decl) : env * decl =
  match d with
  | DParam (v, ai) ->
      let env, v = bind env v in
      (env, DParam (v, ai))
  | DLocal (v, init) ->
      (* The initialiser is in the enclosing scope, so rename it BEFORE binding. *)
      let init = Option.map (uexpr env) init in
      let env, v = bind env v in
      (env, DLocal (v, init))
  | DShared (n, t, sz) ->
      let sz = Option.map (uexpr env) sz in
      let n' = fresh n in
      (SMap.add n n' env, DShared (n', t, sz))

exception Unsupported of string

let rec stmt_has_native = function
  | SNative _ -> true
  | SSeq ss -> List.exists stmt_has_native ss
  | SIf (_, t, el) -> (
      stmt_has_native t
      || match el with Some e -> stmt_has_native e | None -> false)
  | SWhile (_, b) | SFor (_, _, _, _, b) | SPragma (_, b) | SBlock b ->
      stmt_has_native b
  | SMatch (_, cases) -> List.exists (fun (_, b) -> stmt_has_native b) cases
  | SLet (_, _, b) | SLetMut (_, _, b) -> stmt_has_native b
  | _ -> false

(** α-convert every binder in [k]. Raises {!Unsupported} for kernels this pass
    cannot rename soundly, so the caller reports a SKIP with a reason rather
    than validating a kernel it silently left alone. *)
let uniquify_kernel (k : kernel) : kernel =
  if
    stmt_has_native k.kern_body
    || List.exists (fun hf -> stmt_has_native hf.hf_body) k.kern_funcs
  then raise (Unsupported "kernel contains SNative (opaque generated source)") ;
  reset () ;
  (* Helpers are renamed as a group so definitions and call sites agree. Their
     parameters bind only inside their own bodies — helpers are module-level and
     capture nothing, so each starts from the helper-name environment alone. *)
  let henv =
    List.fold_left
      (fun env hf -> SMap.add hf.hf_name (fresh hf.hf_name) env)
      SMap.empty
      k.kern_funcs
  in
  let funcs =
    List.map
      (fun hf ->
        let env, params =
          List.fold_left
            (fun (env, acc) v ->
              let env, v = bind env v in
              (env, v :: acc))
            (henv, [])
            hf.hf_params
        in
        {
          hf with
          hf_name = look henv hf.hf_name;
          hf_params = List.rev params;
          hf_body = ustmt env hf.hf_body;
        })
      k.kern_funcs
  in
  let env, params =
    List.fold_left
      (fun (env, acc) d ->
        let env, d = bind_decl env d in
        (env, d :: acc))
      (henv, [])
      k.kern_params
  in
  let env, locals =
    List.fold_left
      (fun (env, acc) d ->
        let env, d = bind_decl env d in
        (env, d :: acc))
      (env, [])
      k.kern_locals
  in
  {
    k with
    kern_params = List.rev params;
    kern_locals = List.rev locals;
    kern_funcs = funcs;
    kern_body = ustmt env k.kern_body;
  }
