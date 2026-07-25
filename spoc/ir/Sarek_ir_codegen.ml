(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Sarek_ir_codegen - Shared code-generation helpers for GPU backends
 *
 * Hosts logic that was previously duplicated verbatim across the CUDA, OpenCL,
 * Metal, and Vulkan IR generators: type-name mangling and variant type emission.
 ******************************************************************************)

(** Mangle OCaml type name to valid C/GLSL identifier (e.g., "Module.point" ->
    "Module_point") *)
let mangle_name name = String.map (fun c -> if c = '.' then '_' else c) name

(** Alpha-rename kernel-body binders whose name collides with a backend-reserved
    identifier, so a colliding local no longer aliases a param/macro.

    Backends expose scalar params to the body through a mechanism that keys on
    the {e name} (a GLSL [#define] macro, or WGSL's [gen_expr] check against the
    global [scalar_param_names]); a local binder sharing that name is silently
    captured — the GLSL macro rewrites the declared identifier to [pc....] (a
    syntax error), and WGSL redirects every read to the uniform. This pass
    rewrites each colliding binder (and its in-scope references) to a fresh name
    that the exposing mechanism never touches. Initializers, evaluated in the
    outer scope, still expand to the param, so semantics are preserved.

    Covered binder forms: [SLet], [SLetMut], [SFor], and match pattern binders
    (both [SMatch] and [EMatch]). It is a no-op unless a binder genuinely
    shadows — collision-free kernels are byte-identical.

    Backend-specific inputs, and only these, are parameters:
    - [collides name]: whether [name] (as written in the IR, before any escape)
      is a reserved identifier for this backend. GLSL escapes then checks its
      scalar-macro and vector-[_len]-macro sets; WGSL checks the raw
      scalar-param set (it has no [_len] macro).
    - [fresh_name orig n]: mint the [n]-th shadow name for original binder
      [orig] ([n] is 1-based, incremented once per colliding binder). GLSL uses
      [sarek_pc_shadow_<esc>_<n>], WGSL uses [sarek_scalar_shadow_<esc>_<n>].

    The per-call counter starts at 0 and is threaded internally, so each
    invocation (one per kernel) numbers shadows from 1 — matching the backends'
    former reset-then-generate discipline exactly. *)
let rename_shadowing_locals ~collides ~fresh_name body =
  let open Sarek_ir_types in
  let module SM = Map.Make (String) in
  let counter = ref 0 in
  let mint orig =
    incr counter ;
    fresh_name orig !counter
  in
  let ren env name =
    match SM.find_opt name env with Some n -> n | None -> name
  in
  (* Rebind a match pattern's binders: rename each that collides with a reserved
     identifier to a fresh name — used both in the destructuring declaration
     emitted by [gen_match_pattern] and in case-body references (via [env]) — and
     drop any same-named outer mapping for non-colliding binders (they shadow
     it). *)
  let bind_pattern env = function
    | PWild -> (PWild, env)
    | PConstr (cname, names) ->
        let env, names =
          List.fold_left_map
            (fun env name ->
              if collides name then
                let nn = mint name in
                (SM.add name nn env, nn)
              else (SM.remove name env, name))
            env
            names
        in
        (PConstr (cname, names), env)
  in
  let rec re_expr env e =
    match e with
    | EConst _ -> e
    | EVar v -> EVar {v with var_name = ren env v.var_name}
    | EBinop (op, a, b) -> EBinop (op, re_expr env a, re_expr env b)
    | EUnop (op, a) -> EUnop (op, re_expr env a)
    | EArrayRead (arr, i) -> EArrayRead (ren env arr, re_expr env i)
    | EArrayReadExpr (b, i) -> EArrayReadExpr (re_expr env b, re_expr env i)
    | ERecordField (e, f) -> ERecordField (re_expr env e, f)
    | EIntrinsic (ns, n, args) -> EIntrinsic (ns, n, List.map (re_expr env) args)
    | ECast (t, e) -> ECast (t, re_expr env e)
    | ETuple es -> ETuple (List.map (re_expr env) es)
    | EApp (f, args) -> EApp (re_expr env f, List.map (re_expr env) args)
    | ERecord (n, fs) ->
        ERecord (n, List.map (fun (k, v) -> (k, re_expr env v)) fs)
    | EVariant (t, c, args) -> EVariant (t, c, List.map (re_expr env) args)
    | EArrayLen n -> EArrayLen (ren env n)
    | EArrayCreate (t, s, m) -> EArrayCreate (t, re_expr env s, m)
    | EIf (c, t, e) -> EIf (re_expr env c, re_expr env t, re_expr env e)
    | EMatch (s, cases) ->
        (* Rebind pattern binders before each case body (mirrors [SMatch]): a
           binder shadowing a reserved identifier is renamed and its body refs
           follow, otherwise an outer mapping would wrongly substitute them. *)
        EMatch
          ( re_expr env s,
            List.map
              (fun (p, b) ->
                let p, env = bind_pattern env p in
                (p, re_expr env b))
              cases )
  in
  let rec re_lvalue env lv =
    match lv with
    | LVar v -> LVar {v with var_name = ren env v.var_name}
    | LArrayElem (arr, i) -> LArrayElem (ren env arr, re_expr env i)
    | LArrayElemExpr (b, i) -> LArrayElemExpr (re_expr env b, re_expr env i)
    | LRecordField (lv, f) -> LRecordField (re_lvalue env lv, f)
  in
  (* Bind a local: rename it when it collides with a reserved identifier;
     otherwise drop any same-named outer mapping (this fresh local shadows it). *)
  let bind env (v : var) =
    if collides v.var_name then
      let nn = mint v.var_name in
      ({v with var_name = nn}, SM.add v.var_name nn env)
    else (v, SM.remove v.var_name env)
  in
  let rec re_stmt env s =
    match s with
    | SAssign (lv, e) -> SAssign (re_lvalue env lv, re_expr env e)
    | SSeq ss -> SSeq (List.map (re_stmt env) ss)
    | SIf (c, t, eo) ->
        SIf (re_expr env c, re_stmt env t, Option.map (re_stmt env) eo)
    | SWhile (c, b) -> SWhile (re_expr env c, re_stmt env b)
    | SFor (v, lo, hi, dir, b) ->
        let lo = re_expr env lo and hi = re_expr env hi in
        let v', env' = bind env v in
        SFor (v', lo, hi, dir, re_stmt env' b)
    | SMatch (e, cases) ->
        SMatch
          ( re_expr env e,
            List.map
              (fun (p, b) ->
                let p, env = bind_pattern env p in
                (p, re_stmt env b))
              cases )
    | SReturn e -> SReturn (re_expr env e)
    | (SBarrier | SWarpBarrier | SMemFence | SEmpty) as s -> s
    | SExpr e -> SExpr (re_expr env e)
    | SLet (v, e, body) ->
        let e = re_expr env e in
        let v', env' = bind env v in
        SLet (v', e, re_stmt env' body)
    | SLetMut (v, e, body) ->
        let e = re_expr env e in
        let v', env' = bind env v in
        SLetMut (v', e, re_stmt env' body)
    | SPragma (ss, b) -> SPragma (ss, re_stmt env b)
    | SBlock b -> SBlock (re_stmt env b)
    | SNative _ as s -> s
  in
  re_stmt SM.empty body

(** Emit a C/MSL tagged-union variant type (enum + typedef struct + union +
    inline constructors). [type_of_elttype] and [constructor_prefix] are the
    only backend-specific inputs. *)
let gen_variant_def ~type_of_elttype ~constructor_prefix buf (name, constrs) =
  let mangled = mangle_name name in
  (* Enum for tags - use simple names for switch case labels *)
  Buffer.add_string buf "enum { " ;
  List.iteri
    (fun i (cname, _) ->
      if i > 0 then Buffer.add_string buf ", " ;
      Buffer.add_string buf cname ;
      Buffer.add_string buf " = " ;
      Buffer.add_string buf (string_of_int i))
    constrs ;
  Buffer.add_string buf " };\n" ;
  (* Struct with tag and union *)
  Buffer.add_string buf "typedef struct {\n  int tag;\n" ;
  (* Generate union if any constructor has payload *)
  let has_payload = List.exists (fun (_, args) -> args <> []) constrs in
  if has_payload then begin
    Buffer.add_string buf "  union {\n" ;
    List.iter
      (fun (cname, args) ->
        match args with
        | [] -> () (* No payload for this constructor *)
        | [ty] ->
            Buffer.add_string buf "    " ;
            Buffer.add_string buf (type_of_elttype ty) ;
            Buffer.add_string buf (" " ^ cname ^ "_v;\n")
        | _ ->
            (* Multiple args - generate struct *)
            Buffer.add_string buf "    struct { " ;
            List.iteri
              (fun i ty ->
                if i > 0 then Buffer.add_string buf " " ;
                Buffer.add_string buf (type_of_elttype ty) ;
                Buffer.add_string buf (Printf.sprintf " _%d;" i))
              args ;
            Buffer.add_string buf (" } " ^ cname ^ "_v;\n"))
      constrs ;
    Buffer.add_string buf "  } data;\n"
  end ;
  Buffer.add_string buf ("} " ^ mangled ^ ";\n\n") ;
  (* Constructor functions *)
  List.iteri
    (fun _i (cname, args) ->
      Buffer.add_string
        buf
        (constructor_prefix ^ " " ^ mangled ^ " make_" ^ mangled ^ "_" ^ cname
       ^ "(") ;
      (match args with
      | [] -> ()
      | [ty] ->
          Buffer.add_string buf (type_of_elttype ty) ;
          Buffer.add_string buf " v"
      | _ ->
          List.iteri
            (fun j ty ->
              if j > 0 then Buffer.add_string buf ", " ;
              Buffer.add_string buf (type_of_elttype ty) ;
              Buffer.add_string buf (Printf.sprintf " v%d" j))
            args) ;
      Buffer.add_string buf (") {\n  " ^ mangled ^ " r;\n") ;
      Buffer.add_string buf ("  r.tag = " ^ cname ^ ";\n") ;
      (match args with
      | [] -> ()
      | [_] -> Buffer.add_string buf ("  r.data." ^ cname ^ "_v = v;\n")
      | _ ->
          List.iteri
            (fun j _ ->
              Buffer.add_string
                buf
                (Printf.sprintf "  r.data.%s_v._%d = v%d;\n" cname j j))
            args) ;
      Buffer.add_string buf "  return r;\n}\n\n")
    constrs

(** Emit a GLSL variant type. GLSL lacks enum/typedef/union, so tags are
    const-int declarations, the type is a bare struct with flat payload fields,
    and constructors have no qualifier prefix. *)
let gen_variant_def_glsl ~type_of_elttype buf (name, constrs) =
  let mangled = mangle_name name in
  (* Enum constants *)
  List.iteri
    (fun i (cname, _) ->
      Buffer.add_string buf (Printf.sprintf "const int %s = %d;\n" cname i))
    constrs ;
  Buffer.add_char buf '\n' ;
  (* GLSL forbids anonymous nested struct definitions inside a struct, so a
     multi-field (flattened tuple) payload is hoisted to a named struct type
     [<mangled>_<cname>_payload { T0 _0; T1 _1; ... }] declared before the
     variant struct; the member and all accesses keep the [<cname>_v._N] shape
     shared with the other backends. *)
  let payload_struct_name cname =
    Printf.sprintf "%s_%s_payload" mangled cname
  in
  List.iter
    (fun (cname, args) ->
      match args with
      | [] | [_] -> ()
      | _ ->
          Buffer.add_string
            buf
            (Printf.sprintf "struct %s {" (payload_struct_name cname)) ;
          List.iteri
            (fun i ty ->
              if i > 0 then Buffer.add_string buf " " ;
              Buffer.add_string
                buf
                (Printf.sprintf " %s _%d;" (type_of_elttype ty) i))
            args ;
          Buffer.add_string buf " };\n")
    constrs ;
  (* Struct with tag and union-like data *)
  Buffer.add_string buf (Printf.sprintf "struct %s {\n  int tag;\n" mangled) ;
  let has_payload = List.exists (fun (_, args) -> args <> []) constrs in
  if has_payload then begin
    (* GLSL doesn't have unions, so we use the largest payload type *)
    List.iter
      (fun (cname, args) ->
        match args with
        | [] -> ()
        | [ty] ->
            Buffer.add_string
              buf
              (Printf.sprintf "  %s %s_v;\n" (type_of_elttype ty) cname)
        | _ ->
            Buffer.add_string
              buf
              (Printf.sprintf "  %s %s_v;\n" (payload_struct_name cname) cname))
      constrs
  end ;
  Buffer.add_string buf "};\n\n" ;
  (* Constructor functions *)
  List.iteri
    (fun _i (cname, args) ->
      Buffer.add_string
        buf
        (Printf.sprintf "%s make_%s_%s(" mangled mangled cname) ;
      (match args with
      | [] -> ()
      | [ty] ->
          Buffer.add_string buf (type_of_elttype ty) ;
          Buffer.add_string buf " v"
      | _ ->
          List.iteri
            (fun j ty ->
              if j > 0 then Buffer.add_string buf ", " ;
              Buffer.add_string buf (type_of_elttype ty) ;
              Buffer.add_string buf (Printf.sprintf " v%d" j))
            args) ;
      Buffer.add_string buf ") {\n" ;
      Buffer.add_string buf (Printf.sprintf "  %s r;\n" mangled) ;
      Buffer.add_string buf (Printf.sprintf "  r.tag = %s;\n" cname) ;
      (match args with
      | [] -> ()
      | [_] -> Buffer.add_string buf (Printf.sprintf "  r.%s_v = v;\n" cname)
      | _ ->
          List.iteri
            (fun j _ ->
              Buffer.add_string
                buf
                (Printf.sprintf "  r.%s_v._%d = v%d;\n" cname j j))
            args) ;
      Buffer.add_string buf "  return r;\n}\n\n")
    constrs
