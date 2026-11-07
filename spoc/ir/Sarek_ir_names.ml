(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

open Sarek_ir_types
module SM = Map.Make (String)
module SS = Set.Make (String)

(* One binder of a name: the emitted name, the [var_id] it was bound with ([0]
   when it has none: pattern variables, shared arrays, helpers, or an unset
   id) and whether it is an array (a vector parameter, a local or shared
   array). [leaked] marks a scalar whose [let] body has ended: its declaration
   stays in the target block, but only a reference carrying its own [var_id]
   still reaches it. Arrays and scalars of one name are different binders: an array
   reference by name never reaches a scalar, nor the reverse. *)
type entry = {emitted : string; ident : int; array : bool; leaked : bool}

(* [names]: source name -> the binders of that name in scope, innermost first.
   A reference picks the innermost binder that can be its own: a variable
   reference whose [var_id] and the binder's are both set and differ does not
   reach that binder, so a later sibling that still reads an outer binder is
   not captured by an inner one of the same name that was renamed (the
   interpreter resolves by [var_id] too). [visible]: the emitted names a new
   binder must not take. They differ for a parameter of a shader target, which
   is looked up but left out of [visible] (see [rename_params] in the
   interface). *)
type env = {names : entry list SM.t; visible : SS.t}

let is_array_type = function TVec _ | TArray _ -> true | _ -> false

type ctx = {
  reserved : string -> bool;
  rename_params : bool;
  dry : bool;  (** First pass: record every name, rename nothing. *)
  taken : (string, unit) Hashtbl.t;
      (** Every name of the kernel, and minted. *)
  mutable changed : bool;
}

let empty_env = {names = SM.empty; visible = SS.empty}

let note ctx name = Hashtbl.replace ctx.taken name ()

let find_entry env name ~array ~ident =
  match SM.find_opt name env.names with
  | None -> None
  | Some binders ->
      List.find_opt
        (fun e ->
          e.array = array
          && (array
             ||
             if e.leaked then ident <> 0 && e.ident = ident
             else ident = 0 || e.ident = 0 || e.ident = ident))
        binders

let emitted_or_same name = function Some e -> e.emitted | None -> name

(* A scalar name with no identity: a helper, a callee. *)
let look ctx env name =
  note ctx name ;
  emitted_or_same name (find_entry env name ~array:false ~ident:0)

(* The name of an array, as the IR spells array references. *)
let look_array ctx env name =
  note ctx name ;
  emitted_or_same name (find_entry env name ~array:true ~ident:0)

let look_var ctx env (v : var) =
  note ctx v.var_name ;
  emitted_or_same
    v.var_name
    (find_entry
       env
       v.var_name
       ~array:(is_array_type v.var_type)
       ~ident:v.var_id)

(* The C spelling of a name that is not an identifier: [x'] becomes [xp], any
   other stray character an underscore. *)
let sanitize name =
  let spell c =
    if Sarek_reserved_words.is_identifier_char c then String.make 1 c
    else if c = '\'' then "p"
    else "_"
  in
  let body =
    String.concat "" (List.map spell (List.of_seq (String.to_seq name)))
  in
  if body = "" || (body.[0] >= '0' && body.[0] <= '9') then "v_" ^ body
  else body

(* [<name>_<k>] for the smallest [k >= 1] that no name of the kernel, no
   reserved word and no binder in scope uses. *)
let fresh ctx env name =
  let base = sanitize name in
  (* A prefix family ([half_], [get_], [simd_]) reserves every [<base>_<k>]:
     lead with [v_] then, which no target reserves. *)
  let base = if ctx.reserved (base ^ "_1") then "v_" ^ base else base in
  let rec go k =
    let candidate = Printf.sprintf "%s_%d" base k in
    if
      Hashtbl.mem ctx.taken candidate
      || ctx.reserved candidate
      || SS.mem candidate env.visible
    then go (k + 1)
    else begin
      note ctx candidate ;
      candidate
    end
  in
  go 1

(* Bind [name]: renamed when reserved or when a binder of the same name is in
   scope (in C a declaration is in scope in its own initialiser, and a second
   declaration in one block is an error). *)
let bind ?(is_param = false) ?(ident = 0) ?(array = false) ctx env name =
  note ctx name ;
  let renamable = (not is_param) || ctx.rename_params in
  let emitted =
    if ctx.dry || not renamable then name
    else if ctx.reserved name || SS.mem name env.visible then fresh ctx env name
    else name
  in
  if emitted <> name then ctx.changed <- true ;
  let visible =
    (* A shader scalar parameter is exposed by name and left out; a vector
       parameter is a buffer of that name, which a local must still avoid. *)
    if is_param && (not ctx.rename_params) && not array then env.visible
    else SS.add emitted env.visible
  in
  let binders = Option.value ~default:[] (SM.find_opt name env.names) in
  ( emitted,
    {
      names =
        SM.add
          name
          ({emitted; ident; array; leaked = false} :: binders)
          env.names;
      visible;
    } )

let bind_var ?is_param ctx env (v : var) =
  let name, env =
    bind
      ?is_param
      ~ident:v.var_id
      ~array:(is_array_type v.var_type)
      ctx
      env
      v.var_name
  in
  ({v with var_name = name}, env)

let bind_pattern ctx env = function
  | PWild -> (PWild, env)
  | PConstr (constr, names) ->
      let env, names =
        List.fold_left_map
          (fun env n ->
            let n, env = bind ctx env n in
            (env, n))
          env
          names
      in
      (PConstr (constr, names), env)

let rec expr ctx env e =
  let go = expr ctx env in
  match e with
  | EConst _ -> e
  | EVar v -> EVar {v with var_name = look_var ctx env v}
  | EBinop (op, a, b) -> EBinop (op, go a, go b)
  | EUnop (op, a) -> EUnop (op, go a)
  | EArrayRead (a, i) -> EArrayRead (look_array ctx env a, go i)
  | EArrayReadExpr (a, i) -> EArrayReadExpr (go a, go i)
  | ERecordField (a, f) -> ERecordField (go a, f)
  | EIntrinsic (path, name, args) -> EIntrinsic (path, name, List.map go args)
  | ECast (t, a) -> ECast (t, go a)
  | ETuple es -> ETuple (List.map go es)
  | EApp (f, args) -> EApp (go f, List.map go args)
  | ERecord (t, fields) -> ERecord (t, List.map (fun (f, x) -> (f, go x)) fields)
  | EVariant (t, c, args) -> EVariant (t, c, List.map go args)
  | EArrayLen a -> EArrayLen (look_array ctx env a)
  | EArrayCreate (t, size, space) -> EArrayCreate (t, go size, space)
  | EIf (c, a, b) -> EIf (go c, go a, go b)
  | EMatch (scrutinee, cases) ->
      EMatch (go scrutinee, List.map (case_with ctx env expr) cases)

and case_with :
    'a. ctx -> env -> (ctx -> env -> 'a -> 'a) -> pattern * 'a -> pattern * 'a =
 fun ctx env body (pattern, arm) ->
  let pattern, env = bind_pattern ctx env pattern in
  (pattern, body ctx env arm)

let rec lvalue ctx env = function
  | LVar v -> LVar {v with var_name = look_var ctx env v}
  | LArrayElem (a, i) -> LArrayElem (look_array ctx env a, expr ctx env i)
  | LArrayElemExpr (a, i) -> LArrayElemExpr (expr ctx env a, expr ctx env i)
  | LRecordField (lv, f) -> LRecordField (lvalue ctx env lv, f)

(* Fragment names are a namespace of their own (see [coopmat_op]); only the
   buffers they move to and from, and the index expressions, are names of the
   kernel. *)
let coopmat ctx env = function
  | CM_decl _ as op -> op
  | CM_load r ->
      CM_load
        {
          r with
          src = look_array ctx env r.src;
          index = expr ctx env r.index;
          stride = expr ctx env r.stride;
        }
  | CM_store r ->
      CM_store
        {
          r with
          dst = look_array ctx env r.dst;
          index = expr ctx env r.index;
          stride = expr ctx env r.stride;
        }
  | CM_muladd _ as op -> op
  | CM_scale_add _ as op ->
      Sarek_coopmat_scale.map ~buf:(look_array ctx env) ~expr:(expr ctx env) op

(* What a [let] leaves behind for the later siblings of its target block: the
   names it declared stay unavailable ([visible]) and a scalar stays reachable
   by its [var_id], but an ARRAY binder is out of the IR scope that ends with
   its body: an array reference carries no identity, and means the innermost
   array of that name in the IR, which is not this one. *)
let leave_binder before inner (v : var) =
  if not (is_array_type v.var_type) then
    let before_binders =
      Option.value ~default:[] (SM.find_opt v.var_name before.names)
    in
    let mark binders =
      List.map
        (fun e ->
          if List.memq e before_binders then e else {e with leaked = true})
        binders
    in
    {inner with names = SM.update v.var_name (Option.map mark) inner.names}
  else
    let names =
      match SM.find_opt v.var_name before.names with
      | Some binders -> SM.add v.var_name binders inner.names
      | None -> SM.remove v.var_name inner.names
    in
    {inner with names}

(* A statement list is one target block: [SLet] and [SLetMut] are emitted as a
   declaration followed by their body, with no braces, and [SPragma] adds none,
   so the declarations of an earlier sibling are in scope for the later ones
   even though the IR nests only the [SLet] body. [stmt_in_block] returns the
   environment those declarations leave behind; constructs that open a block of
   their own ([SIf], [SWhile], [SFor], [SMatch], [SBlock]) leak nothing. *)
let rec stmt_in_block ctx env s =
  match s with
  | SSeq ss ->
      let env, ss =
        List.fold_left_map
          (fun env s ->
            let s, env = stmt_in_block ctx env s in
            (env, s))
          env
          ss
      in
      (SSeq ss, env)
  | SLet (v, x, b) ->
      let x = expr ctx env x in
      let v', inner = bind_var ctx env v in
      let b, inner = stmt_in_block ctx inner b in
      (SLet (v', x, b), leave_binder env inner v)
  | SLetMut (v, x, b) ->
      let x = expr ctx env x in
      let v', inner = bind_var ctx env v in
      let b, inner = stmt_in_block ctx inner b in
      (SLetMut (v', x, b), leave_binder env inner v)
  | SPragma (p, b) ->
      let b, env = stmt_in_block ctx env b in
      (SPragma (p, b), env)
  | _ -> (stmt ctx env s, env)

(* A statement in a scope of its own: nothing it declares outlives it. *)
and stmt ctx env s =
  let e = expr ctx env and st = stmt ctx env in
  match s with
  | SAssign (lv, x) -> SAssign (lvalue ctx env lv, e x)
  | SSeq _ | SLet _ | SLetMut _ | SPragma _ -> fst (stmt_in_block ctx env s)
  | SIf (c, t, f) -> SIf (e c, st t, Option.map st f)
  | SWhile (c, b) -> SWhile (e c, st b)
  | SFor (v, lo, hi, dir, b) ->
      let lo = e lo and hi = e hi in
      let v, env = bind_var ctx env v in
      SFor (v, lo, hi, dir, stmt ctx env b)
  | SMatch (x, cases) -> SMatch (e x, List.map (case_with ctx env stmt) cases)
  | SReturn x -> SReturn (e x)
  | SExpr x -> SExpr (e x)
  | SBlock b -> SBlock (st b)
  | SCoopmat op -> SCoopmat (coopmat ctx env op)
  | SBarrier | SWarpBarrier | SMemFence | SEmpty | SNative _ -> s

(* {2 Kernel level} *)

(* A helper's parameters are real function parameters on every target (shaders
   included), so a local of the helper must avoid them and they may be
   renamed: only the kernel's own parameters are exposed by name. *)
let bind_helper_params ctx env params =
  List.fold_left_map
    (fun env v ->
      (* A vector parameter of a helper is a global buffer on a shader: it
         keeps its name there, like a kernel parameter. *)
      let is_param =
        (not ctx.rename_params)
        && match v.var_type with TVec _ -> true | _ -> false
      in
      let v, env = bind_var ~is_param ctx env v in
      (env, v))
    env
    params

let bind_decl ctx env = function
  | DParam (v, info) ->
      let v, env = bind_var ~is_param:true ctx env v in
      (env, DParam (v, info))
  | DLocal (v, init) ->
      let init = Option.map (expr ctx env) init in
      let v, env = bind_var ctx env v in
      (env, DLocal (v, init))
  | DShared (name, t, size) ->
      let size = Option.map (expr ctx env) size in
      let name, env = bind ~array:true ctx env name in
      (env, DShared (name, t, size))

let helper ctx global (hf : helper_func) =
  let env, params = bind_helper_params ctx global hf.hf_params in
  {
    hf with
    hf_name = look ctx global hf.hf_name;
    hf_params = params;
    hf_body = stmt ctx env hf.hf_body;
  }

(* Helper functions are module-level names: in scope in every function, so a
   local may not take one. *)
let bind_helpers ctx funcs =
  List.fold_left
    (fun env hf ->
      let _, env = bind ctx env hf.hf_name in
      env)
    empty_env
    funcs

let kernel ctx (k : kernel) =
  note ctx k.kern_name ;
  let global = bind_helpers ctx k.kern_funcs in
  let funcs = List.map (helper ctx global) k.kern_funcs in
  let env, params = List.fold_left_map (bind_decl ctx) global k.kern_params in
  let env, locals = List.fold_left_map (bind_decl ctx) env k.kern_locals in
  {
    k with
    kern_params = params;
    kern_locals = locals;
    kern_funcs = funcs;
    kern_body = stmt ctx env k.kern_body;
  }

(* {2 Entry point} *)

(* [SNative] carries source text built by a closure, whose identifiers this pass
   cannot see: a rename around it could not be sound. *)
let rec has_native = function
  | SNative _ -> true
  | SSeq ss -> List.exists has_native ss
  | SIf (_, t, f) -> has_native t || Option.fold ~none:false ~some:has_native f
  | SWhile (_, b) | SFor (_, _, _, _, b) | SPragma (_, b) | SBlock b ->
      has_native b
  | SMatch (_, cases) -> List.exists (fun (_, b) -> has_native b) cases
  | SLet (_, _, b) | SLetMut (_, _, b) -> has_native b
  | SAssign _ | SReturn _ | SBarrier | SWarpBarrier | SExpr _ | SMemFence
  | SEmpty | SCoopmat _ ->
      false

let kernel_has_native (k : kernel) =
  has_native k.kern_body
  || List.exists (fun hf -> has_native hf.hf_body) k.kern_funcs

let legalize ~reserved ~rename_params (k : kernel) : kernel =
  if kernel_has_native k then k
  else
    let ctx =
      {
        reserved;
        rename_params;
        dry = true;
        taken = Hashtbl.create 64;
        changed = false;
      }
    in
    ignore (kernel ctx k) ;
    let ctx = {ctx with dry = false} in
    let renamed = kernel ctx k in
    if ctx.changed then renamed else k

let legalize_shader (k : kernel) : kernel =
  legalize
    ~reserved:Sarek_reserved_words.invalid_identifier
    ~rename_params:false
    k
