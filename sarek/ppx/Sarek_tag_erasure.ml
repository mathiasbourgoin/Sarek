(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Sarek PPX - Static tag erasure (campaign item L14, scoped tier)
 *
 * Typed-AST -> typed-AST rewrite that runs after defunctionalization
 * (Sarek_defunc) / tail-recursion elimination and BEFORE lowering
 * (Sarek_lower_ir), on the DEVICE-lowered kernel only: the un-erased
 * [native_kernel] captured earlier in Sarek_ppx keeps the tag and stays the
 * OCaml reference path. See roster/ptx-limits-campaign/L14-static-tag-erasure.md.
 *
 * The tag stays in the typed AST for typing / OCaml-side exhaustiveness
 * checking; this pass erases it for code generation wherever the live
 * constructor of a variant-typed *storage slot* is statically determined -
 * the slot is written exactly once by a literal constructor application
 * ([TEConstr]) and is only ever read as the scrutinee of a [match]. This is
 * case-of-known-constructor / iota-reduction applied to the storage
 * representation of a value whose constructor identity is invariant, which is
 * a standard, decades-old compiler technique (design doc SS7).
 *
 * Scoped tier implemented here (S1): a variant-typed plain (immutable)
 * kernel-local `let` binding, so there is exactly one write site by
 * construction:
 *
 *     let s = C payload in ... match s with .. | C x -> rhs | .. ...
 *  ~> let s = payload   in ... let x = s in rhs ...            (unary)
 *     let s = C         in ... match s with .. | C -> rhs | .. ...
 *  ~>                       ... rhs ...                         (nullary, drop)
 *
 * The tag store and the whole branch chain disappear from every backend's
 * emitted code (observable in the IR / CUDA-C / PTX), and behaviour is
 * unchanged: the reduced form runs the C-arm unconditionally, binding the
 * payload for the unary case. Anything with a runtime-selected constructor - a
 * mutable slot, a value from a `TEIf`/`TEMatch`/helper call, or a slot read as
 * a whole variant value rather than only as a `match` scrutinee - is
 * conservatively left tagged, exactly today's behaviour.
 *
 * Scoped tier S2 (this file, below the S1 section): erasing a variant-typed
 * *record field* - the layout-unblock headline of SS0/SS8. Rather than reconcile
 * the `[@@sarek.type]` declaration (consumed by field POSITION by the
 * Interpreter and host marshalling), S2 SYNTHESIZES a device-only all-scalar
 * record with POSITIONAL fields ([_0], [_1], ..) - mirroring L13's `_tup_*`
 * tuple synthesis - so no declaration reconciliation and no runtime
 * registration are needed: positional fields resolve without a registry entry
 * (Sarek_ir_interp_eval.positional_field_index), and threading a nominal
 * `TRecord` type (not a tuple, which falls back to the int32 placeholder in
 * Sarek_lower_ir.elttype_of_typ) into the binding gives every struct-based
 * backend a correct type. See the S2 section header for the exact scope and the
 * empirical backend matrix this unblocks (Interpreter/Vulkan reject the layout,
 * OpenCL the field read, CUDA emits malformed code; only Native tolerated it).
 ******************************************************************************)

open Sarek_types
open Sarek_typed_ast

let is_variant_ty (t : typ) : bool =
  match repr t with TVariant _ -> true | _ -> false

(* -------------------------------------------------------------------------- *)
(* Generic structural map / iter over the typed AST (mirrors the shape of     *)
(* Sarek_defunc.rewrite_desc / iter_children).                                *)
(* -------------------------------------------------------------------------- *)

let map_desc (f : texpr -> texpr) (te : texpr_desc) : texpr_desc =
  match te with
  | TEUnit | TEBool _ | TEInt _ | TEInt32 _ | TEInt64 _ | TEFloat _ | TEDouble _
  | TEGlobalRef _ | TENative _ | TEIntrinsicConst _ | TEVar _ ->
      te
  | TEVecGet (v, i) -> TEVecGet (f v, f i)
  | TEVecSet (v, i, x) -> TEVecSet (f v, f i, f x)
  | TEArrGet (a, i) -> TEArrGet (f a, f i)
  | TEArrSet (a, i, x) -> TEArrSet (f a, f i, f x)
  | TEFieldGet (r, fld, idx) -> TEFieldGet (f r, fld, idx)
  | TEFieldSet (r, fld, idx, v) -> TEFieldSet (f r, fld, idx, f v)
  | TEBinop (op, a, b) -> TEBinop (op, f a, f b)
  | TEUnop (op, a) -> TEUnop (op, f a)
  | TEApp (fn, args) -> TEApp (f fn, List.map f args)
  | TEAssign (n, id, v) -> TEAssign (n, id, f v)
  | TELet (n, id, v, b) -> TELet (n, id, f v, f b)
  | TELetRec (n, id, ps, fn_body, cont) ->
      TELetRec (n, id, ps, f fn_body, f cont)
  | TELetMut (n, id, v, b) -> TELetMut (n, id, f v, f b)
  | TEIf (c, t, e) -> TEIf (f c, f t, Option.map f e)
  | TEFor (v, id, lo, hi, dir, body) -> TEFor (v, id, f lo, f hi, dir, f body)
  | TEWhile (c, b) -> TEWhile (f c, f b)
  | TESeq es -> TESeq (List.map f es)
  | TEMatch (s, cases) -> TEMatch (f s, List.map (fun (p, b) -> (p, f b)) cases)
  | TERecord (n, fields) ->
      TERecord (n, List.map (fun (fl, e) -> (fl, f e)) fields)
  | TEConstr (tn, cn, arg) -> TEConstr (tn, cn, Option.map f arg)
  | TETuple es -> TETuple (List.map f es)
  | TEReturn e -> TEReturn (f e)
  | TECreateArray (size, t, m) -> TECreateArray (f size, t, m)
  | TEPragma (opts, body) -> TEPragma (opts, f body)
  | TEIntrinsicFun (r, c, args) -> TEIntrinsicFun (r, c, List.map f args)
  | TELetShared (n, id, t, size, b) ->
      TELetShared (n, id, t, Option.map f size, f b)
  | TESuperstep (n, d, step, cont) -> TESuperstep (n, d, f step, f cont)
  | TEOpen (path, body) -> TEOpen (path, f body)

let iter_children (f : texpr -> unit) (e : texpr) : unit =
  ignore
    (map_desc
       (fun c ->
         f c ;
         c)
       e.te)

let map_children (f : texpr -> texpr) (e : texpr) : texpr =
  {e with te = map_desc f e.te}

(* -------------------------------------------------------------------------- *)
(* Eligibility + reduction, keyed on a slot variable id                       *)
(* -------------------------------------------------------------------------- *)

let is_slot_var (id : int) (e : texpr) : bool =
  match e.te with TEVar (_, i) -> i = id | _ -> false

(* A constructor arm is reducible only when its payload pattern binds through
   the forms [reduce_matches] actually substitutes: no payload (nullary), a
   single [TPVar], or a binder-free [TPAny] (`C _ -> ...`). Compound payload
   patterns (a multi-arg constructor's [TPTuple], nested [TPConstr], ...) would
   have their binders silently dropped by the reduction, so they make the arm -
   and therefore the slot - ineligible and the tag is retained. *)
let is_reducible_payload (arg : tpattern option) : bool =
  match arg with
  | None -> true
  | Some {tpat = TPVar _; _} | Some {tpat = TPAny; _} -> true
  | Some _ -> false

let is_ctor_case (cname : string) ((p, _) : tpattern * texpr) : bool =
  match p.tpat with
  | TPConstr (_, cn, arg) -> cn = cname && is_reducible_payload arg
  | _ -> false

(* Could this arm match a value built with constructor [cname]? A wildcard or
   variable pattern matches anything; a constructor pattern matches only its
   own constructor; anything else on a variant scrutinee is treated
   conservatively as matching. Used to detect arms that shadow the [cname] arm
   under OCaml's first-match-wins semantics. *)
let arm_matches_ctor (cname : string) ((p, _) : tpattern * texpr) : bool =
  match p.tpat with
  | TPAny | TPVar _ -> true
  | TPConstr (_, cn, _) -> cn = cname
  | _ -> true

(* The reduction keeps the first arm satisfying [is_ctor_case]; that is only
   correct if no earlier arm would also match a [cname] value at runtime.
   Otherwise `match s with _ -> a | C x -> b` (first-match-wins => [a]) would be
   miscompiled to [b]. Returns true when the [cname] arm is reachable. *)
let ctor_arm_reachable (cname : string) (cases : (tpattern * texpr) list) : bool
    =
  let rec go = function
    | [] -> true (* no cname arm at all; caller's List.exists handles that *)
    | case :: _ when is_ctor_case cname case -> true
    | case :: rest -> if arm_matches_ctor cname case then false else go rest
  in
  go cases

(* Every read of the slot variable is the scrutinee of a `match` that has an
   explicit arm for [cname]. Any other use of the variable (which would need
   the whole tagged value) makes the slot ineligible. Variable ids are globally
   unique (Sarek_typed_ast.fresh_var_id), so there is no shadowing to handle. *)
let uses_all_reducible (id : int) (cname : string) (body : texpr) : bool =
  let ok = ref true in
  let rec go (e : texpr) : unit =
    match e.te with
    | TEVar (_, i) when i = id -> ok := false (* bare use outside scrutinee *)
    | TEMatch (scrut, cases) when is_slot_var id scrut ->
        if
          (not (List.exists (is_ctor_case cname) cases))
          || not (ctor_arm_reachable cname cases)
        then ok := false ;
        List.iter (fun (_, rhs) -> go rhs) cases
    | _ -> iter_children go e
  in
  go body ;
  !ok

(* Substitute [repl] for every occurrence of variable [pid]. Used to inline the
   payload-typed slot variable in place of a constructor pattern's bound
   variable, so no fresh `let` is introduced in what may be an expression
   position (Sarek treats `let` as statement-only in some contexts - see the
   note in Sarek_defunc.distribute_app). [repl] is always a bare variable read,
   so substitution never duplicates work. *)
let subst_var (pid : int) (repl : texpr) (e : texpr) : texpr =
  let rec go (e : texpr) : texpr =
    match e.te with TEVar (_, i) when i = pid -> repl | _ -> map_children go e
  in
  go e

(* Replace every `match <slot> with .. | C pat -> rhs | ..` by the C-arm: [rhs]
   with the bound pattern variable substituted by [accessor] (the slot re-read
   at the payload type) for a unary constructor, or [rhs] alone for a nullary
   one. Recurses into the kept arm so nested reads reduce too. *)
let reduce_matches (id : int) (cname : string)
    (accessor : (unit -> texpr) option) (body : texpr) : texpr =
  let rec go (e : texpr) : texpr =
    match e.te with
    | TEMatch (scrut, cases) when is_slot_var id scrut -> (
        let pat, rhs =
          match List.find_opt (is_ctor_case cname) cases with
          | Some (p, rhs) -> (
              match p.tpat with
              | TPConstr (_, _, arg) -> (arg, rhs)
              | _ -> (None, rhs))
          | None ->
              (* Guarded by uses_all_reducible; if a future eligibility-check
                 change breaks that invariant, fail loudly rather than
                 re-entering this same branch forever via [go e]. *)
              failwith
                "Sarek_tag_erasure.reduce_matches: no reducible arm for the \
                 slot's constructor (invariant violated: uses_all_reducible \
                 should have made this slot ineligible)"
        in
        let rhs' = go rhs in
        match (pat, accessor) with
        | Some {tpat = TPVar (_, pid); _}, Some mk -> subst_var pid (mk ()) rhs'
        | _ -> {rhs' with ty = e.ty})
    | _ -> map_children go e
  in
  go body

(* -------------------------------------------------------------------------- *)
(* S2: variant-typed record FIELD erasure (device path, layout unblock)        *)
(*                                                                             *)
(* An immutable kernel-local record binding                                    *)
(*                                                                             *)
(*   let r = { ..; f = C payload; .. } in .. match r.f with .. | C x -> rhs .. *)
(*                                                                             *)
(* whose variant-typed field [f] is written by a *literal* constructor [C]     *)
(* with a substitutable payload, and every read of which is only the scrutinee *)
(* of a [match] with a reducible [C]-arm, is rewritten on the DEVICE path to a *)
(* synthesized all-scalar record with POSITIONAL fields ([_0], [_1], ..),      *)
(* mirroring L13's tuple synthesis (Sarek_lower_ir.tuple_record_fields):       *)
(*                                                                             *)
(*   let r = { _0 = payload; _1 = .. } in .. let x = r._0 in rhs ..            *)
(*                                                                             *)
(* The variant field's tag/branch is erased exactly as S1 does for a slot, and *)
(* the record loses its nested variant entirely. That is what unblocks the     *)
(* backends that reject a nested-variant record field today (empirically:      *)
(* Interpreter and Vulkan/GLSL reject the layout, OpenCL rejects the field     *)
(* read, CUDA emits a malformed constructor). Positional [_i] fields need no   *)
(* runtime registry entry — the interpreter resolves them by position          *)
(* (Sarek_ir_interp_eval.positional_field_index) — and an all-scalar record    *)
(* threads through elttype_of_typ as a proper nominal struct (a *tuple*-typed  *)
(* binding would fall back to the int32 placeholder, so we synthesize a named  *)
(* record, not a tuple). The native path keeps the original record             *)
(* (native_kernel is captured before this pass), so the [@@sarek.type]         *)
(* declaration and host marshalling stay untouched.                            *)
(*                                                                             *)
(* Scope (conservative, same spirit as S1): a plain immutable kernel-local     *)
(* [let]-bound record literal; EVERY variant field is erasable (literal ctor + *)
(* substitutable payload + all reads are reducible match scrutinees); every    *)
(* kept field and every erased unary payload is a scalar primitive; the record *)
(* value never escapes as a whole (only field reads of it appear). Anything    *)
(* else leaves the record entirely untouched (today's behaviour).              *)
(* -------------------------------------------------------------------------- *)

let is_scalar_prim (t : typ) : bool =
  match repr t with
  | TPrim (TInt32 | TBool) | TReg (Int64 | Float32 | Float64 | Int) -> true
  | _ -> false

(* C-identifier-safe tag for a scalar type, used to mangle the synthesized
   record name. Two erased records of identical shape share a name and hence a
   single emitted struct, which is sound (structural). *)
let typ_tag (t : typ) : string =
  match repr t with
  | TPrim TInt32 -> "int32"
  | TPrim TBool -> "bool"
  | TReg Int64 -> "int64"
  | TReg Float32 -> "float32"
  | TReg Float64 -> "float64"
  | TReg Int -> "int"
  | _ -> "x"

let pos_field_name (i : int) : string = Printf.sprintf "_%d" i

(* Per-field plan for an eligible record binding. Slots (the positional fields
   of the synthesized record) are numbered in source field order; a nullary
   erased field carries no data and is dropped, so it consumes no slot. *)
type field_plan =
  | FKeep of int * typ  (** kept scalar field -> positional slot, its type *)
  | FUnary of int * typ * string
      (** erased unary field -> slot, payload type, constructor name *)
  | FNullary of string  (** erased nullary field -> dropped, constructor name *)

(* Classify every field of a record literal. Returns [None] (whole record
   ineligible) if any field is a variant not written by a reducible literal
   constructor, or a kept/payload field that is not a scalar primitive.
   Requires at least one surviving slot and at least one erased variant. *)
let plan_fields (fields : (string * texpr) list) :
    (string * field_plan) list option =
  let exception Bail in
  try
    let slot = ref 0 in
    let next () =
      let i = !slot in
      incr slot ;
      i
    in
    let plan =
      List.map
        (fun (fname, (fexpr : texpr)) ->
          match repr fexpr.ty with
          | TVariant _ -> (
              match fexpr.te with
              | TEConstr (_, cname, None) -> (fname, FNullary cname)
              | TEConstr (_, cname, Some payload) when is_scalar_prim payload.ty
                ->
                  (fname, FUnary (next (), repr payload.ty, cname))
              | _ -> raise Bail)
          | t when is_scalar_prim t -> (fname, FKeep (next (), t))
          | _ -> raise Bail)
        fields
    in
    let has_variant =
      List.exists
        (function _, (FUnary _ | FNullary _) -> true | _ -> false)
        plan
    in
    if !slot = 0 || not has_variant then None else Some plan
  with Bail -> None

(* Constructor name of an erased variant field, if [fname] is one. *)
let variant_field_ctor (plan : (string * field_plan) list) (fname : string) :
    string option =
  match List.assoc_opt fname plan with
  | Some (FUnary (_, _, c)) | Some (FNullary c) -> Some c
  | _ -> None

(* Is [scrut] a read [r.fname] of record variable [rid]? Returns the field. *)
let field_read_of (rid : int) (e : texpr) : string option =
  match e.te with
  | TEFieldGet ({te = TEVar (_, i); _}, fname, _) when i = rid -> Some fname
  | _ -> None

(* Body-usage eligibility: every occurrence of [rid] is a field read; every
   read of an erased variant field is a match scrutinee whose match has a
   reducible arm for that field's constructor; the record never escapes whole.
   Mirrors S1's uses_all_reducible, keyed on field reads instead of the slot
   variable. *)
let record_body_eligible (rid : int) (plan : (string * field_plan) list)
    (body : texpr) : bool =
  let ok = ref true in
  let rec go (e : texpr) : unit =
    match e.te with
    | TEVar (_, i) when i = rid -> ok := false (* whole-record escape *)
    | TEMatch (scrut, cases) -> (
        match
          Option.bind (field_read_of rid scrut) (variant_field_ctor plan)
        with
        | Some cname ->
            if
              (not (List.exists (is_ctor_case cname) cases))
              || not (ctor_arm_reachable cname cases)
            then ok := false ;
            List.iter (fun (_, rhs) -> go rhs) cases (* skip the scrutinee *)
        | None -> iter_children go e)
    | TEFieldGet ({te = TEVar (_, i); _}, fname, _) when i = rid -> (
        (* A field read reached here is NOT an erased-variant match scrutinee. *)
        match variant_field_ctor plan fname with
        | Some _ ->
            ok := false (* bare variant-field read needs the whole value *)
        | None -> () (* scalar field read: fine; do not recurse into the var *))
    | _ -> iter_children go e
  in
  go body ;
  !ok

(* An active record-erasure context: everything the body rewrite needs. *)
type rec_ctx = {
  rc_rid : int;
  rc_rname : string;
  rc_synth_ty : typ;
  rc_plan : (string * field_plan) list;
  rc_loc : Sarek_ast.loc;
}

(* Find the [match] arm for [cname] (guaranteed reducible by eligibility) and
   return its payload pattern and rhs. *)
let find_ctor_arm (cname : string) (cases : (tpattern * texpr) list) :
    tpattern option * texpr =
  match List.find_opt (is_ctor_case cname) cases with
  | Some (p, rhs) -> (
      match p.tpat with TPConstr (_, _, arg) -> (arg, rhs) | _ -> (None, rhs))
  | None ->
      failwith
        "Sarek_tag_erasure.find_ctor_arm: no reducible arm for the record \
         field's constructor (invariant violated: record_body_eligible should \
         have made this record ineligible)"

(* Build a positional field read [r._i : ty] on the synthesized record. *)
let mk_field_read (rc : rec_ctx) (slot : int) (ty : typ) : texpr =
  let rrec =
    {
      te = TEVar (rc.rc_rname, rc.rc_rid);
      ty = rc.rc_synth_ty;
      te_loc = rc.rc_loc;
    }
  in
  {te = TEFieldGet (rrec, pos_field_name slot, slot); ty; te_loc = rc.rc_loc}

(* Try to erase a record binding. Returns the rewritten [TELet] on success, or
   [None] to leave it untouched. [descend] is the recursion used for the
   payload expressions and to continue the general rewrite. *)
let rec try_erase_record (ctxs : rec_ctx list) (name : string) (id : int)
    (value : texpr) (body : texpr) (te_loc : Sarek_ast.loc) : texpr option =
  match value.te with
  | TERecord (_, fields) -> (
      match plan_fields fields with
      | Some plan when record_body_eligible id plan body ->
          (* Survivors in slot order (source order over kept/unary fields). *)
          let survivors =
            List.filter_map
              (fun (fname, (fexpr : texpr)) ->
                match List.assoc fname plan with
                | FKeep (slot, ty) -> Some (slot, descend ctxs fexpr, ty)
                | FUnary (slot, ty, _) ->
                    let payload =
                      match fexpr.te with
                      | TEConstr (_, _, Some p) -> p
                      | _ -> assert false
                    in
                    Some (slot, descend ctxs payload, ty)
                | FNullary _ -> None)
              fields
          in
          let survivors =
            List.sort (fun (a, _, _) (b, _, _) -> compare a b) survivors
          in
          let synth_name =
            "_erec"
            ^ String.concat
                ""
                (List.map (fun (_, _, ty) -> "_" ^ typ_tag ty) survivors)
          in
          let field_tys =
            List.map (fun (slot, _, ty) -> (pos_field_name slot, ty)) survivors
          in
          let synth_ty = TRecord (synth_name, field_tys) in
          let value_fields =
            List.map (fun (slot, e, _) -> (pos_field_name slot, e)) survivors
          in
          let value' =
            {
              te = TERecord (synth_name, value_fields);
              ty = synth_ty;
              te_loc = value.te_loc;
            }
          in
          let rc =
            {
              rc_rid = id;
              rc_rname = name;
              rc_synth_ty = synth_ty;
              rc_plan = plan;
              rc_loc = value.te_loc;
            }
          in
          let body' = descend (rc :: ctxs) body in
          Some {te = TELet (name, id, value', body'); ty = body'.ty; te_loc}
      | _ -> None)
  | _ -> None

(* The unified S2 traversal, carrying the active record-erasure contexts. It (a)
   reduces each erased variant field's match to its constructor arm (binding the
   arm's payload variable to the positional field read), (b) remaps every
   kept-field read to its positional slot, and (c) attempts a fresh erasure at
   every [let]-bound record literal, so S2 recurses into and composes with
   nested erasable records. Everything else recurses structurally. *)
and descend (ctxs : rec_ctx list) (e : texpr) : texpr =
  let find_ctx rid = List.find_opt (fun rc -> rc.rc_rid = rid) ctxs in
  match e.te with
  | TELet (name, id, value, body) -> (
      match try_erase_record ctxs name id value body e.te_loc with
      | Some e' -> e'
      | None -> map_children (descend ctxs) e)
  | TEMatch (scrut, cases) -> (
      match scrut.te with
      | TEFieldGet ({te = TEVar (_, i); _}, fname, _) -> (
          match find_ctx i with
          | Some rc when variant_field_ctor rc.rc_plan fname <> None -> (
              match List.assoc fname rc.rc_plan with
              | FUnary (slot, pty, cname) -> (
                  let arg, rhs = find_ctor_arm cname cases in
                  let rhs' = descend ctxs rhs in
                  match arg with
                  | Some {tpat = TPVar (_, pid); _} ->
                      subst_var pid (mk_field_read rc slot pty) rhs'
                  | _ -> {rhs' with ty = e.ty})
              | FNullary cname ->
                  let _, rhs = find_ctor_arm cname cases in
                  {(descend ctxs rhs) with ty = e.ty}
              | FKeep _ -> map_children (descend ctxs) e)
          | _ -> map_children (descend ctxs) e)
      | _ -> map_children (descend ctxs) e)
  | TEFieldGet ({te = TEVar (_, i); _}, fname, _) -> (
      match find_ctx i with
      | Some rc -> (
          match List.assoc_opt fname rc.rc_plan with
          | Some (FKeep (slot, ty)) -> mk_field_read rc slot ty
          | _ ->
              map_children (descend ctxs) e
              (* variant-field bare read: eligibility bars it *))
      | None -> map_children (descend ctxs) e)
  | _ -> map_children (descend ctxs) e

let erase_record_fields (body : texpr) : texpr = descend [] body

(* -------------------------------------------------------------------------- *)
(* Core rewrite                                                               *)
(* -------------------------------------------------------------------------- *)

let rewrite_body (body : texpr) : texpr =
  let rec rewrite (e : texpr) : texpr =
    match e.te with
    | TELet (name, id, value, body)
      when is_variant_ty value.ty
           && match value.te with TEConstr _ -> true | _ -> false -> (
        match value.te with
        | TEConstr (_tyname, cname, arg) when uses_all_reducible id cname body
          -> (
            match arg with
            | None ->
                (* nullary: drop the binding, reduce every match to its arm *)
                reduce_matches id cname None (rewrite body)
            | Some payload ->
                (* unary: retype the binding to the payload; each match arm
                   binds its pattern var to the (now payload-typed) slot. *)
                let payload' = rewrite payload in
                let accessor () = {payload' with te = TEVar (name, id)} in
                let body' =
                  reduce_matches id cname (Some accessor) (rewrite body)
                in
                {
                  te = TELet (name, id, payload', body');
                  ty = body'.ty;
                  te_loc = e.te_loc;
                })
        | _ -> map_children rewrite e)
    | _ -> map_children rewrite e
  in
  rewrite body

(* -------------------------------------------------------------------------- *)
(* Entry point                                                                *)
(* -------------------------------------------------------------------------- *)

(* Erase variant record fields (S2) first, then variant slots (S1). The two are
   orthogonal — S2 rewrites [let r = {..}] record bindings into all-scalar
   synthesized records, S1 rewrites [let s = C ..] variant-slot bindings — and
   S2 only ever turns a variant field read into a scalar field read, never
   producing a new variant slot, so the composition is confluent. *)
let transform_body (b : texpr) : texpr = rewrite_body (erase_record_fields b)

let erase_tags (kernel : tkernel) : tkernel =
  let module_items' =
    List.map
      (function
        | TMFun (n, r, ps, b) -> TMFun (n, r, ps, transform_body b)
        | TMConst _ as it -> it)
      kernel.tkern_module_items
  in
  {
    kernel with
    tkern_module_items = module_items';
    tkern_body = transform_body kernel.tkern_body;
  }
