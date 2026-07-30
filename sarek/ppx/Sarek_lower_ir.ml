(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Sarek PPX - GPU kernel DSL for OCaml
 *
 * This module lowers the typed AST directly to Sarek_ir_ppx, bypassing
 * the legacy Kirc_Ast. This provides a cleaner IR for the V2 execution path.
 ******************************************************************************)

open Sarek_ast
open Sarek_types
open Sarek_typed_ast
module Ir = Sarek_ir_ppx

(** Mangle type names for C compatibility *)
let mangle_type_name name = String.map (function '.' -> '_' | c -> c) name

(** Convert Sarek_types.typ to Sarek_ir_ppx.elttype *)
let rec elttype_of_typ (ty : typ) : Ir.elttype =
  match repr ty with
  | TPrim TInt32 -> Ir.TInt32
  | TPrim TBool -> Ir.TBool
  | TPrim TUnit -> Ir.TUnit
  | TReg Int64 -> Ir.TInt64
  | TReg Int -> Ir.TInt32 (* int maps to int32 on GPU *)
  | TReg Float16 -> Ir.TFloat16
  | TReg Float32 -> Ir.TFloat32
  | TReg Float64 -> Ir.TFloat64
  | TReg Char ->
      (* WRONG-WIDTH #5. `char` used to lower to [Ir.TInt32] "represented as
         int32". It is not: [Spoc_core.Vector.char] is a Bigarray of OCaml
         chars — ONE byte per element — while the device declaration produced
         from [Ir.TInt32] is `int*`, four. A `char vector` kernel therefore
         compiled clean and strode the buffer at 4x the host's element size,
         with no diagnostic anywhere in the pipeline.

         There is no 1-byte element type in the IR to map it to, so the honest
         answer is to refuse rather than to guess a width. Nothing is lost by
         refusing: [Execute.check_launch_args] already rejects a [Vector.Char]
         argument against a [TInt32] parameter on physical width (see
         test_float16.test_argcheck_width_fallback_for_unmappable_kinds), on
         BOTH the device and the interpreter entry points — so a `char vector`
         kernel could never launch on any backend. All this arm changes is WHEN
         the user is told, and by a message that names the cause.

         Same shape and same remedy as the float16 rejections
         ([lower_param]'s scalar-parameter arm, Sarek_ir_layout, Soa). *)
      Ppxlib.Location.raise_errorf
        ~loc:Ppxlib.Location.none
        "`char` is not a supported Sarek element type: a host `char` is 1 byte \
         (Spoc_core.Vector.char) but the device IR has no 1-byte element type, \
         so it would be accessed through a 4-byte `int`. Use `int32` (and \
         convert on the host) instead."
  | TReg (Custom name) ->
      (* Same class as the [Char] arm above: a registered custom type is a
         user-declared aggregate whose size comes from its registered layout,
         so collapsing it to [Ir.TInt32] ("Custom types handled separately")
         claims a 4-byte scalar for something that is generally neither 4 bytes
         nor a scalar. Record and variant types reach lowering as
         [TRecord]/[TVariant] with their fields, which are handled above; a
         bare [TReg (Custom _)] is a type name that was never registered as a
         Sarek type (it comes from [%sarek_intrinsic]'s fallback for unknown
         type names), and its layout is genuinely unknown here. *)
      Ppxlib.Location.raise_errorf
        ~loc:Ppxlib.Location.none
        "Type %S is not a registered Sarek type, so its device size is \
         unknown. Declare it with [@@sarek.type] (records and variants), or \
         use a built-in scalar type."
        name
  | TVec elem_ty -> Ir.TVec (elttype_of_typ elem_ty)
  | TArr (elem_ty, mem) ->
      let ir_mem =
        match mem with
        | Sarek_types.Global -> Ir.Global
        | Sarek_types.Shared -> Ir.Shared
        | Sarek_types.Local -> Ir.Local
      in
      Ir.TArray (elttype_of_typ elem_ty, ir_mem)
  | TRecord (name, fields) ->
      Ir.TRecord (name, List.map (fun (n, ty) -> (n, elttype_of_typ ty)) fields)
  | TVariant (name, constrs) ->
      Ir.TVariant
        ( name,
          List.map
            (fun (n, ty_opt) -> (n, constr_payload_elttypes ty_opt))
            constrs )
  (* NOTE: TTuple/TFun are NOT rejected here even though this is where the
     original bug report pointed. This function converts the type of *any*
     typed value reachable during lowering, including local/helper function
     bindings inside a kernel body (e.g. `let make_p x y z = ... in ...`,
     see test_visibility_private.ml, test_transpose.ml, bench_nbody.ml) -
     those legitimately have TFun type and must keep lowering (their
     "elttype" is a don't-care placeholder, never actually used as data).
     The real defect is kernel *parameters* of tuple/function type, which
     are rejected explicitly in lower_param below, at the point where a
     type is about to become a formal parameter. *)
  | TTuple _ -> Ir.TInt32
  | TFun _ -> Ir.TInt32
  | TVar _ ->
      Ppxlib.Location.raise_errorf
        ~loc:Ppxlib.Location.none
        "Kernel parameter type is a type variable — polymorphic kernels are \
         not supported. Annotate the parameter with a concrete type (e.g. \
         float32, int32)."

(** Flatten a variant constructor's payload type into its IR field list. A
    multi-argument constructor ([MkPair of float32 * float32]) carries a tuple
    payload type; it is FLATTENED to one IR field per component
    ([TFloat32; TFloat32]) so it lines up field-for-field with the multi-binder
    pattern ([PConstr ("MkPair", ["x"; "y"])]) and the flat [_0/_1] tagged-union
    payload every source generator already emits. A single, non-tuple payload
    stays one field. Only scalar-primitive tuple components flatten; any other
    tuple keeps the (placeholder) single-field mapping. *)
and constr_payload_elttypes (ty_opt : typ option) : Ir.elttype list =
  match ty_opt with
  | None -> []
  | Some ty -> (
      match repr ty with
      | TTuple tys when List.for_all is_scalar_prim_typ tys ->
          List.map elttype_of_typ tys
      | _ -> [elttype_of_typ ty])

(** Scalar-primitive predicate on a source [typ], matching the component set
    accepted for flattened tuple payloads (mirrors [prim_component_elttype]). *)
and is_scalar_prim_typ (t : typ) : bool =
  match repr t with
  | TPrim TInt32 | TPrim TBool | TReg Int64 | TReg Float32 | TReg Float64 ->
      true
  | _ -> false

(* L13: tuple-typed vector elements. A tuple used as a vector element type is
   lowered to a synthesized packed record with positional fields [_0], [_1],
   ..., reusing the record/aggregate codegen path unchanged (the layout is
   computed by Sarek_ir_layout, byte-for-byte identically on host and device).
   Only scalar-primitive components are supported in this tier; nested tuples,
   records, vectors or functions as components are rejected. *)
let elttype_prim_tag (e : Ir.elttype) : string option =
  match e with
  | Ir.TInt32 -> Some "int32"
  | Ir.TInt64 -> Some "int64"
  | Ir.TFloat32 -> Some "float32"
  | Ir.TFloat64 -> Some "float64"
  | Ir.TBool -> Some "bool"
  | _ -> None

(** Scalar-primitive check on the SOURCE type of a tuple component. This must
    match on [Sarek_types.typ] directly, NOT via [elttype_of_typ]: that
    conversion maps [TTuple]/[TFun] to a placeholder [Ir.TInt32] (see its NOTE),
    so routing the primitivity check through it would silently accept nested
    tuples or function components as int32 fields and synthesize a wrong layout.
    Returns the component's IR type iff it is a supported scalar. *)
let prim_component_elttype (t : typ) : Ir.elttype option =
  match repr t with
  | TPrim TInt32 -> Some Ir.TInt32
  | TPrim TBool -> Some Ir.TBool
  | TReg Int64 -> Some Ir.TInt64
  | TReg Float32 -> Some Ir.TFloat32
  | TReg Float64 -> Some Ir.TFloat64
  | _ -> None

let tuple_field_name i = Printf.sprintf "_%d" i

(** Mangled nominal name of the record synthesized for a tuple shape, e.g.
    [_tup_float32_int32] for [float32 * int32]. Kept in sync with the host-side
    [Sarek_tuple_vec] builder so both agree on the element identity. *)
let tuple_record_name (comps : Ir.elttype list) : string =
  "_tup"
  ^ String.concat
      ""
      (List.map
         (fun e ->
           match elttype_prim_tag e with Some t -> "_" ^ t | None -> "_x")
         comps)

(** The single located error raised for any tuple whose components are not all
    scalar primitives — shared by the vector-element path, the kernel-local slot
    path ({!slot_elttype_of_typ}), and the match-scrutinee guard in
    {!lower_stmt}. [loc] should be the offending source expression when known
    ([Ppxlib.Location.none] otherwise). *)
let raise_tuple_component_error ~loc : 'a =
  Ppxlib.Location.raise_errorf
    ~loc
    "Tuple values support only scalar components \
     (float32/float64/int32/int64/bool); nested tuples, records, vectors or \
     functions as tuple components are not supported (applies to vector \
     elements, kernel-local tuple bindings and tuple match scrutinees)."

(** Synthesized record fields (positional [_0.._n]) for a tuple's component
    types. Raises if any component is not a scalar primitive. *)
let tuple_record_fields (tys : typ list) : string * (string * Ir.elttype) list =
  let comps =
    List.map
      (fun t ->
        match prim_component_elttype t with
        | Some e -> e
        | None -> raise_tuple_component_error ~loc:Ppxlib.Location.none)
      tys
  in
  let name = tuple_record_name comps in
  (name, List.mapi (fun i e -> (tuple_field_name i, e)) comps)

let tuple_tmp_counter = ref 0

(** Is [t] a tuple whose components are all scalar primitives (the shape we
    synthesize a record for)? *)
let is_primitive_tuple (t : typ) : bool =
  match repr t with
  | TTuple tys -> List.for_all (fun t -> prim_component_elttype t <> None) tys
  | _ -> false

(** IR type for a value that occupies a *data slot* — a vector element, or a
    kernel-local binding ([let]/[match]-bound). A primitive-component tuple
    becomes its synthesized positional record ([_tup_*], fields [_0.._n]) so the
    struct backends (CUDA/OpenCL/GLSL/Metal) emit the right compound type
    instead of the [elttype_of_typ] placeholder [int] (see that function's
    NOTE); every other type uses the ordinary mapping. A non-primitive tuple
    raises the located tuple-component error via {!tuple_record_fields},
    mirroring vector-of-tuple scope. This is the sole difference from
    {!elttype_of_typ}: the placeholder there stays intact for genuinely non-data
    flows (function-typed helper bindings), while data uses route here. *)
let slot_elttype_of_typ (t : typ) : Ir.elttype =
  match repr t with
  | TTuple tys ->
      let name, fields = tuple_record_fields tys in
      Ir.TRecord (name, fields)
  | _ -> elttype_of_typ t

(** Convert Sarek_types.memspace to Sarek_ir_ppx.memspace *)
let memspace_of_memspace (mem : Sarek_types.memspace) : Ir.memspace =
  match mem with
  | Sarek_types.Global -> Ir.Global
  | Sarek_types.Shared -> Ir.Shared
  | Sarek_types.Local -> Ir.Local

(* REMOVED: [c_type_of_typ] / [record_constructor_strings] /
   [variant_constructor_strings].

   These emitted C struct/union/builder source strings for registered record
   and variant types, which [lower_kernel] returned and [Sarek_quote] passed
   to [Sarek.Kirc_types.register_constructor_string]. That function appends to
   [Kirc_types.constructors], a list that NOTHING in the tree ever reads: the
   C-family backends build variant/record definitions themselves from
   [kern_types]/[kern_variants] via [Sarek_ir_codegen.gen_variant_def], and
   have done so since the Sarek_ir cutover. The strings were write-only.

   They are removed rather than repaired because they could not be repaired.
   [c_type_of_typ] ended in the wildcard [| _ -> "int"] — the
   silently-succeeding shape of the wrong-width family, which declared 2-byte
   float16 members as 4-byte `int`. Replacing that wildcard with the explicit
   rejection the class demands immediately broke a real kernel
   (sarek/tests/e2e/test_static_tag_erasure.ml), because this path is handed
   types it has no C name for and depended on the placeholder to keep going.
   A generator that only works while it is silently wrong, and whose output no
   one reads, is dead weight with a live hazard attached. *)

(* Debug counters for IR lowering *)
let ir_lower_expr_count = ref 0

let ir_lower_stmt_count = ref 0

(* ── module constants referenced from a helper body (backlog-160) ─────────
   A module constant is lexically visible inside a helper, but lowering puts it
   in the KERNEL body while helpers are emitted out-of-line — so the helper names
   an identifier its translation unit never declares. Emitted CUDA, verbatim:

     __device__ float apply(float x) { return (x * scale); }
     __global__ void k(...) { float scale = 2.0f; ... }

   SEVEN paths break that way (CUDA-C, OpenCL, Metal, GLSL, WGSL, PTX and the
   Interpreter); only Native works, because it emits an OCaml [let] the helper
   closes over. A CPU-passes / device-fails divergence.

   THE FIX IS TO PREFIX, NOT TO HOIST. Emitting the constants as top-level
   [const] / [__constant] / [constant] / [__device__ const] declarations was the
   first plan and it is unsound: a module-constant initializer is an arbitrary
   kernel expression — the pipeline explicitly anticipates thread-dependent ones
   (Sarek_convergence analyses thread/block and barrier usage) — and every one of
   those storage classes requires a compile-time-constant initializer. Hoisting
   [let (base : int32) = thread_idx_x] would break a kernel that compiles today.
   Prefixing the [SLet] into the helper body handles both shapes, needs no IR
   field and no backend change, and fixes all seven paths in one place.

   ACCEPTED COST, stated because it is a real semantic change: the initializer is
   evaluated once PER HELPER CALL instead of once per kernel. For a constant that
   is what it says it is, that is redundant work and nothing else. For an
   initializer containing a BARRIER it would change convergence, so that case is
   refused rather than duplicated. *)

(* FREE names of an IR statement: every [EVar]/[LVar] name that is not bound by
   an enclosing binder. [bound] carries the binders in scope at each point.

   The first version took no [bound] and collected every name, including
   locally-bound ones, on the reasoning that over-approximating only costs a dead
   declaration. It does not. The C-family backends emit [SLet] FLAT — [T name =
   e;] followed by the body at the same indent, no braces — so a helper-local
   [let c = ...] that merely SHARES a module constant's name made the constant
   look referenced, got its [SLet] prefixed, and produced two [float c] in one
   block: a redeclaration error on a helper that compiled fine before. Caught by
   review on #362.

   Binders covered: [SLet], [SLetMut], [SFor]. NOT match-pattern binders — those
   stay uncovered, so a pattern-bound name still reads as free. That is the safe
   direction (it can only over-approximate, never emit an undeclared identifier),
   and it is why the fold below refuses a residual collision rather than assuming
   this set is complete. *)
let rec expr_names (e : Ir.expr) (bound : string list)
    (acc : (string, unit) Hashtbl.t) : unit =
  let add n = if not (List.mem n bound) then Hashtbl.replace acc n () in
  let go a = expr_names a bound acc in
  match e with
  | Ir.EVar v -> add v.Ir.var_name
  | Ir.EConst _ -> ()
  | Ir.EBinop (_, a, b) ->
      go a ;
      go b
  | Ir.EUnop (_, a) -> go a
  | Ir.EArrayRead (n, i) ->
      add n ;
      go i
  | Ir.EArrayReadExpr (b, i) ->
      go b ;
      go i
  | Ir.ERecordField (b, _) -> go b
  | Ir.EIntrinsic (_, _, args) -> List.iter go args
  | Ir.ECast (_, a) -> go a
  | Ir.ETuple es -> List.iter go es
  | Ir.EApp (f, args) ->
      go f ;
      List.iter go args
  | Ir.ERecord (_, fs) -> List.iter (fun (_, a) -> go a) fs
  | Ir.EVariant (_, _, args) -> List.iter go args
  | Ir.EArrayLen n -> add n
  | Ir.EArrayCreate (_, sz, _) -> go sz
  | Ir.EIf (c, t, e2) ->
      go c ;
      go t ;
      go e2
  | Ir.EMatch (sc, cases) ->
      go sc ;
      List.iter (fun (_, a) -> go a) cases

and lvalue_names (lv : Ir.lvalue) (bound : string list)
    (acc : (string, unit) Hashtbl.t) : unit =
  let add n = if not (List.mem n bound) then Hashtbl.replace acc n () in
  match lv with
  | Ir.LVar v -> add v.Ir.var_name
  | Ir.LArrayElem (n, i) ->
      add n ;
      expr_names i bound acc
  | Ir.LArrayElemExpr (b, i) ->
      expr_names b bound acc ;
      expr_names i bound acc
  | Ir.LRecordField (b, _) -> lvalue_names b bound acc

and stmt_names (st : Ir.stmt) (bound : string list)
    (acc : (string, unit) Hashtbl.t) : unit =
  let goe e = expr_names e bound acc in
  let gos s = stmt_names s bound acc in
  match st with
  | Ir.SAssign (lv, e) ->
      lvalue_names lv bound acc ;
      goe e
  | Ir.SSeq sts -> List.iter gos sts
  | Ir.SIf (c, t, e) ->
      goe c ;
      gos t ;
      Option.iter gos e
  | Ir.SWhile (c, b) ->
      goe c ;
      gos b
  | Ir.SFor (v, lo, hi, _, b) ->
      (* The bounds are evaluated outside the binding, the body inside it. *)
      goe lo ;
      goe hi ;
      stmt_names b (v.Ir.var_name :: bound) acc
  | Ir.SMatch (sc, cases) ->
      goe sc ;
      List.iter (fun (_, s) -> gos s) cases
  | Ir.SReturn e -> goe e
  | Ir.SExpr e -> goe e
  | Ir.SLet (v, e, b) | Ir.SLetMut (v, e, b) ->
      (* Same asymmetry: [let c = c *. 2.] genuinely references the outer [c], so
         the initializer is walked with the OLD scope. *)
      goe e ;
      stmt_names b (v.Ir.var_name :: bound) acc
  | Ir.SPragma (_, b) -> gos b
  | Ir.SBlock b -> gos b
  | Ir.SBarrier | Ir.SWarpBarrier | Ir.SMemFence | Ir.SEmpty -> ()
  | Ir.SNative _ -> ()

(* Every name BOUND anywhere in a statement, at any depth. Used to detect the
   residual case the free-name fix cannot make safe: a helper that both
   references a module constant and rebinds that same name ([let c = c *. 2.]).
   The reference is real, so the constant must be prefixed; the rebinding then
   emits a second declaration of the same identifier in the same flat block.
   There is no correct code to emit, so the fold refuses. *)
let rec stmt_binders (st : Ir.stmt) (acc : (string, unit) Hashtbl.t) : unit =
  match st with
  | Ir.SLet (v, _, b) | Ir.SLetMut (v, _, b) | Ir.SFor (v, _, _, _, b) ->
      Hashtbl.replace acc v.Ir.var_name () ;
      stmt_binders b acc
  | Ir.SSeq sts -> List.iter (fun s -> stmt_binders s acc) sts
  | Ir.SIf (_, t, e) ->
      stmt_binders t acc ;
      Option.iter (fun s -> stmt_binders s acc) e
  | Ir.SWhile (_, b) -> stmt_binders b acc
  | Ir.SMatch (_, cases) -> List.iter (fun (_, s) -> stmt_binders s acc) cases
  | Ir.SPragma (_, b) | Ir.SBlock b -> stmt_binders b acc
  | Ir.SAssign _ | Ir.SReturn _ | Ir.SExpr _ | Ir.SBarrier | Ir.SWarpBarrier
  | Ir.SMemFence | Ir.SEmpty | Ir.SNative _ ->
      ()

(* Does this initializer contain a synchronising operation? Prefixing would
   duplicate it per call site, which changes convergence, so that case is REFUSED
   rather than duplicated.

   The first version of this function returned [false] unconditionally, reasoning
   that only the statement forms carry a barrier and an [Ir.expr] cannot. The
   first half is true — no [Ir.expr] constructor holds an [Ir.stmt] — and the
   conclusion was still wrong: barriers reach the IR as INTRINSICS
   (Sarek_core_primitives registers [block_barrier], [memory_fence_block],
   [memory_fence_device]; [warp_barrier] joined them in backlog-70), and
   [EIntrinsic] is an expression. A guard that cannot fire is the defect class
   this repository keeps closing, so it is named here rather than quietly left.

   Matched by NAME because that is what the IR carries at this point: lowering
   does not turn these into [SBarrier], it leaves them as intrinsic calls (there
   is no barrier-specific arm in this file). If a name is added to the primitive
   table without being added here, this guard silently stops covering it — which
   is why the list is stated in one place and the refusal message names the
   intrinsic it found. *)
let synchronising_intrinsics =
  ["block_barrier"; "warp_barrier"; "memory_fence_block"; "memory_fence_device"]

let rec expr_barrier (e : Ir.expr) : string option =
  let first =
    List.fold_left
      (fun acc x -> match acc with Some _ -> acc | None -> expr_barrier x)
      None
  in
  match e with
  | Ir.EIntrinsic (_, name, args) ->
      if List.mem name synchronising_intrinsics then Some name else first args
  | Ir.EConst _ | Ir.EVar _ | Ir.EArrayLen _ -> None
  | Ir.EBinop (_, a, b) | Ir.EArrayReadExpr (a, b) -> first [a; b]
  | Ir.EUnop (_, a) | Ir.ECast (_, a) | Ir.ERecordField (a, _) -> expr_barrier a
  | Ir.EArrayRead (_, i) -> expr_barrier i
  | Ir.EArrayCreate (_, sz, _) -> expr_barrier sz
  | Ir.ETuple es -> first es
  | Ir.EApp (f, args) -> first (f :: args)
  | Ir.ERecord (_, fs) -> first (List.map snd fs)
  | Ir.EVariant (_, _, args) -> first args
  | Ir.EIf (c, t, e2) -> first [c; t; e2]
  | Ir.EMatch (sc, cases) -> first (sc :: List.map snd cases)

(** Lowering state *)
type state = {
  fun_map : (string, tparam list * texpr) Hashtbl.t;
  lowering_stack : (string, unit) Hashtbl.t;
  lowered_funs : (string, Ir.helper_func) Hashtbl.t;
      (** Lowered helper functions: name -> helper_func *)
  mutable lowered_funs_order : string list;
      (** Order in which functions were lowered (for dependency ordering) *)
  types : (string, (string * Ir.elttype) list) Hashtbl.t;
      (** Collected record types: type_name -> [(field_name, field_type); ...]
      *)
  variants : (string, (string * Ir.elttype list) list) Hashtbl.t;
      (** Collected variant types: type_name ->
          [(constructor_name, payload_types); ...] *)
  mod_consts : (string, Ir.var * Ir.expr) Hashtbl.t;
      (** Module-level constants, by name, with their LOWERED initializer
          (backlog-160). A helper body that names one must carry its own [SLet]:
          the kernel-body copy is in a different scope, and helpers are emitted
          out-of-line. See [prefix_referenced_consts]. *)
  mutable mod_consts_order : string list;
      (** Declaration order, reversed. Prefixing must be deterministic AND must
          respect dependency order — a later constant may reference an earlier
          one — so the prefix is emitted in declaration order, not hashtable
          order. *)
}

let create_state fun_map =
  {
    fun_map;
    lowering_stack = Hashtbl.create 8;
    lowered_funs = Hashtbl.create 8;
    lowered_funs_order = [];
    types = Hashtbl.create 8;
    variants = Hashtbl.create 8;
    mod_consts = Hashtbl.create 8;
    mod_consts_order = [];
  }

(* There is ONE id allocator PER ID SPACE, and the distinction is load-bearing.
   TERM-variable ids come from the typer's [Sarek_typed_ast.fresh_var_id]; TYPE
   variable ids come from [Sarek_types.fresh_tvar_id]. They are separate
   [Atomic]s, both starting at 0, so they are NOT interchangeable and a value
   from one is meaningless in the other's space.

   This note previously said only "There is ONE id allocator: fresh_var_id",
   which read as though that counter served everything — and while it said so,
   Sarek_typer really did build one tvar from it (backlog-183). Since
   [float_literal_ids] and [numeric_required_ids] are keyed on tvar ids and read
   inside [unify], that leak could reject a legal program. Enforced now by
   scripts/check-tvar-id-allocator.sh rather than by this paragraph. A
   third one used to appear to live here ([fresh_id] over a [next_var_id] field)
   and was deleted with its field — it had no caller, and a commit message that
   called it live was wrong. The tail-recursion transform draws from the typer
   directly, so a transform id cannot collide with a typer id by construction.
   The only other id convention is the NEGATIVE range for tuple temporaries
   below, disjoint from the typer's by sign. *)

(** Register the synthesized [_tup_*] record for a primitive-component tuple in
    the codegen types table, so a kernel-local slot typed by that record (see
    {!make_var}/{!slot_elttype_of_typ}) has its [struct] definition emitted.
    Idempotent; a no-op for non-tuple or non-primitive-tuple types (the latter
    is rejected at the slot-typing site by {!tuple_record_fields}). The
    realistic data sources for a local tuple (a tuple literal, an [if]/[match]
    whose branches are tuple literals, a tuple-typed vector element) already
    register it elsewhere; this call makes the binding site self-sufficient
    regardless. *)
let register_tuple_type (state : state) (ty : typ) : unit =
  match repr ty with
  | TTuple tys when List.for_all (fun t -> prim_component_elttype t <> None) tys
    ->
      let name, fields = tuple_record_fields tys in
      if not (Hashtbl.mem state.types name) then
        Hashtbl.add state.types name fields
  | _ -> ()

(** Convert Sarek_ast.binop to Sarek_ir_ppx.binop *)
let ir_binop (op : binop) (_ty : typ) : Ir.binop =
  match op with
  | Add -> Ir.Add
  | Sub -> Ir.Sub
  | Mul -> Ir.Mul
  | Div -> Ir.Div
  | Mod -> Ir.Mod
  | Eq -> Ir.Eq
  | Ne -> Ir.Ne
  | Lt -> Ir.Lt
  | Le -> Ir.Le
  | Gt -> Ir.Gt
  | Ge -> Ir.Ge
  | And -> Ir.And
  | Or -> Ir.Or
  | Lsl -> Ir.Shl
  | Lsr ->
      (* Never reached: TEBinop (Lsr, _, _) is intercepted in lower_expr
         and rewritten via lower_lsr into a logical-shift expression tree,
         because Ir.Shr is arithmetic on every backend (see
         briefs/fix-critical-semantics-evidence.md, G phase 1). Kept here
         so this match stays a total, honest structural map of
         Sarek_ast.binop. *)
      Ir.Shr
  | Asr -> Ir.Shr (* arithmetic shift; Ir.Shr is arithmetic on every backend *)
  | Land -> Ir.BitAnd
  | Lor -> Ir.BitOr
  | Lxor -> Ir.BitXor

(** Convert Sarek_ast.unop to Sarek_ir_ppx.unop *)
let ir_unop (op : unop) : Ir.unop =
  match op with Neg -> Ir.Neg | Not -> Ir.Not | Lnot -> Ir.BitNot

(** Convert memspace *)
let lower_memspace = function
  | Local -> Ir.Local
  | Shared -> Ir.Shared
  | Global -> Ir.Global

(** Create a var from typed var info. Uses {!slot_elttype_of_typ} (not the bare
    {!elttype_of_typ}) so a kernel-local binding of primitive-tuple type is
    typed by its synthesized [_tup_*] record rather than the placeholder [int] —
    the struct backends then declare and read the slot as the right compound
    type. Function-typed references still fall through to the placeholder. *)
let make_var name id ty mutable_ : Ir.var =
  {
    var_name = name;
    var_id = id;
    var_type = slot_elttype_of_typ ty;
    var_mutable = mutable_;
  }

(** Lower a declaration *)
let lower_decl ~mutable_ id name ty : Ir.decl =
  let v = make_var name id ty mutable_ in
  Ir.DLocal (v, None)

(** Transform a statement to ensure it returns a value. This adds SReturn to
    leaf statements without re-traversing the original AST. *)
let rec make_returning stmt =
  match stmt with
  | Ir.SReturn _ -> stmt (* Already returns *)
  | Ir.SExpr e -> Ir.SReturn e (* Expression -> return it *)
  | Ir.SIf (c, t, Some e) ->
      Ir.SIf (c, make_returning t, Some (make_returning e))
  | Ir.SIf (c, t, None) ->
      Ir.SIf (c, make_returning t, Some (Ir.SReturn (Ir.EConst Ir.CUnit)))
  | Ir.SMatch (e, cases) ->
      Ir.SMatch (e, List.map (fun (p, b) -> (p, make_returning b)) cases)
  | Ir.SSeq stmts -> (
      match List.rev stmts with
      | [] -> Ir.SReturn (Ir.EConst Ir.CUnit)
      | last :: rest -> Ir.SSeq (List.rev (make_returning last :: rest)))
  | Ir.SLet (v, e, body) -> Ir.SLet (v, e, make_returning body)
  | Ir.SLetMut (v, e, body) -> Ir.SLetMut (v, e, make_returning body)
  | Ir.SPragma (opts, body) -> Ir.SPragma (opts, make_returning body)
  | Ir.SFor _ | Ir.SWhile _ | Ir.SAssign _ | Ir.SBarrier | Ir.SWarpBarrier
  | Ir.SMemFence | Ir.SEmpty | Ir.SNative _ ->
      (* These are side-effect statements; return unit after *)
      Ir.SSeq [stmt; Ir.SReturn (Ir.EConst Ir.CUnit)]
  | Ir.SBlock body -> Ir.SBlock (make_returning body)

(** [true] iff [e] is a syntactically trivial IR expression ([EVar] or
    [EConst]). Trivial expressions have no side effects, so duplicating them in
    the tree built by {!lower_lsr} is semantically inert. *)
let is_trivial_ir_expr (e : Ir.expr) : bool =
  match e with Ir.EVar _ | Ir.EConst _ -> true | _ -> false

(** Lower [a lsr b] (logical/unsigned right shift) to an IR expression tree
    built only from existing IR nodes.

    Ir.Shr is emitted as an *arithmetic* (sign-extending) shift by every
    consumer (CUDA/OpenCL/Metal/GLSL/WGSL emit plain [>>] on a signed C/GLSL int
    type; PTX and the interpreter use [shr.s32]/[Int32.shift_right] - see G
    phase 1 in briefs/fix-critical-semantics-evidence.md). There is no IR node
    for a logical shift and none may be added (formal/codegen-ptx models [Shr]
    itself), so [lsr] is expressed via the classic arithmetic-shift identity,
    width-aware via [width_bits]:

    {[
      lshr (a, n)
      =
      if n = 0 then a
      else
        ashr (a, n) lxor (ashr (a, width - 1) lsl ((width - n) land (width - 1)))
    ]}

    [ashr(a, width-1)] is all-1s when [a] is negative and all-0s otherwise;
    shifted left by [(width - n) land (width - 1)] (equal to [width - n] for
    every [n] in [1..width-1]) it isolates exactly the [n] sign-extended bits
    that [ashr(a, n)] filled in, and XOR-ing them off recovers the zero-filled
    logical shift.

    {b Why the [land (width - 1)] mask, not just an [n = 0] guard.} PTX's [selp]
    and WGSL's [select()] (see [Sarek_ir_ptx_expr.ml] and [Sarek_ir_wgsl.ml])
    evaluate BOTH branches of the resulting [EIf] before selecting one - the
    [EIf] only picks which *value* is used, it does not skip *computing* the
    other branch's subexpressions on those backends. So even though the
    then-branch ([a_ir]) is selected when [n = 0], the else-branch's internal
    [Sub(width_bits, b_ir)] is still evaluated, and without masking it would be
    exactly [width_bits] (shift-by-32/64), which is undefined/rejected on some
    backends. Masking with [width_bits - 1] keeps that shift count in
    [0..width_bits-1] for every [n], including [n = 0] (where it reduces to a
    well-defined shift-by-0, unused anyway since the [EIf] selects [a_ir]),
    while leaving the result unchanged for [n] in [1..width_bits-1] (masking a
    value already in range is a no-op). Shift amounts with [n < 0] or
    [n >= width] are unspecified, matching the pre-existing behaviour of
    [Shl]/[Shr] on out-of-range counts.

    {b Duplication / side-effect safety.} [a_ir] and [b_ir] each appear three
    times in the tree above (in [sign_fill], in [arith_shift]/[top_bits], and in
    the final [EIf]'s branches/condition). [Sarek_ir_ppx.expr] is documented as
    "pure, no side effects" and has no let-binding form (only [Ir.stmt]'s
    [SLet]/[SLetMut] bind values, and those wrap a *statement* continuation, not
    an expression one) - so there is no way to evaluate a subexpression once and
    reuse the result across multiple expression positions. Hoisting via a
    synthetic [EApp] call (which would single-evaluate its arguments) is not
    viable either: PTX - the backend this rewrite specifically targets - does
    not implement device function calls ([Sarek_ir_ptx_expr.ml] rejects [EApp]
    outright). Consequently, if [a_ir] or [b_ir] is *not* trivial (e.g. it
    embeds an [EIntrinsic] atomic call), this tree would silently evaluate that
    operand's side effect multiple times. Rather than accept that, this function
    restricts the rewrite to trivial operands ([EVar]/[EConst], see
    {!is_trivial_ir_expr}) and raises a located PPX error for every other case,
    directing the user to hoist the operand into a `let` before the shift. This
    tree is NOT safe for arbitrary operands - only for trivial ones. *)
let lower_lsr ~(loc : Ppxlib.Location.t) (a_ir : Ir.expr) (b_ir : Ir.expr)
    (ty : Ir.elttype) : Ir.expr =
  if not (is_trivial_ir_expr a_ir && is_trivial_ir_expr b_ir) then
    Ppxlib.Location.raise_errorf
      ~loc
      "lsr: this logical-shift operand is not a plain variable or constant. \
       Sarek_ir_ppx.expr has no let-binding form, so the lsr expansion would \
       evaluate this operand multiple times, silently duplicating any side \
       effect it contains (e.g. an atomic intrinsic call). Bind it to a local \
       variable first and retry, e.g. `let x = <expr> in x lsr n`." ;
  let width_bits = match ty with Ir.TInt64 -> 64 | _ -> 32 in
  let const n =
    match ty with
    | Ir.TInt64 -> Ir.EConst (Ir.CInt64 (Int64.of_int n))
    | _ -> Ir.EConst (Ir.CInt32 (Int32.of_int n))
  in
  let sign_fill = Ir.EBinop (Ir.Shr, a_ir, const (width_bits - 1)) in
  let shift_count =
    Ir.EBinop
      ( Ir.BitAnd,
        Ir.EBinop (Ir.Sub, const width_bits, b_ir),
        const (width_bits - 1) )
  in
  let top_bits = Ir.EBinop (Ir.Shl, sign_fill, shift_count) in
  let arith_shift = Ir.EBinop (Ir.Shr, a_ir, b_ir) in
  let logical_shift = Ir.EBinop (Ir.BitXor, arith_shift, top_bits) in
  Ir.EIf (Ir.EBinop (Ir.Eq, b_ir, const 0), a_ir, logical_shift)

(** Convert a typed expression to IR expression *)
let rec lower_expr (state : state) (te : texpr) : Ir.expr =
  incr ir_lower_expr_count ;
  (* Log progress every 10000 calls *)
  if !ir_lower_expr_count mod 10000 = 0 then
    Sarek_debug.log_to_file
      (Printf.sprintf "    [IR] lower_expr progress: %d" !ir_lower_expr_count) ;
  match te.te with
  | TEUnit -> Ir.EConst Ir.CUnit
  | TEBool b -> Ir.EConst (Ir.CBool b)
  | TEInt i -> Ir.EConst (Ir.CInt32 (Int32.of_int i))
  | TEInt32 i -> Ir.EConst (Ir.CInt32 i)
  | TEInt64 i -> Ir.EConst (Ir.CInt64 i)
  | TEFloat f -> (
      (* L17b: a bare float literal is lowered by its resolved type — float64 if
         context unified it to Float64, float32 otherwise (the default). *)
      match repr te.ty with
      | TReg Float64 -> Ir.EConst (Ir.CFloat64 f)
      | _ -> Ir.EConst (Ir.CFloat32 f))
  | TEDouble f -> Ir.EConst (Ir.CFloat64 f)
  | TEVar (name, id) -> (
      match repr te.ty with
      | TFun _ ->
          (* Function reference - just use the name *)
          let v = make_var name id te.ty false in
          Ir.EVar v
      | _ ->
          let v = make_var name id te.ty false in
          Ir.EVar v)
  | TEVecGet (vec, idx) -> (
      match vec.te with
      | TEVar (name, _) -> Ir.EArrayRead (name, lower_expr state idx)
      | _ -> Ir.EArrayReadExpr (lower_expr state vec, lower_expr state idx))
  | TEArrGet (arr, idx) -> (
      match arr.te with
      | TEVar (name, _) -> Ir.EArrayRead (name, lower_expr state idx)
      | _ -> Ir.EArrayReadExpr (lower_expr state arr, lower_expr state idx))
  | TEFieldGet (r, field, _) -> Ir.ERecordField (lower_expr state r, field)
  | TEBinop (Lsr, a, b) ->
      lower_lsr
        ~loc:(Sarek_ast.loc_to_ppxlib te.te_loc)
        (lower_expr state a)
        (lower_expr state b)
        (elttype_of_typ te.ty)
  | TEBinop (And, a, b) ->
      (* Short-circuit && (audit finding H3): a strict Ir.And evaluates both
         operands eagerly on the PTX and Interpreter backends while the
         C-family backends emit C's short-circuiting &&, so the classic
         [i < n && a.(i) > 0.] bounds guard read out of bounds on PTX and
         raised on the Interpreter. Lowering to EIf gives every backend
         short-circuit semantics (the PTX EIf emitter already refuses to
         evaluate memory-reading/effectful branches speculatively). *)
      Ir.EIf (lower_expr state a, lower_expr state b, Ir.EConst (Ir.CBool false))
  | TEBinop (Or, a, b) ->
      (* Short-circuit || - see the && case above. *)
      Ir.EIf (lower_expr state a, Ir.EConst (Ir.CBool true), lower_expr state b)
  | TEBinop (op, a, b) ->
      Ir.EBinop (ir_binop op te.ty, lower_expr state a, lower_expr state b)
  | TEUnop (op, a) -> Ir.EUnop (ir_unop op, lower_expr state a)
  | TEApp (fn, args) -> (
      let args_ir = List.map (lower_expr state) args in
      match fn.te with
      | TEVar (name, _) when Hashtbl.mem state.fun_map name ->
          (* Module-level function call *)
          if Hashtbl.mem state.lowered_funs name then
            (* Already lowered - use cached *)
            Ir.EApp (Ir.EVar (make_var name 0 fn.ty false), args_ir)
          else if Hashtbl.mem state.lowering_stack name then
            (* Recursive call - emit by name *)
            Ir.EApp (Ir.EVar (make_var name 0 fn.ty false), args_ir)
          else
            (* First time - lower the function *)
            let params, body = Hashtbl.find state.fun_map name in
            let ret_ty = repr body.ty in
            (* Wrong-width class, 3rd instance: a module-level helper whose
               RETURN type is a primitive-component tuple ([let mk x y = (x, y)])
               must be typed through the data mapper, not the bare
               [elttype_of_typ] placeholder that collapses [TTuple]/[TFun] to
               [Ir.TInt32]. The body already lowers a primitive tuple literal to
               the synthesized [_tup_*] record (see the [TETuple] case), so a
               placeholder return type declared the helper int-returning while
               its body returned a compound — a silent miscompile. Routing
               through [slot_elttype_of_typ] lowers the return to that same
               [_tup_*] record end-to-end (backends already support an aggregate
               helper return; see the record-arg/ret helper path). Register the
               synthesized record so its [struct] definition is emitted even if
               no other site did. A non-primitive tuple return raises the located
               tuple-component error rather than miscompiling. *)
            register_tuple_type state ret_ty ;
            let name_of_helper = name in
            (* Captured HERE, where [body] is still the helper's texpr: inside the
               prefixing fold below, [body] is the accumulator statement and
               shadows it. *)
            let helper_loc = Sarek_ast.loc_to_ppxlib body.te_loc in
            Hashtbl.add state.lowering_stack name () ;
            let fun_body_ir = lower_stmt state body in
            Hashtbl.remove state.lowering_stack name ;
            (* Use make_returning to add return statements without re-traversing *)
            let fun_body_ir = make_returning fun_body_ir in
            (* backlog-160: give the helper its own copy of every module constant
               it references. Without this the emitted device function names an
               identifier declared only in the kernel body — see the header above
               [expr_names] for the emitted CUDA and for why hoisting to a
               top-level [const] is unsound.

               TRANSITIVE, and that is not decoration: a constant's initializer
               may reference an earlier constant, so a fixpoint is taken over the
               referenced set before prefixing. Prefixed in DECLARATION order, so
               a dependency is declared before its user. Only the constants
               actually referenced are prefixed — prefixing all of them would
               evaluate initializers the helper does not need and, worse, would
               drag an unreferenced barrier into the refusal path. *)
            let referenced =
              let acc = Hashtbl.create 8 in
              stmt_names fun_body_ir [] acc ;
              let rec close () =
                let added = ref false in
                Hashtbl.iter
                  (fun name _ ->
                    match Hashtbl.find_opt state.mod_consts name with
                    | None -> ()
                    | Some (_, init) ->
                        let sub = Hashtbl.create 4 in
                        expr_names init [] sub ;
                        Hashtbl.iter
                          (fun n () ->
                            if
                              Hashtbl.mem state.mod_consts n
                              && not (Hashtbl.mem acc n)
                            then begin
                              Hashtbl.replace acc n () ;
                              added := true
                            end)
                          sub)
                  (Hashtbl.copy acc) ;
                if !added then close ()
              in
              close () ;
              acc
            in
            (* Names the helper binds anywhere, at any depth. A constant that is
               both referenced AND rebound cannot be prefixed: the backends emit
               [SLet] flat, so the prefix and the rebinding are two declarations
               of one identifier in one block. Refused below rather than emitted.
               Caught by review on #362. *)
            let helper_binders =
              let acc = Hashtbl.create 8 in
              stmt_binders fun_body_ir acc ;
              acc
            in
            let fun_body_ir =
              List.fold_left
                (fun body name ->
                  if not (Hashtbl.mem referenced name) then body
                  else if Hashtbl.mem helper_binders name then
                    Ppxlib.Location.raise_errorf
                      ~loc:helper_loc
                      "Helper %S both references module constant %S and binds \
                       a local of the same name. The constant has to be \
                       declared inside the helper to be visible there, and the \
                       generated device code declares locals in one flat \
                       scope, so that would emit two declarations of %S in the \
                       same block. Rename the local, or pass the constant in \
                       as a parameter of %S."
                      name_of_helper
                      name
                      name
                      name_of_helper
                  else
                    match Hashtbl.find_opt state.mod_consts name with
                    | None -> body
                    | Some (v, init) -> (
                        match expr_barrier init with
                        | Some intr ->
                            (* Duplicating a barrier per call site changes
                               convergence, so this is refused rather than
                               silently changed. Names both the constant and the
                               intrinsic: "a barrier somewhere" is not actionable.

                               LOCATED at the helper's body, not
                               [Location.none]. The first version passed
                               [Location.none] while the commit message and the
                               PR body both claimed "a located error" — the claim
                               was wider than the code, and a refusal the user
                               cannot place in their source is barely better than
                               a silent one. This file already threads real
                               locations for comparable refusals (the [lsr] error
                               and the shared-memory rejections), so there was no
                               reason to drop it here. Caught by review on #362. *)
                            Ppxlib.Location.raise_errorf
                              ~loc:helper_loc
                              "Module constant %S is referenced by helper %S \
                               and its initializer calls %S, a synchronising \
                               operation. Making it visible to the helper \
                               means evaluating the initializer once per call, \
                               which would execute %S once per call site and \
                               change convergence. Move the value into a \
                               parameter of %S, or compute it in the kernel \
                               body and pass it in."
                              name
                              name_of_helper
                              intr
                              intr
                              name_of_helper
                        | None -> Ir.SSeq [Ir.SLet (v, init, Ir.SEmpty); body]))
                fun_body_ir
                (List.rev state.mod_consts_order)
            in
            (* Convert tparam list to var list *)
            (* backlog-158: the parameter's identity is the TYPER's id, not its
               position. This was [List.mapi] handing each parameter its index
               while [p.tparam_id] — the id every use site inside the body
               carries — was destructured and discarded. The interpreter's
               [lookup_var] resolves by id before name, so a reference resolved to
               whichever parameter occupied that slot, and the name fallback was
               silently load-bearing: it only rescued the cases where the id
               lookup missed entirely.

               Wrong exactly for 1 <= c <= n-1, where c is the global typer
               counter when these parameters are allocated and n their count
               (c = 0 makes positional and typer ids coincide by accident; c >= n
               makes every lookup miss and fall back safely). A module constant
               declared ahead of the helper puts c at 1, which is why the probe
               needs one and why it must be the first kernel in a fresh file — the
               counter is global and persists.

               Measured, not reasoned: `combine a b c d = a +. b +. c` with
               arguments 1 2 3 4 returned 9 on the Interpreter and 6 on OpenCL,
               Vulkan and Native. Safe to change because every other consumer of
               [hf_params] keys by NAME (PTX, GLSL, WGSL, inline_vec) and no
               golden pins these ids. *)
            let hf_params =
              List.map
                (fun (p : tparam) ->
                  make_var p.tparam_name p.tparam_id p.tparam_type false)
                params
            in
            let helper_func : Ir.helper_func =
              {
                hf_name = name;
                hf_params;
                hf_ret_type = slot_elttype_of_typ ret_ty;
                hf_body = fun_body_ir;
              }
            in
            Hashtbl.add state.lowered_funs name helper_func ;
            state.lowered_funs_order <- name :: state.lowered_funs_order ;
            Ir.EApp (Ir.EVar (make_var name 0 fn.ty false), args_ir)
      | _ -> Ir.EApp (lower_expr state fn, args_ir))
  | TERecord (name, fields) ->
      (* Register the record type if not already registered *)
      if not (Hashtbl.mem state.types name) then begin
        let field_types =
          List.map (fun (n, e) -> (n, elttype_of_typ e.ty)) fields
        in
        Hashtbl.add state.types name field_types
      end ;
      Ir.ERecord (name, List.map (fun (n, e) -> (n, lower_expr state e)) fields)
  | TEConstr (ty_name, constr, arg) ->
      (* Register the variant type if not already registered.
         Get constructors from the expression's type (which has full variant info) *)
      if not (Hashtbl.mem state.variants ty_name) then begin
        match repr te.ty with
        | TVariant (_, constrs) ->
            let constr_types =
              List.map
                (fun (cname, ty_opt) -> (cname, constr_payload_elttypes ty_opt))
                constrs
            in
            Hashtbl.add state.variants ty_name constr_types
        | _ -> ()
      end ;
      (* Flatten a multi-argument constructor's tuple payload into one IR
         argument per component, matching the flattened field registration
         above and the multi-binder pattern side. A literal tuple whose
         components are all scalar primitives is the shape the typer produces
         for [MkPair (a, b)]; everything else stays a single argument. *)
      let args =
        match arg with
        | None -> []
        | Some {te = TETuple comps; ty = tup_ty; _}
          when match repr tup_ty with
               | TTuple tys -> List.for_all is_scalar_prim_typ tys
               | _ -> false ->
            List.map (lower_expr state) comps
        | Some e -> [lower_expr state e]
      in
      Ir.EVariant (ty_name, constr, args)
  | TETuple exprs -> (
      (* L13: a tuple literal whose components are scalar primitives is lowered
         to the same synthesized record ([_0], [_1], ...) used for tuple vector
         elements. This carries the nominal type name so struct-based backends
         (OpenCL/GLSL/CUDA/Metal) emit a typed compound literal rather than a
         bare, type-less brace initializer, and it matches what a tuple vector
         slot stores. Non-primitive tuples keep the generic [Ir.ETuple]. *)
      match repr te.ty with
      | TTuple tys
        when List.for_all (fun t -> prim_component_elttype t <> None) tys ->
          let comps =
            List.map
              (fun t ->
                match prim_component_elttype t with
                | Some e -> e
                | None -> assert false (* guarded above *))
              tys
          in
          let name = tuple_record_name comps in
          if not (Hashtbl.mem state.types name) then
            Hashtbl.add
              state.types
              name
              (List.mapi (fun i e -> (tuple_field_name i, e)) comps) ;
          Ir.ERecord
            ( name,
              List.mapi
                (fun i e -> (tuple_field_name i, lower_expr state e))
                exprs )
      | _ -> Ir.ETuple (List.map (lower_expr state) exprs))
  | TEGlobalRef (name, ty) ->
      let v = make_var name 0 ty false in
      Ir.EVar v
  | TEIntrinsicConst ref -> (
      match ref with
      | Sarek_env.IntrinsicRef (path, name) -> Ir.EIntrinsic (path, name, [])
      | Sarek_env.CorePrimitiveRef name -> Ir.EIntrinsic ([], name, []))
  (* The two f16 width conversions are the only primitives that lower to a
     typed IR cast instead of an intrinsic call. Going through [ECast] (rather
     than an [EIntrinsic] carrying a device format string) is what lets each
     backend emit its own documented narrowing -- CUDA/HIP __float2half, and on
     the interpreter a narrowing through Sarek_float16 -- and it is also what
     makes [Sarek_ir_analysis.kernel_uses_float16] see the conversion, since its
     leaf inspects [ECast] target types. See the "conv_f16" primitives in
     Sarek_core_primitives.ml. *)
  | TEIntrinsicFun (Sarek_env.CorePrimitiveRef "float16_of_float32", _, [arg])
    ->
      Ir.ECast (Ir.TFloat16, lower_expr state arg)
  | TEIntrinsicFun (Sarek_env.CorePrimitiveRef "float32_of_float16", _, [arg])
    ->
      Ir.ECast (Ir.TFloat32, lower_expr state arg)
  | TEIntrinsicFun (ref, _conv, args) -> (
      match ref with
      | Sarek_env.IntrinsicRef (path, name) ->
          Ir.EIntrinsic (path, name, List.map (lower_expr state) args)
      | Sarek_env.CorePrimitiveRef name ->
          Ir.EIntrinsic ([], name, List.map (lower_expr state) args))
  (* If-then-else as expression (returns a value) *)
  | TEIf (cond, then_, Some else_) ->
      Ir.EIf
        (lower_expr state cond, lower_expr state then_, lower_expr state else_)
  | TEIf (cond, then_, None) ->
      (* No else branch - only valid for unit-returning expressions *)
      Ir.EIf (lower_expr state cond, lower_expr state then_, Ir.EConst Ir.CUnit)
  (* Match as expression *)
  | TEMatch (e, _)
    when (match repr e.ty with TTuple _ -> true | _ -> false)
         && not (is_primitive_tuple e.ty) ->
      (* Same non-primitive tuple-scrutinee guard as the statement path (see
         {!lower_stmt}); reachable when a non-primitive tuple match is used in
         value position. *)
      raise_tuple_component_error ~loc:(Sarek_ast.loc_to_ppxlib e.te_loc)
  | TEMatch (e, cases) ->
      let ir_cases =
        List.map
          (fun (pat, body) -> (lower_pattern pat, lower_expr state body))
          cases
      in
      Ir.EMatch (lower_expr state e, ir_cases)
  (* These expression forms require statement context - should be caught by typer *)
  | TEVecSet _ | TEArrSet _ | TEFieldSet _ | TEAssign _ | TELet _ | TELetRec _
  | TELetMut _ | TEFor _ | TEWhile _ | TESeq _ | TEReturn _ | TECreateArray _
  | TENative _ | TEPragma _ | TELetShared _ | TESuperstep _ | TEOpen _ ->
      failwith
        "Internal error: lower_expr called with statement-only expression. \
         This should have been caught by the type checker."

(** Convert a typed expression to IR statement *)
and lower_stmt (state : state) (te : texpr) : Ir.stmt =
  incr ir_lower_stmt_count ;
  match te.te with
  | TEUnit -> Ir.SEmpty
  | TESeq [] -> Ir.SEmpty
  | TESeq [e] -> lower_stmt state e
  | TESeq es -> Ir.SSeq (List.map (lower_stmt state) es)
  | TEVecSet (vec, idx, value) -> (
      match vec.te with
      | TEVar (name, _id) ->
          Ir.SAssign
            (Ir.LArrayElem (name, lower_expr state idx), lower_expr state value)
      | _ ->
          (* Complex base expression - use LArrayElemExpr *)
          Ir.SAssign
            ( Ir.LArrayElemExpr (lower_expr state vec, lower_expr state idx),
              lower_expr state value ))
  | TEArrSet (arr, idx, value) -> (
      match arr.te with
      | TEVar (name, _id) ->
          Ir.SAssign
            (Ir.LArrayElem (name, lower_expr state idx), lower_expr state value)
      | _ ->
          (* Complex base expression - use LArrayElemExpr *)
          Ir.SAssign
            ( Ir.LArrayElemExpr (lower_expr state arr, lower_expr state idx),
              lower_expr state value ))
  | TEFieldSet (r, field, _, value) ->
      let lv = lower_lvalue state r field in
      Ir.SAssign (lv, lower_expr state value)
  | TEAssign (name, id, value) ->
      let v = make_var name id value.ty true in
      Ir.SAssign (Ir.LVar v, lower_expr state value)
  | TELet (name, id, value, body) -> (
      match value.te with
      (* Special case: create_array - need proper array declaration *)
      | TECreateArray (size, elem_ty, mem) ->
          let size_ir = lower_expr state size in
          let v = make_var name id (TArr (elem_ty, mem)) false in
          let body_ir = lower_stmt state body in
          Ir.SLet
            ( v,
              Ir.EArrayCreate
                (elttype_of_typ elem_ty, size_ir, memspace_of_memspace mem),
              body_ir )
      (* Normal let binding *)
      | _ ->
          register_tuple_type state value.ty ;
          let v = make_var name id value.ty false in
          Ir.SLet (v, lower_expr state value, lower_stmt state body))
  | TELetMut (name, id, value, body) ->
      register_tuple_type state value.ty ;
      let v = make_var name id value.ty true in
      Ir.SLetMut (v, lower_expr state value, lower_stmt state body)
  | TELetRec (name, _id, params, fn_body, cont) ->
      (* Register function in fun_map for later inlining when called *)
      Hashtbl.add state.fun_map name (params, fn_body) ;
      lower_stmt state cont
  | TEIf (cond, then_, else_opt) ->
      Ir.SIf
        ( lower_expr state cond,
          lower_stmt state then_,
          Option.map (lower_stmt state) else_opt )
  | TEFor (var, id, lo, hi, dir, body) ->
      let v = make_var var id (TPrim TInt32) true in
      let ir_dir = match dir with Upto -> Ir.Upto | Downto -> Ir.Downto in
      Ir.SFor
        ( v,
          lower_expr state lo,
          lower_expr state hi,
          ir_dir,
          lower_stmt state body )
  | TEWhile (cond, body) ->
      Ir.SWhile (lower_expr state cond, lower_stmt state body)
  | TEMatch (e, [({tpat = TPTuple pats; _}, body)])
    when is_primitive_tuple e.ty
         && List.for_all
              (fun p ->
                match p.tpat with TPVar _ | TPAny -> true | _ -> false)
              pats ->
      (* L13: a single-arm tuple-pattern match is not a variant dispatch; the
         C-like backends (OpenCL/GLSL/CUDA/Metal) would otherwise emit a bogus
         [switch (x.tag)]. Rewrite it to a record destructure: bind the
         scrutinee once, then bind each component to its [_i] field. This works
         uniformly on every backend (field access is already supported). *)
      register_tuple_type state e.ty ;
      let rec_elt = slot_elttype_of_typ e.ty in
      incr tuple_tmp_counter ;
      let tmp_name = Printf.sprintf "__sarek_tup_%d" !tuple_tmp_counter in
      let tmp_var : Ir.var =
        {
          var_name = tmp_name;
          var_id = - !tuple_tmp_counter;
          var_type = rec_elt;
          var_mutable = false;
        }
      in
      let body_ir = lower_stmt state body in
      let bound =
        List.fold_right
          (fun (i, p) acc ->
            match p.tpat with
            | TPVar (name, id) ->
                let field =
                  Ir.ERecordField (Ir.EVar tmp_var, tuple_field_name i)
                in
                Ir.SLet (make_var name id p.tpat_ty false, field, acc)
            | _ -> acc)
          (List.mapi (fun i p -> (i, p)) pats)
          body_ir
      in
      Ir.SLet (tmp_var, lower_expr state e, bound)
  | TEMatch (e, _)
    when (match repr e.ty with TTuple _ -> true | _ -> false)
         && not (is_primitive_tuple e.ty) ->
      (* A tuple-typed match scrutinee that did NOT take the primitive
         single-arm destructure path above is a NON-PRIMITIVE tuple (nested
         tuple / record / vector / function component), reachable with a
         non-variable scrutinee that never passes through [slot_elttype_of_typ]
         (e.g. [let ((a, b), c) = ((x, y), z) in ...], which the parser desugars
         to this match, or the equivalent [match] spelling). Without this guard
         it would lower to [Ir.ETuple] + an [SMatch] and die as a confusing
         backend C error ([switch ((...).tag)]). Raise the same located
         tuple-component error the slot path raises. Only tuple scrutinees reach
         here; variant matches have a [TVariant] scrutinee and fall through. *)
      raise_tuple_component_error ~loc:(Sarek_ast.loc_to_ppxlib e.te_loc)
  | TEMatch (e, cases) ->
      let ir_cases =
        List.map
          (fun (pat, body) -> (lower_pattern pat, lower_stmt state body))
          cases
      in
      Ir.SMatch (lower_expr state e, ir_cases)
  | TEReturn e -> Ir.SReturn (lower_expr state e)
  | TEPragma (opts, body) -> Ir.SPragma (opts, lower_stmt state body)
  | TELetShared (name, id, elem_ty, size_opt, body) ->
      (* Shared memory declaration: __shared__ type name[size]; or __local type name[size]; *)
      let size_ir =
        match size_opt with
        | Some size -> lower_expr state size
        | None -> Ir.EIntrinsic (["Sarek_stdlib"; "Gpu"], "block_dim_x", [])
      in
      let v = make_var name id (TArr (elem_ty, Sarek_types.Shared)) false in
      (* Sibling of the helper-return wrong-width fix, REJECTED (round 2). A
         [let%shared] whose ELEMENT type is a tuple/function was originally typed
         by the bare [elttype_of_typ] placeholder, which collapses [TTuple]/[TFun]
         to [Ir.TInt32] — a silent scalar-collapse miscompile. Routing it through
         [slot_elttype_of_typ] to the synthesized [_tup_*] record (as data slots
         do) was attempted, but a compound in shared memory is NOT supported by
         the whole backend fleet: the PTX backend raises "unsupported construct:
         btype of custom type" and Native raises "Cannot create default value for
         this type" (proven on hardware: OpenCL/Vulkan/Interpreter passed,
         CUDA/PTX-under-ZLUDA and Native failed; the rejection is locked by
         sarek/tests/negative/test_shared_tuple.ml). Shipping a route that
         miscompiles on shared-capable
         devices is worse than a clean rejection, so a tuple/function shared
         element is a located compile error, mirroring [lower_param]'s
         parameter-boundary rejection. (Aggregate shared arrays are a follow-up
         needing per-backend struct-in-__shared__ support first.) *)
      (match repr elem_ty with
      | TTuple _ ->
          Ppxlib.Location.raise_errorf
            ~loc:(Sarek_ast.loc_to_ppxlib te.te_loc)
            "Tuple-typed shared-memory arrays are not supported; declare \
             separate scalar [let%%shared] arrays for each component (a \
             compound in shared memory is not supported across the CUDA/PTX \
             and Native backends)."
      | TFun _ ->
          Ppxlib.Location.raise_errorf
            ~loc:(Sarek_ast.loc_to_ppxlib te.te_loc)
            "Function-typed shared-memory arrays are not supported."
      | _ -> ()) ;
      let elem_ir = elttype_of_typ elem_ty in
      (* Use EArrayCreate with Shared memspace - codegen will emit proper declaration *)
      Ir.SLet
        (v, Ir.EArrayCreate (elem_ir, size_ir, Ir.Shared), lower_stmt state body)
  | TESuperstep (_name, _divergent, step_body, cont) ->
      (* Wrap step_body in SBlock to create C scope for variable isolation *)
      Ir.SSeq
        [
          Ir.SBlock (lower_stmt state step_body);
          Ir.SBarrier;
          lower_stmt state cont;
        ]
  | TEOpen (_path, body) -> lower_stmt state body
  | TECreateArray (_size, _elem_ty, _mem) ->
      (* Standalone array creation - just emit unit *)
      Ir.SExpr (Ir.EConst Ir.CUnit)
  | TENative {gpu; ocaml} -> Ir.SNative {gpu; ocaml}
  (* Pure expressions as statements *)
  | TEBool _ | TEInt _ | TEInt32 _ | TEInt64 _ | TEFloat _ | TEDouble _
  | TEVar _ | TEVecGet _ | TEArrGet _ | TEFieldGet _ | TEBinop _ | TEUnop _
  | TEApp _ | TERecord _ | TEConstr _ | TETuple _ | TEGlobalRef _
  | TEIntrinsicConst _ | TEIntrinsicFun _ ->
      Ir.SExpr (lower_expr state te)

and lower_lvalue (state : state) (r : texpr) (field : string) : Ir.lvalue =
  match r.te with
  | TEVar (name, id) ->
      let v = make_var name id r.ty false in
      Ir.LRecordField (Ir.LVar v, field)
  | TEFieldGet (base, inner_field, _) ->
      Ir.LRecordField (lower_lvalue state base inner_field, field)
  | TEVecGet (vec, idx) -> (
      match vec.te with
      | TEVar (name, _) ->
          Ir.LRecordField (Ir.LArrayElem (name, lower_expr state idx), field)
      | _ ->
          Ir.LRecordField
            ( Ir.LArrayElemExpr (lower_expr state vec, lower_expr state idx),
              field ))
  | TEArrGet (arr, idx) -> (
      match arr.te with
      | TEVar (name, _) ->
          Ir.LRecordField (Ir.LArrayElem (name, lower_expr state idx), field)
      | _ ->
          Ir.LRecordField
            ( Ir.LArrayElemExpr (lower_expr state arr, lower_expr state idx),
              field ))
  | _ ->
      failwith
        "Internal error: lower_lvalue called with non-lvalue expression. This \
         should have been caught by the type checker."

and lower_pattern (pat : tpattern) : Ir.pattern =
  match pat.tpat with
  | TPAny -> Ir.PWild
  | TPVar (name, _) -> Ir.PConstr ("", [name])
  | TPConstr (_ty_name, constr, arg_pat) ->
      let vars =
        match arg_pat with None -> [] | Some p -> extract_pattern_vars p
      in
      Ir.PConstr (constr, vars)
  | TPTuple pats ->
      Ir.PConstr ("tuple", List.concat_map extract_pattern_vars pats)

and extract_pattern_vars (pat : tpattern) : string list =
  match pat.tpat with
  | TPAny -> ["_"]
  | TPVar (name, _) -> [name]
  | TPConstr (_, _, Some p) -> extract_pattern_vars p
  | TPConstr (_, _, None) -> []
  | TPTuple pats -> List.concat_map extract_pattern_vars pats

(** Convert a kernel parameter to IR declaration *)
let lower_param (p : tparam) : Ir.decl =
  (match repr p.tparam_type with
  | TReg Float16 ->
      (* #57 slice 1: the delivered surface is `float16 vector`, NOT a scalar
         f16 parameter — and nothing else in the pipeline stops one.

         It type-checks, and Sarek_ir_cuda maps it to a by-value `__half`
         formal. But [Execute.vector_arg] has no float16 constructor, so the
         only way to supply an argument is [Float32 f], which becomes
         [ArgFloat32] and pushes a 4-byte C float whose address the device then
         reads as a 2-byte __half. Executed on gfx1100 with
         `fun (out : float16 vector) (s : float16) -> out.(tid) <- s` and
         `Float32 3.14159`: HIP produced 0.000476837158 with no error raised,
         while the interpreter produced the correct 3.140625 — the two oracles
         silently disagreed, which is the exact property test_hip_f16 exists to
         guarantee.

         Same class as the f16 rejections already in place for record fields
         (Sarek_ir_layout), SoA fields (Soa.ml) and whole kernels (the five
         backend gates); scalar params were the hole. *)
      Ppxlib.Location.raise_errorf
        ~loc:Ppxlib.Location.none
        "Kernel parameter %S has type float16: f16 is a storage-only element \
         type and cannot be a scalar kernel parameter. Pass a float32 scalar \
         and narrow inside the kernel with float16_of_float32, or use a \
         `float16 vector`."
        p.tparam_name
  | TTuple _ ->
      Ppxlib.Location.raise_errorf
        ~loc:Ppxlib.Location.none
        "Tuple-typed kernel parameters are not supported; pass components as \
         separate parameters."
  | TFun _ ->
      Ppxlib.Location.raise_errorf
        ~loc:Ppxlib.Location.none
        "Function-typed kernel parameters are not supported."
  | TArr (t, _) -> (
      (* An array of tuples or functions would silently collapse to TInt32 in
         elttype_of_typ (wrong stride, garbage layout) — reject at the
         formal-parameter boundary. (L13: local-array-of-tuple aggregate
         support is a follow-up; only global vector elements are lowered.) *)
      match repr t with
      | TTuple _ ->
          Ppxlib.Location.raise_errorf
            ~loc:Ppxlib.Location.none
            "Arrays of tuples are not supported as kernel parameters; declare \
             a record type with [@@sarek.type] instead, or use a vector."
      | TFun _ ->
          Ppxlib.Location.raise_errorf
            ~loc:Ppxlib.Location.none
            "Arrays of functions are not supported as kernel parameters."
      | _ -> ())
  | TVec t -> (
      match repr t with
      | TFun _ ->
          Ppxlib.Location.raise_errorf
            ~loc:Ppxlib.Location.none
            "Vectors of functions are not supported as kernel parameters."
      | TTuple tys ->
          (* L13: a vector of tuples is lowered to a vector of the synthesized
             packed record; validate the components eagerly for a clear error. *)
          ignore (tuple_record_fields tys)
      | _ -> ())
  | _ -> ()) ;
  let elt =
    match repr p.tparam_type with
    | TVec t -> Ir.TVec (slot_elttype_of_typ t)
    | _ -> elttype_of_typ p.tparam_type
  in
  let v =
    {
      Ir.var_name = p.tparam_name;
      var_id = p.tparam_id;
      var_type = elt;
      var_mutable = false;
    }
  in
  if p.tparam_is_vec then
    let elem_ty =
      match repr p.tparam_type with TVec t -> slot_elttype_of_typ t | _ -> elt
    in
    Ir.DParam (v, Some {arr_elttype = elem_ty; arr_memspace = Ir.Global})
  else Ir.DParam (v, None)

(** Lower a complete kernel *)
let lower_kernel (kernel : tkernel) : Ir.kernel =
  (* Reset and log counters *)
  ir_lower_expr_count := 0 ;
  ir_lower_stmt_count := 0 ;
  let kern_name = Option.value kernel.tkern_name ~default:"anon" in
  Sarek_debug.log_to_file
    (Printf.sprintf "  [IR] lower_kernel start: %s" kern_name) ;
  (* Build fun_map from module items (local + registry) *)
  let fun_map = Hashtbl.create 8 in
  let all_mod_items = kernel.tkern_module_items in
  List.iter
    (function
      | TMFun (name, _, params, body) ->
          Hashtbl.replace fun_map name (params, body)
      | _ -> ())
    all_mod_items ;
  let state = create_state fun_map in

  (* Register record types from parameter types (especially vector element types) *)
  let rec register_types_from_typ ty =
    match repr ty with
    | TRecord (name, fields) ->
        if not (Hashtbl.mem state.types name) then begin
          let field_types =
            List.map (fun (n, t) -> (n, elttype_of_typ t)) fields
          in
          Hashtbl.add state.types name field_types
        end ;
        List.iter (fun (_, t) -> register_types_from_typ t) fields
    | TVec elem_ty -> register_types_from_typ elem_ty
    | TArr (elem_ty, _) -> register_types_from_typ elem_ty
    | TTuple tys ->
        (* L13: register the synthesized record for a tuple aggregate so the
           codegen types table knows its field layout, mirroring records.
           Primitivity is checked on the SOURCE types (prim_component_elttype),
           never via elttype_of_typ, whose TTuple/TFun placeholder would let a
           nested tuple register as an int32 field. *)
        (match
           List.map prim_component_elttype tys |> fun opts ->
           if List.for_all Option.is_some opts then
             Some (List.map Option.get opts)
           else None
         with
        | Some comps ->
            let name = tuple_record_name comps in
            if not (Hashtbl.mem state.types name) then
              Hashtbl.add
                state.types
                name
                (List.mapi (fun i e -> (tuple_field_name i, e)) comps)
        | None -> ()) ;
        List.iter register_types_from_typ tys
    | _ -> ()
  in
  List.iter
    (fun (p : tparam) -> register_types_from_typ p.tparam_type)
    kernel.tkern_params ;

  (* Lower module-level constants *)
  let module_items_ir =
    List.fold_right
      (fun item acc ->
        match item with
        | TMConst (name, id, ty, expr) ->
            let v = make_var name id ty false in
            let init = lower_expr state expr in
            (* backlog-160: record it so a helper body that names this constant
               can carry its own SLet. Populated HERE, while the kernel-body copy
               is built, which is before the body is lowered — and helpers are
               lowered lazily FROM the body, so by the time any helper needs a
               constant, every constant is registered.

               The residual ordering case, checked and stated rather than left
               implicit: a helper can also be lowered while a CONSTANT's own
               initializer is lowered (a constant may call a module function). At
               that moment the constants declared after it are not yet in the
               table, so such a helper gets only the earlier ones. That is the
               correct scope — OCaml's own scoping gives the initializer access to
               earlier bindings only — so the limitation matches the language
               rather than being a gap. A helper referencing a LATER constant is
               not expressible. *)
            Hashtbl.replace state.mod_consts name (v, init) ;
            state.mod_consts_order <- name :: state.mod_consts_order ;
            Ir.SSeq [Ir.SLet (v, init, Ir.SEmpty); acc]
        | TMFun _ -> acc)
      all_mod_items
      Ir.SEmpty
  in

  (* Lower body *)
  let body_ir = lower_stmt state kernel.tkern_body in
  Sarek_debug.log_to_file
    (Printf.sprintf
       "  [IR] lower_kernel done: %s (expr=%d, stmt=%d)"
       kern_name
       !ir_lower_expr_count
       !ir_lower_stmt_count) ;
  let full_body =
    match module_items_ir with
    | Ir.SEmpty -> body_ir
    | _ -> Ir.SSeq [module_items_ir; body_ir]
  in

  (* Collect types from state *)
  let types_list =
    Hashtbl.fold (fun name fields acc -> (name, fields) :: acc) state.types []
  in
  (* Collect variant types from state *)
  let variants_list =
    Hashtbl.fold
      (fun name constrs acc -> (name, constrs) :: acc)
      state.variants
      []
  in
  (* Collect helper functions from state, in dependency order *)
  let funcs_list =
    List.rev state.lowered_funs_order
    |> List.filter_map (fun name -> Hashtbl.find_opt state.lowered_funs name)
  in
  {
    Ir.kern_name = Option.value kernel.tkern_name ~default:"sarek_kern";
    (* "kernel" is reserved in OpenCL *)
    kern_params = List.map lower_param kernel.tkern_params;
    kern_locals = [];
    kern_body = full_body;
    kern_types = types_list;
    kern_variants = variants_list;
    kern_funcs = funcs_list;
    kern_native_fn = None;
    (* Native fn is added during quoting *)
  }

(** Get the return value declaration for a kernel *)
let lower_return_value (kernel : tkernel) : Ir.decl option =
  match repr kernel.tkern_return_type with
  | TPrim TUnit -> None
  | ty ->
      let v = make_var "result" 0 ty true in
      Some (Ir.DLocal (v, None))
