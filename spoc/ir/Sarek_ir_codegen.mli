(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Shared code-generation helpers used across GPU backends (CUDA, OpenCL,
    Metal, Vulkan/GLSL). Extracted to avoid duplication of identical
    variant/struct generation logic. *)

(** [reject_feature ~raise_ ~backend ?hint feature k] raises, via [raise_], when
    kernel [k] uses a numeric width [feature] that this backend has not
    implemented yet.

    WHY A WHOLE-KERNEL GATE, and not just the backend's per-element-type match
    arms: those arms only run when the emitter actually asks for a type string,
    and several positions never do. PTX validated aggregate vector element types
    through a [| _ -> ()] fall-through and never inspected [hf_ret_type], so an
    f16 vector parameter the body did not read, or an f16 helper return type,
    produced a complete, valid, silently-wrong module with no diagnostic at all.
    {!Sarek_ir_analysis.kernel_uses} folds params, locals, body, helper params
    AND return types, and record and variant field types, which closes the
    class.

    THIS IS WHERE THE NEXT WIDTH GETS WIRED IN. Adding bf16 should be a
    constructor in {!Sarek_ir_analysis.feature} plus one partial application per
    backend.

    [raise_] is a parameter because each backend raises through its own error
    functor (which stamps the backend tag) and [spoc/ir] deliberately has no
    backend dependencies; it receives the composed reason string. [?hint] is
    omitted by backends whose deferral has no actionable hint. Call it once per
    backend, partially applied, so the [generate] entries stay one-liners. *)
val reject_feature :
  raise_:(string -> unit) ->
  backend:string ->
  ?hint:string ->
  Sarek_ir_analysis.feature ->
  Sarek_ir_types.kernel ->
  unit

(** {1 Measured f16 refusals (#57 slices 2a/2b, single-sourced by #138)}

    Two backends left {!reject_feature} because "not yet supported" was false of
    them: their refusal is a MEASUREMENT, not a queue position. Each is refused
    in two places — the per-element-type arm of [<backend>_type_of_elttype] and
    the whole-kernel [reject_float16_kernel] — and each sentence used to be
    written out at both sites.

    That is not a DRY nit. The sentence carries the number that justifies the
    refusal, and the two OpenCL copies had already drifted apart before this
    constant existed: one opened "float16 is not supported — not because the
    codegen is missing", the other "float16 is refused by measurement, not
    pending implementation", and only the second was the wording the docs and
    the goldens cite. Two copies of a measured claim become two claims.

    Single-sourcing also fixes the direction that matters next: when Mesa stops
    fusing and one of these refusals is lifted, one constant makes every
    remaining reference to it obviously stale. Duplicates just rot.

    The verbatim expectations in
    sarek/tests/codegen_golden/test_cuda_f16_golden.ml are NOT a third copy in
    that sense — they are the golden, deliberately spelled out so that
    re-folding a backend into the shared composer breaks a test rather than
    passing silently, and one case there asserts both sites of each backend
    agree. *)

(** The OpenCL f16 refusal diagnostic. Raised by both
    [Sarek_ir_opencl.opencl_type_of_elttype] and
    [Sarek_ir_opencl.reject_float16_kernel]. *)
val opencl_float16_refusal : string

(** The GLSL f16 refusal diagnostic. Raised by both
    [Sarek_ir_glsl.glsl_type_of_elttype] and
    [Sarek_ir_glsl.reject_float16_kernel]. *)
val glsl_float16_refusal : string

(** Mangle an OCaml type name into a valid C/GLSL identifier (e.g.
    "Module.point" -> "Module_point"). Replaces '.' with '_'. *)
val mangle_name : string -> string

(** Alpha-rename kernel-body binders whose name collides with a backend-reserved
    identifier, rewriting each colliding binder (and its in-scope references) to
    a fresh name so it no longer aliases a scalar param exposed by name. Covers
    [SLet], [SLetMut], [SFor], and [SMatch]/[EMatch] pattern binders; a no-op
    for collision-free kernels.

    [collides name] reports whether [name] (as written in the IR) is reserved
    for this backend. [fresh_name orig n] mints the [n]-th shadow name for
    original binder [orig] ([n] is 1-based, incremented once per colliding
    binder). The counter is internal and starts at 0 on every call, so a single
    per-kernel invocation numbers shadows from 1.

    Shared by the GLSL (push-constant / vector-[_len] macros,
    [sarek_pc_shadow_*]) and WGSL (scalar-param field access,
    [sarek_scalar_shadow_*]) backends. *)
val rename_shadowing_locals :
  collides:(string -> bool) ->
  fresh_name:(string -> int -> string) ->
  Sarek_ir_types.stmt ->
  Sarek_ir_types.stmt

(** {1 Match-expression payload bindings}

    A match STATEMENT ([SMatch]) lowers to a [switch] whose arms are blocks, so
    each backend opens the arm with real destructuring declarations
    ([T r = <scrut>.data.C_v;]). A match EXPRESSION ([EMatch]) lowers to a
    nested ternary / [select()] and has nowhere to declare anything, so every
    backend used to emit the case body with its payload binders left dangling.
    On the shader backends that is a device-compiler error; on the C-family
    backends it is SILENT-WRONG when an unrelated same-named variable is in
    scope — the kernel returns a plausible wrong answer with no diagnostic
    (#75). Expression position admits exactly one binding mechanism,
    substituting the binder by the payload read.

    {!payload_layout} is the single description of where a payload lives, and it
    feeds BOTH the substitution and each backend's [SMatch] declaration
    ({!payload_suffix}), so the two paths cannot drift apart (#94). *)

(** Where a constructor's payloads live inside the emitted variant value. The
    backends genuinely differ — WGSL flattens a multi-payload constructor into
    indexed sibling fields where the others nest — and assuming otherwise emits
    a field that does not exist. *)
type payload_layout = {
  union : string option;
      (** Member the payloads sit under ([Some "data"] for the C-family tagged
          union, [None] where they are fields of the variant struct itself). *)
  indexed : bool;
      (** [true] when payload [i] of a multi-payload constructor is the sibling
          field [<C>_v_<i>]; [false] when it is [<C>_v._<i>]. *)
}

(** CUDA / OpenCL / Metal: [<scrut>.data.<C>_v], [<scrut>.data.<C>_v._<i>]. *)
val c_family_payload_layout : payload_layout

(** GLSL: [<scrut>.<C>_v], [<scrut>.<C>_v._<i>] (multi-payload is a hoisted
    named struct — see {!gen_variant_def_glsl}). *)
val glsl_payload_layout : payload_layout

(** WGSL: [<scrut>.<C>_v], [<scrut>.<C>_v_<i>] — no union and no nested struct.
*)
val wgsl_payload_layout : payload_layout

(** [payload_fields layout ~cname ~arity i] is the field chain from a scrutinee
    value down to payload [i] of [cname], whose pattern binds [arity] names. A
    one-payload constructor is [<C>_v] under every layout. *)
val payload_fields :
  payload_layout -> cname:string -> arity:int -> int -> string list

(** The payload chain as a source suffix ([".data.C_v._0"]) to append to an
    already-rendered scrutinee — what each backend's [SMatch] declaration emits.
*)
val payload_suffix :
  payload_layout -> cname:string -> arity:int -> int -> string

(** The payload chain as an IR projection off a scrutinee expression — the same
    chain {!payload_suffix} renders, for the [EMatch] substitution. *)
val payload_access :
  payload_layout ->
  cname:string ->
  arity:int ->
  int ->
  Sarek_ir_types.expr ->
  Sarek_ir_types.expr

(** [true] iff some case of a match expression binds at least one name, i.e. iff
    {!subst_ematch_payloads} would change anything. Backends guard their rewrite
    arm with it; the rewrite clears the binder lists throughout the subtree, so
    the guard is false on every node of the result and the rewrite neither loops
    nor runs twice. *)
val ematch_binds_payload : (Sarek_ir_types.pattern * 'a) list -> bool

(** [subst_ematch_payloads ~layout ~raise_ scrutinee cases] replaces every
    payload binder in [cases] — and in every match expression NESTED inside them
    — by the corresponding payload read, and clears the binder lists.

    IT REWRITES THE WHOLE SUBTREE IN ONE PASS, deliberately. Substitution
    injects a term built from the OUTER scrutinee, and that term has free
    variables; if an inner match were rewritten by a separate later call, its
    binders would substitute into those variables and capture them. Rewriting
    everything in one traversal binds each binder exactly once, in its own
    scope, and a replacement term installed at a leaf is never revisited.
    Shadowing is handled in the other direction too (an inner binder drops the
    outer mapping).

    The variable pattern [PConstr ("", [x])] (from [match e with x -> ..]) binds
    the whole scrutinee rather than a payload.

    [raise_] reports a binder shape with no lowering and MUST NOT RETURN (same
    convention as {!reject_feature}: each backend raises through its own error
    functor and [spoc/ir] has no backend dependencies). It fires for [EArrayLen]
    or [EArrayRead] of a binder — both need a vector-typed payload, so both are
    unreachable from the current DSL, and both were covered by #73's guard — and
    for a scrutinee containing an atomic intrinsic, which the lowering would
    otherwise execute once per re-emitted copy. *)
val subst_ematch_payloads :
  layout:payload_layout ->
  raise_:(string -> unit) ->
  Sarek_ir_types.expr ->
  (Sarek_ir_types.pattern * Sarek_ir_types.expr) list ->
  (Sarek_ir_types.pattern * Sarek_ir_types.expr) list

(** Emit a C/MSL tagged-union variant type: an [enum] of constructor tags, a
    [typedef struct] with a [tag] field and a [union] of payloads, and one
    inline constructor function per case.

    [type_of_elttype] maps IR element types to backend C type strings.
    [constructor_prefix] is the qualifier emitted before each constructor
    function (e.g. ["__device__ __host__ inline"] for CUDA, ["static inline"]
    for OpenCL/Metal).

    Used by the CUDA, OpenCL, and Metal backends. *)
val gen_variant_def :
  type_of_elttype:(Sarek_ir_types.elttype -> string) ->
  constructor_prefix:string ->
  Buffer.t ->
  string * (string * Sarek_ir_types.elttype list) list ->
  unit

(** {1 C-family shared helpers}

    Shared by the C-family backends (CUDA, OpenCL, Metal). GLSL/WGSL and PTX
    diverge too much to use these. *)

(** [is_vec_type t] is [true] iff [t] is a vector type, i.e. carries an implicit
    trailing [sarek_<name>_length] kernel argument. *)
val is_vec_type : Sarek_ir_types.elttype -> bool

(** Emit an l-value (assignment target / read path): [LVar], [LArrayElem],
    [LArrayElemExpr], [LRecordField]. Identical across the C-family backends;
    [gen_expr] renders array-index subexpressions. *)
val gen_lvalue :
  gen_expr:(Buffer.t -> Sarek_ir_types.expr -> unit) ->
  Buffer.t ->
  Sarek_ir_types.lvalue ->
  unit

(** Emit the array kernel-parameter head
    [<memspace> <elttype>* restrict <name>], the spelling shared by the OpenCL
    and Metal backends. [memspace] maps the array's memory space to its
    qualifier and [type_of_elttype] its element type. CUDA differs (no memspace,
    [__restrict__]) and does not use this. Intended as the [gen_array_param]
    argument to {!gen_param}. *)
val gen_global_array_param :
  memspace:(Sarek_ir_types.memspace -> string) ->
  type_of_elttype:(Sarek_ir_types.elttype -> string) ->
  Buffer.t ->
  Sarek_ir_types.var ->
  Sarek_ir_types.array_info ->
  unit

(** Emit a kernel parameter declaration. Shared skeleton: the scalar case emits
    [<param_type> <name>] plus a [, int sarek_<name>_length] suffix for vectors;
    the array-info case emits [gen_array_param] followed by that same length
    suffix.

    [param_type] spells the scalar/pointer type; [gen_array_param] emits the
    array-parameter head (see {!gen_global_array_param}); [invalid] rejects a
    [DLocal]/[DShared] declaration by raising the backend's located error (it
    never returns). *)
val gen_param :
  param_type:(Sarek_ir_types.elttype -> string) ->
  gen_array_param:
    (Buffer.t -> Sarek_ir_types.var -> Sarek_ir_types.array_info -> unit) ->
  invalid:(unit -> unit) ->
  Buffer.t ->
  Sarek_ir_types.decl ->
  unit

(** Raised by {!sort_type_decls_by_dependency} when the declarations it is given
    form a cycle between DISTINCT declarations, carrying the unplaced
    declarations as ["record <name>"] / ["variant <name>"] strings. The kind is
    part of the payload, not decoration: a record and a variant may share a
    mangled name and are two distinct nodes here, so a bare name list would
    render such a cycle as [t; t] and identify neither. Either kind can be on
    the cycle — record to record, variant to variant, or one through the other.
    Such a cycle has no valid emission order, so it is refused rather than
    emitted in input order. A declaration whose own field or payload type is
    ITSELF is NOT reported here: the self-edge is dropped so the diagnostic
    stays about cycles between declarations, and the backend's field-type
    emission is what reports it.

    It escapes {!gen_type_decls}, so it can surface from any of the five
    backends' [generate_with_types]; a [Printexc] printer is registered for it
    so the carried names reach the user instead of [Type_decl_cycle(_)]. *)
exception Type_decl_cycle of string list

(** One record or variant type declaration. Both kinds travel in one list inside
    {!sort_type_decls_by_dependency} so that a dependency edge crossing between
    them can be ordered (see {!gen_type_decls}). *)
type type_decl =
  | Record_decl of string * (string * Sarek_ir_types.elttype) list
  | Variant_decl of string * (string * Sarek_ir_types.elttype list) list

(** Which of the two kinds a backend family emitted first, back when it ran two
    per-kind loops. It decides the TIE-BREAK in {!gen_type_decls} and nothing
    else: anything with a real dependency edge is reordered either way. Passing
    the order a family's loops already produced is what keeps an edge-free
    kernel's emitted source byte-identical, so committed goldens do not churn.
*)
type tie_break =
  | Variants_first  (** CUDA, OpenCL, Metal, and HIP via the CUDA generator. *)
  | Records_first  (** GLSL and WGSL. *)

(** Mangled names of every record AND variant type reachable from a type,
    through arrays, vectors, variant payloads and nested record fields. Sorted
    and deduplicated.

    Terminates on a cyclic [elttype] value whatever closes the cycle: the
    visited set is keyed on physical node identity, not on a type name, so a
    cycle closed by [TVec]/[TArray] alone — with neither a record nor a variant
    on it — is caught too. Exported so the tests can exercise that termination
    directly; no production caller outside this module uses it. *)
val referenced_type_names : Sarek_ir_types.elttype -> string list

(** Order record and variant declarations so a declaration comes after every
    declaration its own field/payload types reference. Covers all four edge
    directions: record to record and variant to variant (backlog-203, where list
    order is not dependency order because the PPX prepends registry-reachable
    types to the payload's own), and record to variant and variant to record
    (backlog-211, where each family sorted one kind inside its own loop and the
    cross edge was ordered by neither).

    Tie-break is the incoming index, never the name and never the kind, so a
    list already in a valid emission order is returned UNCHANGED — that is the
    property committed goldens rest on, and an edge-free list is a special case
    of it. It is NOT stability in the general sense: a blocked declaration is
    overtaken by later independent ones, so [[a (needs c); b; c]] comes out
    [[b; c; a]], reversing the unrelated [a] and [b]. Input order is preserved
    only among the declarations ready at the same step.

    Node identity is the POSITION in the list. Two same-named declarations are
    therefore two nodes, and a record's edge to a same-named variant is kept — a
    name-keyed self-edge drop would lose exactly that edge. A reference to a
    type that is in the list is an edge to order.

    {b Completeness (backlog-212).} Before any placement, every name any
    declaration's fields/payloads reference is checked against the declared set
    (by mangled name, either kind — the same kind-blind resolution the edges
    above already use, so a set this check accepts as complete can still name a
    same-mangled, wrong-kind declaration; that residual is not what this check
    closes); the self-reference case always passes, because the declaration
    supplying the name is itself in that set. A name that resolves to nothing
    raises {!Undeclared_type_ref} rather than being silently dropped as "not an
    edge" — that used to surface only later, as whichever backend compiler
    happened to run over the generated source complaining about an unknown type
    name. Not every backend is in that "later" set: Metal goes through
    {!gen_c_type_decls} like the rest of the C family and is fully covered here;
    PTX declares no struct types at all (its [generate_with_types] never calls
    {!gen_type_decls}), so an undeclared reference on the PTX path is exactly as
    silent after this check as before it — this closes the gap for every
    struct-DECLARING backend, not for the one backend that declares none. This
    is reachable from ordinary [[@@sarek.type]] source: the PPX registers a
    variant's constructor payload from the constructor's OWN declared type,
    independent of whether the value passed to the constructor ever separately
    registered its own record type.

    Exported for the tests; {!gen_type_decls} is the only production caller.

    @raise Type_decl_cycle
      if the declarations form a cycle between distinct declarations; a
      self-referencing field or payload is dropped, not reported. No current
      caller can reach it: the PPX refuses a self- or forward-referencing record
      field at alignment-resolution time, and fusion only concatenates two
      PPX-produced (hence acyclic) type lists. It is a backstop for hand-built
      IR — tests today, a future front end — where nothing else would notice,
      not a guard anything presently depends on.
    @raise Undeclared_type_ref
      if any declaration's own field or payload types name a record or variant
      that is not itself in [decls] — checked, and reachable from ordinary
      [[@@sarek.type]] source, not only from hand-built IR; see the completeness
      paragraph above. *)
val sort_type_decls_by_dependency : type_decl list -> type_decl list

(** Raised by {!sort_type_decls_by_dependency} when some declaration's own field
    or payload types name a record or variant that is not itself among the
    declarations it was given — a reference with nothing behind it, as opposed
    to {!Type_decl_cycle}'s reference to something present but unorderable. Each
    string in the payload names the referencing declaration's kind and MANGLED
    name, the field or constructor site, and the missing (also mangled) type,
    e.g.
    [{"variant \"probe2\"'s constructor \"At2\" references undeclared type
     \"probe_pt\""}] — both names mangled consistently, never one dotted and the
    other underscored for what is otherwise the same declaration. A [Printexc]
    printer is registered so the message reaches the user instead of
    [Undeclared_type_ref(_)].

    Reachable from ordinary [[@@sarek.type]] source: the PPX registers a
    variant's constructor payload type from the constructor's own declaration,
    independent of whether the value passed to the constructor ever separately
    caused its record type to be registered (a record literal, a parameter of
    that type, or a local array element of that type all register it; extracting
    an existing value of that type from elsewhere — e.g. a differently-typed
    source — does not). See
    sarek/tests/e2e/test_undeclared_variant_payload_record.ml. *)
exception Undeclared_type_ref of string list

(** Emit a backend's record and variant declarations from ONE
    {!sort_type_decls_by_dependency} pass over both lists together, dispatching
    each entry to [emit_record] or [emit_variant]. Every backend that emits
    struct declarations (CUDA, OpenCL, Metal, HIP via CUDA, GLSL, WGSL) goes
    through this; PTX declares no types and does not.

    The two emitters cannot be swapped by accident: their payload types differ
    ([(string * elttype) list] versus [(string * elttype list) list]).

    It orders exactly the declarations it is handed, by the type references
    inside their own fields and payloads. It orders NOTHING ELSE the backend
    emits — headers, bindings and helper functions are the caller's sequencing —
    and it does not make an unorderable or incomplete input emittable: see
    {!Type_decl_cycle} for a cycle, and {!Undeclared_type_ref} for a reference
    to a type that is not in [records]/[variants] at all.

    @raise Type_decl_cycle see {!sort_type_decls_by_dependency}.
    @raise Undeclared_type_ref see {!sort_type_decls_by_dependency}. *)
val gen_type_decls :
  emit_record:
    (Buffer.t -> string * (string * Sarek_ir_types.elttype) list -> unit) ->
  emit_variant:
    (Buffer.t -> string * (string * Sarek_ir_types.elttype list) list -> unit) ->
  tie_break:tie_break ->
  Buffer.t ->
  records:(string * (string * Sarek_ir_types.elttype) list) list ->
  variants:(string * (string * Sarek_ir_types.elttype list) list) list ->
  unit

(** {!gen_type_decls} specialised to the C family (CUDA, OpenCL, Metal, and HIP
    through the CUDA generator): records are emitted as
    [typedef struct { ... } name;] by the shared emitter and variants by
    {!gen_variant_def}, so a backend supplies only [type_of_elttype] and
    [constructor_prefix] — the two things that differ between them.

    This is the only route to the shared C-family record emitter. That emitter
    is deliberately not exported on its own: a record loop with no sort in it is
    the shape backlog-203 fixed.

    @raise Type_decl_cycle see {!sort_type_decls_by_dependency}.
    @raise Undeclared_type_ref see {!sort_type_decls_by_dependency}. *)
val gen_c_type_decls :
  type_of_elttype:(Sarek_ir_types.elttype -> string) ->
  constructor_prefix:string ->
  Buffer.t ->
  records:(string * (string * Sarek_ir_types.elttype) list) list ->
  variants:(string * (string * Sarek_ir_types.elttype list) list) list ->
  unit

(** Emit a GLSL variant type. GLSL has no [enum], [typedef], or [union], so tags
    are [const int] declarations, the type is a bare [struct], payloads are flat
    fields, and constructor functions have no qualifier prefix.

    [type_of_elttype] maps IR element types to GLSL type strings.

    Used by the Vulkan backend. *)
val gen_variant_def_glsl :
  type_of_elttype:(Sarek_ir_types.elttype -> string) ->
  Buffer.t ->
  string * (string * Sarek_ir_types.elttype list) list ->
  unit
