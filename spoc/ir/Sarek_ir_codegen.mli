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

(** Emit C-family record type declarations: one [typedef struct { ... } name;]
    per record, one field per line. Only [type_of_elttype] differs per backend.
*)
val gen_record_typedefs :
  type_of_elttype:(Sarek_ir_types.elttype -> string) ->
  Buffer.t ->
  (string * (string * Sarek_ir_types.elttype) list) list ->
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
