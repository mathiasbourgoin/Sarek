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
    (#75). Expression position admits exactly one binding mechanism —
    substituting the binder by the payload read — and it lives here, once,
    rather than as a per-backend copy (#94). *)

(** [true] iff some case of a match expression binds at least one name, i.e. iff
    {!subst_ematch_payloads} would change anything. Backends guard their rewrite
    arm with it; the rewrite clears the binder lists, so the guard is false on
    the rewritten node and cannot loop. *)
val ematch_binds_payload : (Sarek_ir_types.pattern * 'a) list -> bool

(** [subst_ematch_payloads ~union_field scrutinee cases] replaces every payload
    binder of every case by the corresponding payload read of [scrutinee] and
    clears the binder lists (backends only need the constructor name, for the
    tag test). The access path is the one that backend's [SMatch] arm already
    declares, so the two paths agree by construction.

    [union_field] is the only backend-specific input: [Some "data"] for the
    C-family tagged union ([<scrut>.data.C_v], [<scrut>.data.C_v._<i>] for a
    multi-payload constructor), [None] for GLSL/WGSL, which flatten payloads
    into the variant struct ([<scrut>.C_v]).

    Substitution is capture-avoiding — a nested [EMatch] arm rebinding the same
    name keeps reading its own payload — and the variable pattern
    [PConstr ("", [x])] (from [match e with x -> ...]) binds the whole
    scrutinee. [scrutinee] is duplicated per occurrence, which is
    semantics-preserving because IR expressions are pure and the tag test
    already re-emits it once per case. [EArrayLen] of a payload binder is left
    untouched: a vector payload has no companion length argument, so that shape
    has no correct lowering (pre-existing, unreachable from the DSL). *)
val subst_ematch_payloads :
  union_field:string option ->
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
