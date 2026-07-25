(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Shared code-generation helpers used across GPU backends (CUDA, OpenCL,
    Metal, Vulkan/GLSL). Extracted to avoid duplication of identical
    variant/struct generation logic. *)

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
