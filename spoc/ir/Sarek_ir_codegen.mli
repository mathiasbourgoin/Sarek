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
