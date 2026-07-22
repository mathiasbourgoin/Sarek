(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** PTX kernel emitter: parameter/local declaration, register-block output, PTX
    file header, and top-level generate entry points. *)

open Sarek_ir_types
open Sarek_ir_ptx_types

(** [emit_params buf alloc env params] emits [ld.param] instructions for each
    kernel parameter into [buf], binds each parameter register into [env], and
    records array element types in [alloc.arr_elt_types]. Returns the formatted
    [.param] declaration block string for embedding in the [.entry] header. *)
val emit_params : Buffer.t -> reg_alloc -> env -> decl list -> string

(** [emit_locals buf shared_buf module_buf alloc env locals] emits register
    allocations and optional initialisation moves for each [DLocal] declaration
    ([DLocal] of array type is rejected fail-closed: it carries no size and
    cannot allocate storage). Statically-sized [DShared] declarations emit a
    [.shared] directive to [shared_buf]; a dynamic [DShared] (size [None]) emits
    a module-scope [.extern .shared] incomplete-array directive to [module_buf]
    (one per kernel; the region's byte size is supplied at launch via
    [~shared_mem]). Both bind the base address with a [mov.u32] in [buf].
    [DParam] entries are skipped. *)
val emit_locals :
  Buffer.t -> Buffer.t -> Buffer.t -> reg_alloc -> env -> decl list -> unit

(** [emit_reg_decls buf alloc] emits [.reg] declarations based on the allocator
    high-water marks. Must be called {e after} all [emit_*] calls. *)
val emit_reg_decls : Buffer.t -> reg_alloc -> unit

(** [make_ptx_header ?sm_target ?ptx_version ()] returns the PTX file header
    string ([.version], [.target], [.address_size]). Defaults:
    [sm_target = "sm_86"], [ptx_version = "8.0"]. *)
val make_ptx_header : ?sm_target:string -> ?ptx_version:string -> unit -> string

(** [generate ?sm_target k] translates kernel [k] to a complete PTX string. Uses
    three-phase generation: body → register-count → header concatenation.
    @param sm_target Override the default [sm_86] target for older hardware. *)
val generate : ?sm_target:string -> kernel -> string

(** [generate_with_types ~types k] is [generate k]. Record and variant type
    definitions are not representable as PTX struct types; the [~types] argument
    is accepted for interface compatibility with other backends and is ignored.
*)
val generate_with_types : types:_ -> kernel -> string
