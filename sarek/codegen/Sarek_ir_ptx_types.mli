(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** PTX codegen shared types: register allocator, environment, error helpers,
    and buffer emit primitives.

    All downstream PTX codegen modules open this module. *)

open Sarek_ir_types

(** {1 Error handling} *)

exception Ptx_codegen_error of string

(** [fail msg] raises {!Ptx_codegen_error} with [msg]. *)
val fail : string -> 'a

(** [unsupported what] raises {!Ptx_codegen_error} for an unsupported IR
    construct named [what]. *)
val unsupported : string -> 'a

(** {1 Register allocator} *)

(** Counter-based register allocator. Each PTX type has an independent counter
    so that register names stay readable (e.g. [%r0], [%f0], [%rd0]). *)
type reg_alloc = {
  mutable u32 : int;
  mutable u64 : int;
  mutable f32 : int;
  mutable f64 : int;
  mutable pred : int;
  mutable label : int;
  arr_elt_types : (string, elttype) Hashtbl.t;
  arr_memspaces : (string, unit) Hashtbl.t;
}

(** [make_alloc ()] returns a fresh zeroed allocator. *)
val make_alloc : unit -> reg_alloc

(** Allocate a fresh [.u32] register and return its PTX name ([%rN]). *)
val new_u32 : reg_alloc -> string

(** Allocate a fresh [.u64] register and return its PTX name ([%rdN]). *)
val new_u64 : reg_alloc -> string

(** Allocate a fresh [.f32] register and return its PTX name ([%fN]). *)
val new_f32 : reg_alloc -> string

(** Allocate a fresh [.f64] register and return its PTX name ([%fdN]). *)
val new_f64 : reg_alloc -> string

(** Allocate a fresh [.pred] register and return its PTX name ([%pN]). *)
val new_pred : reg_alloc -> string

(** Allocate a fresh branch label and return its name ([LN]). *)
val new_label : reg_alloc -> string

(** {1 Type mapping} *)

(** [ptx_reg_type_of t] returns the PTX register-type string for [t] (e.g.
    [".u32"], [".f64"]). Raises {!Ptx_codegen_error} for [TRecord] and
    [TVariant]. *)
val ptx_reg_type_of : elttype -> string

(** [new_reg_for_type alloc t] allocates a register appropriate for type [t] and
    returns its PTX name. *)
val new_reg_for_type : reg_alloc -> elttype -> string

(** {1 Environment: variable name -> PTX register name} *)

(** Maps Sarek IR variable names to their PTX register names. *)
type env = (string, string) Hashtbl.t

(** [make_env ()] returns an empty environment. *)
val make_env : unit -> env

(** [env_bind env name reg] binds [name] to register [reg], overwriting any
    previous binding. *)
val env_bind : env -> string -> string -> unit

(** [env_lookup env name] returns the PTX register for [name]. Raises
    {!Ptx_codegen_error} if [name] is unbound. *)
val env_lookup : env -> string -> string

(** {1 Emit helpers} *)

(** [emit buf fmt ...] appends a 4-space-indented line to [buf]. *)
val emit : Buffer.t -> ('a, Buffer.t, unit) format -> 'a

(** [emit_label buf lbl] appends [lbl:] followed by a newline to [buf]. *)
val emit_label : Buffer.t -> string -> unit
