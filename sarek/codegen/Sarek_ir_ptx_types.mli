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
  shared_decls : Buffer.t;
      (** [.shared] declarations discovered while emitting the body; spliced
          into the kernel prologue by [generate]. *)
  funcs : (string, helper_func) Hashtbl.t;
      (** Kernel helper functions ([kern_funcs]), inlined at EApp sites. *)
  mutable inline_stack : string list;
      (** Helper names currently being inlined — recursion guard. *)
  mutable inline_ret : (string option * string) list;
      (** Per-inline (result register, end label) for SReturn lowering. *)
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

(** {1 Environment: variable name -> PTX binding} *)

(** SROA-decomposed aggregate value: a record is one binding per field (in
    declaration order, nested records as nested [ARecord]); a variant is a u32
    tag register plus per-(constructor index) payload bindings. *)
type agg_value =
  | ARecord of (string * binding) list
  | AVariant of {tag_reg : string; payloads : (int * binding list) list}

(** A variable's PTX value: a single scalar register name, or an aggregate
    register set. *)
and binding = Scalar of string | Agg of agg_value

(** Maps Sarek IR variable names to their PTX bindings. *)
type env = (string, binding) Hashtbl.t

(** [make_env ()] returns an empty environment. *)
val make_env : unit -> env

(** [env_bind env name reg] binds [name] to the scalar register [reg],
    overwriting any previous binding. *)
val env_bind : env -> string -> string -> unit

(** [env_bind_binding env name b] binds [name] to [b] (scalar or aggregate),
    overwriting any previous binding. *)
val env_bind_binding : env -> string -> binding -> unit

(** [env_lookup env name] returns the PTX register for the scalar [name]. Raises
    {!Ptx_codegen_error} if [name] is unbound, or if it is bound to an aggregate
    (internal error: scalar lookup on aggregate binding). *)
val env_lookup : env -> string -> string

(** [env_lookup_binding env name] returns the binding for [name]. Raises
    {!Ptx_codegen_error} if [name] is unbound. *)
val env_lookup_binding : env -> string -> binding

(** [length_param_name arr] is the name of the implicit length parameter paired
    with vector/array parameter [arr] ("sarek_<arr>_length") — the single
    definition shared by the param emitter and the [EArrayLen] lookup, matching
    the convention of Execute.expand_to_run_source_args and the C-family code
    generators. *)
val length_param_name : string -> string

(** {1 Emit helpers} *)

(** [emit buf fmt ...] appends a 4-space-indented line to [buf]. *)
val emit : Buffer.t -> ('a, Buffer.t, unit) format -> 'a

(** [emit_label buf lbl] appends [lbl:] followed by a newline to [buf]. *)
val emit_label : Buffer.t -> string -> unit

(** {1 Register-class helpers} *)

(** Register class recovered from a PTX register name prefix ([%fdN] = f64,
    [%fN] = f32, [%rdN] = u64, [%rN] = u32). *)
type reg_class = RU32 | RU64 | RF32 | RF64

(** [reg_class r] is the class of register [r], from its name prefix. *)
val reg_class : string -> reg_class

(** [mov_op_of_class c] is the typed PTX mov opcode for class [c]. *)
val mov_op_of_class : reg_class -> string

(** [new_reg_like alloc r] allocates a fresh register of [r]'s class. *)
val new_reg_like : reg_alloc -> string -> string

(** [mov_scalar buf ~dst ~src] emits a typed mov of [src] into [dst], typed by
    [dst]'s register class. *)
val mov_scalar : Buffer.t -> dst:string -> src:string -> unit

(** [copy_reg buf alloc r] allocates a fresh register of [r]'s class and emits a
    mov from [r] into it — for bindings that must not alias the source register
    (mutable lets, inlined helper parameters). *)
val copy_reg : Buffer.t -> reg_alloc -> string -> string

(** [copy_binding buf alloc b] copies every scalar leaf of [b] into fresh
    registers (leaf-wise {!copy_reg}), preserving the aggregate shape. *)
val copy_binding : Buffer.t -> reg_alloc -> binding -> binding

(** [mov_binding buf ~src ~dst] emits leaf-wise typed movs of [src]'s registers
    into [dst]'s registers. Shapes must be compatible (records matched by field
    name, variant payload slots by constructor tag); raises {!Ptx_codegen_error}
    on shape mismatch. *)
val mov_binding : Buffer.t -> src:binding -> dst:binding -> unit

(** {1 Shared-memory declaration helpers} *)

(** Byte alignment of a [.shared] array of the given element type. *)
val ptx_align_of_elttype : elttype -> int

(** PTX untyped-bits type ([b32]/[b64]) for a [.shared] array declaration. *)
val ptx_btype_of_elttype : elttype -> string

(** {1 Statement-emitter hook}

    Installed by Sarek_ir_ptx_stmt at load time so the expression emitter can
    emit helper-function bodies (statements) when inlining EApp without a
    circular module dependency. *)
val stmt_emitter :
  (Buffer.t -> reg_alloc -> env -> Sarek_ir_types.stmt -> unit) ref
