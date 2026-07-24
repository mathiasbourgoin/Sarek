(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** PTX expression emitter.

    Translates Sarek IR expressions to PTX instruction sequences. All emitters
    return the PTX register holding the result and mutate [buf] and [alloc] as
    side effects. *)

open Sarek_ir_types
open Sarek_ir_ptx_types

(** [emit_expr buf alloc env expr] emits PTX instructions for [expr] into [buf]
    and returns the register name holding the result.

    Raises {!Sarek_ir_ptx_types.Ptx_codegen_error} for IR constructs not yet
    covered by the PTX backend (variants, records, recursive device functions,
    etc.). Non-recursive helper-function calls (EApp) are inlined at the call
    site. *)
val emit_expr : Buffer.t -> reg_alloc -> env -> expr -> string

(** [emit_value buf alloc env expr] is the aggregate-aware emitter: scalar
    expressions delegate to {!emit_expr} (wrapped in [Scalar]); record
    construction ([ERecord]) evaluates each field into an SROA register set, and
    field projection ([ERecordField]) on a local record is pure register
    selection ([Agg] values, no memory traffic — FR-020). *)
val emit_value : Buffer.t -> reg_alloc -> env -> expr -> binding

(** [emit_match_arms buf alloc env scrut arms ~emit_arm] emits a full match on
    the scrutinee binding [scrut]: variant scrutinees get a tag-compare branch
    chain (per-arm [setp.eq] + bra, last/catch-all arm unconditional — never
    selp, FR-022); tuple/record/scalar scrutinees support exactly one
    destructuring arm. [emit_arm] emits one arm body (expression or statement)
    with the arm's pattern variables bound arm-scoped. Raises
    {!Sarek_ir_ptx_types.Ptx_codegen_error} on a non-exhaustive variant match
    (no catch-all arm and at least one constructor uncovered). *)
val emit_match_arms :
  Buffer.t ->
  reg_alloc ->
  env ->
  binding ->
  (pattern * 'a) list ->
  emit_arm:('a -> unit) ->
  unit

(** [emit_binop buf alloc env op e1 e2] emits a binary operation. Type is
    inferred from the register-name prefix of the first operand. *)
val emit_binop : Buffer.t -> reg_alloc -> env -> binop -> expr -> expr -> string

(** [emit_cast buf alloc r_src dst_ty] emits a PTX [cvt.*] instruction if needed
    and returns the destination register. Returns [r_src] unchanged when no
    conversion is needed. *)
val emit_cast : Buffer.t -> reg_alloc -> string -> elttype -> string

(** [emit_intrinsic buf alloc env path name args] emits the PTX sequence for the
    named Sarek intrinsic. [path] disambiguates module-qualified names ("of_int"
    resolves to f32 or f64 by its Float32/Float64 path). Raises
    {!Sarek_ir_ptx_types.Ptx_codegen_error} for unknown intrinsic names. *)
val emit_intrinsic :
  Buffer.t -> reg_alloc -> env -> string list -> string -> expr list -> string
