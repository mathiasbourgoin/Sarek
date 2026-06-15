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

    Raises {!Ptx_codegen_error} for IR constructs not yet covered by the PTX
    backend (variants, records, device function calls, etc.). *)
val emit_expr : Buffer.t -> reg_alloc -> env -> expr -> string

(** [emit_binop buf alloc env op e1 e2] emits a binary operation. Type is
    inferred from the register-name prefix of the first operand. *)
val emit_binop : Buffer.t -> reg_alloc -> env -> binop -> expr -> expr -> string

(** [emit_cast buf alloc r_src dst_ty] emits a PTX [cvt.*] instruction if needed
    and returns the destination register. Returns [r_src] unchanged when no
    conversion is needed. *)
val emit_cast : Buffer.t -> reg_alloc -> string -> elttype -> string

(** [emit_intrinsic buf alloc env path name args] emits the PTX sequence for the
    named Sarek intrinsic. [path] is ignored (reserved for future namespacing).
    Raises {!Ptx_codegen_error} for unknown intrinsic names. *)
val emit_intrinsic :
  Buffer.t -> reg_alloc -> env -> string list -> string -> expr list -> string
