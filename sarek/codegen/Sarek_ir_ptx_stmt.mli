(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@Gmail.com> *)
(******************************************************************************)

(** PTX statement emitter.

    Translates Sarek IR statements to PTX instruction sequences. All emitters
    mutate [buf] and [alloc] as side effects. *)

open Sarek_ir_types
open Sarek_ir_ptx_types

(** [emit_stmt buf alloc env stmt] emits PTX instructions for [stmt] into [buf].

    Raises {!Ptx_codegen_error} for IR constructs not covered by the PTX backend
    (e.g. [SMatch]). *)
val emit_stmt : Buffer.t -> reg_alloc -> env -> stmt -> unit

(** [emit_assign buf alloc env lv e] emits a PTX assignment for lvalue [lv] set
    to expression [e]. Handles scalar, array-element, and indirect array-element
    lvalues. *)
val emit_assign : Buffer.t -> reg_alloc -> env -> lvalue -> expr -> unit
