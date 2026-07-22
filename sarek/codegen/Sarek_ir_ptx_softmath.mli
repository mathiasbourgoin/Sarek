(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Software f64 transcendentals for the PTX backend.

    Pure-IR [helper_func] bodies (fdlibm/Taylor polynomials over native f64 ops
    \+ [f64_bits]/[bits_f64] bitcasts) for the Float64 transcendentals PTX has
    no instruction for: sin, cos, tan, exp, log, log10, pow, sinh, cosh, tanh.
    The PTX emitter registers them on demand and inlines them through the
    existing EApp machinery. Precise tier: max relative error ≤ 1e-12 on the
    documented domains (see the .ml header for domains and coefficient
    provenance). *)

(** [helper_name intrinsic] is the reserved helper name (e.g.
    ["__sarek_f64_sin"]) implementing the Float64 [intrinsic], or [None] when
    the intrinsic has no software f64 implementation. *)
val helper_name : string -> string option

(** [register funcs] adds every softmath helper to [funcs] (idempotent). The
    helpers may call each other (tan → sin/cos; sinh/cosh/tanh/pow → exp;
    log10/pow → log), so they are always registered as a family. *)
val register : (string, Sarek_ir_types.helper_func) Hashtbl.t -> unit

(** All softmath helpers (for direct IR-level evaluation in tests). *)
val all_helpers : unit -> Sarek_ir_types.helper_func list
