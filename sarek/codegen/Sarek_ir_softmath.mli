(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Software f64 transcendentals — backend-shared (PTX and GLSL/Vulkan).

    Pure-IR [helper_func] bodies (fdlibm/Taylor polynomials over native f64 ops
    \+ [f64_bits]/[bits_f64] bitcasts) for the Float64 transcendentals PTX and
    GLSL core have no instruction/overload for: sin, cos, tan, asin, acos, atan,
    atan2, exp, expm1, log, log10, log1p, pow, sinh, cosh, tanh. The PTX emitter
    registers them on demand and inlines them through the EApp machinery; the
    GLSL emitter emits the needed family as top-level functions gated
    per-kernel. Precise tier: max relative error ≤ 1e-12 on the documented
    domains (see the .ml header for domains and coefficient provenance). *)

(** [helper_name intrinsic] is the reserved helper name (e.g.
    ["__sarek_f64_sin"]) implementing the Float64 [intrinsic], or [None] when
    the intrinsic has no software f64 implementation. *)
val helper_name : string -> string option

(** [register funcs] adds every softmath helper to [funcs] (idempotent). The
    helpers may call each other (tan → sin/cos; sinh/cosh/tanh/pow → exp;
    log10/pow → log; atan2 → atan), so they are always registered as a family.
*)
val register : (string, Sarek_ir_types.helper_func) Hashtbl.t -> unit

(** All softmath helpers (for direct IR-level evaluation in tests). *)
val all_helpers : unit -> Sarek_ir_types.helper_func list

(** [helper_by_name n] is the helper whose [hf_name] is [n], if any. *)
val helper_by_name : string -> Sarek_ir_types.helper_func option

(** [callees hf] are the reserved [__sarek_f64_*] helper names [hf] calls
    directly (deduplicated) — used to close the family over cross-calls. *)
val callees : Sarek_ir_types.helper_func -> string list

(** [uses_int64 hf] iff [hf]'s body needs 64-bit integer support (the
    [f64_bits]/[bits_f64] bitcasts or any int64 value/cast/literal). *)
val uses_int64 : Sarek_ir_types.helper_func -> bool

(** [uses_copysign hf] iff [hf]'s body calls the [copysign] intrinsic. *)
val uses_copysign : Sarek_ir_types.helper_func -> bool
