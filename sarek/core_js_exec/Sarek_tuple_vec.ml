(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * jsoo build stub for [Sarek_tuple_vec] (L13 tuple-typed vectors).
 *
 * The interpreter ([Sarek_ir_interp], PR #249) decodes tuple-vector composite
 * element bytes through the shared shape registry in [sarek/tuple_vec/]. That
 * real module cannot be compiled here: its [component]/[pair]/[triple] builders
 * marshal host bytes through [Ctypes] raw pointers (via
 * [Spoc_core.Vector.Custom_helpers]) and depend on the genuine ctypes-backed
 * [spoc_core] — neither is jsoo-compatible, and pulling them in would defeat
 * the FFI-free shim this library exists to be.
 *
 * The interpreter only consumes three surfaces from that module: the
 * [field_layout] and [shape] byte-layout records and [lookup_shape]. This stub
 * reproduces exactly those, with an empty shape registry.
 *
 * This is behaviour-preserving, not a degradation: the shape registry is
 * populated solely as a side effect of [pair]/[triple] running on the host,
 * and those builders never run in the jsoo path (no ctypes, no host device).
 * The real module would therefore also return [None] from [lookup_shape] for
 * every name in this context. On [None] the interpreter yields an empty
 * [VRecord] on read and a no-op on writeback — so tuple-vector decode is simply
 * inert in the browser build, exactly as it is with the real module here.
 ******************************************************************************)

(** Byte layout of one positional field ([_0], [_1], ...) of a tuple element.
    Mirrors [Sarek_tuple_vec.field_layout] in [sarek/tuple_vec/]. *)
type field_layout = {
  fl_name : string;
  fl_elttype : Sarek_ir_types.elttype;
  fl_offset : int;
}

(** Byte layout of a tuple-shape element. Mirrors [Sarek_tuple_vec.shape]. *)
type shape = {sh_name : string; sh_size : int; sh_fields : field_layout list}

(** [lookup_shape name] always returns [None] in the jsoo build: no host-side
    [pair]/[triple] call ever registers a shape here (those need ctypes), so
    there is nothing to look up. *)
let lookup_shape (_name : string) : shape option = None
