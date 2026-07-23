(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Structure-of-Arrays (SoA) storage for custom-type vectors — host side.
 *
 * Tier 1a headline item (host/transfer/accessor half). A custom-type vector
 * (e.g. a [point3d = {x;y;z}] vector) is stored packed AoS today, which forces
 * an N-way-strided (1/N-efficiency) uncoalesced device access whenever a kernel
 * touches only one field. SoA stores each scalar leaf in its own contiguous
 * buffer, restoring full coalescing for single-field access.
 *
 * This module provides the storage *plan* (reusing the leaf enumeration
 * [Sarek_ir_layout] already computes) and the host-side AoS<->SoA transpose.
 * It is deliberately layered below {!Vector}: it operates on raw ctypes
 * pointers, so the caller can transpose between a custom (AoS) vector's host
 * buffer and an array of scalar (SoA leaf) vectors, then transfer each leaf
 * with the ordinary scalar path — no new backend or codegen support needed.
 *
 * The device-side consumption of a SoA vector (lowering [pts.(i).x] to N base
 * pointers + coalesced scalar loads) requires PTX-emitter changes and is the
 * Tier 1b handoff; see docs/optimization/opt-spoc-runtime.md.
 ******************************************************************************)

(** One scalar leaf of a flattened record: its dotted path, scalar type, byte
    offset within the packed AoS element, and scalar byte size. *)
type leaf = {
  path : string;
  ty : Sarek_ir_types.elttype;
  aos_offset : int;
  size : int;
}

(** A SoA storage plan for a flat record type. [aos_stride] is the packed AoS
    element size in bytes (the stride between consecutive elements in AoS). *)
type plan = {name : string; leaves : leaf list; aos_stride : int}

(** Raised when a type is not a v1-supported SoA target (non-record, or a record
    with a nested-record / variant / array field — flat records only for v1). *)
exception Unsupported of string

(** Build a SoA plan from a record's named fields. Reuses
    {!Sarek_ir_layout.record_layout} for leaf offsets/sizes/stride. Every field
    must be a scalar; nested records, variants and arrays are rejected with
    {!Unsupported}. *)
val plan : name:string -> (string * Sarek_ir_types.elttype) list -> plan

(** [plan_of_elttype t] requires [t] to be a [TRecord]. *)
val plan_of_elttype : Sarek_ir_types.elttype -> plan

(** Number of scalar leaves (= number of SoA sub-buffers / kernel base pointers
    a SoA vector of this type needs). *)
val num_leaves : plan -> int

(** Total bytes an AoS buffer of [length] elements occupies
    ([length * aos_stride]). *)
val aos_bytes : plan -> length:int -> int

(** [scatter plan ~aos ~length ~leaves] copies each element's field values from
    the packed AoS buffer [aos] into the per-leaf contiguous buffers [leaves]
    ([leaves.(k)] receives the k-th leaf, in {!plan.leaves} order).
    Bit-preserving copy; works for any scalar leaf types. *)
val scatter :
  plan ->
  aos:unit Ctypes.ptr ->
  length:int ->
  leaves:unit Ctypes.ptr array ->
  unit

(** Inverse of {!scatter}: gather per-leaf SoA buffers back into a packed AoS
    buffer. *)
val gather :
  plan ->
  leaves:unit Ctypes.ptr array ->
  length:int ->
  aos:unit Ctypes.ptr ->
  unit
