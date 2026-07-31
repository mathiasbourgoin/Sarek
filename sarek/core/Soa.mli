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

(** Raised to refuse an operation the SoA layer cannot do without either
    corrupting data or misrepresenting what happened.

    Below is the list of RAISERS, which is closed as of this comment
    ([grep -rn "Unsupported" sarek/core/Soa.ml sarek/core/Soa_vector.ml] to
    re-derive it). The reasons under each raiser are examples, not an
    enumeration — read the message, not this list, to learn why a particular
    call refused.

    - {!plan} and {!plan_of_elttype}: the element type is not a v1-supported SoA
      target. Reached by a non-record; by a nested-record, variant, array or
      vector field; by an f16 or uint8 field; and by a
      [Sarek_ir_layout.record_layout] failure such as a misaligned field.
    - {!Soa_vector.create}, and therefore {!Soa_vector.create_transparent}: the
      element type's [custom_type.ir_fields] is [None], so no flat-record layout
      is derivable at all. Raised before {!plan} is reached.
    - {!Soa_vector.scatter}: a vector's host data is out of date while auto-sync
      is disabled, so the transpose cannot be done correctly and silently.

    That last cause is unlike the others and callers must not conflate them: it
    is a transient property of one VECTOR's state, not a permanent property of a
    TYPE, and the same call succeeds once the host copy is refreshed. A handler
    that reads {!Unsupported} as "this type can never be SoA" and installs a
    permanent AoS fallback is wrong for it. *)
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
