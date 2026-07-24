(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * User-facing Structure-of-Arrays (SoA) custom-vector storage — Tier 1c host
 * half.
 *
 * Opt-in SoA storage for a custom (flat-record) vector. Bundles:
 *   - an ordinary AoS custom {!Vector.t} as the host source of truth (so the
 *     full existing Vector host API — get/set/PPX accessors — works unchanged);
 *   - a {!Soa.plan} (the leaf enumeration reused from Tier 1a);
 *   - N per-leaf host buffers, each a width-matched scalar {!Vector.t} used as a
 *     bit-preserving byte transport (4-byte leaf -> int32 vector, 8-byte leaf ->
 *     int64 vector; the logical leaf type is irrelevant because
 *     {!Soa.scatter}/{!Soa.gather} copy raw words). Each leaf is therefore an
 *     ordinary scalar vector reusing the existing scalar transfer/allocation
 *     path — no new backend or codegen support needed on the host side.
 *
 * The device-side consumption (lowering [pts.(i).x] to N base pointers +
 * coalesced scalar loads) is the #260 PTX emitter; the launch that binds these
 * N leaf buffers as that ABI is {!Sarek.Execute.run_soa}, which is CUDA/PTX
 * only. This module is pure host storage + transpose and is backend-agnostic.
 *
 * This is deliberately layered above {!Vector} + {!Soa} rather than being a new
 * constructor on the core [host_storage] GADT: the PPX [custom_type] carries no
 * record layout (so the field list must be supplied here), and a fully
 * transparent [Vector.create ~layout:SoA] + generic [Execute.run] auto-dispatch
 * would additionally require threading SoA param names through every backend and
 * generalising the 1-buffer-per-device table — deferred (see
 * docs/optimization/tier1b-emitter-soa-handoff.md and the impl brief).
 ******************************************************************************)

(** A per-leaf host buffer, type-erased: a width-matched scalar vector used as a
    bit-preserving byte transport for one SoA leaf. *)
type packed_leaf = Leaf : ('e, 'f) Vector.t -> packed_leaf

(** A SoA custom vector of element type ['a]. *)
type 'a t

(** [create custom ~fields length] builds a SoA custom vector of [length]
    elements. [custom] is the PPX-generated custom type (the AoS source of
    truth); [fields] is the record's field layout [(name, scalar type)].

    {b Precondition — [fields] MUST match the record's declaration order and
       types exactly.} The [custom_type] carries no layout, so this cannot be
    checked: [fields] is the sole description of how the packed AoS element is
    split into leaves. If it disagrees with the actual record layout (wrong
    order, wrong widths, missing/extra field), {!scatter}/{!gather} transpose
    against the wrong byte offsets and the vector carries
    {e silently transposed / corrupted} data with no error. Pass exactly the
    fields the [[@@sarek.type]] record declares, in order — the same list the
    kernel IR's [TRecord] uses.

    Raises {!Soa.Unsupported} if [fields] is not a flat record. *)
val create :
  'a Vector.custom_type ->
  fields:(string * Sarek_ir_types.elttype) list ->
  int ->
  'a t

(** The AoS host vector (source of truth). Use it for the non-PTX host fallback
    (drive it through the ordinary {!Sarek.Execute.run_vectors} AoS path). *)
val aos_vector : 'a t -> ('a, unit) Vector.t

(** The SoA storage plan (leaf enumeration + AoS stride). *)
val plan : 'a t -> Soa.plan

(** The N per-leaf host buffers, in {!Soa.plan} leaf (record declaration) order.
*)
val leaves : 'a t -> packed_leaf array

(** Number of scalar leaves (= number of base pointers a kernel launch binds).
*)
val num_leaves : 'a t -> int

(** Number of elements. *)
val length : 'a t -> int

(** [set t i v] / [get t i] delegate to the AoS host vector (the source of
    truth). *)
val set : 'a t -> int -> 'a -> unit

val get : 'a t -> int -> 'a

(** [scatter t] transposes the AoS host buffer into the N per-leaf host buffers
    (call before transferring the leaves to a device). Ensures the AoS host copy
    is current first. *)
val scatter : 'a t -> unit

(** [gather t] transposes the N per-leaf host buffers back into the AoS host
    buffer (call after reading leaves back from a device that wrote them). *)
val gather : 'a t -> unit
