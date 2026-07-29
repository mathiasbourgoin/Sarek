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
 * N leaf buffers as that ABI is {!Sarek.Soa_launch.run_soa}, which is CUDA/PTX
 * only. This module is pure host storage + transpose and is backend-agnostic.
 *
 * This is deliberately layered above {!Vector} + {!Soa} rather than being a new
 * constructor on the core [host_storage] GADT. A fully transparent
 * [Vector.create ~layout:SoA] + generic [Execute.run] auto-dispatch would
 * require threading SoA param names through every backend and generalising the
 * 1-buffer-per-device table — deferred (see
 * docs/optimization/tier1b-emitter-soa-handoff.md and the impl brief).
 *
 * The layout half of that deferral is now closed: the leaf enumeration is
 * derived from [custom_type.ir_fields] rather than supplied by the caller, so
 * "the custom_type carries no record layout" is no longer a reason this cannot
 * be transparent. What remains is the backend threading and the buffer table.
 ******************************************************************************)

(** A per-leaf host buffer, type-erased: a width-matched scalar vector used as a
    bit-preserving byte transport for one SoA leaf. *)
type packed_leaf = Leaf : ('e, 'f) Vector.t -> packed_leaf

(** A SoA custom vector of element type ['a]. *)
type 'a t

(** [create custom length] builds a SoA custom vector of [length] elements.
    [custom] is the PPX-generated custom type (the AoS source of truth), and the
    leaf layout is {b derived} from it via [custom.ir_fields].

    There is deliberately no [~fields] parameter. There used to be one, on the
    premise that the [custom_type] carried no layout; [ir_fields] now does, and
    the PPX fills it for every [[@@sarek.type]] record from the same source as
    [elem_size]/[get]/[set]. A caller-supplied list was the only way for the
    described layout to disagree with the real one — and that disagreement was
    not an error but {e silently transposed data}, since {!scatter}/{!gather}
    would index at the wrong byte offsets. Deriving makes that unreachable
    instead of checking for it.

    Raises {!Soa.Unsupported} if the element type has no derivable flat-record
    layout ([ir_fields = None], i.e. a variant) or is otherwise not
    SoA-representable. *)
val create : 'a Vector.custom_type -> int -> 'a t

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
