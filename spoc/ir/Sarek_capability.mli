(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** The kernel/backend capability vocabulary (#64, slice 1).

    A "capability" in Sarek is not one boolean per device. Two days of measured
    backend defects (docs/fp-contraction-policy.md) produced facts that live at
    four different layers, and a model that collapses them into a single device
    flag is wrong — specifically, wrong in the direction of silently permitting
    things:

    - Metal has no [double] at all. No device query can tell you that; it is a
      property of the Metal Shading Language.
    - ACO fuses an f32 multiply into the f32→f16 narrowing regardless of what
      the driver advertises. The device reports the feature and the feature is
      broken. A device flag would say "yes".
    - Apple Silicon OpenCL reports no [cl_khr_fp64], but the question that
      actually decides whether a build succeeds there is whether the HOST clang
      can compile [double] for that target.
    - [-cl-fp32-correctly-rounded-divide-sqrt] is illegal unless the device
      advertises [CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT] — and local devices lack
      the bit and accept the flag anyway, so "it was accepted" proves nothing.

    This module is the vocabulary those facts get stated in. It holds no policy
    of its own and no backend dependencies: it lives in [spoc/ir] beside
    {!Sarek_ir_analysis}, whose [feature] type says what a kernel REQUIRES,
    while this says what a target PROVIDES and why it might not.

    {1 Slice scope}

    Slice 1 implements the {b static} half only — the diagnostic for
    {!Backend_structural} absence, where the answer is known from the backend
    alone with no device in hand. The other kinds are named here because naming
    them is what stops the next capability from being modelled as a boolean;
    their probes are later slices. See docs/design/capability-model.md. *)

(** {1 What kind of thing a capability is} *)

(** The layer that decides whether a capability is present. The kind determines
    {e when} the question can be answered, and therefore whether a static
    diagnostic or a launch-time gate is the right instrument. *)
type kind =
  | Backend_structural
      (** The target LANGUAGE cannot express it. Metal has no [double]; WebGPU
          has no [f64]. Decidable from the backend alone, before any device
          exists. Never revisable by buying hardware. This is the only kind
          slice 1 acts on, because it is the only kind that needs no probe. *)
  | Device_optional
      (** The backend supports it but a given device may not: OpenCL
          [cl_khr_fp64] / [cl_khr_fp16], Vulkan [shaderFloat16], CUDA f16 below
          sm_53, tensor cores absent on Pascal (sm_61). Answerable only with a
          device in hand — hence a launch-time gate, or a static one when the
          target device is pinned at compile time. *)
  | Host_toolchain
      (** A property of the HOST compiler or headers, not of the device. On
          Apple Silicon OpenCL the operative question is whether host clang
          compiles [double] for that target, not what the device reports; NVRTC
          needs [cuda_fp16.h] on the include path. Probed by attempting a host
          compile, never by querying a device. *)
  | Toolchain_semantic
      (** The device has the feature, the driver advertises it, and the shader
          compiler mistranslates it anyway. ACO fusing an f32 multiply into the
          f32→f16 narrowing is the type case: three front ends, one backend
          compiler, and pocl on x86 does not do it — which localises the defect
          to the compiler, not the device and not the API. Cannot be queried at
          all. Only measured. A device saying "yes" must NOT override this. *)
  | Policy
      (** We refuse something that works, by decision. Distinct from
          {!Toolchain_semantic}: that kind is the evidence, this is the verdict.
          A [Toolchain_semantic] fact is revised by a new MEASUREMENT; a
          [Policy] refusal is revised by a DECISION. Keeping them apart is what
          lets the diagnostic say which one the author is looking at. *)
  | Flag_legality
      (** A build option that is only legal when a device bit is set —
          [-cl-fp32-correctly-rounded-divide-sqrt] requires
          [CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT]. Its own kind because the
          failure mode is unique: the runtime does not enforce it, so the flag
          is ACCEPTED on devices that lack the bit. Acceptance is not evidence
          of support, which is exactly the inference a boolean model invites. *)

(** Stable lowercase-hyphenated rendering, e.g. ["backend-structural"]. Appears
    in diagnostics, so it is part of the tested message. *)
val kind_name : kind -> string

(** Whether answering this kind requires a device (or host toolchain) to be
    probed. [false] for {!Backend_structural} and {!Policy} — those are
    decidable statically, which is why they can be enforced at codegen.

    This is the predicate that decides static-vs-dynamic, so it is what a later
    slice consults to know which capabilities still need a launch gate. *)
val kind_needs_device : kind -> bool

(** {1 Evidence} *)

(** How we know. Kept in the record because the fp-contraction work established
    that a capability claim without a named device and toolchain is not usable —
    and because "measured here" and "reported by a vendor document" have
    different revisability. *)
type evidence =
  | Measured of string
      (** Observed by us. The string must name device AND toolchain. *)
  | Quoted of string
      (** Taken from a specification or vendor document, not measured here. *)
  | By_construction of string
      (** True of the emitter/type system itself, checkable by reading it. *)

val evidence_text : evidence -> string

(** ["measured"] / ["quoted"] / ["by construction"]. Rendered into the
    diagnostic so a reader can tell a measurement from a citation without
    opening this file. *)
val evidence_provenance : evidence -> string

(** {1 Capabilities} *)

type t = {
  cap_name : string;  (** e.g. ["float64"]. Matches the DSL-visible name. *)
  cap_kind : kind;
  cap_why : string;  (** One sentence: why the target lacks it. *)
  cap_evidence : evidence;
  cap_remedy : string option;  (** What the author should do instead. *)
}

(** {1 Verdicts} *)

(** The result of asking whether a target provides a capability.

    Three-valued on purpose. A two-valued answer forces an unprobed device into
    one bucket, and the bucket it lands in is "permitted" every time somebody
    writes [not unsupported]. *)
type verdict =
  | Available
  | Unavailable of t
  | Unknown of string  (** Could not determine; the string says why. *)

(** [permits Available = true], everything else [false].

    {b The safety property of this module.} [Unknown] does not permit. A device
    or toolchain we failed to probe is refused, not admitted — because every
    defect that motivated #64 was a case of something being permitted by
    default. Anything that needs to know "is this allowed" must go through this
    function rather than pattern-matching [Unavailable] and treating the rest as
    fine. *)
val permits : verdict -> bool

(** The first non-permitting verdict, if any. [None] means every verdict
    permits. Written in terms of {!permits} so [Unknown] cannot leak through. *)
val first_refusal : verdict list -> verdict option

(** {1 Rendering} *)

(** The diagnostic sentence. Names the capability and the target, states the
    kind and the provenance of the evidence, and gives the remedy when there is
    one. Its exact shape is asserted by the tests: a message that does not name
    the capability and the target is the failure mode this whole issue exists to
    remove. *)
val explain : target:string -> t -> string

(** {1 The static half (slice 1)} *)

(** [refuse_if_used ~raise_ ~target cap feature k] raises (through the backend's
    own error functor) when [k] uses [feature] and [target] structurally lacks
    [cap].

    Deliberately NOT {!Sarek_ir_codegen.reject_feature}. That composer says "not
    YET supported (#57 slice 2)" — a claim about a queue position, true of a
    backend nobody has got to. This one is for capabilities the target can never
    have, where a promise of future support would be a lie.

    [raise_] is a parameter for the same reason it is there: [spoc/ir] has no
    backend dependencies, and each backend raises through its own error functor
    so the message carries the right backend tag. *)
val refuse_if_used :
  raise_:(string -> unit) ->
  target:string ->
  t ->
  Sarek_ir_analysis.feature ->
  Sarek_ir_types.kernel ->
  unit

(** {1 Known capabilities} *)

(** Metal has no double precision.

    Until #64 slice 1 this was the project's clearest instance of the defect
    class the issue exists to kill: [Sarek_ir_metal.metal_type_of_elttype]
    mapped [TFloat64] to ["float"] with a comment saying Metal does not support
    double — a SILENT halving of precision, with no refusal anywhere on the
    path. A kernel written against binary64 semantics compiled clean and
    returned binary32 answers. *)
val float64_absent_metal : t
