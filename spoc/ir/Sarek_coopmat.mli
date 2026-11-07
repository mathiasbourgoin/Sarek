(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Cooperative matrix: the DEVICE-facing half.

    Every type and value of {!Sarek_coopmat_types} is re-exported here by
    [include module type of], with its type equalities intact, so
    [Sarek_coopmat.fragment], [Sarek_coopmat.Uint8] and the rest mean exactly
    what they meant before slice 3 split the module. Read
    {!Sarek_coopmat_types}'s interface for the vocabulary and the design
    rationale behind it.

    What is defined HERE is the part that consults {!Sarek_capability}, and it
    is separated for a mechanical reason as much as a conceptual one: the IR
    names a fragment, {!Sarek_capability} depends on the IR, and a vocabulary
    that also depended on {!Sarek_capability} would close a module cycle. *)

(* [struct include ... end] and not the bare module path: the bare form
   re-declares each type ABSTRACTLY, so [Sarek_coopmat.config] would be a
   distinct type from [Sarek_coopmat_types.config] and every value crossing
   between the IR and the capability layer would need a conversion that does
   nothing. This form preserves the type equalities, which is the whole point of
   the re-export. *)
include module type of struct
  include Sarek_coopmat_types
end

(** [verdict ~support cfg] judges a requested configuration against a device.

    [support = None] yields {!Sarek_capability.Unknown}, which
    {!Sarek_capability.permits} refuses. That is the reason this takes an option
    and the reason {!Framework_sig.capabilities} stores an option: an empty
    configuration list and an unprobed device are different facts, and only one
    of them is evidence.

    {b The gate must be keyed on the feature, not on the configuration list},
    and this is measured rather than argued. On this workstation
    [vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR] returns [VK_SUCCESS] and
    FOURTEEN configurations for the Raphael iGPU — a device that does not
    advertise [VK_KHR_cooperative_matrix] and whose [cooperativeMatrix] feature
    reads false. A probe that populated [ds_configs] from that call without
    first checking the extension would report the iGPU as fully capable, and
    this verdict would say [Available] for a device that cannot run the
    instruction. The Vulkan probe therefore leaves [ds_configs] empty unless the
    extension is advertised AND the feature is true; see
    [sarek-vulkan/Vulkan_api_device.ml]. *)
val verdict :
  support:device_support option -> config -> Sarek_capability.verdict

(** The {!Sarek_capability.Device_optional} record for a device that provides no
    cooperative-matrix support at all, or none matching the request. [evidence]
    is the probe's {!device_support.ds_evidence}; {!verdict} passes it, so the
    refusal says which backend rule refused this device. *)
val device_lacks_config : ?evidence:string -> config -> Sarek_capability.t

(** [fragment_verdict ~support f]: does the device offer any configuration one
    of whose four fragments is [f]? A kernel that declares, loads or stores a
    fragment without ever multiplying still needs the device's per-lane layout
    for it, so the launch gate asks this for every fragment as well as
    {!verdict} for every multiply-add. [None] support is [Unknown]. *)
val fragment_verdict :
  support:device_support option -> fragment -> Sarek_capability.verdict

(** {1 The accuracy policy (docs/design/f16-relaxed-accuracy.md §6.1)}

    A float configuration is {!Relaxed_bounded}: the hardware chooses the order
    of the additions, so its result is within a derived bound of the exact value
    but not bit-identical to the interpreter. The owner's decision of 2026-07-27
    is that such a deviation needs an EXPLICIT opt-in and that the launch fails
    without it; these three values are that opt-in. *)

(** [accept_relaxed_accumulation ~reason] admits the float configurations that
    have a derived bound ([f16 * f16 + f32 -> f32] and
    [bf16 * bf16 + f32 -> f32]; the latter also needs
    {!accept_operand_rounding}) for the rest of the process. One-way: there is
    no withdrawal, so a capability check that has passed stays valid. The first
    non-empty [reason] is kept.

    @raise Invalid_argument if [reason] is empty or blank. *)
val accept_relaxed_accumulation : reason:string -> unit

(** The reason recorded by {!accept_relaxed_accumulation} at {b process} scope,
    if it was called. Like the other [*_reason ()] readers of this module it
    does not see [Kernel]-scoped grants: the verdicts take a budget
    ({!policy_verdict_for}). *)
val relaxed_accumulation_reason : unit -> string option

(** {2 Operand rounding (specs/matrix-pattern-rewrite.md)}

    A float32 matrix product computed on matrix units must first round its
    operands to a narrower type. That is a larger deviation than the
    accumulation order {!accept_relaxed_accumulation} admits, so it has its own
    opt-in, per precision: accepting one does not accept another — for the
    matrix rewrite. A hand-built [BFloat16] configuration is admitted by either
    [Operands_bf16] or [Operands_bf16x3], since the configuration cannot tell a
    split term from a plain rounded operand. [Operands_bf16x3] splits each
    operand into three bfloat16 terms ([x = hi + mid + lo]) and keeps the six
    products of order at most two, for near-float32 accuracy. Process-wide and
    one-way, like {!accept_relaxed_accumulation}. *)

type operand_precision = Sarek_policy.operand_precision =
  | Operands_f16
  | Operands_bf16
  | Operands_bf16x3

val operand_precision_name : operand_precision -> string

(** @raise Invalid_argument if [reason] is empty or blank. *)
val accept_operand_rounding :
  precision:operand_precision -> reason:string -> unit

(** The reason recorded for that precision, if it was accepted. *)
val operand_rounding_reason : operand_precision -> string option

(** {2 Automatic float rewrite (specs/matrix-pattern-dynamic.md)}

    [Execute.run_vectors] can rewrite a program's OWN float GEMM-shaped kernels
    onto matrix units, without the program having asked for matrix units. That
    is a separate decision from accepting the accuracy deviation: the accuracy
    opt-ins ({!accept_relaxed_accumulation}, {!accept_operand_rounding}) admit
    matrix-unit configurations for code that asks for them explicitly
    ([Sarek_gemm.Coopmat], [Sarek_matrix.run]); they do NOT enable the automatic
    rewrite. The automatic float rewrite happens only when this opt-in AND the
    accuracy opt-ins required for the configuration were all given. Granting one
    never enables more than it names. int32 (exact) products are rewritten
    automatically without any opt-in.

    This opt-in has no environment variable, like the accuracy opt-ins: it is a
    statement in the program's source. Process-wide, one-way, thread-safe. *)

(** [accept_automatic_float_rewrite ~reason]: the first non-empty [reason] is
    kept; later calls are no-ops.

    @raise Invalid_argument if [reason] is empty or blank. *)
val accept_automatic_float_rewrite : reason:string -> unit

(** The reason recorded by {!accept_automatic_float_rewrite}, if it was called.
*)
val automatic_float_rewrite_reason : unit -> string option

(** [policy_verdict cfg] judges the accuracy regime, independently of any
    device:
    - configurations whose four component types are all integer are [Available]
      (strict: exact accumulation);
    - [f16 * f16 + f32 -> f32] is [Available] after
      {!accept_relaxed_accumulation};
    - [bf16 * bf16 + f32 -> f32] is [Available] after
      {!accept_relaxed_accumulation} AND {!accept_operand_rounding} for
      [Operands_bf16] or [Operands_bf16x3] (its load rounds float32 operands);
    - every other configuration with a float component is [Unavailable] (no
      derived bound, §5.4). A refusal is a {!Sarek_capability.Policy} record
      naming the missing call. The launch gate and the interpreter apply it
      after the device verdict. *)
val policy_verdict : config -> Sarek_capability.verdict

(** [policy_verdict_for budget cfg]: {!policy_verdict} against the grants of
    [budget] (pure). [policy_verdict cfg] is
    [policy_verdict_for (Sarek_policy.budget ~kernels:[]) cfg], the process
    grants only, for a caller with no kernel; a caller that knows the kernel
    passes [Sarek_policy.budget ~kernels:[name]] so that a [Kernel]-scoped
    [Accumulation_order] or [Operand_rounding] grant is honoured for that kernel
    only (the launch gate, the interpreter). *)
val policy_verdict_for :
  Sarek_policy.Budget.t -> config -> Sarek_capability.verdict

(** [policy_decision_for budget cfg]: the same judgement as a decision for the
    explain report. [Refused] names the missing capability (the [cap_name] of
    the refusal); [Applied] says [Exact_by_construction] for an integer
    configuration and otherwise [Granted] with the grants it rests on
    ([Accumulation_order], and the operand rounding of a bf16 operand) and their
    reasons. *)
val policy_decision_for :
  Sarek_policy.Budget.t -> config -> Sarek_policy.decision

(** {1 Scaled accumulation (docs/design/coopmat-scaled-accumulate.md)} *)

(** [scale_add_operands_verdict ~dst_frag ~src_frag]: are these the fragments
    {!Sarek_ir_types.CM_scale_add} is defined on? [dst_frag] must be an f32
    accumulator and [src_frag] an s32 or f32 accumulator of the same shape and
    scope. A u32 source is refused by name: converting it would need a meaning
    of its own (its values above [2^31] are not s32 values), and none is
    defined. [Backend_structural] when refused: no device or decision changes
    it. Shared by the launch gate, the interpreter's gate and the evaluator. *)
val scale_add_operands_verdict :
  dst_frag:fragment -> src_frag:fragment -> Sarek_capability.verdict

(** [scale_add_verdict ~support ~dst_frag ~src_frag] is the launch gate's
    question for one [CM_scale_add] (§3 of the design): the operands as above,
    then whether the device's backend lowers the statement
    ({!device_support.ds_scale_add}). [None] support is [Unknown]; the fragments
    themselves are checked by {!fragment_verdict}, separately. No accuracy
    policy is involved: the statement is Strict. *)
val scale_add_verdict :
  support:device_support option ->
  dst_frag:fragment ->
  src_frag:fragment ->
  Sarek_capability.verdict
