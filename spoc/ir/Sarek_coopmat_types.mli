(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Cooperative-matrix (tensor-core) vocabulary and fragment type.

    backlog-62, slices 2 and 4b of docs/design/f16-relaxed-accuracy.md §7. This
    module is the backend-neutral half: what a cooperative-matrix configuration
    IS, what a device provides, and what a matrix fragment looks like. It holds
    no Vulkan types and no Metal types — the Vulkan probe lives in
    [sarek-vulkan/Vulkan_api_device.ml] and lowers into these values.

    It sits in [spoc/ir] beside {!Sarek_capability} for the same reason that one
    does: {!Framework_sig.capabilities} must be able to report what a device
    provides, and [spoc/framework] may not depend on any backend.

    {1 What this module deliberately does NOT do}

    It does not emit code, and it does not decide accuracy. {!regime} classifies
    a configuration into the two acceptance regimes of the design document's
    §1.6, but it does not admit anything: the device verdict and the accuracy
    policy ({!Sarek_coopmat.verdict}, {!Sarek_coopmat.policy_verdict}) do. The
    emitters are GLSL (integer configurations), CUDA/HIP
    (specs/coopmat-hip-wmma.md) and Metal (specs/coopmat-metal-simdgroup.md);
    nothing in the DSL produces a fragment, so a cooperative-matrix kernel is
    built as IR.

    {1 Why the type admits integers}

    §8 of the design document, adopted as binding by §7 slice 4b. Twelve of the
    fourteen configurations the local RX 7900 XTX advertises are integer;
    [SPV_KHR_cooperative_matrix] states that integer accumulation is exact at
    the precision of the result type; so those configurations are deliverable
    under Sarek's EXISTING strict accuracy contract, with no relaxation, no
    allowlist and no opt-in. {!accumulation_is_exact} is where that distinction
    is computed rather than asserted in prose. Retrofitting integer component
    types after an f16-only fragment type had shipped would be a wide, invasive
    change; admitting them now costs one variant group and one predicate. *)

(** {1 Component types} *)

(** The element type of one matrix operand or result.

    The float half is the two configurations that need the relaxed contract; the
    integer half is the twelve that do not. Closed on what has been MEASURED to
    exist on a device this project can reach — the Vulkan set of
    docs/design/f16-relaxed-accuracy.md §4 — plus [BFloat16], added with its
    measurement on HIP gfx11 (specs/matrix-pattern-rewrite.md). Metal's 8x8
    [bfloat] [simdgroup_matrix] (§7 slice 6) is still unmeasured and not
    lowered. *)
type component_type =
  | Float16
  | BFloat16
      (** bfloat16 operand (specs/matrix-pattern-rewrite.md). A [BFloat16]
          fragment is loaded from a [float32] buffer, each element rounded to
          bfloat16 (round to nearest even): exact for values already
          representable. Lowered on HIP gfx11 only. *)
  | Float32
  | Uint8
  | Sint8
  | Uint32
  | Sint32

(** Stable lowercase rendering: ["f16"], ["bf16"], ["f32"], ["u8"], ["s8"],
    ["u32"], ["s32"]. Matches the column headings of the design document's §4
    table and the [ctype] function of [tools/probes/vulkan_coopmat_probe.c], so
    a probe line and a Sarek diagnostic can be compared by eye. *)
val component_name : component_type -> string

(** Width in bits. *)
val component_bits : component_type -> int

(** Whether the component type is an integer type.

    Load-bearing rather than cosmetic: it is what {!accumulation_is_exact}
    reads, and therefore what decides whether a configuration falls under the
    strict contract or needs the relaxation. *)
val component_is_integer : component_type -> bool

(** {1 Scope} *)

(** The set of invocations that cooperate to hold one matrix.

    All fourteen configurations measured locally are [Subgroup] scope, and
    Metal's [simdgroup_matrix] is a SIMD-group (i.e. subgroup) construct too, so
    [Subgroup] is the only scope any planned backend can currently produce. The
    other three are modelled because {!Sarek_capability}'s lesson is that a
    value you cannot represent becomes a value you silently permit: a device
    reporting [Workgroup] scope must come back as an unrecognised-but-named
    scope, not be quietly rewritten to [Subgroup]. *)
type scope = Subgroup | Workgroup | Device_scope | Queue_family

val scope_name : scope -> string

(** {1 Shapes} *)

(** A [m x n x k] multiply-add shape: [A] is [m x k], [B] is [k x n], and [C]
    and the result are [m x n].

    A record of three ints rather than a closed variant of the shapes seen so
    far, because §7 slice 4b makes non-hard-coded dimensions binding: Vulkan
    measured 16x16x16 and Metal measured 8x8x8, and 8x8 is the ONLY size MSL
    offers, so a type that could only spell 16x16x16 would already be wrong for
    a backend whose numbers are in the same document. *)
type shape = {m : int; n : int; k : int}

val shape_name : shape -> string

(** {1 Configurations} *)

(** One multiply-add configuration a device advertises: [D = A x B + C].

    [cfg_saturating] is Vulkan's [saturatingAccumulation] — integer accumulation
    that clamps instead of wrapping. It is a separate field rather than a
    component type because it is a property of the OPERATION, not of any
    operand: the local device advertises the same [s8 x s8 -> s32] element types
    both with it and without it, and those are two distinct configurations that
    compute different functions. *)
type config = {
  cfg_shape : shape;
  cfg_a : component_type;
  cfg_b : component_type;
  cfg_c : component_type;
  cfg_result : component_type;
  cfg_saturating : bool;
  cfg_scope : scope;
}

(** A one-line rendering, e.g. ["16x16x16 f16*f16+f32->f32 subgroup"]. Appears
    in diagnostics. *)
val config_name : config -> string

(** {1 Accuracy regime — the §8 discriminator} *)

(** Is the accumulation of this configuration exact?

    True exactly when the addend and result component types are both integer.
    [SPV_KHR_cooperative_matrix] states that integer accumulation is performed
    at the precision of the result type and is exact, so such a configuration
    computes the same function as the interpreter and lands under Sarek's
    existing strict contract. Every float configuration is false: the
    specification leaves the ORDER of the [k + 1] additions to the
    implementation, which is a freedom no closed-form model can pin down (design
    document §5.1). *)
val accumulation_is_exact : config -> bool

(** The two acceptance regimes of docs/design/f16-relaxed-accuracy.md §1.6.

    [Strict] means bit-identity with the interpreter is still promised.
    [Relaxed_bounded] means only the derived error bound of §5.2 holds, which
    §6.1 makes an explicit opt-in rather than a diagnostic. *)
type accuracy_regime = Strict | Relaxed_bounded

val regime : config -> accuracy_regime

val regime_name : accuracy_regime -> string

(** {1 What a device provides} *)

(** A device's cooperative-matrix support, as probed.

    [ds_subgroup_size] is here rather than left to the caller because a
    subgroup-scope fragment's storage is distributed across exactly that many
    invocations, so it is part of the calling convention and not an unrelated
    device statistic. See {!components_per_invocation}. *)
type device_support = {
  ds_configs : config list;
  ds_robust_buffer_access : bool;
      (** Vulkan [cooperativeMatrixRobustBufferAccess]: whether out-of-bounds
          cooperative-matrix loads and stores are bounds-checked. *)
  ds_subgroup_size : int;
      (** Invocations per subgroup, as the device reports it — NOT a constant.
          Measured 64 on the RX 7900 XTX under radv / Mesa 26.1.4-arch3.1, where
          [Vulkan_plugin_base] had been hard-coding 32. *)
  ds_advertised_count : int;
      (** How many configurations the DEVICE reported, including any this build
          could not represent and therefore dropped from {!ds_configs}.

          Kept because dropping an unrepresentable configuration is the safe
          direction (it can only cause a refusal) but it is also SILENT, and a
          silent drop hid a real defect once: a wrong [VkComponentTypeKHR]
          enumerant table turned fourteen advertised configurations into six
          plausible-looking wrong ones. The equality
          [ds_advertised_count = List.length ds_configs] is the check that
          catches it, and it is hardware-independent — unlike any assertion
          about which configurations a particular GPU offers. *)
  ds_evidence : string;
      (** How the probe decided, in one sentence a refusal can quote: the
          backend's rule and the device facts it read (e.g. the Vulkan extension
          and feature bits, or the HIP gfx family and warp size).
          {!Sarek_coopmat.device_lacks_config} puts it in the refusal, so a
          refused launch says WHY this device, on this backend, offers no
          configuration — the backends decide by different rules and a single
          backend's wording would mislead on the others. *)
  ds_scale_add : bool;
      (** Whether this backend lowers {!Sarek_ir_types.CM_scale_add} on this
          device (docs/design/coopmat-scaled-accumulate.md §3). False unless a
          lowering exists AND was executed bit-identical to the interpreter on
          such a device: only the HIP probe sets it, on compute capability 11.0
          (gfx1100) with 32-wide wavefronts, the only chip it was executed on.
          Other gfx11 parts (gfx1150, gfx1151...) are lowered but not offered it
          until they are run. {b Known gap:} compute capability does not name
          the exact target, and gfx1101 and gfx1102 (gfx11.0.x targets) report
          11.0 too, so they are admitted although never executed. The exact
          target is only in [hipDeviceProp_t.gcnArchName] (the
          [hipDeviceAttribute*Arch*] attributes are unused in ROCm 7.2's
          [hip_runtime_api.h]), which the HIP bindings do not read; keying on it
          is a follow-up. It is a separate flag rather than a consequence of
          [ds_configs] because the statement has no producer to key on: a
          [CM_decl] or [CM_load] can create its source fragment without any
          multiply-add in the kernel. *)
}

(** [config_matches ~shape ~a ~b ~c ~result cfg] compares the DIMENSIONS and the
    four component types, and deliberately NOT [cfg_saturating] or [cfg_scope].

    Those two are what separates configurations that this predicate is used to
    group: the local device advertises the same [s8 x s8 -> s32] element types
    both saturating and not, and they compute different functions. Callers that
    need an exact match ({!verdict}) test them explicitly on top; callers that
    are searching ({!find_config}) rank them. Folding them in here would make
    the two behaviours indistinguishable at the call site. *)
val config_matches :
  shape:shape ->
  a:component_type ->
  b:component_type ->
  c:component_type ->
  result:component_type ->
  config ->
  bool

(** [find_config ~support ~a ~b ~c ~result ~shape] returns the first advertised
    configuration matching those component types and that shape, preferring a
    non-saturating one. [None] when the device was not probed or advertises no
    such configuration. *)
val find_config :
  support:device_support option ->
  shape:shape ->
  a:component_type ->
  b:component_type ->
  c:component_type ->
  result:component_type ->
  config option

(** {1 The fragment type (slice 4b)} *)

(** Which operand of [D = A x B + C] a fragment holds.

    [Accumulator] covers both [C] and the result [D], which are the same shape
    and the same component type in every configuration measured, and which
    SPIR-V and MSL both give a single fragment role. *)
type use = Matrix_a | Matrix_b | Accumulator

val use_name : use -> string

(** How a matrix is laid out in the buffer a fragment is loaded from or stored
    to, with [index] the buffer position of element (0, 0) and [stride] the
    distance between consecutive rows ([Row_major]: element (r, c) is at
    [index + r * stride + c]) or consecutive columns ([Col_major]: element (r,
    c) is at [index + c * stride + r]).

    A memory property of one load or store, not of the fragment: the same
    fragment may be loaded from a row-major buffer and stored to a column-major
    one. *)
type layout = Row_major | Col_major

val layout_name : layout -> string

(** A cooperative-matrix fragment: the DSL-side value one invocation holds a
    slice of.

    It is NOT a per-invocation array. A fragment is a subgroup-cooperative
    value: the whole subgroup collectively holds [rows x columns] components,
    each invocation holding {!components_per_invocation} of them at an
    implementation-defined position. That indeterminate position is why a
    fragment must not be indexable from the DSL, and it is the reason this type
    exists at all rather than the DSL using an array. *)
type fragment = {
  frag_use : use;
  frag_shape : shape;
  frag_component : component_type;
  frag_scope : scope;
}

(** [fragment_dims f] is [(rows, columns)], derived from {!use} and the shape:
    [A] is [m x k], [B] is [k x n], and an accumulator is [m x n]. Derived
    rather than stored so a fragment cannot be built claiming a size its shape
    does not have. *)
val fragment_dims : fragment -> int * int

(** Total components in the whole fragment, i.e. [rows * columns]. *)
val fragment_components : fragment -> int

(** The fragments a configuration operates on: [(a, b, c, result)]. *)
val fragments_of_config : config -> fragment * fragment * fragment * fragment

(** {2 The subgroup calling convention} *)

(** [components_per_invocation ~subgroup_size f] is how many components of [f]
    each invocation of the cooperating subgroup holds.

    [Error] when [subgroup_size] does not divide {!fragment_components}, which
    is a real constraint and not a defensive check: a 16x16 fragment over a
    64-wide subgroup is 4 components per invocation, and the same fragment over
    a 24-wide subgroup has no distribution at all. It returns a result rather
    than raising because it is the check a codegen slice must perform BEFORE
    emitting anything, and it is the check a device query must survive when a
    driver reports a subgroup size the configuration cannot be laid out over.

    The value itself is deliberately NOT a promise about WHICH components an
    invocation holds. That mapping is implementation-defined in both SPIR-V and
    MSL; only the count is portable. *)
val components_per_invocation :
  subgroup_size:int -> fragment -> (int, string) result

(** [config_fits_subgroup ~subgroup_size cfg] is true when every one of the four
    fragments of [cfg] can be distributed over a subgroup of that size. *)
val config_fits_subgroup : subgroup_size:int -> config -> bool
