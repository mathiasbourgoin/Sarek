# The kernel↔backend capability model

_What a kernel requires, what a target provides, and which layer knows._

**Status:** slice 1 landed; slices 2–5 proposed. **Issue:** #64 (absorbs the
capability half of #57 §7). **Date:** 2026-07-27.

---

## 1. Why a boolean per device is the wrong model

Three days of measured backend defects (`docs/fp-contraction-policy.md`) produced
a set of capability facts that do not live at the same layer:

- **Metal has no `double`.** `Sarek_ir_metal.ml` mapped `TFloat64 -> "float"`
  with a comment saying so and no refusal anywhere on the path. No device query
  can report this; it is a property of the Metal Shading Language.
- **ACO fuses an f32 multiply into the f32→f16 narrowing** regardless of what the
  driver advertises — 620/63488 disagreements via rusticl, up to 5075/63488 via
  RADV, on RX 7900 XTX / gfx1100. pocl on x86 does not fuse, which localises the
  defect to ACO rather than to the device or the API. The device reports the
  feature; the feature is broken. A device flag says "yes".
- **Apple Silicon OpenCL has no `cl_khr_fp64`** — and the question that actually
  decides whether a build succeeds there is whether the *host* clang can compile
  `double` for that target, not what the device reports.
- **`-cl-fp32-correctly-rounded-divide-sqrt` is illegal** unless the device
  advertises `CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT` — and local devices lack the
  bit and accept the flag anyway. Acceptance is not evidence of support.
- **Pascal (sm_61) has no tensor cores**, no bf16, no FP8, and runs f16 at 1/64
  the f32 rate — present-but-catastrophic, which is neither "supported" nor
  "unsupported".

Collapsing these into one boolean per device is not merely lossy. It is lossy in
one direction: every one of them, reduced to a flag, reads as **permitted**.

## 2. The taxonomy — six kinds of capability

Implemented as `Sarek_capability.kind` (`spoc/ir/Sarek_capability.mli`). The kind
determines *when* the question can be answered, and therefore whether a static
diagnostic or a launch gate is the right instrument — that is
`kind_needs_device`.

| Kind | Decided by | Answerable | Example |
|---|---|---|---|
| `Backend_structural` | the target **language** | statically, no device | Metal has no `double`; WebGPU has no `f64` |
| `Device_optional` | the **device** | needs a device | `cl_khr_fp64`, `shaderFloat16`, sm_53 for f16, tensor cores |
| `Host_toolchain` | the **host compiler/headers** | needs a host probe | can Apple clang compile `double` for this target; NVRTC needs `cuda_fp16.h` |
| `Toolchain_semantic` | the **shader compiler** | only measurable | ACO fusing f32 mul into the f16 narrowing |
| `Policy` | **us** | statically | f16 refused on OpenCL because we measured it wrong |
| `Flag_legality` | a **build option × a device bit** | needs a device | `-cl-fp32-correctly-rounded-divide-sqrt` |

Three distinctions do real work and are worth defending:

**`Toolchain_semantic` vs `Device_optional`.** A device flag and a compiler
behaviour are different facts about different components, and the compiler one
must be able to *override a device saying yes*. Any model where a device query
is the final authority gets ACO wrong.

**`Toolchain_semantic` vs `Policy`.** The first is the evidence, the second the
verdict. A `Toolchain_semantic` fact is revised by a new **measurement**; a
`Policy` refusal is revised by a **decision**. Keeping them apart is what lets a
diagnostic tell an author which one they are looking at — and stops a policy
refusal silently outliving the measurement that justified it.

**`Flag_legality` as its own kind.** Its failure mode is unique: the runtime does
not enforce it, so the flag is accepted on devices that lack the bit. That is
precisely the inference a boolean model invites ("it compiled, so it's
supported") and it deserves a name that says otherwise.

## 3. The verdict algebra

```ocaml
type verdict = Available | Unavailable of t | Unknown of string
val permits : verdict -> bool   (* Available only *)
```

Three-valued on purpose. A two-valued answer forces an unprobed device into one
bucket, and the bucket it lands in is "permitted" every time somebody writes
`not unsupported`. **`Unknown` does not permit**: a device or toolchain we failed
to probe is refused, not admitted. This is the module's safety property and it
has a dedicated test with a red-on-mutation proof.

`permits` is written as an explicit match on all three constructors rather than
`v = Available`, so adding a fourth verdict is a compile error at the one place
that decides whether something may run.

## 4. Slicing

**Slice 1 — vocabulary + the static half (this PR).** `Sarek_capability` in
`spoc/ir` (no backend deps, beside `Sarek_ir_analysis`, whose `feature` type says
what a kernel *requires* while this says what a target *provides*). The
`Backend_structural` diagnostic, and Metal's f64 refusal — the one measured fact
that was actively producing wrong answers.

**Slice 1b — route WGSL f64 through the table.** WGSL already refuses f64, but
via a bespoke `has_float64` / `params_have_float64` path with hand-written
strings. Routing it through `Sarek_capability` makes the message uniform and the
kind explicit. **Deliberately not done here**: #141 is auditing the WGSL backend
concurrently and this would collide. No WGSL file is touched by slice 1.

**Slice 2 — the dynamic launch gate.** `Execute.run` (`sarek/execute/Execute.ml`)
is the single point that has both the device and the IR, right beside
`check_launch_args`; `Framework_sig.generate_source` takes no device, so no
codegen path can consult capabilities today. Requires extending
`Framework_sig.capabilities`, which breaks the literal-record tests in
`spoc/framework/test/` — a cost worth paying once, for a set-valued feature field
rather than another bool. Also migrates the OpenCL/GLSL f16 refusals from
hand-written strings to structured `Toolchain_semantic` + `Policy` values, and
picks up the GLSL `int64_t` hole (#142).

> **Correction.** Slice 1 recorded this as a GLSL *fp64* hole — "emits `double`
> while never declaring `GL_ARB_gpu_shader_fp64`". That was **measured false** by
> #141: `glsl_header` takes `~uses_float64` and emits the extension, and
> `glslangValidator` accepts the f64 kernel (exit 0). The real hole is one type
> over — GLSL `int64_t`, whose `#extension` emission is gated on the float64
> conditions only, so a plain `int64 vector` kernel emitted a shader glslang
> rejects. It is `Device_optional`, not `Backend_structural`: GLSL *can* spell
> `int64_t`, but a Vulkan device may not provide `shaderInt64` — so it needs the
> device probe that `kind_needs_device Device_optional = true` calls for and
> that slice 1 deliberately does not build.
>
> **Emitter half fixed in #141**, so #142 is the device probe only. The two
> float64 conditions the `#extension` was gated on were the softmath helpers
> that bit-cast a double and a non-finite f64 literal spelled via
> `int64BitsToDouble`; `Sarek_ir_analysis.Int64` is now OR-ed in, so the line is
> emitted whenever the kernel uses int64 at all. The rejection before the fix
> was `syntax error, unexpected IDENTIFIER` at exit 2, exit 0 after. Regression
> gate: `glsl-validate/int64_only_store`, a validation-only kernel whose only
> wide type is int64 — the shape the corpus lacked, which is why the gap
> survived. Until #142 lands, a device without `shaderInt64` still fails at
> shader load rather than at launch with a Sarek diagnostic: loud and correct,
> but unattributed.

**Slice 3 — host-toolchain and flag-legality probes.** Needs machinery that does
not exist: a host trial-compile, and retention of the OpenCL extension string
(`Opencl_api.ml` parses `CL_DEVICE_EXTENSIONS` and keeps only the fp64 boolean,
so the correctly-rounded-divide-sqrt bit is discarded before anyone could check
it).

**Slice 4 — located diagnostics.** See §5.

**Slice 5 — affinity, not capability.** See §5.

## 5. What the model can and cannot express

Stated explicitly, because a capability model that quietly cannot represent the
facts that motivated it is worse than none.

**Expressible and enforced today:**

- Metal has no `double`. `Backend_structural`, refused at codegen, both at the
  per-element-type arm and at a whole-kernel gate. Both are load-bearing and
  independently tested — an f64 *literal* never reaches the type arm at all, and
  neither does an f64 *local* whose only appearance is its declared type.

  Those two motivating shapes were found independently — the literal by #64
  reasoning down from the capability model, the local by #141 reasoning up from
  the emitted source — and both searches landed on the same detector
  (`Sarek_ir_analysis.kernel_uses Float64`) at the same two `generate` entries.
  Convergence from opposite directions is the argument that {arm, whole-kernel}
  is the *complete* set of entry points, not merely the set someone thought of.

  #141 also revised the severity. Slice 1 described the pre-fix behaviour as "a
  silent halving of precision"; it was worse than that. The IR element type
  fixes the buffer stride as well as the arithmetic, and `Vector.float64` is 8
  bytes per element, so `device float*` strode the host buffer at 4 and every
  element after the first was a bit-half of its neighbour. The kernel did not
  lose precision, it read a different array — a wrong-answer defect, not a
  quality-of-result one.

**Expressible, not yet wired:**

- WGSL/WebGPU has no `f64` — same kind, deferred to slice 1b (#141
  coordination). #141's backend-wide sweep confirms slice 1b is a pure
  refactor and finds nothing for it to fix: WGSL already refuses `TFloat64`,
  `TInt64` and `TFloat16` with located errors, and is the only backend that
  refuses everything it cannot represent at the right width. It was the
  *precedent* Metal should have followed, not a second instance of the defect.
- f16 refused on OpenCL and GLSL — representable as `Toolchain_semantic`
  (evidence: the ACO counts) plus `Policy` (verdict). Currently hand-written
  strings; slice 2 structures them.
- Apple Silicon OpenCL / host clang — `Host_toolchain` exists as a kind; no probe
  machinery exists.
- `-cl-fp32-correctly-rounded-divide-sqrt` — `Flag_legality` exists as a kind;
  the device bit is discarded before it can be read.
- Pascal sm_61 has no tensor cores / bf16 / FP8 — `Device_optional`, and
  `compute_capability` is already in the capabilities record, so the probe is
  cheap. Slice 2.
- **`int64` on Vulkan/GLSL** — `Device_optional`
  (`VkPhysicalDeviceFeatures.shaderInt64` / `GL_ARB_gpu_shader_int64`). #141
  fixed the emitter half; the *device* half is unprobed, so today a device
  without the feature fails at shader load rather than at launch with a Sarek
  diagnostic. #142, slice 2. See the correction under §4 slice 2.

**Correctly NOT in the table — see §5.1 for the rule:**

- **Metal `TBool`** was the case that prompted §5.1, and the numbers behind it:
  MSL `bool` is one byte, the host gives a Sarek `bool` a 4-byte slot
  (`Sarek_ir_layout.scalar_size TBool = 4`, mirroring `Sarek_ppx`), and `bool`
  is an accepted `[@@sarek.type]` record field — so host `{bool;bool;int}` at
  0/4/8, size 12, met an emitted `typedef struct { bool a; bool b; int n; }` at
  0/1/4, size 8. Fixed in the emitter (#141): Metal now emits `int`.

  The instrument that catches this class is not this table but the totality
  sweep — `sarek/tests/codegen_golden/test_backend_type_width_totality.ml`. For
  every backend and every scalar element type it admits exactly **three**
  outcomes, and it is worth stating all three, because a reader who believes it
  is two will misread the third as impossible:

  1. the emitted device type occupies exactly `Sarek_ir_layout.scalar_size`
     bytes — the host's own width;
  2. the mapper **refuses**, with a diagnostic (`Match_failure`, `Not_found`,
     `Invalid_argument` and `Failure` are rejected as refusals — an incomplete
     match is not a policy);
  3. the device type is recorded as having **no memory form at all**, which
     exempts it from the width check.

  Outcome 3 is an escape hatch, and it is the one that could be used to defeat
  the sweep, so it is pinned rather than merely permitted: the complete set
  lives in `expected_no_memory_form`, and
  `test_no_memory_form_set_is_exactly_as_recorded` fails on **any** addition or
  removal. Widening it is a deliberate edit to a literal list, not something a
  codegen change can do quietly.

  Today that set is six entries, and they are there for two different reasons —
  a distinction anyone deciding whether their own case belongs there needs:

  - `TUnit` on all five of Metal, CUDA, OpenCL, GLSL and WGSL — **no object
    representation at all**. C's `void` is not a value, so there is nothing to
    give a width to; WGSL has no unit type whatsoever and the emitter writes a
    comment (`/* unit */`), which is not a type either.
  - `TBool` on WGSL — a **real value the language will not let you put in a
    buffer**. WGSL `bool` exists and is perfectly usable in registers; it is
    simply not host-shareable, and `naga` refuses it in a storage binding
    ("The type is not host-shareable") rather than choosing a width. The
    failure is loud and at shader-load time, never a wrong stride.

  If a candidate is neither — if the target *would* accept it in a buffer at
  some width — then it is outcome 1 or outcome 2, not outcome 3.

  This sweep is the concrete form of §5.1's closing point about complementary
  instruments.

**NOT expressible, and not fixed by any planned slice:**

- **Performance cliffs.** f16 at 1/64 f32 rate on Pascal is *present* and
  *ruinous*. The model is binary present/absent; "supported but catastrophically
  slow" is a third thing it cannot say. This is the affinity half of the backlog
  title and it is genuinely a different model — a cost, not a predicate. Slice 5,
  and it should not be bolted onto `verdict`.
- **Which shader compiler is in the stack.** The ACO fact is a property of
  ACO — not of RADV, not of the device, not of OpenCL-vs-Vulkan (it reproduces
  through three front ends). The model can *say* a capability is
  `Toolchain_semantic`, but it has no way to *identify* the compiler at runtime,
  so such a verdict can only be blanket-per-backend. That over-refuses on pocl,
  which measurably does not fuse — and it now over-refuses on Intel IGC too,
  which does not fuse either (`docs/fp-contraction-policy.md` §11.3, executed on
  Intel Arc / Meteor Lake-P). Closing this needs a compiler-identity probe that
  does not exist.

  > **This gap does not block the f16 barrier, and backlog #144 asked whether it
  > did.** The barrier's own scoping never depended on runtime identification:
  > it is emitted from a single site under `#if defined(__HIP__) ||
  > defined(__HIP_PLATFORM_AMD__)`, and the compilers that could get it wrong
  > never receive the source, because f16 is refused at codegen on OpenCL, GLSL,
  > Metal, WGSL and PTX. A preprocessor conditional is the compiler naming
  > itself, which is strictly stronger than any device-string or probe-based
  > identification this model could add. The gap is real for a *future*
  > `Toolchain_semantic` verdict that must be taken with only a device in hand;
  > it is not real for this one. See `docs/fp-contraction-policy.md` §11.4.
- **Source locations.** `Sarek_ir_types.kernel` carries no location and the IR
  has no per-node locations, so a codegen refusal names the capability and the
  target but not the kernel source line. #64 asked for a *located* error; slice 1
  delivers a named one. The fix is to thread `Sarek_ast.loc` through
  `Sarek_lower_ir` into the IR — slice 4, and a large change on its own.

### 5.1 What does not belong in the table

**A width mismatch with a correct in-language lowering is a codegen bug, not a
missing capability.**

The first real test of this was Metal `TBool`. It looks identical to the f64
case — same backend, same silence, same class of wrong answer — and the table is
right there as a convenient hook. It was correctly declined. A capability entry
is an assertion that a target **cannot provide** something; filing `TBool` there
would make the table claim Metal cannot express booleans, which is false. Metal
has `bool`, and a correct lowering exists (`int`, which CUDA and OpenCL already
emit). What was wrong was the emitter, and the fix belongs in the emitter.

The test is not "is it silent?" or "is it a width mismatch?" — both are true of
`TBool` and of Metal f64 alike. It is: **does a correct lowering exist in the
target language?** If yes, it is a codegen bug however much it resembles a
capability gap. If no, it is a capability.

This matters more as the table grows, because the pressure runs one way: the
table is discoverable, it produces a decent diagnostic for free, and filing
against it feels like progress. Every entry that should have been a codegen fix
is a permanent false claim about what a target can do, and it removes a working
feature from users of that backend.

The corollary is that the capability table and a codegen-correctness sweep are
**complementary instruments, neither subsuming the other**. The sweep finds
wrong lowerings of things the target supports; the table records things the
target does not support. A defect found by one is not evidence about the other,
and "we have a capability model now" is not a reason to stop sweeping.

## 6. Open decision

Inherited from `docs/design/f16-dsl-element-type.md` §7 and still open: should a
capability refusal be a hard error or a warn-and-emulate fallback?

Slice 1 takes **hard error**, but only for `Backend_structural`, where the
question barely arises — there is no in-language emulation to fall back to, and
`Sarek_real64`'s `Fallback_df64` substrate is the remedy the diagnostic names.
The question stays genuinely open for `Device_optional`, where a fallback often
does exist. It should be decided per kind, not once for everything, and it wants
a human call.
