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
  independently tested — an f64 *literal* never reaches the type arm at all.

**Expressible, not yet wired:**

- WGSL/WebGPU has no `f64` — same kind, deferred to slice 1b (#141 coordination).
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
  which measurably does not fuse. Closing this needs a compiler-identity probe
  that does not exist.
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
