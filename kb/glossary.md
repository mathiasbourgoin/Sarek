---
title: Glossary
last-updated: 2026-07-27
status: live-doctrine
owner: human
schema-version: 2
---

# Glossary

**This is an index, not a source of truth.** Every entry names where the authoritative
definition lives and says only enough to let a reader find it. When this file and the
document it points at disagree, the document wins and this file is the bug.

The vocabulary below is spread over roughly six thousand lines of design documents, and
the same word means different things in different halves of it — "the f16 contract" has
two answers that differ in kind, `precise` names both a GLSL qualifier and a claim about
what it buys, and "measured" is not "admitted". That ambiguity has cost real time, which
is the only reason this file exists.

Two entries are anchored by `properties.md` declarations
(`PROP-GLOSSARY-CAPABILITY-KINDS`, `PROP-GLOSSARY-F16-MODEL-NAMES`): if the identifiers
they name are renamed in the source, CI goes red here rather than leaving a glossary that
quietly describes a vocabulary nobody uses any more.

---

## Evidence tiers

Source of truth: `docs/fp-contraction-policy.md` §2, the legend under the backend table.
A claim may not assert more than its tier supports.

| Tier | Means |
| --- | --- |
| `executed` | Ran on the named device and agreed with the interpreter. |
| `machine-code` | The emitted ISA was inspected and has the required shape; nothing was executed. |
| `compiler-output` | An intermediate representation was inspected (PTX, SPIR-V); the layer below is not constrained. |
| `by-construction` | The source hands the compiler nothing it can contract; no flag is relied on. |
| `unverified` | Believed, documented, or inherited from a vendor's documentation — not measured here. |

Qualifiers that appear in practice and are **stronger** than the bare tier:
`executed, element-wise over the whole finite binary16 domain` (all 63488 inputs, not a
sampled maximum) — the document repeatedly contrasts it with agreement between summary
statistics, which is weaker and has misled here before.

**Liveness / positive control.** Not a tier. A mandatory qualifier on any zero: a sweep
reporting 0 disagreements is indistinguishable from a sweep that did not run, so every
null carries a control that must *fail*. See `properties.md`, "the property this project
learned the hard way".

## Capability model

Source of truth: `docs/design/capability-model.md` §2–§3; the types are
`spoc/ir/Sarek_capability.mli`.

The **kind** determines *when* the question can be answered, and therefore whether a
static diagnostic or a launch gate is the right instrument.

| Kind | Decided by | Answerable | Example |
| --- | --- | --- | --- |
| `Backend_structural` | the target **language** | statically, no device | Metal has no `double`; WebGPU has no `f64` |
| `Device_optional` | the **device** | needs a device | `cl_khr_fp64`, `shaderFloat16`, sm_53 for f16 |
| `Host_toolchain` | the **host compiler/headers** | needs a host probe | can Apple clang compile `double` for this target |
| `Toolchain_semantic` | the **shader compiler** | only measurable | ACO *and* LLVM/AMDGPU fusing f32 mul into the f16 narrowing |
| `Policy` | **us** | statically | f16 refused on OpenCL because we measured it wrong |
| `Flag_legality` | a **build option × a device bit** | needs a device | `-cl-fp32-correctly-rounded-divide-sqrt` |

Three distinctions the document defends, because collapsing any of them has produced a
defect:

- `Toolchain_semantic` **must be able to override a device saying yes**. Any model where
  a device query is the final authority gets the AMD f16 fusion wrong.
- `Toolchain_semantic` is the **evidence**; `Policy` is the **verdict**. The first is
  revised by a new measurement, the second by a decision.
- `Flag_legality` is its own kind because the runtime does not enforce it: the flag is
  *accepted* on devices lacking the bit. **Acceptance is not evidence of support.**

**Verdict** — `Available | Unavailable of t | Unknown of string`, with
`permits : verdict -> bool` true for `Available` only.

> **`Unknown` does not permit.** A device or toolchain we failed to probe is refused, not
> admitted. Every defect that motivated the model was something permitted by default.

`permits` is written as an explicit match on all three constructors rather than
`v = Available`, so adding a fourth verdict is a compile error at the one place that
decides whether something may run. Machine-checked as
`PROP-CAP-UNKNOWN-DOES-NOT-PERMIT`.

**The admission test** (`capability-model.md` §5.1) — *does a correct lowering exist in
the target language?* If yes it is a codegen bug, however much it resembles a capability
gap. If no, it is a capability.

## f16 numerics

Source of truth: `docs/design/f16-relaxed-accuracy.md`; measurements under
`docs/measurements/`; per-backend behaviour in `docs/fp-contraction-policy.md` §2, §12,
§13.

**Regime A** — f16 as a **scalar** element type. Accepts on **exact** agreement with one
member of a finite named model set, plus the 1-ulp ceiling. Available because the
deviation has a closed form: one elided rounding, which a host reference can compute
exactly.

**Regime B** — cooperative matrix (`OpCooperativeMatrixMulAddKHR` and its Metal
equivalent). Accepts on a **derived numeric bound**, because the deviation is an ordering
freedom over 17 terms and there is no finite set to enumerate.

> **When quoting the f16 contract, name the regime.** "Sarek's f16 contract" is ambiguous
> and the two answers differ in kind, not degree. Unifying *downward* — putting scalar
> f16 on a numeric bound — admits a known defect (§0.1).

Regimes migrate B → A on measurement alone, and one already has (Metal
`simdgroup_half8x8`).

**The named model set.** Members are named by *which mandated rounding is elided*:

| Name | Elides |
| --- | --- |
| `S_strict` | nothing — the interpreter's semantics, always a member |
| `S_fuse_mul_into_narrowing` | the f32 multiply feeding a narrowing, rounded once |
| `S_absorb_all_into_final_narrowing` | multiply, intermediate narrowing and add, all into one rounding |
| `S_f32_mul_then_absorb_add` | the same, but the multiply keeps its own binary32 result |
| `S_drop_intermediate_narrowing` | **named and deliberately NOT admitted** — the IGC defect. It sits at 1 ulp like the others; the only instrument keeping it out is that it is not on the list. |

Since backlog-151 these are understood as **one rule at three settings**, not four
independent members — see `fp-contraction-policy.md` §13.4 (`R_local_absorb`,
`R_local_absorb_nocontract`, `R_local_absorb_opencl`), and note §12.4's generative rule
was measured **false**.

**`f16_relaxed_ceiling`** — no admitted deviation may exceed 1 ulp **of the binary16
value produced by the narrowing at which the rounding was elided**, not of the kernel's
final value. Derived, not a round number. It is **necessary and not sufficient**: the IGC
dropped-narrowing defect sits exactly at 1 ulp. With more than one intermediate
narrowing, which narrowing to evaluate at is **not settled**.

**Measured ≠ admitted.** All twenty f16 shapes are measured; two are admitted; the other
eighteen stay refused. The relaxation is an **allowlist, not a lifting** — a driver
nobody has swept keeps today's refusal automatically.

**Discriminating shape** — a shape on which the candidate models are not all the same
function. Eight of the twenty are **non-discriminating**, so a device result there is
evidence for nothing. The denominator is 12, not 20; a table of twenty zeros would
otherwise read as twenty confirmations.

**`precise`** — a GLSL qualifier this project emits on every f32/f64 local, lowered by
glslang to SPIR-V `NoContraction`. The invariant is *never delete it*
(`PROP-GLSL-PRECISE-ON-FLOAT-LOCALS`), **not** *it protects these shapes*: on RADV the
ISA is opcode-identical with and without it, and the stronger claim was retracted. The
"RADV ignores `NoContraction`" reading was also retracted — the decoration is
*inapplicable* to a conversion, not disobeyed.

## Process vocabulary

Source of truth: `skills-meta/health-2026-07-27.md`, which is workstation-local — see
`kb/index.md` for why. These names are carried here because they are the taxonomy
`properties.md` and commit messages use, and they should survive the log they came from.

| Class | Means |
| --- | --- |
| `gate-vacuous` | A check that reads green while checking nothing. 12 instances in one session. See `properties.md`. |
| `stale-tooling` | A tool reporting success while doing nothing — `git fetch` that does not update refs, a filter that truncates the data a decision is made on. |
| `agent-isolation` | Work lost or overwritten because an agent touched a shared checkout instead of its own worktree. |
| `evidence` | A verdict stated more confidently than its tier supports; a conclusion later refuted. |
| `schema-drift` | A record whose shape changed under a consumer that kept reading it. |
| `scope` | A reviewer prompt narrow enough that the reviewer walks past a bug in the file it is reading. |
| `git-mechanics` | Commits silently dropped by a rebase; conflicts from independent additions to one file. |
| `parallel-collision` | Two concurrent agents colliding on a shared name or resource. |
| `process-bypass` | A mandated gate skipped, and the skip not recorded. Silence is not a review. |

**Prove red** — a checker is not evidence until it has been mutated and observed to fail
*with the message it promises*. `scripts/check-kb-properties.test.sh` is the worked
example; a `scripts/prove-red.sh` generalising it is approved and not yet built.

**Carrier** — a file a fresh clone executes by itself (`.github/workflows/ci.yml`, the
`Makefile`). Only an executable reference forms an edge; a mention in a comment or a
document does not run anything. The distinction is what separates a tracked gate from a
tracked gate that is inert.

## Tracker references

`backlog-NN` is an **internal** tracker item and is deliberately written without `#`:
GitHub would auto-link a bare `#NN` to whatever holds that number on a counter now past
330, promising a page that does not exist. `PR #290` / `issue #135` are real GitHub
numbers. Full rule: `CONTRIBUTING.md`.

Note when reading source: OCaml comments predating that convention use `#NN` for backlog
items (`Sarek_capability.mli` "#64", "#142"). A cross-reference search needs both
spellings.
