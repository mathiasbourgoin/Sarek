# The relaxed f16 accuracy contract, and the tensor-core paths it unblocks

_What Sarek promises about f16 results when bit-identity with the interpreter is
no longer required, and how backlog-62 (Vulkan cooperative-matrix) and backlog-63 (Metal
`simdgroup_matrix`) are sliced on top of it._

**Status:** design only; the two decisions this document opened were taken by the
project owner on 2026-07-27 and are recorded as decided in **§3** (scalar f16 is
relaxed, as an explicit performance trade) and **§6** (user-facing friction is
proportional to the strength of the evidence). **No refusal is lifted by this
change.** **Tracked as:** backlog-62, backlog-63; depends on the f16 scalar type (backlog-57) and the
capability model (backlog-64). **Date:** 2026-07-27.

**Amended 2026-07-27 (Metal measurement).** §7 slices 5 and 6 are **done** — an
Apple M4 became reachable, both probes ran, and the results are wired in: §2
gains a Metal row (**strict**, 0/63488, no relaxation needed) and the document's
first **Regime B** rows. Two things changed that are not just new rows: **§1.3's
ceiling was wrong** and is corrected to evaluate at the elided narrowing rather
than on the final value, with the counterexample that broke it; and **§8's
integer-coopmat fallback is Vulkan-only**, because Metal has no integer
`simdgroup_matrix` at all. **Still no refusal is lifted by this change.**

**Amended 2026-07-27 (slice 1, ACO measurement).** §7 slice 1 has **run**, on
the two shapes this project has device numbers for. Both ACO stacks are
**admitted under Regime A** on those shapes: every kernel variant is
bit-identical to exactly one named closed-form model on 63488/63488 inputs, on
both local devices, with zero inputs matching no model. **§9.3's open risk is
closed** — RADV's two-narrowing shape matches a model plain *and* with
`precise`, and the `precise` puzzle is reconciled at machine-code tier: the
decoration forbids a multiply-into-**add** contraction and cannot reach a
**conversion** absorbing its operand. §1.2 gains **two** admitted members and
one explicitly **refused** named function. **No refusal is lifted by this
change** — slice 3 is still where that happens, and 18 of the 20 emittable
shapes remain unmeasured and therefore refused.

**Amended 2026-07-27 (backlog-151, the other 18 shapes).** The 18 shapes slice 1
left unmeasured have been swept, on both ACO stacks and all four local devices.
Three things changed and none of them lifts a refusal. **(i)** Slice 1's
candidate generative rule is **false** — absorption is a *local* peephole, not a
whole-tree property, and shapes A11, A12 and B4 break it with the machine code
for each (`fp-contraction-policy.md` §13.4). **(ii)** The corrected rule holds on
12 of 12 discriminating shapes on RADV and 11 of 12 on rusticl, and it means
**§1.2's model set does not grow per shape** — the four members below are *one
rule at three settings*, not four independent functions (§1.2's new note, and
§13.5). **(iii)** Two facts this document asserted are narrowed: the two ACO
front ends are **not** the same function on shapes slice 1 did not sweep, and
§1.3's ceiling needs an evaluation point stated when a shape has more than one
intermediate narrowing (§1.3's new note). Separately, RADV returns `-0` for
`0.0 - x` at `x = +0` on one input, through every barrier — a **failure** under
§1.2 rather than a relaxation, recorded at §13.6. **The other 18 shapes remain
refused**: slice 3 is still where a refusal is lifted.

**Read §0.1 first.** It is one paragraph and it is the argument every other
choice here follows from.

Companion documents, neither of which is edited here:
[`docs/fp-contraction-policy.md`](../fp-contraction-policy.md) is the measurement
record this design consumes;
[`docs/design/capability-model.md`](capability-model.md) is the machinery it
plugs into; [`docs/design/f16-dsl-element-type.md`](f16-dsl-element-type.md) §8
slice 3 is the scope note this document fills in.

---

## 0. The decision, and what it does not decide

The project owner has ruled that **bit-identical agreement with the interpreter
is no longer a requirement for f16**, provided the results are correct,
deterministic, the technical limitation is understood and explained, and all of
it is documented.

That is a genuine relaxation and it is what makes backlog-62 and backlog-63 reachable. It is
also not yet a specification. "Correct" without a bound admits anything; and this
repository has already shipped a gate whose tolerance was wider than the effect
it existed to detect (`docs/fp-contraction-policy.md` §11.5, backlog-118), where
the consequence was a fully collapsed implementation reading green.

This document turns the decision into something a test can fail.

### 0.1 The one argument to read before anything else

The natural way to spend a relaxation is a tolerance: *"within N ulp of the
interpreter"*. **It cannot work here, and the reason is arithmetic on two numbers
this project has already measured**, not a matter of taste.

Two behaviours have been observed on the f16 narrowing. One is a characterised
alternative rounding of the same expression — ACO absorbs the f32 multiply into
the f32→f16 narrowing, rounding once where the DSL mandates twice. The other is a
plain defect — IGC, under a `volatile` barrier, drops the intermediate binary16
narrowing altogether, which is a *different expression*
(`fp-contraction-policy.md` §11.4).

**Both are at most one ulp of binary16.** §11.4's worked divergence is
`x = 0.681640625`, where the device returns `1000.5` and the discipline mandates
`1001` — and 1000.5 and 1001 are **adjacent binary16 values** (1001 lies in the
binade [512, 1024), where the binary16 ulp is 0.5). So any band wide enough to
admit the first is wide enough to admit the second, and any band tight enough to
exclude the second excludes the first as well.

**A pure ulp gate cannot separate a characterised deviation from a plain defect
on this hazard.** That is why the contract in §1.2 is a *set of named,
closed-form reference semantics compared exactly*, and not a tolerance — and why
the 1-ulp ceiling of §1.3 is kept as a necessary check while being explicitly
declared insufficient on its own. Everything else in this document follows from
that one observation.

### 0.2 The two decisions the first draft left open, now taken

This document was first circulated with two questions marked as owner decisions.
Both have been settled and are recorded as decided, with the reasoning kept so it
can be re-derived:

- **Scalar f16: relax — recorded explicitly as a performance trade, not a
  reachability one.** §3.
- **User-facing friction is proportional to the strength of the evidence.** §6.

**The document still declines to apply the relaxation as one undifferentiated
thing**, for the reason set out in §3: the two paths being unblocked cannot hold
bit-identity for *different* reasons, only one of which is unfixable, and letting
the unfixable one carry the fixable one would give the fixable one away for free.

---

## 1. What replaces bit-identity

### 1.1 Not an ulp band. A finite set of admissible rounding semantics.

The short form of this is §0.1 and it is the argument the whole design rests on.
Here is the measurement behind it.

Two behaviours have been observed on the f16 narrowing, and they are not the same
kind of thing:

| behaviour | where measured | what it is |
|---|---|---|
| the f32 multiply is absorbed into the f32→f16 narrowing, rounding **once** where the DSL mandates twice | ACO — hiprtc/gfx1100, rusticl/radeonsi, RADV | a well-defined alternative rounding of the *same* expression |
| the intermediate binary16 narrowing is **dropped entirely** | IGC on Intel Arc, under a `volatile` barrier (`fp-contraction-policy.md` §11.4) | a different expression |

Both are **at most one ulp of binary16**. §11.4's worked divergence is `x =
0.681640625`: the device returns `1000.5` where the discipline mandates `1001`,
and 1000.5 and 1001 are *adjacent* binary16 values (1001 lies in the binade
[512, 1024), where the binary16 ulp is 0.5). So a 1-ulp band admits both, and a
band tight enough to exclude the second excludes the first as well. **A pure ulp
gate cannot separate a characterised deviation from a plain defect here.** That
is not a hypothesis about tolerances in general; it is arithmetic on two numbers
this project has already measured.

### 1.2 The contract for **scalar f16** arithmetic

**Scope, stated before the rule because it is load-bearing.** What follows governs
**f16 as a scalar element type** — narrowings, widenings and the arithmetic
written around them in a DSL kernel. It does **not** govern
`OpCooperativeMatrixMulAddKHR` or its Metal equivalent, which are accepted under a
*different* rule (§5) for a reason that is not a matter of effort. §1.6 sets the
two regimes against each other; read it before quoting either one as "the f16
contract".

> **Sarek's scalar-f16 contract, relaxed form.** For each *(backend, driver,
> expression shape)* Sarek records a finite set of **admissible reference
> semantics**. The device result must be **bit-identical to one member of that
> set**, on every input the gate sweeps. The interpreter's semantics — every
> operation rounded as written, per `fp-contraction-policy.md` §1 — is always a
> member. Any additional member is a *named, closed-form, measured* alternative,
> recorded per backend and per driver. A result matching no member is a
> **failure**, however small the numeric difference.

Two named members exist today, and both are functions a host reference can
compute exactly:

- **`S_strict`** — the interpreter. f16 arithmetic performed in binary32 and
  narrowed by an explicit round-to-nearest-even step at every narrowing the
  source writes. Already implemented twice in-tree, as `ref_discipline` in
  `sarek-vulkan/test/test_vulkan_f16_tripwire.ml` and its port into
  `sarek-opencl/test/test_opencl_f16_tripwire.ml`.
- **`S_fuse_mul_into_narrowing`** — every f32 multiply immediately consumed by an
  f32→f16 narrowing is evaluated at binary32 (or better) and narrowed in a
  **single** rounding. Already implemented as `ref_fused` in the same two files,
  and as the `fusedctl` variant of `tools/probes/opencl_f16_contraction_probe.c`.

**Two further members were added on 2026-07-27 by slice 1**, both measured
element-wise on RADV over the whole finite binary16 domain on two devices
(`fp-contraction-policy.md` §12.4). They exist because a *two*-narrowing shape
gives ACO a second combine that a one-narrowing shape cannot exhibit:

- **`S_absorb_all_into_final_narrowing`** — the multiply, the intermediate
  binary16 narrowing and the f32 add are **all** absorbed into the final
  narrowing: one rounding where the DSL mandates four. RADV emits this as a
  single `v_fma_mixlo_f16` taking `x`, `1.1` and `1000`.
- **`S_f32_mul_then_absorb_add`** — the same, except the multiply keeps its own
  correctly-rounded binary32 result. This is what `precise` buys on RADV, and
  therefore **the model the shipped codegen actually runs under**, since
  `Sarek_ir_glsl.gen_var_decl` emits `precise` on every float local.

**One function is named and deliberately NOT admitted**, and it is the reason
this section is a set of names rather than a bound:

- **`S_drop_intermediate_narrowing`** — the intermediate binary16 narrowing is
  dropped outright, the add still rounding to f32. This is the **IGC defect** of
  `fp-contraction-policy.md` §11.4 — and slice 1 found it is also reachable on
  RADV, exactly, by barriering the f32 intermediates and nothing else. So the
  same closed-form function is a plain defect on one stack and an
  on-demand behaviour on another, it sits at 1 ulp like everything else here,
  and **the only instrument that keeps it out is that it is not on the list**.
  §0.1 argued a tolerance could not separate a characterised deviation from this
  defect; slice 1 turned that argument into a measurement.

> **These four are ONE rule at three settings, not four members — measured
> 2026-07-27 (backlog-151).** Read as a list, §1.2 invites the question this
> design was most exposed to: does the list grow with every expression a user
> writes? It does not, and the reason is that the list is not primitive. All
> four are what a single *local* rule produces on the two shapes slice 1
> measured:
>
> *each f32→f16 narrowing absorbs the single f32 operation immediately feeding
> it — a multiply, an add/sub, or an explicit fma — rounding it once; a multiply
> feeding an addition is separately contracted into a single-rounded **binary32**
> fma; an intermediate binary16 narrowing consumed only by f32 arithmetic may be
> elided; **every other f32 operation keeps its own correctly-rounded binary32
> result**.*
>
> Three booleans — contract, elide, sink-into-a-select — pick the setting, and
> each is a measured property of a (driver, front end, decoration) triple rather
> than a free parameter. `fp-contraction-policy.md` §13.4 has the settings, the
> verification on ten shapes the rule was not fitted to, and the disassembly.
> **So the model set is `{S_strict}` plus one generator, for any shape.** §13.5
> is the argument that this is the difference between a contract and a lookup
> table, and it is the whole reason the 18 shapes were worth sweeping.
>
> The rule is **not** the one slice 1 proposed. That one absorbed the *whole*
> tree and is refuted at machine-code tier: `v_fma_mixlo_f16` takes one
> multiply-add and one conversion, so a `v_floor_f32` or a `v_cndmask_b32`
> between the multiply and the narrowing stops it dead (§13.4).

This is exact agreement, not a tolerance, so it is strictly *tighter* than
today's bit-identity requirement in every direction except the one deviation it
names. It is the same discipline as `Test_helpers.classify_df64_result` — an
allowlist of specific, argued deviations rather than a widened bound — and it is
available to us in a stronger form than df64 got, because this deviation has a
closed form and df64's does not.

### 1.3 The collapse ceiling, kept anyway

The model set is the discriminator; a ceiling is still required, for the case
where a *new* driver produces something that matches no model and we have to
decide whether it is a near-miss or garbage. Mirroring
`Test_helpers.df64_collapsed_ceiling`:

> **`f16_relaxed_ceiling`** — no admitted deviation may exceed **1 ulp of the
> binary16 value produced by the narrowing at which the rounding was elided**,
> with subnormal results compared absolutely against `2^-24` (the binary16
> subnormal spacing). Derivation, not a round number: every deviation in the
> admitted class is the *elision of exactly one round-to-nearest step*, and a
> single elided rounding moves a value by at most half an ulp at the elided step,
> hence at most one ulp of that step's own result. Anything larger is not a
> rounding difference and must not be filed as one.

**The ceiling is evaluated at the elided narrowing, NOT on the kernel's final
value, and that is a correction.** The first version of this section said "the
final value", on the reasoning that half an ulp at the elided step is "at most
one ulp at the final step". **That last inference is false**, and it was measured
false on Metal — the one backend swept with a control able to compute the
admitted model exactly (`fp-contraction-policy.md` §10.14,
`tools/probes/metal_f16_narrowing_probe.m`).

> **The case that broke it.** Shape `f16(f16(x*1.1) + 1000)` at
> **`x = -907.5`**. The exact product is `-998.250021636…`, whose binary32
> rounding is **exactly `-998.25`** — a binary16 tie in the binade [512, 1024),
> where the ulp is 0.5. `S_strict` rounds that tie to even and gets `-998`, then
> `-998 + 1000 = ` **`2.0`**. `S_fuse_mul_into_narrowing` narrows the exact
> product instead, is not sitting on the tie, gets `-998.5`, then
> `-998.5 + 1000 = ` **`1.5`**.
>
> The deviation at the elided narrowing is `-998` against `-998.5`: **exactly 1
> ulp of binary16 there**, as the derivation says. But 2.0 and 1.5 lie in the
> binade [1, 2), where the binary16 ulp is `2^-10`, so on the **final** value the
> same deviation measures **512 ulp** (against the smaller magnitude, 1.5) or
> **256 ulp** (against 2.0) — the count depends on which of the two you take as
> the denominator, and a gate must say which. Either way it is hundreds of ulps
> against a ceiling of one, produced by the *admitted* model on a *conforming*
> device.

The mechanism is general and has nothing to do with Metal: the `+ 1000` cancels
the leading bits, so the result lands in a far smaller binade than the value the
rounding was elided at, and the ulp is re-scaled by the ratio of the two binades.
Any expression in which a narrowing is followed by a cancelling operation does
this. A final-value ceiling is therefore not merely loose — it **rejects correct
results**, which is worse than the failure mode it was written to catch.

**Consequences for the harness.** `classify_f16_result` (§7 slice 0) must
evaluate the ceiling per *narrowing*, which means the reference models have to
expose their intermediates rather than only their final value. That is a small
change to `ref_discipline` / `ref_fused` — they already compute the intermediates
— but it must be made deliberately, because the natural implementation compares
only what the kernel wrote to memory. Where a shape's intermediates are not
observable, the ceiling is **not applicable** and must be reported as such rather
than silently evaluated on the final value.

> **And it needs a point named when a shape has more than one intermediate
> narrowing — measured 2026-07-27 (backlog-151).** "At the elided narrowing" is
> unambiguous for every shape this document had seen, all of which have at most
> one. Shape B4 of the catalogue,
> `f16(f16(f16(x*1.1)+1000)*1.1)`, has two, and the two answers differ by three
> orders of magnitude: at the **outer** narrowing the admitted model exceeds the
> ceiling on **719 of 63488** inputs and reaches 1638 ulp, because two elisions
> separate it from `S_strict` there and the derivation above covers exactly one;
> at the **innermost**, where one elision separates them, it is **0.500044 ulp
> with zero exceedances**. (On the final value the same deviation reaches
> **1.85e+06 ulp**, which is this correction reproduced on a third shape.)
>
> The harness evaluates at the innermost narrowing and reports the
> intermediate-narrowing count, so a shape with more than one reads as
> *partially* covered rather than as a clean pass. **What the ceiling should say
> about the remaining elisions is not settled here** and is work for slice 0,
> which is where `classify_f16_result` is built. `fp-contraction-policy.md`
> §13.7.

The ceiling is **necessary and not sufficient** — §1.1 is the proof, since the
IGC dropped-narrowing defect sits exactly at 1 ulp. Both checks run; both must
pass. Note that §0.1 and §1.1 are unaffected by this correction and were
re-read against it: both argue about a deviation *at the narrowing* — the
dropped-narrowing defect and the fused multiply are compared where they occur —
so the observation that a pure ulp gate cannot separate them stands exactly as
written, and if anything is strengthened, since the correct evaluation point is
now stated rather than left to the reader.

### 1.4 What "deterministic" means, operationally

Promised, and tested:

> **Same input, same physical device, same driver version, same Sarek build, same
> dispatch shape → bit-identical result, across launches within a process and
> across processes.**

Not promised, and **measured false** rather than merely doubted: the same result
across drivers, driver versions, vendors, or architectures. ACO returns one
answer on `f16(x*1.1)` and nvrtc/IGC/pocl return a different one, on the same
source, today (`fp-contraction-policy.md` §2, §11.3). Any statement of the strong
reading would be a false claim, so this document does not make it.

For cooperative-matrix specifically the promise is weaker still, and by
specification rather than by driver behaviour. `SPV_KHR_cooperative_matrix` says
of `OpCooperativeMatrixMulAddKHR`, verbatim: **"The order of the operations is
implementation-dependent"** and **"The internal precision of floating-point
operations is defined by the client API"**; and the mapping of matrix components
to invocations is likewise implementation-dependent. Evidence tier for those two
sentences: **unverified** in this repository's sense — quoted from the Khronos
SPIR-V registry, not measured here. The consequence is that a coopmat result may
legitimately depend on the dispatch shape and the subgroup mapping, so §5's
determinism gate varies the dispatch shape and *reports* rather than asserts
until it has been measured.

Test shape, all three host-only or single-device, none requiring a second
machine: (a) re-run the exhaustive sweep in-process and require bit-identical
output; (b) re-run in a fresh process; (c) for coopmat, vary workgroup size and
matrix tiling and record whether the result moves.

### 1.5 Scope — what the relaxation does NOT touch

Stated because a relaxation that leaks is worse than no relaxation.

- **f32, f64 and `Sarek_df64` contracts are unchanged.** Nothing here licenses a
  wider bound for any of them.
- **A backend that meets `S_strict` is held to `S_strict`.** CUDA/nvrtc and
  HIP/AMDGPU are 0/63488 today (`fp-contraction-policy.md` §2, §7). They do not
  gain an allowance because another backend needed one. The admissible set is
  keyed per backend and per driver; it is not a global widening.
- **Integer cooperative-matrix configurations stay strict.**
  `SPV_KHR_cooperative_matrix` states that integer additions "are performed at
  the precision of *Result Type*, are exact". On the local device 12 of the 14
  advertised configurations are integer (§4). They need no relaxation at all —
  see the recommendation in §8.
- **`Unknown` still does not permit.** The relaxation is an **allowlist**, not a
  lifting. A driver nobody has swept keeps today's refusal, automatically, with
  no new decision required. This is the single most important structural property
  of the design and §6 is how it is enforced.

### 1.6 Two acceptance regimes, and neither is "the contract" unqualified

The first draft of this document stated §1.2 as though it governed all f16 and
then accepted cooperative matrix against a numeric bound in §5. Those are two
incompatible acceptance rules and the document asserted both. They are now
separated explicitly, and the separation is the *same* scalar/coopmat split
already argued in §3.1 — carried into the contract rather than left in the
justification.

| | **Regime A — scalar f16** | **Regime B — cooperative matrix** |
|---|---|---|
| governs | f16 as a scalar element type: narrowings, widenings, the arithmetic around them | `OpCooperativeMatrixMulAddKHR` and its Metal equivalent |
| acceptance rule | **exact** agreement with one member of a finite named model set (§1.2), plus the 1-ulp ceiling (§1.3) | a **derived numeric bound** on `D = A×B + C` (§5) |
| why that rule | the deviation has a **closed form** — it is one elided rounding, and a host reference can compute it exactly | the deviation is an **ordering freedom over 17 terms**; there is no finite set to enumerate |
| can the other rule be used? | a bound would admit a known defect — §0.1 | a model set does not exist to be written down |
| user-facing friction (§6.1) | loud one-time diagnostic | mandatory explicit opt-in |
| can it migrate? | — | **yes** — if a given implementation's accumulation order is ever pinned to a closed form, that configuration moves to Regime A, and §6.1's friction falls with it |

**The distinction is not stylistic and it is not about effort.** Regime A's rule
is available because ACO's combine elides exactly one rounding, which is a
function. Regime B's is not, because the SPIR-V extension grants an ordering
freedom (§1.4) whose admissible results are combinatorially many. Anyone tempted
to unify them should note that unifying *downward* — putting scalar f16 on a
numeric bound — is exactly the move §0.1 shows admits a known defect.

**When quoting this document, name the regime.** "Sarek's f16 contract" is
ambiguous and the two answers differ in kind, not degree.

---

## 2. The per-backend deviation record

These are the numbers this design holds itself to, for **Regime A** (scalar f16,
§1.2). Every row names the device and the driver, per `fp-contraction-policy.md`
§1 corollary 3. Evidence tiers are that document's. **Regime B now has rows too**,
below the Regime A table — Metal's `simdgroup_matrix`, measured 2026-07-27. RADV's
coopmat numerics remain unmeasured (§4, last paragraph).

**SLICE 1 RAN, 2026-07-27, and both candidates met the rule — on the two shapes
it swept.** Every kernel variant on both ACO stacks is bit-identical to exactly
one named model on **63488 / 63488** inputs, on both local devices, with zero
inputs matching no model (`fp-contraction-policy.md` §12). rusticl's
count-and-first-divergence agreement is upgraded to element-wise; RADV's
two-narrowing shape, §9.3's open risk, matches a closed-form model under `precise`
*and* without it, and the `precise` puzzle is reconciled at machine-code tier.

**The admission is per shape, and only two of twenty shapes are ADMITTED.**
§1.2's model set is keyed on *(backend, driver, expression shape)* and always
was, so this is not the per-shape degradation §9.3 feared. **All twenty shapes
are now MEASURED** (backlog-151, `fp-contraction-policy.md` §13) — but measured
is not admitted, and under §1.5 the other 18 stay refused until slice 3 decides.
Read the verdict column as the current status, not as a plan.

**Three things backlog-151 changed in this table's reading, none of which moves a
verdict.**

1. **The rusticl and RADV rows are not the same function.** §12.3 concluded
   "the two ACO front ends behave identically on both shapes"; that was true of
   those two shapes and is false in general. rusticl keeps the intermediate
   narrowing where RADV elides it, does not contract multiply-add without being
   asked, and sinks a narrowing into the arms of a select where RADV does not
   (2863/63488 against RADV's 0 on the same shape). The allowlist has to be
   keyed on the **front end** as well as the backend.
2. **`precise` is inert on more shapes than §9.3's reconciliation implied.** It
   changes the model on exactly two of the twelve discriminating shapes; on the
   rest, including an explicit `fma()`, the disassembly is byte-identical
   (§13.4).
3. **One shape, A10 `f16(0. - x)`, returns a result matching NO model on RADV**
   — `-0` where IEEE requires `+0`, at `x = +0`, through every barrier. Under
   §1.2 that is a **failure**, not a candidate for the admissible set: it is not
   a rounding. rusticl is correct on it. §13.6.

| backend / driver | device(s) | deviation from `S_strict` | matches `S_fuse_mul_into_narrowing`? | evidence | verdict under Regime A |
|---|---|---|---|---|---|
| **CUDA / nvrtc** | GTX 1070 Max-Q, sm_61, CUDA 12.9, driver 580.119.02 | **0 / 63488** | n/a | executed | **strict**, unchanged |
| **HIP / AMDGPU** | RX 7900 XTX (gfx1100), Raphael iGPU (gfx1036), ROCm hiprtc, with the opacity barrier | **0 / 63488**, all 20 emittable shapes | n/a | executed + machine-code | **strict**, unchanged |
| **OpenCL / rusticl** | RX 7900 XTX + Raphael iGPU, Mesa 26.1.4-arch3.1 | **620 / 63488** on `f16(f16(x*1.1)+1000)`; **2912 / 63488** on `f16(x*1.1)` (newly swept, slice 1) | **YES — bit-identical on 63488 / 63488, on BOTH shapes and BOTH devices.** Green control (`volatile __local` round-trip) reproduces `S_strict` exactly; positive control reproduces the fused model exactly | executed, **element-wise** over the whole finite binary16 domain (`fp-contraction-policy.md` §12.3) | **ADMITTED under Regime A** on the two swept shapes; the other 18 shapes are `Unknown` and stay refused |
| **OpenCL / pocl (x86), Intel IGC, Intel oneAPI CPU** | AMD EPYC 7763 (CI), Intel Arc Graphics (MTL), Core Ultra 9 185H | **0 / 63488** | n/a | executed | **strict**; refusal is currently over-broad here |
| **Vulkan / RADV** | RX 7900 XTX (NAVI31) + Raphael iGPU (RAPHAEL_MENDOCINO), Mesa 26.1.4-arch3.1 | **2912 / 63488** on `f16(x*1.1)` (plain and `precise` alike); **5075** plain / **4776** with `precise` on `f16(f16(x*1.1)+1000)` | **YES, each count is a different named model, all bit-identical on 63488 / 63488 on both devices.** `f16(x*1.1)` → `S_fuse_mul_into_narrowing`. `f16(f16(x*1.1)+1000)` plain → `S_absorb_all_into_final_narrowing`; with `precise` → `S_f32_mul_then_absorb_add`, which is **the model the shipped codegen runs under**. `precise` is honoured, not ignored: it forbids a multiply-into-add contraction and cannot reach a conversion absorbing its operand (§9.3) | executed **element-wise**, plus **machine-code** for the reconciliation (`fp-contraction-policy.md` §12.4) | **ADMITTED under Regime A** on the two swept shapes, each with its own named model; the other 18 shapes are `Unknown` and stay refused |
| **Vulkan / ANV** | Intel Arc Graphics (MTL), Mesa 26.1.2-arch3.1 | **0 / 63488** | n/a | executed | **strict**; refusal is currently over-broad here |
| **Metal** | Apple M4, macOS 15.6.1 (24G90), Apple clang 17.0.0 (clang-1700.0.13.5), Metal.framework from the Command Line Tools SDK | **0 / 63488** on `f16(x*1.1)` **and 0 / 63488** on `f16(f16(x*1.1)+1000)`; unchanged under `#pragma METAL fp contract(off)`, `#pragma clang fp contract(off)`, a `volatile thread` barrier and an `as_type` bitcast barrier | n/a — no deviation to model. The `fusedctl` control, on the same source, compile options and dispatch, reproduces `S_fuse_mul_into_narrowing` on **63488 / 63488** and reports 2912 / 620 | executed, **element-wise** over the whole finite binary16 domain | **strict**; refusal is currently over-broad here |
| **WGSL / naga** | — | **never probed** | — | none | **`Unknown` → refused** |
| **PTX** | — | refuses kernel-level f16 by design (backlog-57 slice 2) | — | by-construction | nothing to relax |

**Regime B rows — cooperative matrix (§5), Metal only.** These are the first
numeric measurements of any cooperative-matrix implementation in this project;
§4's Vulkan row records availability and explicitly says nothing about what RADV
*computes*. Instrument: `tools/probes/metal_simdgroup_matrix_probe.m`,
`fp-contraction-policy.md` §10.15. `D = A×B + C` with `C` nonzero throughout, so
the constant is `γ_8` and not the `C = 0` degenerate case; §5.3's exactness
invariant is asserted by the harness (21 binades, 50 bits, binary64 has 53); both
of §5.4's controls fire.

| configuration | device | within §5's bound? | closed-form model? | evidence | regime |
|---|---|---|---|---|---|
| `simdgroup_half8x8 × half8x8 → float8x8`, 8×8×8 | Apple M4, as above | **0 / 65536 outside**; worst 2.67e-07 against `γ_8` = 4.7684e-07 | **YES — bit-equal on 65536 / 65536** to *"initialise the accumulator to `C`, then add the eight products in index order, all in binary32"* | executed, element-wise | **A** (migrated from B — see below) |
| `simdgroup_half8x8 × half8x8 → half8x8`, 8×8×8 | Apple M4, as above | **0 / 65536 outside**; worst 2.12e-03 against `γ_8` = 3.9216e-03 | **no** — 65520 / 65536 against sequential binary16; the 16 stragglers match neither pairwise binary16 nor a binary32 chain narrowed at the end | executed, element-wise | **B**; §5.4's "do not admit without a separate argued case" now rests on a measurement |

> **This is the first configuration to migrate B → A, and that matters more than
> the row.** §1.6's last table row says a Regime B configuration moves to Regime A
> *"if a given implementation's accumulation order is ever pinned to a closed
> form"*, and §6.1's corollary says friction falls with it, "with no new decision
> needed". That was written as a mechanism nobody had exercised. It has now been
> exercised: Metal's f16×f16→f32 `simdgroup_matrix` MulAdd is not merely inside
> the bound, it **is** a named closed-form function of its inputs, agreed
> bit-for-bit on every one of 65536 elements. Under §1.6 it is therefore Regime A,
> and under §6.1 the DSL author gets a **loud one-time diagnostic rather than a
> mandatory `accept_relaxed_f16` opt-in**. No decision was needed to move it; the
> measurement moved it. **The migration path is real rather than theoretical**,
> which is the part that makes the regime split credible rather than a way of
> having two rules.
>
> Scope, stated so the row is not over-read: one implementation, one tile size
> (8×8×8, the only one Metal offers), `C` drawn from one distribution, and
> **nothing here constrains RADV**, whose coopmat numerics remain entirely
> unmeasured. A Regime A verdict is per *(implementation, configuration)*, exactly
> as §1.2's model set is per *(backend, driver, shape)*.
>
> One methodological note, because it is the reason the row says "C FIRST".
> An earlier revision of the probe pinned `C = 0` and reported 65536/65536
> against "sequential binary32". That claim was *underdetermined*, not wrong:
> with `C = 0` the C-first and C-last orders are the same function. A nonzero `C`
> separates them decisively — 65536 against 51850 — so **the closed form could
> not have been identified at all without §5.1's insistence that the operation is
> `A×B + C`.** §5.1 records that correction as having been made on paper; this is
> it reproduced from the measurement side.

Three things the Regime A table says that are easy to miss:

1. **The relaxation is admitted on exactly two stacks** — rusticl and RADV,
   both ACO — **and on two expression shapes**. Everything else either already
   meets `S_strict` or has never been measured. This is a much narrower change
   than "relax f16": it is four (stack, shape) pairs, each with a named function
   attached, and 18 shapes per stack still refused. **Metal joined the strict
   group rather than the candidate group**, which is the outcome that needed no
   relaxation at all.
2. **The RADV two-narrowing shape WAS the open risk, and it is closed.** 5075
   plain against 4776 with `precise` is two different named models, each matched
   bit-for-bit on 63488/63488, and the ISA says why: `NoContraction` forbids
   contracting a multiply into an **addition** and cannot reach a **conversion**
   absorbing its operand. The one-narrowing shape contains no addition, so the
   decoration binds nothing and the ISA is byte-identical; the two-narrowing
   shape contains one, so it binds, the multiply survives as its own
   `v_fma_mix_f32`, and the model changes. Both facts, one mechanism. §9.3
   records the resolution.
3. **pocl, IGC, ANV, the Intel CPU runtime and now Metal are refused today for a
   defect they do not have.** `capability-model.md` §5 already records this as the
   over-refusal that follows from having no compiler-identity probe. It is not
   made worse by this design, and slice 2's `VkPhysicalDeviceDriverProperties`
   plumbing narrows the Vulkan half of it. Metal is the clearest case: it is
   measured strict **element-wise** on both shapes, which is a stronger evidence
   tier than any other row in the table carries, and it is still refused.

---

## 3. Why the relaxation is worth it — and why it is not uniform

> **DECIDED (2026-07-27).** Take the relaxation for scalar f16 as well as for
> cooperative matrix — and record the scalar half explicitly as a **performance
> trade**, because that is what it is. The exact mechanism is kept documented and
> available; it is not being taken away, it is being declined by default on cost
> grounds. The reasoning is below and is deliberately preserved, because the two
> halves of the relaxation are justified by two different arguments and a later
> reader must be able to tell them apart.

**The honest sentence.** This is a **weaker guarantee than Sarek held before.**
Before, an f16 result on a supported backend was bit-identical to the
interpreter, and the interpreter is the definition of what a Sarek program means;
after, on driver stacks that pass slice 1, it is bit-identical to a *named
alternative rounding* of the same program (Regime A), or inside a derived bound
(Regime B). A user who diffs two devices bit-for-bit will now see differences on
those stacks, legitimately. That is the price.

What it buys is that the tensor-core paths become reachable at all. Neither
`VK_KHR_cooperative_matrix` (backlog-62) nor Metal `simdgroup_matrix` (backlog-63) can be
expressed without f16 as a DSL element type on those backends
(`f16-dsl-element-type.md` §1), and both are the funded direction (ibid. §10 Q2).

**But the two halves are not blocked for the same reason, and this matters.**

- **Cooperative-matrix accumulation cannot be bit-reproduced against any fixed
  reference, by specification.** The order of operations is
  implementation-dependent (§1.4). There is no barrier, no flag and no source
  formulation that recovers it. Here the relaxation is the *only* route.
- **Scalar f16 on RADV and rusticl CAN be bit-reproduced, and the mechanism is
  measured.** `fp-contraction-policy.md` §2 records a working barrier for each:
  forcing the f16 **bit pattern** through a global-memory round-trip gives
  **0 / 63488** — on rusticl/radeonsi (`volatile __global` round-trip) and on
  RADV (the f16 bit pattern through global memory), on the RX 7900 XTX and the
  Raphael iGPU, Mesa 26.1.4-arch3.1. It is exact. It was rejected on **cost** — a
  memory round-trip per narrowing, into a scratch buffer neither backend
  currently owns — and not on impossibility.

### 3.1 The scalar decision, stated plainly

**Sarek declines an available exact mechanism for scalar f16 on ACO stacks, and
accepts a characterised one-rounding deviation instead, in exchange for
throughput.** That is a performance choice. It is not the same kind of statement
as the coopmat one, where no mechanism exists at any price.

The trade is taken because the scalar type is a prerequisite for the matrix path,
and a global-memory round-trip **per narrowing, inside a matmul inner loop**,
would consume exactly the bandwidth the tensor-core path is being built to save.
Paying it there would make the unblock self-defeating.

**The exact route stays documented and reachable.** If a user's requirement is
bit-reproducibility across stacks rather than throughput, the mechanism exists,
has been measured at 0/63488 on both ACO stacks, and is written down in
`fp-contraction-policy.md` §2 (the `OpenCL / rusticl (f16 narrowing)` and
`Vulkan / RADV (f16 narrowing)` rows). Nothing in this design deletes it or makes
it harder to reintroduce; a future opt-in barrier mode is a small, self-contained
change on top of the slice-3 codegen. It is declined by default, not removed.

**Consequences to keep in view.** Because this half is a cost decision and not a
physics one, it is the half that should be **revisited when the cost changes** —
a Mesa release that stops fusing (the existing tripwires already go red on
exactly that), a cheaper barrier shape, or a workload whose narrowings are not in
an inner loop. The coopmat half will never be revisited on those grounds, because
no measurement can move a specification.

---

## 4. Cooperative-matrix availability on the local device — measured

**Executed 2026-07-27** on this workstation via
`tools/probes/vulkan_coopmat_probe.c` (committed with this document; it creates a
Vulkan instance, queries, and destroys it — no device, no shader). Evidence
tier: **executed**.

**`VK_KHR_cooperative_matrix` IS advertised locally**, on the discrete GPU only:

| | AMD Radeon RX 7900 XTX (RADV NAVI31) | AMD Ryzen 9 7950X (RADV RAPHAEL_MENDOCINO) |
|---|---|---|
| driver | radv, Mesa 26.1.4-arch3.1, Vulkan 1.4.354 | radv, Mesa 26.1.4-arch3.1, Vulkan 1.4.354 |
| `VK_KHR_cooperative_matrix` | **YES** (extension revision 2) | **no** |
| `cooperativeMatrix` feature | **true** | false |
| `cooperativeMatrixRobustBufferAccess` | true | — |
| `cooperativeMatrixSupportedStages` | `SHADER_STAGE_COMPUTE_BIT` only | — |
| `VK_KHR_shader_float16_int8` / `shaderFloat16` | YES / **true** | YES / **true** |
| `VK_KHR_16bit_storage` / `storageBuffer16BitAccess` | YES / **true** | YES / **true** |

All 14 advertised configurations on NAVI31 are **M=16, N=16, K=16**, **`scope =
subgroup`**:

| A | B | C | Result | saturating | count |
|---|---|---|---|---|---|
| u8/s8 (all four sign combinations) | u8/s8 | u32 | u32 | no | 4 |
| u8/s8 (all four) | u8/s8 | s32 | s32 | no | 4 |
| u8/s8 (all four) | u8/s8 | s32 | s32 | **yes** | 4 |
| **f16** | **f16** | **f16** | **f16** | no | 1 |
| **f16** | **f16** | **f32** | **f32** | no | 1 |

Four consequences for the plan:

1. **backlog-62 is not blocked on hardware.** The path can be built and measured
   locally, end to end.
2. **Only 2 of 14 configurations need the relaxed contract.** The other 12 are
   integer, and the SPIR-V extension states integer accumulation is exact — they
   are deliverable under the *existing* strict contract. See §8.
3. **`shaderFloat16` is available and Sarek does not enable it.**
   `fp-contraction-policy.md` §7(b) flags this as real unplumbed work;
   `sarek-vulkan/Vulkan_api_device.ml` chains no feature structs beyond core
   `VkPhysicalDeviceFeatures`. The probe confirms the feature is *there* to be
   enabled, so it is a plumbing slice and not a hardware question.
4. **The iGPU is a free negative device.** Coopmat is `Device_optional` in
   `capability-model.md`'s taxonomy, and this box has one device that has it and
   one that does not, in the same driver — so the capability gate can be tested
   for both outcomes without any second machine.

**Not measured, and not to be inferred:** nothing here says what RADV's coopmat
MulAdd *computes*. The probe queries availability. The numeric contract of §5 is
unmeasured on every implementation.

---

## 5. The coopmat numeric contract (proposed, §7 slice 4a measures it)

**This is the second acceptance regime of §1.6, not an instance of §1.2's model
set.** `S_strict` and `S_fuse_mul_into_narrowing` are the wrong instruments for a
16×16×16 MulAdd, and not because nobody has written the models down: the freedom
the specification grants is over the *order* of a 17-term summation, which admits
combinatorially many distinct correct results. There is no finite set to enumerate,
so this regime is a derived numeric bound. §1.6 is where the two regimes are set
against each other and why neither is allowed to be called "the contract"
unqualified.

### 5.1 What the operation actually computes

`OpCooperativeMatrixMulAddKHR` computes **`D = A × B + C`**, not `A × B`. Each
output element is

```
D[m][n]  =  ( Σ_{k=0..K-1} A[m][k] · B[k][n] )  +  C[m][n]
```

so for the f16×f16→f32 configuration one output element is a sum of **K products
plus one f32 addend — K + 1 = 17 terms, requiring 16 additions**, in an order the
specification leaves to the implementation. An earlier draft of this section
bounded only the products and compared against the dot product alone; that was
wrong, and a correct result with a nonzero `C` would have failed the gate.

### 5.2 The derivation

- **An f16 × f16 product is exact in binary32.** Two 11-bit significands multiply
  to at most 22 bits, which fits binary32's 24, and the product's exponent range
  is inside binary32's. Evidence tier: **by-construction**. The products
  contribute *no* error at all, whatever order they are formed in.
- **`C[m][n]` is a binary32 value supplied by the caller, so it is exact as
  given.** It contributes no error of its own — but it *does* enter the
  accumulation, so it is one more term to be summed and one more magnitude in the
  denominator.
- **The only freedom left is the order of the 16 binary32 additions.** The
  classical bound for summing `n` exactly-representable terms in *any* order,
  using `n − 1` additions, is `|error| ≤ γ_{n−1} · Σ|terms|` with
  `γ_j = j·u / (1 − j·u)` and `u = 2^-24`. Here `n = K + 1 = 17`, so `j = 16` and
  `γ_16 = 16·2^-24 / (1 − 16·2^-24) = 9.5368e-07`.
- **Stated against `Σ|terms|`, never against the result.** A cancelling dot
  product has unbounded *relative-to-result* error under any summation order, so a
  relative-to-result bound would be a number that cannot fail. The denominator is
  `Σ_k |A[m][k]·B[k][n]| + |C[m][n]|`. **Re-checked against §1.3's correction and
  unaffected:** the defect that forced §1.3 to move off the final value is
  cancellation re-scaling the unit of measure, and this bound already refuses to
  measure against the result for the same underlying reason. Regime B needed no
  change; Regime A did.
- **The bound assumes the implementation's intermediate accumulation is at
  least binary32.** If it is wider, the true error is smaller and the bound still
  holds. If it is *narrower* — an implementation accumulating the f32-result
  configuration in binary16 — the bound does not hold and the gate fails, which
  is precisely the defect §5.4's positive control exists to catch.

> **Proposed contract, f16×f16→f32 coopmat, `D = A × B + C`.** For every output
> element `(m, n)` of a 16×16×16 `OpCooperativeMatrixMulAddKHR`:
>
> ```
> | D_device[m][n] − D_exact[m][n] |  ≤  9.5368e-07 · ( Σ_k |A[m][k]·B[k][n]| + |C[m][n]| )
> ```
>
> where `D_exact` is the **exactly** evaluated `Σ_k A[m][k]·B[k][n] + C[m][n]`.
>
> **Degenerate case `C = 0`:** 16 terms, 15 additions, and the bound tightens to
> `γ_15 = 8.9407e-07 · Σ_k |A[m][k]·B[k][n]|`. The harness should use the tighter
> constant when it has pinned `C = 0`, and must not use it otherwise.

### 5.3 Computing `D_exact` — binary64 is NOT sufficient in general

A second correction to the earlier draft, which asserted that binary64 makes the
reference exact for K = 16. It does not, and the condition is worth stating
because the harness must **assert** it rather than assume it.

Each of the 17 terms is an integer multiple of `2^(e_i − 23)`, where `e_i` is its
binade exponent, and the sum's magnitude is at most `17 · 2^(max e_i + 1)`. So an
exact representation needs

```
(max e − min e) + 24 + 1 + ceil(log2 17)  =  span + 30   bits
```

and binary64 has 53. **Binary64 is exact only when the exponent span of the 17
terms is at most 23 binades.** That is not a formality: f16 inputs range from
subnormals near `2^-24` to `65504 ≈ 2^16`, so their products can span roughly 80
binades — far outside what binary64 can sum exactly.

Two acceptable implementations, and the harness must do one of them explicitly:

- **Restrict the input generator** so the term exponent span is ≤ 23 binades, and
  **assert that invariant in the harness**, failing loudly if a generated case
  violates it. Then binary64 is exact and the reference is cheap.
- **Use an exact accumulator** — exact rational arithmetic, or a fixed-point
  superaccumulator wide enough for binary32's full exponent range — and lift the
  input restriction.

The first is recommended for slice 4a and the restriction is a real narrowing of
coverage that must be recorded alongside the result, not buried in the generator.

### 5.4 Why f32-accumulate first, and the required control

The same derivation does **not** carry over unchanged to the f16×f16→**f16**
configuration, and the reason is worse than a larger constant: with an f16 result
type the products are no longer exact in the accumulation format (22 significand
bits into 11), so the error has a product term as well as an ordering term. Even
ignoring that, the ordering term alone is `γ_16` at `u = 2^-11`, i.e.
**≈ 7.87e-03 · Σ|terms|** — four orders of magnitude looser than the f32 case and
wide enough to hide almost any implementation defect.

**Recommendation: admit the f32-accumulate configuration first, and do not admit
the f16-accumulate one without a separate argued case and its own derivation.**
This is the §11.5 lesson applied before rather than after.

**Two positive controls are required**, without which the bound is a gate that
cannot fail. Each is a deliberately wrong reference that the gate must *reject*:

1. **A binary16-accumulating reference** — catches an implementation that
   accumulates the f32-result configuration at f16, the defect §5.2's last bullet
   names.
2. **A `C`-dropping reference** — catches exactly the error this section was
   written to fix, so that §5.1's correction is pinned by a test rather than only
   by a paragraph. It fires only when the gate is exercised with a nonzero `C`,
   which is therefore mandatory in the input generator.

Both are a few lines each, and they are the difference between a measurement and
a formality.

---

## 6. How a DSL author finds out

> **DECIDED (2026-07-27).** **User-facing friction is proportional to the
> strength of the evidence for the deviation.** Where the deviation is exactly
> characterised — a named closed-form model, agreed with bit-for-bit over an
> exhaustive sweep — the author gets a loud one-time diagnostic and the code
> runs. Where the deviation is only *bounded* — the coopmat case, where the
> specification grants a freedom no reference can pin down — the author must
> explicitly opt in, and the launch fails without it. The rule and its derivation
> are §6.1.

Silence is not an option; it is the failure mode this repository's recent history
is made of. The mechanism is `capability-model.md`'s, used as designed and
without extending it.

**Kinds.** The relaxed path is `Toolchain_semantic` (the evidence: ACO's combine,
with the measured counts of §2) composed with `Policy` (the verdict: admitted, on
this allowlist). `capability-model.md` §5 already lists the current refusals as
"expressible, not yet wired" in exactly this pair; this design turns the verdict
from *refuse* to *admit-on-allowlist* and leaves the kinds alone.

**Verdict algebra unchanged, and this is the safety property.** No fourth
constructor — `capability-model.md` §3 is right that adding one should be a
compile error at the deciding site, and there is no need. Instead:

- `Available` — the (driver, driver version, shape class) triple is on the
  measured allowlist and the exhaustive sweep is green in CI or on a
  workstation gate. Strict semantics.
- `Available` **carrying a relaxation record** — same, but the admitted set has
  more than one member. The record names the model and cites the row of §2.
- `Unknown _` — **everything else, including every driver nobody has swept.**
  Does not permit. Today's refusal is the default and stays the default; a new
  Mesa release does not silently start emitting relaxed f16.

### 6.1 The friction rule, and why it is shaped this way

**The rule.** *How much the user is made to say yes is proportional to how poorly
we know the deviation.*

| evidence for the deviation | what the author gets |
|---|---|
| exact match to a named closed-form model, swept exhaustively (on ACO: `S_fuse_mul_into_narrowing`, `S_absorb_all_into_final_narrowing` and `S_f32_mul_then_absorb_add`, all confirmed element-wise by slice 1 on 2026-07-27) | a **one-time runtime diagnostic** on first launch of an f16 kernel on that device, naming the model and the doc section, on by default; silenceable only by an explicit call, never by accident |
| a *bounded* deviation with no closed form (the coopmat case, §5) | **explicit opt-in required** — `Sarek.accept_relaxed_f16 ~reason` or equivalent, and the launch fails without it |
| unmeasured | refused, as today |

**The derivation, kept because someone will want to re-derive it.** The owner's
ruling asks for four things, and one of them — *"la limitation technique est
comprise"* — is a property of the **user**, not of the code. Nothing in a test
suite can establish it. The only instrument that can is making the user say so.
So the question is not *whether* to ask, but *when* asking earns its cost.

Three constraints pin the answer:

1. **An opt-in is the only evidence of understanding we can collect.** A
   diagnostic evidences that we *told* them; a required call evidences that they
   *read* it. Those are different facts and only the second discharges the
   ruling's third clause.
2. **Mandatory everywhere defeats the unblock.** If every f16 kernel needs an
   acknowledgement before it will run, the tensor-core path is unusable by
   default, and backlog-62/backlog-63 were unblocked in order to be used.
3. **Mandatory nowhere makes the coopmat bound meaningless.** §5's bound is a
   *bound*, not an identity: a user can be inside it and still be getting an
   answer that depends on the driver's accumulation order. Someone who has not
   understood that will read a passing run as a reproducible one.

Constraints 2 and 3 pull in opposite directions, so the split has to fall
somewhere; the strength of the evidence is the right place for it to fall,
because it is exactly the axis along which the *risk of misreading a passing run*
varies. Where the deviation matches a closed-form model bit-for-bit over the
whole finite domain, a user who ignores the diagnostic still gets a result that
is a *specific, documented, deterministic* function of their input — the worst
case is a surprise, not a silent wrongness. Where the deviation is only bounded,
the same ignored diagnostic leaves them believing something that is not true. The
friction is spent where the failure mode is worse.

**Corollary, and it is the useful part of the rule.** Friction is not a fixed
property of a backend — it *falls* as evidence improves. If slice 4a finds that a
given implementation's coopmat MulAdd matches a closed-form accumulation order
exactly, that configuration moves from the opt-in row to the diagnostic row, with
no new decision needed. The rule is written to make that a measurement outcome
rather than a negotiation.

**Also required, and cheap:** the generated device source carries a header
comment naming the semantics in force, so anyone reading a dumped kernel sees it;
and the `Sarek_capability` verdict is queryable before launch so a program can
branch on it rather than discovering it in a diagnostic.

---

## 7. Slicing plan for backlog-62 and backlog-63

Each slice names what it proves and what hardware it needs. Nothing below lifts a
refusal before slice 3.

| # | slice | proves | hardware | lifts a refusal? | status |
|---|---|---|---|---|---|
| **0** | the contract, as a testable classifier | the gate can tell `S_fuse_mul_into_narrowing` from a dropped narrowing | none (host-only) | no | open — and now also carries §1.3's per-narrowing ceiling |
| **1** | element-wise model characterisation of ACO scalar f16 | whether the contract of §1.2 is deliverable at all | RX 7900 XTX + Raphael iGPU (local) | no | **DONE — all 20 shapes measured (2 in slice 1, the other 18 in backlog-151). Deliverable, and §1.2's set is generated rather than enumerated. 18 shapes measured but NOT admitted.** |
| **2** | `shaderFloat16` + driver-identity + capability plumbing | the gate can be keyed on a driver, not a device name | local, both devices | no | **DONE 2026-07-27** — plus the coopmat capability query and the fragment type of slice 4b |
| **3** | GLSL scalar f16, allowlisted | Sarek-*generated* f16 shaders meet the contract | RX 7900 XTX (local) | **yes**, on an allowlist | open |
| **4** | backlog-62 Vulkan coopmat, f16×f16→f32 | the tensor-core path, end to end | RX 7900 XTX (local) | yes (new capability) | **4a/4c DONE for the INTEGER configurations, 2026-07-27 — see below. The f16 half is untouched and 4a's numeric contract is still unmeasured.** |
| **5** | backlog-63 Metal — **scalar f16 first** | the Metal row of §2 | Apple M4 | no | **DONE 2026-07-27 — strict, 0/63488** |
| **6** | backlog-63 Metal `simdgroup_matrix` | the Apple tensor-core path | Apple M4 | yes | **DONE 2026-07-27 — availability + numerics; no integer path** |

### Slice 0 — make the contract fail-able (host-only, lands first)

Promote `ref_discipline` / `ref_fused` out of
`sarek-vulkan/test/test_vulkan_f16_tripwire.ml` and its OpenCL port into a shared
module, and add to `Test_helpers`, mirroring the df64 machinery one-for-one:
`f16_admissible_models`, `f16_known_relaxation` (the allowlist, keyed on
framework × driver predicate × shape class), `f16_relaxed_ceiling`, and
`classify_f16_result` returning `` `Pass | `Xpass | `Known_relaxation | `Fail ``
with **strict XPASS** — a driver that stops deviating turns the run red and names
the arm to delete, exactly as `classify_df64_result` does and for the reason
recorded there.

Calibration, running with no GPU: the two host models must separate on exactly
**620** over the finite binary16 domain — the figure independently reproduced on
hiprtc/gfx1100, rusticl/radeonsi, `fusedctl` on Intel Arc, and now by the Metal
probe's round-to-odd control on the M4 (`fp-contraction-policy.md` §10.14), which
is four unrelated stacks. `test_opencl_f16_tripwire` already does this after
§11.5 and it is the model to copy.

**`f16_relaxed_ceiling` is per-narrowing, not per-result (§1.3), and slice 0 is
where that is built rather than retrofitted.** The models must expose their
intermediate binary16 values so the ceiling can be evaluated where the rounding
was elided; a `classify_f16_result` that compares only final values cannot
implement §1.3 as corrected and will reject correct results on any shape with a
cancelling operation after a narrowing. Add a host-only calibration for exactly
that: the shape `f16(f16(x*1.1)+1000)` at **`x = -907.5`**, where the two models
differ by **1 ulp at the narrowing** and by **512 ulp on the final value** — the
case that broke the first formulation. It must classify as `Known_relaxation`,
and a ceiling evaluated on the final value must be shown to *fail* it, or the
correction is not actually in force.

**Proves, and this is the whole point of the slice:** feed the classifier the
IGC dropped-narrowing signature (4774/63488, first divergence `x = 0.681640625`)
and require **FAIL**; feed it the ACO signature and require `Known_relaxation`.
A gate that cannot separate those two is a gate that admits a defect, and §1.1 is
the argument that no ulp band can.

### Slice 1 — is the contract actually deliverable? (local, decision point)

Today's measurements cover **two** shapes. The HIP f16 shape audit
(`docs/optimization/amdgpu-f16-fusion-shape-audit.md`) enumerates **all 20**
Sarek-emittable f16 expression shapes and sweeps each over all 63488 finite
binary16 inputs. Run that catalogue against `S_strict` and
`S_fuse_mul_into_narrowing` **element-wise**, on rusticl/radeonsi and on RADV,
on the RX 7900 XTX and the Raphael iGPU.

Two outcomes, and they are not equally good:

- **Every shape matches a model exactly.** The contract of §1.2 is deliverable as
  written. Proceed.
- **Some shape matches no model.** Most likely candidate is the RADV
  two-narrowing shape, where 5075 plain against 4776 with `precise` already
  suggests two behaviours rather than one (§2 note 2). Then that shape stays
  refused, the contract becomes per-shape, and **the owner is told before slices
  2–4 are funded** — a per-shape contract is a materially worse thing to document
  and to explain to a user than a per-backend one. This is the open risk of the
  whole design and it is written up as such in **§9.3**, together with the
  mitigation (§7 slice 4b's integer component types, which put an
  existing-strict-contract coopmat path within reach regardless).

> **OUTCOME, 2026-07-27 — the first outcome, on the shapes swept.** Executed on
> both local devices for each stack: `AMD Radeon RX 7900 XTX` and the Raphael
> iGPU, under rusticl/radeonsi and under RADV, Mesa 26.1.4-arch3.1, Vulkan
> 1.4.354. Full record in `fp-contraction-policy.md` §12; instruments are
> `sarek-opencl/probe/probe_opencl_f16_model_agreement.ml`,
> `sarek-vulkan/probe/probe_vulkan_f16_model_agreement.ml` and the shared
> `tools/f16_model_set/`.
>
> | stack, shape, variant | model | agreement |
> |---|---|---|
> | rusticl, `f16(x*1.1)` | `S_fuse_mul_into_narrowing` | 63488 / 63488 |
> | rusticl, `f16(f16(x*1.1)+1000)` | `S_fuse_mul_into_narrowing` | 63488 / 63488 |
> | RADV, `f16(x*1.1)`, plain and `precise` | `S_fuse_mul_into_narrowing` | 63488 / 63488 |
> | RADV, `f16(f16(x*1.1)+1000)`, plain | `S_absorb_all_into_final_narrowing` | 63488 / 63488 |
> | RADV, `f16(f16(x*1.1)+1000)`, `precise` | `S_f32_mul_then_absorb_add` | 63488 / 63488 |
>
> **Zero inputs matched no model, on either stack, on either device.** The
> ceiling of §1.3, evaluated at the elided narrowing, is met everywhere with
> zero exceedances — and the same deviations measure 512 to 1.7e6 ulp on the
> final value, so §1.3's correction is reproduced on 63488 inputs rather than
> resting on one Metal counterexample.
>
> **Two things this slice found that were not on its list.** (i) The
> `double`-based `fusedctl` control does not build on rusticl, which advertises
> no `cl_khr_fp64`; the control had to be rebuilt on an f32 pair with a
> round-to-odd step, the same construction MSL forced on the Metal probe. (ii)
> ACO **defeats** the obvious two-narrowing positive control — a round-to-odd
> product narrowed and then added to 1000 is re-absorbed, landing on
> `S_absorb_all_into_final_narrowing` instead of the fused model. The working
> control has to push the f16 bit pattern through the SSBO as well. A control
> that a compiler can optimise away is not a control, and this one was caught
> only because it reported the wrong model rather than a plausible number.
>
> **What is NOT settled: 18 of the 20 shapes.** The catalogue in
> `docs/optimization/amdgpu-f16-fusion-shape-audit.md` has 20 emittable f16
> shapes and this swept two. Slice 3 must not lift a refusal beyond the shapes a
> model has been measured for. There is a single candidate generative rule that
> produces all five results above — *a narrowing absorbs the whole f32 tree
> feeding it, cut where `NoContraction` forbids a multiply-add* — but its
> evidence tier is **unverified as a general rule**, and the remaining 18 shapes
> are exactly what would confirm or break it (`fp-contraction-policy.md` §12.4).

> **SETTLED, 2026-07-27 (backlog-151) — and the rule was WRONG.** All 18 were
> swept, on RADV and rusticl, on all four local devices, element-wise over the
> whole finite binary16 domain. Full record in `fp-contraction-policy.md` §13;
> raw output in `docs/measurements/f16-shapes-2026-07-27/`.
>
> - **The rule is false.** Absorption is a *local* peephole, and the
>   disassembly is why: `v_fma_mixlo_f16` takes one multiply-add and one
>   conversion, so it cannot reach past anything else. `f16(floor(x*1.1))` and
>   `f16(x>0 ? x*1.1 : x*0.9)` come back **`S_strict`** where the rule predicts
>   absorption, with the intervening `v_floor_f32` / `v_cndmask_b32` visible in
>   the ISA. `f16(f16(f16(x*1.1)+1000)*1.1)` is worse: it matches **no single
>   member of the model set at all** — 63480/63488 against one member and
>   63486/63488 against another — because ACO performs *two* single-rounding
>   events at *two* precisions, contracting `x*1.1+1000` into one binary32 fma
>   and only then absorbing the final multiply.
> - **The corrected rule is local, and it holds** on 12 of 12 discriminating
>   shapes on RADV (plain and `precise`) and 11 of 12 on rusticl, verified on
>   ten shapes it was not fitted to. §13.4 states it; §1.2 carries it.
> - **The model set does NOT grow per shape** — the outcome this slice was a
>   decision point for. It is `{S_strict}` plus one generator with three
>   measured booleans, for any shape a user writes. §13.5.
> - **Only 12 of the 20 shapes can discriminate at all.** On the other eight
>   every model is the same function, so a device sweep of them measures
>   nothing; four of those eight are the shapes the HIP audit already recorded
>   as *demoted in the machine code and clean in the numbers*. §13.2.
> - **One residual and one defect.** rusticl mispredicts on
>   `f16(f16(x*1.1)*1.1)`, where both stacks constant-fold `1.1*1.1` into
>   binary32 `1.21` **across** the intermediate narrowing — a reassociation
>   combine, not an absorption one, identified by the literal `0x3f9ae148` in
>   the ISA and not modelled here (§13.4). And RADV returns `-0` for `0.0 - x`
>   at `x = +0`, through every barrier, which is a §1.2 **failure** rather than
>   a relaxation (§13.6).
>
> **No refusal is lifted by this.** Slice 3 is still where that happens, and it
> now has a rule to gate on rather than four measured points.

**This slice reports its outcome upward before the next one starts.** It is a
decision point, not a task on a list; a green result funds slices 2–4 and a red
one reopens the contract.

Also upgrade the rusticl row of §2 from count-and-first-divergence agreement to
element-wise agreement, which is what §1.2 actually requires and which nobody has
run. **Done** — and the one-narrowing shape, never previously swept on rusticl,
was added: 2912/63488, the same figure and the same model as RADV.

### Slice 2 — plumbing (local, no refusal touched)

- `Vulkan_api_device` chains `VkPhysicalDeviceShaderFloat16Int8Features` and
  `VkPhysicalDevice16BitStorageFeatures` at device creation. Note the
  measurement in `fp-contraction-policy.md` §7(b): RADV accepts f16 shaders today
  *without* the feature enabled, so the current tripwire runs on an un-enabled
  path. That is fine for a driver measurement and is not fine for shipping.
- Plumb `VkPhysicalDeviceDriverProperties` onto `Device.t` so `driverID` /
  `driverName` are available. `fp-contraction-policy.md` §11.7 and the
  `is_anv_device` comment in `Test_helpers` both already ask for this, and the
  f16 allowlist needs a driver key rather than a device-name substring — the ANV
  predicate today would match a future non-Mesa Intel driver.
- `Framework_sig.capabilities` gains `supports_fp16` and a coopmat configuration
  list. This is `capability-model.md`'s slice 2 and it breaks the literal-record
  tests in `spoc/framework/test/`, a cost that document already accepts.

> **OUTCOME, 2026-07-27 — done, and it moved two facts this document asserts.**
> Executed on both local devices (AMD Radeon RX 7900 XTX / RADV NAVI31 and the
> AMD Ryzen 9 7950X iGPU / RADV RAPHAEL_MENDOCINO), radv, Mesa 26.1.4-arch3.1,
> Vulkan 1.4.354. Instruments: `sarek-vulkan/test/test_vulkan_coopmat_capability.ml`
> (device side) and `spoc/ir/test/test_sarek_coopmat.ml` (host side).
>
> Delivered: `VkPhysicalDeviceShaderFloat16Int8Features` and
> `VkPhysicalDevice16BitStorageFeatures` are queried through the
> `VkPhysicalDeviceFeatures2` chain and **requested** in
> `VkDeviceCreateInfo.pNext`, alongside `VK_KHR_cooperative_matrix` where
> advertised; `VkPhysicalDeviceDriverProperties` and
> `VkPhysicalDeviceSubgroupProperties` are on `Device.t`;
> `Framework_sig.capabilities` gains `coopmat : Sarek_coopmat.device_support
> option`; and `Sarek_coopmat` (in `spoc/ir`, beside `Sarek_capability`) carries
> the configuration vocabulary and **slice 4b's fragment type**, integer
> component types included.
>
> **Two corrections to this document.**
>
> 1. **The subgroup size is 64, not 32.** `VkPhysicalDeviceSubgroupProperties.
>    subgroupSize` reads **64** on *both* local devices. `Vulkan_plugin_base` had
>    been reporting a hard-coded 32, which was wrong for both. This is ABI, not
>    a statistic: a 16×16 fragment over 64 invocations is 4 components each, not
>    8, so any codegen slice written against 32 would have been wrong at the
>    calling convention. `warp_size` is now the probed value.
>
> 2. **The configuration query answers for devices that do not support the
>    extension, and §4's negative device is only negative if you check the
>    extension.** `vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR` returns
>    `VK_SUCCESS` and **all fourteen configurations for the Raphael iGPU** — the
>    device this document offers as the free negative device, which does not
>    advertise `VK_KHR_cooperative_matrix` and whose `cooperativeMatrix` feature
>    reads false. Calling an extension entry point on a device without the
>    extension is undefined behaviour, and RADV's undefined behaviour here is a
>    well-formed, plausible, entirely wrong answer. §4's claim is correct as
>    written — it was measured through the extension list — but a probe that
>    enumerated configurations first would have contradicted it, and would have
>    made the `Device_optional` gate unable to refuse anything.
>
> **The refusal was observed.** With the extension check in place, the iGPU
> yields `Unavailable` for 16×16×16 `f16×f16→f32` while the RX 7900 XTX yields
> `Available`, same driver, same Mesa, same run. Verified falsifiable by
> mutation: removing the extension check makes the iGPU report all fourteen
> configurations and both devices permit — and the *gate* test stays green,
> because it compares the verdict against the same list the verdict reads. Two
> further checks catch it (`configurations imply the enabled feature`, and a
> no-drop invariant on the advertised count), and both go red on their
> respective mutants.
>
> **A defect found in the doing, worth recording because it looked like
> nothing.** A wrong `VkComponentTypeKHR` enumerant table decoded the fourteen
> advertised configurations into six — including an `f16 × s8 → s32` and a
> `u8 × u8 → u8` that no hardware advertises — with the other eight silently
> dropped as unrepresentable. Every number looked plausible and every test
> passed. The enumerants are now pinned against `vulkan_core.h` by value, and
> `device_support` carries `ds_advertised_count` so that a dropped configuration
> is a test failure rather than a smaller list.
>
> **`shaderFloat16` enablement did NOT change the f16 measurement.** Controlled
> A/B, one build, feature request toggled: `test_vulkan_f16_tripwire` reports
> 2912/63488 on both arms on both devices, same first divergence, all controls
> green. `fp-contraction-policy.md` §7(b) is closed with the table.
>
> **No refusal is lifted.** `Sarek_ir_glsl` still refuses f16; nothing in Sarek
> emits a cooperative-matrix instruction. Slice 3 and slices 4a/4c are untouched.

### Slice 3 — lift the GLSL scalar-f16 refusal, on an allowlist (local)

`Sarek_ir_glsl.reject_float16_kernel` consults the capability verdict instead of
refusing unconditionally. `Available` only for allowlisted (driver, driver
version) pairs; `Unknown` — every other driver, including every future Mesa —
keeps the current refusal.

Gate: the slice-1 sweep re-run on **Sarek-generated** shaders.
`fp-contraction-policy.md` §7(c) is explicit that the existing tripwire compiles
raw GLSL and measures the driver rather than the codegen, and does not substitute
for a codegen gate. The existing tripwire stays, unchanged, as the driver-side
half.

OpenCL/rusticl gets the same treatment or is deliberately left refused —
`Sarek_ir_opencl`'s refusal comment is right that OpenCL has no tensor-core path
to unlock (`opt-expressivity-gaps.md`: "OpenCL has no portable equivalent"), so
lifting it buys nothing but consistency. **Recommend leaving OpenCL refused** and
saying why, rather than lifting it for symmetry.

### Slice 4 — backlog-62, Vulkan cooperative-matrix (local)

> **OUTCOME, 2026-07-27 — the INTEGER path is done end to end, and the order of
> 4a/4b/4c was deliberately inverted.** Executed on RADV, Mesa 26.1.4-arch3.1,
> Vulkan 1.4.354, on an `AMD Radeon RX 7900 XTX (RADV NAVI31)`, with the
> `AMD Ryzen 9 7950X` iGPU (`RADV RAPHAEL_MENDOCINO`) as the negative device.
> Instruments: `sarek-vulkan/test/test_vulkan_coopmat_integer.ml` (driver side,
> hand-written GLSL), `sarek/tests/e2e/test_coopmat_integer_e2e.ml` (codegen
> side, Sarek IR through `Sarek_ir_glsl`), and
> `sarek-vulkan/probe/probe_vulkan_coopmat_configs.ml` (the advertised table).
>
> **Why the inversion.** This slice leads with f16 because f16 scalar is a
> prerequisite for backlog-63 and for bf16 regardless. But 4a's numeric
> contract — the γ₁₆ bound of §5.2 and the two positive controls of §5.4 — is
> entirely unmeasured, whereas `SPV_KHR_cooperative_matrix` states that INTEGER
> accumulation is exact at the precision of the result type. So the integer
> configurations land under Sarek's **existing strict contract** with no
> relaxation, no allowlist, no opt-in and no bound to derive, and they are what
> got a full DSL-to-executed-result path first. §8's argument, delivered.
>
> **The advertised table, measured.** All fourteen are 16×16×16 subgroup scope.
> Twelve integer: `u8×u8`, `u8×s8`, `s8×u8`, `s8×s8`, each with `→u32`, `→s32`
> and `→s32 saturating`. Two float: `f16×f16→f16` and `f16×f16→f32`. **Every
> integer configuration has 8-bit operands with a 32-bit accumulator** — there
> is no wider integer operand type to fall back on.
>
> **Three device features slice 2 did not request, all mandatory.** Read off the
> SPIR-V rather than guessed: `glslc` on a 16×16×16 u8 `coopMatMulAdd` emits
> `OpCapability Int8`, `StorageBuffer8BitAccess`, `VulkanMemoryModel` and
> `CooperativeMatrixKHR`; slice 2 enables only the last. `shaderInt8`,
> `storageBuffer8BitAccess` and `vulkanMemoryModel` are now queried and
> **requested** at `vkCreateDevice`. **`vulkanMemoryModel` is required by the
> FLOAT path too** — glslang makes `GL_KHR_memory_scope_semantics` a
> prerequisite of `GL_KHR_cooperative_matrix` — so slice 2's coopmat plumbing
> was incomplete independently of the integer work.
>
> **Coverage is exhaustive, not sampled — and the derivation is written out
> because "exhaustive" is usually sampling in disguise.** The full domain of a
> 16×16×16 u8 multiply-add is 256⁵¹² and cannot be enumerated. The domain that
> matters for an EXACTNESS claim can be: there are exactly 65536 ordered pairs
> `(a, b)` of u8 operand values, and the question is whether every one of them
> is multiplied at least once.
>
> Take `A[i][k] = 16k + i` and `B[k][j] = 16t + j` for dispatch `t`, all indices
> in `0..15`. One multiply-add forms the 4096 products `A[i][k] · B[k][j]` over
> the `(i, j, k)` triples. Fix `k`: the A-values are `{16k + i : i}` = the block
> `[16k, 16k+15]`, and the B-values are `{16t + j : j}` = the block
> `[16t, 16t+15]`, so the 256 pairs at that `k` are exactly the 16×16 block
> `[16k, 16k+15] × [16t, 16t+15]`. Ranging `k` over `0..15` gives sixteen blocks
> that are disjoint (their A-ranges are disjoint), for 4096 pairs per dispatch
> with no repeat. Ranging `t` over `0..15` moves the B-range through all sixteen
> of its blocks, so the 256 blocks `(k, t)` tile `[0,255] × [0,255]` exactly.
>
> **65536 pairs, each exactly once, in sixteen dispatches.** Every one with a
> nonzero mixed-sign `C`, and bit-identical to the oracle on every one of the
> 4096 outputs of every tile. Wrapping accumulation is exact too, and that case
> asserts the reference ACTUALLY wrapped rather than merely running.
>
> **The refusal was observed on both halves.** The iGPU advertises no
> `VK_KHR_cooperative_matrix`; its verdict refuses while the RX 7900 XTX permits
> in the same run under the same driver build, and the launch gate refuses on
> six devices (the Vulkan iGPU, both OpenCL devices, Native, and both
> Interpreter devices). The gate assertion relates two **independently
> observed** facts — the extension bit and the verdict — rather than comparing
> the verdict to the list it reads, which is the tautology slice 2's gate test
> fell into.
>
> **Every claim above was proved falsifiable by mutation.** Each row below was
> applied, run on the hardware named above, and observed to produce the failure
> it promises. The list is the inventory of what is pinned; nothing is
> summarised as a count.
>
> | # | mutation | observed |
> |---|---|---|
> | 1 | driver test: shader loads B column-major | `tile 0 diverges at 0: got 19340 want -500` |
> | 2 | driver test: shader drops C | exhaustive and wraparound cases both red |
> | 3 | driver test: `storageBuffer8BitAccess` never requested | **only** the plumbing assertion red — see the finding below |
> | 4 | e2e: codegen emits `ColumnMajor` in `CM_load` | 16 tiles diverge, first `23780` vs `368140` |
> | 5 | e2e: codegen swaps the A and B operands | glslang rejects the shader — `UseA` and `UseB` are not interchangeable types |
> | 6 | e2e: interpreter oracle drops C in `CM_muladd` | 16 tiles diverge, first `-500` vs `0` |
> | 7 | e2e: launch gate returns no configurations | all six refusing devices report `PERMITTED` |
> | 8 | e2e: the stride-1-B control made a no-op | `the stride-1-B control was ACCEPTED on 16 of 16 tiles` |
> | 9 | e2e: the C-dropping control made a no-op | `the C-dropping control was ACCEPTED on 16 of 16 tiles` |
>
> Rows 8 and 9 mutate the CONTROLS rather than the code under test, and they are
> here because a positive control is itself a gate that can rot: if the mutated
> IR a control builds ever stopped differing from the real one, the control
> would be accepted every time and its assertion would pass forever while
> checking nothing. Row 8 exists specifically because that control was
> MISLABELLED — see the note on the stride-1 read below — and a mislabelled
> control is the shape a vacuous one arrives in.
>
> Mutation 1 leaves the identity case (`D = A × I + 0`) **green**, because `I`
> is symmetric. That is the reason the identity case and the exhaustive case are
> two separate tests rather than one.
>
> **The stride-1-B control is a Hankel read, not a transposed B.** `CM_load`
> with stride `s` reads `m[r][c] = buf[base + r·s + c]`, so a stride of 1 gives
> `buf[r + c]`: every row is the previous row shifted by one, and the whole
> 16×16 fragment comes from the first 31 elements of the buffer. The transpose
> would be `buf[c·16 + r]` and is not reachable through a row-major stride at
> all — it needs `gl_CooperativeMatrixLayoutColumnMajor`, which this slice does
> not emit. The control is valid either way, since all a control must do is
> compute a genuinely different function; but it was recorded as "transposed B"
> in an earlier revision of this section and in the test itself, and that was
> wrong. Both are corrected. The driver-side test's `transpose_b` IS a genuine
> transpose — it is a host-side reference, not a stride — and is unchanged.
>
> **FINDING — a device feature can be missing with every numeric test green, and
> only a capability assertion can see it.** This is mutation 3 and it is the most
> instructive result of the slice, so it is recorded as a finding and not as a
> table row. With `storageBuffer8BitAccess` never requested, the shader is a
> specification violation — it declares `OpCapability StorageBuffer8BitAccess`
> against a logical device that never enabled it — and RADV computes the
> **correct answer anyway**. All three numerics tests stay green; only the
> feature assertion goes red. That is backlog-142's failure mode reproduced
> exactly, one feature over: no results-based test can catch it, however
> exhaustive its inputs, because the results are right. The instrument that sees
> it is the capability model, and the reason it exists.
>
> **The order reversal paid beyond its intent.** Building the integer path first
> is what surfaced the three missing device features at all — they were read off
> the SPIR-V that an integer `coopMatMulAdd` emits. Starting from the float side
> would have found neither `shaderInt8` nor `storageBuffer8BitAccess`, and would
> have found `vulkanMemoryModel` only by accident, since a float coopmat shader
> needs it for the same `GL_KHR_memory_scope_semantics` reason and slice 2 had
> already shipped without noticing.
>
> **What this did NOT do**, and each is a refusal in the code rather than an
> omission:
> - **The f16 coopmat path.** `Sarek_ir_glsl` refuses float components, and so
>   does the interpreter — the latter because §5.1's implementation-defined
>   addition order means no strict oracle exists to compare against. 4a's
>   contract is exactly as unmeasured as it was.
> - **The scalar-f16 refusal of slice 3-as-written is untouched.**
> - **Saturating accumulation, column-major layout, workgroup scope.** Each is
>   one enumerant to emit and a second behaviour to verify against an oracle,
>   and none is DELIVERED: the codegen refuses all three.
>
>   Stated precisely, because mutations 1 and 4 above did emit column-major
>   shaders and did run them: column-major has executed only as a MUTATION, to
>   demonstrate that the comparison goes red. It has never been verified against
>   the interpreter, so nothing is known about whether Sarek would emit it
>   correctly — which is exactly why it is refused rather than shipped. "Not
>   executed" would have been the wrong word and it was the word used here
>   before.
> - **The PPX surface.** A coopmat kernel is built as IR directly. Nothing in
>   `[%kernel ...]` produces an `SCoopmat`, and the PPX's parallel `stmt` type
>   deliberately did not gain the constructor.
> - **Metal, CUDA, OpenCL, WGSL, PTX** refuse rather than emit.


Sub-sliced deliberately, because 4a is cheap, is the highest-information step,
and needs nothing from the DSL:

- **4a — hand-written GLSL coopmat shader, driven through the existing
  `sarek-vulkan` dispatch.** Requires `GL_KHR_cooperative_matrix`, the
  `shaderFloat16` plumbing of slice 2, and a 16×16×16 f16→f32 kernel. Measure
  against the exact host reference of §5 — which is `D = A × B + C`, **with a
  nonzero `C` exercised**, not the dot product alone — using the `γ_16` bound of
  §5.2 and one of §5.3's two exact-reference constructions, with the input-span
  invariant asserted rather than assumed if binary64 is chosen. Both positive
  controls of §5.4 are required — a binary16-accumulating reference and a
  `C`-dropping reference must each be *rejected*, which is what shows the gate can
  go red at all. Run the §1.4 determinism tests,
  including the dispatch-shape variation. **Proves the numeric contract of §5 on
  a real implementation** — which is today entirely unmeasured — before any IR
  work is committed to it.
- **4b — the IR fragment type.** The new type class that
  `f16-dsl-element-type.md` §8 slice 3 names and defers. Two shape requirements,
  both cheap now and expensive to retrofit, and **both are binding on this slice
  rather than advisory**:

  - **Dimensions are not hard-coded.** Must accommodate 16×16×16 subgroup-scope
    (Vulkan, measured §4) *and* 8×8×8 (Metal, measured §7 slice 6 — 8×8 is the
    **only** size MSL offers, so this is a hard requirement and not a guess).
  - **Component types admit integers from the start** — `u8`/`s8` operands with
    `u32`/`s32` accumulate, including the saturating variants. This is §8's
    recommendation and it is adopted: 12 of the 14 configurations advertised on
    the local device are integer, `SPV_KHR_cooperative_matrix` states integer
    accumulation is *exact* at the precision of the result type, and those
    configurations are therefore deliverable under Sarek's **existing strict
    contract** with no relaxation, no allowlist and no opt-in. Building the type
    to admit them costs almost nothing at design time and is a wide, invasive
    change afterwards.

    It is also the **fallback if slice 1 goes the wrong way**: if the ACO shapes
    fail to match a closed-form model and the scalar contract has to become
    per-shape, an integer-only coopmat path still lands, still under the strict
    contract, and backlog-62 is not blocked on the accuracy question at all. That
    fallback only exists if 4b was built for it.

    **The fallback is backlog-62's, not backlog-63's — measured, not assumed.** Metal has *no*
    integer `simdgroup_matrix`: the only element types are `half`, `float` and
    `bfloat`, and integers fail a named `is_simdgroup_matrix_element<T>`
    static_assert, so the enumeration is closed (§7 slice 6). An earlier reading
    of this bullet implied every backend had a strict-contract path to fall back
    on. **backlog-63 does not.** On Metal the tensor-core path is reachable only through
    the relaxation, which makes backlog-63 strictly more exposed to it than backlog-62 — if the
    relaxation were withdrawn, backlog-62 could still ship integer configurations and
    backlog-63 could ship nothing. Admitting integer component types remains right for
    the reasons above; it is simply not a universal safety net.

  The *slicing* is deliberately **not** reordered to put integers first — f16
  scalar is a prerequisite for backlog-63 and for bf16 regardless
  (`f16-dsl-element-type.md` §11.1), so the relaxation work is resequenced rather
  than avoided, and an integer-only tensor-core path is not what the intended
  audience means by "tensor cores". The type admits integers; the plan still
  leads with f16.
- **4c — GLSL codegen for the fragment type**, gated on the `Device_optional`
  coopmat capability. The Raphael iGPU is the free negative device (§4).

### Slices 5 and 6 — backlog-63, Metal — **DONE, 2026-07-27**

Both were unschedulable pending access to an Apple GPU. Access was granted, the
probes were run on an Apple M4, and both slices are complete. Recorded in
`fp-contraction-policy.md` §10.14 and §10.15; the rows are in §2 above.

- **Slice 5 — scalar f16 on Metal. Done: it does not fuse.** 0 / 63488 from
  `S_strict` on both swept shapes, element-wise, on the naive kernel and under
  every barrier — and the control goes red on **each** shape separately
  (2912 / 63488 on `f16(x*1.1)`, 620 / 63488 on `f16(f16(x*1.1)+1000)`), so
  neither zero rests on discrimination demonstrated only for the other shape. Instrument `tools/probes/metal_f16_narrowing_probe.m`. Two
  results worth carrying forward beyond the row:
  - `#pragma METAL fp contract(off)` **does not govern this hazard**, in either
    direction — it does not move the plain kernel and does not disturb the
    control. That is not inertness: §10.5 measures the same pragma taking
    `a*b+c` from 8773/8773 to 0/8773 on this device. Contraction of `a*b+c` and
    absorption of a multiply into an f32→f16 narrowing are two behaviours, and
    Metal exhibits neither.
  - The control had to be built differently, and that is a portability note for
    slice 0: **MSL has no `double`**, so the `fusedctl` construction used on
    OpenCL is not expressible. The Metal control reconstructs the exact product
    from a double-float pair with a round-to-odd step. It reproduces **2912** on
    the one-narrowing shape and **620** on the two-narrowing shape; the 620 is
    now four unrelated stacks agreeing on that figure.
  - It also produced the counterexample that corrected §1.3. See there.
- **Slice 6 — `simdgroup_matrix`. Done, and it disproved a planning assumption.**
  MSL offers exactly three instantiations, all 8×8: `half`, `float`, `bfloat`.
  Every other size fails `_valid_simdgroup_matrix_size` and every integer type
  fails `is_simdgroup_matrix_element<T>`, both named `static_assert`s — so this is
  a **closed enumeration, not a sample**. Numerics are in §2's Regime B rows: the
  f32-accumulate configuration matches a closed-form model element-wise and
  migrates to Regime A; the f16-accumulate one does not.

> **The integer-coopmat fallback is Vulkan-only. It does not exist on Metal.**
> §8 and slice 4b both keep an integer configuration as the route that lands
> under Sarek's *existing strict contract* if slice 1 finds no closed-form model
> for the ACO shapes — resting on the fact that 12 of the 14 configurations RADV
> advertises are integer. **That reasoning does not transfer to Metal, and this
> document previously implied a universal fallback.** There is no integer
> `simdgroup_matrix` at all, so **backlog-63 has no strict-contract route**: on Metal it
> is f16 or bf16 or nothing, and it is reachable *only* through the relaxation.
>
> Two consequences the plan has to carry:
> - **Slice 4b's integer requirement keeps its justification but loses its
>   universality.** Admitting integer component types in the IR fragment type is
>   still right — it is nearly free at design time, expensive to retrofit, and it
>   is what keeps backlog-62 deliverable if slice 1 goes badly. But it is a **backlog-62
>   fallback, not a backlog-62-and-backlog-63 fallback**, and slice 4b should say so rather than
>   let a reader infer that every backend has a strict-contract path.
> - **backlog-63 is more exposed to the relaxation than backlog-62 is.** If the relaxation were
>   ever withdrawn, backlog-62 could still ship its integer configurations and backlog-63 could
>   ship nothing. That asymmetry did not exist in the plan as written and should
>   be visible when the two are prioritised against each other.
>
> `bfloat` 8×8 is available, is not in the plan anywhere, and has **not** been
> swept — nothing is claimed about what it computes. It is the obvious cheap
> extension of slice 6 and needs no new hardware access.

**Apple M4 access: requested, granted, used.** Two standalone probe binaries were
built and run in a scratch directory under `/tmp`; nothing installed, no settings
changed, no working tree touched. The absence of Xcode (Command Line Tools only,
so no offline `metal` compiler) turned out not to matter —
`newLibraryWithSource:options:error:` compiles through the driver at runtime,
which is the call under test anyway.

**What slices 5 and 6 did NOT cover**, neither needing new hardware access:
the 20-shape catalogue of
`docs/optimization/amdgpu-f16-fusion-shape-audit.md` (only two shapes were
swept), a `bfloat` 8×8 sweep, subnormal-specific analysis, and Metal **codegen** —
both probes compile hand-written MSL and therefore measure the driver, exactly as
§7(c) of the policy document says of the GLSL tripwire. A Sarek-generated-shader
gate is a separate slice.

### Hardware this plan does not have

- ~~**Apple GPU** — an Apple M4 machine, permission needed. Blocks slices 5–6 entirely.~~
  **Resolved 2026-07-27**: access granted, slices 5 and 6 both executed on an
  Apple M4. What Metal still lacks is not hardware but coverage — the 20-shape
  catalogue, a `bfloat` sweep, and a codegen-side gate.
- **NVIDIA Ampere or newer** — the GTX 1070 is sm_61 with no tensor cores, no
  bf16 and no FP8 (`capability-model.md` §1). Nothing in this plan constrains the
  CUDA/PTX tensor-core path, and no f16 *performance* claim is available from
  local hardware either.
- **AMDVLK, the proprietary AMD Vulkan driver, NVIDIA's Vulkan driver, lavapipe**
  — all unmeasured coopmat implementations. Under §1.5 they are `Unknown` and
  therefore refused, which is the safe direction and requires no extra work.

---

## 8. An argued objection: the cheapest first tensor-core slice needs no relaxation

Recorded because it is a real alternative to the framing of the request.

Of the 14 cooperative-matrix configurations advertised locally (§4), **12 are
integer**, and `SPV_KHR_cooperative_matrix` states that integer accumulation is
exact at the precision of the result type. Those configurations are deliverable
**under Sarek's existing strict contract**, with no accuracy relaxation, no
allowlist and no `accept_relaxed_f16` opt-in.

They exercise every structurally hard part of backlog-62: the `Device_optional`
capability gate, the `VkPhysicalDeviceCooperativeMatrixPropertiesKHR` query, the
`shaderFloat16`-adjacent feature plumbing, the new IR fragment type, the subgroup
ABI, and the codegen. Only the *numerics* differ.

**So an alternative slicing exists in which backlog-62's whole skeleton lands before any
part of the accuracy relaxation is used**, and the relaxation is then applied to
one narrow thing (the f16 accumulate path) with the machinery already proven.
`u8 × u8 → s32` is also a real workload — quantised inference is the main
consumer of integer tensor cores.

Against it: the f16 scalar type is a prerequisite for backlog-63 and for bf16 regardless
(`f16-dsl-element-type.md` §11.1), so the relaxation work is not avoided, only
resequenced; and an integer-only coopmat path is not what "tensor cores" means to
most of the intended audience.

> **RESOLVED (2026-07-27) — the objection is taken in its type-design half and
> declined in its sequencing half.** The plan is **not** reordered. Slice 4b's
> fragment type **must** admit integer component types from the start, and that
> is now a binding requirement of the slice rather than a recommendation — see
> §7 slice 4b. It is nearly free at design time, expensive to retrofit, and it is
> what keeps a strict-contract coopmat path available as a fallback if slice 1
> finds no closed-form model for the ACO shapes.
>
> **AMENDED 2026-07-27, after the Metal measurement.** This whole section is
> scoped to **backlog-62**. It reads as though "the cheapest first tensor-core slice
> needs no relaxation" were a statement about tensor cores in general; it is a
> statement about *Vulkan*, and it holds only because RADV advertises integer
> configurations. **Metal advertises none** — `simdgroup_matrix` exists for
> `half`, `float` and `bfloat` only, integers failing a named static_assert, a
> closed enumeration (§7 slice 6). So the objection's escape route is unavailable
> to backlog-63 entirely, and the sequencing argument is stronger than it looked: f16 is
> not merely a prerequisite for backlog-63 "regardless", it is the **only** element type
> backlog-63 can be built on. Nothing in the resolution changes; its scope is now
> stated.

---

## 9. Where the decision was interpreted, and where it has since been settled

Recorded plainly, because none of these was read off the ruling — each was a
reading. Two have since been decided by the owner and are marked as such; the
rest remain interpretations this design is responsible for.

### 9.1 Settled

3. **The decision was taken about "f16" as one thing; it is two.** Coopmat cannot
   hold bit-identity by specification; scalar f16 on ACO *can*, at the measured
   cost of a global round-trip per narrowing (§3).
   > **DECIDED (2026-07-27): relax both, and record the scalar half explicitly as
   > a performance trade.** The exact barrier stays documented and available. The
   > coopmat reachability argument is not permitted to carry the scalar case;
   > §3.1 states the scalar decision on its own terms, including that it is the
   > half that should be revisited if the cost ever changes.
5. **The opt-in question (§6).** Whether a relaxed path needs an explicit user
   acknowledgement or only a loud diagnostic is a product decision about friction.
   > **DECIDED (2026-07-27): friction proportional to the strength of the
   > evidence** — loud one-time diagnostic where the deviation matches a
   > closed-form model bit-for-bit over an exhaustive sweep, mandatory explicit
   > opt-in where it is only bounded. §6.1 records the derivation and the
   > corollary that friction *falls* as evidence improves, with no new decision
   > required.

### 9.2 Still this design's interpretation, not the owner's words

1. **"Globalement corrects" has no operational meaning and it is the load-bearing
   phrase.** §0.1/§1 choose exact agreement with an admitted model set plus a
   derived ceiling, over an ulp band. The reason is measured, not stylistic, but
   it is still a choice and a different reading would give a different contract.
2. **"Déterministe" — the strong reading is already false.** §1.4 promises the
   weak one (same device, same driver version, same build) and says so. If the
   owner meant the strong reading, the decision cannot be implemented as stated
   on any backend, including the ones that pass today.
4. **Uniform or per-backend was left open, and per-backend is the honest
   answer.** A single global tolerance admitting RADV's 5075/63488 is very loose;
   §2's per-backend, per-driver, per-shape record is what the measurements support
   and it costs an allowlist that must be maintained. The maintenance burden is
   real and is the price of the honesty.
6. **Nobody has said what happens to an existing user's results.** f16 is refused
   on these backends today, so no user can be relying on them — the relaxation
   cannot regress anyone. Worth stating explicitly, because it is the reason this
   can be done at all without a deprecation path.

### 9.3 The open risk — CLOSED, 2026-07-27

**RADV's `precise` behaviour on the two-narrowing shape is now understood, and
it does not break the design.** The risk as written was: 5075/63488 plain
against 4776/63488 with `precise` means the decoration *changes the answer* —
while `fp-contraction-policy.md` §6 shows the same decoration produces
**byte-identical ISA** on the one-narrowing shape, and that the one-narrowing
shape matches a single-rounding model exactly. Those facts looked inconsistent.

**They are consistent, and the mechanism is one sentence.** `NoContraction`
forbids contracting a multiply into an **addition**. A **conversion** absorbing
its operand is a different combine and the decoration does not reach it.

- `f16(x*1.1)` contains no addition. The decoration is emitted, binds nothing,
  and the ISA is byte-identical — `v_fma_mixlo_f16` either way. **RADV is not
  ignoring the decoration here; the decoration is inapplicable.**
- `f16(f16(x*1.1)+1000)` contains one, once ACO has elided the intermediate
  narrowing. The decoration **binds**: the multiply is materialised as its own
  `v_fma_mix_f32` and only the add is absorbed. The ISA changes, and so does
  the model — from `S_absorb_all_into_final_narrowing` to
  `S_f32_mul_then_absorb_add`.

Evidence: **executed** element-wise, 63488/63488 for each model on both local
RADV devices, plus **machine-code** (`RADV_DEBUG=asm`, one shader per run so
each dump is attributable). Record: `fp-contraction-policy.md` §12.4.

**So the contract does not become per-shape in the sense feared.** §1.2's model
set was always keyed on *(backend, driver, expression shape)*; what §9.3 feared
was a shape matching **no** model, which would have forced a refusal that could
not be stated as a function. Every shape swept matches one exactly. What the
result *does* cost is that **`S_fuse_mul_into_narrowing` alone does not cover
RADV** — the admissible set gains two members (§1.2), and the member in force
depends on whether the codegen emits `precise`. That is a real maintenance
obligation and it is the honest price of the outcome.

**What remains open is coverage, not mechanism.** Two of the twenty emittable
shapes are measured. The remaining 18 are `Unknown` and refused under §1.5, and
slice 3 must not lift a refusal past them. The single generative rule that
would collapse the per-shape table back to a per-driver one is stated in
`fp-contraction-policy.md` §12.4 and is **unverified** at that scope.

The mitigation named here — §7 slice 4b's integer component types — is no longer
needed as a fallback for the accuracy question, but stays a binding requirement
of that slice for the reasons §8 gives.

---

## 10. Tests added by this change

**None as gates.** The 2026-07-27 slice-1 amendment adds two probe
**executables** — `sarek-opencl/probe/probe_opencl_f16_model_agreement.ml` and
`sarek-vulkan/probe/probe_vulkan_f16_model_agreement.ml` — and the model library
they share, `tools/f16_model_set/`. They are built by `dune build` and run by
hand; they are deliberately not `(test)` stanzas, because they measure a driver
in order to decide whether a contract is deliverable and would otherwise block
CI on whatever GPU a runner happens to have. The two tripwires that defend the
current refusals are untouched. The original text of this section follows.

**None.** This change adds a design document and three standalone probes, none
wired into dune, matching the convention of the ones already there:
`tools/probes/vulkan_coopmat_probe.c` (read-only Vulkan query), and — from the
2026-07-27 amendment — `tools/probes/metal_f16_narrowing_probe.m` and
`tools/probes/metal_simdgroup_matrix_probe.m`. No Sarek behaviour changes, no
refusal is lifted, and there is nothing new to gate. The test strategy is §7
slice 0, which is the first slice of the follow-on work — and which now has two
extra obligations from the Metal measurement: the ceiling must be evaluated
per-narrowing (§1.3), and the `x = -907.5` case must be a host-only calibration
that a final-value ceiling is shown to fail.
