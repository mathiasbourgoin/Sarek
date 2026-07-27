# The relaxed f16 accuracy contract, and the tensor-core paths it unblocks

_What Sarek promises about f16 results when bit-identity with the interpreter is
no longer required, and how #62 (Vulkan cooperative-matrix) and #63 (Metal
`simdgroup_matrix`) are sliced on top of it._

**Status:** design only; the two decisions this document opened were taken by the
project owner on 2026-07-27 and are recorded as decided in **§3** (scalar f16 is
relaxed, as an explicit performance trade) and **§6** (user-facing friction is
proportional to the strength of the evidence). **No refusal is lifted by this
change.** **Issues:** #62, #63; depends on the f16 scalar type (#57) and the
capability model (#64). **Date:** 2026-07-27.

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

That is a genuine relaxation and it is what makes #62 and #63 reachable. It is
also not yet a specification. "Correct" without a bound admits anything; and this
repository has already shipped a gate whose tolerance was wider than the effect
it existed to detect (`docs/fp-contraction-policy.md` §11.5, backlog #118), where
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
> binary16 result**, measured on the final value, with subnormal results compared
> absolutely against `2^-24` (the binary16 subnormal spacing). Derivation, not a
> round number: every deviation in the admitted class is the *elision of exactly
> one round-to-nearest step*, and a single elided rounding moves a value by at
> most half an ulp at the elided step, which is at most one ulp at the final step.
> Anything larger is not a rounding difference and must not be filed as one.

The ceiling is **necessary and not sufficient** — §1.1 is the proof, since the
IGC dropped-narrowing defect sits exactly at 1 ulp. Both checks run; both must
pass.

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
§1 corollary 3. Evidence tiers are that document's. Regime B has no rows here —
no coopmat implementation's numerics have been measured at all (§4, last
paragraph).

**No stack in this table is settled as relaxable.** Both non-zero rows are
**candidates**, and neither has met the acceptance rule of §1.2: rusticl's
agreement with `S_fuse_mul_into_narrowing` is established only at the level of a
count and a first divergence, not element-wise; RADV's is exact for the
one-narrowing shape and **unmeasured for the two-narrowing shape**. Slice 1 (§7)
is what would convert either from candidate to admitted, and it may convert
neither. Read the verdict column as the current status, not as a plan.

| backend / driver | device(s) | deviation from `S_strict` | matches `S_fuse_mul_into_narrowing`? | evidence | verdict under Regime A |
|---|---|---|---|---|---|
| **CUDA / nvrtc** | GTX 1070 Max-Q, sm_61, CUDA 12.9, driver 580.119.02 | **0 / 63488** | n/a | executed | **strict**, unchanged |
| **HIP / AMDGPU** | RX 7900 XTX (gfx1100), Raphael iGPU (gfx1036), ROCm hiprtc, with the opacity barrier | **0 / 63488**, all 20 emittable shapes | n/a | executed + machine-code | **strict**, unchanged |
| **OpenCL / rusticl** | RX 7900 XTX + Raphael iGPU, Mesa 26.1.4-arch3.1 | **620 / 63488** on `f16(f16(x*1.1)+1000)` | **count and first-divergence agreement** (620, first divergence `x = 5.68359375`, device 1006.5 vs reference 1006 — identical to `fusedctl`'s deliberate single-rounding on Intel). **Element-wise agreement not established.** | executed | **candidate**, blocked on slice 1 |
| **OpenCL / pocl (x86), Intel IGC, Intel oneAPI CPU** | AMD EPYC 7763 (CI), Intel Arc Graphics (MTL), Core Ultra 9 185H | **0 / 63488** | n/a | executed | **strict**; refusal is currently over-broad here |
| **Vulkan / RADV** | RX 7900 XTX (NAVI31) + Raphael iGPU (RAPHAEL_MENDOCINO), Mesa 26.1.4-arch3.1 | **2912 / 63488** on `f16(x*1.1)`; **5075** plain / **4776** with `precise` on `f16(f16(x*1.1)+1000)` | one-narrowing shape: **exact, 0/63488 against a single-rounding model**. Two-narrowing shape: **not measured against any model**, and `precise` changes the count, so a single model does not obviously cover it | executed | **candidate for the one-narrowing shape only**; blocked on slice 1 |
| **Vulkan / ANV** | Intel Arc Graphics (MTL), Mesa 26.1.2-arch3.1 | **0 / 63488** | n/a | executed | **strict**; refusal is currently over-broad here |
| **Metal** | — | **never probed** | — | none | **`Unknown` → refused.** Needs ladon (§7) |
| **WGSL / naga** | — | **never probed** | — | none | **`Unknown` → refused** |
| **PTX** | — | refuses kernel-level f16 by design (#57 slice 2) | — | by-construction | nothing to relax |

Three things this table says that are easy to miss:

1. **The relaxation is a candidate on exactly two stacks** — rusticl and RADV,
   both ACO — and admitted on none of them yet. Everything else either already
   meets `S_strict` or has never been measured. Even in the best case this is a
   much narrower change than "relax f16"; in the worst case (§9.3) it is narrower
   still, covering one shape on one driver.
2. **The RADV two-narrowing shape is the open risk.** 5075 plain against 4776
   with `precise` means the decoration *changes the answer* while
   `fp-contraction-policy.md` §6 shows it produces byte-identical ISA on the
   one-narrowing shape. Those two facts are not obviously consistent and nobody
   has reconciled them. If the two-narrowing shape matches no closed-form model,
   it stays refused and the contract becomes per-shape — a materially worse
   design, and the owner should hear about it before more effort is spent. Slice 1
   is the decision point.
3. **pocl, IGC, ANV and the Intel CPU runtime are refused today for a defect they
   do not have.** `capability-model.md` §5 already records this as the
   over-refusal that follows from having no compiler-identity probe. It is not
   made worse by this design, and slice 2's `VkPhysicalDeviceDriverProperties`
   plumbing narrows the Vulkan half of it.

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
`VK_KHR_cooperative_matrix` (#62) nor Metal `simdgroup_matrix` (#63) can be
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

1. **#62 is not blocked on hardware.** The path can be built and measured
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
  `Σ_k |A[m][k]·B[k][n]| + |C[m][n]|`.
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
| exact match to a named closed-form model, swept exhaustively (`S_fuse_mul_into_narrowing` on ACO, if slice 1 confirms it) | a **one-time runtime diagnostic** on first launch of an f16 kernel on that device, naming the model and the doc section, on by default; silenceable only by an explicit call, never by accident |
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
   default, and #62/#63 were unblocked in order to be used.
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

## 7. Slicing plan for #62 and #63

Each slice names what it proves and what hardware it needs. Nothing below lifts a
refusal before slice 3.

| # | slice | proves | hardware | lifts a refusal? |
|---|---|---|---|---|
| **0** | the contract, as a testable classifier | the gate can tell `S_fuse_mul_into_narrowing` from a dropped narrowing | none (host-only) | no |
| **1** | element-wise model characterisation of ACO scalar f16, all emittable shapes | whether the contract of §1.2 is deliverable at all | RX 7900 XTX (local) | no |
| **2** | `shaderFloat16` + driver-identity + capability plumbing | the gate can be keyed on a driver, not a device name | local, both devices | no |
| **3** | GLSL scalar f16, allowlisted | Sarek-*generated* f16 shaders meet the contract | RX 7900 XTX (local) | **yes**, on an allowlist |
| **4** | #62 Vulkan coopmat, f16×f16→f32 | the tensor-core path, end to end | RX 7900 XTX (local) | yes (new capability) |
| **5** | #63 Metal — **scalar f16 first** | the Metal row of §2, which does not exist | **ladon (M4) — permission needed** | no |
| **6** | #63 Metal `simdgroup_matrix` | the Apple tensor-core path | **ladon** | yes |

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
hiprtc/gfx1100, rusticl/radeonsi and `fusedctl` on Intel Arc.
`test_opencl_f16_tripwire` already does this after §11.5 and it is the model to
copy.

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

**This slice reports its outcome upward before the next one starts.** It is a
decision point, not a task on a list; a green result funds slices 2–4 and a red
one reopens the contract.

Also upgrade the rusticl row of §2 from count-and-first-divergence agreement to
element-wise agreement, which is what §1.2 actually requires and which nobody has
run.

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

### Slice 4 — #62, Vulkan cooperative-matrix (local)

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
    (Vulkan, measured §4) *and* 8×8×8 (`simdgroup_half8x8`, Metal).
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
    contract, and #62 is not blocked on the accuracy question at all. That
    fallback only exists if 4b was built for it.

  The *slicing* is deliberately **not** reordered to put integers first — f16
  scalar is a prerequisite for #63 and for bf16 regardless
  (`f16-dsl-element-type.md` §11.1), so the relaxation work is resequenced rather
  than avoided, and an integer-only tensor-core path is not what the intended
  audience means by "tensor cores". The type admits integers; the plan still
  leads with f16.
- **4c — GLSL codegen for the fragment type**, gated on the `Device_optional`
  coopmat capability. The Raphael iGPU is the free negative device (§4).

### Slices 5 and 6 — #63, Metal (needs ladon; permission not yet requested)

**#63 cannot start at the matrix layer.** `fp-contraction-policy.md` §3 lists
Metal f16 on the "may NOT rely on" list with the note that it has **never been
probed**, and §2's Metal row covers f32 only. There is no Metal row in §2 of this
document because there is no measurement to put in it. So:

- **Slice 5 — scalar f16 on Metal.** A standalone probe in the shape of the
  existing `tools/probes/metal_math_mode_probe.m` and
  `metal_contraction_barrier_probe.m`: exhaustive sweep of all 63488 finite
  binary16 inputs against `S_strict` and `S_fuse_mul_into_narrowing`, with the
  `fusedctl`-style positive control. Note Metal's contraction defence is a
  *source pragma* (`#pragma METAL fp contract(off)`) and none of the compile
  options stop contraction (§2, §10.5 of the policy doc) — so the probe must
  sweep with and without the pragma. This is a **measurement, not a code
  change**, and it produces the Metal row that §2 lacks.
- **Slice 6 — `simdgroup_matrix`.** Enumerate what MSL actually offers on the M4
  (`simdgroup_half8x8` / `simdgroup_float8x8`), then the 4a-equivalent
  hand-written kernel against the §5-style derived bound, recomputed for K = 8.

**Ladon usage: not requested, not used.** An M4 is reachable at
`ssh -i ~/.ssh/id_ed25519 ladon`. Nothing in this document has touched it. The
minimum ask, if permission is granted, is: build and run two standalone probe
binaries in a scratch directory, install nothing, change no settings, touch no
working tree. Slices 5 and 6 are unschedulable until that is agreed.

### Hardware this plan does not have

- **Apple GPU** — ladon, permission needed. Blocks slices 5–6 entirely.
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

They exercise every structurally hard part of #62: the `Device_optional`
capability gate, the `VkPhysicalDeviceCooperativeMatrixPropertiesKHR` query, the
`shaderFloat16`-adjacent feature plumbing, the new IR fragment type, the subgroup
ABI, and the codegen. Only the *numerics* differ.

**So an alternative slicing exists in which #62's whole skeleton lands before any
part of the accuracy relaxation is used**, and the relaxation is then applied to
one narrow thing (the f16 accumulate path) with the machinery already proven.
`u8 × u8 → s32` is also a real workload — quantised inference is the main
consumer of integer tensor cores.

Against it: the f16 scalar type is a prerequisite for #63 and for bf16 regardless
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

### 9.3 The open risk, unchanged and deliberately not closed

**RADV's `precise` behaviour on the two-narrowing shape is not understood, and it
is the thing most likely to break this design.** 5075/63488 plain against
4776/63488 with `precise` means the decoration *changes the answer* — while
`fp-contraction-policy.md` §6 shows the same decoration produces **byte-identical
ISA** on the one-narrowing shape, and that the one-narrowing shape matches a
single-rounding model exactly (0/63488). Those facts are not obviously consistent
and nobody has reconciled them.

If the two-narrowing shape matches **no** closed-form model, the contract of §1.2
becomes **per-shape** rather than per-backend. That is materially worse: harder to
document, harder to explain to a user, and a per-shape allowlist is a maintenance
object of a different order from a per-driver one.

**Slice 1 is therefore structured as a decision point, not a task**, and the
owner should hear the outcome **before slices 2–4 are funded** — not after. See
§7 slice 1. The mitigation, if it goes the wrong way, is §7 slice 4b's integer
component types: an integer-only coopmat path lands under the existing strict
contract and #62 is not blocked on the accuracy question at all.

---

## 10. Tests added by this change

**None.** This change adds a design document and one read-only Vulkan query probe
(`tools/probes/vulkan_coopmat_probe.c`, standalone, not wired into dune —
matching the convention of the four probes already there). No Sarek behaviour
changes, no refusal is lifted, and there is nothing new to gate. The test
strategy is §7 slice 0, which is the first slice of the follow-on work.
