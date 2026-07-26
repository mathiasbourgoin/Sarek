# AMDGPU f16 fusion/demotion: exhaustive shape audit

_Issue #106. Measured 2026-07-25/26 on **AMD Radeon RX 7900 XTX (gfx1100)** and
**AMD Ryzen 9 7950X iGPU (gfx1036)**, ROCm hiprtc, `-ffp-contract=off` forced
last per `Hip_rtc.base_options`. Disassembly via `hipcc`/ROCm LLVM for gfx1100._

## What was already known

An AMDGPU ISel combine fuses an f32 multiply into the f32→f16 narrowing that
consumes it (`v_fma_mixlo_f16`), and demotes a neighbouring f32 add to binary16
(`v_add_f16`). It sits **below** the C-level FP controls, so `-ffp-contract=off`
does not prevent it. `Sarek_ir_cuda.sarek_f32_barrier` — `asm volatile("" :
"+v"(x))` on the narrowing's argument — does.

That was established on **one** expression shape (`f16_midround`, 620 of 63488
finite binary16 inputs disagreeing with the interpreter). The `v_add_f16`
observation showed the class was broader than multiply-then-narrow, but nobody
had enumerated it.

## Why the enumeration is small, and closed

f16 in Sarek is a **storage-only element type**. The typer rejects f16 operands
for every arithmetic, comparison, bitwise and unary operator; there is no f16
literal; there is no f16 math intrinsic; f16 struct fields are rejected at
layout; an f16 scalar kernel parameter is rejected at lowering. The entire f16
surface is two core primitives, `float16_of_float32` and `float32_of_float16`.

So the only f16 value producer is `ECast (TFloat16, e)` for an f32-typed `e`,
and "audit every f16 expression shape" reduces to: **audit the shapes of `e`
that can reach a narrowing**, plus the two paths with no narrowing at all (a
straight f16→f16 copy, and a widening whose result never narrows). That is a
closed enumeration, not a sample.

## Method

Two independent instruments, because neither alone is sufficient.

**Numeric** — `sarek-hip/test/test_hip_f16_shapes.ml` sweeps each shape over
**all 63488 finite binary16 values** (the whole domain, including both zeros and
the entire subnormal range) and compares bit-exactly against the interpreter.
Three columns:

| column | what it runs |
|---|---|
| `shipping` | the ordinary Sarek path (`Execute.run_vectors`) |
| `src+barrier` | the generated HIP source, verbatim, through `run_source` |
| `src-barrier` | the same source with the barrier's `asm volatile` body deleted |

`src+barrier` exists so that `src-barrier` differs from it by exactly one
textual substitution and nothing else, and so that `shipping` vs `src+barrier`
checks the `run_source` path is faithful.

**ISA** — `scripts/f16_shape_isa_audit.sh` disassembles every shape for gfx1100
with and without the barrier.

The second instrument is not redundant. **A numeric null cannot distinguish "not
demoted" from "demoted, but exact on this domain"**, and the second is a latent
hazard: the demotion is present in the machine code and will bite on the next
expression shape that is not so lucky. Three shapes below are exactly that case.

### Calibration before trust

The harness reproduces the known result — shape B1 returns **620** mismatches
with the barrier removed, matching the originally reported 620 of 63488. The
harness was therefore shown to detect this defect class *before* its null
results were relied on. The test enforces this permanently: if removing the
barrier breaks **no** shape, it exits non-zero on the grounds that it has not
been shown to detect demotion at all and its zeros are uninformative.

This matters because the original 620 hid behind a *sampled* f16 test that was
green for months.

## Results

`ship` / `+bar` / `-bar` are mismatch counts out of 63488. Identical on gfx1100
and gfx1036.

| id | shape (`x` = `float32_of_float16 inp.(i)`) | ship | +bar | −bar | ISA with barrier | ISA without barrier | verdict |
|---|---|---|---|---|---|---|---|
| A1 | `narrow x` | 0 | 0 | 0 | `cvt` | *(collapses to a 16-bit load/store)* | unaffected |
| A2 | `narrow (x *. 1.1)` | 0 | 0 | **2913** | `cvt` | `v_fma_mixlo_f16` | **affected** |
| A3 | `narrow (x +. 1000.)` | 0 | 0 | 0 | `cvt` | `v_add_f16` | **demoted, unobservable** |
| A4 | `narrow (x -. 1000.)` | 0 | 0 | 0 | `cvt` | `v_add_f16` | **demoted, unobservable** |
| A5 | `narrow (x /. 3.)` | 0 | 0 | 0 | `cvt` | `cvt` + `v_fma_mix_f32`×2 | unaffected |
| A6 | `narrow (x *. 1.1 +. 1000.)` | 0 | 0 | 0 | `cvt` | `cvt` | unaffected |
| A7 | `narrow ((x +. 1000.) *. 1.1)` | 0 | 0 | **374** | `cvt` | `v_fma_mixlo_f16` | **affected** |
| A8 | `narrow (fma x 1.1 1000.)` | 0 | 0 | **484** | `cvt` + `v_fma_mix_f32` | `v_fma_mixlo_f16` | **affected** |
| A9 | `narrow (sqrt (x *. x))` | 0 | 0 | 0 | `cvt` | `cvt` | unaffected |
| A10 | `narrow (0. -. x)` | 0 | 0 | 0 | `cvt` | `v_sub_f16` | **demoted, unobservable** |
| A11 | `narrow (floor (x *. 1.1))` | 0 | 0 | 0 | `cvt` | `cvt` | unaffected |
| A12 | `narrow (if x>0. then x*.1.1 else x*.0.9)` | 0 | 0 | **2864** | `cvt` | `v_fma_mixlo_f16` | **affected** |
| A13 | `narrow (x *. x)` | 0 | 0 | 0 | `cvt` | `v_mul_f16` | **demoted, unobservable** |
| A14 | `narrow (x *. 1.1 *. 1.1)` | 0 | 0 | **309** | `cvt` | `v_fma_mixlo_f16` | **affected** |
| A15 | `narrow (x *. 1.1 +. x /. 3.)` | 0 | 0 | 0 | `cvt` | `cvt` | unaffected |
| B1 | mid-narrow then `+. 1000.` | 0 | 0 | **620** | `cvt`×2 | `v_add_f16` + `v_fma_mixlo_f16` | **affected** *(the original)* |
| B2 | mid-narrow then `*. 1.1` | 0 | 0 | **5369** | `cvt`×2 | `v_fma_mixlo_f16`×2 | **affected** |
| B3 | mid-narrow of add then `*. 1.1` | 0 | 0 | **963** | `cvt`×2 | `v_add_f16` + `v_fma_mixlo_f16` | **affected** |
| B4 | two mid-narrows then `*. 1.1` | 0 | 0 | **1518** | `cvt`×3 | `v_add_f16` + `v_fma_mixlo_f16`×2 | **affected** |
| C1 | `out.(i) <- inp.(i)` (no cast) | 0 | 0 | 0 | *(load/store only)* | *(load/store only)* | unaffected |

`cvt` = `v_cvt_f16_f32`, a standalone narrowing — the correct shape.
`v_fma_mix_f32` is **not** a demotion: it keeps an f32 result and is how an
explicitly requested `fma()` is meant to be emitted.

### Every shape is correct as shipped

**All 20 shapes, both devices: 0 mismatches out of 63488 on the shipping path.**
No currently-emittable f16 expression shape disagrees with the interpreter.

### The barrier is load-bearing, and one barrier covers everything

9 of 20 shapes break when the barrier is removed; the *same single* mechanism —
`sarek_f32_barrier`'s `asm volatile("" : "+v"(x))` — fixes every one. There is no
shape needing a different or additional barrier. With the barrier present, every
narrowing in every shape is a standalone `v_cvt_f16_f32`.

### Four demotion opcodes, not one

The combine family is wider than the two opcodes previously recorded:

| opcode | what it does | first seen here |
|---|---|---|
| `v_fma_mixlo_f16` | f32 multiply/fma fused **into** the narrowing | already known |
| `v_add_f16` | f32 add/sub demoted to a binary16 add | already known |
| `v_mul_f16` | f32 multiply demoted to a binary16 multiply | **new** (A13) |
| `v_sub_f16` | f32 negate/subtract demoted to binary16 | **new** (A10) |

### The rule

A shape is demoted iff **the operation immediately feeding the narrowing** is an
f32 arithmetic op whose operands the compiler can prove are binary16-representable.
Whether that demotion is *observable* is a separate question:

- **multiply or fma feeding the narrowing** → `v_fma_mixlo_f16`, and always
  observable in these tests (A2, A7, A8, A12, A14, B1–B4). Double rounding
  (round to binary32, then to binary16) genuinely differs from the single
  rounding the fused form performs.
- **add/sub feeding the narrowing** (A3, A4) → `v_add_f16`, **zero** observable
  disagreements across the entire domain.
- **`x *. x`** (A13) → `v_mul_f16`, and provably harmless: a product of two
  binary16 values needs at most 22 significant bits, so it is *exact* in
  binary32 and the demoted binary16 multiply rounds identically. Not luck — a
  theorem about the operand widths.
- **negation** (A10) → `v_sub_f16`, harmless because negation is exact in every
  format.
- **an op that is not the last before the narrowing** is not fused (A6, A15:
  the multiply feeds an add, and the add is what reaches the narrowing).

A3, A4, A10 and A13 are the important rows. They are **demoted in the machine
code and clean in the numbers**. Had this audit been numeric-only it would have
reported them as unaffected, and the conclusion "only multiplies are at risk"
would have been drawn from a measurement that does not support it.

## Reproduction

```
dune exec sarek-hip/test/test_hip_f16_shapes.exe   # numeric, all 63488 inputs
scripts/f16_shape_isa_audit.sh                     # ISA, gfx1100
```

## What this does not settle

- **gfx1100 and gfx1036 only.** Both are RDNA3/RDNA2-era. CDNA (MI200/MI300) and
  older GCN are untested; the combine is an LLVM AMDGPU ISel pattern, so it
  plausibly applies to all AMDGPU subtargets, but plausibly is not measured.
- **One ROCm version**, the one installed on this machine.
- **Sarek-emittable shapes only.** The audit is closed over what the Sarek DSL
  can currently produce. If f16 ever stops being storage-only — an f16 literal,
  an f16 arithmetic operator, an f16 intrinsic — the enumeration reopens and
  this table is no longer complete.
- **Transcendentals are excluded on purpose** (`sin`, `exp`, `log`, `pow`). Per
  `docs/fp-contraction-policy.md` §1 their rounding is the oracle's own and a
  device is not required to match bit-for-bit, so a disagreement there would not
  be evidence of demotion.

## Incidental defect found

`Sarek_stdlib.Float32.abs_float` cannot be used in a kernel at all. Its native
fallback lowers to `Sarek_cpu_runtime.Float32.abs_float`, which does not exist —
the runtime function is named `abs`:

```
Error: Unbound value Sarek.Sarek_cpu_runtime.Float32.abs_float
Hint:   Did you mean Sarek.Sarek_cpu_runtime.Float32.of_float?
```

The same name mismatch affects `expm1`, `log1p`, `hypot`, `copysign`, `fmod` and
`minus`, all declared in `sarek/Sarek_stdlib/Float32.ml` with no counterpart in
`sarek/interp/Sarek_float32.ml`. Not fixed here — out of scope for #106, and it
wants its own test — but it is why this audit uses `0. -. x` for negation and
`sqrt (x *. x)` rather than `sqrt (abs_float x)`.
