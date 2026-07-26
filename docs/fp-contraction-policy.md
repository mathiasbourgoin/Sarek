# Floating-point contraction policy

_Cross-backend policy for what a Sarek DSL author may rely on when a device
compiler is free to fuse, reassociate or flush floating-point operations._

**Status:** normative for this repository. **Issue:** #116 (absorbs #110, #111);
§6 answers #126 and the HIP row answers #106. **Date:** 2026-07-26.

Three separate defects in this project came from the same place — the gap
between the floating-point semantics we assumed a backend had and the ones it
actually had:

- **HIP/AMDGPU.** An ISel combine fused an f32 multiply into the f32→f16
  narrowing that consumed it (`v_fma_mixlo_f16`) and demoted an f32 add to
  binary16 (`v_add_f16`). **620 of 63488** finite binary16 inputs disagreed with
  the interpreter, and 0 with the barrier in place (quoted from
  `sarek-hip/test/test_hip_f16.ml`; measured on RX 7900 XTX / gfx1100).
  `-ffp-contract=off` did not prevent it: the combine sits below the C-level FP
  controls.

  > **Do not quote "373" for this.** That figure appears in-tree for two *other*
  > populations and the two uses are not consistent with each other:
  > `test_hip_f16.ml` attributes it to inputs where the test's own old
  > f64-intermediate reference disagreed with the f32 one (a harness question,
  > not a device one), while `Hip_rtc.ml` attributes it to inputs on which
  > `-ffp-contract=off` changed the result. One of those two is a slip and it is
  > **not resolved here**. 620 is the barrier/ISel-combine count and is the only
  > one this document uses.
- **CUDA/ptxas.** `ptxas` contracted the multiply that closes `quick_two_sum`
  into *both* of its consumers, collapsing `df64` mul/div from ~9e-15 to
  5.92e-08 — plain float32. It survived four years.
- **Vulkan/GLSL.** Vulkan `mul`/`div` degrade to ~5.8e-08 on Mesa RADV
  (measured here, RX 7900 XTX, Mesa 26.1.4-arch3.1) and on Mesa ANV (quoted,
  Intel UHD Graphics 630). Attributed to a `fma` that is not correctly rounded,
  *not* to contraction — but the two failure modes look identical from the
  outside, and telling them apart took a separate measurement.

This document states, per backend, what the compiler is permitted to contract,
what mechanism (if any) actually prevents it, whether that mechanism is
**verified** or merely **believed**, and what a DSL author may rely on.

---

## 1. The rule

**Sarek's floating-point semantics are IEEE-754 with every operation rounded as
written. `float32` arithmetic rounds to binary32 at every step; `float16`
arithmetic is performed in binary32 and narrowed by an explicit
round-to-nearest-even step; subnormals are not flushed; division and square
root are correctly rounded.**

**The interpreter is the oracle.** `sarek/interp/` evaluates the IR with
float32 rounding emulated at every operation — `Sarek_float32.to_float32`
rounds each result through an `Int32` bit round-trip, and `fma` is
`Float.fma` — and it is the definition of what a Sarek program means. (The
intermediate is OCaml's binary64. For `+`, `-`, `*`, `/` and `sqrt` the double
rounding is benign, because binary64 carries more than `2p+2` bits of a
binary32 operand; for the transcendentals it is a rounding the oracle owns and
the device does not have to match bit-for-bit.) When a device
result and the interpreter result differ, **the device is wrong** — that is the
direction of the obligation, and it is why the interpreter is not merely
"another backend". Every backend gate in this repository is ultimately an
agreement-with-the-interpreter gate.

Three corollaries, each of which has been violated at least once:

1. **Contraction is not a performance knob, it is a semantic change.** Fusing
   `a*b + c` removes a rounding the DSL promised. For ordinary code the
   difference is one ulp; for an error-free transformation (Dekker/Knuth
   TwoSum, TwoProd) it destroys the algorithm outright.
2. **A flag that names the hazard is not a mechanism that prevents it.**
   `-ffp-contract=off` did nothing for the AMDGPU combine. Verify at the
   machine-code level or do not claim it.
3. **A guarantee you have not measured on the device *and toolchain* you are
   claiming it for is not a guarantee.** "Verified on CUDA/PTX" was once
   recorded on the strength of a run through ZLUDA on an AMD GPU. Name the
   device and the toolchain, every time.

---

## 2. Per-backend table

Legend for the evidence column:

| tier | meaning |
|---|---|
| **executed** | ran on the named device and agreed with the interpreter |
| **machine-code** | the emitted ISA was inspected and has the required shape; nothing was executed |
| **compiler-output** | an intermediate representation was inspected (PTX, SPIR-V); the layer below is not constrained |
| **by-construction** | the source hands the compiler nothing it can contract; no flag is relied on |
| **unverified** | believed, documented, or inherited from a vendor's documentation — not measured here |

| backend | what the compiler may contract | what actually prevents it | evidence |
|---|---|---|---|
| **Interpreter** | nothing — it is the oracle | it evaluates the IR directly | executed (any host) |
| **HIP / AMDGPU** | `a*b+c`; **and**, below the FP flags, an f32 multiply or fma fused into an f32→f16 narrowing (`v_fma_mixlo_f16`), plus f32 add/sub/mul/negate demotion to binary16 (`v_add_f16`, `v_sub_f16`, `v_mul_f16`) | two mechanisms, both required: `-ffp-contract=off` forced **last** in the hiprtc option array (`Hip_rtc.base_options` / `hiprtc_options`), *and* the `asm volatile("" : "+v"(x))` opacity barrier on every narrowing's argument (`Sarek_ir_cuda.sarek_f32_barrier_decl`). One barrier covers every affected shape; none needs a different one | **executed + machine-code**, RX 7900 XTX / gfx1100 **and** Raphael iGPU / gfx1036, ROCm hiprtc: all 20 Sarek-emittable f16 expression shapes swept over all 63488 finite binary16 inputs, **0 disagreements as shipped**; removing the barrier breaks 9 of 20 (reproducing the original 620 exactly on the `f16_midround` shape), and disassembly shows demotion opcodes in 3 further shapes that are demoted yet numerically clean — see [`docs/optimization/amdgpu-f16-fusion-shape-audit.md`](optimization/amdgpu-f16-fusion-shape-audit.md) |
| **CUDA / nvrtc (f16 narrowing)** | in principle the same fusion — but NVIDIA has no fused multiply-and-convert-to-f16 instruction to fuse *into* | **nothing Sarek emits.** `ptxas` simply declines to absorb `cvt.rn.f16.f32` | **executed**, GTX 1070 Max-Q / sm_61 / CUDA 12.9 / driver 580.119.02: exhaustive sweep of all 63488 finite binary16 inputs, 0 device/interpreter disagreements, with a liveness control proving the sweep can go red (§7). Also machine-code, CUDA 13.3 host tools, sm_75…sm_121 — see §4. Machine-checked by `test_cuda_f16_sass`, which until this change **self-skipped in CI** for want of `nvdisasm` (§7) |
| **CUDA / nvrtc + PTX (f32 `a*b+c`)** | yes, by default (`-fmad=true` is nvrtc's and ptxas's default, and it applies to PTX input too) | **no flag.** `Sarek_df64` denies the compiler a fusable multiply by routing products through `fma` (`mul_rn`) | executed, GTX 1070 Max-Q / sm_61 / CUDA 12.9 / driver 580.119.02: df64 mul 5.92e-08 → 9.07e-15, div 5.64e-08 → 5.08e-15 |
| **CUDA — subnormal flushing** | `-use_fast_math` / `-ftz=true` would flush binary32 subnormals | `Cuda_nvrtc.check_fp_conformance` **rejects** those options at the only point an option array reaches `nvrtcCompileProgram` | machine-code + test, CUDA 13.3: the hazard is reproduced (`FMUL.FTZ`/`FADD.FTZ` at sm_90) and the guard is proved to fire — see §5 |
| **OpenCL** | `FP_CONTRACT` is on by default in OpenCL C | **no flag** — Sarek passes an empty build-option string. Same `mul_rn`-by-construction defence as CUDA | executed, GTX 1070 Max-Q / NVIDIA OpenCL: mul 5.92e-08 → 9.07e-15, sqrt 2.88e-08 → 9.80e-15 with no OpenCL-specific change (quoted). Re-measured here on RX 7900 XTX / Mesa radeonsi: mul 9.07e-15, div 5.08e-15, sqrt 1.08e-14 |
| **OpenCL / rusticl (f16 narrowing)** | an f32 multiply into the f32→f16 narrowing that consumes it — rounding **once** where the DSL mandates twice. Same defect class as HIP/AMDGPU, same ACO backend | **nothing affordable.** Measured non-fixes, all still 620/63488: `#pragma OPENCL FP_CONTRACT OFF`, a `volatile` local, a `volatile __private` pointer, an `as_half`/`as_ushort` bitcast round-trip, and `convert_half_rte`. HIP's `asm volatile("" : "+v"(x))` **does not compile** here — rusticl goes through SPIR-V, where AMDGPU register constraints do not exist. Only a `volatile __global` round-trip and a `volatile __local` (LDS) round-trip work (both 0/63488), and both cost memory traffic per narrowing; the LDS form additionally needs a workgroup-sized allocation this backend does not control. **Consequence: f16 stays REJECTED in `Sarek_ir_opencl`** | **executed**, 2026-07-26, exhaustive sweep of all 63488 finite binary16 inputs on **two** devices — RX 7900 XTX (navi31) and the integrated Raphael iGPU (gfx1036) — rusticl/radeonsi, DRM 3.64, kernel 7.1.2-3-cachyos. Both report **620/63488**, first divergence at `x=5.68359375` (device 1006.5, interpreter 1006), bit-identical to the HIP figure. Liveness control: the `volatile __global` variant of the same harness reports **0/63488**, so the sweep is proven able to go both red and green. Reproducer: `tools/probes/opencl_f16_contraction_probe.c` |
| **Vulkan / GLSL** | contraction and reassociation of float expressions | `precise` on every float local (`Sarek_ir_glsl.gen_var_decl`), which glslang lowers to SPIR-V `NoContraction` — but on RADV nothing needs preventing: the driver does not contract these shapes even without the decoration | **executed + machine-code**, RX 7900 XTX (RADV NAVI31) and Raphael iGPU (RADV RAPHAEL_MENDOCINO), Mesa 26.1.4-arch3.1: 0 of 7 contraction shapes contracted with or without `precise`, ISA opcode-identical between the two builds, explicit `fma()` controls fused 4/4 — see §6. Decoration emission: compiler-output, glslc 2026.2 + glslangValidator, 18 `NoContraction` with `precise` / 0 without. **Mesa ANV not measured — no Intel GPU on this machine.** Separately, `fma` is not correctly rounded on RADV: df64 mul 5.84e-08 / div 5.86e-08, each the measured worst-case relative error over `test_df64`'s own input set on the named device and driver, not a bound |
| **Metal** | Metal's default compile options enable fast math | **nothing.** `Metal_api` passes a null `MTLCompileOptions`, and `Metal_bindings.mtl_device_new_library_with_source` *ignores its `_options` argument entirely* | unverified — no Apple hardware in this project's CI or on the machine this policy was written on. Treat Metal float results as outside the guarantee |
| **WGSL** | unconstrained | nothing | unverified, untested |
| **Native (OCaml host)** | n/a | n/a | float32 is evaluated at OCaml binary64 precision, so error-free transformations cancel; `Sarek_df64` degrades to ~2^-24 there **by design** |

---

## 3. What a DSL author may rely on

**You may rely on, today:**

- **f32 arithmetic agreeing with the interpreter on HIP/AMDGPU**, including f16
  round-trips — confirmed across **every f16 expression shape the DSL can
  emit**, each swept over the whole finite binary16 domain
  ([`docs/optimization/amdgpu-f16-fusion-shape-audit.md`](optimization/amdgpu-f16-fusion-shape-audit.md)),
  on gfx1100 and gfx1036.
- **The f16 discipline agreeing with the interpreter on NVIDIA**, via the
  CUDA/C (nvrtc) path — executed, GTX 1070 Max-Q / sm_61 / CUDA 12.9 / driver
  580.119.02, exhaustive over all 63488 finite binary16 inputs, **0**
  disagreements (§7). Note this is the **CUDA/C** path: the PTX backend
  refuses kernel-level f16 by design (`#57` slice 2, a located
  `Ptx_codegen_error`), so there is nothing to rely on there yet.
- **`Sarek_df64` meeting its precision contract on CUDA/PTX and OpenCL on
  NVIDIA Pascal, and on OpenCL on AMD** — `sqrt` on CUDA/PTX is now included
  (measured worst-case 8.53e-15 over `test_df64`'s input set on GTX 1070 Max-Q
  / sm_61 / CUDA 12.9; the cause was the `sqrt.approx.f32` seed). These are
  sampled maxima on named devices, not bounds — see the caveat at the top of
  `Sarek_df64`'s PRECISION CONTRACT. The OpenCL and Vulkan `sqrt` residuals
  recorded in that header are still open.
- **No `-use_fast_math` / `-ftz=true` reaching nvrtc**, enforced rather than
  documented (§5) — in **both** the inline (`--ftz=true`) and the separated
  (`--ftz true`) spelling, and fail-closed on a bare `--ftz` whose value cannot
  be resolved. The first version of this guard was spelling-shaped and the
  separated form went straight through it; that hole is closed and both
  spellings are now regression-tested against real `libnvrtc`.

**You may NOT rely on:**

- **Metal or WGSL float semantics at all.** Metal in particular compiles with
  fast math on, unopposed.
- **Vulkan `fma` being correctly rounded.** On Mesa RADV it is not, and
  `Sarek_df64` mul/div is ~5.8e-08 there — a *documented*, unfixed deviation,
  not a bug to rediscover. Note this is a distinct failure from contraction:
  RADV does **not** contract (§6).
- **`precise` protecting you on a driver other than RADV.** On RADV it is inert,
  because the driver does not contract in the first place; that is a measured
  property of Mesa 26.1.4-arch3.1 on two devices, not a bound and not a
  guarantee, and it says nothing about ANV, AMDVLK, proprietary drivers, or a
  future Mesa. The decoration is emitted and correct — it has simply never been
  observed to change a result.
- **f16 on NVIDIA through the PTX backend.** The CUDA/C path is now confirmed
  by execution (above), but `Sarek_ir_ptx` still refuses kernel-level f16
  outright, so "f16 works on CUDA" is true only of CUDA/C.
- **f16 on OpenCL, on any AMD/Mesa stack.** Measured, not assumed: rusticl
  /radeonsi fuses the multiply into the narrowing and **620 of 63488** finite
  binary16 inputs disagree with the interpreter, on both the RX 7900 XTX and
  the Raphael iGPU. Unlike HIP, no affordable barrier exists — see the
  `OpenCL / rusticl (f16 narrowing)` row. `Sarek_ir_opencl` therefore rejects
  f16 at codegen, which is a *deliberate* refusal backed by a measurement, not
  an unimplemented feature. Re-test before enabling it: the blocker is a Mesa
  optimiser behaviour and could change with a Mesa release. That re-test is
  automated — `sarek-opencl/test/test_opencl_f16_tripwire.ml` fails when the
  fusion stops happening, so the refusal cannot quietly outlive its reason.
- **A product you compute yourself staying unfused across a `Sarek_df64`
  boundary.** `[@sarek.module]` bodies are inlined into your kernel, so
  `df64_add_f32 acc (x *. y)` re-creates the exact fusable pattern the library
  removed from its own code. Write `mul_rn x y` (or `two_prod x y`) instead.
  A `let` binding does **not** help — contraction happens far below source
  level. This is the one hazard in this document that lives in *caller* code,
  and no gate in this repository can see it.

**If you are adding a backend or a compiler flag:** an FP-relaxing flag is a
semantic change to every kernel, so it needs an interpreter-agreement argument
before it goes in, not after. If a backend's option array can be influenced
from outside, guard it at the point the array is handed to the compiler — not
at the caller — so that the guard also covers whatever the *next* maintainer
adds. `Cuda_nvrtc.check_fp_conformance` and `Hip_rtc.hiprtc_options` are the
two worked examples.

---

## 4. CUDA f16: the barrier bought nothing *at the narrowing*, and has been removed (#110)

The AMDGPU fix added an opacity barrier around every f32→f16 narrowing. A
PTX-flavoured variant, `asm volatile("" : "+f"(x))`, was added to the non-HIP
branch of `sarek_f32_barrier` at the same time — on the assumption that what
was needed on one target was probably needed on the other.

It was buying nothing **at the narrowing site**, and the reason is not the one
first recorded here. Two corrections, both measured:

**NVVM does NOT erase the block.** The barriered PTX keeps the
`// begin inline asm` / `// end inline asm` marker pair and allocates more
virtual registers (`.reg .f32 %f<9>` against `%f<5>`). What is true, and is the
argument that actually settles the deletion, is that the block contributes
**zero PTX instructions**: the two modules differ only by those comment markers
and virtual-register renumbering, so `ptxas` receives an *identical instruction
stream* either way. The identical cubins are therefore structural, not a CUDA
13.3 coincidence.

**And the barrier is not inert in general** — only at a narrowing. Measured
(CUDA 13.3, sm_90, host-side) on `out[i] = sarek_f32_barrier(a[i]*b[i]) + c[i]`:

| | PTX | SASS (`-fmad=true`, the default) | SASS (`-fmad=false`) |
|---|---|---|---|
| without barrier | `fma.rn.f32` | `FFMA` | `FMUL` + `FADD` |
| with barrier | `mul.f32` + `add.f32` | `FFMA` | `FMUL` + `FADD` |

So at a mul→add site it **is** a real NVVM-level contraction barrier — and
`ptxas -O1` and above **re-contract it back to `FFMA` anyway** (`-O0` and
`--fmad=false` do not), leaving the cubins byte-identical again. The practical
consequence is worth stating plainly, because it is a trap:

> **Do not reach for `sarek_f32_barrier` to fix the caller-side `df64` hazard in
> §3 on NVIDIA.** It protects the PTX and `ptxas` undoes it. `mul_rn` works
> because an `fma` cannot be fused a second time — that is a property of the
> instruction, not of a flag or a barrier.

At the f16 narrowing there is nothing for either level to fuse in the first
place: NVIDIA has no fused multiply-and-convert-to-f16 instruction, which is why
the emitted code is unchanged there under every flag tried.

**Measured for this change** (CUDA 13.3, `nvcc`/`ptxas`/`nvdisasm` V13.3.73,
host-side, **no NVIDIA device**), on the current output of
`Sarek_ir_cuda.generate` for the `f16_midround` kernel, with the `"+f"` asm and
with the branch reduced to `return x;`:

| arch | cubin | SASS |
|---|---|---|
| sm_75, sm_80, sm_86, sm_89, sm_90, sm_100, sm_120, sm_121 | **byte-identical** (`cmp`) on all eight | identical (`diff` on `nvdisasm -c`) |

and the arithmetic stream in the *identity* variant is still unfused:

```
HADD2.F32             R0, -RZ, R2.H0_H0   ; f16 -> f32 widening (exact)
FMUL                  R0, R0, 1.10000002  ; the f32 multiply, intact
F2FP.F16.F32.PACK_AB  R0, RZ, R0          ; separate f32 -> f16 narrowing
HADD2.F32             R0, -RZ, R0.H0_H0
FADD                  R0, R0, 1000        ; the f32 add, NOT demoted
F2FP.F16.F32.PACK_AB  R0, RZ, R0
```

(sm_90; sm_75 emits `F2F.F16.F32` instead of the packed `F2FP` form — both are
single-instruction conversions.)

**Decision: remove the `"+f"` asm; keep the identity function.** The function
itself must stay, because the same generated source is compiled by both hiprtc
and nvrtc under an `#if`. What was deleted is the *pretence*. A call site
reading `__float2half(sarek_f32_barrier(x))` advertised a protection that did
not exist on NVIDIA, and this repository has already paid three times for the
distance between advertised and actual FP semantics.

**Nothing was depending on the code shape.** Two independent checks, both run
after the removal: the machine-code comparison above (byte-identical, so there
is nothing to depend on), and the full `test_cuda_f16_sass` gate, which
re-derives the SASS through generated CUDA → nvrtc → PTX → ptxas → cubin →
nvdisasm and still reports `f32 discipline intact on 7/7 architectures`.

**What actually holds the guarantee on NVIDIA is `ptxas`**, not anything Sarek
emits: hand-written PTX with `mul.f32` / `cvt.rn.f16.f32` and no inline asm at
all produces the same unfused sequence. That is a property of a third-party
assembler, which is exactly the kind of property that must be machine-checked
rather than assumed — `sarek-cuda/test/test_cuda_f16_sass.ml` is that check,
and it carries a positive control (`__hmul`/`__hadd`, which *must* be
classified as fused) so it cannot pass vacuously.

Full per-architecture detail, flag sweep and residual risk:
[`docs/optimization/cuda-f16-fusion-sass-audit.md`](optimization/cuda-f16-fusion-sass-audit.md).

---

## 5. CUDA: `-use_fast_math` / `-ftz=true` are refused (#111)

`-use_fast_math` implies `--ftz=true --prec-div=false --prec-sqrt=false
--fmad=true`. `-ftz=true` alone flushes binary32 subnormals. Unlike the HIP
contraction case — where appending `-ffp-contract=off` *last* neutralises
whatever the caller passed — there is no later flag that undoes subnormal
flushing. So these are **rejected**, not warned about.

`Cuda_nvrtc.check_fp_conformance` raises `Fp_conformance_violation` for
`use_fast_math` (any spelling), `ftz=true|1`, `prec-div=false|0`,
`prec-sqrt=false|0`.

**It screens the option ARRAY, not individual strings, and this is the whole
design.** nvrtc accepts an option and its value as two separate array elements,
so a per-element check is spelling-shaped and misses the separated form. The
first version of this guard did exactly that. Measured through these bindings
against real `libnvrtc.so.13.3`:

| option array | compiles? | `.ftz` in PTX | old guard | current guard |
|---|---|---|---|---|
| `["--ftz=true"]` | — | — | REJECT | REJECT |
| `["--ftz"; "true"]` | **yes** | **yes** | **accept** | REJECT |
| `["-ftz"; "true"]` | **yes** | **yes** | **accept** | REJECT |
| `["--prec-div"; "false"]` | yes | n/a | accept | REJECT |
| `["--ftz"; "false"]` | yes | no | accept | accept |

A value-taking name now consumes the following array element when there is no
`=`, and is **fail-closed**: a bare `--ftz` whose value cannot be resolved is
refused, because a guard that cannot tell what a flag is set to must not assume
the safe answer. (`Hip_rtc.fp_relaxing_option_prefixes` never had this hole — it
matches by prefix on options whose value is always inline.) `fmad=true` **warns** rather than rejects, because it is
nvrtc's default and rejecting it would reject the status quo; contraction is
defeated by construction instead (§2). It is applied in two places: at the
entry of `compile_to_ptx`, before libnvrtc is touched — so the check works on a
host with no CUDA at all — and at `compile_with_string_opts`, the single point
where any option array reaches `nvrtcCompileProgram`. The second placement is
the one that matters: it screens flags this module adds *itself*, so a future
maintainer's hardcoded flag is covered, not just a caller's.

**The hazard is real, measured here** (CUDA 13.3, host-side, no device): the
generated `f16_midround` kernel built for sm_90 has plain `FMUL`/`FADD` by
default and `FMUL.FTZ`/`FADD.FTZ` under `-ftz=true`. `1e-5` is already lane 5
of the `test_hip_f16` input array, so a divergence from the interpreter is
reachable from data the suite already uses.

**The guard has been seen to fire.** `sarek-cuda/test/test_cuda_fp_conformance.ml`
holds four cases, and each was proved red by mutating the thing under test:

| mutation | test that went red | message |
|---|---|---|
| drop `use_fast_math` from the reject list | rejection | `"-use_fast_math" must be refused on the CUDA path; the guard accepted it` |
| match on option name, ignoring its value | acceptance | `"-ftz=false" is legitimate and must be accepted; the guard refused it` |
| remove both `check_fp_conformance` calls from `compile_to_ptx` | end-to-end | `compile_to_ptx accepted -use_fast_math and compiled; the guard did not fire on the real entry point` |
| build the "ftz" SASS variant without `-ftz=true` | hazard control | `control is broken: -ftz=true produced no FMUL.FTZ / FADD.FTZ, so the option this guard rejects has not been shown to change anything on this toolchain` |
| restore the per-element (spelling-shaped) matcher | rejection + end-to-end | the separated form compiles again, with `.ftz` in the PTX — the bypass reproduced on demand |
| drop the option under test from the assembled array | assembled-array | `the assembled array no longer contains the option under test, so this check proves nothing` |

The third mutation is the informative one: with the guard removed,
`compile_to_ptx` compiled `-use_fast_math` **successfully** on this host. The
guard is the only thing standing between a caller and a flushed-subnormal
kernel.

---

## 6. Vulkan/GLSL: `precise` is emitted; RADV has nothing to honour

`Sarek_ir_glsl.gen_var_decl` prefixes every `float`/`double` local with
`precise`. The front end does lower it:

**Measured** (glslc 2026.2 and glslangValidator, SPIRV-Tools 1.4.350.1). On the
Sarek-generated `matrix_mul` compute shader, the SPIR-V carries **2
`NoContraction` decorations** on the accumulation `OpFMul`/`OpFAdd` and **0**
when `precise` is stripped. On the 12-shape probe shader described below, **18**
and **0** respectively — both toolchains agree, which matters because the
runtime path compiles through `glslangValidator`, not `glslc`.

### The question this section used to leave open

Whether a driver *obeys* `NoContraction` is a separate question from whether the
decoration is emitted, and the two things written down in this repository
disagreed about the answer. `Sarek_ir_glsl.ml` said `precise` was added
*because* RADV was observed simplifying error-free transformations without it —
which implies RADV honours it. Campaign notes said the opposite, that Mesa ANV
and RADV ignore it. Neither was backed by a recorded measurement.

**It is now measured, and neither claim survives on RADV.**

### The experiment

`sarek/tests/e2e/test_vulkan_no_contraction.ml`. The same shader is compiled
twice, differing **only** by the expansion of a `#define` that adds or omits
`precise` on the float locals, and both are run on the same device, same
driver, same process.

The discriminator is chosen so a contracted evaluation is not a one-ulp wobble.
With `a = b = 1 + 2^-12` and `c = -(1 + 2^-11)`, all exactly representable in
binary32:

```
exact a*b          = 1 + 2^-11 + 2^-24
fl32(a*b)          = 1 + 2^-11          (2^-24 is exactly half an ulp at 1.0;
                                         ties-to-even drops it)
fl32(fl32(a*b)+c)  = 0                  <- multiply then add
fl32(fma(a,b,c))   = 2^-24 = 5.96e-08   <- contracted
```

Exactly zero against 5.96e-08. There is no tolerance to choose and no way to
read the result ambiguously.

**The contracted target is measured, not modelled.** This document establishes
in the same table that RADV's `fma` is not correctly rounded, so predicting
"what a fused evaluation would return" from an exact-fma model can be wrong on
precisely this driver — and wrong in the dangerous direction, because a
mispredicted target makes a genuinely contracted result match neither candidate
and the shape reads as clean. Every contraction shape is therefore compared
against a **sibling shape that asks the device for the fused value of the same
operands** via an explicit `fma()`, never against an IEEE model. Those sibling
shapes are themselves integrity-checked: an explicit `fma()` that fails to fuse
means the harness is not being handed a fused value at all, and the test fails
rather than reporting. (At these particular operands the device's `fma` does
agree with IEEE — the harness prints that comparison — but nothing depends on
it doing so.)

Twelve shapes are probed, because "honoured" need not be one yes/no: a driver
may contract a mul-add but not a mul-sub, and `precise` forbids reassociation as
well as contraction. The seven contraction-sensitive shapes cover multiply into
add, into subtract from either side, the same expression with and without a
named intermediate, the TwoProd error term, and — the shape `precise` was
actually put in the codegen for — a **loop-carried `acc += a*b` accumulation**
with a non-constant trip count, i.e. the `matrix_mul` inner loop. Two further
shapes probe reassociation.

### The result

**Measured on RX 7900 XTX (RADV NAVI31) and on the Raphael iGPU (RADV
RAPHAEL_MENDOCINO), Mesa 26.1.4-arch3.1, 2026-07-26:**

| | no `precise` | `precise` |
|---|---|---|
| contraction shapes contracted | **0 of 7** | **0 of 7** |
| reassociation shapes reassociated | 0 of 2 | 0 of 2 |
| explicit `fma()` controls fused | 4 of 4 | 4 of 4 |

Identical on both devices. Confirmed at machine-code level: with `RADV_DEBUG=asm`
the emitted RDNA ISA is **opcode-for-opcode identical** between the `precise` and
non-`precise` builds on both devices, and in the *non*-`precise` build — where
the driver is completely free to fuse — the multiply and the add are separate
`v_mul_f32` / `v_add_f32` instructions. The `v_fma_f32` instructions present are
exactly the explicitly-requested `fma()` calls.

**So: RADV does not contract these shapes even when nothing forbids it.**

- The claim that **RADV ignores `NoContraction`** is not supported: the driver
  never produced a contracted result, decoration or no decoration.
- The claim that **`precise` was needed because RADV was contracting** is also
  not supported: with `precise` stripped, RADV still did not contract, and the
  ISA is unchanged.

What is *not* established is that RADV would honour `NoContraction` if it ever
did want to contract. This is an **absence of the hazard on this driver and this
version**, not a demonstration of obedience. The honest status of `precise` on
RADV is therefore: **inert, and free** — it costs nothing (identical ISA) and it
is the correct thing to emit for portability, but on RADV today it is not what
is holding anything up. Keep it; do not credit it.

**Not run: Mesa ANV.** There is no Intel GPU on this machine. The ANV half of
the original disagreement is a **hardware gap**, not a null result — recorded in
§7's still-open list.

### This does not explain the df64 deviation, and never could

`Sarek_df64` mul/div sit at ~5.8e-08 on RADV (re-measured for this document on
RX 7900 XTX, Mesa 26.1.4-arch3.1, `test_df64`, 2026-07-25: mul 5.84e-08, div
5.86e-08, against add 5.33e-15, sub 6.51e-15, sqrt 1.08e-14, and the
interpreter's mul 9.07e-15 / div 5.08e-15 on the same run). Every figure here is
the **measured worst-case relative error over that test's own input set**, on the
named device and driver — a maximum observed, not a bound proved, and agreement
between two such maxima is agreement between summary statistics rather than a
demonstration of element-wise identity. The same shape appears on the Raphael
iGPU.

That deviation was already known to have a different cause, and this measurement
independently confirms it. `Sarek_df64` mul/div sat at ~5.8e-08 both **before and
after** the `mul_rn` contraction barrier; since that barrier works by removing
the fusable multiply, a contraction-shaped failure would have been fixed by it,
and it was not. The recorded cause — RADV's GLSL `fma` is not correctly rounded
— is further supported by extending `mul_rn` into `two_sum`/`quick_two_sum`
*regressing* RADV (add 5.33e-15 → 1.15e-07). The present result closes the loop:
RADV is not contracting at all, so contraction cannot be the explanation.

## 7. What NVIDIA hardware settled, and what is still open

This section was written when there was no NVIDIA GPU on this project's
machines. A GTX 1070 Max-Q (**sm_61 Pascal, CUDA 12.9, driver 580.119.02**)
became available on 2026-07-26 and closed two of its three bullets. Note sm_61
is *below* the sm_75…sm_121 range the host-side sweeps cover, so it is a
genuinely different sample rather than a repeat.

**CLOSED — f16 executed on NVIDIA hardware.** The interpreter-agreement claim
for f16 is no longer HIP-only. Exhaustive over all **63488** finite binary16
inputs, on the `f16_midround` kernel (the one whose mid-expression narrowing is
the whole point of the discipline):

| check | result |
|---|---|
| interpreter vs CPU reference | 0 / 63488 |
| CUDA vs CPU reference | 0 / 63488 |
| **CUDA == interpreter, bit-identical** | **0 / 63488** |
| liveness control (`cuda(scale)` vs `interp(midround)`) | **63085 / 63488** — the sweep can go red |

That also answers the narrow hardware question this bullet used to name: the
conversion **does** round to nearest-even on ties, since the domain sweep
contains the ties and none disagreed. At sm_61 the narrowing is `F2F.F16.F32`
(the unpacked form §4 notes sm_75 emits) rather than the packed `F2FP`.

A zero here would be worthless without the liveness control — see the last row.
Reporting an exhaustive agreement without showing the harness can produce a
nonzero is the failure mode this table is shaped to avoid.

**CLOSED — the driver JIT agrees with offline `ptxas`.** These runs load PTX
through `cuModuleLoadData`, so the assembling compiler was the *driver's* ptxas
(580.119.02), not `/opt/cuda/bin/ptxas`. It produced interpreter-identical f16
results and a `Sarek_df64` that meets its contract, so the driver JIT is no
longer an unconstrained variable — on this driver and this architecture.

**CLOSED — a second toolkit version.** These measurements are **CUDA 12.9**,
between CI's 12.6 and this document's 13.3. The `ptxas`-declines-to-fuse
property now has two independent toolkit samples rather than one.

**Independently confirmed: the removed CUDA barrier really was inert.** §4
concluded that from byte-identical cubins with no device. Executing the
barrier-free codegen on real hardware gives the same 0/63488 — and while the
barrier still existed, neutralising it left the sm_61 SASS **byte-identical**
(`F2F.F32.F16.RZ / FMUL32I / F2F.F16.F32 / F2F.F32.F16.RZ / FADD /
F2F.F16.F32`, every mandated rounding its own instruction). Two methods, two
architectures ranges, same answer.

**Still open** (the first is not an NVIDIA gap, but it is a hardware gap and
this is where they are collected):

- **No Intel GPU: the ANV half of §6 is unrun.** The `precise` /
  `NoContraction` question was originally disputed for **both** Mesa RADV and
  Mesa ANV. §6 settles RADV by measurement. ANV is **not** measured and cannot
  be measured here — the experiment requires executing on the driver in
  question. Nothing in §6 licenses any statement about ANV in either direction,
  and the campaign note claiming ANV ignores `NoContraction` remains unverified.
  One run closes it on any Intel box:
  `dune build @e2e-gpu` enumerates every Vulkan device present and needs no
  configuration. AMDVLK and the proprietary AMD Vulkan driver are unmeasured for
  the same reason — different SPIR-V consumers on the very same GPU.
- **f16 on the PTX backend.** `Sarek_ir_ptx` refuses kernel-level f16 (`#57`
  slice 2). Everything above is the **CUDA/C** path.
- **One GPU, one architecture.** sm_61 has no tensor cores, no bf16 and no FP8,
  so nothing here constrains `mma`, bf16 or FP8 contraction, and no f16
  *performance* claim can be made from it (f16 runs at 1/64 f32 rate on GP104).
  Those need Ampere or newer.
- **Still one driver.** The driver-JIT result above is 580.119.02 on Pascal.
  **If the SASS gate ever fails on CI at 12.6 while passing at 13.3, that
  difference is still the finding.**

  > **This was not actually being checked.** `nvdisasm` ships in `cuda-nvdisasm`,
  > not in `cuda-nvcc`, and it was **absent from the built CI image** — verified
  > by running `command -v nvdisasm` inside it, where `ptxas` and `nvcc` both
  > resolve and `nvdisasm` does not. So `test_cuda_f16_sass` self-skipped in the
  > only place 12.6 is exercised, and nothing could have surfaced a 12.6-vs-13.3
  > divergence. Fixed in this change: `ci/Dockerfile` installs
  > `cuda-nvdisasm-12-6` and `ci/assert-toolchain.sh` now fails the build when
  > `nvdisasm` is missing or cannot disassemble a freshly assembled probe cubin.
  > Until an image built with that change has run, **treat every "the gate
  > checks this" statement in §4 as holding at CUDA 13.3 only.**
- **`FMUL.FTZ` was observed, subnormal divergence was not.** §5 shows the flag
  changes the instruction; it does not show a wrong answer, because that
  requires running the kernel.
- **`Sarek_df64`'s `sqrt` residual — CUDA/PTX resolved, OpenCL and Vulkan still
  open.** The one-line experiment this bullet used to describe as needing an
  NVIDIA device has been run. On CUDA/PTX the `sqrt.approx.f32` Newton seed was
  the cause: emitting `sqrt.rn.f32` moves the measured worst-case relative error
  from 1.42e-14 to 8.53e-15 in `test_df64` and from 1.68e-14 to 8.87e-15 in
  `test_real64` (GTX 1070 Max-Q / sm_61 / CUDA 12.9), each figure being the max
  over that test's own input set. Both coincide with the interpreter's figure
  for the same inputs — agreement between summary statistics, not a
  demonstration of element-wise identity. The OpenCL residual (1.81e-14) is the **same
  bug class in a different backend** — `clBuildProgram` is called with no
  options and OpenCL's default permits a 3-ulp `sqrt`; adding
  `-cl-fp32-correctly-rounded-divide-sqrt` moves it to 8.87e-15, measured on the
  same device, tracked as #136. The **Vulkan** residual (1.68e-14 on NVIDIA,
  while Intel UHD 630 passes at 1.17e-14) has no established cause — that one is
  still "do not promote a hypothesis to a cause".
- **Metal is entirely unverified** (no Apple hardware), and it is the one
  backend currently compiled with fast math on.

---

## 8. Where the mechanisms live

| file | what it carries |
|---|---|
| `sarek-hip/Hip_rtc.ml` | `-ffp-contract=off` forced last; warning for caller-supplied fast-math options |
| `sarek/codegen/Sarek_ir_cuda.ml` | `sarek_f32_barrier` — load-bearing on HIP, a documented identity on NVIDIA |
| `sarek-cuda/Cuda_nvrtc.ml` | `check_fp_conformance` — rejects subnormal-flushing / approximate-div options |
| `sarek/Sarek_df64/Sarek_df64.ml` | the `mul_rn` contraction barrier, its per-backend precision table, and the caller-side hazard |
| `sarek/codegen/Sarek_ir_glsl.ml` | `precise` on float locals → SPIR-V `NoContraction` |
| `sarek-cuda/test/test_cuda_f16_sass.ml` | the f16 SASS gate (with positive control) |
| `sarek-cuda/test/test_cuda_fp_conformance.ml` | the nvrtc FP-option guard and its hazard control |
| `sarek-hip/test/test_hip_rtc_options.ml` | proves `-ffp-contract=off` stays last whatever the caller passes |
| `sarek/tests/e2e/test_df64.ml` | the per-backend precision measurement this policy quotes |
| `sarek/tests/e2e/test_vulkan_no_contraction.ml` | the §6 experiment: `precise` vs not, same device/driver/run, contracted targets taken from the device's own `fma` (`e2e-gpu` alias) |
| `sarek-hip/test/test_hip_f16_shapes.ml` | every f16 expression shape swept over all 63488 finite binary16 inputs, with a barrier-removed control that must go red (`e2e-hip` alias) |
| `scripts/f16_shape_isa_audit.sh` | the ISA half of that audit — catches shapes demoted in machine code but numerically clean |
