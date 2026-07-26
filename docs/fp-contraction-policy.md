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
- **Vulkan/RADV, f16.** RADV's ACO backend absorbs an f32→f16 narrowing into
  whatever arithmetic feeds it, and `precise`/`NoContraction` does not stop it —
  the decoration is emitted, and the emitted ISA is byte-identical with and
  without it. **2912 of 63488** finite binary16 inputs disagree with the
  interpreter on a single narrowing, **5075** on a two-narrowing expression.
  Third front end onto the same ACO backend as HIP and rusticl, and a wider
  combine than either. f16 stays refused on this backend (§2, §6).

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
| **OpenCL** | `FP_CONTRACT` is on by default in OpenCL C, and no build option turns it off | for contraction, **no flag** — same `mul_rn`-by-construction defence as CUDA. For div/sqrt, `Opencl_fp.conformance_options` requests `-cl-fp32-correctly-rounded-divide-sqrt`, **gated** on `CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT` in the device's `CL_DEVICE_SINGLE_FP_CONFIG`; `Opencl_fp.check_fp_conformance` **rejects** the relaxing `-cl-*` options at `Opencl_api.Program.build`, the single point an option string reaches `clBuildProgram` (§9) | executed, GTX 1070 Max-Q / NVIDIA OpenCL: mul 5.92e-08 → 9.07e-15, sqrt 2.88e-08 → 9.80e-15 with no OpenCL-specific change (quoted). Re-measured here on RX 7900 XTX / Mesa radeonsi: mul 9.07e-15, div 5.08e-15, sqrt 1.08e-14 |
| **OpenCL / rusticl (f16 narrowing)** | an f32 multiply into the f32→f16 narrowing that consumes it — rounding **once** where the DSL mandates twice. Same defect class as HIP/AMDGPU, same ACO backend | **nothing affordable.** Measured non-fixes, all still 620/63488: `#pragma OPENCL FP_CONTRACT OFF`, a `volatile` local, a `volatile __private` pointer, an `as_half`/`as_ushort` bitcast round-trip, and `convert_half_rte`. HIP's `asm volatile("" : "+v"(x))` **does not compile** here — rusticl goes through SPIR-V, where AMDGPU register constraints do not exist. Only a `volatile __global` round-trip and a `volatile __local` (LDS) round-trip work (both 0/63488), and both cost memory traffic per narrowing; the LDS form additionally needs a workgroup-sized allocation this backend does not control. **Consequence: f16 stays REJECTED in `Sarek_ir_opencl`** | **executed**, 2026-07-26, exhaustive sweep of all 63488 finite binary16 inputs on **two** devices — RX 7900 XTX (navi31) and the integrated Raphael iGPU (gfx1036) — rusticl/radeonsi, DRM 3.64, kernel 7.1.2-3-cachyos. Both report **620/63488**, first divergence at `x=5.68359375` (device 1006.5, interpreter 1006), bit-identical to the HIP figure. Liveness control: the `volatile __global` variant of the same harness reports **0/63488**, so the sweep is proven able to go both red and green. Reproducer: `tools/probes/opencl_f16_contraction_probe.c` |
| **Vulkan / RADV (f16 narrowing)** | an f32→f16 narrowing absorbs whatever arithmetic feeds it (`v_fma_mixlo_f16`) — the multiply, and also the f32 **add**: the plain two-narrowing kernel compiles to a *single* fused instruction, one rounding where the DSL mandates three. Same ACO backend as HIP and rusticl, reached through a third front end, but a **wider** combine than either | **nothing affordable, and `precise` is not it.** `precise` → SPIR-V `NoContraction` IS honoured (it keeps the f32 multiply as its own `v_fma_mix_f32`) and still leaves 2912/63488, because absorbing a *conversion* is a different combine from contracting `a*b+c`. An f16 bitcast round-trip changes nothing. A `volatile` SSBO round-trip on the f32 intermediates makes ACO drop the intermediate narrowing **entirely** instead (4774/63488). Only forcing the f16 *bit pattern* through global memory works (0/63488), at a global round-trip per narrowing into a scratch buffer this backend does not control. **Consequence: f16 stays REJECTED in `Sarek_ir_glsl`** | **executed**, 2026-07-26, exhaustive sweep of all 63488 finite binary16 inputs on **two** devices — RX 7900 XTX (**RADV NAVI31**) and the integrated Raphael iGPU (**RADV RAPHAEL_MENDOCINO**) — Mesa 26.1.4-arch3.1, Vulkan 1.4.354. Both report identical counts: **2912/63488** on `f16(x*1.1)` (plain and `precise` alike), **5075/63488** on `f16(f16(x*1.1)+1000)` plain, **4776/63488** with `precise`. Calibration: the same host oracle reproduces the independently measured **620** on the HIP/OpenCL kernel shape, and the barriered kernel reports **0/63488**, so the sweep is proven able to go both red and green. Gate: `sarek-vulkan/test/test_vulkan_f16_tripwire.ml` |
| **OpenCL / pocl on x86 (f16 narrowing)** | in principle the same fusion — but nothing in this stack performs it | **nothing needed.** The naive narrowing already round-trips through binary16 exactly, so the barrier that rusticl requires is unnecessary here | **executed on CI**, 2026-07-26, quoted device `AMD EPYC 7763 64-Core Processor` under pocl on a GitHub-hosted runner: exhaustive sweep of all 63488 finite binary16 inputs, **0** disagreements between the naive and `volatile __local`-barriered narrowings. Observed as a CI failure of `test_opencl_f16_tripwire` before that test was scoped, i.e. the number was produced by a harness that was at the time *trying* to find a difference — so it is a null with the sweep demonstrably live. **This is what localises the defect:** the same source, swept the same way, fuses on ACO and does not fuse here, so the locus is *the ACO backend*, not *OpenCL*. That in turn is the second independent reason to read rusticl and HIP/AMDGPU as one bug seen through two front ends rather than two bugs. Guarded by `test_opencl_f16_tripwire`'s locus check, which fails if any non-ACO implementation is found to fuse |
| **Vulkan / GLSL** | contraction and reassociation of float expressions | `precise` on every float local (`Sarek_ir_glsl.gen_var_decl`), which glslang lowers to SPIR-V `NoContraction` — but on RADV nothing needs preventing *for these shapes*: the driver does not contract them even without the decoration. It is **not** the decoration that is protecting them; RADV was separately observed ignoring `NoContraction` on a combine it does want to perform (§6, f16 narrowing) | **executed + machine-code**, RX 7900 XTX (RADV NAVI31) and Raphael iGPU (RADV RAPHAEL_MENDOCINO), Mesa 26.1.4-arch3.1: 0 of 7 contraction shapes contracted with or without `precise`, ISA opcode-identical between the two builds, explicit `fma()` controls fused 4/4 — see §6. Decoration emission: compiler-output, glslc 2026.2 + glslangValidator, 18 `NoContraction` with `precise` / 0 without. **Mesa ANV not measured — no Intel GPU on this machine.** Separately, `fma` is not correctly rounded on RADV: df64 mul 5.84e-08 / div 5.86e-08, each the measured worst-case relative error over `test_df64`'s own input set on the named device and driver, not a bound |
| **Metal** | contraction of `a*b+c` — **measured, and NOT preventable by any compile option**; separately, both math defaults are the fast one (`mathMode = MTLMathModeFast`, `mathFloatingPointFunctions = ...Fast`, read from a fresh `MTLCompileOptions`) | **two mechanisms, both required**: `#pragma METAL fp contract(off)` in every generated kernel (`Sarek_ir_metal.metal_fp_contract_pragma`) for contraction, *and* `mathMode = Safe` + `mathFloatingPointFunctions = Precise` in `Metal_bindings.mtl_compile_options_conformant` for math-function accuracy (falling back to the deprecated `fastMathEnabled = NO` before macOS 15) | **executed**, Apple M4 / macOS 15.6.1 (24G90) / Apple clang 17.0.0: on the 8773 of 65536 elements where the device's own `fma` differs from the separately-rounded value, `a*b+c` is contracted 8773/8773 under every compile-option setting **including `mathMode=Safe`**, and 0/8773 with the pragma. Options are honoured (16017 and 22135 of 65536 math-function results change), so Metal is not in rusticl's ignore-it class. **Interpreter agreement now executed on the same device**: `test_df64` and `test_real64` PASS on every op and reproduce the interpreter's figures exactly (mul 9.07e-15, add 5.33e-15, sub 6.51e-15, div 5.08e-15, sqrt 8.53e-15) — sampled maxima over each test's input set, not bounds, and agreement between summary statistics rather than element-wise identity. f16 and subnormals unprobed; two record/variant kernels do not compile at all (§10.11) |
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
- **`Sarek_df64` meeting its precision contract on Metal, for f32 scalar
  kernels** — executed on an Apple M4 / macOS 15.6.1 (24G90) / Apple clang
  17.0.0: `test_df64` and `test_real64` PASS on every op and reproduce the
  interpreter's worst-case figures exactly (§10.7). Read the scope narrowly:
  one device, one OS, summary maxima rather than element-wise identity, no f16,
  no subnormals, and **not** kernels using records or variants — those do not
  compile on Metal at all (§10.11).
- **No `-use_fast_math` / `-ftz=true` reaching nvrtc**, enforced rather than
  documented (§5) — in **both** the inline (`--ftz=true`) and the separated
  (`--ftz true`) spelling, and fail-closed on a bare `--ftz` whose value cannot
  be resolved. The first version of this guard was spelling-shaped and the
  separated form went straight through it; that hole is closed and both
  spellings are now regression-tested against real `libnvrtc`.

**You may NOT rely on:**

- **WGSL float semantics at all.**
- **Metal f16, Metal subnormals, and Metal record/variant kernels.** Metal
  **f32 arithmetic** has moved OFF this list: `test_df64` and `test_real64`
  agree with the interpreter on every op on an Apple M4 / macOS 15.6.1 / Apple
  clang 17.0.0 (§10.7), with contraction defeated by
  `#pragma METAL fp contract(off)` in every generated kernel (§10.5). What is
  still NOT covered: f16 (never probed on Metal), subnormals (never probed),
  element-wise identity (the agreement is between summary maxima), any device
  or OS other than that one, and kernels using **records or variants**, which
  do not compile on Metal at all (§10.11).
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
  That tripwire asserts **only on ACO devices** (it keys on `"ACO"` in the
  OpenCL device name, which is where Mesa reports its shader compiler) and
  **skips visibly** elsewhere, naming the devices it rejected. Scoping it was
  not cosmetic: unscoped, it failed on a pocl/x86 CI runner that correctly does
  not fuse — a false positive on a blocking gate, which is the polarity that
  gets assertions deleted. On a runner with no ACO device the tripwire is
  therefore a no-op, and the C probe is the only artifact CI exercises; the
  tripwire is in practice a developer-workstation gate.
- **f16 on Vulkan/GLSL, on Mesa RADV.** Measured, not assumed, and *worse* than
  the OpenCL case: **2912 of 63488** finite binary16 inputs disagree with the
  interpreter on a single narrowing, and **5075** on the two-narrowing shape,
  on both the RX 7900 XTX (RADV NAVI31) and the Raphael iGPU (RADV
  RAPHAEL_MENDOCINO). `Sarek_ir_glsl` therefore rejects f16 at codegen — a
  *deliberate* refusal backed by a measurement, not an unimplemented feature.
  **Do not reason from `precise`.** This backend already emits `precise` on
  every float local, and #106/#126 measured RADV not to contract 7 f32 shapes;
  neither is evidence about this combine, and the `precise` variant still
  disagrees on 2912 inputs. Re-test before enabling: the blocker is a Mesa
  optimiser behaviour and could change with a Mesa release. That re-test is
  automated — `sarek-vulkan/test/test_vulkan_f16_tripwire.ml` fails when the
  fusion stops, and separately fails if `precise` starts working, so neither
  half of the refusal can quietly outlive its reason. It asserts **only on RADV
  devices** (it keys on `"RADV"` in the Vulkan device name — driver identity,
  not device model, because the Raphael iGPU's name reads like a CPU and
  reproduces the defect identically) and **skips visibly** elsewhere, naming
  the devices it rejected.
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

What is *not* established by the experiment above is that RADV would honour
`NoContraction` if it ever did want to contract. That was left open here, and
**#57 slice 2b closed it — negatively.**

### RADV *does* want to contract somewhere, and there it ignores `NoContraction`

Measured 2026-07-26, RX 7900 XTX (RADV NAVI31) and the Raphael iGPU (RADV
RAPHAEL_MENDOCINO), Mesa 26.1.4-arch3.1. The shape is `float16_t(x * 1.1)` — a
combine the twelve shapes above do not probe, because it is not `a*b+c`: it is a
multiply absorbed by the **conversion** that consumes it.

| | no `precise` | `precise` |
|---|---|---|
| `NoContraction` decorations in the SPIR-V (`glslangValidator` → `spirv-dis`, reproduced by `tools/probes/vulkan_f16_narrowing_probe.sh`) | **0** | **1**, on the `OpFMul` |
| emitted RDNA ISA | `v_fma_mixlo_f16 0xcccd, v1, neg(0)` | **byte-identical** |
| disagreements with the interpreter, all 63488 finite binary16 inputs | 2912 | **2912** |
| matches a single-rounding model | exactly, 0/63488 | exactly, 0/63488 |

The decoration is emitted, by the same `glslangValidator` the Vulkan runtime
path actually uses; the driver produces opcode-for-opcode identical machine code
either way; and that machine code performs one rounding where the SPIR-V asks
for two. **That is a `NoContraction` violation, observed.**

So the honest status of `precise` on RADV is *not* "inert, and free". It is:

- **inert** for the seven f32 `a*b+c`-family shapes above — RADV does not
  contract them with or without it, so nothing is being held up;
- **ignored** for the multiply-into-narrowing combine, where RADV *does*
  contract and the decoration does not stop it.

Keep emitting it — it is correct for portability and costs nothing. Do not
credit it with a guarantee on RADV, in either direction: it is not what makes
f32 safe here, and it is not enough to make f16 safe here. The consequence for
f16 is the `Vulkan / RADV (f16 narrowing)` row in §2 and the refusal in
`Sarek_ir_glsl`; the consequence for f32 is that the phrase "contraction-safe by
front-end declaration" describes what Sarek *emits*, never what RADV *obeys*.

**Generalisable, and the reason this was missed for a whole slice:** the twelve
shapes were chosen to probe *contraction* as the term is normally used —
`a*b+c`. A conversion is also an operation that can absorb its operand, and it
is not in that family. A null result is only as broad as its shape catalogue.

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
  same device. **Fixed in backlog #136 (§9)** — but the fix could not be
  re-measured here: the red does not reproduce on this machine's OpenCL devices
  (rusticl/radeonsi reports `sqrt` 9.68e-15, a PASS) and rusticl ignores FP
  build options outright. The **Vulkan** residual (1.68e-14 on NVIDIA,
  while Intel UHD 630 passes at 1.17e-14) has no established cause — that one is
  still "do not promote a hypothesis to a cause".
- **f16 on Vulkan/GLSL: what slice 2b did NOT close.** The refusal is measured
  and gated, but three things remain unmeasured and must not be read into it.
  (a) **Only RADV.** No non-RADV Vulkan implementation has been measured for
  this combine — not ANV, not AMDVLK, not NVIDIA, not lavapipe — so "Vulkan
  fuses" is not a claim this repository makes; "RADV fuses" is. The tripwire
  deliberately carries no "non-RADV does not fuse" cross-check for that reason,
  unlike its OpenCL sibling, which has pocl data behind it.
  (b) **The `shaderFloat16` device feature is not enabled.** Vulkan requires it
  before a shader may use the SPIR-V `Float16` capability;
  `Vulkan_api_device` chains no feature structs beyond core
  `VkPhysicalDeviceFeatures`, and RADV accepts the shaders anyway. The
  measurement stands — the defect is visible in the ISA, and the barriered
  control returns bit-exact results on the same un-enabled path — but that
  plumbing is real work that enabling f16 here would have to do first, and it
  is not done.
  (c) **No Sarek-generated shader was involved.** The tripwire compiles raw
  GLSL, because `Sarek_ir_glsl` refuses f16; it measures the driver, not the
  codegen. If the refusal is ever lifted, the codegen's own output needs its own
  exhaustive interpreter-agreement gate — this one does not substitute.
- **Metal: no longer unverified, and the measurement changed the fix.** An
  Apple M4 became available on 2026-07-26. Measured there (macOS 15.6.1, Apple
  clang 17.0.0): both of Metal's math defaults are the fast one, the compile
  options ARE honoured — and **none of them stops contraction**, which needs a
  source pragma instead. Full account and the three things still open in §10.
  Interpreter agreement has since been executed on that M4: `test_df64` and
  `test_real64` PASS on every op and reproduce the interpreter's figures
  exactly (§10.7), so Metal **f32** has left the §3 "may NOT rely on" list.
  Metal f16, subnormals, and record/variant kernels have not — the last of
  those because they do not compile (§10.11).

---

## 8. Where the mechanisms live

| file | what it carries |
|---|---|
| `sarek-hip/Hip_rtc.ml` | `-ffp-contract=off` forced last; `-fhip-fp32-correctly-rounded-divide-sqrt` and `-fno-gpu-flush-denormals-to-zero` set explicitly (§9); warning for caller-supplied fast-math options |
| `sarek-opencl/Opencl_fp.ml` | the OpenCL build-option string: the capability-gated correctly-rounded-div/sqrt request, and `check_fp_conformance` rejecting the relaxing `-cl-*` options |
| `sarek-opencl/test/test_opencl_fp_conformance.ml` | that guard and its two anti-vacuity controls (§9.5) |
| `sarek/codegen/Sarek_ir_metal.ml` | `metal_fp_contract_pragma` — the ONLY measured Metal contraction defence (§10.5) |
| `sarek-metal/Metal_bindings.ml` | `mtl_compile_options_conformant` — `mathMode=Safe` + `fpFunctions=Precise`, with the pre-macOS-15 fallback (§10.6) |
| `tools/probes/opencl_build_options_probe.c` | which OpenCL build options a stack accepts, and whether they do anything — with plumbing and FP liveness controls |
| `tools/probes/metal_math_mode_probe.m` | Metal's real defaults, and whether its options change results (M4) |
| `tools/probes/metal_contraction_barrier_probe.m` | the Metal contraction barrier sweep, and the deprecated/modern API equivalence |
| `sarek/codegen/Sarek_ir_cuda.ml` | `sarek_f32_barrier` — load-bearing on HIP, a documented identity on NVIDIA |
| `sarek-cuda/Cuda_nvrtc.ml` | `check_fp_conformance` — rejects subnormal-flushing / approximate-div options |
| `sarek/Sarek_df64/Sarek_df64.ml` | the `mul_rn` contraction barrier, its per-backend precision table, and the caller-side hazard |
| `sarek/codegen/Sarek_ir_glsl.ml` | `precise` on float locals → SPIR-V `NoContraction`; and the measured f16 refusal |
| `sarek-vulkan/test/test_vulkan_f16_tripwire.ml` | the RADV f16-fusion tripwire, its calibration and its green control |
| `tools/probes/vulkan_f16_narrowing_probe.sh` | standalone reproducer for the emitted-but-ignored `NoContraction`; needs no device |
| `sarek-cuda/test/test_cuda_f16_sass.ml` | the f16 SASS gate (with positive control) |
| `sarek-cuda/test/test_cuda_fp_conformance.ml` | the nvrtc FP-option guard and its hazard control |
| `sarek-hip/test/test_hip_rtc_options.ml` | proves `-ffp-contract=off` stays last whatever the caller passes, that the two §9 conformance defaults are set explicitly, and that the relaxing-option warning list covers what was measured to matter (with an anti-vacuity control) |
| `sarek/tests/e2e/test_df64.ml` | the per-backend precision measurement this policy quotes |
| `sarek/tests/e2e/test_vulkan_no_contraction.ml` | the §6 experiment: `precise` vs not, same device/driver/run, contracted targets taken from the device's own `fma` (`e2e-gpu` alias) |
| `sarek-hip/test/test_hip_f16_shapes.ml` | every f16 expression shape swept over all 63488 finite binary16 inputs, with a barrier-removed control that must go red (`e2e-hip` alias) |
| `scripts/f16_shape_isa_audit.sh` | the ISA half of that audit — catches shapes demoted in machine code but numerically clean |

---

## 9. OpenCL and HIP: what an unset option was choosing (backlog #136)

`Opencl_api.Program.build` called `clBuildProgram` with an **empty option
string**, and every caller in the tree passed nothing. An empty option string
is not "no policy". It is a policy chosen by omission — whatever each vendor
decided its default should be — and OpenCL's default permits a `sqrt` of up to
3 ulp and a divide of up to 2.5 ulp, against §1's requirement that both are
correctly rounded.

That is the same shape as the PTX `sqrt.approx.f32` defect (§7): a faster, less
accurate default selected by passing nothing.

### 9.1 The red, and where it lives

**The red is real and it is quoted, not re-measured.** On a GTX 1070 Max-Q
(sm_61, CUDA 12.9, driver 580.119.02), `test_real64`'s df64 fallback measured
OpenCL `sqrt` at **1.81e-14** against a 1.42e-14 tolerance — a FAIL — and
building with `-cl-fp32-correctly-rounded-divide-sqrt` moved it to **8.87e-15**,
a PASS coinciding with the interpreter's figure for the same input set.
Measured during #298 on that device.

**That device is not on this machine, and the red does not reproduce here.**
Measured 2026-07-26, `test_real64` on rusticl/radeonsi 26.1.4-arch3.1: OpenCL
`sqrt` 9.68e-15 on both the RX 7900 XTX and the Raphael iGPU — already passing.
Say so plainly rather than implying the fix was observed to move anything
locally: it was not, and it could not have been.

### 9.2 The instrument, and why its cost figures are worthless here

`tools/probes/opencl_build_options_probe.c`, 2^20 inputs per device, carries two
controls and they disagree:

| variant | sqrt vs correctly-rounded | bit-differs from baseline |
|---|---|---|
| baseline (empty option string) | ≤1 ulp | — |
| `-cl-fp32-correctly-rounded-divide-sqrt` (the fix) | ≤1 ulp | **0 / 1048576** |
| `-cl-fast-relaxed-math` (**FP liveness control**) | ≤1 ulp | **0 / 1048576** |
| `-DSAREK_PROBE_SCALE=…` (**plumbing control**) | 3 ulp | 1048576 / 1048576 |

The plumbing control **passes**: the option string really does reach rusticl's
compiler and the comparison can go non-zero. The FP liveness control **fails**:
even `-cl-fast-relaxed-math` changes nothing.

**So on this stack "the flag changed nothing" is indistinguishable from "the
flag was discarded", and no accuracy or cost conclusion may be drawn from these
devices.** The probe prints −3.0% and +0.2% for the two devices; those are
run-to-run noise around an unchanged kernel and **must not be reported as a
measured cost**. The honest statement of cost on this machine is: *not
measurable here*. The CUDA analogue (~12% on `bench_nbody` at sm_61 for
`sqrt.rn.f32`) is a different backend on a different device and does not
transfer.

### 9.3 Why the flag is capability-gated

The OpenCL spec permits `-cl-fp32-correctly-rounded-divide-sqrt` only when the
device's `CL_DEVICE_SINGLE_FP_CONFIG` contains
`CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT`; otherwise `clBuildProgram` returns
`CL_INVALID_BUILD_OPTIONS`. An unconditional flag would therefore fail **every
kernel build** on every device lacking the capability — a total failure, worse
than the numerical one it fixes.

**This machine would not have caught that.** Measured: both local devices report
`CL_DEVICE_SINGLE_FP_CONFIG = 0x6` (`INF_NAN | ROUND_TO_NEAREST`, with neither
`CL_FP_DENORM` nor `CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT`) and **both accept the
flag with `CL_SUCCESS` anyway** — rusticl departing from the spec in the
permissive direction. The gate exists because the local stack cannot be trusted
to reveal its absence. (Control: the deliberately invalid
`-cl-this-option-does-not-exist` *is* refused, so "ACCEPTED" is not the probe's
answer to everything.)

### 9.4 The full audit — every default now chosen rather than inherited

| option | decision | why |
|---|---|---|
| `-cl-fp32-correctly-rounded-divide-sqrt` | **set**, gated on the device bit | default off ⇒ 3-ulp `sqrt`, 2.5-ulp divide; `Sarek_df64`'s Newton/Karp step squares its seed error and has no margin for it |
| `-cl-denorms-are-zero` | **never passed; refused from callers** | asks the device to flush binary32 subnormals, which §1 forbids. Its default (absent) is already conformant |
| `-cl-fast-relaxed-math` | **never passed; refused** | implies unsafe-math + finite-math, and permits `native_*` substitution |
| `-cl-unsafe-math-optimizations` | **never passed; refused** | permits reassociation, which destroys an error-free transformation outright (§1 corollary 1) |
| `-cl-finite-math-only` | **never passed; refused** | assumes no NaN/Inf; the oracle assumes no such thing |
| `-cl-no-signed-zeros` | **never passed; refused** | discards the sign of zero, which `Sarek_df64` renormalisation relies on |
| `-cl-mad-enable` | **never passed; refused** | the contraction hazard by name |
| `-cl-single-precision-constant` | **never passed; refused** | silently demotes double literals |
| `-cl-opt-disable` | **allowed** from callers | conservative; cannot relax FP semantics |
| `FP_CONTRACT` | **not addressable** | on by default in OpenCL C and **no build option turns it off**. `#pragma OPENCL FP_CONTRACT OFF` was measured on this stack and does not work (`Sarek_ir_opencl`'s `TFloat16` rejection: 620/63488 survive it). Contraction remains defeated **by construction** via `mul_rn`, as on CUDA |

**Subnormals are flushed here regardless.** Both local devices lack
`CL_FP_DENORM`, so f32 subnormals do not survive on this hardware whatever the
build options say. That is a device property Sarek cannot correct; it is
recorded, not fixed.

Options are **refused** rather than countered because — unlike HIP, where
appending `-ffp-contract=off` last neutralises whatever the caller passed —
OpenCL has no build option that undoes `-cl-fast-relaxed-math`. Same reasoning
as §5.

### 9.5 The guard has been seen to fire

`sarek-opencl/test/test_opencl_fp_conformance.ml`. Every case was proved red by
mutating the thing under test:

| mutation | test that went red |
|---|---|
| drop `-cl-fast-relaxed-math` from the reject list | rejection, token matching, assembly, **and** the real-device refusal |
| ignore the capability gate (pass the flag unconditionally) | "off when the device does not advertise" |
| make `conformance_options` always empty | "on when it does", assembled-string, caller-options |
| remove `check_fp_conformance` from `build_options` | "raises instead of assembling", real-device refusal |
| prefix-match instead of whole-token match | token matching (`-cl-mad-enable-that-is-not-a-real-option` wrongly refused) |
| drop the caller's options while assembling | caller-options-survive |
| make the `CL_DEVICE_SINGLE_FP_CONFIG` query return 0 | **fp-config-query-is-live** |

The last is the one that matters most. The capability gate is *off* whenever the
device lacks the bit — which is the case on every device here — so a silently
broken query would disable the gate everywhere and make every "correctly gated
off" assertion vacuous. Asserting the query returns non-zero on a real device is
the only thing separating "the device says no" from "we never asked".

### 9.6 HIP does NOT share the omission — but it had a neighbouring one

The same question was put to the HIP path, since it is the same family and
likely the same oversight. It is not: **HIP's inherited defaults are already
conformant.**

Measured on this machine 2026-07-26, ROCm 7.2.4 / AMD clang 22.0.0git, gfx1100,
on `out = sqrtf(a) + a/b` via `clang++ -x hip --cuda-device-only -O3 -S`:

| build | divide / sqrt shape | `.amdhsa_float_denorm_mode_32` |
|---|---|---|
| what Sarek passed (`-ffp-contract=off`) | refined: `v_div_scale_f32`, `v_div_fmas_f32`, `v_div_fixup_f32`; `v_sqrt_f32` + `v_fma_f32` Newton residuals (13 fp instructions) | **3** (subnormals preserved) |
| `-fno-hip-fp32-correctly-rounded-divide-sqrt` (**liveness control**) | bare `v_rcp_f32` / `v_sqrt_f32` with `v_frexp`/`v_ldexp` scaling, **no `v_div_fixup_f32`** (10) | 3 |
| `-fgpu-flush-denormals-to-zero` (**liveness control**) | refined | **0** (flushed) |
| **with the two flags now set explicitly** | **identical** to row 1, all 13 instructions | 3 |

So setting them costs nothing, and both controls confirm the comparison can go
non-identical. What it buys, since `hiprtc_options` appends `base_options`
**last**: a caller passing `-fgpu-flush-denormals-to-zero` or
`-fno-hip-fp32-correctly-rounded-divide-sqrt` is now neutralised by last
occurrence — verified, denorm mode returns to 3 and `v_div_fixup_f32` returns to
the output. And a future clang default change becomes a no-op instead of a
silent regression, which is the entire lesson of the OpenCL case.

**The limit of the append-last defence, measured:** `-ffast-math` from a caller
removes `v_div_fixup_f32` and is **not** rescued by appending
`-fhip-fp32-correctly-rounded-divide-sqrt`, because it sets the per-instruction
`afn` fast-math flag rather than changing the lowering default. No trailing
clang option was found that undoes the `-cl-*` spellings at all.

**HIP's actual gap was silence.** `fp_relaxing_option_prefixes` warned on four
prefixes and passed these through unwarned, each measured to change gfx1100
codegen: `-Ofast`, `-cl-fast-relaxed-math`, `-cl-unsafe-math-optimizations`,
`-fapprox-func` (all degrade divide/sqrt exactly like `-ffast-math`),
`-fgpu-flush-denormals-to-zero`, `-fno-hip-fp32-correctly-rounded-divide-sqrt`,
and `-munsafe-fp-atomics` (swaps the `global_atomic_cmpswap_b32` CAS loop for a
hardware `global_atomic_add_f32`). All are now in the list.

**Still open, deliberately.** The first four are *not neutralisable by anything*,
which is precisely the condition under which §5 says to **reject** rather than
warn. That is a caller-visible behaviour change and is not made here. It is the
recommended follow-up, and until it lands a caller can still relax HIP float
semantics by passing `-ffast-math` and reading a warning.

---

---

## 10. Metal: measured on an M4 — and the compile options were the wrong lever (backlog #125)

This section was written as "a statement about code, not behaviour", because
there was no Apple hardware. An Apple M4 became available on 2026-07-26, and the
measurement **contradicted the fix it was defending**.

**Device and toolchain, named as §1 corollary 3 requires:** Apple M4, macOS
15.6.1 (24G90), arm64, Apple clang 17.0.0 (clang-1700.0.13.5), Metal.framework
from the Command Line Tools SDK. No Xcode, therefore no offline `metal`
compiler — and it does not matter, because `newLibraryWithSource:options:error:`
compiles through the driver at runtime, which is exactly the call under test.

### 10.1 What was wrong before

`Metal_bindings.mtl_device_new_library_with_source` took an `_options` parameter
and ignored it; `Metal_api` passed null regardless. So every Sarek Metal kernel
took Metal's defaults and there was **no route to change them** — not "no
policy", but an *unsettable* wrong default.

### 10.2 Metal's defaults are fast on TWO knobs, not one

Read from a freshly constructed `MTLCompileOptions`:

| property | default | meaning |
|---|---|---|
| `mathMode` | `2` = `MTLMathModeFast` | aggressive unsafe FP optimisation |
| `mathFloatingPointFunctions` | `0` = `...Fast` | single-precision math functions resolve to `metal::fast` |

"Metal defaults to fast math" was previously quoted from Apple's documentation.
It is now measured — and it is **two** independent knobs. Setting `mathMode`
alone leaves math functions on the fast path.

The enum values are **read from the SDK** (`MTLLibrary.h:241-246`, `258-262`),
not guessed. The previous revision of `Metal_bindings.ml` deliberately avoided
`setMathMode:` because these values could not be checked; that objection is
retired by reading them.

### 10.3 The options ARE honoured — liveness, not plumbing

A compile that succeeds proves plumbing, not semantics. That distinction is what
made the OpenCL FP liveness control valuable (§9.2), so the same question was put
to Metal. Over 65536 inputs on `sqrt(a) + 1/a`, against the default:

| setting | results changed |
|---|---|
| `mathMode = Safe` | **16017 / 65536** |
| `mathMode = Safe` + `fpFunctions = Precise` | **22135 / 65536** |
| `fastMathEnabled = NO` | **22135 / 65536** |
| `mathMode = Fast`, `fastMathEnabled = YES` | 0 (confirming the default) |

**Metal is therefore NOT in rusticl's class.** It does not accept these options
and discard them. And the two-knob point is quantified: 16017 against 22135.

### 10.4 But the options do NOT touch contraction — and that broke the fix

The kernel under test is `o = a*b + c`. The separately-rounded reference is
computed in **two kernel passes** with the product round-tripped through device
memory, so it cannot itself be fused; the device's **own** `fma` is read from the
device rather than modelled (§6 records why an IEEE model risks a false "they
agree"); and elements are restricted to the **8773 of 65536** where those two
differ, since contraction is unobservable anywhere else.

| build | contracted |
|---|---|
| default options | 8773 / 8773 |
| `mathMode = MTLMathModeSafe` | **8773 / 8773** |
| `mathMode=Safe` + `fpFunctions=Precise` | **8773 / 8773** |
| `fastMathEnabled = NO` | **8773 / 8773** |
| **`#pragma METAL fp contract(off)`** | **0 / 8773** |

`a*b+c` is bit-identical across every compile-option setting (0/65536 differ).

**This is §1 corollary 2 for the third time in this document** — a flag that
names the hazard is not a mechanism that prevents it. `-ffp-contract=off` did
nothing for the AMDGPU combine; `precise`/`NoContraction` does nothing for the
RADV f16 narrowing (§6); `mathMode = Safe` does nothing for Metal contraction.
Had the M4 not become available, this change would have shipped `mathMode = Safe`
believing it was a contraction defence. It is not one, and it is no longer
described as one.

### 10.5 What actually prevents it

A full barrier sweep, every variant built with `mathMode=Safe` +
`fpFunctions=Precise` (`tools/probes/metal_contraction_barrier_probe.m`):

| candidate | contracted |
|---|---|
| plain `a*b+c` | 8773 / 8773 |
| **`#pragma METAL fp contract(off)`** | **0 / 8773** — adopted |
| `#pragma METAL fp math_mode(safe)` | 8773 / 8773 |
| `#pragma clang fp contract(off)` | 0 / 8773 |
| `volatile thread` local | 0 / 8773 |
| `threadgroup volatile` round-trip | 0 / 8773 |
| device round-trip | 0 / 8773 |
| `as_type` bitcast round-trip | 0 / 8773 |
| `precise::` namespace | does not compile — no such namespace in MSL |

Note `#pragma METAL fp math_mode(safe)` fails exactly as the `mathMode`
*property* does: on Metal, math mode and contraction are orthogonal.

`Sarek_ir_metal` emits `#pragma METAL fp contract(off)` in every generated
kernel. It is chosen over the working alternatives because it is file-scoped,
costs no register or memory traffic, and needs no per-expression codegen change —
the same reasoning that put `precise` on GLSL locals (§6). All nine byte-exact
generated Metal goldens were compiled on the M4 to confirm the pragma is accepted
in the position Sarek emits it. Gated by `test_codegen_golden`'s
`metal_contraction_pragma` group, which carries an anti-vacuity control that
fails if the kernel list is empty.

### 10.6 Which API spelling, settled by measurement

`fastMathEnabled` is deprecated since macOS 15.0 in favour of `mathMode`, but
`mathMode` does not exist before macOS 15.0 / iOS 18.0, so both are needed: the
modern pair when present, the deprecated boolean as fallback, selected by
`respondsToSelector:`.

**They are equivalent, measured.** `fastMathEnabled = NO` and
`mathMode=Safe + fpFunctions=Precise` are **bit-identical over 65536 elements**
of `sqrt + reciprocal + sin + log + exp` (0 differ). The pre-macOS-15 fallback is
exact, not degraded — measured rather than assumed.

### 10.7 Interpreter agreement: measured, and it holds

`test_df64` and `test_real64` have now been run on the M4. **Metal agrees with
the interpreter oracle on every operation**, worst-case relative error over each
test's own input set:

| op | Metal | Interpreter | tol | |
|---|---|---|---|---|
| `mul` | 9.07e-15 | 9.07e-15 | 1.42e-14 | PASS |
| `add` | 5.33e-15 | 5.33e-15 | 7.11e-15 | PASS |
| `sub` | 6.51e-15 | 6.51e-15 | 7.11e-15 | PASS |
| `div` | 5.08e-15 | 5.08e-15 | 1.42e-14 | PASS |
| `sqrt` | 8.53e-15 | 8.53e-15 | 1.42e-14 | PASS |
| `of_i32` | 0 | 0 | 0 | PASS |

`test_real64`'s df64 fallback on the same device: add 5.31e-15, sub 6.94e-15,
mul 8.73e-15, div 5.26e-15, sqrt 8.87e-15 — all PASS. `dot(2^20)` 2.89e-13,
identical to the interpreter's.

These are **sampled maxima on one device over one input set, not bounds**, and
Metal reproducing the interpreter's figure is agreement between two summary
statistics — no element-wise comparison was run. Same caveat as the CUDA
figures in §7.

### 10.8 The pragma does NOT move df64 — and that is the expected answer

Rebuilt with `metal_fp_contract_pragma` emptied and re-run on the same device:
**every df64 and real64 figure above is unchanged, to the digit.**

That is not a failure of the pragma. It is `Sarek_df64` working as designed.
The library defeats contraction **by construction** — `mul_rn` routes products
through `fma` so there is no fusable multiply left for any compiler to fuse —
which is exactly why §2 records "no flag" as the CUDA and OpenCL mechanism too.
**`Sarek_df64` is immune to contraction by design, so it cannot be an instrument
for detecting it.** Expecting it to move the way `sqrt.rn.f32` did on NVIDIA
conflates two different mechanisms: that was an approximate *seed inside the
algorithm*, which `mul_rn` does not protect against; this is contraction, which
it does.

So the pragma's value is precisely for the code `Sarek_df64` cannot protect —
**caller-written kernels**. That is the §3 hazard that "lives in *caller* code,
and no gate in this repository can see": a user writing `a*b + c`, or
`df64_add_f32 acc (x *. y)`, re-creates the fusable pattern the library removed
from its own code. On the M4 that pattern is contracted 8773/8773 without the
pragma and 0/8773 with it (§10.4). The synthetic probe is the right instrument
and the only one; df64 passing either way is consistent with both.

### 10.9 What is STILL not established

- **Element-wise identity.** The agreement above is between summary statistics.
- **One device, one OS.** M4 / macOS 15.6.1 only. Nothing here constrains older
  Metal versions, Intel Macs, or the `fastMathEnabled` fallback path — which was
  measured for *equivalence* on macOS 15 but has never run on the pre-15 OS that
  is its only reason to exist.
- **Subnormals.** Not probed at all on Metal.
- **f16 on Metal.** Not probed. The three other backends that reach an ACO or
  NVIDIA compiler each needed a separate exhaustive f16 sweep before anything
  could be said; Metal has had none.
- **The two kernels that do not compile** (§10.11) are excluded from every
  figure above, because they never ran.

### 10.10 Running the full suite on the M4: one more environmental finding

`dune runtest` on the M4 is green except for **six `opencl_validation_sweep`
cases, all `float64`**. Apple's clang rejects them outright:

```
warning: unsupported OpenCL extension 'cl_khr_fp64' - ignoring
error: use of type 'double' requires cl_khr_fp64 support
```

**Environmental, not a regression.** Apple Silicon's OpenCL has no `cl_khr_fp64`
at all, so the host-clang validation of any generated f64 OpenCL kernel cannot
pass there. This branch touches no OpenCL codegen (its OpenCL changes are all in
`sarek-opencl/` runtime bindings) and no f64 path, so it cannot be the cause.
Worth noting nonetheless: cases 7-8 of the same sweep **SKIP** on f64 while
16-20 **FAIL**, so the sweep's own capability gating is inconsistent — some f64
cases detect the missing extension and some do not.

**FIXED (#140).** The investigation found something worse than "some paths
check": *neither* did. Cases 7-8 skipped for a completely unrelated reason — a
hardcoded `validation_exclusions` entry for `Float64.abs_float` /
`Float64.copysign`, which are user-callable but absent from
`Sarek_pure_registry.float64_list` and therefore die at codegen — and would have
skipped identically on a machine with full fp64. Cases 16-20 simply ran and hit
the toolchain. There was no fp64 predicate anywhere in the gate.

Note also what the capability *is* here. The sweep never touches a device; it
compiles with `clang -x cl`. So the authority is not the device's `cl_khr_fp64`
(`Opencl_api.ml`'s `supports_fp64`, which the e2e suite uses) but whether *this
clang* can compile a `double` kernel — Apple clang's `arm64-apple-darwin` target
does not list the extension, so the `#pragma OPENCL EXTENSION cl_khr_fp64 :
enable` the production path emits is ignored with a warning and `double` is an
error. `Opencl_clang.fp64_available ()` establishes that by compiling a `double`
probe, the same way `available ()` establishes clang itself. The sweep now asks
it **before** the exclusion list, so on a toolchain without fp64 all *seven*
float64 cases report one verdict for one stated reason.

Reproduced on Linux rather than argued: `SAREK_OPENCL_GATE_NO_FP64=1` adds
`-cl-ext=-cl_khr_fp64` to every invocation, which makes clang 22.1.6 emit the
identical `error: use of type 'double' requires cl_khr_fp64 support`. With the
new predicate disabled that reproduces the M4 split exactly — 7, 8 SKIP and
16-20 FAIL; with it enabled, seven uniform SKIPs. `ci/assert-toolchain.sh` now
carries an fp64 positive control so the skip cannot become CI's normal outcome,
and `test_opencl_gate.ml` re-runs itself under the switch so a suppression that
did nothing would fail rather than pass.

All six `metal_contraction_pragma` cases pass on the M4.

### 10.11 A separate defect this uncovered

Compiling the nine byte-exact generated Metal goldens on the M4: **seven compile,
two do not.** `record_kernel` and `variant_kernel` emit `constant Point2* &pts` /
`constant Opt* &out`, and Metal rejects the pointee address space: *"invalid
address space qualification for buffer pointee type ... valid address space
qualifications are device and constant"*.

**Pre-existing and unrelated to the FP work** — a control run with the pragma
stripped from those same two sources fails identically. It is a codegen
correctness bug that had never been observable, because nothing in this project
had ever compiled Metal on Apple hardware.

**FIXED (#139), and the design decision is `device`.** The cause: the
`DParam (v, None)` arm of `gen_param_metal` treated *every* parameter without an
`array_info` as a scalar and emitted `constant <ty> &name`; for a vec-typed `v`,
`metal_type_of_elttype` returns a pointer type, so the emission was a reference
to a pointer whose pointee had no address space.

`constant` is not a live option for a Sarek `vec`, so the choice is not close:
objects in `constant` are read-only for the lifetime of the kernel (MSL 3.2
§4.2/§4.3) and both offending kernels *write* through the parameter
(`pts[idx] = ...`, `out[idx] = ...`), `constant` carries an
implementation-defined size limit and expects per-dispatch-invariant data, and
every other backend already lowers a vec parameter to a mutable global pointer
(CUDA `T* __restrict__`, OpenCL `__global T* restrict`). Metal's own
`metal_param_type` and `metal_memspace Global` already said `device`; this arm
was the single place in the backend that disagreed with them.

The gate that would have caught it now exists: `metal_validation_sweep`, with
`Metal_gate.Metal_addrspace` (layer 1, pure text, no toolchain — so it runs on
the Linux machines where the defect was introduced) and
`Metal_gate.Metal_compile` (layer 2, `xcrun metal`, macOS only, honest skip with
a stated reason elsewhere). Layer 1's red path is driven permanently by
`sarek/tests/unit/test_metal_gate.ml`, which feeds it the pre-fix golden
verbatim.

### 10.12 A third defect, found the same way (#132)

`wgsl_validation_sweep`'s corpus contained no multi-field variant payload and,
worse, no `SMatch` at all — `variant_kernel` only *constructs* variants — so
every accessor and every `switch` the WGSL match emitter can produce was
unreachable from any executable gate. The field-naming half of #132 had already
been closed by PR #306 (`EMatch` and all five `SMatch` paths now share one
`payload_layout`, and WGSL's `indexed = true` spelling `.MkPair_v_0` matches its
flat struct). What the coverage hole was still hiding was a different, live bug:
the emitter wrote `default:` only when the source match had a wildcard arm, and
WGSL requires exactly one default clause in every `switch`. Every exhaustive
match therefore produced a module `naga` rejects with *"missing default case"* —
i.e. the ordinary case. C's `switch` needs no default and GLSL's likewise, so
WGSL was the odd one out here too, exactly as it was for payload spelling.

Adding `smatch_multi_payload` to the sweep turned it red on the first run;
emitting a synthetic empty `default` when no `PWild` arm is present turned it
green.
