# Floating-point contraction policy

_Cross-backend policy for what a Sarek DSL author may rely on when a device
compiler is free to fuse, reassociate or flush floating-point operations._

**Status:** normative for this repository. **Tracked as:** backlog-116 (absorbs backlog-110, backlog-111);
§6 answers backlog-126, the HIP row answers backlog-106, and §11 answers backlog-123.
**Date:** 2026-07-27.

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
  (measured here, RX 7900 XTX, Mesa 26.1.4-arch3.1) and on Mesa ANV (measured,
  Intel Arc Graphics / Meteor Lake-P, Mesa 26.1.2-arch3.1 — §11.1; and quoted
  on Intel UHD Graphics 630, a different generation that the Arc measurement
  does *not* speak for). Attributed to a `fma` that is not correctly rounded,
  *not* to contraction — but the two failure modes look identical from the
  outside, and telling them apart took a separate measurement, now made on both
  drivers.
- **Vulkan/RADV, f16.** RADV's ACO backend absorbs an f32→f16 narrowing into
  whatever arithmetic feeds it, and `precise`/`NoContraction` does not stop it —
  the decoration is emitted, and the emitted ISA is byte-identical with and
  without it *on the one-narrowing shape*, where there is no addition for it to
  bind to (§12.4). **2912 of 63488** finite binary16 inputs disagree with the
  interpreter on a single narrowing, **5075** on a two-narrowing expression.
  **All of those counts are now named closed-form functions, matched
  bit-for-bit on 63488/63488** — see §12.
  Second front end onto the same ACO backend as rusticl/radeonsi, and a wider
  combine than either it or HIP — whose identical-looking fusion comes from a
  *different* compiler, LLVM's AMDGPU backend (§2, "Two AMD compilers"). f16
  stays refused on this backend (§2, §6).

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

### Two AMD compilers, not one

Three rows below are AMD GPUs, and it is tempting — this document did it — to
read them as one compiler behind several front ends. **They are two compilers:**

| stack | front end | compiler that emits the ISA |
|---|---|---|
| HIP / hiprtc (ROCm) | HIP C++ | **LLVM's AMDGPU backend** — ROCm ships its own clang/LLVM; `libamd_comgr` is linked against `libLLVMAMDGPUCodeGen`, and §9.6's HIP measurements are `AMD clang 22.0.0git` |
| OpenCL / rusticl on radeonsi | OpenCL C → SPIR-V | **ACO**, Mesa's shader compiler — named by Mesa in the device string itself: `AMD Radeon RX 7900 XTX (radeonsi, navi31, ACO, DRM 3.64, …)` |
| Vulkan / RADV | GLSL → SPIR-V | **ACO**, same compiler, second front end |

Evidence: **by-construction** (what the shipped toolchains are made of). No
measurement below changes, and no measurement below moves tier because of this
note.

**What the agreement therefore means.** hiprtc and rusticl produce the *same*
620 of 63488 disagreements with the same first divergence at `x = 5.68359375`.
Read as one compiler seen twice, that is a single bug. Read correctly, it is the
stronger statement: **the f32-multiply-into-f16-narrowing fusion is present in
both of AMD's GPU compilers independently — it is an AMD-toolchain-wide
behaviour, not an ACO one.** RADV adds a third data point on the ACO side with a
*wider* combine.

**Why the distinction is load-bearing and not pedantry.** "The locus is ACO" is
the sentence someone would use to scope a future barrier, a denylist or a
detection predicate. Scoped to Mesa/ACO, such a predicate misses the
LLVM/AMDGPU path that HIP takes; scoped to ROCm, it misses rusticl and RADV. The
`#if defined(__HIP_PLATFORM_AMD__)` guard on `sarek_f32_barrier` (§4, §11.4) is
correct precisely because it keys on **the toolchain compiling that source**,
which no cross-compiler generalisation is needed to get right. Where a predicate
below *is* keyed on the string `"ACO"` (§3's OpenCL tripwire), that key selects
Mesa devices and only Mesa devices — see the note there.

**What the non-AMD negatives still establish.** pocl/x86, Intel IGC, the Intel
oneAPI CPU runtime and NVIDIA all decline to fuse (§11.3). That still rules out
*OpenCL* and *SPIR-V* as the locus. It no longer supports "ACO specifically",
because the one non-Mesa stack that does fuse is AMD's own.

| backend | what the compiler may contract | what actually prevents it | evidence |
|---|---|---|---|
| **Interpreter** | nothing — it is the oracle | it evaluates the IR directly | executed (any host) |
| **HIP / AMDGPU** | `a*b+c`; **and**, below the FP flags, an f32 multiply or fma fused into an f32→f16 narrowing (`v_fma_mixlo_f16`), plus f32 add/sub/mul/negate demotion to binary16 (`v_add_f16`, `v_sub_f16`, `v_mul_f16`) | two mechanisms, both required: `-ffp-contract=off` forced **last** in the hiprtc option array (`Hip_rtc.base_options` / `hiprtc_options`), *and* the `asm volatile("" : "+v"(x))` opacity barrier on every narrowing's argument (`Sarek_ir_cuda.sarek_f32_barrier_decl`). One barrier covers every affected shape; none needs a different one | **executed + machine-code**, RX 7900 XTX / gfx1100 **and** Raphael iGPU / gfx1036, ROCm hiprtc: all 20 Sarek-emittable f16 expression shapes swept over all 63488 finite binary16 inputs, **0 disagreements as shipped**; removing the barrier breaks 9 of 20 (reproducing the original 620 exactly on the `f16_midround` shape), and disassembly shows demotion opcodes in 3 further shapes that are demoted yet numerically clean — see [`docs/optimization/amdgpu-f16-fusion-shape-audit.md`](optimization/amdgpu-f16-fusion-shape-audit.md) |
| **CUDA / nvrtc (f16 narrowing)** | in principle the same fusion — but NVIDIA has no fused multiply-and-convert-to-f16 instruction to fuse *into* | **nothing Sarek emits.** `ptxas` simply declines to absorb `cvt.rn.f16.f32` | **executed**, GTX 1070 Max-Q / sm_61 / CUDA 12.9 / driver 580.119.02: exhaustive sweep of all 63488 finite binary16 inputs, 0 device/interpreter disagreements, with a liveness control proving the sweep can go red (§7). Also machine-code, CUDA 13.3 host tools, sm_75…sm_121 — see §4. Machine-checked by `test_cuda_f16_sass`, which until this change **self-skipped in CI** for want of `nvdisasm` (§7) |
| **CUDA / nvrtc + PTX (f32 `a*b+c`)** | yes, by default (`-fmad=true` is nvrtc's and ptxas's default, and it applies to PTX input too) | **no flag.** `Sarek_df64` denies the compiler a fusable multiply by routing products through `fma` (`mul_rn`) | executed, GTX 1070 Max-Q / sm_61 / CUDA 12.9 / driver 580.119.02: df64 mul 5.92e-08 → 9.07e-15, div 5.64e-08 → 5.08e-15 |
| **CUDA — subnormal flushing** | `-use_fast_math` / `-ftz=true` would flush binary32 subnormals | `Cuda_nvrtc.check_fp_conformance` **rejects** those options at the only point an option array reaches `nvrtcCompileProgram` | machine-code + test, CUDA 13.3: the hazard is reproduced (`FMUL.FTZ`/`FADD.FTZ` at sm_90) and the guard is proved to fire — see §5 |
| **OpenCL** | `FP_CONTRACT` is on by default in OpenCL C, and no build option turns it off | for contraction, **no flag** — same `mul_rn`-by-construction defence as CUDA. For div/sqrt, `Opencl_fp.conformance_options` requests `-cl-fp32-correctly-rounded-divide-sqrt`, **gated** on `CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT` in the device's `CL_DEVICE_SINGLE_FP_CONFIG`; `Opencl_fp.check_fp_conformance` **rejects** the relaxing `-cl-*` options at `Opencl_api.Program.build`, the single point an option string reaches `clBuildProgram` (§9) | executed, GTX 1070 Max-Q / NVIDIA OpenCL: mul 5.92e-08 → 9.07e-15, sqrt 2.88e-08 → 9.80e-15 with no OpenCL-specific change (quoted). Re-measured here on RX 7900 XTX / Mesa radeonsi: mul 9.07e-15, div 5.08e-15, sqrt 1.08e-14 |
| **OpenCL / rusticl (f16 narrowing)** | an f32 multiply into the f32→f16 narrowing that consumes it — rounding **once** where the DSL mandates twice. Same defect class as HIP/AMDGPU, and bit-for-bit the same count — but a **different compiler**: ACO here, LLVM's AMDGPU backend there (see "Two AMD compilers" above) | **nothing affordable.** Measured non-fixes, all still 620/63488: `#pragma OPENCL FP_CONTRACT OFF`, a `volatile` local, a `volatile __private` pointer, an `as_half`/`as_ushort` bitcast round-trip, and `convert_half_rte`. HIP's `asm volatile("" : "+v"(x))` **does not compile** here — rusticl goes through SPIR-V, where AMDGPU register constraints do not exist. Only a `volatile __global` round-trip and a `volatile __local` (LDS) round-trip work (both 0/63488), and both cost memory traffic per narrowing; the LDS form additionally needs a workgroup-sized allocation this backend does not control. **Consequence: f16 stays REJECTED in `Sarek_ir_opencl`** | **executed**, 2026-07-26, exhaustive sweep of all 63488 finite binary16 inputs on **two** devices — RX 7900 XTX (navi31) and the integrated Raphael iGPU (gfx1036) — rusticl/radeonsi, DRM 3.64, kernel 7.1.2-3-cachyos. Both report **620/63488**, first divergence at `x=5.68359375` (device 1006.5, interpreter 1006), bit-identical to the HIP figure. Liveness control: the `volatile __global` variant of the same harness reports **0/63488**, so the sweep is proven able to go both red and green. Reproducer: `tools/probes/opencl_f16_contraction_probe.c`. **Upgraded 2026-07-27 to ELEMENT-WISE model agreement (backlog-62 slice 1, §12.3):** the plain kernel is bit-identical to `S_fuse_mul_into_narrowing` on **63488 / 63488** inputs on both devices, on this shape *and* on `f16(x*1.1)` (2912/63488 against the discipline, never previously swept here). The `double`-based `fusedctl` control does **not** build on rusticl, which advertises no `cl_khr_fp64`; the replacement control carries the exact product as an f32 pair and rounds to odd. Instrument: `sarek-opencl/probe/probe_opencl_f16_model_agreement.ml` |
| **Vulkan / RADV (f16 narrowing)** | an f32→f16 narrowing absorbs whatever arithmetic feeds it (`v_fma_mixlo_f16`) — the multiply, and also the f32 **add**: the plain two-narrowing kernel compiles to a *single* fused instruction, one rounding where the DSL mandates three. Same ACO backend as rusticl, reached through a second front end — HIP's identical-count defect comes from LLVM's AMDGPU backend instead — and a **wider** combine than either | **nothing affordable, and `precise` is not it.** `precise` → SPIR-V `NoContraction` IS honoured (it keeps the f32 multiply as its own `v_fma_mix_f32`) and still leaves 2912/63488, because absorbing a *conversion* is a different combine from contracting `a*b+c`. An f16 bitcast round-trip changes nothing. A `volatile` SSBO round-trip on the f32 intermediates makes ACO drop the intermediate narrowing **entirely** instead (4774/63488). Only forcing the f16 *bit pattern* through global memory works (0/63488), at a global round-trip per narrowing into a scratch buffer this backend does not control. **Consequence: f16 stays REJECTED in `Sarek_ir_glsl`** | **executed**, 2026-07-26, exhaustive sweep of all 63488 finite binary16 inputs on **two** devices — RX 7900 XTX (**RADV NAVI31**) and the integrated Raphael iGPU (**RADV RAPHAEL_MENDOCINO**) — Mesa 26.1.4-arch3.1, Vulkan 1.4.354. Both report identical counts: **2912/63488** on `f16(x*1.1)` (plain and `precise` alike), **5075/63488** on `f16(f16(x*1.1)+1000)` plain, **4776/63488** with `precise`. Calibration: the same host oracle reproduces the independently measured **620** on the HIP/OpenCL kernel shape, and the barriered kernel reports **0/63488**, so the sweep is proven able to go both red and green. Gate: `sarek-vulkan/test/test_vulkan_f16_tripwire.ml`. **Upgraded 2026-07-27 to ELEMENT-WISE model agreement (backlog-62 slice 1, §12.4):** each of those counts is a named closed-form function matched on **63488 / 63488** on both devices — plain `f16(x*1.1)` and `precise` `f16(x*1.1)` are `S_fuse_mul_into_narrowing`; plain `f16(f16(x*1.1)+1000)` is `S_absorb_all_into_final_narrowing` (a **single** `v_fma_mixlo_f16` over x, 1.1 and 1000); the same shape with `precise` is `S_f32_mul_then_absorb_add` (`v_fma_mix_f32` then `v_fma_mixlo_f16`); the f32-barriered variant is `S_drop_intermediate_narrowing`. `precise` is **honoured**, not ignored — it forbids a multiply-into-ADD contraction and cannot reach a conversion absorbing its operand, which is why it is byte-identical on the shape with no addition and changes the answer on the shape with one. Instrument: `sarek-vulkan/probe/probe_vulkan_f16_model_agreement.ml` |
| **OpenCL / pocl on x86 (f16 narrowing)** | in principle the same fusion — but nothing in this stack performs it | **nothing needed.** The naive narrowing already round-trips through binary16 exactly, so the barrier that rusticl requires is unnecessary here | **executed on CI**, 2026-07-26, quoted device `AMD EPYC 7763 64-Core Processor` under pocl on a GitHub-hosted runner: exhaustive sweep of all 63488 finite binary16 inputs, **0** disagreements between the naive and `volatile __local`-barriered narrowings. Observed as a CI failure of `test_opencl_f16_tripwire` before that test was scoped, i.e. the number was produced by a harness that was at the time *trying* to find a difference — so it is a null with the sweep demonstrably live. **This is what localises the defect:** the same source, swept the same way, fuses on an AMD GPU compiler and does not fuse here, so the locus is *the AMD GPU compilers*, not *OpenCL* and not *SPIR-V*. Note what it does **not** localise: rusticl and HIP/AMDGPU do not share a compiler (see "Two AMD compilers" above), so their identical 620 is two compilers agreeing, not one bug seen twice. **Confirmed by a second, independent negative on a real GPU** — Intel Arc Graphics under the Intel Compute Runtime / IGC, a compiler sharing no lineage with Mesa: 0/63488, with the sweep calibrated on the same run against the known 620 (§11.3). Guarded by `test_opencl_f16_tripwire`'s locus check, which fails if any OpenCL implementation outside its `"ACO"` device-string scope is found to fuse — and which was itself wrong until §11.5. Read that predicate as "not Mesa", not as "not AMD": an AMD GPU reached through ROCm's OpenCL would be compiled by LLVM's AMDGPU backend, i.e. by a compiler this document expects to fuse, while sitting outside the key |
| **Vulkan / GLSL** | contraction and reassociation of float expressions | `precise` on every float local (`Sarek_ir_glsl.gen_var_decl`), which glslang lowers to SPIR-V `NoContraction` — but on RADV nothing needs preventing *for these shapes*: the driver does not contract them even without the decoration. It is **not** the decoration that is protecting them; RADV was separately observed ignoring `NoContraction` on a combine it does want to perform (§6, f16 narrowing) | **executed + machine-code**, RX 7900 XTX (RADV NAVI31) and Raphael iGPU (RADV RAPHAEL_MENDOCINO), Mesa 26.1.4-arch3.1: 0 of 7 contraction shapes contracted with or without `precise`, ISA opcode-identical between the two builds, explicit `fma()` controls fused 4/4 — see §6. Decoration emission: compiler-output, glslc 2026.2 + glslangValidator, 18 `NoContraction` with `precise` / 0 without. **Mesa ANV now measured too** (§11.2): Intel Arc Graphics (Meteor Lake-P), Mesa 26.1.2-arch3.1, same 0 of 7 / 0 of 7 with `fma()` controls 4/4 — and unlike RADV, ANV does not fuse the f16 narrowing either, so no combine has been found on ANV where `NoContraction` is ignored. Separately, `fma` is not correctly rounded on RADV: df64 mul 5.84e-08 / div 5.86e-08, each the measured worst-case relative error over `test_df64`'s own input set on the named device and driver, not a bound; ANV shows the same signature (mul 5.84e-08 / div 5.86e-08, §11.1) |
| **Metal** | contraction of `a*b+c` — **measured, and NOT preventable by any compile option**; separately, both math defaults are the fast one (`mathMode = MTLMathModeFast`, `mathFloatingPointFunctions = ...Fast`, read from a fresh `MTLCompileOptions`) | **two mechanisms, both required**: `#pragma METAL fp contract(off)` in every generated kernel (`Sarek_ir_metal.metal_fp_contract_pragma`) for contraction, *and* `mathMode = Safe` + `mathFloatingPointFunctions = Precise` in `Metal_bindings.mtl_compile_options_conformant` for math-function accuracy (falling back to the deprecated `fastMathEnabled = NO` before macOS 15) | **executed**, Apple M4 / macOS 15.6.1 (24G90) / Apple clang 17.0.0: on the 8773 of 65536 elements where the device's own `fma` differs from the separately-rounded value, `a*b+c` is contracted 8773/8773 under every compile-option setting **including `mathMode=Safe`**, and 0/8773 with the pragma. Options are honoured (16017 and 22135 of 65536 math-function results change), so Metal is not in rusticl's ignore-it class. **Interpreter agreement now executed on the same device**: `test_df64` and `test_real64` PASS on every op and reproduce the interpreter's figures exactly (mul 9.07e-15, add 5.33e-15, sub 6.51e-15, div 5.08e-15, sqrt 8.53e-15) — sampled maxima over each test's input set, not bounds, and agreement between summary statistics rather than element-wise identity. **The f16 narrowing is now probed and does NOT fuse** — 0/63488 from the discipline on **both** swept shapes, element-wise, with a validated positive control that goes red on **both**: the same control reports **2912/63488** on `f16(x*1.1)` and **620/63488** on `f16(f16(x*1.1)+1000)`, the latter being the figure already measured on hiprtc/gfx1100, rusticl/radeonsi and Intel Arc (§10.14). Two shapes, two independently-nonzero controls, two zeros; the pragma changes nothing there, in either direction, because there is no fusion to prevent. Subnormals still unprobed; two record/variant kernels do not compile at all (§10.11). **f64 is refused outright** — MSL has no `double`, and until backlog-141 `TFloat64` was silently emitted as `float`, striding an 8-byte-per-element host buffer at 4; `Sarek_real64` (df64) is the supported route and is the figure quoted above (§10.13) |
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
  refuses kernel-level f16 by design (backlog-57 slice 2, a located
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
- **Metal subnormals and Metal record/variant kernels.** Metal **f32
  arithmetic** has moved OFF this list: `test_df64` and `test_real64` agree with
  the interpreter on every op on an Apple M4 / macOS 15.6.1 / Apple clang 17.0.0
  (§10.7), with contraction defeated by `#pragma METAL fp contract(off)` in
  every generated kernel (§10.5). **The Metal f16 narrowing has now moved off
  too**, on the strongest evidence in this document: element-wise agreement with
  the discipline on all 63488 finite binary16 inputs, on **both** swept shapes,
  each with its own validated positive control proven able to go red — 2912/63488
  on `f16(x*1.1)` and 620/63488 on `f16(f16(x*1.1)+1000)` (§10.14). What is still NOT covered: subnormals
  (never probed), f32 element-wise identity (that agreement is still between
  summary maxima — the f16 result *is* element-wise), the f16 shapes beyond the
  two swept here, any device or OS other than that one, and kernels using
  **records or variants**, which do not compile on Metal at all (§10.11).
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
  OpenCL device name, which is where Mesa reports its shader compiler — so the
  key means "a Mesa stack", not "an AMD GPU": ROCm's OpenCL on the same card
  compiles through LLVM's AMDGPU backend and is out of scope, see §2's "Two AMD
  compilers") and
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
  every float local, and backlog-106/backlog-126 measured RADV not to contract 7 f32 shapes;
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

## 4. CUDA f16: the barrier bought nothing *at the narrowing*, and has been removed (backlog-110)

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

## 5. CUDA: `-use_fast_math` / `-ftz=true` are refused (backlog-111)

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
**backlog-57 slice 2b closed it — negatively.**

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

> **AMENDED 2026-07-27 (backlog-62 slice 1, §12.4): "ignored" is the wrong word, and
> the right one changes what may be inferred.** The decoration is *inapplicable*
> here, not disobeyed. `NoContraction` forbids contracting a multiply into an
> **addition**; a **conversion** absorbing its operand is a different combine
> and the decoration does not reach it. This shape contains no addition, so
> there is nothing for the decoration to bind to — hence the byte-identical
> ISA. On the two-narrowing shape `f16(f16(x*1.1)+1000)` there *is* an addition
> once ACO has elided the intermediate narrowing, the decoration **binds**, the
> multiply is materialised as its own `v_fma_mix_f32`, and the emitted ISA and
> the numeric answer both change (5075 → 4776 disagreements, each an exact
> match to a different named model). Everything measured in this section stands
> exactly as recorded; what changes is that RADV has **not** been observed
> violating `NoContraction`, and the sentence "that is a `NoContraction`
> violation, observed" above should be read as superseded by §12.4. The
> practical conclusion is unchanged and if anything firmer: `precise` is not
> enough to make f16 safe here, because it constrains only one of the two
> combines in play.

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

**Now run: Mesa ANV — see §11.2.** When this section was written there was no
Intel GPU on this machine, and the ANV half of the original disagreement was a
**hardware gap** rather than a null result. An Intel Meteor Lake Arc machine
became available on 2026-07-27. ANV reproduces RADV's 0 of 7 / 0 of 7 on the
twelve shapes, and — unlike RADV — does not perform the f16
multiply-into-narrowing combine either (0/63488 against the discipline, with the
fused model separating at the calibrating 2912 on the same run). So the campaign
note claiming ANV *ignores* `NoContraction` is **not supported**; neither is any
claim that ANV honours it, because ANV never contracted when it was free to.
That measurement is Xe-LPG and says nothing about the Gen9.5 UHD 630 the
original ANV figures came from (§11.0).

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

- **CLOSED — the ANV half of §6 is run.** An Intel Meteor Lake-P machine became
  available on 2026-07-27 and the whole bullet is discharged in **§11**: the
  `NoContraction` question (§11.2), the df64 ANV allowlist entry that had never
  been executed (§11.1), and — as a bonus the bullet did not ask for — the
  independent OpenCL implementation that confirms the f16 fusion belongs to
  AMD's GPU compilers rather than to OpenCL (§11.3). The campaign note claiming ANV ignores `NoContraction` is **not
  supported**. Scope: **Xe-LPG only**, which is not the Gen9.5 UHD 630 the
  quoted ANV numbers came from (§11.0). **AMDVLK and the proprietary AMD Vulkan
  driver remain unmeasured** — different SPIR-V consumers on the very same GPU.
- **f16 on the PTX backend.** `Sarek_ir_ptx` refuses kernel-level f16 (backlog-57
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
  same device. **Fixed in backlog-136 (§9)** — but the fix could not be
  re-measured here: the red does not reproduce on this machine's OpenCL devices
  (rusticl/radeonsi reports `sqrt` 9.68e-15, a PASS) and rusticl ignores FP
  build options outright. The **Vulkan** residual (1.68e-14 on NVIDIA,
  while Intel UHD 630 passes at 1.17e-14) has no established cause — that one is
  still "do not promote a hypothesis to a cause". The red does not reproduce on
  Intel Arc/ANV either: `test_real64` reports `sqrt` 1.17e-14 there, a PASS, and
  `test_df64` 9.57e-15 (§11.1). Intel's OpenCL, unlike rusticl, passes both at
  the interpreter's own figures (8.87e-15 / 8.53e-15).
- **f16 on Vulkan/GLSL: what slice 2b did NOT close.** The refusal is measured
  and gated, but three things remain unmeasured and must not be read into it.
  (a) **Effectively only RADV.** "Vulkan fuses" is not a claim this repository
  makes; "RADV fuses" is. One non-RADV implementation has since been measured —
  **Mesa ANV on Meteor Lake Arc does not fuse** (§11.2, 0/63488 against the
  discipline with the fused model calibrating at 2912) — which strengthens the
  scoping rather than weakening it. AMDVLK, NVIDIA and lavapipe remain
  unmeasured. The tripwire still carries no "non-RADV does not fuse"
  cross-check, unlike its OpenCL sibling: one data point is thin ground for a
  gate, and §11.5 is a demonstration of what a cross-check built on a
  not-quite-valid oracle does when it meets new hardware.
  (b) ~~**The `shaderFloat16` device feature is not enabled.**~~ **CLOSED
  2026-07-27 (backlog-62 slice 2).** Vulkan requires the feature before a shader
  may use the SPIR-V `Float16` capability; `Vulkan_api_device` used to chain no
  feature structs beyond core `VkPhysicalDeviceFeatures`, and RADV accepted the
  shaders anyway. It now queries `VkPhysicalDeviceShaderFloat16Int8Features` and
  `VkPhysicalDevice16BitStorageFeatures` through the `VkPhysicalDeviceFeatures2`
  chain and **requests both in `VkDeviceCreateInfo.pNext`**, alongside
  `VK_KHR_cooperative_matrix` where advertised.

  **The measurement is unchanged by enabling it, and that is itself a result.**
  Run as a controlled A/B on 2026-07-27 — one build, the feature request
  toggled, `test_vulkan_f16_tripwire` run on each arm, on the RX 7900 XTX (RADV
  NAVI31) **and** the Raphael iGPU, radv / Mesa 26.1.4-arch3.1, Vulkan 1.4.354:

  | arm | RX 7900 XTX | Raphael iGPU |
  |---|---|---|
  | `shaderFloat16` **not** requested (the pre-slice-2 path) | 2912/63488 | 2912/63488 |
  | `shaderFloat16` requested | 2912/63488 | 2912/63488 |

  First divergence identical on every arm (`x = 8.94069672e-07`, device
  `0x0011`, discipline `0x0010`), the `precise` variant identical
  (2912/63488 from the discipline, 0/63488 from the single-rounding model), and
  all three calibration controls green throughout. So the caveat named a real
  gap in the plumbing but **not** a confound in the measurement: ACO's
  absorption of the f32→f16 narrowing does not depend on whether the feature was
  requested. Evidence tier: **executed**, both arms, same binary, same devices.

  Only the one-narrowing shape was re-run this way; the 4774/63488
  two-narrowing figure was not re-measured under the A/B.

  What this does **not** do is lift any refusal — `Sarek_ir_glsl` still refuses
  f16, and that is slice 3.
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
  exactly (§10.7), so Metal **f32** has left the §3 "may NOT rely on" list, and
  **Metal f16 left it too on 2026-07-27** (§10.14, element-wise over the whole
  finite binary16 domain, on both swept shapes, each with a control reproducing
  its own nonzero figure — 2912 and 620 respectively). Metal subnormals and record/variant kernels have not — the last of
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
| `tools/probes/metal_f16_narrowing_probe.m` | the exhaustive Metal f16 narrowing sweep against the named rounding models, with the round-to-odd positive control (§10.14) |
| `tools/probes/metal_simdgroup_matrix_probe.m` | which `simdgroup_matrix` configurations exist on the M4, and what the 8×8×8 f16 MulAdd computes (§10.15) |
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

## 9. OpenCL and HIP: what an unset option was choosing (backlog-136)

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
Measured during PR #298 on that device.

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

## 10. Metal: measured on an M4 — and the compile options were the wrong lever (backlog-125)

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
- ~~**f16 on Metal.** Not probed.~~ **Probed on 2026-07-27 — see §10.14.** The
  f16 narrowing meets the discipline on all 63488 finite binary16 inputs, on
  both swept shapes, each with a positive control that reproduces its own
  nonzero figure (2912 and 620) on the same source and dispatch. This bullet is kept struck through rather than deleted
  because §3's "may NOT rely on" list leaned on it.
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

**FIXED (backlog-140).** The investigation found something worse than "some paths
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

**FIXED (backlog-139), and the design decision is `device`.** The cause: the
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

### 10.12 A third defect, found the same way (backlog-132)

`wgsl_validation_sweep`'s corpus contained no multi-field variant payload and,
worse, no `SMatch` at all — `variant_kernel` only *constructs* variants — so
every accessor and every `switch` the WGSL match emitter can produce was
unreachable from any executable gate. The field-naming half of backlog-132 had already
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

### 10.13 A fourth defect, found the same way (backlog-141): Metal silently halved f64

Not a contraction defect, but it belongs here because it is a *float-semantics*
claim the codebase was making and getting wrong, and because §10's Metal work is
what makes the correct answer available.

`Sarek_ir_metal.metal_type_of_elttype` mapped `TFloat64` to `"float"`, under the
comment *"Metal doesn't support double precision"* — a true statement used to
justify the wrong conclusion. There was **no refusal anywhere**: a kernel
declared over `float64` compiled, ran, and returned a plausible wrong answer.

**Captured before the fix.** For `out.(i) <- inp.(i) *. 2.0` in float64, the
emitter produced, verbatim:

```cpp
kernel void f64_scale(device float* out [[buffer(0)]], constant int &sarek_out_length [[buffer(1)]], device float* inp [[buffer(2)]], ...
```

and for an f64 *local*:

```cpp
  float x = 0.10000000000000001;
```

both with no diagnostic on any channel. Note that the cost is **worse than
halved precision**: the IR element type also fixes the buffer stride, and
`Spoc_core.Vector.float64` is 8 bytes per element, so `device float*` strode the
host's buffer at 4 — every element after the first was a bit-half of a
neighbour. The kernel did not lose precision, it read a different array.

**The fix is a refusal, not a widening**, because MSL has no `double` on any
Apple GPU and never will. Both the element-type arm and a whole-kernel gate
(`Sarek_ir_metal.reject_float64_kernel`, on the existing
`Sarek_ir_analysis.kernel_uses Float64` detector, so an f64 that appears only as
a `CFloat64` literal or an f64-typed local is also caught) render the same
`Sarek_capability` row — `float64_absent_metal`, kind `Backend_structural`,
carrying the remedy — so the fact is stated once and the diagnostic says which
*kind* of unavailability the author is looking at. See
[`docs/design/capability-model.md`](design/capability-model.md).

Both entry points were found twice, independently: backlog-64 reasoned down from the
capability model and was motivated by the f64 **literal** assigned into an f32
buffer; backlog-141 reasoned up from the emitted source and was motivated by the f64
**local**. Neither shape reaches the other's detector, and both searches landed
on the same `kernel_uses Float64` gate at the same two `generate` entries. That
is the argument that {arm, whole-kernel} is the complete set rather than the set
somebody happened to think of.

**`Sarek_real64` is that route and it was never broken.** df64 is *double-float*
— an unevaluated pair of binary32 giving ~2⁻⁴⁶ — so it needs no hardware fp64;
that is its purpose. `Metal_api.Device` reports `supports_fp64 = false`, and
`Sarek_real64` already selects `Fallback_df64` on that basis. Measured on the M4
(§10.7): mul 9.07e-15, add 5.33e-15, sub 6.51e-15, div 5.08e-15, sqrt 8.53e-15.

**The contradicting comment is the reason this survived.** `Metal_api.ml`
documented the field as *"Metal always supports FP64 on macOS"* — the exact
opposite of the truth, eleven lines above the `supports_fp64 = false` that sets
it. A comment that says the opposite of its code is not a documentation nit; it
is how a defect gets read past. Corrected.

#### The class, swept

The point fix closes one arm. The class is *any place a requested element type
is mapped to a narrower one without a refusal*, and it is now swept by
`sarek/tests/codegen_golden/test_backend_type_width_totality.ml`: for every
backend and every scalar IR element type, the mapper must either raise a
diagnostic or name a device type of exactly `Sarek_ir_layout.scalar_size` bytes.
That file is the backend twin of `sarek/tests/unit/test_type_width_totality.ml`,
which enforces the same rule on the *front* half (source type → IR element
type) and stops at the IR.

The full sweep found **two** silent narrowings, both on Metal, and **no others**:

| backend | `TInt64` | `TFloat16` | `TFloat64` | `TBool` |
|---|---|---|---|---|
| **Metal** | `long` ✓ 8 | refused | **`float` ✗ 4 vs 8** | **`bool` ✗ 1 vs 4** |
| **CUDA** | `long long` ✓ 8 | `__half` ✓ 2 | `double` ✓ 8 | `int` ✓ 4 |
| **OpenCL** | `long` ✓ 8 | refused | `double` ✓ 8 | `int` ✓ 4 |
| **GLSL** | `int64_t` ✓ 8 | refused | `double` ✓ 8 | `bool` ✓ 4 (see below) |
| **WGSL** | refused | refused | refused | `bool` — no memory form (see below) |
| **PTX** | `.u64` ✓ 8 | refused | `.f64` ✓ 8 | `.u32` ✓ 4 |

`TBool` is the second Metal narrowing and it is reachable — and it is a codegen
bug, not a capability gap, by the test `docs/design/capability-model.md` §5.1
sets out: not "is it silent" nor "is it a width mismatch" (both are equally true
of it and of f64), but **does a correct lowering exist in the target language?**
For f64 there is none; for bool there is. The host gives a
Sarek `bool` a 4-byte slot (`Sarek_ppx`'s field-size mapping, mirrored by
`Sarek_ir_layout.scalar_size TBool = 4`), MSL `bool` is **one** byte, and `bool`
is an accepted `[@@sarek.type]` record field type. Host `{bool;bool;int}` lays
out at 0/4/8, size 12; the emitted `typedef struct { bool a; bool b; int n; }`
lays out at 0/1/4, size 8. Metal now emits `int`, which is what CUDA and OpenCL
already did.

The other two `bool` spellings are **not** silent, and this was checked rather
than assumed:

- **GLSL** — glslang lowers a `bool` member of an std430 storage buffer to a
  32-bit uint, i.e. the host's own width. Verified on the emitted bool-record
  shader: `glslangValidator -V --target-env vulkan1.2` exits 0, and `spirv-dis`
  shows `%Flagged_0 = OpTypeStruct %uint %int`, `OpMemberDecorate ... Offset 0`
  / `Offset 4`, `ArrayStride 8`.
- **WGSL** — `bool` has no memory form at all there, and `naga` refuses it in a
  storage binding rather than choosing a width: *"The type is not
  host-shareable"*. A bool that reaches a buffer is a hard validation failure at
  shader-load time, never a wrong stride.

#### A fifth defect the same sweep turned up: GLSL int64 with no extension

Loud rather than silent, so not the backlog-141 narrowing class, but broken all the
same. `glsl_type_of_elttype` maps `TInt64` to `int64_t` at the correct width —
but that spelling does not exist under plain `#version 450`; it needs
`#extension GL_ARB_gpu_shader_int64 : require`. That extension was gated on the
two *float64* conditions only (the softmath helpers that bit-cast a double, and
a non-finite f64 literal spelled via `int64BitsToDouble`), because nothing in
the corpus reached int64 any other way. A kernel over a plain `int64 vector`,
with no float64 anywhere, therefore emitted:

```glsl
layout(std430, set=0, binding = 0) buffer Buffer_outv { int64_t outv[]; };
```

with no `#extension` line. `glslangValidator` rejects it —
`ERROR: :7: '' : syntax error, unexpected IDENTIFIER`, exit 2 — reproduced
before the fix and exit 0 after. The gate is `glsl-validate/int64_only_store`, a
validation-only kernel (no golden) whose only wide type is int64, which is
exactly the shape the corpus lacked. `Sarek_ir_analysis.feature` gained an
`Int64` constructor to drive it.

The *fp64* extension, by contrast, was already correct and is the measurement
that retracted a slice-1 claim: `glsl_header` takes
`~uses_float64:(kernel_uses_float64 k)` and emits
`#extension GL_ARB_gpu_shader_fp64 : require`, and `glslangValidator` accepts
the generated f64 kernel at exit 0. `docs/design/capability-model.md` §4
records the retraction in place. The *capability* half of int64-on-Vulkan —
`VkPhysicalDeviceFeatures.shaderInt64`, which is `Device_optional` and needs a
device probe — is backlog-142; only the emitter half is fixed here.

#### And one in the precedent test itself

`sarek/tests/unit/test_type_width_totality.ml` builds its source-type
enumeration by unfolding a total successor chain, specifically so the list
cannot drift from the type. Its `unfold` pushed the *successor* onto the
accumulator instead of the element just visited, so it dropped the **first**
element of every chain and duplicated the **last**: `T.Int` and
`T.TPrim T.TInt32` were never swept by any assertion in that file. Its
anti-vacuity check did not catch it, because one duplicate exactly compensated
for one omission — the lengths were still 7, 3 and 10.

Found by the backend twin, whose pinned-exemption-set assertion reported the
duplicate directly. Both `unfold`s are fixed, and both files now assert
**distinctness** as well as length, which is the property a count alone cannot
establish.

### 10.14 The f16 narrowing on Metal: **it does not fuse** (backlog-63 slice 5)

§10.9 listed f16 as the largest thing Metal had never been probed for, and
[`docs/design/f16-relaxed-accuracy.md`](design/f16-relaxed-accuracy.md) §2 has a
Metal row reading *"never probed → `Unknown` → refused"* for the same reason.
This is the measurement that fills it.

**Device and toolchain, as §1 corollary 3 requires:** Apple M4, macOS 15.6.1
(24G90), arm64, Apple clang 17.0.0 (clang-1700.0.13.5), Metal.framework from the
Command Line Tools SDK. Instrument:
`tools/probes/metal_f16_narrowing_probe.m`, standalone, in the shape of the two
Metal probes already in that directory. Evidence tier: **executed**, element-wise
over the whole finite binary16 domain — not a sampled maximum.

Two shapes, the two the other backends were swept on, each over all 63488 finite
binary16 inputs, deviation from the discipline (`S_strict`):

| variant | `f16(x*1.1)` | `f16(f16(x*1.1)+1000)` |
|---|---|---|
| plain (naive codegen) | **0 / 63488** | **0 / 63488** |
| `#pragma METAL fp contract(off)` | 0 / 63488 | 0 / 63488 |
| `#pragma clang fp contract(off)` | 0 / 63488 | 0 / 63488 |
| `volatile thread` barrier | 0 / 63488 | 0 / 63488 |
| `as_type` bitcast barrier | 0 / 63488 | 0 / 63488 |
| **FUSEDCTL — positive control** | **2912 / 63488** | **620 / 63488** |

**Metal is in the strict class.** No barrier is needed, none is emitted, and the
refusal of f16 in `Sarek_ir_metal` is currently over-broad in the same way §11.3
found pocl's and IGC's to be.

**Why the zeros are readable.** §11.5 is this document's own record of a null
that was not a result, so the control is the point. MSL has no `double`, so the
OpenCL probe's `(double)x * (double)1.1f` control is not expressible; the Metal
control reconstructs the exact product from a double-float pair and applies a
round-to-odd step before the narrowing. It is then **validated against the host
reference element-wise** — 0 of 63488 differ from
`S_fuse_mul_into_narrowing` — on **both** shapes. It reproduces **2912** on the
one-narrowing shape and **620** on the two-narrowing shape, the latter being the
figure independently measured on hiprtc/gfx1100, on rusticl/radeonsi, and by
`fusedctl` on Intel Arc. **Both zeros therefore sit next to a nonzero from the
same control**, on the same source, the same compile options and the same
dispatch as the variants reporting 0 — the discrimination is demonstrated per
shape, not carried from one shape to the other. A
host calibration runs first and refuses to print any device number unless the
binary16 round-trip is clean and the two models separate on 2912 and 620.

**The pragma does not govern this hazard — and does not need to.** `#pragma
METAL fp contract(off)` changes nothing on either shape, in either direction: it
does not move the plain kernel (already strict) and it does not disturb the
control (620 with and without). That is **not** the pragma being inert; §10.5
measures it taking `a*b+c` from 8773/8773 to 0/8773 on this device. Contraction
of `a*b+c` and absorption of a multiply into an f32→f16 narrowing are two
behaviours, the pragma is measured on one of them, and Metal exhibits neither.
`Sarek_ir_metal.metal_fp_contract_pragma` keeps its §10.5 justification
unchanged and gains no new one here.

**A defect this surfaced in the relaxed contract itself, not in Metal.**
`f16-relaxed-accuracy.md` §1.3 proposes `f16_relaxed_ceiling` — *"no admitted
deviation may exceed 1 ulp of the binary16 result, measured on the final
value"*. The validated control computes the admitted model exactly, so it can be
used to test the ceiling, and **the admitted model violates it**. At
`x = -907.5`: the exact product is `-998.25000216…`, whose binary32 rounding is
exactly `-998.25` — a binary16 tie in the binade [512, 1024) — which rounds to
even, giving `S_strict` `-998 + 1000 = 2`. The single-rounding model narrows the
exact product, is not at the tie, gets `-998.5`, and gives `1.5`. The deviation
*at the elided narrowing* is `-998` against `-998.5` — **exactly 1 ulp of
binary16 there**, as the derivation says. On the **final** value it is
**512 ulp** measured against 1.5, or **256 ulp** measured against 2.0; the count
depends on which of the two supplies the denominator, and a gate must say which.
Either way it is hundreds of ulps against a ceiling of one, because the `+1000`
cancels the leading bits away and re-scales the ulp.

The §1.3 derivation is sound where it is stated — an elided rounding moves a
value by at most half an ulp *at the elided step*. What does not follow is the
restatement on the final value, once a later operation can cancel. The ceiling
has to be evaluated at the narrowing where the rounding was elided. This is a
property of the contract and not of any device; it is recorded here because this
is where it was measured, and the design document's own §1.3 is where it needs
fixing.

**Determinism (§1.4 a and b):** the whole sweep is bit-identical across two runs
in separate processes.

**Not established.** Only the two shapes above — the 20-shape catalogue of
`docs/optimization/amdgpu-f16-fusion-shape-audit.md` has not been run on Metal.
Subnormal *inputs* are in the sweep (all 63488 finite values are), but no
subnormal-specific analysis was done. One device, one OS. And this measures the
**driver**, from hand-written MSL: it is not a codegen gate, exactly as §7(c)
says of the GLSL tripwire.

### 10.15 `simdgroup_matrix` on the M4: three configurations, none of them integer

The backlog-63 prerequisite, measured the same day with
`tools/probes/metal_simdgroup_matrix_probe.m` on the same device and toolchain.
Evidence tier: **executed**.

`simdgroup_matrix<T, Cols, Rows>` instantiates for exactly three configurations,
and every other candidate is a compile-time refusal with a named
`static_assert`, so the enumeration is closed rather than sampled:

| configuration | available |
|---|---|
| `half` 8×8 | **yes** |
| `float` 8×8 | **yes** |
| `bfloat` 8×8 | **yes** |
| 16×16, 8×16, 16×8, 4×4, 32×8, 8×32 (any type) | no — `_valid_simdgroup_matrix_size` |
| `char`, `uchar`, `short`, `int` 8×8 | no — `is_simdgroup_matrix_element<T>` |

`threadExecutionWidth` is 32, so one 8×8 fragment is one full simdgroup. GPU
families Apple1–9 and Metal3 are all supported and the runtime compiler accepts
MSL 2.4 through 3.2.

**There is no integer `simdgroup_matrix`, and that removes a fallback.**
`f16-relaxed-accuracy.md` §8 and §7 slice 4b keep an integer cooperative-matrix
path as the route that lands *under the existing strict contract* if the ACO
scalar shapes match no closed-form model — 12 of the 14 configurations RADV
advertises are integer. **That fallback is Vulkan-only.** backlog-63 has no
strict-contract route available to it at all; on Metal it is f16/bf16 or
nothing. `bfloat` 8×8 is available, is not in the current plan, and has not been
swept — nothing here says what it computes.

**What the f16 MulAdd computes.** 1024 independent 8×8×8 problems, 65536 output
elements, `D = A×B + C` with **`C` nonzero throughout**, against an exactly
computed host reference.

`f16-relaxed-accuracy.md` §5.1 records that an earlier draft of that section
bounded only the products and would have failed a correct result with a nonzero
`C`. The first version of *this* probe made the mirror-image mistake — it pinned
`C = 0` — and it cost a wrong conclusion, so it is worth stating plainly: with
`C = 0` the "C added first" and "C added last" accumulation orders are the **same
function**, and the probe reported 65536/65536 against "sequential binary32"
without being able to say which. A nonzero `C` separates them decisively. The
constant is therefore `γ_8` and not the `γ_7` of the `C = 0` degenerate case,
which §5.2 says must not be used unless `C` is pinned.

Two anti-vacuity problems were found and fixed before any number below was
trusted, and both are the same failure: a check that could not fail.

1. **The input set.** The first version drew operands as multiples of 2⁻⁹ in
   [−1, 1] — every product was then a multiple of 2⁻¹⁸ below 8, the sum of 8 of
   them was exactly representable in binary32, **every accumulation order
   agreed**, and it reported 64/64 exact while measuring nothing. The operands
   now carry a full 11-bit significand over an 11-wide exponent range, so
   accumulation order is observable: sequential and pairwise binary32 differ on
   **23291 / 65536**.
2. **The reference's own exactness.** §5.3 warns that binary64 is *not*
   sufficient in general and must be asserted. The harness now computes the term
   exponent span and refuses to print any device number unless it fits: measured
   **21 binades, needing 50 bits against binary64's 53**.

Both of §5.4's required controls run, each a deliberately wrong reference the
bound must reject: the **binary16 accumulator**, rejected on 65433 / 65536, and
the **`C`-dropping reference**, rejected on 65498 / 65536. The exact reference is
rejected by its own bound on 0.

| | `half8x8 × half8x8 → float8x8` | `half8x8 × half8x8 → half8x8` |
|---|---|---|
| bit-equal to **sequential binary32, `C` FIRST** | **65536 / 65536** | 7 / 65536 |
| bit-equal to sequential binary32, `C` last | 51850 / 65536 | 9 / 65536 |
| bit-equal to pairwise binary32, `C` last | 34914 / 65536 | 11 / 65536 |
| bit-equal to **sequential binary16, `C` first** | 9 / 65536 | **65520 / 65536** |
| bit-equal to pairwise binary16 | 10 / 65536 | 32712 / 65536 |
| bit-equal to the exact dot product | 538 / 65536 | 0 / 65536 |
| worst error / Σ\|terms\| | 2.67e-07 (`γ_8` = 4.7684e-07) | 2.12e-03 (`γ_8` = 3.9216e-03) |
| elements outside the bound | 0 / 65536 | 0 / 65536 |

**The f32-accumulate configuration matches a named closed-form model
element-wise on every element**: initialise the accumulator to `C`, then add the
eight products in index order, all in binary32. That is stronger than
`f16-relaxed-accuracy.md` §5 anticipated — §5 proposes a *bound* because the
accumulation order is implementation-dependent, and on this implementation the
order is observable and is pinned. §1.6's migration row and §6.1's corollary
apply directly: this configuration moves from Regime B to Regime A, and its
friction falls from a mandatory opt-in to a diagnostic, with no new decision
needed. **The closed form could not have been identified without §5.1's
insistence that the operation is `A×B + C`** — 65536 against 51850 is the whole
difference between naming the order and guessing it.

The model cannot separate "exact products then sequential adds" from an fma chain
and does not need to — an f16 × f16 product is exact in binary32 (11 + 11 = 22
bits inside 24), so those are the same function. Evidence tier for *that* step:
**by-construction**.

**The f16-accumulate configuration matches no closed-form model**: 16 elements in
65536 differ from sequential binary16, and match neither pairwise binary16 nor a
binary32 chain narrowed at the end. §5.4 recommends not admitting that
configuration on the width of its bound; the recommendation now also rests on the
measurement.

**Determinism (§1.4).** Bit-identical across two processes, and bit-identical
across threadgroup sizes 32 / 64 / 128 / 256 — 0/65536 differ in each case. §1.4
records that a coopmat result may legitimately move with the dispatch shape and
that nobody had measured whether it does. On this device, at this tile size, it
does not.

**Not measured: throughput.** Whether the M4 has dedicated matrix hardware
behind these three instantiations is not answered here and cannot be read off
availability; that needs a benchmark against a scalar kernel.

---

## 11. Measured on Intel: Meteor Lake Arc, ANV and the Intel Compute Runtime (backlog-123)

An Intel machine became available on 2026-07-27. It is the first Intel GPU this
project has ever executed on, and it closes three things at once: the df64 ANV
allowlist entry that was carried on a quotation, the `NoContraction` half of §6
that §7 lists as a hardware gap, and the "is the f16 narrowing fusion ACO's or
OpenCL's" question.

**The machine and the toolchains.** Intel Core Ultra 9 185H (Meteor Lake-P),
Intel Arc Graphics integrated GPU. Three distinct compilers, and every claim
below names which one it is about:

| stack | device string | compiler |
|---|---|---|
| **Vulkan** | `Intel(R) Arc(tm) Graphics (MTL)` | Mesa **ANV** 26.1.2-arch3.1, Vulkan 1.4.348, `driverID = VK_DRIVER_ID_INTEL_OPEN_SOURCE_MESA` |
| **OpenCL (GPU)** | `Intel(R) Arc(TM) Graphics` | Intel Compute Runtime (NEO) / **IGC** — `/usr/lib/intel-opencl/libigdrcl.so`, wholly independent of Mesa |
| **OpenCL (CPU)** | `Intel(R) Core(TM) Ultra 9 185H` | Intel oneAPI CPU runtime — `/opt/intel/oneapi/compiler/latest/lib/libintelocl.so` |

The two OpenCL platforms are separate ICDs; the probe below picks between them
with `OCL_ICD_FILENAMES`, since it scans the first platform that has devices.

### 11.0 What a Meteor Lake measurement does and does not establish

**It is not a check on the UHD 630 figures.** The ANV numbers this document and
`Sarek_df64.ml` previously quoted — mul/div ~5.8e-08 — are from **Intel UHD
Graphics 630 (CFL GT2)**, a Gen9.5 part. This machine is **Xe-LPG**: a different
architecture, a different generation, a different ISA, and a decade of driver
work in between. The quoted figure and the measured one **agree**, and that is
worth knowing, but the agreement is a second data point about ANV, not a
confirmation of the first. Had they disagreed, the finding would have been that
the deviation is generation-dependent, and neither number would have been
"corrected" by the other. Nothing below should be read as evidence about UHD
630, and the `is_anv_device` predicate — which matches both — remains a claim
about the *pair*, held open by whichever measurement is weakest.

### 11.1 df64 on ANV: the allowlist entry is real, and stays

`Test_helpers.df64_known_deviation`'s `Vulkan`/`mul|div`/`is_anv_device` arm had
never been executed. It is a **strict xfail**, so the two outcomes were "it
still deviates" or "the run goes red with STALE ALLOWLIST ENTRY and the arm gets
deleted". It deviates.

Measured on `Intel(R) Arc(tm) Graphics (MTL)`, Mesa ANV 26.1.2-arch3.1, Vulkan
1.4.348, 2026-07-27. Every figure is the **measured worst-case relative error
over that test's own input set** on that device and driver — a maximum observed,
not a bound proved:

| op | `test_df64` | `test_real64` (fallback-df64) | contract |
|---|---|---|---|
| mul | **5.84e-08** | **5.93e-08** | 1.42e-14 |
| div | **5.86e-08** | **5.83e-08** | 1.42e-14 |
| add | 5.33e-15 | 5.31e-15 | 7.11e-15 |
| sub | 6.51e-15 | 6.94e-15 | 7.11e-15 |
| sqrt | 9.57e-15 | 1.17e-14 | 1.42e-14 |
| of_i32 | 0 | — | 0 |

Both suites exit 0. `test_real64`'s **native** f64 path is clean on ANV (0 on
add/sub/mul/sqrt, 2.22e-16 on div), so this is specifically the df64 emulation,
not the device's binary64.

That is RADV's pattern exactly — mul and div collapse to float32 precision while
add, sub and sqrt meet the strict bound — and the RADV arm attributes it to a
GLSL `fma` that is not correctly rounded, which destroys TwoProd's error term
while leaving the paths that do not use `fma` intact.

**Contraction is now ruled out on ANV as well, rather than assumed.** §11.2
reports 0 of 7 contraction shapes contracted on this device with *and* without
`precise`, so the compiler is not fusing the multiply that closes
`quick_two_sum`. What remains is the `fma` explanation. Stated at its true
strength: this is an **inference from elimination plus an identical error
signature**, not a direct sweep of `fma` correct-rounding on ANV. The three
`fma` anchor triples that `test_vulkan_no_contraction` does check agree with
IEEE on this device — which is a 3-point sample, so it neither establishes
correct rounding nor contradicts the attribution.

**The predicate stays keyed on the vendor string, and here is the search that
justifies it.** `is_anv_device` matches `"intel"` in the device name rather than
a driver token, which would also match a future non-Mesa Intel Vulkan driver. A
real driver token *does* exist on this driver:
`VkPhysicalDeviceDriverProperties` reports `driverName = "Intel open-source Mesa
driver"` and `driverID = VK_DRIVER_ID_INTEL_OPEN_SOURCE_MESA`, and that struct
is Vulkan 1.2 core, so it is reachable on every device this backend supports.
**Sarek does not plumb it**: `sarek-vulkan/Vulkan_api_device.ml` fills
`Device.name` from `VkPhysicalDeviceProperties::deviceName` and queries nothing
else, and ANV puts no driver token there (`Intel(R) Arc(tm) Graphics (MTL)` —
just as UHD 630 gave `Intel(R) UHD Graphics 630 (CFL GT2)`). Narrowing the key
is therefore a **Vulkan-backend change**, not a test change, and it is
deliberately not bundled into a measurement commit. Until then the vendor string
is the strongest key available at that call site, and the failure direction is
safe: a non-Mesa Intel Vulkan driver that *meets* the contract trips the
strict-XPASS branch and names the arm to delete, rather than passing silently.

### 11.2 `NoContraction` on ANV: the §6 gap, closed

§7's first "still open" bullet — *"No Intel GPU: the ANV half of §6 is unrun"* —
is now **CLOSED**. `sarek/tests/e2e/test_vulkan_no_contraction.ml`, the same
twelve-shape experiment §6 describes, run unchanged on ANV:

| | no `precise` | `precise` |
|---|---|---|
| contraction shapes contracted | **0 of 7** | **0 of 7** |
| reassociation shapes reassociated | 0 of 2 | 0 of 2 |
| explicit `fma()` controls fused | 4 of 4 | 4 of 4 |

Identical to RADV, including the four explicit-`fma()` integrity controls
firing, which is what shows the harness was handed a genuinely fused value to
compare against. As required by §6's own caveat, the contracted target is the
device's own `fma` result and not an IEEE model.

So the same reading as RADV, and for the same reason: **INCONCLUSIVE for
contraction** on this family of shapes. ANV does not contract them even when
nothing forbids it, so `precise` has nothing to suppress and the experiment
cannot say whether it would be honoured if it did.

**The follow-up that settled it negatively on RADV settles it positively here.**
§6 closes the RADV case with the shape RADV *does* want to contract —
`float16_t(x * 1.1)`, a multiply absorbed by the conversion consuming it — where
RADV emits `v_fma_mixlo_f16`, produces byte-identical ISA with and without the
decoration, and disagrees with the interpreter on 2912/63488. Run on ANV via
`sarek-vulkan/test/test_vulkan_f16_tripwire.ml`, with its scope predicate
temporarily widened to reach a non-RADV device:

| | no `precise` | `precise` |
|---|---|---|
| disagreements with Sarek's f16 discipline, all 63488 finite binary16 inputs | **0** | **0** |
| disagreements with the single-rounding (fused) model | **2912** | **2912** |

The 2912 is the calibration and it matters: the harness's fused model separates
from the discipline on exactly the number RADV produces, on this device, on this
run — so the 0 is a live null and not a broken sweep.

**ANV does not perform the combine.** Therefore:

- The campaign note claiming **ANV ignores `NoContraction`** is **not
  supported**, on either shape family. It never produced a contracted result.
- Nothing here shows ANV *honours* it either. On both families ANV declined to
  contract when it was free to, so the decoration was never load-bearing. The
  honest status of `precise` on ANV is the same as its status on RADV's f32
  shapes: **inert**, because nothing needs preventing — with the difference that
  on RADV there is a known combine where the decoration is demonstrably
  *ignored*, and on ANV no such combine has been found.

This does **not** license enabling f16 on Vulkan. `Sarek_ir_glsl`'s refusal is a
backend-wide code path, and it is still warranted by RADV; the refusal would
have to become per-driver to exploit this, and the `shaderFloat16` plumbing §7
notes as missing is still missing.

### 11.3 The f16 narrowing on Intel OpenCL: **it does not fuse** — the AMD localisation confirmed

This is the question §2's pocl row calls *"what localises the defect"*. The
reading before this run: rusticl and HIP are two front ends onto one ACO
backend (620/63488 identical disagreements), RADV is a third and fuses worse,
and pocl on x86 does not fuse — so the locus is ACO. pocl is a single negative,
and pocl is also an LLVM-based CPU stack, which is not very far from ACO's
family. Intel's IGC is a wholly independent implementation on a real GPU.

> **The premise of that reading was wrong, and the correction is recorded in §2
> ("Two AMD compilers").** hiprtc does *not* compile through ACO: it compiles
> through **LLVM's AMDGPU backend**, while rusticl/radeonsi and RADV compile
> through **ACO**, Mesa's shader compiler. Those are two different compilers.
> The measurements are untouched — 620/63488 on both, same first divergence at
> `x = 5.68359375` — but what they establish is **not** "one ACO bug seen
> through two front ends". It is that the f32-multiply-into-f16-narrowing fusion
> is present in **both** AMD GPU compilers: an AMD-toolchain-wide behaviour.
> Everything below stands as measured; read "ACO" in it as "the AMD GPU
> compilers" wherever it is used as a *locus* rather than as the name of the
> specific Mesa component a given figure came from. Evidence tier for the
> correction: **by-construction** (toolchain composition); no figure changes
> tier.

`tools/probes/opencl_f16_contraction_probe.c`, unmodified apart from the new
control described below, exhaustive over all 63488 finite binary16 inputs, on
the same `f16(f16(x*1.1) + 1000)` shape:

| variant | Arc Graphics (IGC) | Core Ultra 9 185H (oneAPI CPU) |
|---|---|---|
| **`fusedctl` — positive control** | **620 / 63488** | **620 / 63488** |
| `plain` (naive codegen) | **0 / 63488** | **0 / 63488** |
| `fpcontract` (`#pragma OPENCL FP_CONTRACT OFF`) | 0 | 0 |
| `convert` (`convert_half_rte`) | 0 | 0 |
| `bitcast` / `bitcast2` | 0 / 0 | 0 / 0 |
| `volatile` (volatile `__local` scalar) | **4774** | 0 |
| `vpriv` (volatile `__private` pointer) | **4774** | 0 |
| `vglobal` (volatile `__global` round-trip) | **4774** | 0 |
| `vlocal` (volatile LDS round-trip) | **4774** | 0 |

**The calibration is exact, and it is why the 0 is a result.** A null from an
exhaustive sweep is worth nothing without showing the same sweep can go nonzero
on the same device and run. `fusedctl` is a new variant added for this: a kernel
that performs the fusion *deliberately*, by computing `(half)((double)x *
(double)1.1f)` — both operands are exactly representable in binary64, so that is
the exact f32 product narrowed in **one** rounding, which is precisely what
`v_fma_mixlo_f16` does. It is checked against the untouched host reference. On
both Intel devices it reports **620/63488, first divergence at x=5.68359375
(device 1006.5, reference 1006)** — the same count *and the same first
divergence point* as ACO. So the harness, on this device and toolchain,
reproduces the known positive to the input, and `plain = 0` is a genuine null.

**Result: neither Intel compiler fuses the f32 multiply into the f32→f16
narrowing.** The naive codegen is already correct on both. This is the **second
independent negative** for a non-AMD OpenCL implementation, and the first on a
GPU with a vendor compiler sharing no lineage with either AMD compiler. The
reading in §2 is confirmed rather than widened on the axis it actually
constrains: **the defect is AMD's GPU compilers', not OpenCL's and not
SPIR-V's.** Had Intel fused, the consequence would have been the opposite one —
a widespread vendor behaviour Sarek must defend against everywhere.

What this run does *not* narrow is which AMD compiler. Both of them fuse
(§2), so a negative from a third vendor cannot separate them, and no
measurement here was ever going to: separating ACO from LLVM/AMDGPU needs two
AMD stacks on the same card, which is exactly the pair already measured.

### 11.4 Unexpected: on IGC, the barrier that fixes ACO **breaks** a correct narrowing

The 4774 column above is a first-contact finding nobody was looking for.

On the Arc GPU, all four `volatile`-based barriers — the ones measured in slice
2a to be the *only* things that defeat the fusion on rusticl — take a narrowing
that is **correct** on IGC and make it **wrong on 4774 of 63488 inputs**. First
divergence at `x = 0.681640625`: the device returns 1000.5 where the discipline
mandates 1001, which is the signature of the **intermediate binary16 narrowing
being dropped entirely** (with the mid rounding, `0.75 + 1000 = 1000.75` ties to
even → 1001; without it, `1000.7498… → 1000.5`). It is the same failure mode,
and the same 4774, that §2 records for RADV when a `volatile` SSBO round-trip is
applied to the f32 intermediates — reached on a completely unrelated compiler,
which is what makes the interpretation credible.

The Intel **CPU** runtime does not do this, so it is IGC-specific, not an Intel
stack property.

**Where it happens.** The SPIR-V is faithful for both kernels. Compiled with
`ocloc -device mtl` and disassembled with `spirv-dis`, the plain kernel carries
`OpFMul` → `OpFConvert %half` → `OpFConvert %float` → `OpFAdd` → `OpFConvert
%half`, and the barriered kernel carries the identical chain with the two
`OpStore`/`OpLoad` pairs marked `Volatile|Aligned`. Every mandated rounding is
present in both. So the divergence is introduced **below SPIR-V, inside IGC**,
folding the `f32(f16(x))` pair across the volatile boundary — a fold that is
only valid when the value is exactly representable in binary16. Evidence tier:
**compiler-output** for that localisation, not machine-code — Gen ISA
disassembly needs `iga64`, which is not on this machine, and `ocloc disasm`
without it emits only the ELF section list.

**Policy consequence, and it is not a small one:** a contraction barrier
measured on one backend is not portable, and applying one unconditionally can
*introduce* the defect it was written to prevent. Any future decision to emit a
narrowing barrier in `Sarek_ir_opencl` must be per-implementation and gated on
an exhaustive agreement sweep for that implementation, never applied backend-wide
on the strength of an ACO measurement.

**What Sarek actually ships today, checked rather than assumed (backlog-144).**
The obvious reading of the paragraph above is that the barrier needs to become
conditional on the shader compiler, and that Sarek therefore needs a way to
*identify* ACO at runtime — the gap §5 of
[`docs/design/capability-model.md`](design/capability-model.md) lists as **NOT
expressible**. That work is not needed, because the barrier is already scoped,
by two facts that are structural rather than probabilistic:

1. **The barrier is emitted from one site and it is preprocessor-scoped.**
   `Sarek_ir_cuda.sarek_f32_barrier_decl` puts the `asm volatile("" : "+v"(x))`
   body inside `#if defined(__HIP_PLATFORM_AMD__)`; the other arm is a bare
   identity (§4). Its only runtime consumers are `Hip_plugin` (hiprtc) and
   `Cuda_plugin` / `Cuda_c_plugin` (nvrtc). Evidence: **by-construction**.
   (That guard read `defined(__HIP__) || defined(__HIP_PLATFORM_AMD__)` until
   §11.4a below.)
2. **IGC cannot receive that source.** f16 is refused outright by
   `Sarek_ir_opencl`, `Sarek_ir_glsl`, `Sarek_ir_metal` and `Sarek_ir_wgsl` — at
   the per-element-type arm *and* at a whole-kernel gate, across every public
   `generate*` entry point — and `Sarek_ir_ptx` refuses it too. §11.4's 4774 was
   measured on `tools/probes/opencl_f16_contraction_probe.c`, a research probe,
   not on generated Sarek code. Evidence: **by-construction**, machine-checked by
   `sarek/tests/codegen_golden/test_cuda_f16_golden.ml`.

**A device-string denylist or a runtime ACO probe would both be weaker than what
is already there.** The `#if` is the compiler identifying *itself* at the moment
it compiles the source; a `CL_DEVICE_VENDOR` match or a `VK_DRIVER_ID` match is a
guess about a stack from the outside, and a boot-time fusion probe measures a
device that never sees this code. Neither would be consulted on the only path
that emits the barrier. **The right shape of the backlog-144 fix was therefore a gate,
not a mechanism** — the scoping was correct and unpinned, and "correct and
unpinned" is how §11.5's own tripwire came to encode a wrong claim.

That gate exists now. `test_f16_barrier_is_amd_scoped` requires the opacity body
to lie between the AMD guard and its `#else`, and the non-AMD arm to contain no
`asm`/`volatile` once comments are stripped. It was **proved red** by giving the
non-AMD arm the AMD `"+v"` barrier — a mutation the pre-existing `"+f"`
assertion does not see, and under which that new case is the only failure.

### 11.4a The guard now names the platform, and the hazard it was recorded for does not exist (backlog-146)

§11.4 closed with this, and it is the sentence this section exists to correct:

> **Not verified, and deliberately not fixed here:** the guard's
> `defined(__HIP__)` disjunct is also true under `__HIP_PLATFORM_NVIDIA__`,
> where `"+v"` is not a valid constraint.

The second half is right and is now measured. The first half is **not supported
by the toolchain's own headers**, and recording an unverified hazard is only
useful if the hazard is stated correctly — this one was not.

**What was measured, on this host (ROCm 7.2.53211, `libhiprtc.so.7`),
reproducer `tools/probes/hip_macro_probe.c`:**

| probe | result | tier |
|---|---|---|
| macros hiprtc predefines | `__HIP__` **defined**, `__HIP_PLATFORM_AMD__` **defined**, `__HIP_PLATFORM_NVIDIA__` **not**, legacy `__HIP_PLATFORM_HCC__` **not** | executed |
| `#if defined(__HIP__) \|\| defined(__HIP_PLATFORM_AMD__)` (old) | selects the AMD arm, `"+v"` compiles | executed |
| `#if defined(__HIP_PLATFORM_AMD__)` (new) | selects the AMD arm, `"+v"` compiles | executed |
| a guard nothing defines (**liveness control**) | takes the other arm — `error: NON_AMD_ARM_TAKEN` | executed |
| `"+v"` under `clang --target=nvptx64-nvidia-cuda` | **rejected** — `invalid output constraint '+v' in asm`; `"+f"` accepted | executed |
| `clang -x hip --offload-arch=sm_61` | **refused** — `unsupported HIP gpu architecture`, on ROCm clang 22.0.0git *and* upstream clang 22.1.6 | executed |

The arm-selection probes carry their own control: the non-AMD arm holds an
`#error`, so "it compiled" is what proves the AMD arm was taken rather than
merely that nothing broke — and the last row shows the same harness reporting
the other arm on the same run, so `COMPILES` is a live result rather than a
`#error` that turned out to be unreachable.

**Why `__HIP__ && __HIP_PLATFORM_NVIDIA__` cannot arise.** `hip/hip_common.h`
auto-enables `__HIP_PLATFORM_AMD__` whenever `__clang__ && __HIP__`, and
auto-enables `__HIP_PLATFORM_NVIDIA__` only for `__NVCC__`, or for clang-CUDA
**without** `__HIP__`; `hip/linker_types.h` then hard-`#error`s unless exactly
one platform macro is set. So under HIP's own headers `__HIP__` **implies** the
AMD platform, and the NVIDIA-platform route (`hipcc` → `nvcc`, or the
`nvidia_detail/` wrappers) is the one on which `__HIP__` is never defined at all.
Evidence: **by-construction**, reading the shipped headers.

**Decision: narrow the guard anyway, to `#if defined(__HIP_PLATFORM_AMD__)`.**
This is a **clarity change, not a bug fix**, and it is worth being exact about
which:

- *It fixes no reachable defect.* Nothing was mis-served by the old guard on any
  configuration reachable with the compilers on this host.
- *It cannot lose the AMD arm.* Measured directly above, and structurally
  guaranteed by `hip_common.h`'s implication.
- *It removes an ambiguity that has now cost two rounds of analysis.* The asm
  body is AMD-ISA-specific; the guard should name the target, not the source
  language. A reader who sees `__HIP__` in a guard around AMDGPU inline asm is
  right to worry, and twice now has.
- *The residual risk is named rather than absorbed:* ROCm older than the 4.x
  `__HIP_PLATFORM_HCC__` → `__HIP_PLATFORM_AMD__` rename would define only the
  legacy macro. No such toolchain exists here, so this is **unverified**. Its
  failure direction is the bad one — the AMD arm is skipped and the f16
  discipline fails *silently*, not loudly. That is why the guard is pinned by a
  test rather than left to review.

**The gate.** `test_f16_barrier_is_amd_scoped` gains a third assertion, on the
guard **line** rather than the whole source: the barrier's `#if` must not
contain `defined(__HIP__)`. (The f16 *include* above it legitimately keys on
`__HIP__` — there the question really is "is this HIP", and both arms of getting
it wrong fail loudly at compile time.) Proved red by restoring the disjunction,
in **both orderings**, because they are caught by different assertions and only
one of them exercises the new code:

| mutation | case that went red | message |
|---|---|---|
| `defined(__HIP__) \|\| defined(__HIP_PLATFORM_AMD__)` | `f32 barrier is scoped to the AMD toolchain` | *the f32 barrier must be emitted under the AMD toolchain guard `"#if defined(__HIP_PLATFORM_AMD__)"`; it was not found at all* — the **pre-existing** exact-guard assertion, which the leading disjunct breaks before the new check is reached |
| `defined(__HIP_PLATFORM_AMD__) \|\| defined(__HIP__)` | same case | *the barrier guard must not key on `__HIP__`* — the **new** assertion; this ordering is the one that slips past the exact-guard search |

One failing case in each run, out of 114 in that suite, and nothing else moved.
The second row is the one that matters: without it the new check would be
unreachable, since the first assertion would always fire first — a check that
cannot fail, in the shape a reviewer would not notice.

**Still unverified after this**, and settleable in one sitting by anyone with the
toolchain: what a genuine HIP-on-NVIDIA compile predefines. The prediction from
the headers is that it defines neither `__HIP__` nor `__HIP_PLATFORM_AMD__`
(the compiler is `nvcc`), so the identity arm is taken and nothing changes.
Settling it needs no new code: build `tools/probes/hip_macro_probe.c` against a
HIP-on-NVIDIA install and read the four macro rows.

### 11.5 The defect this surfaced in our own gate

`dune runtest` across `sarek`, `sarek-opencl`, `sarek-vulkan` and `spoc` on this
machine produced **exactly one failure**, and it was ours.

`test_opencl_f16_tripwire`'s locus cross-check —
`non_aco_implementations_do_not_fuse`, the guard §2's pocl row leans on — went
red on Intel Arc and reported:

> THE LOCUS-IS-ACO CLAIM IS NOW TOO NARROW. `Intel(R) Arc(TM) Graphics` is NOT
> an ACO device, yet 4774 of 63488 finite binary16 inputs differ between the
> naive and barriered narrowings — **it fuses too.**

(That is the text as it stood. The check's wording was corrected again under
backlog-145 — "not an ACO device" is the right *scope* statement but the wrong *locus*
statement, since the locus is both AMD GPU compilers and the `"ACO"` key selects
Mesa stacks only. §2, "Two AMD compilers".)

Both sentences are false, and §11.3 shows the plain kernel is correct on all
63488 inputs. The check compared the plain kernel against the **barriered**
kernel with no oracle. Two kernels computing the same expression that disagree
prove one of them is wrong; they do not say which. On ACO the barriered one is
right, which is where the reading came from — the file described the barriered
kernel as *"simultaneously the control and the oracle"*. On IGC the barriered
one is the wrong one, so the check saw the 4774, blamed the plain kernel, and
instructed the reader, in its own failure text, to weaken a documented claim
that is correct.

Its self-check did not catch this, and the reason is instructive: the barriered
kernel's correctness was pinned at **one hand-computed input**, `x = 1.0`. Intel
passes that point — at `x = 1.0` the dropped rounding does not change the final
result — while being wrong on 4774 others. A pin a broken oracle passes is not a
check.

Fixed here, and the fix is a real oracle rather than a narrower predicate:

- `test_opencl_f16_tripwire` gains the **host binary16 reference** ported from
  its Vulkan sibling (`ref_discipline` / `ref_fused`), which needed one from the
  start because on RADV no affordable barrier works at all.
- Two **calibration** cases now run on every invocation, host-only so they also
  run on GPU-less CI: the host rounding round-trips every finite binary16 input,
  and the two host models separate on exactly **620** — the figure independently
  measured on hiprtc/gfx1100, on rusticl/radeonsi, and now by `fusedctl` on
  Intel Arc. A null from the locus check is only readable because that positive
  reproduces alongside it.
- The **oracle-validity** check is now an exhaustive sweep of the barriered
  kernel against the discipline, replacing the one-point pin. On both ACO
  devices it is 0/63488, so the barrier really does work there — previously
  assumed, now measured.
- The **locus check** compares the plain kernel against the discipline, not
  against the barrier. It reports the barrier's own deviation alongside, without
  asserting on it: on Intel that number is 4774 as shipped, and asserting it
  would pin a permanent red with no action available. The place the barrier's
  validity is load-bearing is the in-scope path, and that *is* asserted.

Verified on both sides after the change: green on RX 7900 XTX + Raphael iGPU
(rusticl/radeonsi, ACO), where the in-scope tripwire still reports 620/63488 and
the barriered sweep is 0/63488; green on both Intel devices, where the locus
check reports `plain vs discipline 0/63488` and prints the 4774.

> **A construction argument is not a measurement — recorded because it looks
> obviously right.** The first attempt at a universal oracle split the
> expression across two kernel launches with the binary16 intermediate in a
> `__global` buffer, on the reasoning that no compiler can fuse across a
> dispatch. That reasoning is sound and the conclusion is wrong: the fusion is
> multiply-into-narrowing and **both** live in the first kernel, so the dispatch
> boundary separates the wrong pair. Measured on RX 7900 XTX, the two-pass
> construction reproduces ACO's fused answer exactly — 620/63488 from the
> discipline. It was caught only because it was swept against a device whose
> answer was already known.

### 11.6 Everything else the first Intel contact produced

Nothing else. Beyond the locus check, the full `dune runtest` across `sarek`,
`sarek-opencl`, `sarek-vulkan` and `spoc` is green on this machine, including
`test_df64`, `test_real64`, `test_vulkan_no_contraction`,
`test_opencl_fp_conformance` and both f16 tripwires. `test_real64` reports
`fp64=true` on all three Intel devices and the native binary64 path is exact on
every op.

Two environmental notes, neither a defect in this repository:

- `dune build @all` fails in that opam switch for want of `js_of_ocaml`,
  `js_of_ocaml-ppx` and `qcheck-core`, which gate `sarek/core_js/webgpu`,
  `sarek/transpile/web` and `formal/codegen-ptx/test`. Nothing was installed;
  the measurements were built as explicit targets.
- `tools/probes/opencl_f16_contraction_probe.c` scans only the first OpenCL
  platform that reports devices, so on a multi-platform host it cannot reach the
  second without help. `OCL_ICD_FILENAMES` selects the ICD; the usage comment
  now says so.

### 11.7 What is still open after this

- **No Gen ISA.** §11.4's localisation of the IGC fold is compiler-output tier.
  `iga64` would settle it at machine-code tier.
- **`fma` correct-rounding on ANV is not directly measured.** §11.1's
  attribution is elimination plus signature. A dedicated sweep of GLSL `fma`
  against a correctly-rounded reference would settle it for both ANV and RADV,
  and neither has one.
- **Still one Intel generation.** Xe-LPG only. Nothing here constrains Gen9.5
  (the quoted UHD 630), Xe-HPG discrete Arc, or Xe2.
- **The `is_anv_device` predicate is still vendor-keyed**, pending the
  `VkPhysicalDeviceDriverProperties` plumbing described in §11.1.
- **AMDVLK and the proprietary AMD Vulkan driver remain unmeasured**, as §7
  notes — different SPIR-V consumers on the same GPU.

---

## 12. ACO's f16 models, measured element-wise — and the `precise` reconciliation (backlog-62 slice 1)

**Executed 2026-07-27** on this workstation. This section is the measurement
record for slice 1 of
[`docs/design/f16-relaxed-accuracy.md`](design/f16-relaxed-accuracy.md), which
that document structures as a **decision point**: §1.2 accepts a device result
only when it is bit-identical to a **named closed-form model**, and until now
no ACO figure in this document had ever been compared against one
element-wise. §2's rusticl row was a *count* and a *first divergence*; §6's
RADV row said the two-narrowing shape matched no model at all, and §9.3 of the
design named that the thing most likely to break the design.

**Both are now settled, and the answer is the good one on both stacks.** Every
kernel variant swept below matches exactly one named model on **63488 / 63488**
inputs, on **both** local devices, with 0 inputs matching no model.

Instruments, committed with this section:
`sarek-opencl/probe/probe_opencl_f16_model_agreement.ml`,
`sarek-vulkan/probe/probe_vulkan_f16_model_agreement.ml`, and the model set
they share, `tools/f16_model_set/f16_model_set.ml`. They are **executables, not
tests**: they measure a driver in order to decide whether a contract is
deliverable, and the gates defending the two refusals
(`test_opencl_f16_tripwire.ml`, `test_vulkan_f16_tripwire.ml`) are untouched.

### 12.1 The models

Four named, closed-form functions, each computed exactly on the host. Naming is
by **which mandated rounding is elided**, because §1.2 requires a member of the
admissible set to be a function, not a description of what a device did.

| model | what it elides |
|---|---|
| `S_strict` | nothing — the interpreter |
| `S_fuse_mul_into_narrowing` | the f32 rounding of the multiply, absorbed into the narrowing that consumes it |
| `S_absorb_all_into_final_narrowing` | multiply, intermediate binary16 narrowing **and** f32 add, all absorbed into the final narrowing: **one** rounding where the DSL mandates four |
| `S_f32_mul_then_absorb_add` | the intermediate narrowing and the add, with the multiply keeping its own correctly-rounded binary32 result |
| `S_drop_intermediate_narrowing` | the intermediate binary16 narrowing only; the add still rounds to f32 (the IGC signature of §11.4) |

**The arithmetic is not ordinary OCaml floats, and that is load-bearing.** Three
of these round to binary16 in a single step from a sum that binary64 cannot hold
exactly: the exact product `x · fl32(1.1)` is an integer multiple of `2^-47`
while the addend reaches `2^9`, so the exact sum spans up to 65 bits against
binary64's 53. Evaluating `p +. 1000.0` would round first, making the model a
*different function* — and it would differ from the intended one exactly at the
binary16 ties, which is where §1.3's counterexample lives. Every single-rounding
model therefore goes through `two_sum` (exact, unevaluated) and a
round-to-odd-style single-step rounding.

### 12.2 Calibration — run before any device number is read

Four host-only checks, all green, on every run:

1. `f16_bits` re-encodes all **63488** finite binary16 values to their own bit
   patterns.
2. `S_strict` and `S_fuse_mul_into_narrowing` separate on exactly **620** on
   `f16(f16(x*1.1)+1000)` — the figure independently reproduced on
   hiprtc/gfx1100, rusticl/radeonsi, `fusedctl` on Intel Arc and the M4's
   round-to-odd control (§10.14).
3. …and on exactly **2912** on `f16(x*1.1)` — RADV's recorded figure for that
   shape.
4. §1.3's counterexample reproduces: at `x = -907.5` the two models differ by
   **1 ulp at the intermediate narrowing** and by **512 ulp on the final
   value**.

**The full pairwise separation matrix is printed**, which is the guard against
the failure this project has already hit once in a neighbouring form: a model
set whose members coincide on the swept inputs reports "exact agreement" while
discriminating nothing. Over the finite binary16 domain the separations are
fixed numbers, and three of them are the figures this document already records:

| pair | inputs where they differ |
|---|---|
| `S_strict` vs `S_fuse_mul_into_narrowing` | 620 |
| `S_strict` vs `S_absorb_all_into_final_narrowing` | **5075** |
| `S_strict` vs `S_f32_mul_then_absorb_add` | **4776** |
| `S_strict` vs `S_drop_intermediate_narrowing` | **4774** |

5075, 4776 and 4774 are §2's recorded RADV counts for the plain kernel, the
`precise` kernel and the f32-barriered kernel respectively — **reproduced by
closed-form host functions before any device was touched**. One caution the
matrix also surfaces and which is not visible from the counts:
`S_f32_mul_then_absorb_add` and `S_drop_intermediate_narrowing` differ on only
**2** of 63488 inputs, so telling those two apart rests on two inputs. It is a
real discrimination — both are exercised below and land on opposite sides — but
it is thin, and a future variant of either model should not be believed to be
distinguished by a sweep that does not hit those two.

### 12.3 OpenCL / rusticl — element-wise, and the count-agreement upgraded

**Executed on rusticl/radeonsi, Mesa 26.1.4-arch3.1, DRM 3.64, kernel
7.1.2-3-cachyos, on both local devices**: `AMD Radeon RX 7900 XTX (radeonsi,
navi31, ACO, …)` and `AMD Ryzen 9 7950X 16-Core Processor (radeonsi,
raphael_mendocino, ACO, …)`. Identical results on both.

| kernel | model matched | disagreements |
|---|---|---|
| `f16(x*1.1)`, plain | **`S_fuse_mul_into_narrowing`** | **0 / 63488** |
| `f16(f16(x*1.1)+1000)`, plain | **`S_fuse_mul_into_narrowing`** | **0 / 63488** |
| GREEN CONTROL — `volatile __local` round-trip, both shapes | `S_strict` | 0 / 63488 |
| POSITIVE CONTROL — deliberate fusion | `S_fuse_mul_into_narrowing` | 0 / 63488 |

Evidence tier: **executed, element-wise over the whole finite binary16 domain**.

Two things this adds beyond upgrading the tier:

- **The one-narrowing shape had never been swept on rusticl.** It reports
  **2912 / 63488** against the discipline — the same figure as RADV, and exact
  agreement with the same model. The two ACO front ends behave identically on
  both shapes.
- **The `fusedctl` control had to be rebuilt.**
  `tools/probes/opencl_f16_contraction_probe.c` builds it on `double` and says
  a device without `cl_khr_fp64` should fail loudly. **rusticl on this box is
  such a device** — it does not advertise the extension and the build is
  rejected. The control here carries the exact product as an unevaluated f32
  pair (Dekker's `twoProd` via `fma`) and rounds to odd before narrowing, the
  same construction the Metal probe needed because MSL has no `double`
  (§10.14). Round-to-odd then round-to-nearest is exact here because binary32
  has 24 significand bits, binary16 has 11, and 24 ≥ 2·11 + 2 — with no margin,
  which is why it is stated rather than assumed.

### 12.4 Vulkan / RADV — the two-narrowing shape matches a model, and `precise` is reconciled

**Executed on RADV, Mesa 26.1.4-arch3.1, Vulkan 1.4.354, on both local
devices**: `AMD Radeon RX 7900 XTX (RADV NAVI31)` and `AMD Ryzen 9 7950X
16-Core Processor (RADV RAPHAEL_MENDOCINO)`. Identical results on both.

| kernel | model matched | vs `S_strict` |
|---|---|---|
| `f16(x*1.1)`, plain | **`S_fuse_mul_into_narrowing`**, 0 / 63488 | 2912 |
| `f16(x*1.1)`, `precise` | **`S_fuse_mul_into_narrowing`**, 0 / 63488 | 2912 |
| `f16(f16(x*1.1)+1000)`, plain | **`S_absorb_all_into_final_narrowing`**, 0 / 63488 | **5075** |
| `f16(f16(x*1.1)+1000)`, `precise` | **`S_f32_mul_then_absorb_add`**, 0 / 63488 | **4776** |
| `f16(f16(x*1.1)+1000)`, volatile SSBO on the **f32 intermediates only** | **`S_drop_intermediate_narrowing`**, 0 / 63488 | **4774** |
| GREEN CONTROL — f16 **bit pattern** through the SSBO, both shapes | `S_strict`, 0 / 63488 | 0 |
| POSITIVE CONTROL — deliberate fusion, both shapes | `S_fuse_mul_into_narrowing`, 0 / 63488 | — |

Evidence tier: **executed, element-wise**, plus **machine-code** for the
reconciliation below.

**Every one of §2's RADV numbers is now a named function.** 5075, 4776 and 4774
were three unexplained counts; they are `S_absorb_all_into_final_narrowing`,
`S_f32_mul_then_absorb_add` and `S_drop_intermediate_narrowing`, each matched
bit-for-bit on all 63488 inputs on two devices.

#### The reconciliation, at machine-code tier

§6 records that `precise` produces **byte-identical ISA** on the one-narrowing
shape while §2 records it *changing the count* on the two-narrowing shape.
Those two facts are consistent, and the ISA says why. `RADV_DEBUG=asm`,
RX 7900 XTX (RADV NAVI31), one shader compiled per run so each dump is
attributable:

| variant | the arithmetic ACO emits |
|---|---|
| `f16(x*1.1)` plain | `v_fma_mixlo_f16 v2, 0xcccd, v1, neg(0) op_sel_hi:[0,1,1]` |
| `f16(x*1.1)` `precise` | **byte-identical to the above** |
| `f16(f16(x*1.1)+1000)` plain | `v_fma_mixlo_f16 v2, 0xcccd, v1, s1 op_sel_hi:[0,1,0]` — a **single** instruction taking `x`, `1.1` and `1000` |
| `f16(f16(x*1.1)+1000)` `precise` | `v_fma_mix_f32 v1, 0xcccd, v1, neg(0)` **then** `v_fma_mixlo_f16 v2, 1.0, v1, 0x63d0` — the multiply survives as its own correctly-rounded f32 operation, and `0x63d0` is binary16 `1000.0` |

> **The rule that explains all four.** `NoContraction` forbids contracting a
> multiply into an **addition**. It does not forbid a **conversion** absorbing
> what feeds it — that is a different combine and the decoration does not reach
> it.
>
> On the one-narrowing shape there is no addition at all, so the decoration has
> nothing to bind to: it is emitted, it binds nothing, and the ISA is
> byte-identical. On the two-narrowing shape there *is* one, once ACO has
> elided the intermediate narrowing — so the decoration bites, the multiply is
> materialised as `v_fma_mix_f32`, and the model moves from
> `S_absorb_all_into_final_narrowing` to `S_f32_mul_then_absorb_add`. **The
> decoration is honoured in both cases. It changes the answer only where there
> was something for it to forbid.**

This **corrects nothing in §6 and completes it**: §6's "inert here, ignored
there" reading of `precise` on RADV was right about the observations and
incomplete about the mechanism. "Ignored" is the wrong word — on the
one-narrowing shape the decoration is *inapplicable*, because a conversion
absorbing its operand is not a contraction. On the two-narrowing shape it is
**applied**, and still insufficient, because it constrains only one of the two
combines in play.

#### A candidate generative rule — stated as a hypothesis, not a measurement

All four observed behaviours are produced by one rule:

> *An f32→f16 narrowing absorbs the entire f32 expression tree feeding it,
> evaluating it exactly and rounding once — intermediate binary16 narrowings
> included, hence elided — cut wherever SPIR-V `NoContraction` forbids a
> multiply-add contraction, at which cut a correctly-rounded binary32 value is
> materialised.*

Evidence tier: **unverified as a general rule.** It is consistent with four
kernel variants across two expression shapes on two devices, and with the ISA
for each; it is not a measurement of the 20-shape catalogue in
[`docs/optimization/amdgpu-f16-fusion-shape-audit.md`](optimization/amdgpu-f16-fusion-shape-audit.md).
It matters because it is the difference between a contract keyed on a driver
and a lookup table keyed on every expression a user might write, and confirming
or breaking it is what the remaining 18 shapes would do.

### 12.5 §1.3's ceiling, applied — and why the correction was not academic

The design's `f16_relaxed_ceiling` is evaluated **at the narrowing where the
rounding was elided**, not on the final value. Both denominators are computed
and printed, because §1.3 requires a gate to say which value it measures the
ulp against.

| stack / shape | model | worst deviation **at the elided narrowing** | inputs over 1 ulp | the same deviation **on the final value** |
|---|---|---|---|---|
| rusticl, `f16(x*1.1)` | `S_fuse_mul_into_narrowing` | 1 ulp | 0 | 1 ulp |
| rusticl, `f16(f16(x*1.1)+1000)` | `S_fuse_mul_into_narrowing` | 1 ulp | 0 | **512 ulp** |
| RADV, `f16(f16(x*1.1)+1000)` plain | `S_absorb_all_into_final_narrowing` | 0.500044 ulp | 0 | **1.68e+06 ulp** |
| RADV, `f16(f16(x*1.1)+1000)` `precise` | `S_f32_mul_then_absorb_add` | 0.5 ulp | 0 | **1.68e+06 ulp** |
| RADV, f32-barriered | `S_drop_intermediate_narrowing` | 0.5 ulp | 0 | **1.68e+06 ulp** |

Both denominators — S_strict's value at the narrowing and the model's own value
there — give the same figures to the digits shown, so nothing here depends on
the choice; a gate must still name one.

**Every admitted deviation is inside the ceiling at the elided narrowing, and
every one of them blows through it by three to six orders of magnitude on the
final value.** §1.3 was corrected on a single Metal counterexample; this is the
correction reproduced on 63488 inputs on a second vendor's hardware, on models
that were not the one it was derived from. A final-value ceiling would reject
correct results from an admitted model on **every** ACO stack, not just at
`x = -907.5`.

The 0.500044 rather than 0.5 for `S_absorb_all_into_final_narrowing` is
expected and is not slack: that model presents the *exact* product where
`S_strict` presents a doubly-rounded one, so the gap is half a binary16 ulp plus
half a binary32 ulp, and `0.5 + 2^-13/2 ≈ 0.50006`.

### 12.6 What slice 1 did NOT measure

- **Two of the 20 emittable f16 shapes.**
  `docs/optimization/amdgpu-f16-fusion-shape-audit.md` enumerates 20; this
  measured `f16(x*1.1)` and `f16(f16(x*1.1)+1000)`. The other 18 are
  **unmeasured on ACO** and under §1.5 stay refused.
- **One constant, one addend.** `1.1` and `1000` throughout, as in every prior
  measurement of this hazard, so the counts are comparable — but the models are
  confirmed against one operand pattern, not a family.
- **Sarek-generated shaders.** Both probes compile hand-written kernels and
  therefore measure the driver, exactly as §7(c) says of the GLSL tripwire. A
  codegen-side gate is §7 slice 3.
- **Determinism beyond re-run stability.** The Vulkan sweep was executed in
  several separate processes during this work and reported bit-identical
  counts each time, which is §1.4(b); §1.4(a)'s in-process repeat and the
  dispatch-shape variation were not run.
- **Non-ACO Vulkan and non-Mesa OpenCL on AMD.** Unchanged from §7 and §11.7.

## 13. The other 18 shapes: §12.4's generative rule is FALSE, and the corrected one is local (backlog-151)

**Executed 2026-07-27** on this workstation, on all four local ACO devices. This
is the measurement §12.4 asked for in its last paragraph: it stated one
candidate rule that produced all five of slice 1's results —

> *An f32→f16 narrowing absorbs the entire f32 expression tree feeding it,
> evaluating it exactly and rounding once — intermediate binary16 narrowings
> included, hence elided — cut wherever SPIR-V `NoContraction` forbids a
> multiply-add contraction, at which cut a correctly-rounded binary32 value is
> materialised.*

— marked it **unverified as a general rule**, and named the remaining 18 of the
20 emittable shapes of
[`docs/optimization/amdgpu-f16-fusion-shape-audit.md`](optimization/amdgpu-f16-fusion-shape-audit.md)
as exactly what would confirm or break it. **It is broken**, on three shapes, on
both stacks, with the machine code for each. A corrected rule is stated in §13.4
and holds on 12 of 12 discriminating shapes on RADV and 11 of 12 on rusticl.

This matters structurally rather than as coverage. §12.4 put it plainly: the
rule "is the difference between a contract keyed on a driver and a lookup table
keyed on every expression a user might write". §13.5 answers that question, and
the answer is neither of those two.

Instruments, committed with this section:
`tools/f16_shape_catalogue/f16_shape_catalogue.ml` (the catalogue, the exact
evaluator and the rule as code), its host-only probe
`tools/f16_shape_catalogue/probe/probe_f16_shape_separation.ml`, and a
`--catalogue` mode added to slice 1's two probes rather than a third and fourth
probe. Raw output: `docs/measurements/f16-shapes-2026-07-27/`.

### 13.1 The models are GENERATORS, and they are pinned to slice 1 before anything is read

§12 hand-wrote seven closed forms for two shapes. Twenty shapes cannot be
handled that way without the answer to §13.5's question being decided by the
instrument. So each of §1.2's named members is restated as a **policy** — a
decision about which mandated roundings are elided — applied to an expression
tree. Five policies, twenty shapes.

The hinge is calibration, and it runs before any device is touched. The generic
evaluator must reproduce §12's **seven hand-written closed forms bit-for-bit on
all 63488 inputs**, on `f16(x*1.1)` and `f16(f16(x*1.1)+1000)`, and must
reproduce §12.2's separations — **2912, 620, 5075, 4776, 4774** — computed from
the generic evaluator rather than from those closed forms. The probes exit
non-zero and print nothing else if it fails. Without that, agreement on the
other 18 shapes would be a statement about a new instrument rather than about
the thing slice 1 measured.

The arithmetic is Shewchuk floating-point expansions plus a sticky flag for
division, for the reason §12.1 gives: three of the models round to binary16 in a
single step from a sum spanning 65 bits, and evaluating that in binary64 would
round first and make the model a *different function* — differing exactly at the
binary16 ties, which is where §1.3's counterexample lives.

The device source and the host model are generated from the **same tree**, so
they cannot drift apart and have a sweep measure the drift.

### 13.2 Eight of the twenty shapes cannot discriminate at all, and that is the first result

A shape on which all five policies are the same function over the whole finite
binary16 domain returns "matches `S_strict`, 0/63488" and that sentence measures
nothing. Twenty such rows would read as twenty confirmations. So the host pass
counts the **distinct functions** the policies induce per shape, and labels the
degenerate ones:

| distinct models | shapes |
|---|---|
| **1 — NON-DISCRIMINATING** | A1, A3, A4, A5, A9, A10, A13, C1 |
| 2 | A2, A7, A8, A11, A12, A14, A15 |
| 3 | A6 |
| 4 | B2, B3 |
| 5 | B1, B4 |

**So the honest denominator is 12, not 20.** A1 and C1 have no arithmetic to
elide; A3/A4 (add/sub into the narrowing), A9 (`sqrt(x*x)` is `|x|` exactly),
A10 (negation is exact) and A13 (`x*x` needs 22 bits and is exact in binary32)
are the four shapes the HIP audit already recorded as **demoted in the machine
code and clean in the numbers** — the same fact, reached from the model side.
A5 (`x/3`) joins them: rounding the exact quotient once and rounding it twice
agree on every one of the 63488 inputs.

### 13.3 What the two ACO stacks return, shape by shape

Both RADV devices returned identical results, and both rusticl devices returned
identical results. Counts are disagreements with `S_strict` out of 63488;
`R_*` are the corrected rule's instances (§13.4). Evidence tier: **executed,
element-wise over the whole finite binary16 domain**, on `AMD Radeon RX 7900 XTX
(RADV NAVI31)` and `AMD Ryzen 9 7950X (RADV RAPHAEL_MENDOCINO)`, radv / Mesa
26.1.4-arch3.1 / Vulkan 1.4.354, and on `AMD Radeon RX 7900 XTX (radeonsi,
navi31, ACO, DRM 3.64, 7.1.2-3-cachyos)` and its Raphael iGPU equivalent.

| shape | models | RADV plain | RADV `precise` | rusticl |
|---|---|---|---|---|
| A1 `narrow x` | 1 | 0 | 0 | 0 |
| A2 `narrow (x*1.1)` | 2 | **2912** `R_local_absorb` | 2912, same | 2912, same |
| A3 `narrow (x+1000)` | 1 | 0 | 0 | 0 |
| A4 `narrow (x-1000)` | 1 | 0 | 0 | 0 |
| A5 `narrow (x/3)` | 1 | 0 | 0 | 0 |
| A6 `narrow (x*1.1+1000)` | 3 | **484** `R_local_absorb` | **2** `R_local_absorb_nocontract` | **2** `R_local_absorb_opencl` |
| A7 `narrow ((x+1000)*1.1)` | 2 | **374** `R_local_absorb` | 374, same | 374, same |
| A8 `narrow (fma x 1.1 1000)` | 2 | **484** `R_local_absorb` | 484, same | 484, same |
| A9 `narrow (sqrt (x*x))` | 1 | 0 | 0 | 0 |
| A10 `narrow (0-x)` | 1 | **1 input matches NO model** | same | 0 |
| A11 `narrow (floor (x*1.1))` | 2 | **0 — `S_strict`** | 0 | 0 |
| A12 `narrow (x>0 ? x*1.1 : x*0.9)` | 2 | **0 — `S_strict`** | 0 | **2863** `R_local_absorb_opencl` |
| A13 `narrow (x*x)` | 1 | 0 | 0 | 0 |
| A14 `narrow (x*1.1*1.1)` | 2 | **308** `R_local_absorb` | 308, same | 308, same |
| A15 `narrow (x*1.1 + x/3)` | 2 | **744** `R_local_absorb` | **0** `R_local_absorb_nocontract` | **0**, same |
| B1 `narrow (narrow (x*1.1)+1000)` | 5 | **5075** `R_local_absorb` | **4776** `R_local_absorb_nocontract` | **620** `R_local_absorb_opencl` |
| B2 `narrow (narrow (x*1.1)*1.1)` | 4 | **17036** `R_local_absorb` | 17036, same | 17036, **the one residual** |
| B3 `narrow (narrow (x+1000)*1.1)` | 4 | **7803** `R_local_absorb` | 7803, same | **963** `R_local_absorb_opencl` |
| B4 `narrow (narrow (narrow (x*1.1)+1000)*1.1)` | 5 | **10707 — matched NO SINGLE MODEL under §12.4's rule** | **10707** `R_local_absorb_nocontract` | **1518** `R_local_absorb_opencl` |
| C1 f16→f16 copy | 1 | 0 | 0 | 0 |

Five of these counts land within one of the HIP audit's barrier-removed figures
for the same shape — A7 374/374, A8 484/484, A14 308/309, and on rusticl B1
620/620, B3 963/963, B4 1518/1518 — which is three unrelated compilers agreeing
on a shape-by-shape signature rather than on one number.

**Zero inputs matched no model anywhere except A10**, and A10 is §13.6.

### 13.4 Why §12.4's rule is false, at machine-code tier

Three shapes break it, and the disassembly says the same thing about all three:
the absorbing instruction is `v_fma_mixlo_f16`, which takes **one** multiply-add
and **one** conversion. It is a peephole. It cannot reach past an operation that
is neither.

`RADV_DEBUG=asm`, RX 7900 XTX (RADV NAVI31), one shader compiled per process so
each dump is attributable:

| shape / variant | what ACO emits | consequence |
|---|---|---|
| **A11** `narrow (floor (x*1.1))`, plain **and** `precise` — byte-identical | `v_fma_mix_f32 v1, 0xcccd, v1` **·** `v_floor_f32_e32 v1, v1` **·** `v_fma_mixlo_f16 v2, 1.0, v1, neg(0)` | the multiply is materialised as its own correctly-rounded f32; the narrowing absorbs only an **identity multiply by 1.0**, i.e. nothing. `S_strict`. |
| **A12** `narrow (x>0 ? x*1.1 : x*0.9)`, plain **and** `precise` — byte-identical | three `v_fma_mix_f32` **·** `v_cmp_lt_f32` **·** `v_cndmask_b32` **·** `v_fma_mixlo_f16 v4, 1.0, v1, neg(0)` | same: both products are materialised at f32, the select is not absorbable, the narrowing absorbs an identity multiply. `S_strict`. |
| **B4** plain | `v_fma_mix_f32 v1, 0xcccd, v1, s1` **·** `v_fma_mixlo_f16 v2, 0xcccd, v1, neg(0)` | **two** single-rounding events at **two different precisions**: `x*1.1+1000` contracted into one *binary32* fma, then the final `*1.1` absorbed into the narrowing. No whole-tree model is that. |

B4 is the strongest form of the failure. It matched
`S_absorb_all_into_final_narrowing` on 63480 of 63488 inputs and
`S_f32_mul_then_absorb_add` on 63486 — **no single member of the model set
describes it**, while every individual input matches some member. §1.2 requires
bit-identity to *one* member on *every* input, so that is a failure of the
contract as written, and it is the first time the two readings of §1.2 have come
apart. The harness reports them differently for that reason.

> **The corrected rule, and it is stated as a peephole because that is what the
> silicon has.**
>
> *Each f32→f16 narrowing absorbs the single f32 operation immediately feeding
> it — a multiply, an add/sub, or an explicit fma — evaluating it exactly from
> its operands and rounding once. Independently, a multiply feeding an addition
> is contracted into a single-rounded **binary32** fma. An intermediate binary16
> narrowing whose value is consumed only by f32 arithmetic may be elided. **Every
> other f32 operation keeps its own correctly-rounded binary32 result.**
> `NoContraction` removes the contraction clause only: it does not reach the
> narrowing's own absorption, nor a plain multiply, nor an explicit fma.*
>
> Evidence tier: **executed**, element-wise on 63488 inputs × 12 discriminating
> shapes × 2 variants × 4 devices, plus **machine-code** for eleven shapes.

It is **one semantics with three boolean knobs**, and the knobs are not a
degree of freedom left open — each is a measured property of a (driver,
decoration) pair:

| instance | contract mul+add? | elide intermediate narrowings? | sink a narrowing into a select's arms? | measured on |
|---|---|---|---|---|
| `R_local_absorb` | yes | yes | no | RADV, plain |
| `R_local_absorb_nocontract` | no | yes | no | RADV, `precise` — **what the shipped codegen runs under** |
| `R_local_absorb_opencl` | no | no | **yes** | rusticl |

All three reduce onto §1.2's existing named members on the two shapes slice 1
measured, and the calibration asserts it: on A2 all three are
`S_fuse_mul_into_narrowing`; on B1 they are `S_absorb_all_into_final_narrowing`,
`S_f32_mul_then_absorb_add` and `S_fuse_mul_into_narrowing` respectively. **The
four names slice 1 admitted are not four independent members. They are one rule
seen at three settings on two shapes**, and that could not be seen with two
shapes.

**Two things the knobs settle that slice 1 could not.**

- **`precise` does NOT cut an explicit `fma()`.** A8's plain and `precise`
  disassembly are byte-identical — one `v_fma_mixlo_f16 v3, v1, 0xcccd, v2` —
  so `NoContraction` binds nothing there. That is the literal reading of §12.4's
  own words: an author-written fma is not a *contraction* of anything. The
  eager reading, which also cuts it, was implemented first and refuted by this
  shape; the superseded run is kept at
  `docs/measurements/f16-shapes-2026-07-27/vulkan-radv-eager-cut.txt` so the
  choice reads as a measurement rather than as a preference.
- **The two ACO front ends are NOT the same function.** §12.3 concluded "the two
  ACO front ends behave identically on both shapes", which was true of the two
  shapes and is false in general. rusticl **keeps** the intermediate narrowing
  where RADV elides it (B1, B3, B4), does **not** contract multiply-add (A6,
  A15 — it matches the `precise` model with no decoration asked for), and
  **sinks** a narrowing into the arms of a select where RADV does not (A12:
  2863/63488 against RADV's 0). The backend is shared; the front ends are not,
  and the f16 allowlist has to be keyed on the pair.

**The one residual: B2, on rusticl.** `narrow(narrow(x*1.1)*1.1)` is the only
discriminating shape where the corrected rule mispredicts — rusticl elides the
intermediate narrowing there while keeping it on B1, B3 and B4. The ISA names
the mechanism and it is not absorption: both stacks emit a single
`v_fma_mixlo_f16 v2, 0xe148, v1`, whose literal `0x3f9ae148` is **binary32
1.21** — the compiler has **reassociated and constant-folded `1.1*1.1` across
the intermediate narrowing**, which requires eliding it. That is a
reassociation combine sitting beside the absorption one, and this document does
not model it. RADV's `precise` variant blocks it (A14 `precise` emits two
instructions where plain emits one), which is the expected behaviour of a
decoration that forbids reassociation.

### 13.5 Does the model set grow per shape? No — but only after the correction

This is the question §12.4 said the 18 shapes existed to answer.

**Under §12.4's rule as written, the answer would have been yes, and badly.** B4
plain matches no member, so a per-shape table would need a new entry for it; and
because that entry is "one binary32 fma then one absorbed multiply", the entry
for B4 tells you nothing about any other shape. That is a lookup table.

**Under the corrected rule, the answer is no.** §1.2's model set is
`{S_strict} ∪ {R_local_absorb(shape, knobs)}` — **two names and three bits**,
for twenty shapes and for any shape a user writes. The four members §1.2 lists
today are the instances that rule produces on the two shapes slice 1 measured,
which is why they read as four independent functions.

So the correction is not cosmetic and it is not a downgrade. It replaces four
measured points with a rule that generates them, and the rule was verified on
ten shapes it was not fitted to. The cost is that the rule is **per (driver,
front end, decoration)** rather than per backend, which §12.3 had provisionally
merged.

### 13.6 A10: RADV returns `-0` for `0.0 - x` at `x = +0`, and the barrier does not fix it

Shape A10 is `narrow(0. - x)`. It is non-discriminating — every model is the
same function — and on **one** input, `x = +0`, **both RADV devices return
`0x8000` (`-0`) where IEEE-754 round-to-nearest requires `+0`**: `(+0) − (+0)`
is `+0` in every rounding mode except round-toward-negative. rusticl returns
`+0` and is correct.

Three things make this worth a subsection rather than a footnote:

- **It survives the barrier.** The green control forces every temporary through
  the volatile SSBO and still returns `-0`, so this is not the absorption hazard
  and no amount of barriering removes it. The likely transform is `0 - x → -x`,
  which is valid only under a no-signed-zeros assumption nobody asked for.
- **Under §1.2 as written it is a FAILURE, not a relaxation.** "A result
  matching no member is a failure, however small the numeric difference" — and
  this difference is a sign bit on a zero, which is as small as a difference
  gets while still being one. It is not a candidate for the admissible set,
  because it is not a rounding at all.
- **A count-only sweep reports it as `1 / 63488` and nothing else.** It was
  identifiable only because the harness prints the device bit pattern next to
  what each model wanted, at the first input matching none. The equivalent
  hazard in the models themselves was also caught this way and is recorded in
  the code: an early revision of the division model dropped the sign of `-0/3`,
  and it showed up as A5 disagreeing with the device on exactly one input.

Not filed as a Sarek defect here — Sarek's f16 GLSL path is refused today and
§13 lifts nothing — but it is the shape a future slice-3 gate will trip on
first, and it is the reason the gate must sweep both zeros.

### 13.7 §1.3's ceiling is derived for ONE elided rounding, and B4 has two

§1.3 evaluates the ceiling **at the narrowing where the rounding was elided**,
deriving 1 ulp from "every deviation in the admitted class is the elision of
exactly one round-to-nearest step". B4 is the first shape with **two**
intermediate narrowings, and it therefore has two candidate evaluation points.
Measured at the outer of them, the admitted model exceeds the ceiling on **719
of 63488 inputs**, reaching 1638 ulp — because two elisions separate the model
from `S_strict` there, and the derivation covers one.

Measured at the **innermost** narrowing, where exactly one elision separates
them, the worst deviation is **0.500044 ulp with zero exceedances** — the same
figure and the same explanation as §12.5's `S_absorb_all` row (half a binary16
ulp plus half a binary32 one). On the **final** value the same deviation reaches
**1.85e+06 ulp**, reproducing §1.3's correction on a third shape.

**§1.3 does not say which narrowing to evaluate at when there is more than one,
and it must.** The harness evaluates at the innermost and prints the
intermediate-narrowing count so a shape with more than one is read as *partially*
covered rather than as a clean pass. Stating the rest of the elisions is left to
§1.3 rather than settled here.

### 13.8 What this did NOT measure

- **Any driver but ACO.** Nothing here says what nvrtc, IGC, ANV, pocl or Metal
  do on the 18 shapes; all of them are 0/63488 on the two shapes slice 1 swept
  and are refused today regardless.
- **A `precise` equivalent on OpenCL.** rusticl was swept plain only. OpenCL C
  has no `NoContraction` spelling this project has measured, so
  `R_local_absorb_opencl`'s `contract = false` is the *default* rusticl
  behaviour rather than a decoration's effect.
- **Constant folding across a narrowing**, the B2 residual of §13.4. Its
  mechanism is identified at machine-code tier and it is not modelled.
- **One constant, one addend, one divisor.** `1.1`, `1000`, `3` and `0.9`
  throughout, as in every prior measurement of this hazard, so the counts stay
  comparable — but the rule is confirmed against one operand pattern per shape,
  not a family. A15's fdiv happens to agree with correctly-rounded division on
  all 63488 inputs; SPIR-V permits it not to, and a different divisor might
  expose that.
- **The source spelling.** The shapes are emitted in three-address form, one
  temporary per operation, which is what `Sarek_ir_glsl.gen_var_decl` produces.
  A12 in particular is a ternary rather than an `if`/`else`, and A11's `floor`
  result is a named temporary; a different spelling could give a compiler a
  different peephole to match. This measures a driver on a codegen shape, which
  is what §7(c) says of every probe in this document.
- **Sarek-generated shaders.** Still §7 slice 3, unchanged.
