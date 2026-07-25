# Floating-point contraction policy

_Cross-backend policy for what a Sarek DSL author may rely on when a device
compiler is free to fuse, reassociate or flush floating-point operations._

**Status:** normative for this repository. **Issue:** #116 (absorbs #110, #111).
**Date:** 2026-07-25.

Three separate defects in this project came from the same place — the gap
between the floating-point semantics we assumed a backend had and the ones it
actually had:

- **HIP/AMDGPU.** An ISel combine fused an f32 multiply into the f32→f16
  narrowing that consumed it (`v_fma_mixlo_f16`) and demoted an f32 add to
  binary16 (`v_add_f16`). 620 of 63488 finite binary16 inputs disagreed with the
  interpreter. `-ffp-contract=off` did not prevent it: the combine sits below
  the C-level FP controls.
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
| **HIP / AMDGPU** | `a*b+c`; **and**, below the FP flags, an f32 multiply into an f32→f16 narrowing, plus f32 add demotion to binary16 | two mechanisms, both required: `-ffp-contract=off` forced **last** in the hiprtc option array (`Hip_rtc.base_options` / `hiprtc_options`), *and* the `asm volatile("" : "+v"(x))` opacity barrier on every narrowing's argument (`Sarek_ir_cuda.sarek_f32_barrier_decl`) | executed, RX 7900 XTX / gfx1100, ROCm hiprtc: exhaustive sweep of all finite binary16 inputs, 373 disagreements before, 0 after |
| **CUDA / nvrtc (f16 narrowing)** | in principle the same fusion | **nothing Sarek emits.** `ptxas` simply declines to absorb `cvt.rn.f16.f32` | machine-code, CUDA 13.3 host tools, sm_75…sm_121 — see §4 |
| **CUDA / nvrtc + PTX (f32 `a*b+c`)** | yes, by default (`-fmad=true` is nvrtc's and ptxas's default, and it applies to PTX input too) | **no flag.** `Sarek_df64` denies the compiler a fusable multiply by routing products through `fma` (`mul_rn`) | executed, GTX 1070 Max-Q / sm_61 / CUDA 12.9 / driver 580.119.02: df64 mul 5.92e-08 → 9.07e-15, div 5.64e-08 → 5.08e-15 |
| **CUDA — subnormal flushing** | `-use_fast_math` / `-ftz=true` would flush binary32 subnormals | `Cuda_nvrtc.check_fp_conformance` **rejects** those options at the only point an option array reaches `nvrtcCompileProgram` | machine-code + test, CUDA 13.3: the hazard is reproduced (`FMUL.FTZ`/`FADD.FTZ` at sm_90) and the guard is proved to fire — see §5 |
| **OpenCL** | `FP_CONTRACT` is on by default in OpenCL C | **no flag** — Sarek passes an empty build-option string. Same `mul_rn`-by-construction defence as CUDA | executed, GTX 1070 Max-Q / NVIDIA OpenCL: mul 5.92e-08 → 9.07e-15, sqrt 2.88e-08 → 9.80e-15 with no OpenCL-specific change (quoted). Re-measured here on RX 7900 XTX / Mesa radeonsi: mul 9.07e-15, div 5.08e-15, sqrt 1.08e-14 |
| **Vulkan / GLSL** | contraction and reassociation of float expressions | `precise` on every float local (`Sarek_ir_glsl.gen_var_decl`), which glslang lowers to SPIR-V `NoContraction` | compiler-output (measured, §6): glslc 2026.2 / SPIRV-Tools 1.4.350.1 emits 2 `NoContraction` decorations for the generated matmul shader, 0 with `precise` stripped. **Whether a given driver honours `NoContraction` is NOT established here** — see §6. Separately, `fma` is not correctly rounded on RADV: df64 mul 5.84e-08 / div 5.86e-08, measured on RX 7900 XTX, Mesa 26.1.4-arch3.1 |
| **Metal** | Metal's default compile options enable fast math | **nothing.** `Metal_api` passes a null `MTLCompileOptions`, and `Metal_bindings.mtl_device_new_library_with_source` *ignores its `_options` argument entirely* | unverified — no Apple hardware in this project's CI or on the machine this policy was written on. Treat Metal float results as outside the guarantee |
| **WGSL** | unconstrained | nothing | unverified, untested |
| **Native (OCaml host)** | n/a | n/a | float32 is evaluated at OCaml binary64 precision, so error-free transformations cancel; `Sarek_df64` degrades to ~2^-24 there **by design** |

---

## 3. What a DSL author may rely on

**You may rely on, today:**

- **f32 arithmetic agreeing with the interpreter on HIP/AMDGPU**, including f16
  round-trips. This is the only backend where the f16 discipline has been
  confirmed by execution.
- **`Sarek_df64` meeting its precision contract on CUDA/PTX and OpenCL on
  NVIDIA Pascal, and on OpenCL on AMD** — with the `sqrt` residual recorded in
  `Sarek_df64`'s header still open.
- **No `-use_fast_math` / `-ftz=true` reaching nvrtc**, enforced rather than
  documented (§5).

**You may NOT rely on:**

- **Metal or WGSL float semantics at all.** Metal in particular compiles with
  fast math on, unopposed.
- **Vulkan `fma` being correctly rounded.** On Mesa RADV it is not, and
  `Sarek_df64` mul/div is ~5.8e-08 there — a *documented*, unfixed deviation,
  not a bug to rediscover.
- **f16 on NVIDIA hardware.** The codegen question is settled at machine-code
  level; no f16 kernel has ever been executed on an NVIDIA GPU (§7).
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

## 4. CUDA f16: the barrier was inert, and has been removed (#110)

The AMDGPU fix added an opacity barrier around every f32→f16 narrowing. A
PTX-flavoured variant, `asm volatile("" : "+f"(x))`, was added to the non-HIP
branch of `sarek_f32_barrier` at the same time — on the assumption that what
was needed on one target was probably needed on the other.

It was doing nothing. The assembly template is empty, NVVM erases the block and
coalesces the register, and the emitted machine code is unchanged.

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
`prec-sqrt=false|0`. `fmad=true` **warns** rather than rejects, because it is
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

The third mutation is the informative one: with the guard removed,
`compile_to_ptx` compiled `-use_fast_math` **successfully** on this host. The
guard is the only thing standing between a caller and a flushed-subnormal
kernel.

---

## 6. Vulkan/GLSL: `precise` is emitted, and honouring it is the driver's job

`Sarek_ir_glsl.gen_var_decl` prefixes every `float`/`double` local with
`precise`. The front end does lower it:

**Measured** (glslc 2026.2, SPIRV-Tools 1.4.350.1, on the Sarek-generated
`matrix_mul` compute shader taken verbatim from
`benchmarks/descriptions/generated/matrix_mul_generated.md`): the SPIR-V carries
**2 `NoContraction` decorations** on the accumulation `OpFMul`/`OpFAdd`, and
**0** when `precise` is stripped from the same source.

That is a **compiler-output** claim and it stops there. `NoContraction` is a
requirement placed on the SPIR-V consumer, i.e. the driver. Whether a given
driver honours it is a separate question that this measurement does not touch.

**This is an open question in this repository, and the two things written down
about it disagree.** `Sarek_ir_glsl.ml` says `precise` was added *because* RADV
was observed simplifying error-free transformations without it — which implies
RADV honours it. Campaign notes state the opposite, that Mesa ANV and RADV
ignore it. Neither is backed by a recorded, reproducible measurement in-tree.
**Do not restate either as fact.**

What *is* measured, and what it does and does not tell us: **re-measured for
this document on RX 7900 XTX under RADV, Mesa 26.1.4-arch3.1** (`test_df64`,
2026-07-25) — mul 5.84e-08, div 5.86e-08, against add 5.33e-15, sub 6.51e-15,
sqrt 1.08e-14, and the interpreter's mul 9.07e-15 / div 5.08e-15 on the same
run. The same shape appears on the Raphael iGPU under RADV. `Sarek_df64` mul/div
sat at ~5.8e-08 both **before and after** the
`mul_rn` contraction barrier. Since that barrier works by removing the fusable
multiply, a contraction-shaped failure would have been fixed by it. It was not.
So RADV's deviation has a different cause, consistent with the recorded one —
RADV's GLSL `fma` is not correctly rounded, which is independently supported by
the measurement that extending `mul_rn` into `two_sum`/`quick_two_sum`
*regressed* RADV (add 5.33e-15 → 1.15e-07). That argues RADV is not silently
contracting; it does **not** establish that RADV honours `NoContraction`.

**The experiment that would settle it** (not run): a kernel computing
`precise float p = a*b; out = p + c;` and, separately, `out = fma(a,b,c)`, on
inputs chosen so that `fl(fl(a*b)+c) ≠ fma(a,b,c)`, executed on RADV and on
ANV. If the two kernels agree, the driver contracted despite `NoContraction`.
Until that is run, treat Vulkan float32 as *contraction-safe by front-end
declaration only*.

---

## 7. What cannot be verified without NVIDIA hardware

There is no NVIDIA GPU on the machine this policy was written on. Host-side
`ptxas`/`nvdisasm`/`nvcc` (CUDA 13.3) cover more architectures than any single
GPU would, but they cover a different question. Explicitly still open:

- **No f16 CUDA kernel has ever been executed on NVIDIA hardware.** The
  interpreter-agreement claim for f16 is HIP-only. The remaining gap is narrow —
  a hardware conversion-rounding question (does `F2FP.F16.F32` round to
  nearest-even on ties?), not a codegen one — but it is not closed.
- **Offline `ptxas` is not the driver JIT.** Sarek loads PTX through
  `cuModuleLoadData`, so on a real machine the assembling compiler is the
  *driver's* ptxas, a different build from `/opt/cuda/bin/ptxas`. Nothing here
  constrains it.
- **One toolkit version.** Every CUDA measurement in this document is CUDA 13.3.
  CI runs 12.6. The result is stable across eight architectures and every
  optimisation level and flag combination tried, which is an argument for it
  being a durable `ptxas` property rather than a 13.3 accident — an argument,
  not a measurement. **If the SASS gate ever fails on CI at 12.6 while passing
  at 13.3, that difference is the finding.**
- **`FMUL.FTZ` was observed, subnormal divergence was not.** §5 shows the flag
  changes the instruction; it does not show a wrong answer, because that
  requires running the kernel.
- **`Sarek_df64`'s `sqrt` residual on NVIDIA** (1.42e-14 CUDA/PTX, 1.68e-14 /
  1.81e-14 in `test_real64`) is unexplained. The leading hypothesis —
  `sqrt.approx.f32` as the Newton seed — has a one-line experiment
  (`sqrt.rn.f32` instead) that needs an NVIDIA device to evaluate. Recorded in
  `Sarek_df64`'s header; do not promote it to a cause.
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
