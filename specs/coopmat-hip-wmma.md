---
name: roster-spec
type: spec
status: live
feature: Cooperative matrix (SCoopmat) on HIP via RDNA3 WMMA; CUDA codegen via nvcuda::wmma (compile-only); explicit opt-in for float accumulation
brief: downstream LLM prefill (matrix-matrix GEMMs on the RX 7900 XTX matrix units)
date: 2026-10-10
version: 1.0.0
---

# Spec — coopmat-hip-wmma

Decisions not dictated by the brief were **made autonomously by the agent** under
the owner's delegation and are marked *(agent decision)*. Owner rules: a
semantics-preserving optimisation may be default; anything that can change results
is explicit opt-in; an unsupported backend refuses through the capability model and
never falls back silently.

Binding prior decisions this spec implements rather than re-opens:
`docs/design/f16-relaxed-accuracy.md` §5 (float cooperative matrix is *Regime B*: a
derived bound, not bit-identity) and §6.1 (a bounded deviation needs an explicit
opt-in and the launch fails without it), both decided 2026-07-27.

## What exists before this change

- `SCoopmat` (`CM_decl`, `CM_load`, `CM_store`, `CM_muladd`) is IR only; no PPX
  surface. Row-major only; subgroup scope only.
- GLSL/Vulkan lowers the integer configurations; float components are refused by
  the GLSL emitter and by the interpreter. Every other emitter refuses `SCoopmat`.
- A backend that does not probe leaves `capabilities.coopmat` `None` ("not
  probed"), which the launch gate turns into a refusal; Vulkan, HIP, CUDA/C and
  Metal probe.

## Supported configurations per backend

All shapes are `M = N = K = 16` on the HIP, CUDA and Vulkan backends (Metal uses 8x8x8, see its row), scope `Subgroup`. Each `CM_load` / `CM_store`
names its memory layout: row-major (element (r, c) at `index + r·stride + c`) or
column-major (`index + c·stride + r`), for any fragment (see "Layouts" below).

| backend | device | A × B + C → D | regime | lowering | validation |
|---|---|---|---|---|---|
| HIP | gfx11 (RDNA3), wave32 | f16 × f16 + f32 → f32 | Relaxed_bounded, opt-in | `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32` | RX 7900 XTX (gfx1100) |
| HIP | gfx11, wave32 | {u8,s8} × {u8,s8} + s32 → s32 | Strict | `__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(sgn_a, a, sgn_b, b, c, false)` | RX 7900 XTX, bit-identical |
| HIP | gfx11, wave32 | {u8,s8} × {u8,s8} + u32 → u32 | Strict | same instruction (wrapping 32-bit sum, identical bits) | RX 7900 XTX, bit-identical |
| HIP | any other gfx (gfx1036 iGPU, gfx10, gfx9/CDNA, gfx12) or warp size ≠ 32 | none | — | refused by the capability gate | gfx1036 refusal observed |
| CUDA | sm_70+ (f16), sm_72+ (8-bit) | f16 × f16 + f32 → f32; u8 × u8, s8 × s8 + s32 → s32 | as above | `nvcuda::wmma` | **compile-only** (NVRTC + ptxas when present); CUDA/C reports the probe `Cuda_coopmat` (sm_70+/sm_72+, warp 32), ZLUDA devices declined (it rejects `wmma.*` and unsigned 8-bit `mma`); CUDA/PTX refuses |
| Metal | Apple GPU (`simdgroup_matrix`) | f16 × f16 + f32 → f32, 8×8×8 only | Relaxed_bounded, opt-in (bit-equal to the interpreter) | `simdgroup_multiply_accumulate` | see `specs/coopmat-metal-simdgroup.md` |
| OpenCL, WGSL, PTX, Native | — | none | — | refused (unchanged) | refusal tests |

Not offered anywhere by this change, each refused with a named reason:
saturating integer accumulation, f16 accumulate (f16 result; §5.4 of the design
document — no derived bound), bf16 and 4-bit integer operands (no component type),
workgroup scope, any shape other than 16×16×16 on HIP/CUDA.

### Layouts

| backend | row-major | column-major |
|---|---|---|
| Interpreter (oracle) | `index + r·stride + c` | `index + c·stride + r` |
| HIP gfx11 | A: `sarek_cm_load_a_*` (16 contiguous elements per lane at `p + i + l16·s`); B: `sarek_cm_load_b_*` (16 elements `s` apart at `p + i + l16`); accumulator `*_acc_*` | the other operand helper: column-major B is the contiguous `load_a_*` form, column-major A the strided one; accumulator `*_acc_cm_*` (`p + i + l16·s + h`, elements 2 apart) |
| CUDA | `row_major` fragment type, `mem_row_major` | `col_major` in the operand's fragment TYPE (so every load of one fragment must use one layout: a fragment loaded in both is a named `#error` in the CUDA arm), `mem_col_major` for accumulators; **compile-only** |
| GLSL (Vulkan) | `gl_CooperativeMatrixLayoutRowMajor` | `gl_CooperativeMatrixLayoutColumnMajor` |
| Metal | `transpose_matrix` flag of `simdgroup_load` / `simdgroup_store` (specs/coopmat-metal-simdgroup.md) | same flag |
| OpenCL, WGSL, PTX | refused (no cooperative matrices) | refused |

Executed: `sarek/tests/e2e/test_coopmat_layout_e2e.ml`, every row/column
combination of A, B, C and D on HIP gfx1100 (u8, f16 and bf16 configurations)
and on Vulkan RADV (u8; GLSL refuses f16), against a host reference on
small-integer inputs: equal for integer configurations, within 1e-4 relative
for f16/bf16 (the gfx1100 matrix unit does not round those sums exactly; a
misplaced element would be off by at least 1). On gfx1100 a column-major B fragment is two
`global_load_b128` per lane where the row-major one is 16 `global_load_d16(_hi)`
loads (`spoc-pipeline-logs/pkgF/F2-isa-*.txt`). The helpers take a
`long long` stride, so `l16·s`, `e·s` and `2j·s` do not overflow `int` past a
stride of 2^31/15.

*(agent decision)* **Builtins, not rocWMMA.** hipcc/hiprtc in ROCm 7.2.4 accept
`__builtin_amdgcn_wmma_f32_16x16x16_f16_w32` and
`__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32` for gfx1100 (measured, see Evidence).
The builtins need no include path (hiprtc has none by default; rocWMMA needs
`-I/opt/rocm/include` and a header-only template library at every JIT compile), and
their per-lane register layout is documented, so Sarek's element-wise row-major
`CM_load` semantics — any start index, any stride, no alignment precondition — map
onto them exactly. rocWMMA's `load_matrix_sync` adds its own layout and alignment
rules that the IR does not promise.

*(agent decision)* **CUDA uses `nvcuda::wmma`, not `mma.sync` PTX.** No NVIDIA GPU
is available, so the lowering cannot be executed. `nvcuda::wmma` leaves the
fragment-to-lane mapping to NVIDIA's header, so the generated code makes no layout
claim that only hardware could check; the inline-PTX alternative would encode four
per-lane tables nobody here can execute. Its preconditions (pointer 32-byte aligned,
`ldm` a multiple of 16 bytes) are checked in the generated code, which traps when
they fail rather than computing on a misaligned fragment.

## Fragment and layout semantics (HIP lowering)

Unchanged IR semantics (`Sarek_coopmat_types`): `CM_load` with start `i` and stride
`s` fills `m[r][c] = buf[i + r*s + c]`; `CM_store` is its inverse;
`CM_muladd` computes `D = A × B + C`; `CM_decl` creates a zero fragment.
The values of `i` and `s` must be uniform across the subgroup.

Per-lane register layout on gfx11 wave32 (lane `l`, `l16 = l mod 16`, `h = l / 16`):

- A (16×16, `m × k`): lane holds row `l16`, elements `k = 0..15` (lanes 16–31 hold
  a copy of lanes 0–15). f16: 16 halves; 8-bit: 16 bytes packed little-endian in
  four 32-bit words.
- B (16×16, `k × n`): lane holds column `l16`, elements `k = 0..15`.
- C/D (16×16 f32 or i32): element `j` (0..7) of lane `l` is `m[2j + h][l16]`.

The generated source carries a header comment naming the semantics in force for
any Relaxed_bounded configuration (design document §6.1, "also required").

Buffer element types: an f16 fragment loads from a `float16` buffer, an f32
accumulator from a `float32` buffer, an 8-bit fragment from a `uint8` buffer (the
signedness is the fragment's, applied by the instruction, as in the interpreter),
and a 32-bit integer accumulator from an `int32` buffer. Any other pairing is a
codegen error naming the fragment and the buffer. The buffer may be a kernel
parameter or a shared array.

## Capability (per device)

- HIP probe *(agent decision)*: a device is offered the HIP configurations above
  iff `hipDeviceComputeCapability` reports major 11 (the gfx11 family — gfx1100
  reports 11.0, the Raphael iGPU gfx1036 reports 10.3) and its warp size is 32. Any
  other device is reported as probed with no configuration (`Some` with an empty
  list), so the verdict is `Unavailable` with a reason naming the HIP rule, not
  `Unknown`. gfx12 also has WMMA, with different builtins and layouts; it is
  refused until it is measured.
- `ds_subgroup_size` = the device warp size (32). `ds_robust_buffer_access` = false.
- The refusal text of `Sarek_coopmat.device_lacks_config` becomes backend-neutral;
  each probe states its own evidence in a new `ds_evidence` field.
- CUDA keeps `coopmat = None` ("not probed" → refused). Metal is probed since specs/coopmat-metal-simdgroup.md.

## Opt-in for float accumulation

- `Sarek_coopmat.accept_relaxed_accumulation : reason:string -> unit` — process-wide
  and one-way; `reason` must be non-empty (`Invalid_argument` otherwise).
- `Sarek_coopmat.policy_verdict cfg`: Strict configurations → `Available`;
  `f16 × f16 + f32 → f32` → `Available` after the opt-in, otherwise `Unavailable`
  with a `Policy` capability whose remedy names the call and the document; every
  other float configuration → `Unavailable` (`Policy`, no derived bound).
- Applied in `Execute.check_device_capabilities` after the device verdict, and in
  the interpreter's capability check.
- *(agent decision)* Why not treat "the kernel contains SCoopmat" as the opt-in:
  the recorded owner decision (§6.1) requires that the user be made to say yes to a
  bounded deviation, and a kernel can be written by someone other than the person
  who launches it (a downstream library). The call is one line.

## Interpreter

With the opt-in, the interpreter evaluates `f16 × f16 + f32 → f32` in one fixed
order: `acc := C[m][n]`, then for `k = 0..15`, `acc := fl32(acc + A[m][k]·B[k][n])`
(the product is exact in binary32). This is one admissible member of the bounded
set, not "the" answer; its role is a deterministic reference that itself satisfies
the bound. Loading an f16 fragment rounds to binary16 (identity on a `float16`
buffer). Without the opt-in the interpreter refuses as before.

## Scaled accumulation (`CM_scale_add`)

Added 2026-11-07, design and evidence in
`docs/design/coopmat-scaled-accumulate.md` (§2.3 semantics, §6 implementation).
Per element, `dst = (dst + (f32(src) * r[i]) * c[j]) - p[i] * q[j]`, each step
one binary32 operation rounded to nearest even, in that order, never contracted;
`src` is an s32 or f32 accumulator, `dst` an f32 accumulator, the scales float32
buffers (global or shared). Lowered here in registers with the accumulator
layout above (element `e` of lane `l`: row `2e + l/16`, column `l mod 16`) by the
preamble's `sarek_cm_scale_add` template; non-contraction rests on hiprtc's
`-ffp-contract=off`. Strict: bit-identical to the interpreter on gfx1100
(`sarek/tests/e2e/test_coopmat_scale_add_e2e.ml`). The CUDA arm of the shared
source is a named `#error`; the HIP probe sets `ds_scale_add` on gfx1100 (compute capability 11.0,
wave32) only, the one chip it was executed on.

## Exactness contract (tests)

- Integer configurations: bit-identical to the interpreter (Strict).
- `f16 × f16 + f32 → f32`: for each output, `|D − D_exact| ≤ γ_n · (Σ_k |A·B| + |C|)`,
  `γ_n = n·u/(1 − n·u)`, `u = 2⁻²⁴`, `n` = number of additions (16 for one
  `CM_muladd`; `K` for a GEMM chaining `K/16` of them onto an initial `C`).
  `D_exact` is computed in binary64 and the harness asserts the condition that
  makes binary64 exact (term exponent span + 25 + ⌈log₂(n+1)⌉ ≤ 53).
- Positive controls the gate must reject: a binary16-accumulating reference and a
  C-dropping reference, with a nonzero C.
- Determinism: repeated launches give identical bits.

## User stories

### US-1: f16 GEMM on the RX 7900 XTX matrix units (P0)
1. **Given** the opt-in and a 16×16×16 f16×f16+f32→f32 kernel, **When** launched on
   gfx1100, **Then** every output is within the γ₁₆ bound of the exact value, the
   two controls are rejected, and two runs are bit-identical.
2. **Given** a tiled GEMM kernel built from SCoopmat (M, N, K multiples of 16),
   **When** run on gfx1100 and on the interpreter, **Then** both are within γ_K of
   the exact result.
3. **Given** no opt-in, **When** the same kernel is launched on gfx1100 or on the
   interpreter, **Then** the launch raises a capability refusal naming
   `accept_relaxed_accumulation`.

### US-2: integer configurations on HIP are exact (P0)
1. **Given** u8×u8+s32→s32, **When** run on gfx1100, **Then** bit-identical to the
   interpreter over all 65536 operand pairs (`test_coopmat_hip_e2e`, which
   reuses the tiling of the Vulkan-only `test_coopmat_integer_e2e`), and a tile
   at a nonzero index and stride inside a wider buffer is bit-identical too.
2. **Given** s8×s8, u8×s8, s8×u8 with s32 and u32 accumulators, **Then** bit-identical.

### US-3: clean refusal (P0)
1. **Given** any coopmat kernel, **When** launched on the gfx1036 iGPU, **Then** the
   gate refuses with `Unavailable`, naming the HIP rule.
2. **When** launched on CUDA, Metal, OpenCL, Native, **Then** refused before codegen.
3. **Given** a HIP kernel with a shape other than 16×16×16, a saturating or
   f16-accumulate configuration, or a mismatched buffer type, **Then** codegen raises
   an unsupported-construct error.
4. **Given** a coopmat kernel launched with a block whose thread count is not a
   multiple of the subgroup size, **Then** the launch is refused (a partial wave
   would execute the matrix instruction with inactive lanes, which is undefined).

### US-4: CUDA codegen compiles (P1)
1. **Given** f16 and u8/s8 coopmat kernels, **When** CUDA source is generated and
   NVRTC is available, **Then** it compiles for sm_80; when ptxas and nvdisasm are
   available, the SASS contains `HMMA` / `IMMA`.

## Requirements

- **FR-001**: `Sarek_coopmat.device_support` gains `ds_evidence : string`; the
  Vulkan probe fills it; `device_lacks_config` uses it and is backend-neutral.
- **FR-002**: `Sarek_coopmat.accept_relaxed_accumulation`, `relaxed_accumulation_reason`,
  `policy_verdict` as above; `Execute.check_device_capabilities` and
  `Sarek_interp_capability` apply `policy_verdict`.
- **FR-003**: HIP `capabilities.coopmat` is `Some` per the probe rule above.
- **FR-004**: `Sarek_ir_cuda` emits the HIP arm (guarded by `__HIP_PLATFORM_AMD__`)
  and the CUDA arm (`nvcuda::wmma`) for `SCoopmat`; preamble only when the kernel
  uses coopmat; `TUint8` maps to `unsigned char`.
- **FR-005**: The interpreter evaluates f16×f16+f32→f32 per the fixed order above
  when opted in.
- **FR-006**: `Execute` refuses a coopmat launch whose block size is not a multiple
  of the device's coopmat subgroup size.
- **FR-007**: Tests per the user stories; benchmark per Success criteria.

## Success criteria

- SC-1: HIP f16 and integer e2e green on gfx1100; refusal observed on gfx1036,
  Native, Interpreter-without-opt-in, OpenCL, Vulkan iGPU.
- SC-2: CUDA NVRTC compile gate green (CI has NVRTC).
- SC-3: Benchmark M=256, K=3072, N ∈ {1024, 3072, 12288}, f16 inputs, f32 output:
  SCoopmat GEMM vs `Sarek_gemm.sgemm_tiled_kernel` (f32) vs rocBLAS
  `rocblas_gemm_ex` (f16→f32), GPU events, ≥ 3 rounds × 20 timed launches, median
  and min–max reported, with the machine state stated.
- SC-4: full `dune build` and `dune runtest` pass.

## Results

### Correctness (tartarus, 2026-10-10)

`sarek/tests/e2e/test_coopmat_hip_e2e.exe` on the RX 7900 XTX (gfx1100), ROCm
7.2.4, kernel 7.1.2-3-cachyos; log `spoc-pipeline-logs/coopmat-hip-wmma/`:

- Integer, all eight HIP configurations (u8/s8 × u8/s8, s32 and u32
  accumulators): 16 tiles each covering all 65536 8-bit operand pairs,
  bit-identical to the interpreter; the C-dropping control rejected on 16/16.
- f16 × f16 + f32 → f32, 64×64×128 and 256×256×3072 tiled GEMM: every output
  within γ_K of the exact result; worst |err| / bound = 0.044 (K = 128) and
  0.0073 (K = 3072). The interpreter's sequential order is within the bound too
  (worst 0.028) and equals the device bit-for-bit on only 212 of 4096 outputs:
  the matrix unit does not round like a sequential binary32 loop, which is why
  the comparison is a bound and not equality. Two runs and three tilings
  (1×1/w1, 2×1/w2, 2×2/w2) are bit-identical. Controls rejected: binary16
  accumulation (3982/4096 and 25009/65536 outputs outside the bound), C dropped
  (all outputs outside).
- Refusals observed: f16 before the opt-in on HIP gfx1100 and Vulkan NAVI31
  (policy reason, naming the call), on the HIP iGPU gfx1036 (HIP rule), on
  OpenCL ×2, Vulkan iGPU, Native, Interpreter ×2; the interpreter's own entry
  point; a 16-thread block.
- Mutations, each observed red then reverted: B loaded as rows (all f16
  outputs outside the bound), accumulator rows `j + 8h` instead of `2j + h`
  (all eight integer configurations diverge on 16/16 tiles), policy verdict
  removed from the launch gate (gfx1100 and NAVI31 permit f16 before the
  opt-in).

CUDA arm (no device): NVRTC 12.6 compiles the f16, u8, s8 and u8→u32 tiles
(`test_cuda_nvrtc_gate`), and ptxas sm_80 + nvdisasm show `HMMA` for f16 and
`IMMA` for u8/s8 (`test_cuda_coopmat_sass`, local CUDA 12.6 NVRTC/ptxas and
nvdisasm 12.8). **Not executed on NVIDIA hardware.**

### Benchmark (tartarus, 2026-10-10, ~23:15 CEST)

RX 7900 XTX, ROCm 7.2.4, shared with other users (busy 0 % and 49 % before
the two sessions). `sarek-hip/test/bench_hip_coopmat_gemm.exe 3 10 10` and
`tools/probes/rocblas_gemm_f16.cpp`: hipEvents around batches of 10
back-to-back launches, 10 batches × 3 rounds per session, two sessions; the
first round of each is slower (clock ramp) and is included. A, B f16, C/D f32,
row-major, D = A·B + C. Median per launch, TFLOP/s = 2MNK/t, [min–max over
both sessions' 60 samples]. Raw: `spoc-pipeline-logs/coopmat-hip-wmma/bench2.txt`.

| kernel | N = 1024 | N = 3072 | N = 12288 |
|---|---|---|---|
| Sarek scalar f32 `sgemm_tiled_kernel` | 506 / 488 µs — 3.2–3.3 TFLOP/s | 1499 / 1521 µs — 3.2 | 5808 / 6056 µs — 3.2–3.3 |
| SCoopmat 1×1 fragments, 4 waves | 67 / 80 µs — **20–24** | 222 / 223 µs — 21.7 | 979 / 1014 µs — 19–20 |
| SCoopmat 2×2, 2 waves (default) | 92 / 82 µs — 17.5–19.7 | 135 / 114 µs — **35.7–42.5** | 427 / 423 µs — 45.2–45.7 |
| SCoopmat 4×2, 2 waves | 143 / 144 µs — 11.2 | 172 / 149 µs — 28–32 | 382 / 365 µs — **50.6–53.0** |
| rocBLAS `gemm_ex` f16→**f32** | 157 / 153 µs — 10.3–10.6 | 330 / 348 µs — 13.9–14.6 | 1197 / 1236 µs — 15.6–16.2 |
| hipBLASLt f16→f32 (top heuristic) | 105 / 111 µs — 14.6–15.3 | 286 / 289 µs — 16.7–16.9 | 1035 / 1044 µs — 18.5–18.7 |
| rocBLAS `gemm_ex` f16→**f16** (different output type) | 63 / 58 µs — 25.7–27.9 | 84 / 86 µs — 56.4–57.2 | 231 / 236 µs — 81.9–83.7 |

Read: the matrix-unit GEMM is 5.5–16× the scalar Sarek GEMM at these shapes
(default tiling 5.5–15×; best tiling per shape 7–16×) and,
with f32 output, faster than both vendor libraries in this ROCm (whose f32-output
path on gfx1100 does not reach their tuned kernels). rocBLAS with an f16 output
is 1.3–1.6× faster than the best SCoopmat tiling: the SCoopmat kernel loads
fragments straight from global memory with no LDS staging or double buffering,
which is the obvious next step and is not in this change.

## What a downstream consumer needs to adopt it

- **Call** `Sarek_coopmat.accept_relaxed_accumulation ~reason:"..."` once at
  start-up (the result is bounded, not bit-identical: say so in its docs).
- **Kernel:** `Sarek_gemm.Coopmat.run ~device ?tiling ~a ~b ~c ~d ~m ~n ~k ()`
  computes `d = a·b + c`; or take `Sarek_gemm.Coopmat.ir` and launch it with
  `launch_dims`. Shapes must tile (`check_shape`): M a multiple of 16·frags_m
  (32 by default), N of 16·frags_n, K of 16, and the wave-tile count of `waves`.
  For prefill pad the token count M up to the tile height.
- **Layout:** row-major, A = activations M×K **f16**, B = weights **K×N f16**,
  or B stored transposed (N×K, llama-style weights or token-major activations)
  with `~b_layout:Col_major`; C/D f32 M×N (C = bias or a zero buffer; D may
  then feed the f32 pipeline). Transposed B loads contiguously but each lane
  reads 32 bytes per k step from a different row: faster than row-major B while
  B stays in cache (M = 256, K = 3072, N = 1024: 1.3–1.4× for the 1×1, 2×2 and
  2×4 tilings), mixed at N = 3072, and 1.5–5× slower when it does not
  (N = 12288; `spoc-pipeline-logs/pkgF/F2-bench-coopmat-gemm-c41-summary.txt`).
- **f16 conversion:** activations f32 → f16 with a small Sarek kernel
  (`float16_of_float32`) per layer before the GEMM; weights converted (and
  dequantised, for quantised GGUF tensors) to f16 once at load. f16 range is
  ±65504: activations beyond it need scaling.
- **Devices:** gfx11 HIP only. Everything else refuses with a capability error,
  so a consumer keeps its existing path as the explicit alternative (no silent
  fallback inside Sarek).

## Evidence

- E-1 (tartarus, 2026-10-10, ROCm 7.2.4 hipcc, RX 7900 XTX gfx1100): a standalone
  HIP program using both builtins with the layout above: f16 16×16×16 max |err|
  vs binary64 7.6e-06 on values of magnitude ~10–60; iu8 (u8 × s8) 0 mismatches of
  256. Program: `spoc-pipeline-logs/coopmat-hip-wmma/wmma_layout.hip`.
- E-2: `rocminfo` reports gfx1100 (wavefront 32) and gfx1036 (wavefront 32).
- E-3: Results above (correctness, CUDA compile, benchmark).
