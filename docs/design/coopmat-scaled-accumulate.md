# Scaled accumulation of a cooperative-matrix fragment

_An operation that adds a per-row and per-column scaled copy of one accumulator
fragment (int32 or f32) into an f32 accumulator, for quantised GEMMs in the style
of llama.cpp's MMQ kernels and for the per-row rescale of online softmax._

**Status:** implemented 2026-11-07 (§6): the interpreter (oracle) and HIP
gfx11 (executed, bit-identical on an RX 7900 XTX, offered on gfx1100 only); CUDA/PTX, CUDA/C, Vulkan,
Metal, OpenCL, WGSL and Native refuse it with named reasons. Until then the
implementation was deferred, because the measurement in §1 shows that 8-bit
WMMA is not measurably faster than f16 WMMA on gfx1100 (ratio 1.02, inside an
8 % order effect): an int8 path gains only the dequantisation work it
removes, not matrix throughput. **Date:** 2026-10-12. **Requested by:** a downstream
user (an int8 prompt path in the style of llama.cpp's MMQ).

## 1. Decisive measurement: iu8 vs f16 WMMA peak on gfx1100

The integer path is only worth a new IR operation if the integer matrix unit is
faster than the f16 one. It is not measurably faster.

**Kernel.** One `SCoopmat` IR kernel per configuration, lowered by the existing
HIP coopmat code (`sarek/codegen/Sarek_ir_cuda_coopmat.ml`, unchanged): A and B
loaded once, four independent accumulators, a loop of `n = 8192` iterations each
issuing four `CM_muladd` (`acc_k = A * B + acc_k`), the four accumulators stored
at the end. Block 256 threads (8 waves), grid 1536 blocks, no other memory
traffic. Ops counted as `2 * 16^3` per `CM_muladd`. Operands are all 1 (u8) and
all 1.0 (f16), accumulators start at 0, so both configurations switch similar
bits. The generated source of both kernels, compiled with
`hipcc --offload-arch=gfx1100 -O3 -ffp-contract=off --cuda-device-only -S`
(ROCm 7.2.4), has exactly four `v_wmma_i32_16x16x16_iu8` (resp.
`v_wmma_f32_16x16x16_f16`) in the loop body.

**Machine and conditions.** RX 7900 XTX (gfx1100), ROCm 7.2.4, only the dGPU
visible (`HIP_VISIBLE_DEVICES=0`), each run under `scripts/gpu-kernel-log-gate.sh`
(exit 0, clean log) and serialised behind the GPU lock shared with other
sessions. Other sessions were loading the CPU (load average 3.4 to 4.7). Best of
5 launches after one warm-up; per run the two configurations alternate twice;
two runs, one per order. The bench source, the generated kernels, their ISA and
the raw output (with date, command and load) are kept outside the repository.

| run | 1st pair (TOPS) | 2nd pair (TOPS) |
|---|---|---|
| f16 first | f16 116.7, iu8 125.4 | f16 128.1, iu8 133.4 |
| iu8 first | iu8 118.7, f16 122.0 | iu8 129.5, f16 130.8 |

**Result.** Whatever comes second in a pair is faster (presumably the clock
ramping up; clocks were not recorded), so the order effect (up to 8 %) is larger than any difference
between the configurations. Means over the four positions: iu8 126.8, f16 124.4
TOPS, a ratio of 1.02. On gfx1100 the 8-bit and f16 WMMA rates are equal within
this noise; there is no 2x. An earlier run with all-zero f16 operands showed f16
ahead (iu8/f16 0.96 to 0.97 at n = 8192); zero operands likely draw less power, so that run is not
used. RDNA4 (gfx12) has a higher 8-bit rate per AMD's published figures, but
Sarek's HIP lowering does not cover gfx12 (`specs/coopmat-hip-wmma.md`).

**Consequence for the consumer.** An int8 prompt path on gfx1100 can save the
dequantise-to-f16 pass, the f16 narrowing of activations and half of the
activation bytes. It cannot save matrix-unit time. Whether that is worth the
work is a downstream measurement (its trace splits dequantisation from GEMM time), not
a SPOC one.

## 2. The operation

### 2.1 What is missing

Fragments are opaque: `CM_decl`, `CM_load`, `CM_store`, `CM_muladd`
(`spoc/ir/Sarek_ir_types.ml`). A quantised GEMM accumulates each 32-element K
block exactly in int32 and must then apply that block's scales in f32:

```
acc_f32[i][j] = (acc_f32[i][j] + ((float) acc_i32[i][j] * d_w[i]) * d_x[j]) - m_w[i] * s_x[j]
```

(`d_w`, `m_w`: per-row weight scale and min of the block; `d_x`, `s_x`:
per-column activation scale and scaled sum). Today this needs a `CM_store` of the
int32 fragment to shared memory, a scalar pass, and a `CM_load` back, per K block.

### 2.2 Proposed IR

One new member of `coopmat_op`, so the existing exhaustive matches are forced to
consider it:

```ocaml
| CM_scale_add of {
    dst : string;            (* f32 accumulator fragment, read and written *)
    dst_frag : Sarek_coopmat_types.fragment;
    src : string;            (* s32 or f32 accumulator, same shape; u32 refused *)
    src_frag : Sarek_coopmat_types.fragment;
    row : scale;             (* r.(i) = row.buf.(row.index + i), i < m *)
    col : scale option;      (* c.(j) = col.buf.(col.index + j), j < n; None = 1 *)
    offset : (scale * scale) option;   (* (p, q): subtract p.(i) * q.(j) *)
    accumulate : bool;       (* false: overwrite dst instead of adding to it *)
  }
and scale = {buf : string; index : expr}   (* f32 buffer, global or shared *)
```

`index` is uniform across the subgroup, as for `CM_load`. `dst` and `src` may
name the same f32 fragment: with `col = None`, no `offset` and
`accumulate = false` that is the per-row rescale online softmax needs
(`acc[i][j] <- acc[i][j] * r[i]`). Typed records, no `Obj`.

### 2.3 Semantics: an exact order, no grant

For every element, with every operation correctly rounded to binary32 (round to
nearest even) and **no contraction**:

```
t1 = f32(src[i][j])          (* int -> f32 conversion; identity for f32 src *)
t2 = t1 * r[i]
t3 = t2 * c[j]               (* t3 = t2 when col = None *)
t4 = dst[i][j] + t3
t5 = p[i] * q[j]             (* only with offset *)
dst[i][j] = t4 - t5          (* dst[i][j] = t4 without offset *)
```

With `accumulate = false`, `t4 = t3` (the old `dst` is not read); that is the
softmax rescale. Bit-identity is a property of this operation; an f16 `src`
produced under the relaxed contract is already outside it.

Why this is Strict and needs no `Sarek_policy` grant:

- int32 accumulation in `CM_muladd` is already exact (Strict,
  `specs/coopmat-hip-wmma.md`).
- Every step above is a single IEEE operation with a fixed order. A device that
  performs exactly these operations produces the interpreter's bits. The
  int-to-float step is exact when `|src| <= 2^24`, which holds for one MMQ block
  (at most `2^19` for s8 x s8 and about `2^21` for u8 x u8 over K = 32, both
  below `2^24`); larger values are still deterministic on HIP, whose conversion
  is correctly rounded; on Vulkan only under `RoundingModeRTE` and
  `DenormPreserve` for 32-bit floats (`SPV_KHR_float_controls`, to verify on
  RADV). A u32 `src` is refused at code generation and by the gate (§3), not
  converted.
- The only way a device can differ is contraction of `t2 * c[j]` with the add,
  or of `p[i] * q[j]` with the subtraction. Each lowering therefore emits
  non-contractible operations (§3). A fused form would remove two roundings per
  element; it is not proposed, because the scale step is a few VALU operations
  per lane against the K-loop's loads and WMMAs, so there is no measured cost to
  buy back. If one appears, the fused form becomes a new named deviation in
  `Sarek_policy` with its bound (first order, both forms' final roundings
  included: `|fused - strict| <= u * (|t3| + |t5| + 2|t4| + 2|dst'|)` per
  element, `dst'` the final value, `u = 2^-24`; to be derived and tested before it becomes a grant),
  not a silent compiler choice.

### 2.4 Interpreter reference

The interpreter keeps whole fragments per subgroup (`coopmat_key` in
`sarek/interp/Sarek_ir_interp_eval.ml`); the leader lane applies §2.3 element by
element (as implemented, with exact binary32 rounding rather than
`Sarek_float32`, which flushes subnormals: §6). This is the oracle every backend is compared
against bit for bit. NaN payloads are not compared; subnormals are preserved,
matching HIP's `-fno-gpu-flush-denormals-to-zero`.

## 3. Lowering per backend (coverage table)

| backend | lowering | status if implemented | reason |
|---|---|---|---|
| Interpreter | §2.4, leader lane, whole fragment | reference | oracle |
| HIP gfx11 | direct, registers only: accumulator element `e` of lane `l` is row `2e + h`, column `l16` (`h = l / 16`, `l16 = l mod 16`, the layout already used by `sarek_cm_load_acc_*`). Each lane loads `c[l16]` and `q[l16]` once and `r`, `p` for its 8 rows; `v_cvt_f32_i32`, then plain `*`, `+`, `-`. Non-contraction rests on hiprtc's `-ffp-contract=off` (`base_options` in `sarek-hip/Hip_rtc.ml`); HIP's `__f*_rn` are plain operators in ROCm 7.2 and guarantee nothing. The e2e test checks the ISA for the absence of `v_fma`/`v_fmac` in the scale step | executed on RX 7900 XTX (gfx1100), bit-identical; offered on gfx1100 only (§6) | the only backend with a documented accumulator layout and a device here |
| Vulkan (KHR cooperative matrix) | the accumulator layout is implementation-defined, so no per-lane form. Portable route: convert the int32 fragment to an f32 accumulator-use matrix (`coopmat<float, …>(src)` is element-wise), load `r` and `c` as broadcast matrices (column-major with stride 0 for `r`, row-major with stride 0 for `c`), multiply component-wise, add. To verify on RADV before relying on it: that stride 0 is accepted and that component-wise `*` on cooperative matrices is supported; otherwise a shared-memory round trip (`coopMatStore`, scalar §2.3, `coopMatLoad`), which needs a per-subgroup scratch array declared by the lowering. The products need `precise` to forbid contraction; also to verify: that `precise` (NoContraction) is honoured on cooperative-matrix arithmetic and that the float_controls modes of §2.3 are available | refused as implemented (§6); target: executed on RADV (gfx1100) by whichever route passes those checks; neither route has been run yet | the plan's coverage decision (HIP and Vulkan executed) |
| CUDA (`nvcuda::wmma`) | the accumulator fragment's element-to-(row, column) map is not documented, so no direct form. Either a shared-memory round trip as for Vulkan, or a move of the 8-bit lowering to `mma.sync` (whose PTX fragment layouts are documented). Until one is done: refused at code generation with a named reason | refused at code generation | no NVIDIA GPU; ZLUDA declines `wmma` |
| Metal | integer `src`: refused, as every integer configuration already is (no integer `simdgroup_matrix`; `no_integer_reason` in `sarek/codegen/Sarek_ir_metal_coopmat.ml`). f32 `src` (softmax rescale): threadgroup round trip, later | integer refused with the existing named reason | hardware |
| OpenCL, WGSL, PTX, Native | refused, as every `SCoopmat` statement is | refused | no cooperative matrices |

The capability gate checks `CM_scale_add` on its own, not through a producer:
`CM_decl`/`CM_load` can create an accumulator with no `CM_muladd` in the
kernel, so "a muladd produced `src`" is not an invariant to rely on. The
statement contributes its own requirement to the gate (alongside the
configurations `kernel_coopmat_configs` collects): `src_frag` is an s32 or f32
accumulator and `dst_frag` an f32 accumulator, each of a shape in some
configuration the device offers, and the device's support record states that
its backend lowers `CM_scale_add` (a new boolean in the coopmat support record,
false by default, set by the HIP probe and by the Vulkan probe once its route
is validated). A device without it refuses at launch with a named reason; the
code generators of CUDA, Metal (f32) and the others refuse too, never computing
it wrongly.

## 4. Fragment loads of 8-bit operands (plan item S2): no change needed

What `sarek_cm_load_*_i8` compiles to on gfx1100 (ROCm 7.2.4, `-O3`, preamble of
`sarek/codegen/Sarek_ir_cuda_coopmat_preamble.ml` unchanged):

| fragment form | global memory | shared memory |
|---|---|---|
| contiguous (`load_a_i8`: A row-major, B column-major) | one `global_load_b128` per lane | one `ds_load_b128` per lane |
| strided (`load_b_i8`: B row-major, A column-major) | 16 `global_load_u8` per lane | not measured (expected: a gather) |

The contiguous forms are already vector loads, with no alignment precondition in
the IR (gfx11 accepts the unaligned access; a 16-byte aligned `index + l16 *
stride` avoids split accesses). The strided form is a gather by construction; the
remedy is the layout, not the helper: store the activations token-major and load B
column-major, as `CM_load`'s `layout` field already allows. On CUDA,
`load_matrix_sync` (under the existing 32-byte alignment check) leaves the load
width to NVIDIA's header, and on Vulkan `coopMatLoad` leaves it to the driver;
neither was inspected.

## 5. If the work is resumed

1. Tests first, failing before: interpreter unit tests of §2.3 on chosen values
   (a product whose contraction would change the bits; `|src| > 2^24`; an offset
   whose rounding matters), a HIP e2e test bit-identical to the interpreter over
   random int32 fragments and scales, a Vulkan e2e test, and refusal tests for
   CUDA, Metal, OpenCL, WGSL, PTX and Native.
2. IR member, interpreter, HIP lowering, Vulkan lowering (after the RADV checks in
   §3), refusals.
3. The coverage table above in the PR, a reason for each empty cell.

## 6. Implementation (2026-11-07)

**IR.** As §2.2, with typed records instead of the tuple and the short field
names: `row : coopmat_scale`, `col : coopmat_scale option`,
`offset : coopmat_offset option`, where `coopmat_scale = {scale_buf; scale_index}`
and `coopmat_offset = {off_row; off_col}` (`spoc/ir/Sarek_ir_types.ml`). Passes
that rename buffers, rewrite or fold expressions, or collect the arrays a kernel
reads see all four scales through `spoc/ir/Sarek_coopmat_scale.ml`.

**Interpreter.** `sarek/interp/Sarek_coopmat_scale_ref.ml`, one rounding per
step in the §2.3 order. It does not use `Sarek_float32`: that module flushes
binary32 subnormals to zero, which HIP (`-fno-gpu-flush-denormals-to-zero`)
does not do. Each step is computed in binary64 and rounded once to binary32 by
the IEEE conversion, which is the correctly rounded binary32 result of one
multiplication, addition or subtraction (binary64 has at least `2 * 24 + 2`
significant bits). A scale buffer that does not hold float32 values is refused, as is a scale
index past the end of its buffer; the HIP lowering, like `CM_load`, does not
check scale bounds on the device.

**HIP gfx11.** A template helper in the coopmat preamble
(`sarek_cm_scale_add<COL, OFF, ACC>`), one call per statement, in registers with
the accumulator layout of §3. Non-contraction rests on hiprtc's
`-ffp-contract=off` alone: a `#pragma clang fp contract(off)` in the helper was
tried and does not help, because clang's `-ffp-contract=fast` disregards it (the
helper with the pragma, compiled with `-ffp-contract=fast` for gfx1100, has
`v_fma`).

**Device support record.** `ds_scale_add`, true only from the HIP probe on
compute capability 11.0 (gfx1100) with 32-wide wavefronts, the only chip the
statement has been executed on. Other gfx11 parts (gfx1150, gfx1151, ...) are
covered by the lowering but are not offered the statement until it is run on
them; the gate names the reason (lowered for gfx11, executed only on gfx1100).
Known gap: the rule keys on compute capability, which does not name the exact
target; gfx1101 and gfx1102 (gfx11.0.x targets) also report 11.0 and are
admitted without having been executed. The exact target is only in
`hipDeviceProp_t.gcnArchName` (ROCm 7.2 `hip/hip_runtime_api.h`: the former
`hipDeviceAttributeArch`, `GcnArch` and `GcnArchName` attributes are now
`hipDeviceAttributeUnused3..5`), and reading it means binding the versioned
`hipDeviceProp_tR0600` struct; keying `ds_scale_add` on `gfx1100` exactly is a
follow-up.
The launch gate asks `Sarek_coopmat.scale_add_verdict` for every statement: the
operands (f32 destination, s32 or f32 source of the same shape and scope; u32
named), then `ds_scale_add`. The gate is evaluated once per (kernel, device),
through the memoised capability check, not per launch.

**CUDA/PTX and CUDA/C: refused.** The two routes of §3 were weighed:

- a shared-memory round trip needs a per-warp scratch tile (1 KB for a 16x16
  binary32 tile) whose total depends on the warps per block, which code
  generation does not know: a static worst case (32 warps, 32 KB) takes two
  thirds of the 48 KB static shared-memory limit from every kernel using the
  statement, and a dynamic allocation is a change to the launch path;
- `mma.sync` has documented fragment layouts, but moving the 8-bit lowering to
  it rewrites the loads, stores and products of the existing wmma lowering.

Neither can be executed here (no NVIDIA GPU; ZLUDA declines wmma), and both
would ship as compile-checked code computing on a layout or a scratch scheme
nobody has run. The CUDA arm of the shared CUDA/HIP source is a named `#error`,
the PTX generator refuses at code generation, and both CUDA probes report
`ds_scale_add = false`, so the launch gate refuses first.

**Vulkan: refused.** The GLSL lowering has no f32 cooperative-matrix fragment
at all (it emits the integer configurations only), and none of the RADV checks
of §3 has been run. What would be needed: an f32 accumulator fragment in the
GLSL lowering, then on RADV (gfx1100) a test that a stride-0
`coopMatLoad` broadcasts, that component-wise `*` and `+` on accumulator
matrices compile and run, that `precise` (NoContraction) is honoured on them
(an ISA check as for HIP), and that `RoundingModeRTE` and `DenormPreserve` for
32-bit floats are available; or, failing the first two, the shared-memory round
trip with its scratch-size question as for CUDA.

**Metal, OpenCL, WGSL, Native: refused.** Metal: an integer source with the
existing no-integer reason, an f32 source because the threadgroup round trip is
not written. OpenCL and WGSL refuse every cooperative-matrix statement; Native
and the backends that do not probe cooperative matrices refuse at the gate.

**Tests.** `sarek/tests/unit/test_coopmat_scale_add.ml` (chosen values whose
bits change if a step is fused, reordered or flushed, every flag combination
against an independent reference, refusals, the gate verdict, the generators);
`sarek/tests/e2e/test_coopmat_scale_add_e2e.ml` (the gate on every device; on
gfx11, bit identity with the interpreter over 18 variants of 64 random tiles,
and the gfx1100 ISA with a `-ffp-contract=fast` control).

