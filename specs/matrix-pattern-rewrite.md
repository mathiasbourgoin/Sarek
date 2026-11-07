---
name: roster-spec
type: spec
status: live
feature: Automatic rewrite of recognised GEMM nests into cooperative-matrix (SCoopmat) tiles, exact for int32, opt-in for float
brief: owner request 2026-10-11 (matrix pattern detection — PR 2 of 3: automatic optimisation)
date: 2026-10-11
version: 1.0.0
---

# Spec — matrix-pattern-rewrite

Builds on specs/matrix-pattern-advice.md (PR 1, the recogniser). Decisions not
dictated by the request are *(agent decision)*. Owner rules: a
semantics-preserving optimisation may be default; anything that can change
results is explicit opt-in; an unsupported device falls back explicitly.

## Scope and sequencing

*(agent decision, coordination)* The rewritten kernel has a different launch
geometry from the original (one subgroup per output tile instead of one thread
per output), so substituting it needs the actual launch: grid, block and scalar
arguments. This PR delivers the rewrite and an **explicit launch entry point**
that applies it (`Sarek_matrix.run`, a drop-in for `Execute.run_vectors`); the
**default** application inside `Execute` arrives with PR 3, as a per-launch
specialisation hook on the deferred-launch window that the dynamic-fusion work
is adding to `Execute` (coordinator instruction 2026-10-11: no concurrent
`Execute` edits). In `Sarek_matrix.run` the exact (integer) rewrite is on by
default and float rewrites need the opt-ins below; PR 3 carries the same policy
into `Execute` (`Faithful` default = exact rewrites only).

## Preconditions (the rewrite is declined otherwise, and the original runs)

Static (kernel), on top of PR 1's `Gemm` finding:
- the kernel is *only* the nest: its sole memory write is the store to C (no
  other array write, atomic, `SExpr`, `SReturn`, `[%native]`), and it has
  exactly one GEMM finding;
- each of M, N and at most one batch index is one of: a global index along one
  axis (`global_idx_y`, or `thread_idx_y + block_dim_y * block_idx_y`, same for
  x, z); an enclosing `for` variable with uniform bounds; or the pair
  `t / d`, `t mod d` with `t = global_thread_id` (or `global_idx_x`) and `d`
  uniform;
- every guard on the path is a conjunction of comparisons whose one side is
  affine-equal to the M index, the N index, the batch index or the linear id
  `t`, and whose other side is uniform;
- strides, offsets, bounds, `init` and epilogue reference no `block_dim_*` or
  `grid_dim_*` (the rewritten launch has its own).
- the path to the nest has no `else` branch, `while` or `match` case (the
  recogniser marks them: its guard list would not be the whole condition), and
  every enclosing `for` is the row, column or batch loop (another loop repeats
  or skips the product);
- the batch index, if any, is a global index or a loop, not a quotient or
  remainder of the thread id.

The rewritten grid (columns on x, rows on y, batches on z) must fit the
device's `max_grid_dims`, and an int64 scalar used in a bound must fit int32;
otherwise the original runs.

Dynamic (launch): the scalar arguments and the original grid/block give the
covered region: `m < Me`, `n < Ne`, `z < Ze`, and for a `t / d` mapping also
`m * d + n < T` (`T` = threads along x) — exactly the outputs the original
writes. The device must offer the configuration (`Sarek_coopmat.verdict`) and
the policy must admit it.

## The rewritten kernel

Same parameters as the original, in the same order, plus int32 parameters
`sarek_mm_me`, `sarek_mm_ne`, `sarek_mm_ze`, `sarek_mm_d`, `sarek_mm_t`.
One block = 4 waves of 32 (2×2), each wave 2×2 fragments of 16×16: a 64×64
block tile; grid = ⌈Ne/64⌉ × ⌈Me/64⌉ × Ze. For each K step of 16 the block
stages the A (64×16) and B (16×64) tiles from global memory into shared memory,
converting to the configuration's operand type and **zero-filling** every
element outside `[0, Me) × [k_lo, k_hi]` (A) or `[k_lo, k_hi] × [0, Ne)` (B) —
no out-of-range global read, any M, N, K, either layout, any offset and batch
stride. Both shared tiles are `[outer][k]` with rows padded by 16 bytes: A as
`[m][k]`, B transposed as `[n][k]`, so A fragments load row-major and B
fragments column-major, each 16 contiguous elements per lane (`ds_load_b128`
on gfx1100). Each thread stages 8 consecutive k of one row of A (column of
B): eight guarded reads (a select, not a branch), their conversions, then
eight contiguous shared stores, which merge into one `ds_store_b128` for
f16 (`spoc-pipeline-logs/pkgF/F2-isa-after-c42.txt`). Consecutive threads
take consecutive groups when k is the operand's contiguous dimension (A
row-major, B column-major), consecutive rows (columns) otherwise, so global
reads stay coalesced. Fragments are loaded from
shared memory (`CM_load`), multiplied (`CM_muladd`), stored to a shared tile,
and each covered output is written with the original epilogue applied to
`init + acc` (the original `init` and epilogue expressions, re-evaluated).

Element cases:

| case | staging | products per tile step | accumulation | result |
|---|---|---|---|---|
| int32 | 4 unsigned byte planes per operand (`uint8` shared, low byte of `x lsr 8i`) | 10 (`i + j ≤ 3`), u8×u8+s32→s32 | 4 accumulators by `s = i + j`, combined `Σ acc_s lsl 8s` (wrapping) | **bit-identical** (mod 2³²) |
| float16 operands | float16 | 1, f16×f16+f32→f32 | f32 | γ_K bound (existing regime B) |
| float32, `F16` | round to float16 | 1 | f32 | operand rounding + γ_K |
| float32, `BF16` | round to bfloat16 (RNE) | 1, bf16×bf16+f32→f32 | f32 | operand rounding + γ_K |
| float32, `BF16x3` | x = hi + mid + lo, each bfloat16 (RNE of the residual) | 6 (`i + j ≤ 2`) | f32 | near f32: measured vs γ_K |

## Subgroup width and backends (H4)

A block is **four waves of one subgroup each** (2 × 2 over the 64 × 64 tile),
so it has `4 × subgroup` threads: 128 on HIP gfx11 and CUDA warps, 256 on RADV
(subgroup 64). `Sarek_matrix_rewrite.kernel ?subgroup` and `launch_dims
?subgroup` take the width (32 or 64; any other is declined); the wave of a
thread is `thread_idx_x / subgroup`. Staging gives each thread one or more
groups of 8 consecutive k, and guards off the threads past the last group when
the threads outnumber the groups (256 threads, 128 groups); the 32-wide output
is byte-identical to the unparameterised kernel (golden digests,
`test_matrix_template_golden`). `Matrix_specialise` reads the width from the
device's cooperative-matrix probe (`ds_subgroup_size`); the interpreter's
cooperative-matrix subgroup is fixed at 32 lanes, so a 64-lane wave runs there
as two interpreter subgroups: `test_matrix_subgroup` (via
`subgroup_override_for_tests`) checks the staging guard, the strides and the
block size against the oracle, **not** the wave-to-subgroup mapping. That
evidence is the Vulkan e2e run on RADV and the mutation that forces width 32
there (wrong results, `H4-mutation-vk-sg32.txt`).

Admission: Vulkan is rewritten automatically only at the measured width, 64
(a 32-wide Vulkan device only on `Sarek_matrix.run`, until measured); the
launch is checked against the device's grid, threads-per-block and
shared-memory limits; the direct float16 variant is laid out for 32-wide waves
only and falls back to staged at any other width. Hand-written `SCoopmat`
kernels on CUDA/C (sm_70+ float16, sm_72+ 8-bit) are admitted by the general
launch gate now that CUDA/C reports a probe: compiler-output evidence only,
never executed on NVIDIA hardware.

| Backend | before | after | evidence |
|---|---|---|---|
| HIP gfx11 | int32 + float | unchanged (byte-identical IR) | executed on RX 7900 XTX (HIP, ROCm), golden digests |
| Vulkan | none | int32 byte planes, on by default (exact) | executed on RX 7900 XTX (RADV, Mesa 26.1.4), bit-identical to the original and the host; float: not offered (GLSL has no float fragment) |
| CUDA/C | none (`coopmat = None`) | int32 byte planes (sm_72+, warp 32) via `Cuda_coopmat`, **explicit request only** (`Sarek_matrix.run`; the automatic path declines: owner rule, untested = opt-in) | compiler-output: NVRTC sm_80 PTX has the u8 `wmma`; not executed: ZLUDA rejects `wmma.*` and unsigned 8-bit `mma` (`specs/zluda-mma-nvrtc13.md` F-1), so ZLUDA devices are declined; no NVIDIA hardware |
| CUDA/PTX | none | none (no coopmat lowering) | by construction |
| Metal | none | f16 operands (f16×f16+f32→f32) on 8×8×8 fragments, float opt-ins only; **integer declined: no integer `simdgroup_matrix`** | codegen: compiler-output (golden on tartarus); interpreter: executed (`test_matrix_fragment`); Apple M4 (macOS 15, 2026-10-12): executed, `test_matrix_rewrite_metal` passes, bit-identical to the original, the interpreter and the host chain; bench 9 pairs, f16 GEMM, rewrite off vs on: n=512 910 vs 189 us (4.81x), n=1024 6802 vs 966 us (7.04x), n=2048 53785 vs 7294 us (7.37x), load 2.6 with a co-running download |
| OpenCL, Native | none | none | unchanged |
| Interpreter | all | unchanged (+ 64-lane layouts checkable) | executed |

## Fragment shape and Metal (H4b)

The templates are parameterised by the **fragment edge** `frag` (m = n = k),
`Sarek_matrix_rewrite.kernel ?frag` and `config_of ?frag`: 16 (the default,
byte-identical output, `test_matrix_template_golden`) or 8. A wave's tile stays
32 × 32 and the block tile 64 × 64, so staging, the accumulator tile and the
output loop are unchanged; only the fragment grid changes: 2 × 2 fragments of
16 × 16, or 4 × 4 of 8 × 8 (16 accumulators per wave). A step of 16 along
the reduction is `16 / frag` chained multiply-accumulates per fragment, in k
order, into the same accumulator.

*(agent decision, the Open decision of `specs/coopmat-metal-simdgroup.md`)* Route
chosen: **parameterise the templates by fragment shape**, not lower a
16×16×16 statement as 2×2 tiles of 8×8 with two chained k-steps in the Metal
generator. Reasons: the HIP/Vulkan IR is untouched (golden digests unchanged),
the Metal generator and its files are untouched, and the tile composition is
visible in the IR, so the interpreter can execute and test it. The claim that
two chained 8-wide multiply-accumulates are the same C-first sequential
binary32 chain as one 16-wide one is **tested**, not assumed:
`test_matrix_fragment` (fragment 8 and 16 on the interpreter, bit-identical to
the original kernel and to the host chain over k, with a positive control: a
C-after-chain result differs from the C-first original), and
`test_matrix_rewrite_metal` on the GPU. Mutation: running one k-step of 8 per
step of 16 fails 9 of the cases.

The device-admission half lives in `Matrix_admission` (split out of `Matrix_specialise`, which had passed 500 lines). `Matrix_admission.fragment_of` takes the fragment edge from the device: the first of
`Sarek_matrix_scheme.fragment_edges` (16, then 8) whose configuration the device
advertises (`fragment_of`); nothing names Metal there. Admission is stricter than the advertisement: a fragment edge other than 16 has been executed on Metal's f16 operands only, so a Vulkan, CUDA or HIP device advertising 8x8x8 alone is declined with that reason (`fragment_refusal`; the interpreter, which holds whole matrices, is exempt). The direct float16 variant is
laid out for 16 × 16 fragments and stays staged at 8.

**Metal admission.** Lowered: `F16_operands` (float16 operands, or float32 rounded
to float16 under `accept_operand_rounding`). Not lowered, each declined with its
reason: the integer byte planes (**Metal has no integer `simdgroup_matrix`**: MSL
offers `half`, `float` and `bfloat` elements only, and the byte-plane scheme
needs `u8 × u8 + s32`, so the exact rewrite cannot be ported and an integer
product runs as the original, with the reason in the report); the bfloat16
lowerings (the bfloat 8×8 fragment is unmeasured and the IR's bf16 fragment
loads a float32 buffer rounded to bfloat16). Policy is unchanged: the float
rewrite needs `accept_relaxed_accumulation` and, automatically,
`accept_automatic_float_rewrite` (#440); `Sarek_matrix.run` does not need the
latter. Metal's launch is four waves of 32: 128 threads, 22.5 KiB of threadgroup
memory for f16 (the device limit is 32 KiB; `fits_device` checks it).

**Exactness on Metal.** The multiply-accumulate is a closed form (accumulator,
then the 8 products added in index order, binary32; `specs/coopmat-metal-simdgroup.md`),
and a product of two binary16 values is exact in binary32, so with an initial
value of zero the rewritten result is the sequential chain over k that the
original kernel and the interpreter compute: **bit-identical**, not merely
bounded. With a non-zero initial value the rewrite adds it after the chain
(`init + sum`) while the original starts the chain from it: they differ, as on
HIP, and `test_matrix_rewrite_metal` asserts the rewrite equals the C-after
model and the original the C-first one.

## Codegen and IR additions

- `ECast (TUint8, e)`: the low 8 bits of an int32 (narrowing store into a
  `uint8` array). Lowered by the CUDA/HIP generator (`(unsigned char)`) and the
  interpreter when this rewrite was written; since then `TUint8` and `TInt8`
  are general 8-bit storage types on every backend but WGSL (reads, stores and
  casts, no 8-bit arithmetic; see `spoc/ir/Sarek_ir_int8.mli`).
- Component `BFloat16` (`Sarek_coopmat_types`): a bf16 operand fragment loaded
  from a **float32** buffer, each element rounded to bfloat16 (round to
  nearest even); exact for values already representable. HIP gfx11:
  `__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32` (operands as 16 × i16,
  measured to compile with ROCm 7.2.4 hipcc for gfx1100); interpreter: same
  rounding, sequential f32 accumulation as for f16; HIP probe advertises
  bf16×bf16+f32→f32 on gfx11; CUDA: refused (nvcuda::wmma bf16 is sm_80
  `__nv_bfloat16` from a bf16 buffer — not this load semantics); others refuse.
  Accuracy policy: like f16 (Relaxed_bounded; products exact in f32).
- Bf16 rounding of a float32 in kernel code (the split): through the existing
  int32 bitcasts (`bits + 0x7fff + ((bits lsr 16) land 1)`, high half kept);
  NaN/Inf operands are out of scope of the γ bound and documented.

## Opt-ins (`Sarek_coopmat`)

- float16 operands: the existing `accept_relaxed_accumulation ~reason`.
- a `BFloat16` configuration (any IR, not only the rewrite) needs
  `accept_operand_rounding` for `Operands_bf16` or `Operands_bf16x3` in
  addition to `accept_relaxed_accumulation`: its load rounds float32 operands.
- Range: the bounds assume finite operands within the target type's normal
  range — float16 overflows above 65504 and loses relative accuracy below
  6.1e-5; bfloat16 shares float32's exponent range. NaN stays NaN (load helper
  and in-kernel split alike).
- float32 operands: `accept_operand_rounding ~precision ~reason` with
  `precision ∈ {F16, BF16, BF16x3}`, per precision, process-wide, one-way, empty
  reason refused; plus `accept_relaxed_accumulation` (the accumulation is
  reordered too). `Sarek_matrix.run ~precision` chooses among accepted ones
  (default: none → float32 GEMMs are not rewritten).

## Entry point (`sarek.matrix`)

`Sarek_matrix.run ?precision ~device ~ir ~args ~block ~grid () : outcome` —
`Rewritten of {config; lowering}` or `Original of reason` (why not: not a GEMM,
precondition, device, policy). Plans are memoised per (IR, device, precision) so
the rewritten IR is physically stable and hits `Execute`'s launch memo (#424).

## Tests and measurements

- Interpreter: int32 rewrite bit-identical to the original kernel on the
  interpreter, shapes with M, N, K not multiples of 16 (1×1×1, 17×33×5,
  64×64×64, 100×37×129), all corpus layouts (transposed A/B, 1-D mapping, batched,
  epilogue, init from C, guards), random values over the full int32 range.
- HIP gfx1100: the same, original on HIP vs rewritten on HIP (bit-identical);
  float cases: max error / γ_K·Σ|a·b| per precision, controls (dropping a
  product plane must exceed the bound); refusals and fallbacks (iGPU gfx1036,
  OpenCL, Native: `Original` with the reason).
- Bench (RX 7900 XTX, shared GPU, ≥3 sessions, median and min–max): scalar
  original vs rewritten (int32, f16, f32-F16, f32-BF16, f32-BF16x3) vs
  `Sarek_gemm.Coopmat` (f16) at M = 256, K = 3072, N ∈ {1024, 3072}, and an
  edge shape.

## Requirements

- FR-001 `ECast TUint8` narrowing; FR-002 `BFloat16` component + HIP lowering
  + probe + interpreter; FR-003 `accept_operand_rounding`; FR-004
  `Sarek_matrix_rewrite` (plan: preconditions, region, rewritten IR);
  FR-005 `Sarek_matrix.run`; FR-006 tests above; FR-007 bench + results here.

## Results (tartarus, 2026-10-11)

RX 7900 XTX (gfx1100), ROCm 7.2.4, GPU shared with other agents. Logs:
`spoc-pipeline-logs/matrix-pattern/pr2-*`.

**Correctness.** `sarek/tests/e2e/test_matrix_rewrite.exe`: int32, six kernels
(2-D, 1-D quotient, 1-D and 2-D short launches, transposed A and B, batched
with init-from-C and epilogue) × shapes 1×1×1, 17×33×5, 64×64×64, 100×37×129
(interpreter: 1×1×1, 17×33×5, 40×24×20), random operands over the whole int32
range: the rewrite is **bit-identical** to the original kernel on gfx1100 and
on both interpreters, and the original equals the host reference; outputs the
original leaves alone stay untouched. Declined, with the reason, on the 7950X
iGPU (HIP gfx1036), OpenCL ×2, Vulkan ×2 (NAVI31 advertises the integer
configuration but has 64-wide subgroups), Native; a kernel with a second write
is declined everywhere.
`test_matrix_rewrite_float.exe`: before the opt-ins a float32 product runs as
the original. Max |err| / (γ_K Σ|a·b|), gfx1100, shapes 64×64×128, 100×37×129,
256×256×1024, operands uniform in [-1, 1):

| precision | ×γ_K | own bound used by the test |
|---|---|---|
| f16 operands (relaxed accumulation) | 0.010–0.028 | γ_K |
| f32 as f16 | 0.77–15.8 | 2u₁₆ + u₁₆² + γ_K |
| f32 as bf16 | 6.0–114 | 2u_bf16 + u_bf16² + γ_K |
| f32 as bf16x3 | 0.04–0.15 | 2⁻²⁴ + γ_{6K} |

On the interpreter every float result is also bit-identical to a host model of
its fixed summation order (per step of 16, per product, per k).
Mutations, each red then reverted: dropping one byte-plane product (all int
cases on gfx1100 and the interpreter); dropping the `m·d + n < t` coverage
predicate (the 1-D short launches); dropping the three order-2 products of the
bf16x3 split (the interpreter-order model, 954/960 outputs).

**Speed** (`sarek-hip/test/bench_matrix_rewrite.exe 3 10 10`, three sessions,
range of the three session medians, µs per launch (TFLOP/s)):

| kernel | 256×1024×3072 | 256×3072×3072 | 1000×1000×1000 |
|---|---|---|---|
| int32 naive (scalar) | 2104–2906 (0.55–0.77) | 5303–5800 (0.83–0.91) | 1596–1624 (1.23–1.25) |
| int32 rewritten | 1034–1577 (1.02–1.56) | 2315–2532 (1.91–2.09) | 1108–1114 (1.80) |
| f16 naive (scalar) | 1857–1983 (0.81–0.87) | 5112–7886 (0.61–0.95) | 1424–1473 (1.36–1.40) |
| f16 rewritten | 557–559 (2.88–2.89) | 583–588 (8.22–8.29) | 356–360 (5.55–5.61) |
| f32 naive (scalar) | 2086–2151 (0.75–0.77) | 5204–5246 (0.92–0.93) | 1478–1559 (1.28–1.35) |
| f32 rewritten, f16 operands | 637–641 (2.51–2.53) | 660–665 (7.26–7.32) | 432–433 (4.62–4.63) |
| f32 rewritten, bf16 | 988–989 (1.63) | 1016–1026 (4.71–4.76) | 676–677 (2.95–2.96) |
| f32 rewritten, bf16x3 | 2158–2175 (0.74–0.75) | 4250–4443 (1.09–1.14) | 2150–2230 (0.90–0.93) |
| `Sarek_gemm.Coopmat` f16 (hand-written, 2×2/w2) | 93–94 (17.2–17.3) | 138–139 (34.8–35.0) | — (needs multiples of 32) |
| `Sarek_gemm` tiled f32 (scalar, hand-written) | 497–506 (3.19–3.24) | 1441–1473 (3.28–3.35) | 594–599 (3.34–3.37) |

The rewrite is 1.4–2.5× (int32) and 3.4–9× (f16) faster than the naive kernel
it replaces, and faster than the hand-tiled scalar f32 GEMM for f16 and f32-as-f16
from N = 3072, but 4–6× slower than the hand-written SCoopmat GEMM, which loads
fragments straight from global memory and requires multiples of its tile, f16
row-major operands and no epilogue. The staged kernel pays for its generality
(any size, layout, batch, element case) with a shared-memory round trip and two
barriers per step of 16. A size-class variant that loads directly from global
memory when the region is aligned is a PR 3 item (launch-time specialisation).
Measured and kept: padding shared rows by 16 bytes (no change within noise);
measured and reverted: steps of 32/64 per barrier pair (slower: fewer resident
blocks). Measured and kept (2026-10-11, C-42): B staged transposed and loaded
column-major, 8-element staging groups; outputs bit-identical
(`spoc-pipeline-logs/pkgF/F2-bits-c41-vs-c42.txt`), staged kernels 1.3–3.5×
faster (f32/f16 256×3072×3072 659 → 269 µs, int32 2664 → 766 µs;
`spoc-pipeline-logs/pkgF/F2-bench-matrix-rewrite-c42-summary.txt`).
