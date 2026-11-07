---
name: roster-spec
type: spec
status: live
feature: f32 -> f16 narrowing barrier only where it changes results; explicit single-rounding opt-in
brief: global review C-40 (package F, item F3)
date: 2026-10-11
version: 1.0.0
---

# Spec — the f32 → f16 narrowing barrier (C-40)

The CUDA/HIP code generator narrows f32 to binary16 as
`__float2half(sarek_f32_barrier(x))`. On AMD the barrier is
`asm volatile("" : "+v"(x))`: it stops LLVM's AMDGPU backend from absorbing the
operation that produces `x` into the narrowing (`v_fma_mixlo_f16`,
`v_add_f16`, ...), which rounds once where the DSL and the interpreter round
twice (round to binary32, then to binary16). Why the barrier is needed and why
no compiler flag replaces it: `docs/fp-contraction-policy.md` §2 and
`docs/optimization/amdgpu-f16-fusion-shape-audit.md`.

The barrier costs. On gfx1100 it keeps the multiply out of the narrowing and,
because the asm is volatile, it also keeps the narrowed values from being
packed and their stores merged, and the loads feeding them from being merged
(`F3-isa-before.txt`: 32 `ds_store_b16` and 8 `global_load_b32` per unit
instead of 4 `ds_store_b128` and 2 `global_load_b128`). A downstream LLM inference engine measured −20 %
on its fused Q4_K GEMM.

This spec removes the barrier where it provably changes nothing (default on)
and adds an explicit opt-in for the rest.

## 1. Rules

| # | Rule |
|---|---|
| FR-001 | A narrowing keeps the barrier unless `Sarek_ir_f16_narrowing.form` proves one of §2's cases. Anything the analysis cannot prove keeps it. |
| FR-002 | Case T1 (exact value, safe top) and case T2 (binary16 sum) are emitted as a bare `__float2half(e)`. |
| FR-003 | Case T1 whose top is a product that may be a zero of the wrong sign is emitted as `sarek_narrow_exact_product(a, b)` (§2.3). |
| FR-004 | `Sarek_f16_policy.accept_single_rounding scope ~reason` removes every barrier from the kernels in `scope` (§4). It is separate from `Sarek_coopmat.accept_relaxed_accumulation`; neither implies the other. |
| FR-005 | Default results are bit-identical to the interpreter, including the sign of zero. Only FR-004 may change them. |

## 2. When one rounding equals two

Notation: `round32`, `round16` are IEEE round-to-nearest-even to binary32 and
binary16. The narrowed expression is `e`; the interpreter computes
`round16(v32)`, `v32` being `e` evaluated with every operation rounded to
binary32. Without the barrier the device may evaluate `e` with any of the
following, all measured on gfx1100 (shape audit; §2.3):

- unchanged: f32 operations, then `v_cvt_f16_f32`;
- the top operation absorbed into the narrowing: `v_fma_mixlo_f16(a, b, c)`
  computes `round16(a * b + c)` from the exact value — for a lone product with
  `c = +0`, for a lone sum as `a * 1 + c`;
- the top operation demoted to binary16 (`v_add_f16`, `v_sub_f16`,
  `v_mul_f16`) when its operands are binary16 values: `round16(a op b)`.

So any evaluation rounds the exact value of the top operation once, or rounds
it to binary32 first; operations below the top are evaluated as without the
barrier (the barrier only ever separated the top from the narrowing).

### 2.1 T1 — the value is exact

**Theorem.** If every operation in `e` has an exact result representable in
binary32 (no rounding happens anywhere in `e`), then `v32 = v`, the exact real
value, and every evaluation above yields `round16(v)` — up to the sign of a
zero result, see §2.3.

Proof: `round32(v) = v`; a fused or demoted evaluation of the top computes
`round16` of the top's exact value, which is `v` because the operands are
exact. ∎

Conditions the analysis enforces at every operation (`representable`):

- significand: the value is `k · 2^q` with `|k| ≤ 2^24`;
- no subnormal binary32 result: every nonzero result is `≥ 2^-126` in
  magnitude (`q ≥ −126`), so a flush-to-zero mode cannot change it;
- no overflow: `|k| · 2^q ≤ max binary32`.

Leaves are values no compiler can re-derive: a binary16 value widened
(`k ≤ 2047`, `q ∈ [−24, 5]`), an int32 converted to float whose range is known
and `≤ 2^24` in magnitude, a binary32 constant, a scalar kernel argument, a
read of a global f32 array that the kernel never writes nor passes on, and
`Int32.float_of_bits` of a read-only int32 array element or of a scalar
int32 argument. Over anything else `float_of_bits` is not a leaf: LLVM folds
`float_of_bits (bits_of_float p)` to `p`, and forwards an int stored earlier
in the kernel to a later load, then fuses `p` into the narrowing (review:
132/16252928 on hip:0). A leaf alone is exact (nothing to fuse). Variables are
looked through to every binding of their name (`let`, mutable `let`,
assignment, loop bounds); a name with a binding the analysis cannot follow
(pattern variable, helper parameter — a helper is inlined, so its caller's
arithmetic reaches it) is unknown.

Integer ranges come from constants, masks (`x land m`, `m ≥ 0`), shifts by
constants, `mod`/division by positive constants, sums and products (any
int32 overflow gives the full range), loop bounds.

Non-finite inputs: a binary16 leaf may be ±∞ or NaN. IEEE arithmetic on them
involves no rounding, so both evaluations give the same infinity; NaN results
are NaN in both, with possibly different sign or payload bits — NaN bits are
not part of the f16 discipline (the f16 test harnesses compare NaN by
NaN-ness).

### 2.2 T2 — a binary16 sum

**Theorem (Figueroa, double rounding).** For binary16 values `a`, `b`:
`round16(round32(a ± b)) = round16(a ± b)`, because binary32 has
`24 ≥ 2·11 + 1` significand bits. A sum's sign of zero is the same in
binary16 and binary32 arithmetic. So `e = a ± b` with `a`, `b` binary16 values
(widened binary16, binary16 constants, integers of magnitude ≤ 2048, negations
and variables of those) needs no barrier even when `a ± b` is not exact in
binary32 (shape audit A3/A4/A10).

### 2.3 The sign of zero (measured)

The absorbed product is `v_fma_mixlo_f16(a, b, 0)` with a **+0** addend
(`F3-plusv-uniform-isa.txt`: `v_fma_mixlo_f16 v1, s2, s3, 0`). When `a · b` is
`−0` (e.g. a negative scale times a zero quant), `−0 + +0 = +0`, where IEEE
multiplication and the interpreter give `−0`. Measured on hip:0 with a T1-only
analysis (`F3-signed-zero-hazard.txt`): 31999 of 16252928 Q8_0 outputs
(`d · int8`, every binary16 `d`) came out `+0` instead of `−0`; nothing else
differed. A negation absorbed into its operand has the same hazard.

So a T1 narrowing whose top (through variables and conditionals) is a product
or a negation is a bare narrowing only when it cannot be a zero of the wrong
sign: the product is never zero (`nonzero`) or never negative (`nonneg`, so
its zero is `+0`); a negation's operand is never zero. `nonneg` holds for
non-negative integers, positive constants, `+0`, and sums and products of
such values; a difference `a - b` is analysed as `a + (-b)` and is never
`nonneg` (review: `(i - j) * k` of bytes is `-0` when `i < j` and `k = 0`).
Otherwise:

- a product written at the top, `e = a * b`, is emitted as
  `sarek_narrow_exact_product(a, b)`: the bare narrowing of `a * b` with
  `sign(a) xor sign(b)` or-ed into the result's sign bit. For an exact product
  that is the sign of the exact value in every case — a nonzero value keeps its
  sign through rounding, and a zero product's IEEE sign is that xor — so the
  result equals `round16(v32)` bit for bit, fused or not. The ISA keeps the
  fusion and the merged stores, plus two integer operations per value
  (`F3-isa-after.txt`, q8h).
- a product behind a variable, or a negation, keeps the barrier.

## 3. Implementation

`sarek/codegen/Sarek_ir_f16_narrowing.ml`: facts per function body
(bindings by name, array element types, global/written arrays), int32 ranges,
the exact-value bound `{kmax; qlo; qhi; binary16; nonzero; nonneg}`, `form`.
Each query has a fuel of 10 000 variable lookups; running out answers
"unknown". `Sarek_ir_cuda` computes facts for the kernel and for each helper
and asks `form` at each `ECast (TFloat16, _)`. The `sarek_f32_barrier` and
`sarek_narrow_exact_product` declarations are emitted in every f16 kernel.

Only the CUDA-C generator (CUDA and HIP) emits the barrier; OpenCL and GLSL
refuse f16, Metal does not fuse (`docs/fp-contraction-policy.md` §2). On
NVIDIA the barrier is an identity and `ptxas` does not absorb the narrowing,
so the change is codegen-only there (compile-only: no NVIDIA device).

## 4. The opt-in: single rounding

```ocaml
Sarek_f16_policy.accept_single_rounding (Kernel "my_kernel") ~reason:"..."
(* : (unit, Sarek_f16_policy.refusal) result *)
```

A blank reason (`Blank_reason`) and `Kernel "sarek_kern"`
(`Default_kernel_name`: the name of every unnamed `[%kernel]`, library
kernels included) are refused and record nothing. Process-wide, one-way, thread-safe, like the `Sarek_coopmat` opt-ins, and
separate from them. `Kernel name` matches `kern_name`; unnamed `[%kernel]`
kernels are all `"sarek_kern"`; a fused kernel is covered when every one of
its parts is (`Sarek_policy.register_fused`; a grant on the generated fused
name counts for nothing). The policy is read when the source is generated and
the launch memo is keyed on it: a kernel generated before the call is
regenerated, without barriers, by its next launch (M2). The source of a covered
kernel says so in a comment.

What changes, per narrowing, when the compiler fuses:

- `v = a * b`: `round16(a · b)` instead of `round16(round32(a · b))` — at most
  one binary16 ulp, only near a binary16 tie; a `−0` product may come out `+0`;
- `v = a * b ± c` fused into one `v_fma_mix`: `round16(a · b ± c)` instead of
  `round16(round32(round32(a · b) ± c))` — at most one binary16 ulp plus
  `|a · b| · 2^-24`, the binary32 rounding error of the product, which is
  large relative to the result when `a · b ± c` cancels.

The device result is the more accurate one. The interpreter keeps rounding
twice: which fusions a compiler performs is not specified, so it cannot be
modelled bit for bit. A differential test of a covered kernel compares within
this bound (`test_hip_f16_single_rounding`).

## 5. Evidence

Machine: tartarus, RX 7900 XTX (gfx1100, hip:0), ROCm 7.2.4, 2026-10-11. Logs
in `/mnt/ssd-external-2to/spoc-pipeline-logs/pkgF/` (`F3-*`).

| What | Where | Result |
|---|---|---|
| Host proof of T1 (every binary16 × every int8/uint8, with the sign restoration; Q6_K triple products) and T2 (every binary16 against 128 others, ±), with a control | `sarek/tests/codegen_golden/test_f16_narrowing_theorem.ml` | all agree; control f16(x·1.1f) 2912/63488 differ |
| T2 exhaustively: all 63488² = 4 030 726 144 pairs of finite binary16 values, `+` and `−` (independent review; C with `_Float16`: `spoc-pipeline-logs/pkgF/scratch-F3-review/t2.c`, `t2-out.txt`) | outside the repository | 0 and 0 disagree; control x·1.1 1794 |
| Analysis vs reference evaluation, 4000 random expressions × 200 inputs, int leaves from three independent inputs | `test_f16_narrowing_property.ml` | 0 failures; 470 before the review fixes; with the sign rule removed, 2628 |
| Which narrowings keep the barrier | `test_f16_narrowing_golden.ml` | 26 cases |
| hip:0 vs f32-then-f16 model, bit for bit, ~16M values each: binary16 sums, byte products/sums, differences of bytes times a byte, Q8_0 and Q6_K signed products; Q4_K and a bitcast round trip keep 4 barriers | `sarek-hip/test/test_hip_f16_narrowing_exact.ml` | 0 differ each; Q8_0 without the sign restoration: 31999 differ, all ±0; before the review fixes: differences 4194240 (±0), bitcast 132 |
| Every emittable f16 shape, all 63488 binary16 inputs, shipping path | `sarek-hip/test/test_hip_f16_shapes.ml` | 0 mismatches as shipped (A1, A3, A4, A10 now without barrier) |
| Opt-in within the bound | `sarek-hip/test/test_hip_f16_single_rounding.ml` | 0 outside the bound |
| Time, dequantise-to-LDS microbenchmark | `sarek-hip/test/bench_f16_narrowing.ml`, `F3-bench-ab-raw.txt` | Q8_0 with binary16 scale: 932 → 182 µs (default); Q4_K as the downstream engine writes it: unchanged by default, 870 → 170 µs with the opt-in |

## 6. Limits and findings

- The downstream engine's kernels read their scales as f32 bits (`Int32.float_of_bits`), an
  arbitrary binary32 value: `d · q` is not provably exact, so their barriers
  stay by default. Reading a binary16 scale as binary16 makes Q8_0 and Q6_K
  products exact (T1). Q4_K/Q5_K subtract a second product with an
  independent scale: not exact in general, opt-in only.
- `"+v"` (C-40 side note): on gfx1100 there is no scalar float ALU, so a
  uniform product is computed in a VGPR anyway and the constraint costs
  nothing; on gfx1150 (scalar float ALU) it adds one `v_mov_b32` per uniform
  narrowing (`F3-plusv-uniform-isa.txt`). An `"s"` constraint is wrong for a
  divergent value and the generator has no uniformity information at this
  point; not changed.
- A non-volatile `asm("" : "+v"(x))` still blocks the fusion and lets the
  loads merge, but the stores stay 16-bit and the microbenchmark is slower
  (≈970 µs against ≈870 µs): not adopted.
