# lsr and automatic unroll: measurements (2026-10-10)

Spec: `specs/codegen-lsr-unroll.md`.

## Conditions

- tartarus: Ryzen 9 7950X, RX 7900 XTX (gfx1100), ROCm 7.2 (HIP 7.2.53211,
  AMD clang 22.0.0git), Linux 7.1.2-3-cachyos, 2026-10-10 21:50–21:53 CEST.
- The machine was shared: load average 12–17 from other jobs during the runs.
  Repeated runs of one configuration differ by up to 30 % (see the raw
  summaries). A difference is claimed only when the ranges of the per-run
  medians of the two configurations do not overlap (e.g. lsr q5_K hot: main
  30.1–38.9, branch 23.4–24.3 µs).
- ladon (Apple M4, macOS) for the Metal and Apple OpenCL e2e only.

## Benchmark: a downstream engine's `bench/kbench.exe`

A downstream LLM inference engine (wave-per-row K-quant matvecs), built in a dune
workspace together with a Sarek tree, so the same kbench source runs against
three code generators:

- `main`: Sarek `origin/main` at `a120b92c`;
- `branch-nounroll`: this branch with `SAREK_AUTO_UNROLL=0` (lsr change only);
- `branch`: this branch (lsr change and unroll hint).

Two kernel sources:

- shipped: the engine's kernels as committed. They already avoid `lsr` (bytes
  extracted with `asr` and a mask) because of this very codegen problem.
- `lsr-` prefix: the same kernels with that workaround reverted to `lsr`
  (`downstream-matvec-asr-to-lsr.diff`; every changed shift is masked or acts on
  a non-negative value, so results are identical). This is the form a kernel
  author writes.

Command, three interleaved rounds of the three configurations:
`kbench.exe 7 exp_` on the HIP device (with or without `SAREK_AUTO_UNROLL=0`). The `gpu`
column is GPU time between events around 200 launches behind a blocker kernel;
the summary is the median of 7 repetitions. Raw summaries in `kbench/`.

Median of the three rounds, µs (`gpu` column):

| Case | Source | main | branch-nounroll | branch |
|---|---|---|---|---|
| exp_down q5_K hot | lsr | 31.59 | 23.65 | 23.88 |
| exp_down q5_K cold | lsr | 36.29 | 29.27 | 31.05 |
| exp_down6 q6_K hot | lsr | 39.16 | 33.00 | 32.53 |
| exp_down6 q6_K cold | lsr | 34.12 | 33.08 | 31.90 |
| exp_gate q4_K hot | lsr | 22.71 | 22.90 | 21.50 |
| exp_gate q4_K cold | lsr | 27.18 | 27.48 | 27.88 |
| exp_down q5_K hot | shipped | 26.93 | 26.42 | 25.43 |
| exp_down q5_K cold | shipped | 32.58 | 31.44 | 30.40 |
| exp_down6 q6_K hot | shipped | 32.08 | 31.37 | 32.70 |
| exp_down6 q6_K cold | shipped | 34.59 | 32.56 | 33.60 |
| exp_gate q4_K hot | shipped | 21.58 | 21.38 | 20.76 |
| exp_gate q4_K cold | shipped | 27.35 | 27.35 | 27.68 |

Reading:

- The lsr change speeds up the `lsr` form of q5_K (hot 31.6 → 23.7–23.9) and
  q6_K (hot 39.2 → 32.5–33.0). q4_K does not change beyond noise.
- With the native shift, the `lsr` form is as fast as the hand-written `asr`
  form: its gfx1100 code is identical (table below).
- The unroll hint changes nothing measurable here: branch and branch-nounroll
  compile to the same gfx1100 code for all three kernels, so their
  differences are noise.

## Generated code

`generated/q5_k-lsr-main-vs-branch.diff`: the HIP source of the `lsr` q5_K
kernel before and after (8 expansions replaced by casts, one `#pragma
unroll`). Source size 4036 → 3692 bytes. `generated/q5_k-shipped-main-vs-branch.diff`:
the shipped kernel gains only the pragma (and one `w lsr 3l` in q8_0, not shown).

`isa-stats.txt`: static instruction counts of `sarek_kern` compiled offline
with `hipcc -O3 -ffp-contract=off --offload-arch=gfx1100` (the runtime uses
hiprtc with `-ffp-contract=off`, so this approximates, not reproduces, the
runtime code object):

- q5_K, `lsr` form: main has 193 VALU and 2 `global_load_b128` because clang
  leaves the `wi` loop rolled (its body is inflated by the expansion). Branch:
  274 VALU and 12 `b128`, the loop unrolled — identical to the shipped `asr`
  form.
- `ws-main-lsr-pragma`: main's expansion plus `#pragma unroll` on that loop
  (what the engine measured as 40.7 → 35.9 µs): 320 VALU and 12 `b128`. The
  pragma does make clang unroll a body it otherwise declines.
- With the native shift, clang unrolls these loops by itself, so
  `branch-nounroll` and `branch` are identical.

## Correctness

- `e2e-tartarus.txt`, `e2e-ladon.txt`: `test_lsr_unroll.exe` on every device
  (HIP ×2, CUDA/PTX through ZLUDA ×2, OpenCL ×2, Vulkan ×2, Native,
  Interpreter ×2; Metal, Apple OpenCL, Native, Interpreter on ladon). All
  pass.
- A mutation (CUDA spelling with a signed cast) makes `lsr32`/`expr` fail on
  both HIP devices (427/768 and 396/768 rows), so the e2e is not vacuous.
- The engine's differential kernel tests (`test/kernels/test_kernels.exe`, 19
  tests) pass on HIP, Native and Interpreter against this branch, for both the
  shipped and the `lsr` kernel sources.
