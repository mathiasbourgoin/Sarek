# backlog-151 — the 18 unmeasured f16 shapes: preserved run output, 2026-07-27

The numbers in
[`docs/fp-contraction-policy.md`](../../fp-contraction-policy.md) §13 are read
off these files. They are committed for the same reason slice 1's are
(`../f16-slice1-2026-07-27/README.md`): a measurement quoted only as prose
cannot be checked.

**What this settles.** `fp-contraction-policy.md` §12.4 closed slice 1 with one
candidate generative rule — *"a narrowing absorbs the whole f32 tree feeding it,
cut where `NoContraction` binds"* — marked **unverified as a general rule**, and
named the remaining 18 of the 20 emittable shapes as what would confirm or break
it. It is **broken**, and the corrected rule is in §13.

## What was run

All on one workstation: AMD Ryzen 9 7950X, Linux 7.1.2-3-cachyos, Mesa
26.1.4-arch3.1, DRM 3.64. No remote machine.

| file | command | devices |
|---|---|---|
| `host-separation.txt` | `dune exec tools/f16_shape_catalogue/probe/probe_f16_shape_separation.exe` | none — host only |
| `vulkan-radv.txt` | `dune exec sarek-vulkan/probe/probe_vulkan_f16_model_agreement.exe -- --catalogue` | `AMD Radeon RX 7900 XTX (RADV NAVI31)`, `AMD Ryzen 9 7950X 16-Core Processor (RADV RAPHAEL_MENDOCINO)`, radv / Mesa 26.1.4-arch3.1 / Vulkan 1.4.354 |
| `opencl-rusticl.txt` | `dune exec sarek-opencl/probe/probe_opencl_f16_model_agreement.exe -- --catalogue` | `AMD Radeon RX 7900 XTX (radeonsi, navi31, ACO, DRM 3.64, 7.1.2-3-cachyos)` and the Raphael iGPU equivalent |
| `vulkan-radv-isa.txt` | `RADV_DEBUG=asm dune exec sarek-vulkan/probe/probe_vulkan_f16_model_agreement.exe -- --shape <id> --variant <plain\|precise>` | RX 7900 XTX (RADV NAVI31) |
| `vulkan-radv-eager-cut.txt` | superseded run, kept as evidence — see below | both RADV devices |

`--shape`/`--variant` compiles **one** shader per process, so each
`RADV_DEBUG=asm` dump is attributable to a named shape and variant. A full
catalogue run compiles sixty shaders in sequence and reading its dump would mean
guessing which belongs to which, which is not a machine-code evidence tier.

## Why `vulkan-radv-eager-cut.txt` is kept

The rule's cut clause says *"cut wherever `NoContraction` forbids a multiply-add
**contraction**"*. That has two readings, and shape **A8** — `narrow(fma x 1.1
1000.)` — separates them: does the cut also apply to an fma the *author wrote*,
which is not a contraction of anything?

`vulkan-radv-eager-cut.txt` is the run under the **eager** reading, where it
does. A8 `precise` reports `S_absorb_all` there, against a prediction of the cut
model, so the eager reading is refuted and the literal one is what the code
implements. The ISA agrees at machine-code tier: A8's `plain` and `precise`
disassembly are **byte-identical**, a single `v_fma_mixlo_f16 v3, v1, 0xcccd,
v2`. Keeping the superseded run is the difference between "the code implements
the literal reading" and "the measurement chose it".

**Read only A8 out of that file.** It predates two other fixes and its A5 and
A10 rows are wrong for a reason unrelated to the cut: the division model dropped
the sign of `-0/3`, so it reports `x = -0` as unmatched on A5 where the device is
correct. That bug is itself worth noting — it was found by the same instrument,
printing the device bit pattern beside each model's at the first unmatched input,
and it is fixed in the committed code.

## How to read them

Each sweep prints one line per model; `0 / 63488 disagreements` marked `<==
EXACT, element-wise` is the model the device is bit-identical to. Three things
are new relative to slice 1's files and matter when reading:

- **`N distinct models`.** How many distinct functions the five slice-1 policies
  induce on that shape over the whole domain. **`1 distinct model` means the
  shape is NON-DISCRIMINATING**: every policy is the same function, so whatever
  the device returns is not evidence for or against anything. Eight of the
  twenty shapes are in that state. A table of twenty zeros would otherwise read
  as twenty confirmations.
- **`NO SINGLE MODEL (mixture)`** is distinguished from **`NO MODEL (N
  unmatched)`**. §1.2 requires bit-identity to *one* member on *every* input,
  which is stronger than every input matching *some* member. Shape B4 `plain`
  under the old rule was the first case where those two differ.
- **the reporting control**, printed before any device number: a host-computed
  **non-strict** model is fed through the same classifier and the harness must
  *name that model*. Slice 1's re-absorbed control was caught only because the
  harness reported the wrong model rather than an implausible count; a
  count-only sweep over twenty shapes would have missed the same thing twenty
  times.

## Reproducing

```
dune exec tools/f16_shape_catalogue/probe/probe_f16_shape_separation.exe
dune exec sarek-vulkan/probe/probe_vulkan_f16_model_agreement.exe -- --catalogue
dune exec sarek-opencl/probe/probe_opencl_f16_model_agreement.exe -- --catalogue
```

The first needs no GPU. It runs the calibration that pins the generic evaluator
to slice 1's seven hand-written closed forms — bit-for-bit on all 63488 inputs,
on the two shapes slice 1 measured on a device — and refuses to print anything
else until that passes. Nothing in the other two files is readable if it fails,
because the generator would then not be the thing slice 1 measured.
