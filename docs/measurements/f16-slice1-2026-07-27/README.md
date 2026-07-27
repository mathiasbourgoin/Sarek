# #62 slice 1 — preserved run output, 2026-07-27

The numbers in
[`docs/fp-contraction-policy.md`](../../fp-contraction-policy.md) §12 and in
[`docs/design/f16-relaxed-accuracy.md`](../../design/f16-relaxed-accuracy.md)
§2 and §7 slice 1 are read off these files. They are committed because a
measurement quoted only as prose cannot be checked, and because the two
tripwires this project already ships were both corrected after a number in a
document turned out not to be what the harness computed.

## What was run

All on one workstation: AMD Ryzen 9 7950X, Linux 7.1.2-3-cachyos, Mesa
26.1.4-arch3.1, DRM 3.64.

| file | command | devices |
|---|---|---|
| `opencl-rusticl.txt` | `dune exec sarek-opencl/probe/probe_opencl_f16_model_agreement.exe` | `AMD Radeon RX 7900 XTX (radeonsi, navi31, ACO, DRM 3.64, 7.1.2-3-cachyos)`, `AMD Ryzen 9 7950X 16-Core Processor (radeonsi, raphael_mendocino, ACO, …)` |
| `vulkan-radv.txt` | `dune exec sarek-vulkan/probe/probe_vulkan_f16_model_agreement.exe` | `AMD Radeon RX 7900 XTX (RADV NAVI31)`, `AMD Ryzen 9 7950X 16-Core Processor (RADV RAPHAEL_MENDOCINO)`, radv / Mesa 26.1.4-arch3.1 / Vulkan 1.4.354 |
| `vulkan-radv-isa.txt` | `RADV_DEBUG=asm dune exec sarek-vulkan/probe/probe_vulkan_f16_model_agreement.exe -- --variant {s1_plain,s1_precise,s2_plain,s2_precise}` | RX 7900 XTX (RADV NAVI31) |

The ISA capture uses `--variant` deliberately: it compiles **one** shader per
process, so each `RADV_DEBUG=asm` dump is attributable to a named variant. A
full run compiles nine shaders in sequence and reading its dump would mean
guessing which belongs to which, which is not a machine-code evidence tier.

## How to read them

Each sweep prints one line per model. `0 / 63488 disagreements` marked `<==
EXACT, element-wise` is the model the device is bit-identical to. A line
`*** N inputs match NO named model ***` is the failure this slice existed to
look for; there are none in these files.

The controls are printed **before** the measurements, and there are three
kinds:

- **green** — a barriered kernel that must reproduce `S_strict` exactly. Until
  it does, a disagreement elsewhere could be a wrong buffer layout rather than
  a statement about the compiler.
- **positive** — a kernel that performs the fusion deliberately and must
  reproduce `S_fuse_mul_into_narrowing` exactly. Without it, a zero cannot be
  distinguished from a sweep that did not happen.
- **host, before any device is touched** — the model round-trip, the 620 and
  2912 separations, §1.3's `x = -907.5` case, and the full pairwise separation
  matrix. The matrix is the guard against a model set whose members coincide on
  the swept inputs and therefore discriminate nothing.

## Reproducing

The probes need no arguments and no GPU-specific setup beyond a Mesa driver:

```
dune exec sarek-opencl/probe/probe_opencl_f16_model_agreement.exe
dune exec sarek-vulkan/probe/probe_vulkan_f16_model_agreement.exe
```

Add `-- --host-only` to run the calibration and the separation matrix with no
device at all; that half is deterministic and machine-independent, and it is
the half that must be believed before any device number is read.
