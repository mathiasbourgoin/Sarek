---
layout: page
title: Device & Backend Selection
---

# Device & Backend Selection

Sarek auto-detects every available backend at startup (CUDA, OpenCL, Vulkan, Metal, plus
the CPU **Native** and **Interpreter** backends) and exposes each physical device. This
page explains how to list those devices and how to pick — or exclude — a specific backend
or device.

## List available devices

```bash
dune exec -- sarek-device-info
```

This prints every detected device across all backends with its framework and
capabilities, e.g.:

```
[0] AMD Radeon RX 7900 XTX (radeonsi, navi31) (OpenCL)
[1] AMD Radeon RX 7900 XTX (RADV NAVI31) (Vulkan)
[2] CPU Native (Parallel, 32 cores) (Native)
[3] CPU Interpreter (Sequential) (Interpreter)
```

The bracketed index is the device id used by `-d <id>` below. Note that the *same*
physical GPU usually appears more than once — once per framework that can drive it (e.g.
OpenCL and Vulkan).

## Disable backends with environment variables

Set any of these to `1` to keep that backend from registering at all. This is the
simplest way to avoid a flaky or unsupported driver, or to force execution onto a
particular backend.

| Variable | Effect |
|---|---|
| `SPOC_DISABLE_CUDA=1` | Skip the CUDA backend |
| `SPOC_DISABLE_OPENCL=1` | Skip the OpenCL backend |
| `SPOC_DISABLE_VULKAN=1` | Skip the Vulkan backend |
| `SPOC_DISABLE_METAL=1` | Skip the Metal backend |
| `SPOC_DISABLE_GPU=1` | Skip **all** GPU backends (CUDA, OpenCL, Vulkan, Metal) |

Examples:

```bash
# Run only on Vulkan (skip the OpenCL view of the same GPU)
SPOC_DISABLE_OPENCL=1 dune exec -- sarek-device-info

# Force CPU-only execution (no GPU backends register)
SPOC_DISABLE_GPU=1 dune exec -- my_program.exe
```

## Test-only strictness variable

`SAREK_REQUIRE_PTX=1` is read by one test, `sarek/tests/e2e/test_soa_emitter_equiv.ml`,
not by the library. Unset (the default everywhere, including CI), a host with no
CUDA/PTX device silently skips that test's SoA legs and the binary still exits 0 —
the normal, desired outcome on the vast majority of hosts. Set to exactly `1` on a
host that is expected to have a CUDA/PTX device (e.g. a ZLUDA-equipped machine), it
turns "no device reported framework `CUDA/PTX`" into a failure, so a regression that
silently drops the device is caught instead of reading as a pass. It is opt-in-by-hand:
no dune rule, Makefile target or CI job sets it today.

```bash
SAREK_REQUIRE_PTX=1 LD_LIBRARY_PATH=$HOME/opt/zluda \
  dune exec sarek/tests/e2e/test_soa_emitter_equiv.exe
```

## Select a device in the example / benchmark programs

The bundled e2e examples and benchmarks accept backend/device flags:

| Flag | Effect |
|---|---|
| `-d <id>` | Run on the device with this index (from `sarek-device-info`) |
| `--vulkan` | Use the first Vulkan device |
| `--native` | Use the CPU Native (parallel) backend |
| `--interpreter` | Use the CPU Interpreter backend |
| `--metal` | Use the first Metal device |
| `--benchmark` | Run across all available devices |

```bash
# Vector-add on the Vulkan device specifically
dune exec sarek/tests/e2e/test_vector_add.exe -- --vulkan

# …on device id 2
dune exec sarek/tests/e2e/test_vector_add.exe -- -d 2
```

> Tip: when several frameworks expose the same GPU, the *default* device is usually the
> first GPU found (often the OpenCL view). Use `--vulkan` or `-d <id>` to target a
> different framework for the same hardware.

## See also

- [Backends](backends.html) — what each backend supports.
- [Troubleshooting](faq.html#troubleshooting) — what to do when a backend faults or a
  device misbehaves.
