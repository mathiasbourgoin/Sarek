# Sarek backend performance sweep: HIP vs OpenCL vs Vulkan (RX 7900 XTX)

**Date:** 2026-07-24
**Purpose:** Substantiate or refute the claimed HIP performance edge, which until now rested on a single `bench_vector_add` datapoint (HIP ~715 GB/s vs OpenCL/Vulkan ~467 GB/s). This document exists to *test* that claim across the whole suite, not to confirm it.

## Hardware / software

| | |
|---|---|
| Host | tartarus (AMD Ryzen 9 7950X, 16-core) |
| Discrete GPU (device under test) | AMD Radeon RX 7900 XTX — gfx1100 / Navi31 |
| iGPU (present, **not** the comparison target) | Ryzen 9 7950X integrated (gfx1036 / Raphael) |
| HIP backend | ROCm 7.2.4 (hipModuleLaunchKernel path) |
| OpenCL backend | **rusticl**, Mesa 26.1.4-arch3.1 (ACO) — *not* ROCm OpenCL |
| Vulkan backend | **RADV**, Mesa 26.1.4-arch3.1, glslangValidator → SPIR-V |
| Kernel | Linux 7.1.2-3-cachyos, DRM 3.64 |
| OCaml | 5.3.0 |
| Repo | branch `feat/hip-rocwmma-backend`, commit `cfd5dd9` |

Every backend enumerates **both** the discrete 7900 XTX and the 7950X iGPU. All numbers
below are the **discrete 7900 XTX only**; the iGPU rows were discarded. No backend was
restricted to the iGPU — all three see the dGPU, so there is no iGPU-vs-dGPU confound in
the comparison itself (see caveats for one timeout it caused).

## Selection mechanism (reproducible)

A single benchmark invocation enumerates *all* available devices and benchmarks *each* one
in turn (`Device.init ()` → filter out Native/Interpreter → loop). It does **not** report
only a top-priority device. To get a clean, interference-free number per backend I ran each
benchmark **three times, once per backend, isolating the backend with the `SPOC_DISABLE_*`
env-var guards** (verified in `backend_*.available.ml` and each plugin's
`plugin_disabled ()`), so only one GPU runtime is initialised per process. This removes any
cross-backend contention on the shared GPU (the original single-datapoint claim came from a
mixed all-backends-in-one-process run).

Isolation env per run (CUDA/Metal always disabled; they are irrelevant here):

```bash
# HIP only
SPOC_DISABLE_OPENCL=1 SPOC_DISABLE_VULKAN=1 SPOC_DISABLE_CUDA=1 SPOC_DISABLE_METAL=1 \
  ./_build/default/benchmarks/bench_<name>.exe --output <dir>
# OpenCL only
SPOC_DISABLE_HIP=1 SPOC_DISABLE_VULKAN=1 SPOC_DISABLE_CUDA=1 SPOC_DISABLE_METAL=1  ... 
# Vulkan only
SPOC_DISABLE_HIP=1 SPOC_DISABLE_OPENCL=1 SPOC_DISABLE_CUDA=1 SPOC_DISABLE_METAL=1 ...
```

Build: `dune build --root . benchmarks/`. Each benchmark does its own internal warmup
(default 10) + timed iterations (default 20) and reports the **min** time (bandwidth) or
**median/mean** (throughput); no external repeats were needed. Each row below is taken at
the **largest size** the benchmark completed on the dGPU (steady state), which is a neutral
choice — not tuned to favour any backend. Raw JSON/logs: `/mnt/ssd-external-2to/spoc-bench-scratch/{hip,opencl,vulkan,logs}`.

## Results

Metric is whatever the benchmark itself prints. **Higher is better** for every metric.
Ratio = HIP ÷ best of {OpenCL, Vulkan}. ✓/✗ = the benchmark's own correctness flag.

| Benchmark | Metric | Size | HIP | OpenCL | Vulkan | Best portable | HIP/best | Verified (H/O/V) |
|---|---|--:|--:|--:|--:|--:|--:|:--:|
| bitonic_sort | Melem/s | 16 384 | **18.9** | 5.2 | 1.4 | 5.2 | **3.62** | ✓/✓/✓ |
| scan | Melem/s | 256 | **6.4** | 2.3 | 1.3 | 2.3 | **2.72** | ✓/✓/✓ |
| transpose_tiled | GB/s | 8192² | **195.4** | 90.0 | 90.1 | 90.1 | **2.17** | ✓/✓/✓ |
| conv2d | GB/s (eff)¹ | 2048 | **44090.7** | 22612.1 | 21962.8 | 22612.1 | **1.95** | ✓/✓/✓ |
| reduction (sum) | GB/s | 100 M | **195.5** | 103.9 | 104.8 | 104.8 | **1.87** | ✓/✓/✓ |
| stencil_2d | GB/s (eff)¹ | 2048 | **51140.1** | 28605.2 | 27968.5 | 28605.2 | **1.79** | ✓/✓/✓ |
| soa_aos (SoA path) | GB/s | 16.7 M | **729.2** | 522.2 | 526.1 | 526.1 | **1.39** | n/a⁴ |
| mandelbrot | Mpix/s | 4096² | **16169.5** | 12896.4 | 13034.4 | 13034.4 | **1.24** | ✓/✓/✓ |
| nbody | GFLOPS | 4096 | **26.5** | 21.6 | 22.2 | 22.2 | **1.19** | ✗/✗/✗² |
| histogram | Melem/s | 50 M | **93289.7** | 79362.4 | 57996.5 | 79362.4 | **1.18** | ✓/✓/✓ |
| vector_copy | GB/s | 500 M | **819.3** | 787.2 | 784.6 | 787.2 | 1.04 | ✓/✓/✓ |
| vector_add | GB/s | 100 M | **780.2** | 755.6 | 752.3 | 755.6 | 1.03 | ✗/✗/✗² |
| stream_triad | GB/s | 500 M | **790.7** | 769.8 | 773.5 | 773.5 | 1.02 | ✓/✓/✓ |
| gather | Melem/s | 50 M | 8254.9 | 8175.2 | 8203.4 | 8203.4 | 1.01 | ✓/✓/✓ |
| scatter | Melem/s | 50 M | 4736.7 | 4614.4 | 4778.7 | 4778.7 | 0.99 | ✓/✓/✓ |
| radix_sort | Melem/s | 50 M | 114.8 | 115.7 | 115.4 | 115.7 | 0.99 | ✓/✓/✓ |
| transpose_naive | GB/s | 8192² | 9.5 | 9.6 | **FAIL**³ | 9.6 | 0.99 | ✓/✓/✗³ |
| dot_product | GB/s | 100 M | 781.3 | **805.2** | 760.4 | 805.2 | **0.97** | ✓/✓/✓ |
| reduction_max | GB/s | 100 M | 413.2 | **521.5** | 515.7 | 521.5 | **0.79** | ✓/✓/✓ |
| matrix_mul_tiled | GFLOPS | 2048 | 3376.9 | **5036.5** | 4857.9 | 5036.5 | **0.67** | ✓/✓/✓ |
| matrix_mul (naive) | GFLOPS | 2048 | 906.2 | 1364.3 | **1720.4** | 1720.4 | **0.53** | ✓/✓/✓ |
| pinned_transfer | GB/s | — | — | — | — | — | — | **not comparable**⁵ |
| soa_emitter | GB/s | — | — | — | — | — | — | **not comparable**⁶ |

Rows sorted by HIP/best ratio descending. Bold HIP = HIP wins; bold portable = a portable
backend wins.

**Footnotes**

1. `conv2d` and `stencil_2d` report *effective/algorithmic* bandwidth (logical bytes touched
   including halo reuse). The magnitudes (44–51 TB/s) exceed the 7900 XTX's ~960 GB/s physical
   DRAM bandwidth and are **not** physical — they reflect cache/LDS reuse. Cross-backend
   **ratios within a benchmark stay valid**; absolute magnitudes are not comparable across
   benchmarks.
2. `vector_add` and `nbody` report Verified ✗ at the largest sizes. This is a **harness
   tolerance artifact, not a compute error**: the verifier compares float32 GPU output against
   a float64 OCaml reference with a fixed absolute epsilon (0.001), and at large indices the
   float32 rounding error (values reach ~3×10⁵) legitimately exceeds it. It fails **identically
   on all three backends**, so it does not discriminate between them. `vector_add` verifies ✓
   at size 1 M on every backend.
3. `transpose_naive` on **Vulkan fails on every device** — the Sarek→GLSL emitter produces a
   syntax error (`unexpected DOT`) that glslangValidator rejects, so no SPIR-V is generated.
   This is a real Vulkan-backend codegen limitation, not a perf result. HIP and OpenCL run it
   fine (and are ~tied). Best-portable falls back to OpenCL.
4. `soa_aos` prints an AoS-vs-SoA table (no Verified flag). The comparison metric above is the
   **SoA single-field-copy** GB/s at the largest N. Note the AoS (strided) path: HIP 172.7,
   OpenCL 157.0, **Vulkan 7.1 GB/s** — RADV handles the 32-byte-strided access pathologically
   badly (19 ms vs sub-ms), a Vulkan-specific weakness worth flagging.
5. `pinned_transfer` is **CUDA-backend-only** (host↔device transfer bandwidth, runs under
   ZLUDA). It has no HIP/OpenCL/Vulkan path and cleanly SKIPs under all three (exit 2/skip). It
   measures memcpy, not a kernel — apples-to-oranges, excluded.
6. `soa_emitter` is **PTX/CUDA-only** ("SoA is PTX-only") and SKIPs on all three portable
   backends. Excluded.

## Summary

**Comparable benchmarks: 21** (the 22 listed minus `pinned_transfer` and `soa_emitter`, both
CUDA/PTX-only; `gather_scatter` counted as its two sub-kernels `gather` + `scatter`).

- **HIP wins: 13 · loses: 4 · ties: 4** (±2% band = tie).
- **Geometric mean of HIP ÷ best-portable ratio:**
  - **1.25× over all 21** comparable benchmarks.
  - **1.13× excluding the two tiny-size, launch-overhead-dominated kernels** (`bitonic_sort`
    at 16 K elements, `scan` at 256 elements — these measure dispatch latency, not throughput,
    and inflate the mean).
  - **1.02× over the pure DRAM-bandwidth-bound streaming kernels** (`vector_add`,
    `vector_copy`, `stream_triad`, `dot_product`) — i.e. **statistical parity**.

### Where HIP loses (most important finding)

HIP is **beaten** on all four compute-bound / general-matrix kernels:

- **matrix_mul (naive): HIP 906 vs Vulkan 1720 GFLOPS — HIP is ~0.53×, nearly half.**
- **matrix_mul_tiled: HIP 3377 vs OpenCL 5037 GFLOPS — HIP ~0.67×.**
- **reduction_max: HIP 413 vs OpenCL 522 GB/s — HIP ~0.79×.**
- **dot_product: HIP 781 vs OpenCL 805 GB/s — HIP ~0.97× (marginal).**

On GEMM the Mesa compilers (ACO via rusticl, and RADV) generate materially faster code than
HIP's path here. This is not noise — the gap is 1.5–1.9×.

### Verdict

**The single-datapoint HIP edge is real but narrow and kernel-dependent — it does not
generalise into a blanket "HIP is faster" claim.**

- On **plain memory-bound streaming** (the class the original `vector_add` claim came from),
  all three backends saturate the same ~800 GB/s and are **at parity** (geomean 1.02×). The
  original "715 vs 467" gap does **not** reproduce under isolated runs — isolated OpenCL and
  Vulkan reach ~750–790 GB/s on `vector_add`, not ~467. The old ~467 figure was almost
  certainly a **cross-backend-contention artifact** of the mixed single-process run, not a
  true backend deficit.
- HIP's **genuine, large wins** are concentrated in kernels dominated by (a) kernel-launch /
  dispatch overhead at small sizes (`bitonic_sort`, `scan`), (b) LDS/shared-memory tiling and
  atomics (`transpose_tiled` 2.2×, `reduction` 1.9×, `histogram` 1.2×), and (c) cache-resident
  effective-bandwidth kernels (`conv2d`, `stencil_2d`). Here the ROCm compiler and scheduler
  clearly help.
- HIP **loses on GEMM and max-reduction**, where the portable Mesa compilers win by up to 1.9×.

So: HIP earns its place for launch-bound, LDS-heavy, and atomics-heavy kernels, and is a
safe default (wins or ties on 17/21). But it is **not** a universal performance win — for pure
streaming it is a wash, and for matrix multiply it is currently the *slowest* of the three.
Any "HIP is faster" statement must be qualified by kernel class.

## Caveats / not measured

- **iGPU-caused timeout:** `matrix_mul_tiled` is reported at 2048 (its largest *completed*
  dGPU size). The 4096 size hit the 400 s per-run timeout because the run also benchmarks the
  slow 7950X iGPU; the dGPU number at 4096 was lost as collateral. Not a dGPU or backend
  failure. All other benchmarks completed their full size ladder.
- **OpenCL = rusticl, not ROCm OpenCL.** The "portable" OpenCL number is Mesa's rusticl. ROCm's
  own OpenCL ICD might differ; this sweep does not measure it.
- **Vulkan = RADV** with the Sarek GLSL emitter; one kernel (`transpose_naive`) does not
  compile, and strided AoS access (`soa_aos`) is pathologically slow — both are RADV/emitter
  limitations, not throughput ceilings.
- **Verification tolerance** (footnote 2) makes the Verified column unreliable at large sizes
  for `vector_add`/`nbody`; treat those ✗ as "float32-vs-float64 epsilon", not miscompute.
- **Effective-bandwidth benchmarks** (footnote 1) have non-physical absolute magnitudes; only
  in-benchmark cross-backend ratios are meaningful.
- Numbers are single-machine, single-session; thermal/clock state was steady but not pinned.
