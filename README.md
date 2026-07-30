# Sarek - GPU Computing for OCaml

**SIMT Abstraction for Runtime Extensible Kernels**

[![Build Status](https://github.com/mathiasbourgoin/Sarek/actions/workflows/ci.yml/badge.svg)](https://github.com/mathiasbourgoin/Sarek/actions)

Sarek is a PPX-based DSL that lets you write GPU kernels directly in OCaml syntax. Kernels compile to multiple backends (CUDA, HIP, OpenCL, Vulkan, Metal) without code changes, plus WGSL for the browser and two CPU backends.

## What is Sarek?

**Sarek** is the user-facing DSL and compiler. Write kernels in OCaml with `[%kernel ...]`, and Sarek compiles them to GPU code at build time.

**SPOC** (SIMT Programming for OCaml) is the underlying runtime providing device abstraction, plugin architecture, and backend infrastructure.

## Recent Development

This codebase has undergone significant modernization (2024-2026):

- **OCaml 5.4 support** with effect handlers and domains
- **Code quality improvements** across all GPU backends
- **Structured error handling** replacing untyped exceptions
- **Plugin-based architecture** for extensible backend support
- **Test coverage** with unit and end-to-end tests
- **Documentation** for all major components
- **WGSL/WebGPU codegen** — a transpiler target emitting WGSL for browser-side execution
- **In-browser Playground** — live kernel transpiler at [mathiasbourgoin.github.io/Sarek/playground.html](https://mathiasbourgoin.github.io/Sarek/playground.html)
- **Interactive Learn course** — edit and run Sarek kernels on your GPU in the browser at [mathiasbourgoin.github.io/Sarek/learn/](https://mathiasbourgoin.github.io/Sarek/learn/)
- **PTX direct emitter** — `Sarek_ir_ptx` emits NVIDIA PTX directly from Sarek IR, bypassing NVRTC. It is the **default** device path of the CUDA backend (`Cuda_ptx_plugin`); the NVRTC/C path remains available as `Cuda_c_plugin`
- **HIP backend** (`sarek-hip`) — native ROCm/hiprtc backend for AMD GPUs, not going through ZLUDA. Measured against OpenCL and Vulkan on an RX 7900 XTX in [docs/benchmarks/hip-vs-opencl-vulkan-2026-07-24.md](docs/benchmarks/hip-vs-opencl-vulkan-2026-07-24.md)
- **float16 element type** — `float16` vectors and kernel values, with compute-in-f32 semantics and explicit narrowing. Codegen exists on the CUDA-C / HIP path only (both go through `Sarek_ir_cuda`), and the Interpreter also runs `f16`: it lists `Float16` in `Sarek_interp_capability.device_features` and rounds through IEEE binary16 in `Sarek_float16`. The remaining targets refuse it — but *not* all for the same kind of reason, and the difference matters:
  - **refused by measurement** — OpenCL (rusticl/radeonsi) and GLSL/Vulkan (RADV's ACO): the driver absorbs the f32→f16 narrowing into the arithmetic feeding it, so a measured fraction of binary16 inputs disagrees with the interpreter — 620/63488 on OpenCL, 2912/63488 on GLSL — and the available remedy does not work (no affordable barrier on the OpenCL path; `precise` does not prevent it on GLSL). Devices, counts and reproducers in [docs/fp-contraction-policy.md](docs/fp-contraction-policy.md)
  - **refused by an emitter-internal invariant** — PTX: `Sarek_ir_ptx_types` derives a value's register class from the register *name* prefix (`%f`/`%fd`/`%rd`), so adding a `%h` class requires auditing every such guard first
  - **not implemented yet** — Metal and WGSL: both arms raise `unsupported_construct` and say so ("not yet supported (#57 slice 2)"); these are deferrals, not measured refusals
  - Native does not list `Float16` among its device features at all; it carries f16 vectors as a storage type only

  See [docs/design/f16-dsl-element-type.md](docs/design/f16-dsl-element-type.md) and the `TFloat16` arms of each generator
- **Structure-of-Arrays device layout** — a record-typed vector parameter can be lowered as SoA (one coalesced device array per scalar leaf) instead of packed AoS; CUDA/PTX only, via `Spoc_core.Soa_vector` + `Sarek.Soa_launch.run_soa`. Design and measurements in [docs/optimization/tier1b-emitter-soa-handoff.md](docs/optimization/tier1b-emitter-soa-handoff.md)
- **In-tree kernel libraries**, all written in pure Sarek rather than per-backend:
  `Sarek_gemm` (shared-memory tiled SGEMM; every backend with a block-shared-memory
  model — i.e. all but the sequential Interpreter), `Sarek_worklist` (dynamic
  parallelism over an atomic work queue, serving the use cases CUDA CDP targets
  without any device-side launch mechanism), `Sarek_df64` / `Sarek_real64`
  (double-float software extended precision for devices with no usable `float64`)
- **Machine-checked formal models** — three Rocq projects under `formal/` (`convergence-safety`, `type-safety`, `codegen-ptx`), rebuilt from scratch and `coqchk`-verified by a dedicated CI job. Counts are not hand-maintained: `scripts/check-formal-proofs.sh` regenerates each `proof-ledger.json` and fails on drift
- **Executable codegen gates in CI** — the CI image ships CUDA 12.6 `ptxas`/`nvdisasm`/NVRTC, `glslangValidator`, `naga` and OpenCL-capable `clang`, and `ci/assert-toolchain.sh` fails the job if any of them is missing, so a self-skipping gate can no longer report success

The framework is actively maintained and uses modern OCaml features while preserving compatibility with existing SPOC code.

**Note**: This recent rework was completed with assistance from AI agents. Feedback, bug reports, and contributions are welcome via [GitHub Issues](https://github.com/mathiasbourgoin/Sarek/issues).

## Features

### GPU Kernel Development

Write GPU kernels in OCaml syntax using the `[%kernel ...]` PPX extension:

```ocaml
let vector_add =
  [%kernel
    fun (a : float32 vector) (b : float32 vector) (c : float32 vector) (n : int32) ->
      let open Sarek_stdlib.Std in
      let tid = global_thread_id in
      if tid < n then c.(tid) <- a.(tid) + b.(tid)]
```

Kernels compile to multiple backends automatically without code changes.

### Backend Support

Devices that appear in `Device.init` / `sarek-device-info`:

| Backend | Framework name | Target | Status | Documentation |
|---------|----------------|--------|--------|---------------|
| **CUDA/PTX** | `CUDA/PTX` | NVIDIA GPUs (and AMD via ZLUDA) | ✓ default CUDA path | [sarek/codegen/Sarek\_ir\_ptx.ml](sarek/codegen/Sarek_ir_ptx.ml) |
| **CUDA/C** | `CUDA/C` | NVIDIA GPUs, via NVRTC | ✓ | [sarek-cuda/](sarek-cuda/) |
| **HIP** | `HIP` | AMD GPUs, via ROCm/hiprtc | ✓ | [sarek-hip/Hip\_plugin.ml](sarek-hip/Hip_plugin.ml) — no package README yet; measurements in [docs/benchmarks/](docs/benchmarks/hip-vs-opencl-vulkan-2026-07-24.md) |
| **OpenCL** | `OpenCL` | Multi-vendor GPUs/CPUs | ✓ | [sarek-opencl/](sarek-opencl/) |
| **Vulkan** | `Vulkan` | Cross-platform GPUs, GLSL→SPIR-V | ✓ | [sarek-vulkan/](sarek-vulkan/) |
| **Metal** | `Metal` | Apple Silicon/Intel Macs | ✓ | [sarek-metal/](sarek-metal/) |
| **Native** | `Native` | CPU (parallel, OCaml 5 domains) | ✓ | [sarek/plugins/native/](sarek/plugins/native/) |
| **Interpreter** | `Interpreter` | CPU (debugging) | ✓ | [sarek/plugins/interpreter/](sarek/plugins/interpreter/) |

Not a device — a code-generation target only:

| Target | What it produces | Status | Documentation |
|--------|------------------|--------|---------------|
| **WGSL** | WGSL compute shaders (`@compute @workgroup_size(...)`) for a JavaScript WebGPU host; validated in CI with `naga` | ✓ codegen; **no OCaml-side runtime** — `sarek/plugins/webgpu` is an inert stub whose `is_available ()` returns `false`, so WGSL never appears in device enumeration | [sarek/codegen/Sarek\_ir\_wgsl.ml](sarek/codegen/Sarek_ir_wgsl.ml) |

The CUDA family registers as two frameworks (`CUDA/PTX` and `CUDA/C`); use
`Device.filter_cuda ()` to match both.

> **The published backend page is stale against the two tables above, and is
> deliberately not linked from them.** `gh-pages/docs/backends.md` — which is
> what [mathiasbourgoin.github.io/Sarek/](https://mathiasbourgoin.github.io/Sarek/)
> serves — still has a single `CUDA` row with no `HIP` row and no
> `CUDA/PTX`-vs-`CUDA/C` split, and its "CUDA Backend" section says the backend
> targets NVIDIA GPUs "using the CUDA Driver API and NVRTC", which is the
> `CUDA/C` path, not the PTX path that is now the default. Its *WGSL* section
> (WGSL is a transpiler target, not a device plugin) does agree with the table
> above. Bringing the published page in line is a website change with its own
> audience and its own row-by-row verification, so it is filed as follow-up work
> rather than folded in here — but it is named here so the divergence is not
> silent, and so this table is not read as endorsing that page.

### Core Features

- **Type Safety**: GADTs and phantom types for compile-time guarantees
- **Zero-Copy**: Efficient memory sharing between host and device
- **Automatic Selection**: Runtime backend selection based on available hardware
- **Intrinsics**: Extensive library of GPU intrinsics (math, atomics, barriers). The PTX emitter lowers 83 intrinsic names natively, enumerated from its own dispatch registry (`Sarek_ir_ptx_expr.intrinsic_registry`)
- **Custom Types**: Support for records and variants in kernels, laid out with the aligned C-ABI rule shared by the host PPX, the C-family backends and the PTX emitter
- **Numeric widths**: `int32`, `int64`, `float32`, `float64`; `float16` on the CUDA-C/HIP path and in the Interpreter (refused elsewhere — see the float16 bullet above for which refusals are measured and which are unimplemented); `Sarek_df64` / `Sarek_real64` for software extended precision where a device has no usable `float64`
- **Capability refusals over silent fallback**: a kernel using a width or feature a target cannot honour is refused with a stated reason and provenance rather than quietly downgraded — see [docs/design/capability-model.md](docs/design/capability-model.md)
- **Debug Logging**: Controlled via `SAREK_DEBUG` environment variable

### Framework Architecture

```
spoc/              Low-level SDK and plugin interface
├── framework/     Plugin registration and backend interface
├── ir/            Intermediate representation (IR)
└── registry/      Intrinsic function registry

sarek/             Runtime and PPX compiler
├── core/          Device abstraction and memory management (incl. Soa_vector)
├── core_base/     Vector representation and host storage
├── codegen/       All code generators: PTX, CUDA C, OpenCL C, GLSL, MSL, WGSL
├── execute/       Launch pipeline (Execute, Soa_launch)
├── framework/     Framework integration
├── interp/        IR interpreter
├── ppx/           Sarek PPX compiler
├── sarek/         Unified execution dispatcher
├── transpile/     Standalone / in-browser transpiler
├── plugins/       Native and Interpreter backends (+ inert WebGPU stub)
└── Sarek_*/       Kernel libraries: stdlib, gemm, worklist, df64, real64,
                   float64, geometry, tuple_vec

GPU Backends:
├── sarek-cuda/    NVIDIA CUDA backend (CUDA/PTX default, CUDA/C via NVRTC)
├── sarek-hip/     AMD ROCm/HIP backend
├── sarek-opencl/  OpenCL backend (multi-vendor)
├── sarek-vulkan/  Vulkan/GLSL backend
└── sarek-metal/   Apple Metal backend

Formal models (Rocq):
├── formal/convergence-safety/   Barrier-safety analysis
├── formal/type-safety/          Sarek PPX type system
└── formal/codegen-ptx/          PTX emission + aggregate byte layout
```

## The PTX Direct Emitter (`Sarek_ir_ptx`)

> **Assembled by `ptxas` in CI on every run; GPU *execution* in CI is still absent.**
> The emitter is no longer an experiment sitting beside the CUDA backend — it *is*
> the CUDA backend's default device path. What CI cannot do is run a kernel: no
> runner has a GPU, so every device e2e test skips there and device validation
> remains manual (RX 7900 XTX under ZLUDA, GTX 1070).

`Sarek_ir_ptx` emits NVIDIA PTX directly from Sarek IR, bypassing NVRTC entirely. It is the default device path of the CUDA backend (`Cuda_ptx_plugin`); the NVRTC/C path remains available as `Cuda_c_plugin`.

**What works:**
- Scalar and vector kernels (int32/int64, float32/float64), global loads/stores, barriers
- Records and variants with the aligned C-ABI aggregate layout (proven in `formal/codegen-ptx/theories/PtxLayout.v`), match expressions, static tag erasure
- Helper functions via `EApp` inlining (`sarek.inline` budget-controlled)
- Static and dynamic shared memory (`.shared`, module-scope `extern .shared`), per-thread `.local` arrays
- Atomics: add/min/max/and/or/xor/exch, CAS, wrapping inc/dec, 64-bit add/exch
- float64 softmath library (trig, exp/log family, hypot, fma, …) with an interpreter oracle
- Parameterised SM target (`?sm_target`, default `sm_86`); `Cuda_api.Kernel.load_from_ptx` adapts `.target` to the device's actual SM (tested: GTX 1070, sm_61; AMD RX 7900 XTX via ZLUDA)

**How the generated PTX is validated:**

- **`ptxas` assembly, on the whole intrinsic surface.** `sarek/tests/unit/test_ptx_intrinsic_sweep.ml`
  builds one kernel per registered intrinsic name and width — 83 names today —
  enumerated from the emitter's *own* dispatch registry
  (`Sarek_ir_ptx_expr.intrinsic_registry`), not from a hand-picked list, and
  assembles each with `ptxas`. Adding an intrinsic to the emitter without adding
  a sweep recipe fails `test_every_name_has_a_recipe`, so the gate cannot drift
  behind the emitter. `sarek/tests/unit/test_ptx_snapshot.ml` assembles a
  smaller set of regression and SoA kernels on top of its PTX-text assertions.
- **The gate cannot self-skip green in CI.** Both gates skip cleanly when `ptxas`
  is absent, which is right on a developer machine and was a disaster in CI:
  the earlier image had no CUDA at all, so the gates printed SKIP and the job
  passed having validated nothing. `ci/Dockerfile` now installs CUDA 12.6
  `ptxas`/`nvdisasm`/NVRTC + headers, and `ci/assert-toolchain.sh` runs before
  any test step and **fails** if `ptxas` is missing or cannot assemble a probe
  module — it runs each tool rather than `command -v`-ing it, because a
  present-but-broken binary skips just as silently as an absent one.
- **`nvdisasm` SASS inspection** for the f16 f32-discipline gate
  (`sarek-cuda/test/test_cuda_f16_sass.ml`), likewise asserted present.

**Known gaps:**
- **Proof-to-production link is uneven.** The *aggregate byte layout* half is
  extracted: `formal/codegen-ptx/extraction/LayoutExtract.v` extracts
  `theories/PtxLayout.v` to OCaml, and the conformance suite runs the theory's
  own definitions against `Sarek_ir_layout`. The *expression/statement/kernel*
  half is not: `formal/codegen-ptx/test/test_codegen_ptx_conformance.ml` still
  checks a hand-written OCaml mirror of the Rocq definitions, and the five `.ml`
  files `Extract.v` produces are committed but linked by nothing. On that half
  the *production* emitter is not compared to the model at all — the mirror is
  compared to the mirror's own spec. The only cases in that suite that invoke
  production code are the 7 in the `ptx-dshared` group, which call the real
  `Sarek_codegen.Sarek_ir_ptx_kernel.generate` and inspect its PTX text against
  the shared-memory acceptance criteria of `specs/ptx-dshared-formal.md` — a
  string/exception check on the emitter's output, still not a check against the
  Rocq model. So: extraction links theory to production on the layout half; on
  the expr/stmt/kernel half neither extraction nor conformance reaches the
  production emitter.
- **No GPU execution in CI.** See the note above; device validation is manual.
- **Warp primitives are emitted but not reachable from kernel source.**
  `SWarpBarrier` and `SMemFence` are lowered by all six generators —
  `bar.warp.sync` on PTX, `__syncwarp()` on CUDA, `sub_group_barrier` on OpenCL,
  `simdgroup_barrier` on Metal, `subgroupBarrier()` on GLSL and WGSL, each pinned
  by `sarek/tests/unit/test_sync_stmt_emission.ml`. What is missing is the front
  end: no PPX surface syntax constructs either statement, so today they are
  reachable only from hand-built IR. The front end does declare four
  synchronising intrinsic *names* — `block_barrier`, `warp_barrier`,
  `memory_fence_block`, `memory_fence_device`
  (`Sarek_lower_ir.synchronising_intrinsics`) — but lowering leaves all four as
  `EIntrinsic` expression calls and never builds `SWarpBarrier` or `SMemFence`
  from them, which is why those two statements have no surface spelling. Warp
  **shuffle**/vote/ballot are a separate, larger gap — not modelled in the IR
  and emitted by no backend.
- **`f16` is refused by this emitter.** `Sarek_ir_ptx_types` has no `TFloat16`
  register type; f16 codegen exists on the CUDA-C/HIP path only.

**Intended purpose:** foundation for formal verification of the CUDA backend, developed in `formal/codegen-ptx/` alongside `specs/ptx-records-variants.md` and `specs/ptx-dshared-formal.md`.

## Installation

### Prerequisites

- OCaml 5.4.0+ (local opam switch included in repository)
- dune 3.15+
- GPU backends (optional):
  - **CUDA**: NVIDIA driver + CUDA toolkit (see CUDA requirements below)
  - **OpenCL**: OpenCL implementation for your device
  - **Vulkan**: Vulkan SDK + glslangValidator or Shaderc
  - **Metal**: macOS 10.13+ (included with Xcode)

The Native (CPU parallel) and Interpreter (CPU sequential) backends work without any GPU drivers.

#### CUDA Requirements

For NVIDIA GPUs, especially newer architectures:

- **CUDA Toolkit**: 12.9 or later recommended
- **Driver Version**: 
  - CUDA 12.9 requires driver 575+
  - CUDA 13.1+ requires driver 580+
- **Blackwell GPUs** (RTX 5000 series, compute capability 12.0):
  - Minimum: CUDA 12.9 + driver 575
  - Recommended: CUDA 13.1 + driver 580+

**Note**: The "CUDA Version" shown by `nvidia-smi` indicates the maximum CUDA runtime API version your driver supports. This may differ from your installed CUDA toolkit version, which is normal. For example, driver 575 with CUDA toolkit 12.9 will show "CUDA Version: 12.9" in `nvidia-smi`.

#### AMD GPUs

Two paths, both real:

- **`sarek-hip`** — the native ROCm path (hiprtc → gfx code object →
  `hipModuleLaunchKernel`), no ZLUDA involved. It reuses the CUDA-C generator
  verbatim, because HIP C++ is source-compatible with the CUDA-C subset Sarek
  emits. Measured against OpenCL and Vulkan on an RX 7900 XTX in
  [docs/benchmarks/hip-vs-opencl-vulkan-2026-07-24.md](docs/benchmarks/hip-vs-opencl-vulkan-2026-07-24.md).
- **CUDA/PTX under ZLUDA** — described below; this is the path the PTX emitter's
  device evidence comes from.

##### CUDA/PTX via ZLUDA

The CUDA backend also runs on AMD GPUs through [ZLUDA](https://github.com/vosen/ZLUDA), a CUDA implementation on top of ROCm. ZLUDA ships the CUDA driver API but not NVRTC, so only the CUDA/PTX backend (the default) is available. Records, variants, match expressions and shared memory are all supported by the PTX emitter, so typical Sarek kernels run unmodified; only kernels that explicitly select the CUDA/C (NVRTC) backend will not run.

```bash
# Prerequisites: ROCm (tested with 7.2) and a supported AMD GPU (e.g. RDNA3)
# Download a ZLUDA release and point the dynamic loader at it:
LD_LIBRARY_PATH=/path/to/zluda dune exec -- sarek-device-info
# → AMD Radeon RX 7900 XTX [ZLUDA] (CUDA/PTX)
```

Tested on an RX 7900 XTX with ZLUDA v7-preview.3: the CUDA/PTX backend matches or exceeds the OpenCL and Vulkan backends on memory-bound benchmarks.

### Installing via OPAM

SPOC is not yet published to the OPAM repository, but you can use OPAM to install from source with all dependencies:

```bash
# Clone repository
git clone https://github.com/mathiasbourgoin/Sarek.git
cd Sarek

# Install dependencies via OPAM (OCaml 5.4+)
opam update
opam install . --deps-only --working-dir

# Build all backends
dune build

# Or build only specific backends you need
dune build sarek sarek-cuda
dune build sarek sarek-opencl
```

Backends detect compatible drivers at runtime. You can install backends even without corresponding GPU drivers - they will simply not be available for use.

### Building from Source

```bash
# Clone and use local opam switch
cd SPOC
opam install . --deps-only

# Build all packages
dune build

# Build specific backend
dune build sarek-cuda
dune build sarek-opencl
```

The framework uses dynamic linking, so you can build without GPU drivers installed. GPU support is detected at runtime.

### Verifying Installation

```bash
# List all available devices
dune exec -- sarek-device-info

# Run unit tests
dune runtest

# Run fast benchmarks (Native + OpenCL if available)
make benchmarks-fast

# Run full benchmark suite on all available devices
make benchmarks
```

The fast benchmarks use small problem sizes and complete in ~20 seconds, while the full benchmark suite exercises all backends with larger datasets.

**Benchmark Suite**: 22 benchmarks (`benchmarks/bench_*.ml`) covering compute-bound (matrix multiplication naive/tiled, Mandelbrot, n-body, conv2d), memory-bound (vector add/copy, stream triad, reduction, stencil, gather/scatter), irregular (histogram, bitonic and radix sort, scan), and layout/transfer optimisation patterns (transpose naive vs tiled, AoS vs SoA, pinned transfer). Results are published to an [interactive web viewer](https://mathiasbourgoin.github.io/Sarek/benchmarks/) with multiple visualization modes.

## Usage

### Basic Example

```ocaml
open Sarek
module Device = Spoc_core.Device
module Vector = Spoc_core.Vector

(* Define a kernel *)
let saxpy =
  [%kernel
    fun (a : float32 vector) (x : float32 vector)
        (y : float32 vector) (alpha : float32) (n : int32) ->
      let open Sarek_stdlib.Std in
      let i = global_thread_id in
      if i < n then y.(i) <- alpha *. x.(i) +. a.(i)]

let () =
  (* Initialize framework *)
  let devs = Device.init ~frameworks:["CUDA"; "OpenCL"; "Native"; "Interpreter"] () in
  let dev = devs.(0) in

  (* Get IR from kernel *)
  let _, kirc = saxpy in
  let ir = match kirc.Sarek.Kirc_types.body_ir with
    | Some ir -> ir | None -> failwith "No IR" in

  (* Create vectors *)
  let n = 1024 in
  let a = Vector.create Vector.float32 n in
  let x = Vector.create Vector.float32 n in
  let y = Vector.create Vector.float32 n in

  (* Execute kernel *)
  let block = Execute.dims1d 256 in
  let grid  = Execute.dims1d ((n + 255) / 256) in
  Execute.run_vectors ~device:dev ~ir ~args:[Vec a; Vec x; Vec y; Float 2.5; Int n]
    ~block ~grid ()
```

### Backend Selection

```ocaml
(* List available devices *)
let devices = Device.all () in
Array.iter (fun dev ->
  Printf.printf "%s (%s)\n"
    dev.Device.name
    dev.Device.framework
) devices

(* Select specific backend. CUDA backends register as "CUDA/PTX" and
   "CUDA/C"; use filter_cuda to match the whole family. *)
let cuda_devices = Device.filter_cuda () in
let opencl_devices = Device.by_framework "OpenCL" in
```

See [sarek/sarek/README.md](sarek/sarek/README.md) for comprehensive usage documentation.

## Testing

```bash
# Run all tests
dune runtest

# Run specific backend tests
dune test sarek-cuda
dune test sarek-opencl

# Run with specific backend
SAREK_BACKEND=cuda dune runtest
```

See [COVERAGE.md](COVERAGE.md) for coverage measurement instructions.

## Troubleshooting

### CUDA Issues

**Error: `CUDA_ERROR_UNKNOWN(222)` when loading PTX on new GPUs**

This error typically occurs on newer GPU architectures (e.g., Blackwell/RTX 5000 series) with mismatched CUDA versions:

- **Solution**: Ensure you have CUDA 12.9+ installed with driver 575+
- **Check versions**:
  ```bash
  nvidia-smi                    # Shows driver version and API level
  nvcc --version                # Shows installed CUDA toolkit version
  ```
- **Common cause**: CUDA 13.1 requires driver 580+. If you have driver 575, use CUDA 12.9 instead.

#### PTX compilation succeeds but module loading fails

Sarek automatically handles forward compatibility by compiling PTX for `compute_90` on compute capability 9.0+ devices. The CUDA driver then JIT-compiles for your actual hardware (e.g., sm_120 for RTX 5070 Ti). This requires:
- CUDA toolkit 12.9+ (for Blackwell GPU support)
- Compatible driver version (see requirements above)

#### Verifying CUDA setup

```bash
# Check if CUDA devices are detected
nvidia-smi

# Verify Sarek can find devices
dune exec -- sarek-device-info

# Check driver API compatibility
cat /proc/driver/nvidia/version
```

### OpenCL Issues

If OpenCL is not detecting your device, ensure you have the appropriate ICD (Installable Client Driver) installed:
- **NVIDIA**: Install NVIDIA driver with OpenCL support
- **AMD**: Install ROCm or AMDGPU-PRO driver
- **Intel**: Install Intel OpenCL runtime

## Documentation

- [GitHub Pages](http://mathiasbourgoin.github.io/Sarek/) - User guides, tutorials, and API docs
- [CONTRIBUTING.md](CONTRIBUTING.md) - Contribution guidelines
- [CHANGES.md](CHANGES.md) - Changelog
- [Backend Documentation](sarek-cuda/) - Individual backend READMEs

For API documentation, see inline comments and README files in each package directory.

### Contributing to Documentation

Documentation sources are in `gh-pages/` directory:
- User guides: `gh-pages/docs/*.md`
- Jekyll layouts: `gh-pages/_layouts/`
- API docs: Auto-generated from code comments via `odoc`

Changes merged to `main` branch automatically deploy to GitHub Pages via CI.

## Requirements

- **OCaml**: 5.4.0+ (uses domains, effects)
- **System**: 64-bit Linux, macOS, Windows (limited testing)
- **GPU**: Optional - Native and Interpreter backends work on any system

## Project History

This work originates from Mathias Bourgoin's PhD thesis at UPMC-LIP6 laboratory (Paris) and was partially funded by the [OpenGPU](http://opengpu.net/) project. Development continued at Verimag laboratory (Grenoble, 2014-2015) and LIFO laboratory (Orléans, 2015-2018).

Current maintainer: Mathias Bourgoin ([Nomadic Labs](https://nomadic-labs.com))

## License

See [LICENSE.md](LICENSE.md) for license information.

## Resources

- **GitHub Pages**: [http://mathiasbourgoin.github.io/Sarek/](http://mathiasbourgoin.github.io/Sarek/)
- **GitHub Actions**: [Build status and CI](https://github.com/mathiasbourgoin/Sarek/actions)
- **Issues**: [Bug reports and feature requests](https://github.com/mathiasbourgoin/Sarek/issues)
