# Sarek - GPU Computing for OCaml

**SIMT Abstraction for Runtime Extensible Kernels**

[![Build Status](https://github.com/mathiasbourgoin/SPOC/actions/workflows/ci.yml/badge.svg)](https://github.com/mathiasbourgoin/SPOC/actions)

Sarek is a PPX-based DSL that lets you write GPU kernels directly in OCaml syntax. Kernels compile to multiple backends (CUDA, OpenCL, Vulkan, Metal) without code changes.

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

The framework is actively maintained and uses modern OCaml features while preserving compatibility with existing SPOC code.

**Note**: This recent rework was completed with assistance from AI agents. Feedback, bug reports, and contributions are welcome via [GitHub Issues](https://github.com/mathiasbourgoin/SPOC/issues).

## Features

### GPU Kernel Development

Write GPU kernels in OCaml syntax using the `[%kernel ...]` PPX extension:

```ocaml
let%kernel vector_add (a : float32 vector) (b : float32 vector) (c : float32 vector) =
  let idx = get_global_id 0 in
  c.(idx) <- a.(idx) + b.(idx)
```

Kernels compile to multiple backends automatically without code changes.

### Backend Support

| Backend | Target | Status | Documentation |
|---------|--------|--------|---------------|
| **CUDA** | NVIDIA GPUs | ✓ | [sarek-cuda/](sarek-cuda/) |
| **OpenCL** | Multi-vendor GPUs/CPUs | ✓ | [sarek-opencl/](sarek-opencl/) |
| **Vulkan** | Cross-platform GPUs | ✓ | [sarek-vulkan/](sarek-vulkan/) |
| **Metal** | Apple Silicon/Intel Macs | ✓ | [sarek-metal/](sarek-metal/) |
| **Native** | CPU (parallel) | ✓ | [sarek/plugins/native/](sarek/plugins/native/) |
| **Interpreter** | CPU (debugging) | ✓ | [sarek/plugins/interpreter/](sarek/plugins/interpreter/) |

### Core Features

- **Type Safety**: GADTs and phantom types for compile-time guarantees
- **Zero-Copy**: Efficient memory sharing between host and device
- **Automatic Selection**: Runtime backend selection based on available hardware
- **Intrinsics**: Extensive library of GPU intrinsics (math, atomics, barriers)
- **Custom Types**: Support for records and variants in kernels
- **Debug Logging**: Controlled via `SAREK_DEBUG` environment variable

### Framework Architecture

```
spoc/              Low-level SDK and plugin interface
├── framework/     Plugin registration and backend interface
├── ir/            Intermediate representation (IR)
└── registry/      Intrinsic function registry

sarek/             Runtime and PPX compiler
├── core/          Device abstraction and memory management
├── framework/     Framework integration
├── ppx/           Sarek PPX compiler
├── sarek/         Unified execution dispatcher
└── plugins/       Native and Interpreter backends

GPU Backends:
├── sarek-cuda/    NVIDIA CUDA backend
├── sarek-opencl/  OpenCL backend (multi-vendor)
├── sarek-vulkan/  Vulkan/GLSL backend
└── sarek-metal/   Apple Metal backend
```

## Installation

### Prerequisites

- OCaml 5.4.0+ (local opam switch included in repository)
- dune 3.20+
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

### Installing via OPAM

SPOC is not yet published to the OPAM repository, but you can use OPAM to install from source with all dependencies:

```bash
# Clone repository
git clone https://github.com/mathiasbourgoin/SPOC.git
cd SPOC

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

## Usage

### Basic Example

```ocaml
open Sarek

(* Define a kernel *)
let%kernel saxpy (a : float32 vector) (x : float32 vector) 
                  (y : float32 vector) (alpha : float) =
  let i = get_global_id 0 in
  y.(i) <- alpha *. x.(i) +. a.(i)

let () =
  (* Initialize framework *)
  let device = Device.get_device 0 in
  
  (* Create vectors *)
  let n = 1024 in
  let a = Vector.create Float32 n in
  let x = Vector.create Float32 n in
  let y = Vector.create Float32 n in
  
  (* Execute kernel *)
  saxpy ~grid:(n/256, 1, 1) ~block:(256, 1, 1) a x y 2.5
```

### Backend Selection

```ocaml
(* List available devices *)
let devices = Device.list_devices () in
List.iter (fun dev ->
  Printf.printf "%s (%s)\n" 
    (Device.name dev) 
    (Device.backend_name dev)
) devices

(* Select specific backend *)
let cuda_device = Device.get_device_by_backend "CUDA" in
let opencl_device = Device.get_device_by_backend "OpenCL" in
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

- [ARCHITECTURE.md](ARCHITECTURE.md) - System architecture and design
- [CONTRIBUTING.md](CONTRIBUTING.md) - Contribution guidelines
- [PROJECT_STATUS.md](PROJECT_STATUS.md) - Current project status
- [Backend Documentation](sarek-cuda/) - Individual backend READMEs

For API documentation, see inline comments and README files in each package directory.

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

- **GitHub Pages**: [http://mathiasbourgoin.github.io/SPOC/](http://mathiasbourgoin.github.io/SPOC/)
- **GitHub Actions**: [Build status and CI](https://github.com/mathiasbourgoin/SPOC/actions)
- **Issues**: [Bug reports and feature requests](https://github.com/mathiasbourgoin/SPOC/issues)
