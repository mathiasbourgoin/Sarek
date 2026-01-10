---
layout: index_sample
title: Sarek Backends
---

# Supported Backends

Sarek is designed to be backend-agnostic. You write your kernel once in OCaml, and Sarek compiles it to the appropriate shading language (CUDA C, OpenCL C, GLSL, MSL) or executes it natively on the CPU.

| Backend | Target | Status | Requirements |
|---------|--------|--------|--------------|
| **CUDA** | NVIDIA GPUs | Stable | NVIDIA Driver + CUDA Toolkit |
| **OpenCL** | Multi-vendor (AMD, Intel, NVIDIA) | Stable | OpenCL Runtime / ICD |
| **Vulkan** | Cross-platform (Linux, Windows, Android) | Stable | Vulkan SDK + glslangValidator |
| **Metal** | Apple Silicon & Intel Macs | Stable | macOS 10.13+ |
| **Native** | CPU (Multicore) | Stable | OCaml 5.4.0+ |
| **Interpreter**| CPU (Debug) | Stable | None |

## CUDA Backend (`sarek-cuda`)

Targets NVIDIA GPUs using the CUDA Driver API and NVRTC (Runtime Compilation).

- **Features**: Shared memory, atomics, warp intrinsics, dynamic parallelism.
- **Performance**: Native CUDA performance; uses JIT compilation to optimize for the specific GPU architecture.
- **Setup**: Requires `libcuda.so` and `libnvrtc.so` in your library path.

```bash
opam install sarek-cuda
```

## OpenCL Backend (`sarek-opencl`)

Targets a wide range of devices including AMD GPUs, Intel integrated graphics, and FPGAs.

- **Features**: Work-group barriers, local memory, math intrinsics.
- **Compatibility**: Tested on NVIDIA, AMD, and Intel platforms.
- **Setup**: Requires an OpenCL ICD loader (e.g., `ocl-icd-libopencl1` on Linux).

```bash
opam install sarek-opencl
```

## Vulkan Backend (`sarek-vulkan`)

Uses GLSL compute shaders and SPIR-V for modern cross-platform GPU support.

- **Pipeline**: Generates GLSL -> Compiles to SPIR-V (via `glslang` or `shaderc`) -> Executes on Vulkan.
- **Features**: Push constants, SSBOs (Storage Buffers), specialization constants.
- **Platform**: Ideal for modern Linux desktops, Android devices, and Windows.

```bash
opam install sarek-vulkan
```

## Metal Backend (`sarek-metal`)

Native support for Apple hardware (M1/M2/M3 chips) using Metal Shading Language (MSL).

- **Features**: Threatgroup memory, SIMD-group functions.
- **Limitations**: No double precision (`float64`) support on most Apple hardware.
- **Platform**: macOS and iOS only.

```bash
opam install sarek-metal
```

## Native CPU Backend (`sarek.native`)

Executes kernels directly on the host CPU without GPU compilation.

- **Mechanism**: Uses OCaml 5 Domains for parallel execution.
- **Use Case**: High-performance fallback when no GPU is available, or for debugging logic with standard debuggers.

## Interpreter Backend (`sarek.interpreter`)

Walks through the Sarek IR (Intermediate Representation) step-by-step.

- **Use Case**: Deep debugging of kernel logic, verification of IR transformations, and educational purposes.
- **Performance**: Slow (interpreted), but provides full visibility into execution.
