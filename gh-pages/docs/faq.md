---
layout: page
title: FAQ
---

# Frequently Asked Questions

## General Questions

### Do I need an NVIDIA GPU to use Sarek?
No. While Sarek supports **CUDA** for NVIDIA hardware, it also supports **OpenCL** and **Vulkan**, which work on AMD GPUs, Intel Integrated Graphics, and even some FPGAs. If no GPU is available, Sarek can fall back to the **Native CPU** backend.

### Can I run Sarek on macOS?
Yes. Sarek has a dedicated **Metal** backend for Apple Silicon (M1/M2/M3) and Intel-based Macs. It also supports OpenCL on macOS.

### Does Sarek work on Windows?
Sarek has limited testing on Windows, but it should work via the OpenCL and Vulkan backends. Using **WSL2** is recommended for the best experience.

## Technical Questions

### What is the overhead of using OCaml for GPU programming?
Sarek uses a JIT compilation approach for GPU backends. The logic you write in OCaml is compiled into native GPU code (CUDA C, GLSL, etc.). Once the kernel is compiled and cached, the execution performance is comparable to hand-written C/CUDA code.

### Does the OCaml Garbage Collector interfere with the GPU?
No. The GPU has its own memory. Data is explicitly transferred between the OCaml heap and the GPU memory via `Vector` objects. The GC manages the OCaml "handle" to the GPU memory, but does not touch the data residing on the device.

### Can I use custom OCaml types in my kernels?
Yes. By using the `[@@sarek.type]` attribute on record definitions, Sarek automatically generates the corresponding C structures for the GPU and handles the memory layout.

### Is OCaml 5 required?
Yes. The latest version of Sarek leverages **OCaml 5 Effects and Domains** for the Native CPU backend to provide high-performance parallel execution on multi-core processors.

## CUDA Backend Questions

### What CUDA version do I need for newer GPUs?
For **Blackwell architecture** GPUs (RTX 5000 series, compute capability 12.0):
- **Minimum**: CUDA 12.9 with driver 575+
- **Recommended**: CUDA 13.1 with driver 580+

Sarek automatically handles forward compatibility by compiling PTX for `compute_90` and letting the driver JIT-compile for newer architectures.

### I get `CUDA_ERROR_UNKNOWN(222)` when loading kernels. What's wrong?
This error typically occurs with newer GPU architectures when there's a version mismatch:

1. **Check your versions**:
   ```bash
   nvidia-smi        # Shows driver version and API level
   nvcc --version    # Shows CUDA toolkit version
   ```

2. **Common causes**:
   - CUDA 13.1 with driver < 580 (downgrade to CUDA 12.9)
   - CUDA toolkit older than 12.9 on Blackwell GPUs (upgrade toolkit)

3. **Verify your setup**:
   ```bash
   dune exec -- sarek-device-info
   ```

### What does "CUDA Version" in nvidia-smi mean?
The "CUDA Version" shown by `nvidia-smi` (e.g., "12.9") is the maximum CUDA runtime API version your **driver** supports, not your installed toolkit version. It's normal and expected for these to differ. For example, driver 575 supports CUDA API 12.9, even if you have CUDA toolkit 13.1 installed (though it won't work fully without driver 580+).

### How do I list available CUDA devices?
Use the device info utility:
```bash
dune exec -- sarek-device-info
```

This shows all detected devices across all backends (CUDA, OpenCL, Vulkan, Metal, Native, Interpreter) with their capabilities.

## Troubleshooting

### A kernel crashes with "the context is lost" / "soft recovery" (amdgpu)

```
amdgpu: The CS has cancelled because the context is lost. This context is
guilty of a soft recovery.
```

This is a **GPU-level fault** reported by the driver, not a Sarek error — the OpenCL
runtime submitted work the driver couldn't complete, so it reset the GPU. On integrated
AMD GPUs (e.g. Renoir APUs) driven by Mesa's legacy **clover** OpenCL, this happens even
for tiny workloads.

What to do, in order:

1. **Confirm it's the driver, not your kernel** — run on the CPU backend:
   ```bash
   dune exec sarek/tests/e2e/test_vector_add.exe -- --native
   ```
   If that passes, your kernel is fine and the OpenCL stack is the problem.
2. **Switch OpenCL implementation** from clover to the maintained **rusticl**:
   ```bash
   RUSTICL_ENABLE=radeonsi dune exec -- your_program.exe
   ```
3. **Use the Vulkan backend instead** (more robust than clover on APUs):
   ```bash
   dune exec sarek/tests/e2e/test_vector_add.exe -- --vulkan
   ```
4. Or disable the failing backend entirely — see
   [Device & Backend Selection](device_selection.html) (`SPOC_DISABLE_OPENCL=1`).

Note: ROCm does not support most integrated APUs, so installing it will not fix this.

### "No devices available" / a backend doesn't appear

- Run `dune exec -- sarek-device-info` to see what is actually detected.
- Check the backend's loader is installed (OpenCL ICD, Vulkan loader, CUDA driver).
- Make sure you haven't left a `SPOC_DISABLE_*` variable set (see
  [Device & Backend Selection](device_selection.html)).

### A specific backend hangs or faults

Isolate it with the disable variables, then run only the others:

```bash
SPOC_DISABLE_OPENCL=1 dune exec -- your_program.exe   # skip OpenCL
SPOC_DISABLE_GPU=1     dune exec -- your_program.exe   # CPU only
```

### Results differ slightly between backends

Floating-point results can differ across backends (different rounding, fused
multiply-add, math-library implementations). The e2e tests compare against a CPU baseline
with a tolerance rather than exact equality. If a backend's result is wildly wrong (not
just last-bit differences), file an issue with the device, driver version, and kernel.
