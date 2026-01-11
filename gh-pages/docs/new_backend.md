---
layout: page
title: Building a New Backend
---

# Building a New Backend Plugin

Sarek is designed to be extensible. Adding a new target (e.g., WebGPU, HIP, or a custom FPGA backend) involves implementing the standard `BACKEND` signature.

## 1. Choosing an Execution Model

Backends in Sarek follow one of two primary execution models:

- **JIT (Just-In-Time)**: The backend generates source code (CUDA C, OpenCL C, GLSL) from Sarek IR at runtime, compiles it using a GPU driver, and launches the binary.
- **Direct**: The backend executes pre-compiled OCaml functions. This is how the **Native CPU** backend works, using OCaml 5 Domains for parallelism without any runtime compilation.

Your implementation must specify this in the `execution_model` field of the signature.

## 2. Implementing Intrinsics

**Intrinsics** are the bridge between Sarek's high-level OCaml syntax and backend-specific hardware features (like thread IDs, barriers, or specialized math functions).

### The Intrinsic Registry
Every backend provides an `INTRINSIC_REGISTRY`. This is where you map Sarek names to your backend's implementation.

For a **JIT backend**, an intrinsic implementation is usually a code-generation function:
```ocaml
(* Mapping Sarek's 'barrier()' to OpenCL's 'barrier(CLK_LOCAL_MEM_FENCE)' *)
Registry.register "barrier" (fun () -> "barrier(CLK_LOCAL_MEM_FENCE);")
```

For a **Direct backend** (Native), it is an actual OCaml function:
```ocaml
(* Mapping thread_idx_x to a function that reads the current domain state *)
Registry.register "thread_idx_x" (fun state -> state.tid_x)
```

## 3. Handling Native Code Blocks

Sarek allows users to embed backend-specific code directly using `[%native.backend "..."]`. 

If you are building a backend named `mygpu`, you must ensure your code generator looks for `SNative ("mygpu", code)` nodes in the IR. This allows power users to bypass the compiler and use hand-optimized assembly or specialized hardware features.

## 4. The `BACKEND` Signature Modules

Your plugin must implement several core modules:

- **Device**: Logic to enumerate hardware and query `capabilities` (max threads, shared memory size, FP64 support).
- **Memory**: Functions to allocate device buffers and transfer data. You should support:
    - `host_to_device` / `device_to_host`
    - `alloc_zero_copy`: (Optional) for devices that share memory with the CPU.
- **Kernel**: Handles the lifecycle of a kernel:
    - `compile`: Takes a source string and produces a runnable handle.
    - `launch`: Marshals `exec_arg` values into the hardware's expected format.

## 5. Plugin Registration

Register your backend with a `priority`. SPOC uses this to automatically select the best available device (e.g., CUDA is usually priority 100, while Native CPU is 50).

```ocaml
let () =
  Spoc_framework.Framework_registry.register_backend
    ~priority:80 
    (module MyNewBackend : Spoc_framework.Framework_sig.BACKEND)
```

## 6. Testing Your Backend

1. **IR Tests**: Ensure your translator produces valid source code for all Sarek IR nodes (loops, matches, record assignments).
2. **Intrinsic Tests**: Verify that all standard Sarek intrinsics are correctly mapped.
3. **E2E Tests**: Run the suite in `sarek/tests/e2e`. A backend is considered "stable" if it passes the `test_vector_add`, `test_reduce`, and `test_matmul` benchmarks.