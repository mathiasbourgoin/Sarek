--- 
layout: page
title: Getting Started with Sarek
---

# Getting Started with Sarek

Sarek is a high-performance framework for GPGPU programming in OCaml. It allows you to write kernels directly in OCaml syntax and execute them on various backends including CUDA, OpenCL, Vulkan, and Metal.

## Installation

### Prerequisites

- **OCaml**: 5.4.0+ (required for effects and domains support)
- **Dune**: 3.20+
- **GPU Drivers** (optional):
  - **CUDA**: NVIDIA drivers + CUDA Toolkit
  - **OpenCL**: OpenCL runtime (Intel NEO, ROCm, etc.)
  - **Vulkan**: Vulkan SDK + glslangValidator
  - **Metal**: macOS 10.13+ (included with Xcode)

### Installing via Opam

Sarek is available via Opam. To install the core package and specific backends:

```bash
# Core packages
opam install sarek spoc

# Install specific GPU backends (optional)
opam install sarek-cuda      # For NVIDIA GPUs
opam install sarek-opencl    # For OpenCL devices
opam install sarek-vulkan    # For Vulkan support
opam install sarek-metal     # For Apple Silicon/Macs
```

## Your First Kernel: Vector Addition

Here is a complete example of a vector addition kernel. Sarek uses the `[%kernel ...]` syntax to define code that runs on the GPU.

```ocaml
open Sarek

(* 1. Define the kernel *)
let%kernel vector_add (a : float32 vector) (b : float32 vector) (c : float32 vector) =
  (* Get the global thread ID *)
  let idx = get_global_id 0 in
  
  (* Perform computation if within bounds *)
  (* Note: Arrays use unsafe access syntax inside kernels for performance *)
  c.(idx) <- a.(idx) + b.(idx)

let () =
  (* 2. Initialize input data *)
  let n = 1024 in
  let a = Vector.create Float32 n in
  let b = Vector.create Float32 n in
  let c = Vector.create Float32 n in
  
  (* Fill vectors with data *)
  for i = 0 to n - 1 do
    Vector.set a i (float_of_int i);
    Vector.set b i (float_of_int (i * 2));
  done;

  (* 3. Select a device (auto-detects available GPU) *)
  let device = Device.get_default () in
  Printf.printf "Using device: %s\n" (Device.name device);

  (* 4. Execute the kernel *)
  (* Grid: 4 blocks, Block: 256 threads -> 1024 total threads *)
  Execute.run vector_add 
    ~device 
    ~grid:(4, 1, 1) 
    ~block:(256, 1, 1) 
    [Vec a; Vec b; Vec c];

  (* 5. Check results *)
  let result = Vector.get c 10 in
  Printf.printf "c[10] = %f\n" result
```

## Shared Memory & Synchronization

Sarek supports advanced GPU features like shared memory and barriers. Here is an example of a parallel reduction (summing a vector).

```ocaml
let%kernel reduce_sum (input : float32 vector) (output : float32 vector) (n : int32) =
  (* Allocate shared memory for the thread block *)
  let%shared sdata = Array.create Float32 256 in
  
  let tid = thread_idx_x in
  let gid = get_global_id 0 in
  
  (* Load data into shared memory *)
  sdata.(tid) <- if gid < n then input.(gid) else 0.0;
  
  (* Synchronize all threads in the block *)
  barrier ();

  (* Tree reduction in shared memory *)
  let stride = ref 128 in
  while !stride > 0 do
    if tid < !stride then
      sdata.(tid) <- sdata.(tid) +. sdata.(tid + !stride);
    barrier ();
    stride := !stride / 2
  done;

  (* Write the block result to global memory *)
  if tid = 0 then
    output.(block_idx_x) <- sdata.(0)
```

## Compilation

Build your project with `dune`:

```lisp
(executable
 (name my_program)
 (libraries sarek spoc)
 (preprocess (pps sarek.ppx)))
```

Run it:

```bash
dune exec ./my_program.exe
```

## Next Steps

- **[Examples](../examples/)** - Learn through practical examples (vector add, matrix multiply, reduction, transpose, mandelbrot)
- **[Concepts](concepts.html)** - Understand Sarek's design and programming model
- **[Benchmarks](../benchmarks/)** - See performance data across different GPUs and backends
- **[Backends](backends.html)** - Learn about CUDA, OpenCL, Vulkan, and Metal support
- **[API Documentation](../spoc_docs/index.html)** - Complete API reference
