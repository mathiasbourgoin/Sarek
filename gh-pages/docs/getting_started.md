--- 
layout: page
title: Getting Started with Sarek
---

# Getting Started with Sarek

Sarek is a high-performance framework for GPGPU programming in OCaml. It allows you to write kernels directly in OCaml syntax and execute them on various backends including CUDA, OpenCL, Vulkan, and Metal.

## Installation

### Prerequisites

- **OCaml**: 5.4.0+ (required for effects and domains support)
- **Dune**: 3.15+
- **GPU Drivers** (optional):
  - **CUDA**: NVIDIA drivers + CUDA Toolkit
  - **OpenCL**: OpenCL runtime (Intel NEO, ROCm, etc.)
  - **Vulkan**: Vulkan SDK + glslangValidator
  - **Metal**: macOS 10.13+ (included with Xcode)

### Installing from Source

Sarek is not yet in the official opam repository. Install from source:

```bash
# Clone the repository
git clone https://github.com/mathiasbourgoin/Sarek.git
cd Sarek

# Install dependencies
opam install . --deps-only -y

# Build
dune build

# Optional: install locally in your opam switch
opam install .
```

GPU backends (CUDA, OpenCL, Vulkan, Metal) are automatically detected and enabled based on available drivers and SDKs on your system.

## Your First Kernel: Vector Addition

Here is a complete example of a vector addition kernel. Sarek uses the `[%kernel ...]` syntax to define code that runs on the GPU.

```ocaml
open Sarek
module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer

(* 1. Define the kernel *)
let vector_add =
  [%kernel
    fun (a : float32 vector) (b : float32 vector) (c : float32 vector) (n : int32) ->
      let open Sarek_stdlib.Std in
      let tid = global_thread_id in
      if tid < n then c.(tid) <- a.(tid) + b.(tid)]

let () =
  (* 2. Initialize input data *)
  let n = 1024 in
  let a = Vector.create Vector.float32 n in
  let b = Vector.create Vector.float32 n in
  let c = Vector.create Vector.float32 n in

  (* Fill vectors with data *)
  for i = 0 to n - 1 do
    Vector.set a i (float_of_int i);
    Vector.set b i (float_of_int (i * 2));
  done;

  (* 3. Select a device (auto-detects best available GPU/CPU) *)
  let devs = Device.init ~frameworks:["CUDA"; "OpenCL"; "Native"; "Interpreter"] () in
  let dev = Device.best () in
  Printf.printf "Using device: %s\n" dev.Device.name;
  ignore devs;

  (* 4. Get IR and execute the kernel *)
  let _, kirc = vector_add in
  let ir = match kirc.Sarek.Kirc_types.body_ir with
    | Some ir -> ir | None -> failwith "No IR" in
  let block = Execute.dims1d 256 in
  let grid  = Execute.dims1d ((n + 255) / 256) in
  Execute.run_vectors ~device:dev ~ir ~args:[Vec a; Vec b; Vec c; Int n]
    ~block ~grid ();
  Transfer.flush dev;

  (* 5. Check results *)
  let result = Vector.get c 10 in
  Printf.printf "c[10] = %f\n" result
```

## Shared Memory & Synchronization

Sarek supports advanced GPU features like shared memory and barriers. Here is an example of a parallel reduction (summing a vector).

```ocaml
let reduce_sum =
  [%kernel
    fun (input : float32 vector) (output : float32 vector) (n : int32) ->
      let open Sarek_stdlib.Std in
      (* Allocate shared memory for the thread block *)
      let%shared (sdata : float32) = 256l in

      let tid = thread_idx_x in
      let gid = global_thread_id in

      (* Load data into shared memory *)
      sdata.(tid) <- if gid < n then input.(gid) else 0.0;

      (* Synchronize all threads in the block *)
      barrier ();

      (* Tree reduction in shared memory *)
      let stride = ref 128l in
      while !stride > 0l do
        if tid < !stride then
          sdata.(tid) <- sdata.(tid) +. sdata.(tid + !stride);
        barrier ();
        stride := !stride / 2l
      done;

      (* Write the block result to global memory *)
      if tid = 0l then
        output.(block_idx_x) <- sdata.(0l)]
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
- **[Backends](backends.html)** - Learn about CUDA, OpenCL, Vulkan, Metal, and WebGPU/WGSL support
- **[API Documentation](../spoc_docs/index.html)** - Complete API reference
- **[Try it in the browser](../playground.html)** - Live kernel transpiler (no install needed)
- **[Interactive Learn course](../learn/)** - Run kernels on your GPU straight from the page
