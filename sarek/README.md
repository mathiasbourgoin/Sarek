# Sarek - OCaml DSL for GPU Computing

Sarek is a Domain-Specific Language (DSL) for writing GPU kernels in OCaml. It provides a PPX syntax extension (`[%kernel ...]`) that allows writing GPU code using OCaml syntax, with automatic compilation to multiple backend targets.

## What is Sarek?

Sarek sits on top of the SPOC framework and provides:

- **PPX Compiler**: Transforms OCaml syntax into GPU code
- **Type System**: Compile-time type checking for GPU kernels
- **Runtime**: Execution dispatcher and device management
- **Standard Library**: Math functions and GPU intrinsics
- **Multi-Backend**: Single source compiles to CUDA, OpenCL, Vulkan, Metal, CPU

## Directory Structure

### Core Components

- **[ppx/](ppx/)** - Sarek PPX Compiler
  - Parses `[%kernel ...]` syntax
  - Type checking and inference
  - IR generation and lowering
  - Code quotation

- **[core/](core/)** - Runtime Core
  - Device abstraction (`Device.ml`)
  - Vector management (`Vector.ml`)
  - Memory operations (`Memory.ml`, `Transfer.ml`)
  - Kernel arguments (`Kernel_arg.ml`)

- **[sarek/](sarek/)** - Execution Layer
  - Unified execution dispatcher (`Execute.ml`)
  - Kernel registry (`Sarek_registry.ml`)
  - CPU runtime (`Sarek_cpu_runtime.ml`)
  - IR interpreter (`Sarek_ir_interp.ml`)

- **[framework/](framework/)** - Framework Integration
  - Backend registry (`Framework_registry.ml`)
  - Kernel cache (`Framework_cache.ml`)
  - Error handling (`Framework_error.ml`)

### Libraries

- **[Sarek_stdlib/](Sarek_stdlib/)** - GPU Standard Library
  - Math functions (sin, cos, exp, sqrt, etc.)
  - Atomic operations
  - Barrier/synchronization primitives
  - Float32 operations

- **[Sarek_float64/](Sarek_float64/)** - Double Precision Support
  - Float64 type and operations
  - Backend-specific implementations
  - Requires GPU double precision support

- **[ppx_intrinsic/](ppx_intrinsic/)** - Intrinsic Definition PPX
  - `%sarek_intrinsic` extension
  - Dual code generation (GPU + host)
  - Backend extensibility

### CPU Backends

- **[plugins/native/](plugins/native/)** - Native CPU Backend
  - Direct execution of pre-compiled functions
  - OCaml 5 domains for parallelism
  - Zero compilation overhead

- **[plugins/interpreter/](plugins/interpreter/)** - Interpreter Backend
  - IR interpretation at runtime
  - Debugging and validation
  - No GPU required

### Tests

- **[tests/unit/](tests/unit/)** - Unit Tests
  - PPX compiler tests
  - Type inference tests
  - IR tests

- **[tests/e2e/](tests/e2e/)** - End-to-End Tests
  - Vector operations, matrix multiplication
  - Reduction, scan, sort algorithms
  - Complex types and intrinsics
  - Run with `make benchmarks` or `make benchmarks-fast`

## Quick Start

### Writing Your First Kernel

```ocaml
open Sarek
module Device = Spoc_core.Device
module Vector = Spoc_core.Vector

(* Define a kernel *)
let vector_add =
  [%kernel
    fun (a : float32 vector) (b : float32 vector) (c : float32 vector) (n : int32) ->
      let open Sarek_stdlib.Std in
      let tid = global_thread_id in
      if tid < n then c.(tid) <- a.(tid) + b.(tid)]

(* Use it *)
let () =
  let devs = Device.init ~frameworks:["CUDA"; "OpenCL"; "Native"; "Interpreter"] () in
  let dev = devs.(0) in
  let n = 1024 in
  let a = Vector.create Vector.float32 n in
  let b = Vector.create Vector.float32 n in
  let c = Vector.create Vector.float32 n in

  (* Initialize a and b... *)

  (* Get IR and execute on GPU *)
  let _, kirc = vector_add in
  let ir = match kirc.Sarek.Kirc_types.body_ir with
    | Some ir -> ir | None -> failwith "No IR" in
  let block = Execute.dims1d 256 in
  let grid  = Execute.dims1d ((n + 255) / 256) in
  Execute.run_vectors ~device:dev ~ir ~args:[Vec a; Vec b; Vec c; Int n]
    ~block ~grid ()
```

### Available Intrinsics

Sarek provides GPU intrinsics that work across all backends:

```ocaml
(* Thread indexing *)
get_global_id 0      (* Global thread index *)
get_local_id 0       (* Local thread index within block *)
get_group_id 0       (* Block/workgroup index *)
get_global_size 0    (* Total number of threads *)
get_local_size 0     (* Block size *)

(* Synchronization *)
barrier ()           (* Block-level barrier *)
memfence ()          (* Memory fence *)

(* Atomic operations *)
atomic_add arr idx value
atomic_sub arr idx value
atomic_cas arr idx expected desired

(* Math functions *)
sin x, cos x, tan x
exp x, log x, sqrt x, pow x y
```

See [Sarek_stdlib/README.md](Sarek_stdlib/README.md) for the complete intrinsic reference.

## Compilation Pipeline

```
OCaml Source with [%kernel ...]
          ↓
    Sarek PPX (ppx/)
          ↓
  Typed IR (Sarek_ir_types)
          ↓
    Backend Selection
          ↓
  ┌───────┴────────┬───────┬─────────┐
  ↓                ↓       ↓         ↓
CUDA C          OpenCL   GLSL     Native
NVRTC           Compiler  SPIR-V  Direct
  ↓                ↓       ↓         ↓
Kernel Execution on Device
```

## Backend Support

Kernels compile to multiple backends automatically:

| Backend | Package | Target | Documentation |
|---------|---------|--------|---------------|
| CUDA | sarek-cuda | NVIDIA GPUs | [sarek-cuda/README.md](../sarek-cuda/README.md) |
| OpenCL | sarek-opencl | Multi-vendor | [sarek-opencl/README.md](../sarek-opencl/README.md) |
| Vulkan | sarek-vulkan | Cross-platform | [sarek-vulkan/README.md](../sarek-vulkan/README.md) |
| Metal | sarek-metal | Apple devices | [sarek-metal/README.md](../sarek-metal/README.md) |
| Native | Built-in | CPU parallel | [plugins/native/README.md](plugins/native/README.md) |
| Interpreter | Built-in | CPU debug | [plugins/interpreter/README.md](plugins/interpreter/README.md) |

## Type System

Sarek enforces GPU-safe type constraints at compile time:

### Supported Types

- **Scalars**: `int32`, `int64`, `float` (float32), `bool`
- **Vectors**: `float32 vector`, `int32 vector` (global memory)
- **Arrays**: `float32 array`, `int32 array` (shared memory)
- **Records**: Custom record types (registered via PPX)
- **Variants**: Simple variants (registered via PPX)

### Type Inference

The PPX performs full type inference with unification:

```ocaml
let inferred =
  [%kernel
    fun (v : float32 vector) ->
      let open Sarek_stdlib.Std in
      let idx = global_thread_id in  (* inferred as int32 *)
      let x = v.(idx) in             (* inferred as float *)
      let y = sin x in               (* inferred as float *)
      v.(idx) <- y +. 1.0]           (* type checked *)
```

Type errors are reported at compile time with helpful messages.

## Testing

```bash
# Run all unit tests
dune runtest

# Run fast benchmarks (CI-friendly, ~20s)
make benchmarks-fast

# Run full benchmark suite
make benchmarks

# Run with specific backend
dune exec sarek/tests/e2e/test_vector_add.exe -- --cuda
dune exec sarek/tests/e2e/test_vector_add.exe -- --native
```

## Coverage

Current test coverage: ~48% (unit tests)

Generate coverage locally:
```bash
./scripts/coverage-unit.sh
# Report at: _coverage/unit-report/index.html
```

## Documentation

- **[ARCHITECTURE.md](../ARCHITECTURE.md)** - System architecture
- **[PROJECT_STATUS.md](../PROJECT_STATUS.md)** - Project status
- **[CONTRIBUTING.md](../CONTRIBUTING.md)** - Contribution guidelines
- Component READMEs in each subdirectory

## Examples

See [tests/e2e/](tests/e2e/) for working examples:

- **test_vector_add.ml** - Basic vector operations
- **test_matrix_mul.ml** - Matrix multiplication (naive + tiled)
- **test_reduce.ml** - Parallel reduction patterns
- **test_histogram.ml** - Atomic operations, shared memory
- **test_mandelbrot.ml** - Complex algorithms
- **test_polymorphism.ml** - Polymorphic kernels

## Getting Help

- Check component-specific READMEs for detailed information
- See [../CONTRIBUTING.md](../CONTRIBUTING.md) for contribution guidelines
- Report issues at [GitHub Issues](https://github.com/mathiasbourgoin/SPOC/issues)
