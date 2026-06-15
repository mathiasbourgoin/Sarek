# Sarek Standard Library

This directory contains the Sarek standard library modules that define types and intrinsic functions for GPU kernel programming.

## Architecture Overview

The Sarek stdlib follows a **ppx_deriving-style architecture** where:

1. **Types and intrinsics are defined only in stdlib modules** using `%sarek_intrinsic`
2. **Stdlib auto-registers** intrinsics into compile-time and runtime registries at module load
3. **The kernel PPX queries registries** instead of hardcoding intrinsics
4. **Device code is resolved at JIT time** via `IntrinsicRef` pattern

```
┌─────────────────────┐     ┌──────────────────────┐
│  Stdlib Modules     │     │  Sarek PPX           │
│  (Float32, Gpu...)  │     │  (kernel compiler)   │
│                     │     │                      │
│  %sarek_intrinsic   │────▶│  Sarek_ppx_registry  │
│  definitions        │     │  (compile-time)      │
└─────────────────────┘     └──────────────────────┘
          │                           │
          │                           │
          ▼                           ▼
┌─────────────────────┐     ┌──────────────────────┐
│  Sarek_registry     │     │  Kirc / Gen.ml       │
│  (runtime)          │◀────│  (code generation)   │
│                     │     │                      │
│  Device code lookup │     │  IntrinsicRef →      │
│  at JIT time        │     │  device code string  │
└─────────────────────┘     └──────────────────────┘
```

## Module Structure

- **`Float32.ml`** - 32-bit float type and math functions (sin, cos, sqrt, etc.)
- **`Int32.ml`** - 32-bit integer type and operations
- **`Int64.ml`** - 64-bit integer type and operations
- **`Gpu.ml`** - GPU thread/block intrinsics (thread_idx_x, block_barrier, etc.)
- **`Sarek_stdlib.ml`** - Re-exports all modules and forces initialization

**Note:** Float64 is NOT part of stdlib - it's in a separate library (`sarek_float64`)
because not all devices support double precision. Link `sarek_float64` only when needed.

## Defining Intrinsics

Use `%sarek_intrinsic` to define types or functions:

### Defining a Type

```ocaml
let%sarek_intrinsic float32 =
  {device = (fun _ -> "float"); ctype = Ctypes.float}
```

The record contains:
- `device`: `device -> string` - returns the C type name for the target device
- `ctype`: Ctypes type for FFI marshalling

### Defining a Function

```ocaml
let%sarek_intrinsic (sin : float32 -> float32) =
  {device = dev "sinf" "sin"; ocaml = Stdlib.sin}
```

The record contains:
- `device`: `device -> string` - returns device code (function name or format string)
- `ocaml`: OCaml implementation for host-side execution

### Device Code Patterns

The `device` field can return:

1. **Function name** (e.g., `"sinf"`) - generates `sinf(arg)`
2. **Format string** (e.g., `"(%s + %s)"`) - arguments are substituted via sprintf

Helper for CUDA/OpenCL differences:
```ocaml
let dev cuda opencl d = Sarek_registry.cuda_or_opencl d cuda opencl

(* CUDA: sinf, OpenCL: sin *)
let%sarek_intrinsic (sin : float32 -> float32) =
  {device = dev "sinf" "sin"; ocaml = Stdlib.sin}

(* Same for both *)
let%sarek_intrinsic (add : float32 -> float32 -> float32) =
  {device = dev "(%s + %s)" "(%s + %s)"; ocaml = ( +. )}
```

### Defining Constants

```ocaml
let%sarek_intrinsic (thread_idx_x : int32) =
  {device = dev "threadIdx.x" "get_local_id(0)"}
```

## Creating a New Stdlib Module

1. **Create the module file** (e.g., `MyLib.ml`):

```ocaml
(* MyLib.ml - Custom intrinsics for Sarek *)

(* Helper for CUDA/OpenCL *)
let dev cuda opencl d = Sarek.Sarek_registry.cuda_or_opencl d cuda opencl

(* Define a custom function *)
let%sarek_intrinsic (my_func : float32 -> float32) =
  {device = dev "my_cuda_func" "my_opencl_func"; ocaml = fun x -> x *. 2.0}

(* Define a custom constant *)
let%sarek_intrinsic (my_const : int32) =
  {device = dev "MY_CUDA_CONST" "MY_OPENCL_CONST"}
```

2. **Add to dune** in your library:

```dune
(library
 (name my_sarek_lib)
 (libraries sarek sarek_ppx_intrinsic)
 (preprocess (pps sarek_ppx_intrinsic)))
```

3. **Export from a wrapper module** and force initialization:

```ocaml
(* My_sarek_stdlib.ml *)
module MyLib = MyLib

let () = ignore MyLib.my_func  (* Force initialization *)
```

4. **Use in kernels**:

```ocaml
let my_kernel = [%kernel fun (v : float32 vector) (n : int32) ->
  let open MyLib in
  let i = thread_idx_x in
  if i < n then v.(i) <- my_func v.(i)
]
```

## Module Scoping

When you write `let open Float32 in sqrt x`, the `open_module` function brings all `Float32.*` intrinsics into scope under short names.

Legacy aliases are supported:
- `Std` → `Gpu` (backward compatibility)
- `Math` → `Float32`

Auto-opened modules (available without explicit open):
- `Gpu` - thread/block intrinsics
- `Float32` - basic math functions

## How It's Wired

### Compile Time

1. `%sarek_intrinsic` PPX processes definitions in stdlib modules
2. Generates registration code that calls `Sarek_ppx_registry.register_*`
3. When stdlib is linked, registration runs at module load time
4. Kernel PPX queries `Sarek_ppx_registry` for type checking

### Runtime (JIT)

1. `%sarek_intrinsic` also generates `Sarek_registry.register_*` calls
2. When kernel is compiled (`Kirc.gen`), code generator encounters `IntrinsicRef`
3. `Gen.ml` looks up device code via `Sarek_registry.fun_device_code`
4. Device code string is emitted (with sprintf for format strings)

### Key Files

- `Sarek_ppx_intrinsic/Sarek_ppx_intrinsic.ml` - PPX that processes `%sarek_intrinsic`
- `Sarek_ppx/lib/Sarek_ppx_registry.ml` - Compile-time registry for type checking
- `Sarek/Sarek_registry.ml` - Runtime registry for device code lookup
- `Sarek/Gen.ml` - Code generator that resolves `IntrinsicRef`
- `Sarek_ppx/lib/Sarek_env.ml` - Typing environment with `open_module`

## Testing

Unit tests are in `Sarek_test/unit/`:
- `test_env.ml` - Tests intrinsic lookup and module opening
- `test_typer.ml` - Tests type checking with intrinsics

E2E tests in `Sarek_test/e2e/`:
- `test_vector_add.ml` - Basic kernel execution test

Run tests:
```bash
dune runtest
```
