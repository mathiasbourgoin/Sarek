# sarek-metal - Metal Backend Plugin for SPOC/Sarek

**Package**: `sarek-metal`  
**Library**: `sarek-metal.plugin`

The Metal backend plugin enables SPOC/Sarek to compile and execute GPU kernels on Apple Silicon and Intel Macs using Metal Shading Language (MSL) and Metal compute pipelines.

## Overview

The Metal backend translates Sarek IR (Intermediate Representation) into Metal C source code (MSL), compiles it at runtime using Metal's JIT compiler, and executes kernels on Metal-compatible GPUs.

### Key Features

- **Pure OCaml**: Uses `ctypes-foreign` for FFI with Metal framework
- **JIT Compilation**: Runtime compilation with Metal compiler
- **Apple Platforms**: macOS 10.13+, iOS 11+, iPadOS

### Supported Devices

- **Apple Silicon**: M1, M2, M3, M4 series
- **Intel Macs**: Discrete and integrated GPUs with Metal support
- **iOS/iPadOS**: A11+ chips with Metal 2+

## Architecture

### Execution Flow

1. **User writes kernel** using `[%kernel ...]` syntax
2. **Sarek PPX** compiles to typed IR
3. **Backend generates** Metal C source code
4. **Metal compiler** builds compute pipeline
5. **Runtime executes** kernel on selected device

### Module Organization

```
sarek-metal/
├── Metal_error.ml          # Structured error handling (14 lines)
├── Metal_types.ml          # Metal type definitions
├── Metal_bindings.ml       # ctypes FFI bindings
├── Metal_api.ml            # High-level API wrappers
├── Metal_plugin_base.ml    # Base plugin implementation
├── Metal_plugin.ml         # Plugin registration
├── Sarek_ir_metal.ml       # IR → Metal C codegen (2,758 lines total)
└── test/
    ├── test_metal_error.ml      # Error tests (6)
    └── test_sarek_ir_metal.ml   # Codegen tests (14)
```

## Error Handling

All errors use structured error types from `Backend_error`:

```ocaml
(* Codegen errors *)
Metal_error.unknown_intrinsic "my_func"
Metal_error.invalid_arg_count "atomic_add" 2 3
Metal_error.unsupported_construct "EArrayCreate" "reason"

(* Runtime errors *)
Metal_error.device_not_found 5 3
Metal_error.backend_unavailable "No Metal device found"
Metal_error.no_device_selected "kernel_launch"

(* Plugin errors *)
Metal_error.library_not_found "Metal" []
Metal_error.feature_not_supported "raw pointer arguments"
```

## Metal C Specifics

### Thread Indexing

| Sarek Intrinsic | Metal C Equivalent | Description |
|-----------------|-------------------|-------------|
| `thread_idx_x()` | `__metal_tid.x` | Local thread index in threadgroup (X) |
| `block_idx_x()` | `__metal_tpg.x` | Threadgroup index (X) |
| `global_idx_x()` | `__metal_gid.x` | Global thread index (X) |
| `grid_dim_x()` | `__metal_num_groups.x` | Number of threadgroups (X) |

### Synchronization

| Sarek Intrinsic | Metal C Equivalent | Description |
|-----------------|-------------------|-------------|
| `barrier()` | `threadgroup_barrier(mem_flags::mem_threadgroup)` | Threadgroup barrier |
| `warp_barrier()` | `sub_group_threadgroup_barrier(...)` | Subgroup barrier |
| `memfence()` | `threadgroup_barrier(mem_flags::mem_device)` | Device memory fence |

### Atomic Operations

Metal uses C++14 atomic operations:

| Sarek Intrinsic | Metal C Equivalent | Memory Order |
|-----------------|-------------------|--------------|
| `atomic_add` | `atomic_fetch_add_explicit(..., memory_order_relaxed)` | Relaxed |
| `atomic_sub` | `atomic_sub(...)` | Relaxed |
| `atomic_min` | `atomic_min(...)` | Relaxed |
| `atomic_max` | `atomic_max(...)` | Relaxed |

### Memory Spaces

| Sarek Memory Space | Metal Qualifier |
|--------------------|-----------------|
| `Global` | `device` |
| `Shared` | `threadgroup` |
| `Local` | (register) |

### Type Mapping

| Sarek IR Type | Metal C Type | Notes |
|---------------|--------------|-------|
| `TInt32` | `int` | 32-bit signed |
| `TInt64` | `long` | 64-bit signed |
| `TFloat32` | `float` | 32-bit floating point |
| `TFloat64` | `float` | **No double precision support** |
| `TBool` | `bool` | Boolean (0/1) |

**Important**: Metal does not support double precision (`float64`/`double`) on most devices. The backend maps `TFloat64` to `float`.

## Usage Examples

### Example 1: Vector Addition

```ocaml
open Spoc_core

let%kernel vector_add (a : float32 vec) (b : float32 vec) (c : float32 vec) (n : int32) =
  let idx = global_idx_x () in
  if idx < n then
    c.{idx} <- a.{idx} +. b.{idx}

let n = 1024 in
let a = Vector.create Float32 n in
let b = Vector.create Float32 n in
let c = Vector.create Float32 n in

(* Initialize data *)
for i = 0 to n - 1 do
  Vector.set a i (float i);
  Vector.set b i (float i *. 2.0)
done

(* Select Metal device *)
let dev = Device.get_by_name "Metal" in
Device.select dev;

(* Execute kernel *)
Execute.run vector_add
  ~grid:{x=4; y=1; z=1}
  ~block:{x=256; y=1; z=1}
  [VecArg a; VecArg b; VecArg c; IntArg (Int32.of_int n)]
```

### Example 2: Matrix Transpose with Shared Memory

```ocaml
let%kernel transpose (input : float32 vec) (output : float32 vec) (width : int32) =
  (* Shared memory tile *)
  let tile = shared_array_create float32 256 in
  
  let idx_x = global_idx_x () in
  let idx_y = global_idx_y () in
  let local_x = thread_idx_x () in
  let local_y = thread_idx_y () in
  
  (* Load into shared memory *)
  if idx_x < width && idx_y < width then begin
    tile.{local_y * 16 + local_x} <- input.{idx_y * width + idx_x};
    barrier ()  (* Synchronize threadgroup *)
  end;
  
  (* Write transposed *)
  let out_x = block_idx_y () * 16 + local_x in
  let out_y = block_idx_x () * 16 + local_y in
  if out_x < width && out_y < width then
    output.{out_y * width + out_x} <- tile.{local_x * 16 + local_y}
```

### Example 3: Histogram with Atomics

```ocaml
let%kernel histogram (data : int32 vec) (bins : int32 vec) (n : int32) (num_bins : int32) =
  let idx = global_idx_x () in
  if idx < n then begin
    let value = data.{idx} in
    let bin = value mod num_bins in
    if bin >= 0 && bin < num_bins then
      ignore (atomic_add bins.{bin} 1l)  (* Atomic increment *)
  end
```

## Installation

### Requirements

**Runtime**:
- macOS 10.13+ (High Sierra) or iOS 11+
- Metal-compatible GPU

**Build**:
- Xcode command line tools
- OCaml 5.4.0+

### macOS

```bash
# Install Xcode command line tools
xcode-select --install

# Metal framework is included with macOS
# No additional packages needed
```

### Verify Installation

```bash
# Check Metal support
system_profiler SPDisplaysDataType | grep -i metal
```

## Testing

The Metal backend includes unit tests:

```bash
# Run all tests
dune test sarek-metal/test

# Run specific test suite
_build/default/sarek-metal/test/test_metal_error.exe
_build/default/sarek-metal/test/test_sarek_ir_metal.exe
```

### Test Coverage

**Error Tests** (6 tests):
- Codegen, runtime, and plugin errors
- Exception handling
- Error prefix formatting
- Utility functions

**Codegen Tests** (14 tests):
- Literals, operations, statements
- Barrier and thread intrinsics
- Atomic operations
- Type mapping and helper functions

## Design Principles

### Type Safety

Uses GADTs and phantom types for typed memory and kernel arguments.

### Structured Errors

Uses shared `Backend_error` module for consistent error handling.

### Code Organization

Code generation uses extracted helper functions for maintainability.

### Limitations

- **No double precision**: Metal doesn't support `float64` on most devices
- **No device-to-device transfers**: Use staging buffer
- **No raw pointers**: Metal doesn't support pointer arguments in kernels
- **macOS/iOS only**: Metal is Apple-exclusive technology

## Troubleshooting

### "Metal framework not found"

Ensure Xcode command line tools are installed:
```bash
xcode-select --install
```

### "No Metal device found"

Check if your Mac supports Metal:
```bash
system_profiler SPDisplaysDataType | grep -i metal
```

Older Macs (pre-2012) may not support Metal.

### "Library 'libobjc' not found"

This should not happen on macOS as `libobjc` is part of the system. If it occurs, reinstall Xcode command line tools.

### Float64 Precision Issues

Metal does not support double precision. If you need `float64`, use the CUDA or OpenCL backend instead.

## License

See main SPOC repository for license information.

## Contributing

When contributing to the Metal backend:

1. Run all tests: `dune test sarek-metal/test`
2. Follow code quality guidelines (see AGENTS.md)
3. No `failwith` - use `Metal_error.raise_error`
4. No pretentious language in docs or commits

## Related Documentation

- **SPOC Framework**: Main SPOC README
- **Sarek PPX**: `sarek/README.md`
- **Backend Error**: `spoc/framework/Backend_error.ml`
- **CUDA Backend**: `sarek-cuda/README.md`
- **OpenCL Backend**: `sarek-opencl/README.md`
- **Vulkan Backend**: `sarek-vulkan/README.md`
