# sarek-vulkan - Vulkan/GLSL Backend Plugin for SPOC/Sarek

**Package**: `sarek-vulkan`  
**Library**: `sarek-vulkan.plugin`

The Vulkan backend plugin enables SPOC/Sarek to compile and execute GPU kernels on Vulkan-compatible devices using GLSL compute shaders and SPIR-V. Uses ctypes-foreign for FFI bindings.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Core Modules](#core-modules)
- [Code Generation](#code-generation)
- [Error Handling](#error-handling)
- [API Reference](#api-reference)
- [Usage Examples](#usage-examples)
- [GLSL Intrinsics](#glsl-intrinsics)
- [Testing](#testing)
- [Installation](#installation)
- [Design Principles](#design-principles)

## Overview

The Vulkan backend is one of several GPU backends supported by SPOC/Sarek. It translates Sarek IR (Intermediate Representation) into GLSL compute shader code, compiles it to SPIR-V, and executes kernels on Vulkan devices.

```
┌─────────────────────────────────────────┐
│       User Code with [%kernel]          │
│         (OCaml syntax)                  │
├─────────────────────────────────────────┤
│          Sarek PPX Compiler             │
│     (Parse → Type → Lower → Quote)      │
├─────────────────────────────────────────┤
│         Sarek IR (Typed AST)            │
├─────────────────────────────────────────┤
│      Vulkan Backend (this package)      │  ← IR → GLSL translation
│    ┌─────────────────────────────┐     │
│    │  Sarek_ir_glsl.ml           │     │  - GLSL code generation
│    │  → GLSL compute shader      │     │  - Intrinsic mapping
│    └─────────────────────────────┘     │
├─────────────────────────────────────────┤
│       glslangValidator/Shaderc          │  ← SPIR-V compilation
│    ┌─────────────────────────────┐     │
│    │  Vulkan_api.ml              │     │  - Compile GLSL → SPIR-V
│    │  → SPIR-V binary            │     │  - Shader modules
│    └─────────────────────────────┘     │
├─────────────────────────────────────────┤
│         Vulkan Runtime                  │  ← Execution
│    ┌─────────────────────────────┐     │
│    │  Vulkan_api.ml              │     │  - Compute pipelines
│    │  → Device operations        │     │  - Memory management
│    └─────────────────────────────┘     │  - Command buffers
└─────────────────────────────────────────┘
```

### Key Features

- **Pure OCaml**: Uses `ctypes-foreign` for FFI
- **GLSL Shaders**: Generate compute shaders with layout qualifiers
- **SPIR-V**: Two compilation paths (glslangValidator or Shaderc library)
- **Multi-Vendor**: Works across vendors (NVIDIA, AMD, Intel, mobile)
- **Kernel Cache**: Global on-disk cache at `~/.cache/sarek/vulkan/`

### Supported Devices

The Vulkan backend works with any Vulkan-compatible device:
- **Desktop GPUs**: NVIDIA (GTX/RTX), AMD (Radeon), Intel Arc
- **Integrated GPUs**: Intel HD/Iris/Xe, AMD APU
- **Mobile**: Mali, Adreno (via Android/Linux)
- **Embedded**: Raspberry Pi 4+ (V3DV driver)

## Architecture

### Execution Flow

1. **User writes kernel** using `[%kernel ...]` syntax
2. **Sarek PPX** compiles to typed IR
3. **Backend generates** GLSL compute shader source
4. **Compiler produces** SPIR-V binary (glslangValidator or Shaderc)
5. **Vulkan creates** shader module and compute pipeline
6. **Runtime executes** kernel via command buffers

### Module Organization

```
sarek-vulkan/
├── Vulkan_error.ml         # Structured error handling (14 lines)
├── Vulkan_types.ml         # Vulkan type definitions (312 lines)
├── Vulkan_bindings.ml      # ctypes FFI bindings (376 lines)
├── Vulkan_api.ml           # High-level API wrappers (1,457 lines)
├── Vulkan_plugin_base.ml   # Base plugin implementation (250 lines)
├── Vulkan_plugin.ml        # Plugin registration (313 lines)
├── Sarek_ir_glsl.ml        # IR → GLSL codegen (1,090 lines)
├── Shaderc.ml              # Shaderc library bindings (268 lines)
└── test/
    ├── test_vulkan_error.ml      # Error tests (6)
    └── test_sarek_ir_glsl.ml     # Codegen tests (14)
```

### GLSL vs OpenCL/CUDA

Key differences from other backends:

| Feature | GLSL/Vulkan | OpenCL C | CUDA C |
|---------|-------------|----------|--------|
| **Syntax** | GLSL (stricter) | C99-like | C++-like |
| **Thread ID** | `gl_GlobalInvocationID.x` | `get_global_id(0)` | `threadIdx.x` |
| **Barriers** | `barrier()` | `barrier(...)` | `__syncthreads()` |
| **Shared Memory** | `shared` qualifier | `__local` | `__shared__` |
| **Atomics** | `atomicAdd` | `atomic_add` | `atomicAdd` |
| **Layout** | Required layout qualifiers | Not needed | Not needed |

## Core Modules

### Vulkan_error.ml

Structured error handling using the shared `Backend_error` functor:

```ocaml
(* Error types *)
type codegen_error   (* IR → GLSL translation errors *)
type runtime_error   (* Device/compilation/execution errors *)
type plugin_error    (* Library loading/feature support errors *)

(* Constructors *)
val unknown_intrinsic : string -> t
val invalid_arg_count : string -> int -> int -> t
val compilation_failed : string -> string -> t
val device_not_found : int -> int -> t
val library_not_found : string -> string list -> t

(* Utilities *)
val raise_error : t -> 'a
val to_string : t -> string
val with_default : default:'a -> (unit -> 'a) -> 'a
val to_result : (unit -> 'a) -> ('a, t) result
```

### Sarek_ir_glsl.ml

IR → GLSL code generation:

```ocaml
(* Generate GLSL compute shader from IR *)
val generate : 
  ?block:Framework_sig.dims ->
  ~types:(string * custom_type) list ->
  Sarek_ir_types.kernel -> string

(* Type conversion *)
val glsl_type_of_elttype : Sarek_ir_types.elttype -> string

(* Thread intrinsics *)
val glsl_thread_intrinsic : string -> string

(* Helper functions *)
val gen_var_decl : Buffer.t -> string -> string -> elttype -> expr -> unit
val gen_array_decl : Buffer.t -> string -> string -> elttype -> expr -> unit
val indent_nested : string -> string
```

### Vulkan_api.ml

High-level Vulkan operations:

```ocaml
module Device : sig
  type t
  val init : unit -> unit
  val get : int -> t
  val count : unit -> int
  val synchronize : t -> unit
end

module Memory : sig
  val malloc : Device.t -> int -> vk_buffer * vk_device_memory
  val host_to_device : src:('a, 'b, 'c) Bigarray.Array1.t -> dst:vk_buffer -> unit
  val device_to_host : src:vk_buffer -> dst:('a, 'b, 'c) Bigarray.Array1.t -> unit
end

module Kernel : sig
  type t
  val compile : Device.t -> name:string -> source:string -> t
  val compile_cached : Device.t -> name:string -> source:string -> t
  val launch : t -> args:args -> grid:dims -> block:dims -> shared_mem:int -> stream:Stream.t option -> unit
end
```

## Code Generation

### GLSL Compute Shader Structure

Generated shaders follow this structure:

```glsl
#version 450
#extension GL_EXT_shader_atomic_float : enable  // For atomicAdd(float)
#extension GL_ARB_gpu_shader_int64 : enable     // For int64_t

// Workgroup size from block dimensions
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// Buffer bindings (storage buffers)
layout(std430, binding = 0) buffer buf_input {
  float input_data[];
};

layout(std430, binding = 1) buffer buf_output {
  float output_data[];
};

// Push constants for scalar arguments
layout(push_constant) uniform PushConstants {
  int size;
  float factor;
} pc;

// Shared memory
shared float shared_data[256];

// Kernel function
void main() {
  int idx = int(gl_GlobalInvocationID.x);
  if (idx >= pc.size) return;
  
  output_data[idx] = input_data[idx] * pc.factor;
  
  barrier();  // Workgroup synchronization
}
```

### Type Mapping

| Sarek IR Type | GLSL Type | Notes |
|---------------|-----------|-------|
| `TInt32` | `int` | 32-bit signed |
| `TInt64` | `int64_t` | Requires GL_ARB_gpu_shader_int64 |
| `TFloat32` | `float` | 32-bit floating point |
| `TFloat64` | `double` | 64-bit floating point |
| `TBool` | `bool` | Boolean |
| `TRecord` | `struct` | Custom structs |
| `TVariant` | `struct` with `int tag` | Tagged unions |
| `TVec` | Storage buffer | Arrays via SSBO |

### Layout Qualifiers

GLSL requires explicit layout qualifiers:

```glsl
// Workgroup size (mandatory for compute shaders)
layout(local_size_x = 256) in;

// Storage buffer bindings
layout(std430, binding = 0) buffer buf_name { ... };

// Push constants for scalars
layout(push_constant) uniform PushConstants { ... };

// Shared memory (workgroup-local)
shared float data[256];
```

## Error Handling

All errors use structured error types:

### Codegen Errors

```ocaml
(* Unknown intrinsic function *)
let _ = Vulkan_error.raise_error 
  (Vulkan_error.unknown_intrinsic "my_func")
(* [Vulkan Codegen] Unknown intrinsic: my_func *)

(* Wrong number of arguments *)
let _ = Vulkan_error.raise_error
  (Vulkan_error.invalid_arg_count "atomic_add" 2 3)
(* [Vulkan Codegen] Intrinsic 'atomic_add' expects 2 arguments but got 3 *)
```

### Runtime Errors

```ocaml
(* Device not found *)
let _ = Vulkan_error.raise_error
  (Vulkan_error.device_not_found 5 3)
(* [Vulkan Runtime] Device 5 not found (valid range: 0-2) *)

(* Compilation failed *)
let _ = Vulkan_error.raise_error
  (Vulkan_error.compilation_failed "shader source" "syntax error at line 10")
(* [Vulkan Runtime] Compilation failed: syntax error at line 10 *)
```

### Plugin Errors

```ocaml
(* Library not found *)
let _ = Vulkan_error.raise_error
  (Vulkan_error.library_not_found "vulkan" ["/usr/lib"; "/opt/vulkan"])
(* [Vulkan Plugin] Library 'vulkan' not found in [/usr/lib, /opt/vulkan] *)

(* Feature not supported *)
let _ = Vulkan_error.raise_error
  (Vulkan_error.feature_not_supported "raw pointer arguments")
(* [Vulkan Plugin] Feature not supported: raw pointer arguments *)
```

## API Reference

### Device Management

```ocaml
(* Initialize Vulkan *)
Vulkan.Device.init ()

(* Get device by index *)
let dev = Vulkan.Device.get 0

(* Get device count *)
let n = Vulkan.Device.count ()

(* Device information *)
let name = dev.name  (* "Intel(R) Arc(tm) Graphics (MTL)" *)
let caps = dev.capabilities  (* supports_fp64, supports_atomics, etc. *)
```

### Memory Operations

```ocaml
(* Allocate device memory *)
let buffer, memory = Vulkan.Memory.malloc dev byte_size

(* Copy host → device *)
Vulkan.Memory.host_to_device ~src:host_array ~dst:buffer

(* Copy device → host *)
Vulkan.Memory.device_to_host ~src:buffer ~dst:host_array

(* Free memory *)
Vulkan.Memory.free dev buffer memory
```

### Kernel Execution

```ocaml
(* Compile GLSL kernel *)
let glsl_source = "..." in
let kernel = Vulkan.Kernel.compile dev ~name:"my_kernel" ~source:glsl_source

(* Or use cached version *)
let kernel = Vulkan.Kernel.compile_cached dev ~name:"my_kernel" ~source:glsl_source

(* Set arguments *)
let args = Vulkan.Kernel.create_args () in
Vulkan.Kernel.set_arg_buffer args 0 input_buffer ;
Vulkan.Kernel.set_arg_buffer args 1 output_buffer ;
Vulkan.Kernel.set_arg_int32 args 2 1024l ;

(* Launch kernel *)
Vulkan.Kernel.launch kernel
  ~args
  ~grid:{x=1024; y=1; z=1}
  ~block:{x=256; y=1; z=1}
  ~shared_mem:0
  ~stream:None
```

## Usage Examples

### Example 1: Vector Addition

```ocaml
open Spoc_core

(* Define kernel using Sarek PPX *)
let%kernel vector_add (a : float32 vec) (b : float32 vec) (c : float32 vec) (n : int32) =
  let idx = global_idx_x () in
  if idx < n then
    c.{idx} <- a.{idx} +. b.{idx}

(* Create vectors *)
let n = 1024 in
let a = Vector.create Float32 n in
let b = Vector.create Float32 n in
let c = Vector.create Float32 n in

(* Initialize data *)
for i = 0 to n - 1 do
  Vector.set a i (float i);
  Vector.set b i (float i *. 2.0)
done

(* Select Vulkan device *)
let dev = Device.get_by_name "Vulkan" in
Device.select dev;

(* Execute kernel *)
Execute.run vector_add
  ~grid:{x=4; y=1; z=1}
  ~block:{x=256; y=1; z=1}
  [VecArg a; VecArg b; VecArg c; IntArg (Int32.of_int n)]

(* Verify results *)
for i = 0 to n - 1 do
  assert (Vector.get c i = float i *. 3.0)
done
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
    barrier ()  (* Synchronize workgroup *)
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

### Example 4: External GLSL Shader

```ocaml
(* Load external GLSL shader *)
let glsl_source = {glsl|
#version 450
layout(local_size_x = 256) in;

layout(std430, binding = 0) buffer buf_in {
  float input_data[];
};

layout(std430, binding = 1) buffer buf_out {
  float output_data[];
};

void main() {
  uint idx = gl_GlobalInvocationID.x;
  output_data[idx] = input_data[idx] * 2.0;
}
|glsl}

(* Execute external shader *)
Execute.run_source
  ~lang:Framework_sig.GLSL_Source
  ~kernel_name:"double"
  ~grid:{x=4; y=1; z=1}
  ~block:{x=256; y=1; z=1}
  [VecArg input; VecArg output]
```

## GLSL Intrinsics

### Thread Indexing

| Sarek Intrinsic | GLSL Equivalent | Description |
|-----------------|-----------------|-------------|
| `thread_idx_x()` | `gl_LocalInvocationID.x` | Local thread index in workgroup (X) |
| `thread_idx_y()` | `gl_LocalInvocationID.y` | Local thread index in workgroup (Y) |
| `thread_idx_z()` | `gl_LocalInvocationID.z` | Local thread index in workgroup (Z) |
| `block_idx_x()` | `gl_WorkGroupID.x` | Workgroup index (X) |
| `block_idx_y()` | `gl_WorkGroupID.y` | Workgroup index (Y) |
| `block_idx_z()` | `gl_WorkGroupID.z` | Workgroup index (Z) |
| `block_dim_x()` | `gl_WorkGroupSize.x` | Workgroup size (X) |
| `block_dim_y()` | `gl_WorkGroupSize.y` | Workgroup size (Y) |
| `block_dim_z()` | `gl_WorkGroupSize.z` | Workgroup size (Z) |
| `global_idx_x()` | `gl_GlobalInvocationID.x` | Global thread index (X) |
| `global_idx_y()` | `gl_GlobalInvocationID.y` | Global thread index (Y) |
| `global_idx_z()` | `gl_GlobalInvocationID.z` | Global thread index (Z) |
| `grid_dim_x()` | `gl_NumWorkGroups.x` | Number of workgroups (X) |
| `grid_dim_y()` | `gl_NumWorkGroups.y` | Number of workgroups (Y) |
| `grid_dim_z()` | `gl_NumWorkGroups.z` | Number of workgroups (Z) |

### Synchronization

| Sarek Intrinsic | GLSL Equivalent | Description |
|-----------------|-----------------|-------------|
| `barrier()` | `barrier()` | Workgroup barrier (all memory) |
| `warp_barrier()` | `subgroupBarrier()` | Subgroup barrier |
| `memfence()` | `memoryBarrier()` | Memory fence |
| `memfence_local()` | `memoryBarrierShared()` | Shared memory fence |
| `memfence_global()` | `memoryBarrierBuffer()` | Global memory fence |

### Atomic Operations

Requires `#extension GL_EXT_shader_atomic_float : enable` for float atomics:

| Sarek Intrinsic | GLSL Equivalent | Types |
|-----------------|-----------------|-------|
| `atomic_add(addr, val)` | `atomicAdd(addr, val)` | int, uint, float |
| `atomic_sub(addr, val)` | `atomicAdd(addr, -val)` | int, uint |
| `atomic_min(addr, val)` | `atomicMin(addr, val)` | int, uint |
| `atomic_max(addr, val)` | `atomicMax(addr, val)` | int, uint |
| `atomic_and(addr, val)` | `atomicAnd(addr, val)` | int, uint |
| `atomic_or(addr, val)` | `atomicOr(addr, val)` | int, uint |
| `atomic_xor(addr, val)` | `atomicXor(addr, val)` | int, uint |
| `atomic_cas(addr, cmp, val)` | `atomicCompSwap(addr, cmp, val)` | int, uint |

### Math Functions

GLSL provides standard math functions:

| Category | Functions |
|----------|-----------|
| **Trigonometric** | `sin`, `cos`, `tan`, `asin`, `acos`, `atan` |
| **Exponential** | `pow`, `exp`, `log`, `exp2`, `log2`, `sqrt` |
| **Common** | `abs`, `sign`, `floor`, `ceil`, `fract`, `mod`, `min`, `max`, `clamp` |
| **Geometric** | `length`, `distance`, `dot`, `cross`, `normalize` |

## Testing

The Vulkan backend includes unit tests:

```bash
# Run all tests
dune test sarek-vulkan/test

# Run specific test suite
_build/default/sarek-vulkan/test/test_vulkan_error.exe
_build/default/sarek-vulkan/test/test_sarek_ir_glsl.exe
```

### Test Coverage

**Error Tests** (6 tests):
- Codegen errors (unknown_intrinsic, invalid_arg_count, unsupported_construct)
- Runtime errors (device_not_found, compilation_failed, no_device_selected)
- Plugin errors (library_not_found, unsupported_source_lang, feature_not_supported)
- Exception handling (Backend_error wrapper)
- Error prefix formatting
- Utility functions

**Codegen Tests** (14 tests):
- Literals (int32, int64, float32, float64, bool)
- Operations (add, sub, mul)
- Statements (empty, assignment, if, while, for)
- Barrier intrinsics
- Thread intrinsics
- Atomic operations
- Type mapping
- Helper functions

## Installation

### Requirements

**Runtime**:
- Vulkan loader (`libvulkan.so.1` / `vulkan-1.dll`)
- Vulkan-compatible GPU driver

**Optional** (for GLSL → SPIR-V compilation):
- `glslangValidator` (command-line tool), OR
- `libshaderc` (library)

### Ubuntu/Debian

```bash
# Vulkan runtime
sudo apt install libvulkan1 mesa-vulkan-drivers

# GLSL compiler (choose one)
sudo apt install glslang-tools     # glslangValidator
sudo apt install libshaderc1       # Shaderc library
```

### Arch Linux

```bash
sudo pacman -S vulkan-icd-loader vulkan-intel vulkan-radeon
sudo pacman -S glslang    # or shaderc
```

### macOS

Vulkan on macOS requires MoltenVK:

```bash
brew install molten-vk
brew install glslang    # or shaderc
```

### Verify Installation

```bash
# Check Vulkan devices
vulkaninfo | grep deviceName

# Test GLSL compiler
echo 'void main() {}' | glslangValidator --stdin -S comp
```

## Design Principles

### Type Safety

Uses GADTs and phantom types for typed memory and kernel arguments.

### Structured Errors

Uses shared `Backend_error` module for consistent error handling.

### Code Organization

Code generation uses extracted helper functions for maintainability.

### Performance Considerations

- **Kernel Cache**: Compiled SPIR-V cached at `~/.cache/sarek/vulkan/`
- **Push Constants**: Scalars passed via push constants (fast)
- **Storage Buffers**: Arrays via SSBOs (efficient)
- **Descriptor Sets**: Reused across kernel invocations

### Limitations

- **No device-to-device transfers**: Use staging buffer
- **No memset**: Not implemented
- **No raw pointers**: GLSL doesn't support pointer arguments
- **SPIR-V direct loading**: Not yet implemented

### Future Improvements

- Subgroup operations (wave intrinsics)
- Multiple queue families
- Async compute
- VK_KHR_shader_float16
- Bindless descriptor sets

## Troubleshooting

### "Library 'vulkan' not found"

Install Vulkan loader:
```bash
sudo apt install libvulkan1
```

### "glslangValidator failed"

Install GLSL compiler:
```bash
sudo apt install glslang-tools
```

Or use Shaderc library:
```bash
sudo apt install libshaderc1
```

### "No compute queue family found"

Your device doesn't support compute operations. Try a different device:
```ocaml
let devs = Device.list_all () in
List.iter (fun d -> Printf.printf "%s\n" d.name) devs
```

### "Device X not found"

Check available devices:
```ocaml
let n = Vulkan.Device.count () in
Printf.printf "Found %d devices\n" n
```

### SPIR-V Compilation Errors

Enable debug logging to see GLSL source:
```bash
SAREK_DEBUG=kernel ./my_program.exe
```

Compiled shader saved to `/tmp/sarek_debug.comp` and `/tmp/sarek_debug.spv`.

## License

See main SPOC repository for license information.

## Contributing

When contributing to the Vulkan backend:

1. Run all tests: `dune test sarek-vulkan/test`
2. Verify e2e tests: `dune build @sarek/tests/e2e/all`
3. Follow code quality guidelines (see AGENTS.md)
4. No `failwith` - use `Vulkan_error.raise_error`
5. No pretentious language in docs or commits

## Related Documentation

- **SPOC Framework**: Main SPOC README
- **Sarek PPX**: `sarek/README.md`
- **Backend Error**: `spoc/framework/Backend_error.ml`
- **OpenCL Backend**: `sarek-opencl/README.md`
- **CUDA Backend**: `sarek-cuda/README.md`
