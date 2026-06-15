# sarek-opencl - OpenCL Backend Plugin for SPOC/Sarek

**Package**: `sarek-opencl`  
**Library**: `sarek-opencl.plugin`

The OpenCL backend plugin enables SPOC/Sarek to compile and execute GPU kernels on OpenCL-compatible devices (GPUs, CPUs, FPGAs) using the OpenCL runtime. Uses ctypes-foreign for FFI bindings.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Core Modules](#core-modules)
- [Code Generation](#code-generation)
- [Error Handling](#error-handling)
- [API Reference](#api-reference)
- [Usage Examples](#usage-examples)
- [OpenCL Intrinsics](#opencl-intrinsics)
- [Testing](#testing)
- [Installation](#installation)
- [Design Principles](#design-principles)

## Overview

The OpenCL backend is one of several GPU backends supported by SPOC/Sarek. It translates Sarek IR (Intermediate Representation) into OpenCL C code, compiles it at runtime, and executes kernels on OpenCL devices.

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
│      OpenCL Backend (this package)      │  ← IR → OpenCL C translation
│    ┌─────────────────────────────┐     │
│    │  Sarek_ir_opencl.ml         │     │  - Code generation
│    │  → OpenCL C source          │     │  - Intrinsic mapping
│    └─────────────────────────────┘     │
├─────────────────────────────────────────┤
│         OpenCL Compiler                 │  ← Runtime compilation
│    ┌─────────────────────────────┐     │
│    │  Opencl_api.ml              │     │  - Build programs
│    │  → Binary/SPIR              │     │  - Error reporting
│    └─────────────────────────────┘     │
├─────────────────────────────────────────┤
│         OpenCL Runtime                  │  ← Execution
│    ┌─────────────────────────────┐     │
│    │  Opencl_api.ml              │     │  - Kernel execution
│    │  → Device operations        │     │  - Memory management
│    └─────────────────────────────┘     │  - Platform control
└─────────────────────────────────────────┘
```

### Key Features

- **Pure OCaml**: Uses `ctypes-foreign` for FFI
- **JIT Compilation**: Runtime compilation with OpenCL compiler
- **Multi-Platform**: Works across vendors (NVIDIA, AMD, Intel)
- **Device Discovery**: Automatic platform/device enumeration

### Supported Devices

The OpenCL backend works with any OpenCL-compatible device:
- **GPUs**: NVIDIA, AMD, Intel Arc
- **CPUs**: Intel, AMD with OpenCL drivers
- **Integrated GPUs**: Intel HD/Iris, AMD APU
- **FPGAs**: Altera/Intel FPGA SDK for OpenCL

## Architecture

### Execution Flow

1. **User writes kernel** using `[%kernel ...]` syntax
2. **Sarek PPX** compiles to typed IR
3. **Backend generates** OpenCL C source code
4. **OpenCL compiler** builds program for device
5. **Runtime executes** kernel on selected device

### Module Organization

```
sarek-opencl/
├── Opencl_error.ml         # Structured error handling
├── Opencl_types.ml         # OpenCL type definitions (514 lines)
├── Opencl_bindings.ml      # ctypes FFI bindings (431 lines)
├── Opencl_api.ml           # High-level API wrappers (678 lines)
├── Opencl_plugin_base.ml   # Base plugin implementation (460 lines)
├── Opencl_plugin.ml        # Plugin registration (291 lines)
├── Sarek_ir_opencl.ml      # IR → OpenCL C codegen (863 lines)
└── test/
    ├── test_opencl_error.ml      # Error tests (6)
    └── test_sarek_ir_opencl.ml   # Codegen tests (13)
```

## Core Modules

### 1. Opencl_error.ml (14 lines)

**Purpose**: Structured error handling using shared `Backend_error` module.

```ocaml
include Spoc_framework.Backend_error.Make(struct let name = "OpenCL" end)
exception Opencl_error = Spoc_framework.Backend_error.Backend_error
```

All error messages are prefixed with backend name:
- `[OpenCL Codegen]` - Code generation errors
- `[OpenCL Runtime]` - Device/compilation errors  
- `[OpenCL Plugin]` - Library/feature errors

### 2. Opencl_types.ml (514 lines)

**Purpose**: OpenCL type definitions for ctypes.

**Key Types**:
```ocaml
type cl_platform_id
type cl_device_id
type cl_context
type cl_command_queue
type cl_program
type cl_kernel
type cl_mem
type cl_event

type cl_device_type =
  | CL_DEVICE_TYPE_CPU
  | CL_DEVICE_TYPE_GPU
  | CL_DEVICE_TYPE_ACCELERATOR
  | CL_DEVICE_TYPE_ALL
```

### 3. Opencl_bindings.ml (431 lines)

**Purpose**: ctypes-foreign FFI bindings to libOpenCL.so.

**Library Loading**:
```ocaml
let opencl_lib = lazy (
  try Some (Dl.dlopen ~filename:"libOpenCL.so" ~flags:[Dl.RTLD_NOW])
  with _ -> None
)
```

**Foreign Function Declarations**:
```ocaml
(* Platform management *)
val clGetPlatformIDs : cl_uint -> cl_platform_id ptr -> cl_uint ptr -> cl_int
val clGetPlatformInfo : cl_platform_id -> cl_platform_info -> size_t -> unit ptr -> size_t ptr -> cl_int

(* Device management *)
val clGetDeviceIDs : cl_platform_id -> cl_device_type -> cl_uint -> cl_device_id ptr -> cl_uint ptr -> cl_int
val clGetDeviceInfo : cl_device_id -> cl_device_info -> size_t -> unit ptr -> size_t ptr -> cl_int

(* Context management *)
val clCreateContext : ... -> cl_context
val clReleaseContext : cl_context -> cl_int

(* Command queue *)
val clCreateCommandQueue : cl_context -> cl_device_id -> cl_command_queue_properties -> cl_int ptr -> cl_command_queue
val clFinish : cl_command_queue -> cl_int

(* Program compilation *)
val clCreateProgramWithSource : cl_context -> cl_uint -> string ptr -> size_t ptr -> cl_int ptr -> cl_program
val clBuildProgram : cl_program -> cl_uint -> cl_device_id ptr -> string -> unit ptr -> unit ptr -> cl_int
val clGetProgramBuildInfo : cl_program -> cl_device_id -> cl_program_build_info -> size_t -> unit ptr -> size_t ptr -> cl_int

(* Memory operations *)
val clCreateBuffer : cl_context -> cl_mem_flags -> size_t -> unit ptr -> cl_int ptr -> cl_mem
val clEnqueueWriteBuffer : cl_command_queue -> cl_mem -> cl_bool -> size_t -> size_t -> unit ptr -> cl_uint -> cl_event ptr -> cl_event ptr -> cl_int
val clEnqueueReadBuffer : cl_command_queue -> cl_mem -> cl_bool -> size_t -> size_t -> unit ptr -> cl_uint -> cl_event ptr -> cl_event ptr -> cl_int

(* Kernel execution *)
val clCreateKernel : cl_program -> string -> cl_int ptr -> cl_kernel
val clSetKernelArg : cl_kernel -> cl_uint -> size_t -> unit ptr -> cl_int
val clEnqueueNDRangeKernel : cl_command_queue -> cl_kernel -> cl_uint -> size_t ptr -> size_t ptr -> size_t ptr -> cl_uint -> cl_event ptr -> cl_event ptr -> cl_int
```

### 4. Opencl_api.ml (678 lines)

**Purpose**: High-level safe wrappers around OpenCL API.

**Module Structure**:
```ocaml
module Platform : sig
  val get_all : unit -> platform array
  val name : platform -> string
  val vendor : platform -> string
  val version : platform -> string
end

module Device : sig
  val count : unit -> int
  val get : int -> device
  val name : device -> string
  val device_type : device -> cl_device_type
  val max_compute_units : device -> int
  val max_work_group_size : device -> int
  val max_work_item_dimensions : device -> int
  val max_work_item_sizes : device -> int array
  val global_mem_size : device -> int64
  val local_mem_size : device -> int64
  val supports_fp64 : device -> bool
end

module Context : sig
  val create : device -> context
  val destroy : context -> unit
end

module Queue : sig
  val create : context -> queue
  val finish : queue -> unit
  val destroy : queue -> unit
end

module Memory : sig
  val alloc : context -> cl_mem_flags -> int -> buffer
  val write : queue -> buffer -> 'a ptr -> int -> unit
  val read : queue -> buffer -> 'a ptr -> int -> unit
  val release : buffer -> unit
end

module Program : sig
  val create_with_source : context -> string -> program
  val build : program -> ?options:string -> unit -> unit
  val get_build_log : program -> device -> string
  val release : program -> unit
end

module Kernel : sig
  val create : program -> string -> kernel
  val set_arg_buffer : kernel -> int -> buffer -> unit
  val set_arg_int32 : kernel -> int -> int32 -> unit
  val set_arg_int64 : kernel -> int -> int64 -> unit
  val set_arg_float32 : kernel -> int -> float -> unit
  val set_arg_float64 : kernel -> int -> float -> unit
  val enqueue_nd_range : 
    queue -> kernel -> 
    ?local_size:int array -> 
    global_size:int array -> 
    unit -> unit
  val release : kernel -> unit
end
```

### 5. Sarek_ir_opencl.ml (863 lines)

**Purpose**: IR to OpenCL C code generation.

**Key Functions**:
```ocaml
(** Generate OpenCL C code from Sarek IR *)
val generate : Sarek_ir_types.kernel -> string

(** Generate code for specific device (enables SNative) *)
val generate_for_device : Device.t -> Sarek_ir_types.kernel -> string

(** Expression generation *)
val gen_expr : Buffer.t -> Sarek_ir_types.expr -> unit

(** Statement generation (refactored, 134 lines) *)
val gen_stmt : Buffer.t -> string -> Sarek_ir_types.stmt -> unit

(** Helper: Match case generation *)
val gen_match_case : Buffer.t -> string -> string -> Sarek_ir_types.pattern -> Sarek_ir_types.stmt -> unit

(** Helper: Array declaration with __local *)
val gen_array_decl : Buffer.t -> string -> Sarek_ir_types.var -> Sarek_ir_types.elttype -> Sarek_ir_types.expr -> Sarek_ir_types.mem_space -> Sarek_ir_types.stmt -> unit
```

**Code Generation Features**:
- Intrinsic registry for extensible builtins
- Record/variant type support with C struct generation
- Pattern matching via switch statements
- Shared memory with `__local` qualifier
- Atomic operations
- Pragma support (`#pragma unroll`, etc.)
- Float64 extension detection (`cl_khr_fp64`)

## Code Generation

### Intrinsic Mapping

OpenCL provides different intrinsics than CUDA. The backend maps Sarek intrinsics to OpenCL equivalents:

| Sarek Intrinsic | OpenCL C | Description |
|-----------------|----------|-------------|
| `thread_idx_x` | `get_local_id(0)` | Thread ID within work-group |
| `block_idx_x` | `get_group_id(0)` | Work-group ID |
| `block_dim_x` | `get_local_size(0)` | Work-group size |
| `grid_dim_x` | `get_num_groups(0)` | Number of work-groups |
| `global_idx_x` | `get_global_id(0)` | Global thread ID |
| `barrier()` | `barrier(CLK_LOCAL_MEM_FENCE)` | Work-group barrier |
| `atomic_add` | `atomic_add` | Atomic addition |

### Type Mapping

| Sarek Type | OpenCL C Type |
|------------|---------------|
| `TInt32` | `int` |
| `TInt64` | `long` |
| `TFloat32` | `float` |
| `TFloat64` | `double` |
| `TBool` | `int` |
| `TVec(TFloat32, 4)` | `float4` |
| `TArray(TInt32, Local)` | `__local int*` |

### Example Code Generation

**Input IR**:
```ocaml
[%kernel
  fun[@opencl] (a : float array) (b : float array) (c : float array) ->
    let i = thread_idx_x + block_idx_x * block_dim_x in
    if i < Array.length a then
      c.(i) <- a.(i) +. b.(i)
]
```

**Generated OpenCL C**:
```c
__kernel void kernel_0(
  __global float* a, int sarek_a_length,
  __global float* b, int sarek_b_length,
  __global float* c, int sarek_c_length
) {
  int i = get_local_id(0) + get_group_id(0) * get_local_size(0);
  if (i < sarek_a_length) {
    c[i] = a[i] + b[i];
  }
}
```

## Error Handling

The OpenCL backend uses the shared `Backend_error` module from `spoc.framework` for structured error handling.

### Module Overview

```ocaml
(** OpenCL-specific error module *)
module Opencl_error = Backend_error.Make(struct let name = "OpenCL" end)

(** Exception type *)
exception Opencl_error = Backend_error.Backend_error
```

All error messages are prefixed with the backend name and category:
- `[OpenCL Codegen]` - Code generation errors
- `[OpenCL Runtime]` - Runtime errors (device, compilation, memory)
- `[OpenCL Plugin]` - Plugin errors (library loading, unsupported features)

### Codegen Errors

```ocaml
(* Unknown intrinsic *)
let x = Intrinsics.unknown_func () in ...
→ Error: [OpenCL Codegen] Unknown intrinsic: unknown_func

(* Unsupported construct *)
match [] with [] -> ...  (* empty match *)
→ Error: [OpenCL Codegen] Unsupported construct 'EMatch': empty match expression

(* Invalid argument count *)
atomic_add(x)  (* needs 2 or 3 args *)
→ Error: [OpenCL Codegen] Intrinsic 'atomic_add' expects 2 arguments but got 1
```

### Runtime Errors

```ocaml
(* Device not found *)
let dev = Device.get 5 in ...  (* only 2 devices available *)
→ Error: [OpenCL Runtime] Device ID 5 not found (available: 0-1)

(* Compilation failure *)
let prog = Program.build program in ...  (* syntax error in kernel *)
→ Error: [OpenCL Runtime] Compilation failed for:
         kernel void test() { invalid syntax }
         
         Compiler log:
         error: expected ';' at line 1
```

### Plugin Errors

```ocaml
(* Library not found *)
(* When libOpenCL.so is missing *)
→ Error: [OpenCL Plugin] Library 'libOpenCL.so' not found in: /usr/lib, /usr/local/lib, /opt/lib

(* Unsupported source language *)
execute_kernel ~source_lang:"CUDA_Source" ...
→ Error: [OpenCL Plugin] Source language not supported: CUDA_Source

(* Feature not supported *)
copy_device_to_device src dst  (* D2D not implemented *)
→ Error: [OpenCL Plugin] Feature not supported: device-to-device copy
```

### Error Recovery

```ocaml
(* Use with_default for safe fallback *)
let device_count = 
  Opencl_error.with_default ~default:0 (fun () ->
    Opencl_api.Device.count ()
  )

(* Convert to Result type *)
let result = Opencl_error.to_result (fun () ->
  compile_and_run_kernel ()
)
match result with
| Ok () -> print_endline "Success"
| Error err -> Printf.eprintf "Error: %s\n" (Opencl_error.to_string err)
```

## API Reference

### Device Management

```ocaml
(* Count available OpenCL devices *)
val Device.count : unit -> int

(* Get device by global index *)
val Device.get : int -> device

(* Query device properties *)
val Device.name : device -> string
val Device.max_compute_units : device -> int
val Device.max_work_group_size : device -> int
val Device.global_mem_size : device -> int64
val Device.local_mem_size : device -> int64
val Device.supports_fp64 : device -> bool
```

### Context and Queue

```ocaml
(* Create OpenCL context for device *)
val Context.create : device -> context

(* Create command queue *)
val Queue.create : context -> queue

(* Wait for all operations to complete *)
val Queue.finish : queue -> unit
```

### Memory Operations

```ocaml
(* Allocate device buffer *)
val Memory.alloc : context -> cl_mem_flags -> int -> buffer

(* Transfer host → device *)
val Memory.write : queue -> buffer -> 'a ptr -> int -> unit

(* Transfer device → host *)
val Memory.read : queue -> buffer -> 'a ptr -> int -> unit

(* Release buffer *)
val Memory.release : buffer -> unit
```

### Program Compilation

```ocaml
(* Create program from OpenCL C source *)
val Program.create_with_source : context -> string -> program

(* Compile program for device *)
val Program.build : program -> ?options:string -> unit -> unit

(* Get build log (useful for errors) *)
val Program.get_build_log : program -> device -> string
```

### Kernel Execution

```ocaml
(* Create kernel from compiled program *)
val Kernel.create : program -> string -> kernel

(* Set kernel arguments *)
val Kernel.set_arg_buffer : kernel -> int -> buffer -> unit
val Kernel.set_arg_int32 : kernel -> int -> int32 -> unit
val Kernel.set_arg_float32 : kernel -> int -> float -> unit

(* Execute kernel *)
val Kernel.enqueue_nd_range :
  queue -> kernel ->
  ?local_size:int array ->
  global_size:int array ->
  unit -> unit
```

## Usage Examples

### Example 1: Vector Addition

```ocaml
open Sarek
open Spoc_core

(* Define kernel *)
let vector_add = [%kernel
  fun[@opencl] (a : float array) (b : float array) (c : float array) ->
    let i = global_idx_x in
    if i < Array.length a then
      c.(i) <- a.(i) +. b.(i)
]

(* Execute on OpenCL device *)
let () =
  let n = 1024 in
  let a = Vector.create Float32 n in
  let b = Vector.create Float32 n in
  let c = Vector.create Float32 n in
  
  (* Initialize on CPU *)
  Vector.init a (fun i -> float i) ;
  Vector.init b (fun i -> float i *. 2.0) ;
  
  (* Select OpenCL device *)
  let dev = Device.get 0 in  (* First OpenCL device *)
  
  (* Execute kernel *)
  Execute.run vector_add ~dev [|a; b; c|] ~grid:(dims_1d n) ~block:(dims_1d 256) ;
  
  (* Read results *)
  let result = Vector.to_array c in
  Printf.printf "c[0] = %f\n" result.(0)  (* 0.0 *)
```

### Example 2: Matrix Multiplication (Work-Groups)

```ocaml
let matmul = [%kernel
  fun[@opencl] (a : float array) (b : float array) (c : float array) (n : int) ->
    let row = global_idx_y in
    let col = global_idx_x in
    
    if row < n && col < n then begin
      let sum = ref 0.0 in
      for k = 0 to n - 1 do
        sum := !sum +. a.(row * n + k) *. b.(k * n + col)
      done ;
      c.(row * n + col) <- !sum
    end
]

let () =
  let n = 512 in
  let size = n * n in
  let a = Vector.create Float32 size in
  let b = Vector.create Float32 size in
  let c = Vector.create Float32 size in
  
  (* Initialize matrices... *)
  
  let dev = Device.get 0 in
  Execute.run matmul ~dev 
    [|a; b; c; Vector.of_int32 (Int32.of_int n)|]
    ~grid:(dims_2d n n)
    ~block:(dims_2d 16 16)
```

### Example 3: Reduction with Shared Memory

```ocaml
let reduction = [%kernel
  fun[@opencl] (input : float array) (output : float array) ->
    (* Shared memory for work-group *)
    let shared : float array = Array.create_local 256 0.0 in
    
    let tid = thread_idx_x in
    let gid = global_idx_x in
    
    (* Load into shared memory *)
    if gid < Array.length input then
      shared.(tid) <- input.(gid)
    else
      shared.(tid) <- 0.0 ;
    
    barrier() ;
    
    (* Reduction in shared memory *)
    let stride = ref (block_dim_x / 2) in
    while !stride > 0 do
      if tid < !stride then
        shared.(tid) <- shared.(tid) +. shared.(tid + !stride) ;
      barrier() ;
      stride := !stride / 2
    done ;
    
    (* Write result *)
    if tid = 0 then
      output.(block_idx_x) <- shared.(0)
]
```

### Example 4: Histogram with Atomics

```ocaml
let histogram = [%kernel
  fun[@opencl] (data : int array) (hist : int array) (bins : int) ->
    let gid = global_idx_x in
    
    if gid < Array.length data then begin
      let value = data.(gid) in
      let bin = value mod bins in
      atomic_add hist.(bin) 1  (* Atomic increment *)
    end
]
```

### Example 5: Device Queries

```ocaml
(* List all OpenCL devices *)
let () =
  Printf.printf "OpenCL devices:\n" ;
  for i = 0 to Device.count () - 1 do
    let dev = Device.get i in
    Printf.printf "  [%d] %s\n" i (Device.name dev) ;
    Printf.printf "      Compute units: %d\n" (Device.max_compute_units dev) ;
    Printf.printf "      Global memory: %Ld bytes\n" (Device.global_mem_size dev) ;
    Printf.printf "      Local memory: %Ld bytes\n" (Device.local_mem_size dev) ;
    Printf.printf "      FP64 support: %b\n" (Device.supports_fp64 dev)
  done
```

## OpenCL Intrinsics

### Work-Item Functions

| Intrinsic | OpenCL Equivalent | Description |
|-----------|-------------------|-------------|
| `thread_idx_x` | `get_local_id(0)` | Local thread ID (x) |
| `thread_idx_y` | `get_local_id(1)` | Local thread ID (y) |
| `thread_idx_z` | `get_local_id(2)` | Local thread ID (z) |
| `block_idx_x` | `get_group_id(0)` | Work-group ID (x) |
| `block_idx_y` | `get_group_id(1)` | Work-group ID (y) |
| `block_idx_z` | `get_group_id(2)` | Work-group ID (z) |
| `block_dim_x` | `get_local_size(0)` | Work-group size (x) |
| `block_dim_y` | `get_local_size(1)` | Work-group size (y) |
| `block_dim_z` | `get_local_size(2)` | Work-group size (z) |
| `grid_dim_x` | `get_num_groups(0)` | Number of work-groups (x) |
| `grid_dim_y` | `get_num_groups(1)` | Number of work-groups (y) |
| `grid_dim_z` | `get_num_groups(2)` | Number of work-groups (z) |
| `global_idx_x` | `get_global_id(0)` | Global thread ID (x) |
| `global_idx_y` | `get_global_id(1)` | Global thread ID (y) |
| `global_idx_z` | `get_global_id(2)` | Global thread ID (z) |
| `global_size` | `get_global_size(0)` | Total threads (x) |

### Synchronization

| Intrinsic | OpenCL Equivalent | Description |
|-----------|-------------------|-------------|
| `barrier()` | `barrier(CLK_LOCAL_MEM_FENCE)` | Work-group barrier |
| `mem_fence()` | `mem_fence(CLK_GLOBAL_MEM_FENCE)` | Global memory fence |

### Atomic Operations

| Intrinsic | OpenCL Equivalent | Description |
|-----------|-------------------|-------------|
| `atomic_add(addr, val)` | `atomic_add(&addr, val)` | Atomic addition |
| `atomic_sub(addr, val)` | `atomic_sub(&addr, val)` | Atomic subtraction |
| `atomic_min(addr, val)` | `atomic_min(&addr, val)` | Atomic minimum |
| `atomic_max(addr, val)` | `atomic_max(&addr, val)` | Atomic maximum |

### Math Functions

Standard OpenCL math functions are available through the intrinsic registry:
- `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2`
- `sqrt`, `rsqrt`, `pow`, `exp`, `log`, `log2`, `log10`
- `floor`, `ceil`, `round`, `trunc`, `fabs`
- `fmin`, `fmax`, `fmod`, `copysign`

## Testing

### Running Tests

```bash
# Build tests
dune build sarek-opencl/test

# Run error tests (6 tests)
_build/default/sarek-opencl/test/test_opencl_error.exe

# Run IR codegen tests (13 tests)
_build/default/sarek-opencl/test/test_sarek_ir_opencl.exe

# Run with coverage
dune runtest --instrument-with bisect_ppx sarek-opencl/test
bisect-ppx-report html  # Generate coverage report
```

### Test Coverage

**Error Tests** (test_opencl_error.ml - 6 tests):
- Codegen errors: unsupported_construct, invalid_arg_count, unknown_intrinsic
- Runtime errors: device_not_found, compilation_failed, no_device_selected
- Plugin errors: unsupported_source_lang, library_not_found, feature_not_supported
- Utilities: with_default, to_result, error_equality

**IR Codegen Tests** (test_sarek_ir_opencl.ml - 13 tests):
- Literals: int32, int64, float32, float64, bool
- Operations: binary ops, assignments
- Control flow: if, while, for, return
- Synchronization: barriers
- Declarations: let, let mut, arrays
- Blocks and pragmas

## Installation

### System Requirements

- **OpenCL Runtime**: ICD loader (libOpenCL.so)
- **OpenCL Drivers**: Vendor-specific drivers
  - NVIDIA: CUDA Toolkit (includes OpenCL)
  - AMD: ROCm or Adrenalin drivers
  - Intel: NEO compute runtime

### Install OpenCL (Ubuntu/Debian)

```bash
# Install ICD loader
sudo apt-get install ocl-icd-libopencl1

# NVIDIA (if using NVIDIA GPU)
sudo apt-get install nvidia-cuda-toolkit

# Intel (CPU or integrated GPU)
sudo apt-get install intel-opencl-icd

# AMD (if using AMD GPU)
# Install ROCm or Adrenalin drivers from AMD website
```

### Install via opam

```bash
# Install sarek-opencl
opam install sarek-opencl

# Or build from source
git clone https://github.com/yourusername/SPOC.git
cd SPOC
opam install . --deps-only
dune build sarek-opencl
```

### Verify Installation

```bash
# Check if OpenCL devices are visible
clinfo  # If installed
# or
_build/default/sarek/tests/e2e/test_vector_add.exe  # Should list OpenCL devices
```

## Design Principles

### Pure OCaml FFI

All OpenCL bindings use `ctypes-foreign`:

```ocaml
let clGetPlatformIDs = 
  foreign "clGetPlatformIDs" 
    (cl_uint @-> ptr cl_platform_id @-> ptr cl_uint @-> returning cl_int)
```

### Structured Error Handling

Uses shared `Backend_error` module for consistent error handling:

```ocaml
include Backend_error.Make(struct let name = "OpenCL" end)
```

### Code Organization

Code generation uses extracted helpers for maintainability:
- `gen_stmt`: 190 → 134 lines (30% reduction)
- Helper functions: `gen_match_case`, `gen_array_decl`
- Named constants: `small_buffer_size`, `large_buffer_size`
- No unsafe patterns (List.hd replaced with pattern matching)

### 4. Multi-Vendor Support

Works with any OpenCL implementation:
- NVIDIA GPUs (via CUDA OpenCL)
- AMD GPUs (via ROCm)
- Intel GPUs (via NEO)
- Intel/AMD CPUs (via OpenCL runtimes)

### 5. Runtime Compilation (JIT)

Kernels compile at runtime:
- Device-specific optimizations
- No pre-compilation needed
- Supports all OpenCL versions

## Troubleshooting

### OpenCL Library Not Found

```
Error: [OpenCL Plugin] Library 'libOpenCL.so' not found in: /usr/lib, /usr/local/lib, /opt/lib
```

**Solution**: Install OpenCL ICD loader
```bash
sudo apt-get install ocl-icd-libopencl1
```

### No OpenCL Devices Found

```
Error: [OpenCL Runtime] Device ID 0 not found (available: 0--1)
```

**Solution**: Install device drivers
- NVIDIA: `sudo apt-get install nvidia-cuda-toolkit`
- Intel: `sudo apt-get install intel-opencl-icd`
- AMD: Install ROCm from AMD website

### Compilation Failed

```
Error: [OpenCL Runtime] Compilation failed for:
...
Compiler log:
error: expected ';' at line 5
```

**Solution**: Check generated OpenCL C code. Enable debug logging:
```bash
SAREK_DEBUG=kernel ./your_program
```

### Double Precision Not Supported

```
Error: Device does not support double precision
```

**Solution**: Check device capabilities or use float32:
```ocaml
let supports_fp64 = Device.supports_fp64 dev
```

## Performance Tips

### Work-Group Sizing

- Use multiples of warp/wavefront size (32 for NVIDIA, 64 for AMD)
- Stay within device limits (`max_work_group_size`)
- Consider occupancy vs shared memory usage

### Memory Access Patterns

- Coalesce global memory accesses (adjacent threads access adjacent memory)
- Use `__local` memory for shared data within work-groups
- Minimize global memory transactions

### Atomics

- Minimize atomic operations (they serialize)
- Use local memory for work-group reductions
- Consider algorithmic alternatives when possible

## Related Documentation

- [SPOC/Sarek Main README](../README.md)
- [Sarek Framework](../sarek/framework/README.md)
- [CUDA Backend](../sarek-cuda/README.md)
- [Backend Error Module](../spoc/framework/Backend_error.md)

## License

See [LICENSE.md](../LICENSE.md)
