# sarek-cuda - CUDA Backend Plugin for SPOC/Sarek

**Package**: `sarek-cuda`  
**Library**: `sarek-cuda.plugin`

The CUDA backend plugin enables SPOC/Sarek to compile and execute GPU kernels on NVIDIA GPUs using the CUDA runtime and NVRTC (NV Runtime Compilation) library. Uses ctypes-foreign for FFI bindings.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Core Modules](#core-modules)
- [Code Generation](#code-generation)
- [Error Handling](#error-handling)
- [API Reference](#api-reference)
- [Usage Examples](#usage-examples)
- [CUDA Intrinsics](#cuda-intrinsics)
- [Testing](#testing)
- [Installation](#installation)
- [Design Principles](#design-principles)

## Overview

The CUDA backend is one of five GPU backends supported by SPOC/Sarek. It translates Sarek IR (Intermediate Representation) into CUDA C code, compiles it at runtime using NVRTC, and executes kernels on NVIDIA GPUs.

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
│       CUDA Backend (this package)       │  ← IR → CUDA C translation
│    ┌─────────────────────────────┐     │
│    │  Sarek_ir_cuda.ml           │     │  - Code generation
│    │  → CUDA C source            │     │  - Intrinsic mapping
│    └─────────────────────────────┘     │
├─────────────────────────────────────────┤
│          NVRTC Compiler                 │  ← Runtime compilation
│    ┌─────────────────────────────┐     │
│    │  Cuda_nvrtc.ml              │     │  - Compile to PTX
│    │  → PTX bytecode             │     │  - Error reporting
│    └─────────────────────────────┘     │
├─────────────────────────────────────────┤
│         CUDA Driver API                 │  ← Execution
│    ┌─────────────────────────────┐     │
│    │  Cuda_api.ml                │     │  - Module loading
│    │  → Kernel execution         │     │  - Memory management
│    └─────────────────────────────┘     │  - Device control
└─────────────────────────────────────────┘
```

### Key Features

- **Pure OCaml**: Uses `ctypes-foreign` for FFI
- **JIT Compilation**: Runtime compilation via NVRTC
- **CUDA Features**: Shared memory, atomics, barriers, synchronization
- **Intrinsics**: Thread IDs, math functions, atomics, memory fences
- **Debug Logging**: Controlled via `SAREK_DEBUG` environment variable

## Architecture

### Module Structure

```
sarek-cuda/
├── Cuda_error.ml          (169 lines)  - Structured error types
├── Cuda_types.ml          (446 lines)  - CUDA type definitions
├── Cuda_bindings.ml       (516 lines)  - FFI bindings via ctypes
├── Cuda_nvrtc.ml          (338 lines)  - NVRTC compiler interface
├── Cuda_api.ml            (455 lines)  - High-level CUDA API
├── Cuda_plugin_base.ml    (224 lines)  - Plugin base implementation
├── Cuda_plugin.ml         (262 lines)  - Plugin registration
├── Sarek_ir_cuda.ml       (919 lines)  - IR → CUDA C code generation
└── test/
    ├── test_cuda_error.ml     (148 lines)  - Error handling tests
    └── test_sarek_ir_cuda.ml  (267 lines)  - Code generation tests
```

### Execution Flow

1. **Kernel Definition**: User writes kernel in OCaml syntax with `[%kernel ...]`
2. **PPX Processing**: Sarek PPX compiles to typed IR (Sarek_ir_types.kernel)
3. **Code Generation**: `Sarek_ir_cuda.generate` translates IR to CUDA C source
4. **Compilation**: `Cuda_nvrtc` compiles source to PTX bytecode
5. **Loading**: `Cuda_api.Module.load_ptx` loads PTX into CUDA context
6. **Execution**: `Cuda_api.Kernel.launch` executes on GPU
7. **Results**: Memory transferred back to host via `Cuda_api.Memory.memcpy_dtoh`

### Plugin Registration

The CUDA backend auto-registers with the framework at library load time:

```ocaml
(* From Cuda_plugin.ml *)
let () =
  Framework_registry.register_backend
    ~priority:100  (* Default priority for CUDA backend *)
    (module Plugin : Framework_sig.BACKEND)
```

## Core Modules

### 1. Cuda_error.ml (169 lines)

**Purpose**: Structured error handling for all CUDA operations.

**Error Categories**:
- `codegen_error`: IR translation errors (unsupported constructs, type errors)
- `runtime_error`: CUDA runtime errors (device not found, compilation failed)
- `plugin_error`: Backend availability errors (missing libraries, unsupported features)

**Key Types**:
```ocaml
type codegen_error =
  | Unknown_intrinsic of { name : string }
  | Invalid_arg_count of { intrinsic : string; expected : int; got : int }
  | Unsupported_construct of { construct : string; reason : string }
  | Type_error of { expr : string; expected : string; got : string }
  | Invalid_memory_space of { decl : string; space : string }

type runtime_error =
  | No_device_selected of { operation : string }
  | Device_not_found of { device_id : int; max_devices : int }
  | Compilation_failed of { source : string; log : string }
  | Module_load_failed of { ptx_size : int; reason : string }

type plugin_error =
  | Unsupported_source_lang of { lang : string; backend : string }
  | Backend_unavailable of { reason : string }
  | Library_not_found of { library : string; paths : string list }
```

**Helper Functions**:
```ocaml
val to_string : cuda_error -> string
val raise_error : cuda_error -> 'a
val with_default : default:'a -> (unit -> 'a) -> 'a
val to_result : (unit -> 'a) -> ('a, cuda_error) result
```

### 2. Cuda_types.ml (446 lines)

**Purpose**: CUDA type definitions for ctypes bindings.

**Key Types**:
- Device management: `cuda_device`, `cuda_context`
- Module/function: `cuda_module`, `cuda_function`
- Memory: `cuda_device_ptr` (GPU memory pointers)
- Streams/events: `cuda_stream`, `cuda_event`
- Compilation: `nvrtc_program` (NVRTC programs)

**Enums and Constants**:
```ocaml
type cuda_mem_copy_kind =
  | HostToHost
  | HostToDevice
  | DeviceToHost
  | DeviceToDevice

type cuda_device_attribute =
  | MaxThreadsPerBlock
  | MaxBlockDimX
  | MaxGridDimX
  | SharedMemoryPerBlock
  | ComputeCapabilityMajor
  (* ... 30+ more attributes *)
```

### 3. Cuda_bindings.ml (516 lines)

**Purpose**: Low-level FFI bindings to CUDA Driver API and NVRTC via ctypes-foreign.

**Library Loading**:
```ocaml
let cuda_lib = Dl.dlopen ~filename:"libcuda.so" ~flags:[Dl.RTLD_NOW]
let nvrtc_lib = Dl.dlopen ~filename:"libnvrtc.so" ~flags:[Dl.RTLD_NOW]
```

**Foreign Function Declarations**:
```ocaml
(* Device management *)
val cuDeviceGet : int -> cuda_device ptr -> cuda_result
val cuDeviceGetCount : int ptr -> cuda_result
val cuDeviceGetName : string -> int -> cuda_device -> cuda_result

(* Context management *)
val cuCtxCreate : cuda_context ptr -> uint -> cuda_device -> cuda_result
val cuCtxDestroy : cuda_context -> cuda_result

(* Memory operations *)
val cuMemAlloc : cuda_device_ptr ptr -> size_t -> cuda_result
val cuMemcpyHtoD : cuda_device_ptr -> unit ptr -> size_t -> cuda_result

(* Module/kernel operations *)
val cuModuleLoadData : cuda_module ptr -> string -> cuda_result
val cuModuleGetFunction : cuda_function ptr -> cuda_module -> string -> cuda_result
val cuLaunchKernel : cuda_function -> ... -> cuda_result

(* NVRTC compilation *)
val nvrtcCreateProgram : nvrtc_program ptr -> string -> string -> ... -> nvrtc_result
val nvrtcCompileProgram : nvrtc_program -> int -> string ptr -> nvrtc_result
val nvrtcGetPTX : nvrtc_program -> string -> nvrtc_result
```

**Error Handling**:
All functions return result codes (`cuda_result`, `nvrtc_result`) which are checked and converted to structured errors.

### 4. Cuda_nvrtc.ml (338 lines)

**Purpose**: High-level interface to NVRTC (NVIDIA Runtime Compilation).

**Key Functions**:
```ocaml
(** Compile CUDA C source to PTX bytecode *)
val compile_to_ptx : 
  ?options:string list ->  (* e.g., ["-arch=compute_70"] *)
  ?name:string ->          (* Program name for error messages *)
  string ->                (* CUDA C source code *)
  string                   (* PTX bytecode *)

(** Get compiler version *)
val version : unit -> int * int  (* major, minor *)

(** Add default options for architecture *)
val default_options : int -> string list
```

**Compilation Process**:
1. Create NVRTC program from source string
2. Add compiler options (architecture, optimization flags)
3. Compile to PTX
4. Extract PTX bytecode
5. Cleanup resources

**Error Messages**:
```ocaml
(* Example compilation error *)
exception Cuda_error (
  Compilation_failed {
    source = "__global__ void kernel() {\n  ...\n}";
    log = "error: identifier \"foo\" is undefined\n  at line 5"
  }
)
```

### 5. Cuda_api.ml (455 lines)

**Purpose**: High-level safe wrappers around CUDA Driver API.

**Module Structure**:
```ocaml
module Device : sig
  val get : int -> device
  val count : unit -> int
  val name : device -> string
  val compute_capability : device -> int * int
  val max_threads_per_block : device -> int
  (* ... 20+ device query functions *)
end

module Context : sig
  val create : ?flags:int -> device -> context
  val destroy : context -> unit
  val set_current : context -> unit
end

module Memory : sig
  val alloc : int -> device_ptr
  val free : device_ptr -> unit
  val memcpy_htod : dest:device_ptr -> src:'a ptr -> size:int -> unit
  val memcpy_dtoh : dest:'a ptr -> src:device_ptr -> size:int -> unit
  val memcpy_dtod : dest:device_ptr -> src:device_ptr -> size:int -> unit
end

module Module : sig
  val load_ptx : string -> cuda_module
  val get_function : cuda_module -> string -> cuda_function
end

module Kernel : sig
  type launch_config = {
    grid_dim : int * int * int;
    block_dim : int * int * int;
    shared_mem : int;
    stream : cuda_stream option;
  }
  
  val launch : cuda_function -> launch_config -> unit ptr array -> unit
end

module Stream : sig
  val create : unit -> cuda_stream
  val destroy : cuda_stream -> unit
  val synchronize : cuda_stream -> unit
end

module Event : sig
  val create : unit -> cuda_event
  val record : cuda_event -> cuda_stream option -> unit
  val synchronize : cuda_event -> unit
  val elapsed_time : cuda_event -> cuda_event -> float
end
```

**Constants**:
```ocaml
(* Maximum lengths for queries *)
let max_device_name_length = 256
let max_ptx_header_preview = 200

(* Memory alignment *)
let alignment = 256  (* CUDA memory alignment requirement *)
```

### 6. Sarek_ir_cuda.ml (919 lines)

**Purpose**: Translate Sarek IR (typed AST) to CUDA C source code.

**Main Entry Point**:
```ocaml
val generate : Sarek_ir_types.kernel -> string
  (** Generate complete CUDA C source from kernel IR *)

val generate_for_device : device:Device.t -> Sarek_ir_types.kernel -> string
  (** Generate with device-specific optimizations *)
```

**Code Generation Functions**:
```ocaml
(* Expression generation *)
val gen_expr : Buffer.t -> Sarek_ir_types.expr -> unit
val gen_lvalue : Buffer.t -> Sarek_ir_types.lvalue -> unit
val gen_intrinsic : Buffer.t -> string list -> string -> expr list -> unit

(* Statement generation *)
val gen_stmt : Buffer.t -> string -> Sarek_ir_types.stmt -> unit
  (* Refactored with helper functions: *)
and gen_record_assign : Buffer.t -> string -> lvalue -> (string * expr) list -> unit
and gen_match_case : Buffer.t -> string -> string -> pattern -> stmt -> unit
and gen_array_decl : Buffer.t -> var -> elttype -> expr -> memspace -> unit

(* Declaration generation *)
val gen_param : Buffer.t -> Sarek_ir_types.decl -> unit
val gen_local : Buffer.t -> string -> Sarek_ir_types.decl -> unit
val gen_helper_func : Buffer.t -> Sarek_ir_types.helper_func -> unit

(* Type conversion *)
val cuda_type_of_elttype : Sarek_ir_types.elttype -> string
  (* TInt32 → "int", TFloat64 → "double", etc. *)
```

**Generated Code Structure**:
```cuda
// Auto-generated by Sarek CUDA backend

// Device helper functions
__device__ int helper_func(int x) { ... }

// Main kernel
__global__ void kernel_name(
    int* arr, int sarek_arr_length,
    float value
) {
    // Thread ID intrinsics
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Shared memory declarations
    __shared__ float shared_data[256];
    
    // Kernel body
    if (tid < sarek_arr_length) {
        shared_data[threadIdx.x] = arr[tid] * value;
        __syncthreads();
        arr[tid] = shared_data[threadIdx.x];
    }
}
```

**Intrinsic Mapping**:
```ocaml
(* Thread intrinsics *)
"thread_idx_x" → "threadIdx.x"
"block_idx_x"  → "blockIdx.x"
"block_dim_x"  → "blockDim.x"
"grid_dim_x"   → "gridDim.x"
"global_idx"   → "threadIdx.x + blockIdx.x * blockDim.x"

(* Math intrinsics *)
"sin", "cos", "sqrt" → "sin", "cos", "sqrt"  (* Direct mapping *)
"fma" → "fma"  (* Fused multiply-add *)

(* Atomic intrinsics *)
"atomic_add" → "atomicAdd"
"atomic_cas" → "atomicCAS"

(* Synchronization *)
"barrier"      → "__syncthreads()"
"warp_barrier" → "__syncwarp()"
"mem_fence"    → "__threadfence()"
```

### 7. Cuda_plugin.ml (262 lines)

**Purpose**: Plugin interface implementation and registration.

**Backend Implementation**:
```ocaml
module Plugin : Framework_sig.BACKEND = struct
  let name = "CUDA"
  let version = (12, 0, 0)  (* CUDA 12.0 *)
  let execution_model = Framework_sig.JIT  (* Just-In-Time compilation *)
  
  let is_available () =
    try
      let count = Cuda_api.Device.count () in
      count > 0
    with _ -> false
  
  module Device = ... (* Device management interface *)
  module Stream = ... (* Stream operations *)
  module Memory = ... (* Memory allocation/transfer *)
  module Kernel = ... (* Kernel compilation/launch *)
  module Event = ...  (* Event synchronization *)
  
  module Intrinsics = Cuda_intrinsics  (* 57 CUDA intrinsics *)
  
  let generate_source ?block kernel =
    Some (Sarek_ir_cuda.generate kernel)
  
  let execute_direct ~device ~source ~kernel_name ~grid ~block ~args () =
    (* JIT compile and execute *)
    let ptx = Cuda_nvrtc.compile_to_ptx source in
    let module_ = Cuda_api.Module.load_ptx ptx in
    let func = Cuda_api.Module.get_function module_ kernel_name in
    Cuda_api.Kernel.launch func { grid_dim = grid; block_dim = block; ... } args
end
```

**Priority System**:
```ocaml
(* Backend priority affects selection order *)
let () = Framework_registry.register_backend ~priority:100 (module Plugin)

(* Priority values used by backends:
   - CUDA: 100
   - OpenCL: 80
   - Vulkan: 70
   - Native: 50 (CPU fallback)
   - Interpreter: 30
*)
```

## Code Generation

### Type System

CUDA C types are generated from Sarek IR types:

```ocaml
(* Scalar types *)
TInt32   → "int"
TInt64   → "long long"
TFloat32 → "float"
TFloat64 → "double"
TBool    → "int"  (* C99 doesn't have bool in device code *)

(* Vector types (Bigarray-backed) *)
TVec (TFloat32, n) → "float*"
TVec (TFloat64, n) → "double*"

(* Custom types (records, variants) *)
TCustom "MyRecord" → "struct MyRecord"
TCustom "MyVariant" → "union MyVariant_data"  (* with tag field *)
```

### Record Types

Records are translated to C structs:

```ocaml
(* Sarek IR *)
type point = { x: float; y: float }

(* Generated CUDA C *)
struct point {
    float x;
    float y;
};
```

**Record Assignment**:
```ocaml
(* IR: lvalue = { x = 1.0; y = 2.0 } *)
(* Generated (field-by-field, no compound literals): *)
lvalue.x = 1.0;
lvalue.y = 2.0;
```

### Variant Types

Variants are translated to tagged unions:

```ocaml
(* Sarek IR *)
type result = Ok of int | Error of string

(* Generated CUDA C *)
enum result_tag { Ok, Error };

union result_data {
    int Ok_v;
    char* Error_v;
};

struct result {
    enum result_tag tag;
    union result_data data;
};
```

**Pattern Matching**:
```ocaml
(* IR: match x with Ok n -> ... | Error msg -> ... *)
(* Generated: *)
switch (x.tag) {
    case Ok: {
        int n = x.data.Ok_v;
        ...
        break;
    }
    case Error: {
        char* msg = x.data.Error_v;
        ...
        break;
    }
}
```

### Memory Spaces

CUDA supports three memory spaces:

```ocaml
type memspace =
  | Global   (* Default: accessible by all threads *)
  | Shared   (* Block-level: __shared__ qualifier *)
  | Local    (* Thread-private: registers or local memory *)

(* Examples *)
let arr_global = Array.create ~mem:Global 1024    (* Global memory *)
let arr_shared = Array.create ~mem:Shared 256     (* __shared__ float arr_shared[256] *)
```

### Control Flow

All OCaml control structures are supported:

**If/Else**:
```ocaml
(* IR: if cond then stmt1 else stmt2 *)
if (cond) {
    stmt1;
} else {
    stmt2;
}
```

**While Loop**:
```ocaml
(* IR: while cond do body done *)
while (cond) {
    body;
}
```

**For Loop**:
```ocaml
(* IR: for i = 0 to 9 do body done *)
for (int i = 0; i <= 9; i++) {  (* Note: <= for inclusive upper bound *)
    body;
}

(* IR: for i = 9 downto 0 do body done *)
for (int i = 9; i >= 0; i--) {
    body;
}
```

### Synchronization

CUDA barriers and memory fences:

```ocaml
(* Block-level barrier *)
SBarrier → "__syncthreads();"

(* Warp-level barrier (CUDA 9.0+) *)
SWarpBarrier → "__syncwarp();"

(* Memory fence (ensures memory consistency) *)
SMemFence → "__threadfence();"
```

### Pragmas

Compiler hints for optimization:

```ocaml
(* IR: #pragma unroll *)
SPragma (["unroll"], body) →
  "#pragma unroll\n" ^ gen_stmt body

(* Loop unrolling with factor *)
SPragma (["unroll"; "4"], for_loop) →
  "#pragma unroll 4\n" ^ gen_stmt for_loop
```

## Error Handling

The CUDA backend uses the shared `Backend_error` module from `spoc.framework` for structured error handling. This provides consistency across all GPU backends (CUDA, OpenCL, Vulkan, Metal).

### Module Overview

```ocaml
(** CUDA-specific error module *)
module Cuda_error = Backend_error.Make(struct let name = "CUDA" end)

(** Exception type for compatibility *)
exception Cuda_error = Backend_error.Backend_error
```

All error messages are prefixed with the backend name and category:
- `[CUDA Codegen]` - Code generation errors (IR translation)
- `[CUDA Runtime]` - Runtime errors (device, compilation, memory)
- `[CUDA Plugin]` - Plugin errors (library loading, unsupported features)

### Codegen Errors

```ocaml
(* Unknown intrinsic *)
let x = Intrinsics.unknown_func () in ...
→ Error: Unknown_intrinsic { name = "unknown_func" }

(* Type mismatch *)
let x : int = float_value in ...
→ Error: Type_error { expr = "float_value"; expected = "int"; got = "float" }

(* Unsupported construct *)
let native_code = [%native.cuda "..."] in ...  (* requires device context *)
→ Error: Unsupported_construct { construct = "SNative"; reason = "requires device context" }
```

### Runtime Errors

```ocaml
(* No device available *)
let dev = Device.get 0 in ...  (* but no CUDA devices present *)
→ Error: No_device_selected { operation = "get_device" }

(* Compilation failure *)
let ptx = compile_to_ptx "invalid syntax" in ...
→ Error: Compilation_failed { 
     source = "invalid syntax"; 
     log = "error: expected ';' at line 1"
   }

(* Device not found *)
let dev = Device.get 5 in ...  (* only 2 devices available *)
→ Error: Device_not_found { device_id = 5; max_devices = 2 }
```

### Plugin Errors

```ocaml
(* Library not found *)
(* When libcuda.so is missing *)
→ Error: Library_not_found { 
     library = "libcuda.so"; 
     paths = ["/usr/lib"; "/usr/local/lib"]
   }

(* Unsupported source language *)
execute_kernel ~source_lang:"GLSL" ...  (* CUDA only supports CUDA C *)
→ Error: Unsupported_source_lang { lang = "GLSL" }
  Message: "[CUDA Plugin] Source language not supported: GLSL"
```

### Error Recovery

```ocaml
(* Use with_default for safe fallback *)
let device_count = 
  Cuda_error.with_default ~default:0 (fun () ->
    Cuda_api.Device.count ()
  )

(* Convert to Result type *)
let result = Cuda_error.to_result (fun () ->
  compile_and_run_kernel ()
)
match result with
| Ok () -> print_endline "Success"
| Error err -> Printf.eprintf "Error: %s\n" (Cuda_error.to_string err)
```

## API Reference

### Device Management

```ocaml
module Device : sig
  type t
  
  (** Get device by ID (0-indexed) *)
  val get : int -> t
  
  (** Count available CUDA devices *)
  val count : unit -> int
  
  (** Get device name (e.g., "NVIDIA GeForce RTX 3090") *)
  val name : t -> string
  
  (** Get compute capability (e.g., (8, 6) for sm_86) *)
  val compute_capability : t -> int * int
  
  (** Query device properties *)
  val max_threads_per_block : t -> int
  val max_block_dim_x : t -> int
  val max_grid_dim_x : t -> int
  val shared_memory_per_block : t -> int
  val total_constant_memory : t -> int
  val warp_size : t -> int
  val max_registers_per_block : t -> int
  val clock_rate : t -> int
  val memory_clock_rate : t -> int
  val global_memory_bus_width : t -> int
  val l2_cache_size : t -> int
end
```

### Context Management

```ocaml
module Context : sig
  type t
  
  (** Create new context for device *)
  val create : ?flags:int -> Device.t -> t
  
  (** Destroy context and free resources *)
  val destroy : t -> unit
  
  (** Set as current context for calling thread *)
  val set_current : t -> unit
  
  (** Get current context *)
  val get_current : unit -> t option
end
```

### Memory Management

```ocaml
module Memory : sig
  type device_ptr  (* Opaque pointer to GPU memory *)
  
  (** Allocate device memory (bytes) *)
  val alloc : int -> device_ptr
  
  (** Free device memory *)
  val free : device_ptr -> unit
  
  (** Copy host → device *)
  val memcpy_htod : dest:device_ptr -> src:'a ptr -> size:int -> unit
  
  (** Copy device → host *)
  val memcpy_dtoh : dest:'a ptr -> src:device_ptr -> size:int -> unit
  
  (** Copy device → device *)
  val memcpy_dtod : dest:device_ptr -> src:device_ptr -> size:int -> unit
  
  (** Async copies with streams *)
  val memcpy_htod_async : Stream.t -> dest:device_ptr -> src:'a ptr -> size:int -> unit
  val memcpy_dtoh_async : Stream.t -> dest:'a ptr -> src:device_ptr -> size:int -> unit
  
  (** Memory info *)
  val get_info : unit -> int * int  (* free, total in bytes *)
end
```

### Module and Kernel Management

```ocaml
module Module : sig
  type t
  
  (** Load PTX bytecode into module *)
  val load_ptx : string -> t
  
  (** Get kernel function from module *)
  val get_function : t -> string -> Kernel.t
  
  (** Unload module *)
  val unload : t -> unit
end

module Kernel : sig
  type t
  type launch_config = {
    grid_dim : int * int * int;    (* Grid dimensions (blocks) *)
    block_dim : int * int * int;   (* Block dimensions (threads) *)
    shared_mem : int;              (* Shared memory bytes *)
    stream : Stream.t option;      (* Optional stream *)
  }
  
  (** Launch kernel with configuration and arguments *)
  val launch : t -> launch_config -> unit ptr array -> unit
  
  (** Get kernel properties *)
  val max_threads_per_block : t -> int
  val shared_size_bytes : t -> int
  val const_size_bytes : t -> int
  val local_size_bytes : t -> int
  val num_regs : t -> int
end
```

### Stream Management

```ocaml
module Stream : sig
  type t
  
  (** Create new stream *)
  val create : unit -> t
  
  (** Create with priority *)
  val create_with_priority : int -> t
  
  (** Destroy stream *)
  val destroy : t -> unit
  
  (** Wait for all operations in stream to complete *)
  val synchronize : t -> unit
  
  (** Check if stream has completed *)
  val query : t -> bool
end
```

### Event Management

```ocaml
module Event : sig
  type t
  
  (** Create event *)
  val create : unit -> t
  
  (** Record event in stream *)
  val record : t -> Stream.t option -> unit
  
  (** Wait for event to complete *)
  val synchronize : t -> unit
  
  (** Check if event has occurred *)
  val query : t -> bool
  
  (** Elapsed time between two events (milliseconds) *)
  val elapsed_time : t -> t -> float
  
  (** Destroy event *)
  val destroy : t -> unit
end
```

## Usage Examples

### Example 1: Vector Addition

```ocaml
open Sarek

(* Define kernel using Sarek syntax *)
let vector_add = [%kernel
  fun (a : float array) (b : float array) (c : float array) ->
    let tid = Intrinsics.global_idx () in
    if tid < Array.length a then
      c.(tid) <- a.(tid) +. b.(tid)
]

(* Execute on GPU *)
let () =
  (* Create vectors *)
  let n = 1_000_000 in
  let a = Vector.init n (fun i -> float i) in
  let b = Vector.init n (fun i -> float (i * 2)) in
  let c = Vector.create n 0.0 in
  
  (* Execute kernel *)
  Execute.run
    ~grid:(1024, 1, 1)
    ~block:(256, 1, 1)
    vector_add a b c ;
  
  (* Results are in c *)
  Printf.printf "c[0] = %f\n" (Vector.get c 0)
```

### Example 2: Matrix Multiplication

```ocaml
let matmul = [%kernel
  fun (a : float array) (b : float array) (c : float array)
      (m : int) (n : int) (k : int) ->
    let row = Intrinsics.block_idx_y () * Intrinsics.block_dim_y () + 
              Intrinsics.thread_idx_y () in
    let col = Intrinsics.block_idx_x () * Intrinsics.block_dim_x () + 
              Intrinsics.thread_idx_x () in
    
    if row < m && col < n then begin
      let sum = ref 0.0 in
      for i = 0 to k - 1 do
        sum := !sum +. a.(row * k + i) *. b.(i * n + col)
      done ;
      c.(row * n + col) <- !sum
    end
]

let () =
  let m, n, k = 1024, 1024, 1024 in
  let a = Vector.create (m * k) 1.0 in
  let b = Vector.create (k * n) 2.0 in
  let c = Vector.create (m * n) 0.0 in
  
  Execute.run
    ~grid:((n + 15) / 16, (m + 15) / 16, 1)
    ~block:(16, 16, 1)
    matmul a b c m n k
```

### Example 3: Shared Memory and Barriers

```ocaml
let reduce_sum = [%kernel
  fun (input : float array) (output : float array) (n : int) ->
    (* Allocate shared memory *)
    let shared = Array.create ~mem:Shared 256 0.0 in
    
    let tid = Intrinsics.thread_idx_x () in
    let gid = Intrinsics.global_idx () in
    
    (* Load data into shared memory *)
    shared.(tid) <- if gid < n then input.(gid) else 0.0 ;
    
    (* Synchronize threads *)
    Intrinsics.barrier () ;
    
    (* Reduction in shared memory *)
    let stride = ref 128 in
    while !stride > 0 do
      if tid < !stride then
        shared.(tid) <- shared.(tid) +. shared.(tid + !stride) ;
      Intrinsics.barrier () ;
      stride := !stride / 2
    done ;
    
    (* Write result *)
    if tid = 0 then
      output.(Intrinsics.block_idx_x ()) <- shared.(0)
]
```

### Example 4: Atomic Operations

```ocaml
let histogram = [%kernel
  fun (input : int array) (bins : int array) (n : int) (num_bins : int) ->
    let tid = Intrinsics.global_idx () in
    
    if tid < n then begin
      let value = input.(tid) in
      let bin = value mod num_bins in
      
      (* Atomic increment of histogram bin *)
      Intrinsics.atomic_add bins.(bin) 1 |> ignore
    end
]
```

### Example 5: Device Query

```ocaml
open Sarek_cuda

let () =
  (* Check CUDA availability *)
  let count = Cuda_api.Device.count () in
  Printf.printf "Found %d CUDA device(s)\n" count ;
  
  (* Query each device *)
  for i = 0 to count - 1 do
    let dev = Cuda_api.Device.get i in
    Printf.printf "\nDevice %d: %s\n" i (Cuda_api.Device.name dev) ;
    
    let major, minor = Cuda_api.Device.compute_capability dev in
    Printf.printf "  Compute Capability: %d.%d\n" major minor ;
    Printf.printf "  Max Threads/Block: %d\n" 
      (Cuda_api.Device.max_threads_per_block dev) ;
    Printf.printf "  Shared Memory/Block: %d bytes\n"
      (Cuda_api.Device.shared_memory_per_block dev) ;
    Printf.printf "  Warp Size: %d\n" 
      (Cuda_api.Device.warp_size dev)
  done
```

## CUDA Intrinsics

The CUDA backend supports 57 intrinsics organized into categories:

### Thread Identification

```ocaml
(* Thread indices within block *)
thread_idx_x : unit -> int   (* threadIdx.x *)
thread_idx_y : unit -> int   (* threadIdx.y *)
thread_idx_z : unit -> int   (* threadIdx.z *)

(* Block indices within grid *)
block_idx_x : unit -> int    (* blockIdx.x *)
block_idx_y : unit -> int    (* blockIdx.y *)
block_idx_z : unit -> int    (* blockIdx.z *)

(* Block dimensions *)
block_dim_x : unit -> int    (* blockDim.x *)
block_dim_y : unit -> int    (* blockDim.y *)
block_dim_z : unit -> int    (* blockDim.z *)

(* Grid dimensions *)
grid_dim_x : unit -> int     (* gridDim.x *)
grid_dim_y : unit -> int     (* gridDim.y *)
grid_dim_z : unit -> int     (* gridDim.z *)

(* Global thread ID (1D) *)
global_idx : unit -> int     (* threadIdx.x + blockIdx.x * blockDim.x *)
```

### Math Functions

```ocaml
(* Trigonometric *)
sin   : float -> float
cos   : float -> float
tan   : float -> float
asin  : float -> float
acos  : float -> float
atan  : float -> float
atan2 : float -> float -> float

(* Hyperbolic *)
sinh  : float -> float
cosh  : float -> float
tanh  : float -> float

(* Exponential/logarithmic *)
exp   : float -> float
exp2  : float -> float    (* 2^x *)
log   : float -> float    (* Natural log *)
log2  : float -> float    (* Base-2 log *)
log10 : float -> float    (* Base-10 log *)

(* Power/root *)
pow   : float -> float -> float
sqrt  : float -> float
rsqrt : float -> float    (* 1/sqrt(x) - faster than division *)
cbrt  : float -> float    (* Cube root *)

(* Rounding *)
floor : float -> float
ceil  : float -> float
round : float -> float
trunc : float -> float

(* Other *)
fabs  : float -> float    (* Absolute value *)
fma   : float -> float -> float -> float  (* Fused multiply-add: a*b+c *)
min   : 'a -> 'a -> 'a
max   : 'a -> 'a -> 'a
```

### Atomic Operations

```ocaml
(* Atomic arithmetic *)
atomic_add : 'a array -> int -> 'a -> 'a    (* Atomic add, returns old value *)
atomic_sub : 'a array -> int -> 'a -> 'a    (* Atomic subtract *)
atomic_exch : 'a array -> int -> 'a -> 'a   (* Atomic exchange *)
atomic_min : 'a array -> int -> 'a -> 'a    (* Atomic minimum *)
atomic_max : 'a array -> int -> 'a -> 'a    (* Atomic maximum *)
atomic_inc : int array -> int -> int -> int (* Atomic increment with wrap *)
atomic_dec : int array -> int -> int -> int (* Atomic decrement with wrap *)

(* Atomic bitwise *)
atomic_and : int array -> int -> int -> int (* Atomic AND *)
atomic_or  : int array -> int -> int -> int (* Atomic OR *)
atomic_xor : int array -> int -> int -> int (* Atomic XOR *)

(* Compare-and-swap *)
atomic_cas : 'a array -> int -> 'a -> 'a -> 'a
  (* atomicCAS(addr, compare, val): if *addr == compare then *addr = val *)
```

### Synchronization

```ocaml
(* Block-level barrier *)
barrier : unit -> unit           (* __syncthreads() *)

(* Warp-level barrier (CUDA 9.0+) *)
warp_barrier : unit -> unit      (* __syncwarp() *)

(* Memory fence *)
mem_fence : unit -> unit         (* __threadfence() *)
mem_fence_block : unit -> unit   (* __threadfence_block() *)
mem_fence_system : unit -> unit  (* __threadfence_system() *)
```

### Warp Operations (CUDA 9.0+)

```ocaml
(* Warp shuffle *)
warp_shuffle : 'a -> int -> 'a              (* __shfl_sync *)
warp_shuffle_up : 'a -> int -> 'a           (* __shfl_up_sync *)
warp_shuffle_down : 'a -> int -> 'a         (* __shfl_down_sync *)
warp_shuffle_xor : 'a -> int -> 'a          (* __shfl_xor_sync *)

(* Warp vote *)
warp_all : bool -> bool                     (* __all_sync *)
warp_any : bool -> bool                     (* __any_sync *)
warp_ballot : bool -> int                   (* __ballot_sync *)
```

## Testing

The CUDA backend includes comprehensive unit tests:

### Test Suite Overview

```bash
# Run all tests
dune test sarek-cuda/test

# Run specific test
_build/default/sarek-cuda/test/test_cuda_error.exe
_build/default/sarek-cuda/test/test_sarek_ir_cuda.exe
```

### test_cuda_error.ml (6 tests)

**Error Construction and Formatting**:
- Codegen errors (unsupported_construct, type_error, invalid_memory_space)
- Runtime errors (no_device_selected, compilation_failed, device_not_found)
- Plugin errors (unsupported_source_lang, library_not_found)

**Error Utilities**:
- `with_default`: Safe fallback on error
- `to_result`: Convert exceptions to Result type
- Error equality and comparison

### test_sarek_ir_cuda.ml (13 tests)

**Expression Generation**:
- Literals: integers, floats, booleans (CUDA uses 0/1 for bool)
- Variables and binary operations (+, *, /, etc.)

**Statement Generation**:
- Assignment statements
- Control flow: if/else, while, for (upto/downto)
- Special constructs: barriers, pragmas, blocks
- Let bindings: immutable and mutable

**Test Coverage**:
```
Test Results:
  ✓ expressions (2 tests)
    - basic literals
    - operations
  ✓ statements (11 tests)
    - basics
    - assignment
    - if statement
    - while loop
    - for loop
    - return
    - barriers
    - let binding
    - let mut
    - block
    - pragma

Total: 19 tests, 100% passing
```

## Installation

### Prerequisites

- OCaml 5.4.0+
- NVIDIA GPU with CUDA support
- CUDA Toolkit 11.0+ (libcuda.so, libnvrtc.so)
- CUDA drivers installed

### Build from Source

```bash
# Clone repository
git clone https://github.com/yourusername/SPOC.git
cd SPOC

# Install dependencies
opam install . --deps-only

# Build CUDA backend
LIBRARY_PATH=/opt/cuda/targets/x86_64-linux/lib:$LIBRARY_PATH \
  dune build sarek-cuda

# Run tests
dune test sarek-cuda/test
```

### Environment Variables

```bash
# Required: CUDA library path
export LD_LIBRARY_PATH=/opt/cuda/lib64:$LD_LIBRARY_PATH

# Optional: Debug logging
export SAREK_DEBUG=kernel    # Log generated kernel source
export SAREK_DEBUG=all       # Log everything
```

### Verify Installation

```ocaml
(* Check CUDA availability *)
open Sarek_cuda

let () =
  match Cuda_api.Device.count () with
  | 0 -> print_endline "No CUDA devices found"
  | n -> Printf.printf "Found %d CUDA device(s)\n" n
```

## Design Principles

### Type Safety

Uses GADTs and phantom types for typed device pointers and kernel arguments:
```ocaml
type 'a device_ptr = private int64
type _ kernel_arg =
  | Scalar : ('a, 'b) scalar_kind * 'a -> ('a, 'b) kernel_arg
  | Vector : ('a, 'b) Vector.t -> ('a, 'b) kernel_arg
```

### Pure OCaml FFI

All CUDA bindings use `ctypes-foreign` for FFI without C stubs:
```ocaml
let cuDeviceGet = 
  Foreign.foreign "cuDeviceGet"
    (int @-> ptr cuda_device @-> returning cuda_result)
```

### Structured Errors

Uses inline record types for structured error handling:
```ocaml
(* Good: Structured error *)
type codegen_error =
  | Unknown_intrinsic of { name : string }

(* Bad: String-based error *)
failwith "unknown intrinsic: foo"
```

### 4. Separation of Concerns

- **Cuda_bindings.ml**: FFI bindings only
- **Cuda_api.ml**: Safe wrappers + error handling
- **Sarek_ir_cuda.ml**: Code generation logic
- **Cuda_plugin.ml**: Framework integration

### 5. Code Organization

**Small, focused functions**:
- Before: `gen_stmt` (210 lines)
- After: `gen_stmt` (140 lines) + 3 helpers (gen_record_assign, gen_match_case, gen_array_decl)

### 6. Testability

**Unit tests for all major components**:
- Error handling (6 tests)
- Code generation (13 tests)
- Total coverage: 19 tests, 100% passing

## Performance Considerations

### Memory Coalescing

Access global memory in coalesced patterns when possible:

```ocaml
(* Good: Coalesced access *)
let tid = Intrinsics.global_idx () in
let value = input.(tid) in  (* Each thread accesses consecutive memory *)

(* Bad: Strided access *)
let tid = Intrinsics.global_idx () in
let value = input.(tid * 32) in  (* Poor memory coalescing *)
```

### Shared Memory

Use shared memory for frequently accessed data:

```ocaml
(* Allocate shared memory *)
let shared = Array.create ~mem:Shared 256 0.0 in

(* Load from global → shared *)
shared.(tid) <- input.(gid) ;
Intrinsics.barrier () ;

(* Fast access from shared memory *)
let value = shared.(tid) in
```

### Warp Divergence

Minimize branching within warps:

```ocaml
(* Good: All threads in warp take same path *)
if tid < n then begin
  (* All threads execute this *)
  arr.(tid) <- arr.(tid) * 2.0
end

(* Bad: Divergent branches *)
if tid mod 2 = 0 then
  (* Half the threads execute this *)
  arr.(tid) <- arr.(tid) * 2.0
else
  (* Other half execute this - warp divergence! *)
  arr.(tid) <- arr.(tid) * 3.0
```

### Occupancy

Maximize GPU occupancy by tuning block size:

```ocaml
(* Check device limits *)
let max_threads = Cuda_api.Device.max_threads_per_block dev in
let shared_mem = Cuda_api.Device.shared_memory_per_block dev in

(* Common block sizes: 128, 256, 512 threads *)
let block_size = min 256 max_threads in
let grid_size = (n + block_size - 1) / block_size in

Execute.run
  ~grid:(grid_size, 1, 1)
  ~block:(block_size, 1, 1)
  kernel ...
```

## Troubleshooting

### Common Issues

**1. Library not found**
```
Error: Library_not_found { library = "libcuda.so"; paths = [...] }
```
Solution: Install NVIDIA drivers and set `LD_LIBRARY_PATH`

**2. No devices found**
```
Error: No_device_selected { operation = "get_device" }
```
Solution: Check `nvidia-smi` output, verify GPU is detected

**3. Compilation failed**
```
Error: Compilation_failed { source = "..."; log = "error: ..." }
```
Solution: Enable debug logging with `SAREK_DEBUG=kernel` to see generated source

**4. Out of memory**
```
Error: Device_operation_failed { operation = "cuMemAlloc"; ... }
```
Solution: Reduce problem size or use memory pools

### Debug Logging

Enable comprehensive logging:

```bash
# Log kernel source code
export SAREK_DEBUG=kernel

# Log all CUDA operations
export SAREK_DEBUG=all

# Log specific components
export SAREK_DEBUG=kernel,memory,device
```

## Related Packages

- **sarek.framework**: Plugin system and backend registration
- **spoc.ir**: Sarek IR types (AST for kernels)
- **sarek.runtime**: High-level execution interface
- **sarek-opencl**: OpenCL backend (alternative to CUDA)
- **sarek-vulkan**: Vulkan/GLSL backend

## License

MIT License - See LICENSE file for details

## Authors

Part of the SPOC/Sarek project

## Version History

- **1.0.0** (2026-01): Production release
  - Structured error handling (Phase 1)
  - Code organization improvements (Phase 2)
  - Comprehensive test suite (Phase 3)
  - Complete documentation (Phase 4)
