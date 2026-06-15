# sarek.framework - Plugin System for GPU Backends

**Package**: `sarek.framework`  
**Library**: `sarek.framework.registry`  
**Test Coverage**: 48% (framework) | 89-96% (tests)  
**Lines of Code**: 328 (core) + 476 (tests)

The framework package provides the plugin system that enables SPOC/Sarek to support multiple GPU backends (CUDA, OpenCL, Vulkan, Native CPU, Interpreter) through a unified interface. It handles backend registration, intrinsic function management, and persistent kernel caching.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Core Modules](#core-modules)
- [Backend Registration](#backend-registration)
- [Intrinsic Registry](#intrinsic-registry)
- [Kernel Cache](#kernel-cache)
- [Error Handling](#error-handling)
- [API Reference](#api-reference)
- [Usage Examples](#usage-examples)
- [Testing](#testing)
- [Design Principles](#design-principles)

## Overview

The framework package sits at the foundation of SPOC/Sarek's multi-backend architecture:

```
┌─────────────────────────────────────────┐
│          User Code / Sarek PPX          │
├─────────────────────────────────────────┤
│     Sarek Runtime / Execute.ml          │
├─────────────────────────────────────────┤
│      Framework Registry (this)          │  ← Plugin discovery & selection
├─────────────────────────────────────────┤
│  ┌──────┬────────┬────────┬────────┐   │
│  │ CUDA │ OpenCL │ Vulkan │ Native │   │  ← Backend plugins
│  └──────┴────────┴────────┴────────┘   │
└─────────────────────────────────────────┘
```

### Key Features

- **Plugin Architecture**: Backends register themselves at runtime via first-class modules
- **Priority-Based Selection**: Choose best available backend based on priorities (e.g., CUDA=100, OpenCL=80, Native=50)
- **Intrinsic Management**: 57 standard GPU intrinsics with backend-specific implementations
- **Persistent Caching**: On-disk kernel cache with statistics tracking (hits/misses/puts)
- **Type Safety**: No `Obj.t`, structured error types, clean abstraction boundaries
- **Debug Logging**: Built-in logging controlled via `SAREK_DEBUG` environment variable

## Architecture

### Module Structure

```
sarek/framework/
├── Framework_registry.ml      (88 lines)  - Backend plugin registration
├── Intrinsic_registry.ml     (195 lines)  - GPU intrinsic management
├── Framework_cache.ml        (107 lines)  - Persistent kernel cache
└── Framework_error.ml         (61 lines)  - Structured error types
```

### Plugin Interface

Backends implement the `Framework_sig.BACKEND` signature:

```ocaml
module type BACKEND = sig
  val name : string
  val version : int * int * int
  val is_available : unit -> bool
  val execution_model : execution_model  (* JIT | Direct | Custom *)
  
  module Device : sig ... end      (* Device management *)
  module Stream : sig ... end      (* Stream/queue operations *)
  module Memory : sig ... end      (* Memory allocation/transfers *)
  module Kernel : sig ... end      (* Kernel compilation/launch *)
  module Event  : sig ... end      (* Event synchronization *)
  
  module Intrinsics : INTRINSIC_REGISTRY  (* Backend-specific intrinsics *)
  
  val generate_source : ?block:dims -> Sarek_ir_types.kernel -> string option
  val execute_direct : ... -> unit
end
```

## Core Modules

### 1. Framework_registry.ml

Central registry for backend plugins. Backends register themselves and are selected based on:
- **Availability**: `is_available()` check (GPU drivers, hardware present)
- **Priority**: Higher priority backends are preferred (default: 50)
- **Name**: Direct lookup by backend name

**Key Functions**:
- `register_backend ~priority (module B : BACKEND)` - Register a backend plugin
- `find_backend : string -> (module BACKEND) option` - Find backend by name
- `best_backend : unit -> (module BACKEND) option` - Get highest-priority available backend
- `available_backends : unit -> (module BACKEND) list` - List all available backends
- `priority : string -> int` - Get backend priority

**Priority Guidelines**:
- CUDA: 100 (best performance on NVIDIA GPUs)
- OpenCL: 80 (good cross-platform support)
- Vulkan: 70 (modern graphics-compute hybrid)
- Native: 50 (CPU fallback, always available)
- Interpreter: 30 (debugging/testing)

### 2. Intrinsic_registry.ml

Manages GPU intrinsic functions (built-ins like `thread_id_x`, `atomic_add`, `sin`). Each backend registers its own implementations with appropriate code generation.

**Intrinsic Types**:
- **Uniform**: Safe in all contexts (e.g., `sin`, `sqrt`)
- **Sync**: Requires uniform execution (e.g., `block_barrier`, `memory_fence`)
- **Divergent**: Thread-divergent operations (e.g., `thread_id_x`, `warp_ballot`)

**Standard Intrinsics** (57 total):

*Thread Intrinsics (14)*:
```
thread_id_x, thread_id_y, thread_id_z    - Thread indices within block
block_id_x, block_id_y, block_id_z       - Block indices within grid
block_dim_x, block_dim_y, block_dim_z    - Block dimensions
grid_dim_x, grid_dim_y, grid_dim_z       - Grid dimensions
global_id_x, global_id_y, global_id_z    - Global thread indices
warp_id                                   - Warp/wavefront ID
```

*Synchronization (4)*:
```
block_barrier      - Block-level barrier (__syncthreads)
warp_barrier       - Warp-level barrier
memory_fence       - Memory fence
memory_fence_block - Block-level memory fence
```

*Atomic Operations (9)*:
```
atomic_add, atomic_sub, atomic_min, atomic_max
atomic_and, atomic_or, atomic_xor
atomic_exch, atomic_cas
```

*Math Functions (30)*:
```
sin, cos, tan, asin, acos, atan, atan2
sqrt, rsqrt, cbrt, exp, exp2, exp10
log, log2, log10, pow
floor, ceil, trunc, round, rint
abs, fabs, fmin, fmax, fma
copysign, ldexp, frexp
```

**Key Functions**:
- `Make()` - Create per-backend intrinsic registry (functor)
- `register : string -> intrinsic_impl -> unit` - Register an intrinsic
- `find : string -> intrinsic_impl option` - Look up intrinsic by name
- `list_all : unit -> string list` - List all registered intrinsics
- `Global.register : backend:string -> string -> intrinsic_impl -> unit` - Register in global registry
- `Global.find : backend:string -> string -> intrinsic_impl option` - Find across all backends

**Convergence Safety**:
```ocaml
let is_safe_in_divergent_flow impl =
  match impl.intr_convergence with
  | Uniform -> true    (* Always safe *)
  | Sync -> false      (* Requires uniform control flow *)
  | Divergent -> true  (* Designed for divergent contexts *)

let requires_uniform_execution impl = impl.intr_convergence = Sync
```

### 3. Framework_cache.ml

Persistent on-disk cache for compiled kernels (PTX, SPIR-V, etc.). Follows XDG Base Directory specification: `~/.cache/spoc/`.

**Statistics Tracking**:
```ocaml
type stats = {
  hits : int;        (* Cache hits - kernel found *)
  misses : int;      (* Cache misses - kernel not found *)
  puts : int;        (* Kernels stored *)
  errors : int;      (* I/O or other errors *)
}

let hit_rate () : float =
  let stats = get_stats () in
  if stats.hits + stats.misses = 0 then 0.0
  else float_of_int stats.hits /. float_of_int (stats.hits + stats.misses) *. 100.0
```

**Key Functions**:
- `get_cache_dir : unit -> string` - Get cache directory path
- `compute_key : dev_name:string -> driver_version:string -> source:string -> string` - Compute cache key (MD5)
- `get : key:string -> string option` - Retrieve cached kernel
- `put : key:string -> data:string -> unit` - Store compiled kernel
- `get_stats : unit -> stats` - Get cache statistics
- `reset_stats : unit -> unit` - Reset statistics counters
- `print_stats : unit -> unit` - Print formatted statistics to stdout

**Cache Key Computation**:
```ocaml
(* MD5 hash of: device_name + driver_version + source_code *)
let key = compute_key
  ~dev_name:"NVIDIA GeForce RTX 3090"
  ~driver_version:"535.183.01"
  ~source:"__global__ void kernel() { ... }"
(* → "a3f2c8b1d4e5f6a7b8c9d0e1f2a3b4c5" *)
```

### 4. Framework_error.ml

Structured error types for all framework operations:

```ocaml
type framework_error =
  | Backend_not_found of { name : string }
  | No_backends_available of { reason : string }
  | Backend_unavailable of { name : string; reason : string }
  | Plugin_registration_failed of { name : string; reason : string }
  | Intrinsic_not_found of { name : string; backend : string option }
  | Intrinsic_registration_failed of { name : string; reason : string }
  | Cache_error of { operation : string; reason : string }
```

**Key Functions**:
- `raise_error : framework_error -> 'a` - Raise framework error as exception
- `to_string : framework_error -> string` - Convert error to human-readable message
- `print_error : framework_error -> unit` - Print error to stderr
- `with_default : default:'a -> (unit -> 'a) -> 'a` - Execute with default fallback

## Backend Registration

### How Backends Register

Backends typically register themselves in their module initialization:

```ocaml
(* plugins/cuda/Cuda_backend.ml *)
module Cuda_backend : BACKEND = struct
  let name = "CUDA"
  let version = (12, 0, 0)
  let is_available () = 
    try Cuda_driver.init (); true
    with _ -> false
  
  let execution_model = JIT
  
  module Intrinsics = Intrinsic_registry.Make()
  
  (* Register CUDA-specific intrinsics *)
  let () =
    Intrinsics.register "thread_id_x"
      (make_divergent_intrinsic ~name:"thread_id_x" ~codegen:"threadIdx.x");
    Intrinsics.register "block_barrier"
      (make_sync_intrinsic ~name:"block_barrier" ~codegen:"__syncthreads()");
    (* ... 55 more intrinsics ... *)
  
  (* ... Device, Stream, Memory, Kernel, Event modules ... *)
end

(* Auto-register with high priority *)
let () =
  Framework_registry.register_backend ~priority:100 (module Cuda_backend)
```

### Backend Selection Flow

```
1. User Code: let dev = Device.default ()
               ↓
2. Execute.ml: Framework_registry.best_backend ()
               ↓
3. Registry checks each backend:
   - CUDA.is_available() → checks for CUDA drivers/GPU
   - OpenCL.is_available() → checks for OpenCL runtime
   - Native.is_available() → always true (CPU fallback)
               ↓
4. Returns highest-priority available backend:
   - CUDA (priority 100) if available
   - OpenCL (priority 80) if CUDA unavailable
   - Native (priority 50) as fallback
```

### Execution Models

Backends declare their execution model:

- **JIT (Just-In-Time)**: Generate source code, compile with GPU driver
  - CUDA: Generate PTX, compile with `nvrtc`
  - OpenCL: Generate OpenCL C, compile with `clBuildProgram`
  - Vulkan: Generate GLSL, compile with `glslangValidator` → SPIR-V
  
- **Direct**: Execute pre-compiled OCaml functions directly
  - Native: Parallel CPU execution of OCaml code
  
- **Custom**: Full control over execution
  - Interpreter: Interpret Sarek IR directly

## Intrinsic Registry

### Creating a Backend Registry

```ocaml
module My_backend = struct
  module Intrinsics = Intrinsic_registry.Make()
  
  let () =
    (* Register thread intrinsics *)
    Intrinsics.register "thread_id_x"
      (make_divergent_intrinsic ~name:"thread_id_x" ~codegen:"get_local_id(0)");
    
    (* Register sync intrinsics *)
    Intrinsics.register "block_barrier"
      (make_sync_intrinsic ~name:"block_barrier" ~codegen:"barrier(CLK_LOCAL_MEM_FENCE)");
    
    (* Register math intrinsics *)
    Intrinsics.register "sin"
      (make_simple_intrinsic ~name:"sin" ~codegen:"sin");
end
```

### Intrinsic Types

```ocaml
(* Simple/Uniform - safe in all contexts *)
let sin_impl = make_simple_intrinsic ~name:"sin" ~codegen:"sin"
(* convergence = Uniform *)

(* Sync - requires uniform control flow *)
let barrier_impl = make_sync_intrinsic ~name:"block_barrier" ~codegen:"__syncthreads()"
(* convergence = Sync *)

(* Divergent - thread-dependent *)
let tid_impl = make_divergent_intrinsic ~name:"thread_id_x" ~codegen:"threadIdx.x"
(* convergence = Divergent *)
```

### Global Registry

The global registry tracks intrinsics across all backends:

```ocaml
(* Register backend-specific implementation *)
Intrinsic_registry.Global.register 
  ~backend:"CUDA" 
  "thread_id_x"
  cuda_thread_id_impl;

Intrinsic_registry.Global.register
  ~backend:"OpenCL"
  "thread_id_x"
  opencl_thread_id_impl;

(* Find for specific backend *)
let impl = Intrinsic_registry.Global.find ~backend:"CUDA" "thread_id_x"

(* Find all backends supporting an intrinsic *)
let backends = Intrinsic_registry.Global.backends_for "thread_id_x"
(* → ["CUDA"; "OpenCL"; "Vulkan"; "Native"] *)
```

## Kernel Cache

### Cache Usage

```ocaml
(* Compute cache key *)
let key = Framework_cache.compute_key
  ~dev_name:(Device.name dev)
  ~driver_version:"535.183.01"
  ~source:ptx_source

(* Try to retrieve from cache *)
match Framework_cache.get ~key with
| Some cached_binary ->
    (* Cache hit! Use cached kernel *)
    Kernel.load_from_binary dev cached_binary
| None ->
    (* Cache miss - compile kernel *)
    let binary = Compiler.compile source in
    (* Store in cache for next time *)
    Framework_cache.put ~key ~data:binary;
    Kernel.load_from_binary dev binary
```

### Cache Statistics

```ocaml
(* Get current statistics *)
let stats = Framework_cache.get_stats ()
Printf.printf "Hits: %d, Misses: %d, Hit rate: %.1f%%\n"
  stats.hits stats.misses (Framework_cache.hit_rate ());

(* Print formatted statistics *)
Framework_cache.print_stats ();
(* Output:
   Cache Statistics:
     Hits:   142
     Misses: 18
     Puts:   18
     Errors: 0
     Hit Rate: 88.8%
*)

(* Reset statistics *)
Framework_cache.reset_stats ()
```

### Cache Location

Cache follows XDG Base Directory specification:

- Linux: `~/.cache/spoc/`
- macOS: `~/Library/Caches/spoc/`
- Windows: `%LOCALAPPDATA%\spoc\cache\`

Cache files are named by MD5 hash of cache key.

## Error Handling

### Using Structured Errors

```ocaml
(* Raise framework error *)
Framework_error.raise_error
  (Backend_not_found { name = "CUDA" })

(* Convert to string *)
let msg = Framework_error.to_string
  (Cache_error { operation = "read"; reason = "Permission denied" })
(* → "Cache error during read: Permission denied" *)

(* Execute with default fallback *)
let backend = Framework_error.with_default
  ~default:(module Native_backend)
  (fun () -> Framework_registry.best_backend () |> Option.get)
```

### Error Types

```ocaml
(* Backend not found *)
Backend_not_found { name = "CUDA" }

(* No available backends *)
No_backends_available { reason = "No GPU drivers found" }

(* Backend exists but unavailable *)
Backend_unavailable { 
  name = "CUDA"; 
  reason = "No CUDA-capable GPU detected" 
}

(* Plugin registration failed *)
Plugin_registration_failed {
  name = "MyBackend";
  reason = "Duplicate backend name"
}

(* Intrinsic not found *)
Intrinsic_not_found {
  name = "my_intrinsic";
  backend = Some "CUDA"
}

(* Cache operation failed *)
Cache_error {
  operation = "write";
  reason = "Disk full"
}
```

## API Reference

### Framework_registry

```ocaml
val register_backend : ?priority:int -> (module BACKEND) -> unit
  (* Register a backend plugin. Default priority: 50 *)

val find_backend : string -> (module BACKEND) option
  (* Find backend by name. Returns None if not found *)

val best_backend : unit -> (module BACKEND) option
  (* Get highest-priority available backend *)

val available_backends : unit -> (module BACKEND) list
  (* List all available (is_available = true) backends *)

val all_backend_names : unit -> string list
  (* List names of all registered backends *)

val priority : string -> int
  (* Get priority of backend. Returns 50 if not found *)
```

### Intrinsic_registry

```ocaml
module Make () : INTRINSIC_REGISTRY
  (* Create per-backend intrinsic registry *)

module type INTRINSIC_REGISTRY = sig
  type intrinsic_impl
  
  val register : string -> intrinsic_impl -> unit
  val find : string -> intrinsic_impl option
  val list_all : unit -> string list
end

(* Intrinsic constructors *)
val make_simple_intrinsic : name:string -> codegen:string -> intrinsic_impl
val make_sync_intrinsic : name:string -> codegen:string -> intrinsic_impl
val make_divergent_intrinsic : name:string -> codegen:string -> intrinsic_impl

(* Convergence checks *)
val is_safe_in_divergent_flow : intrinsic_impl -> bool
val requires_uniform_execution : intrinsic_impl -> bool

(* Global registry *)
module Global : sig
  val register : backend:string -> string -> intrinsic_impl -> unit
  val find : backend:string -> string -> intrinsic_impl option
  val find_all : string -> (string * intrinsic_impl) list
  val backends_for : string -> string list
  val list_all : unit -> string list
end

(* Standard intrinsic groups *)
module Thread_intrinsics : sig
  val is_thread_intrinsic : string -> bool
  (* thread_id_x, block_id_x, global_id_x, etc. *)
end

module Sync_intrinsics : sig
  val is_sync_intrinsic : string -> bool
  (* block_barrier, warp_barrier, memory_fence *)
end

module Atomic_intrinsics : sig
  val is_atomic_intrinsic : string -> bool
  (* atomic_add, atomic_cas, etc. *)
end

module Math_intrinsics : sig
  val is_math_intrinsic : string -> bool
  (* sin, cos, sqrt, exp, etc. *)
end
```

### Framework_cache

```ocaml
type stats = {
  hits : int;
  misses : int;
  puts : int;
  errors : int;
}

val get_cache_dir : unit -> string
  (* Get cache directory path *)

val compute_key : dev_name:string -> driver_version:string -> source:string -> string
  (* Compute cache key (MD5 hash) *)

val get : key:string -> string option
  (* Retrieve cached kernel. Returns None on miss *)

val put : key:string -> data:string -> unit
  (* Store compiled kernel in cache *)

val get_stats : unit -> stats
  (* Get cache statistics *)

val reset_stats : unit -> unit
  (* Reset statistics counters to zero *)

val hit_rate : unit -> float
  (* Calculate hit rate: hits / (hits + misses) * 100 *)

val print_stats : unit -> unit
  (* Print formatted statistics to stdout *)
```

### Framework_error

```ocaml
type framework_error =
  | Backend_not_found of { name : string }
  | No_backends_available of { reason : string }
  | Backend_unavailable of { name : string; reason : string }
  | Plugin_registration_failed of { name : string; reason : string }
  | Intrinsic_not_found of { name : string; backend : string option }
  | Intrinsic_registration_failed of { name : string; reason : string }
  | Cache_error of { operation : string; reason : string }

exception Framework_error of framework_error

val raise_error : framework_error -> 'a
val to_string : framework_error -> string
val print_error : framework_error -> unit
val with_default : default:'a -> (unit -> 'a) -> 'a
```

## Usage Examples

### Example 1: Basic Backend Registration

```ocaml
module My_backend : BACKEND = struct
  let name = "MyBackend"
  let version = (1, 0, 0)
  let is_available () = true  (* Always available *)
  let execution_model = Custom
  
  module Intrinsics = Intrinsic_registry.Make()
  let () =
    Intrinsics.register "test_intrinsic"
      (make_simple_intrinsic ~name:"test" ~codegen:"test()");
  
  (* ... implement all required modules ... *)
end

(* Register with priority *)
let () =
  Framework_registry.register_backend ~priority:75 (module My_backend)
```

### Example 2: Backend Selection

```ocaml
(* Get best available backend *)
let backend = match Framework_registry.best_backend () with
  | Some b -> b
  | None -> failwith "No backends available"

(* Find specific backend *)
let cuda = match Framework_registry.find_backend "CUDA" with
  | Some (module B) -> B.name (* "CUDA" *)
  | None -> "CUDA not found"

(* List all available backends *)
let available = Framework_registry.available_backends ()
List.iter (fun (module B : BACKEND) ->
  Printf.printf "%s (priority: %d)\n" B.name
    (Framework_registry.priority B.name)
) available
```

### Example 3: Intrinsic Management

```ocaml
(* Create backend-specific registry *)
module Intrinsics = Intrinsic_registry.Make()

(* Register intrinsics *)
let () =
  Intrinsics.register "thread_id_x"
    (make_divergent_intrinsic ~name:"thread_id_x" ~codegen:"threadIdx.x");
  
  Intrinsics.register "sin"
    (make_simple_intrinsic ~name:"sin" ~codegen:"sin");

(* Look up intrinsic *)
match Intrinsics.find "thread_id_x" with
| Some impl -> Printf.printf "Codegen: %s\n" impl.intr_codegen
| None -> Printf.printf "Not found\n"

(* List all intrinsics *)
let all = Intrinsics.list_all ()
Printf.printf "Registered: %s\n" (String.concat ", " all)
```

### Example 4: Kernel Caching

```ocaml
(* Compile with caching *)
let compile_cached dev source =
  let key = Framework_cache.compute_key
    ~dev_name:(Device.name dev)
    ~driver_version:"12.0"
    ~source in
  
  match Framework_cache.get ~key with
  | Some binary ->
      Printf.printf "Cache hit!\n";
      binary
  | None ->
      Printf.printf "Cache miss - compiling...\n";
      let binary = expensive_compilation source in
      Framework_cache.put ~key ~data:binary;
      binary

(* Monitor cache performance *)
let () =
  Printf.printf "Hit rate: %.1f%%\n" (Framework_cache.hit_rate ());
  Framework_cache.print_stats ()
```

### Example 5: Error Handling

```ocaml
(* Safe backend lookup with fallback *)
let get_backend_safe name =
  Framework_error.with_default
    ~default:(module Native_backend : BACKEND)
    (fun () ->
      match Framework_registry.find_backend name with
      | Some b -> b
      | None -> Framework_error.raise_error
          (Backend_not_found { name })
    )

(* Handle specific errors *)
try
  let backend = Framework_registry.best_backend ()
    |> Option.get
in
  (* Use backend *)
  ()
with
| Framework_error.Framework_error err ->
    Printf.eprintf "Framework error: %s\n"
      (Framework_error.to_string err);
    exit 1
```

## Testing

The framework package has comprehensive test coverage:

### Test Structure

```
sarek/framework/test/
├── test_framework_registry.ml     (8 tests)   - Registry operations
├── test_intrinsic_registry.ml    (17 tests)   - Intrinsic management
├── test_framework_cache.ml       (12 tests)   - Cache operations
├── test_framework_integration.ml (16 tests)   - Integration tests
└── dummy_backend.ml                           - Test backend implementation
```

### Running Tests

```bash
# Run all framework tests
dune runtest sarek/framework/test

# Run specific test suite
dune exec sarek/framework/test/test_framework_registry.exe
dune exec sarek/framework/test/test_intrinsic_registry.exe
dune exec sarek/framework/test/test_framework_cache.exe
dune exec sarek/framework/test/test_framework_integration.exe

# Run with coverage
BISECT_FILE=_coverage/framework dune runtest sarek/framework/test
bisect-ppx-report html
```

### Test Results

- **53 tests total**: 37 unit + 16 integration
- **All tests passing** in <0.002s per suite
- **Test coverage**: 89-96% of test code
- **Module coverage**: Cache 61%, Registry 68%, Intrinsic tested
- **Overall project coverage**: 48.03% (+6.2pp from framework tests)

### Dummy Backend

The `dummy_backend.ml` provides a minimal BACKEND implementation for testing:

```ocaml
module Dummy_backend : BACKEND = struct
  let name = "Dummy"
  let version = (1, 0, 0)
  let is_available () = true
  let execution_model = Custom
  
  (* Minimal implementations of all required modules *)
  module Device = struct
    type t = unit
    let count () = 1
    let capabilities () = { is_cpu = true; ... }
    (* ... *)
  end
  
  (* 3 test intrinsics *)
  let () =
    Intrinsics.register "test_thread_id" (make_divergent_intrinsic ...);
    Intrinsics.register "test_barrier" (make_sync_intrinsic ...);
    Intrinsics.register "test_add" (make_simple_intrinsic ...)
end
```

## Design Principles

### 1. Type Safety

- **No `Obj.t`**: Use first-class modules for plugin abstraction
- **Structured errors**: No string-based error handling
- **Clean interfaces**: Well-defined module signatures

### 2. Separation of Concerns

- **Framework is pure**: No backend-specific code
- **Plugins are independent**: Backends don't depend on each other
- **Registry is neutral**: No preference for specific backends

### 3. Extensibility

- **Plugin architecture**: Add new backends without modifying framework
- **Custom intrinsics**: Backends can define their own intrinsics
- **Priority system**: Control backend selection behavior

### 4. Performance

- **Persistent caching**: Avoid recompilation across runs
- **Statistics tracking**: Monitor cache effectiveness
- **Minimal overhead**: Registry lookup is O(1) hashtable

### 5. Developer Experience

- **Debug logging**: `SAREK_DEBUG=all` for detailed output
- **Clear errors**: Structured errors with context
- **Tests**: 53 tests covering error and execution paths
- **Documentation**: Inline comments + README

## Debug Logging

Enable debug logging via `SAREK_DEBUG` environment variable:

```bash
# Framework-specific logging
SAREK_DEBUG=framework ./my_app

# All debug output
SAREK_DEBUG=all ./my_app
```

**Log Output**:
```
[Framework] Registering backend: CUDA (priority: 100)
[Framework] Backend CUDA available: true
[Framework] Best backend: CUDA (priority: 100)
```

## Performance Considerations

### Backend Selection

- Registry lookup is O(1) hashtable operation
- `is_available()` checks may involve system calls (check once at startup)
- Priority comparison is O(n) where n = number of backends (~5)

### Intrinsic Registry

- Per-backend registries are hashtables (O(1) lookup)
- Global registry uses hashtable of lists (O(1) + O(m) where m = backends per intrinsic)
- Standard intrinsics initialized at module load time

### Kernel Cache

- Cache key computation: MD5 hashing (~1μs)
- Cache lookup: File I/O (~100μs) vs compilation (~100ms-1s)
- Typical hit rate: 80-95% after warmup
- Cache size: ~10KB per kernel, ~1MB per 100 kernels

## Related Packages

- **spoc.framework** (`spoc/framework/`) - Backend interface definitions (Framework_sig.ml)
- **sarek.core** (`sarek/core/`) - Core abstractions (Device, Vector, Transfer)
- **sarek.sarek** (`sarek/sarek/`) - Sarek runtime (Execute, Fusion, Interpreter)
- **plugins/cuda** - CUDA backend implementation
- **plugins/opencl** - OpenCL backend implementation
- **plugins/vulkan** - Vulkan backend implementation
- **plugins/native** - Native CPU backend
- **plugins/interpreter** - IR interpreter backend

## Contributing

When adding features to the framework:

1. **Maintain type safety**: No `Obj.t` or unsafe casts
2. **Add tests**: Unit tests + integration tests with dummy backend
3. **Update documentation**: Inline comments + README
4. **Check coverage**: Target 60%+ coverage for new code
5. **Follow patterns**: Study existing backends for examples
6. **Debug logging**: Add `debugf` calls for important operations

## License

See [LICENSE.md](../../LICENSE.md) in repository root.

---

**Version**: 1.0.0  
**Last Updated**: 2025-01-07  
**Maintainer**: SPOC/Sarek Team
