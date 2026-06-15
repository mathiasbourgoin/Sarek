# Backend_error - Shared Error Handling for GPU Backends

**Package**: `spoc.framework`  
**Module**: `Spoc_framework.Backend_error`

Generic structured error types for all GPU backend operations (CUDA, OpenCL, Vulkan, Metal, Native, Interpreter).

## Overview

`Backend_error` provides a unified error handling system for GPU backends, eliminating code duplication across backend implementations. Each backend instantiates the `Make` functor with their name to get backend-specific error constructors.

### Error Categories

1. **Codegen Errors** - IR to backend source translation failures
2. **Runtime Errors** - Device operations, compilation, memory management
3. **Plugin Errors** - Missing libraries, unsupported features

## Usage

### Basic Usage - Direct Construction

```ocaml
open Spoc_framework.Backend_error

(* Create errors with explicit backend name *)
let err1 = unknown_intrinsic ~backend:"CUDA" "my_func"
let err2 = compilation_failed ~backend:"OpenCL" "kernel void foo() {" "syntax error"
let err3 = library_not_found ~backend:"Vulkan" "libvulkan.so" ["/usr/lib"]

(* Convert to string *)
let msg = to_string err1  (* "[CUDA Codegen] Unknown intrinsic: my_func" *)

(* Raise as exception *)
let () = raise_error err1
```

### Recommended Usage - Backend-Specific Module

Each backend should instantiate the functor to get clean constructors:

```ocaml
(* In cuda_backend/Cuda_error.ml *)
module E = Spoc_framework.Backend_error.Make(struct let name = "CUDA" end)

(* Now use without ~backend parameter *)
let err1 = E.unknown_intrinsic "my_func"
let err2 = E.compilation_failed "source" "log"
let err3 = E.raise_error (E.device_not_found 5 2)
```

### Error Handling Patterns

```ocaml
(* Pattern 1: Raise exceptions *)
if not (is_valid intrinsic) then
  E.raise_error (E.unknown_intrinsic intrinsic_name)

(* Pattern 2: Return Result type *)
let compile source =
  E.to_result (fun () ->
    match compile_kernel source with
    | Ok kernel -> kernel
    | Error msg -> E.raise_error (E.compilation_failed source msg))

(* Pattern 3: Fallback with default value *)
let get_device () =
  E.with_default ~default:None (fun () ->
    Some (find_device ()))
```

## API Reference

### Error Types

#### Codegen Errors
- `Unknown_intrinsic {name}` - Unrecognized intrinsic function
- `Invalid_arg_count {intrinsic; expected; got}` - Wrong argument count
- `Unsupported_construct {construct; reason}` - IR construct not supported
- `Type_error {expr; expected; got}` - Type mismatch
- `Invalid_memory_space {decl; space}` - Invalid memory qualifier
- `Unsupported_type {type_name; backend}` - Type not supported (e.g., fp64 without extension)

#### Runtime Errors
- `No_device_selected {operation}` - Operation needs device but none set
- `Device_not_found {device_id; max_devices}` - Device ID out of range
- `Compilation_failed {source; log}` - Kernel compilation failed
- `Module_load_failed {size; reason}` - Failed to load compiled module
- `Kernel_launch_failed {kernel_name; reason}` - Kernel launch failed
- `Memory_allocation_failed {bytes; reason}` - Device memory allocation failed
- `Memory_copy_failed {direction; bytes; reason}` - Host-device transfer failed
- `Context_error {operation; reason}` - GPU context operation failed
- `Synchronization_failed {reason}` - Device sync failed

#### Plugin Errors
- `Unsupported_source_lang {lang; backend}` - Source language not supported
- `Backend_unavailable {reason}` - Backend not available (drivers, devices missing)
- `Library_not_found {library; paths}` - Required library not found
- `Initialization_failed {backend; reason}` - Backend init failed
- `Feature_not_supported {feature; backend}` - Feature not supported

### Core Functions

```ocaml
val to_string : t -> string
(** Convert error to human-readable message with [Backend] prefix *)

val raise_error : t -> 'a
(** Raise error as Backend_error exception *)

val print_error : t -> unit
(** Print error to stderr *)

val with_default : default:'a -> (unit -> 'a) -> 'a
(** Execute function, return default on Backend_error *)

val to_result : (unit -> 'a) -> ('a, t) result
(** Execute function, wrap in Result type *)

val result_to_string : ('a, t) result -> ('a, string) result
(** Convert Error variant to error message string *)
```

### Make Functor

```ocaml
module Make (B : sig val name : string end) : sig
  (* Codegen error constructors *)
  val unknown_intrinsic : string -> t
  val invalid_arg_count : string -> int -> int -> t
  val unsupported_construct : string -> string -> t
  val type_error : string -> string -> string -> t
  val invalid_memory_space : string -> string -> t
  val unsupported_type : string -> t

  (* Runtime error constructors *)
  val no_device_selected : string -> t
  val device_not_found : int -> int -> t
  val compilation_failed : string -> string -> t
  val module_load_failed : int -> string -> t
  val kernel_launch_failed : string -> string -> t
  val memory_allocation_failed : int64 -> string -> t
  val memory_copy_failed : string -> int -> string -> t
  val context_error : string -> string -> t
  val synchronization_failed : string -> t

  (* Plugin error constructors *)
  val unsupported_source_lang : string -> t
  val backend_unavailable : string -> t
  val library_not_found : string -> string list -> t
  val initialization_failed : string -> t
  val feature_not_supported : string -> t

  (* Re-exported utilities *)
  val raise_error : t -> 'a
  val print_error : t -> unit
  val with_default : default:'a -> (unit -> 'a) -> 'a
  val to_result : (unit -> 'a) -> ('a, t) result
  val to_string : t -> string
end
```

## Migration Guide

### Migrating Existing Backend

For a backend with existing error handling (e.g., CUDA with `Cuda_error.ml`):

1. **Create backend-specific module** (keep existing interface):

```ocaml
(* Cuda_error.ml - new implementation *)
include Spoc_framework.Backend_error.Make(struct let name = "CUDA" end)

(* Optional: Add CUDA-specific errors if needed *)
type cuda_specific_error = ...
```

2. **Update call sites** - No changes needed if using same constructor names!

```ocaml
(* Before *)
let err = Cuda_error.unknown_intrinsic "func"
let () = Cuda_error.raise_error err

(* After - same code! *)
let err = Cuda_error.unknown_intrinsic "func"
let () = Cuda_error.raise_error err
```

3. **Update exception handling**:

```ocaml
(* Before *)
try ... with Cuda_error.Cuda_error err -> ...

(* After *)
try ... with Spoc_framework.Backend_error.Backend_error err -> ...
```

### For New Backends

```ocaml
(* backend/Backend_error.ml *)
include Spoc_framework.Backend_error.Make(struct let name = "MyBackend" end)
```

That's it! All error handling is provided by the shared module.

## Design Rationale

### Why Framework-Level?

- **Low-level dependency**: Lives in `spoc.framework` (no circular deps)
- **Shared by all backends**: CUDA, OpenCL, Vulkan, Metal, Native, Interpreter
- **Alongside backend signature**: Natural place for cross-backend infrastructure

### Why Not Exceptions-First?

- **Structured data**: Error records preserve context (names, IDs, paths)
- **Inspectable**: Errors can be analyzed programmatically
- **Testable**: Easy to construct and verify in tests
- **Exception interop**: `raise_error` wrapper when needed

### Why Parameterized by Backend Name?

- **Clear error origin**: Messages show "[CUDA]", "[OpenCL]", etc.
- **Type safety**: Still just one error type, no per-backend exceptions
- **Easy debugging**: Immediately know which backend failed

## Code Savings

Without shared error module:
- CUDA: ~170 lines (Cuda_error.ml)
- OpenCL: ~170 lines (would need similar)
- Vulkan: ~170 lines (would need similar)
- Metal: ~170 lines (would need similar)
- Native: ~170 lines (would need similar)
- **Total**: ~850 lines of duplicated error handling

With shared error module:
- Backend_error.ml: ~350 lines (generic)
- Per backend: `include Backend_error.Make(...)` (1 line)
- **Savings**: ~500 lines (60%)

## Testing

See `spoc/framework/test/test_backend_error.ml` for comprehensive test coverage:

- Codegen error construction and formatting
- Runtime error messages with context
- Plugin error handling
- Multiple backend isolation
- Exception handling (raise, catch, default, result)

Run tests:
```bash
dune build spoc/framework/test/test_backend_error.exe
_build/default/spoc/framework/test/test_backend_error.exe
```

## Future Enhancements

Potential additions:

1. **Error codes**: Numeric codes for programmatic error classification
2. **Source locations**: File/line info for better debugging
3. **Error chaining**: Wrap underlying errors (e.g., OS errors)
4. **Severity levels**: Warning vs Error vs Fatal
5. **i18n support**: Localized error messages

## Related Modules

- `sarek/framework/Framework_error.ml` - Framework-level errors (plugin registry, intrinsics)
- `sarek/core/Error.ml` - Core runtime errors (device, memory, kernel execution)
- `spoc/framework/Framework_sig.ml` - Backend interface signatures
