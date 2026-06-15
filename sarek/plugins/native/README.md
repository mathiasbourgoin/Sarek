# Native Plugin - CPU Execution Backend

**Package**: `sarek.native`  
**Execution Model**: Direct (pre-compiled OCaml functions)

CPU backend that executes pre-compiled OCaml functions directly without JIT compilation. Kernels are registered by the Sarek PPX at module load time.

## Overview

The Native backend provides CPU execution for Sarek kernels using OCaml 5 domains for parallelism. Unlike GPU backends that generate and compile source code at runtime, the Native backend executes functions that were compiled with the rest of your application.

### Execution Flow

1. **Kernel Definition**: `[%kernel fun ...]` in OCaml code
2. **PPX Processing**: Sarek PPX generates both GPU IR and native OCaml function
3. **Registration**: Native function auto-registered at module load time
4. **Execution**: Direct function call with typed arguments (no compilation)

### Key Features

- Direct execution of pre-compiled functions
- OCaml 5 domains for parallel execution
- Zero compilation overhead at runtime
- Type-safe argument passing via `Framework_sig.exec_arg`
- Supports both parallel and sequential execution modes

## Architecture

### Module Structure

```
sarek/plugins/native/
├── Native_error.ml              # Structured error handling
├── Native_plugin_base.ml        # Core implementation (678 lines)
├── Native_plugin.ml             # BACKEND interface (393 lines)
└── test/
    └── test_native_error.ml     # Error tests
```

### Device Model

The Native backend creates virtual devices representing CPU cores:

- **Parallel Device**: Multi-core execution using OCaml 5 domains
- **Sequential Device**: Single-threaded execution for debugging

## Usage

```ocaml
(* Kernel automatically works on Native backend *)
let%kernel vector_add (a : float32 vector) (b : float32 vector) (c : float32 vector) =
  let idx = get_global_id 0 in
  c.(idx) <- a.(idx) + b.(idx)

(* Execute on Native backend *)
let () =
  let device = Sarek.Device.get_device 0 in  (* CPU device *)
  vector_add ~block:(256, 1, 1) ~grid:(n/256, 1, 1) a b c
```

The Native backend is selected automatically when no GPU is available or can be explicitly requested.

## Limitations

- No external GPU source execution (CUDA/OpenCL/GLSL)
- No raw pointer arguments
- Performance limited by CPU capabilities
- No custom type accessors (get/set on custom buffers)

## Testing

```bash
# Run error tests
dune test sarek/plugins/native
```

## Implementation Details

### Kernel Registration

Kernels register themselves via a global hashtable:

```ocaml
val native_kernels : 
  (string, Framework_sig.exec_arg array -> 
           int * int * int -> int * int * int -> unit) Hashtbl.t
```

### Memory Model

- **Zero-copy**: Native backend always uses zero-copy (host and device share memory)
- **Bigarray storage**: All buffers backed by Bigarray for efficient access
- **Type safety**: GADT-based buffer types prevent type errors

### Parallelism

Uses OCaml 5 domains to parallelize across grid dimensions. The parallel execution strategy depends on the grid size and available cores.
