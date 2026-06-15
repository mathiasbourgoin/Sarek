# SPOC - SDK Package for OCaml Computing

The `spoc` package provides the foundational layer for GPU computing in OCaml.
It defines the core abstractions shared by all backends (CUDA, OpenCL, Vulkan,
Native CPU) without depending on any specific GPU runtime.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              User Code                                       │
│                     (Kernels written with %kernel)                          │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
┌──────────────────────────────────────▼──────────────────────────────────────┐
│                            Sarek Runtime                                     │
│         (sarek package: Execute, Kirc_kernel, Sarek_ir)                     │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
┌──────────────────────────────────────▼──────────────────────────────────────┐
│                           SPOC SDK Layer                                     │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐  │
│  │   framework/    │  │      ir/        │  │        registry/            │  │
│  │                 │  │                 │  │                             │  │
│  │  Framework_sig  │  │ Sarek_ir_types  │  │     Sarek_registry          │  │
│  │  Typed_value    │  │ Sarek_ir_pp     │  │                             │  │
│  │  Device_type    │  │ Sarek_ir_analysis│ │                             │  │
│  └────────┬────────┘  └────────┬────────┘  └──────────────┬──────────────┘  │
│           │                    │                          │                  │
│           │      Pure types, no FFI dependencies          │                  │
└───────────┼────────────────────┼──────────────────────────┼──────────────────┘
            │                    │                          │
┌───────────▼────────────────────▼──────────────────────────▼──────────────────┐
│                            Backend Plugins                                   │
│       CUDA        OpenCL        Vulkan        Native CPU     Interpreter     │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Package Structure

```
spoc/
├── framework/          # Plugin interface definitions
│   ├── Framework_sig.ml   - Backend plugin interface (BACKEND module type)
│   ├── Typed_value.ml     - Type-safe value representation
│   └── Device_type.ml     - Device abstraction (alias to Framework_sig.device)
│
├── ir/                 # GPU kernel IR representation
│   ├── Sarek_ir_types.ml  - IR type definitions (elttype, expr, stmt, kernel)
│   ├── Sarek_ir_pp.ml     - Pretty printing for debugging
│   └── Sarek_ir_analysis.ml - Static analysis (float64 detection)
│
└── registry/           # Runtime type and function registry
    └── Sarek_registry.ml  - Type/function lookup for code generation
└── [../sarek/core/](../sarek/core/)     # Runtime core (Device, Memory, Vector, Kernel, Transfer)
└── [../sarek/ppx/](../sarek/ppx/)       # Sarek PPX compiler (kernel syntax frontend)
└── [../sarek/plugins/](../sarek/plugins/)  # Backend plugin implementations
```

## Design Principles

### 1. Pure Types, No FFI

The spoc package contains no FFI bindings or runtime dependencies.
This allows:
- Compilation without GPU drivers installed
- Use in pure OCaml environments (js_of_ocaml, etc.)
- Clean separation between interface and implementation

### 2. Backend Independence

All types are defined generically. The `Framework_sig.BACKEND` module type
specifies what a backend must provide:
- Device enumeration and management
- Memory allocation and transfer
- Kernel compilation and execution
- Stream and event handling

### 3. Type Safety Without Obj.t

The `Typed_value` module provides existential type wrappers that preserve
type safety across module boundaries:

```ocaml
(* Scalar types implement SCALAR_TYPE *)
module type SCALAR_TYPE = sig
  type t
  val name : string
  val size : int
  val to_primitive : t -> primitive
  val of_primitive : primitive -> t
end

(* Values are wrapped with their type module *)
type scalar_value = SV : (module SCALAR_TYPE with type t = 'a) * 'a -> scalar_value

(* Kernel arguments preserve types *)
type exec_arg =
  | EA_Int32 of int32
  | EA_Int64 of int64
  | EA_Float32 of float
  | EA_Float64 of float
  | EA_Scalar : (module SCALAR_TYPE with type t = 'a) * 'a -> exec_arg
  | EA_Vec of (module EXEC_VECTOR)
```

### 4. Extensible Types

The IR uses extensible types for forward compatibility:

```ocaml
(* Backend-specific kernel args extend this type *)
type kargs = ..

(* Each backend adds its variant *)
type kargs += CUDA_kargs of Cuda_backend.Kernel.args
type kargs += OpenCL_kargs of Opencl_backend.Kernel.args
```

## Key Types

### Device Capabilities

```ocaml
type capabilities = {
  max_threads_per_block : int;
  max_block_dims : int * int * int;
  max_grid_dims : int * int * int;
  shared_mem_per_block : int;
  total_global_mem : int64;
  compute_capability : int * int;
  supports_fp64 : bool;
  supports_atomics : bool;
  warp_size : int;
  is_cpu : bool;
  (* ... *)
}
```

### Execution Model

```ocaml
type execution_model =
  | JIT     (* CUDA, OpenCL - runtime compilation *)
  | Direct  (* Native CPU - pre-compiled functions *)
  | Custom  (* Interpreter, LLVM - custom pipeline *)
```

### IR Element Types

```ocaml
type elttype =
  | TInt32 | TInt64 | TFloat32 | TFloat64 | TBool | TUnit
  | TRecord of string * (string * elttype) list
  | TVariant of string * (string * elttype list) list
  | TArray of elttype * memspace
  | TVec of elttype
```

## Usage

### For Backend Implementers

Implement the `Framework_sig.BACKEND` module type:

```ocaml
module My_backend : Framework_sig.BACKEND = struct
  let name = "MyBackend"
  let version = (1, 0, 0)
  let is_available () = (* check runtime availability *)
  let execution_model = JIT

  module Device = struct (* ... *) end
  module Memory = struct (* ... *) end
  module Kernel = struct (* ... *) end
  (* ... *)
end
```

### For PPX/Compiler Implementers

Use IR types to represent compiled kernels:

```ocaml
let kernel : Sarek_ir_types.kernel = {
  kern_name = "vector_add";
  kern_params = [ (* ... *) ];
  kern_body = (* ... *);
  (* ... *)
}

(* Check if float64 extension needed *)
if Sarek_ir_analysis.kernel_uses_float64 kernel then
  print_endline "Requires FP64 support"

(* Pretty print for debugging *)
Sarek_ir_pp.print_kernel kernel
```

### For Runtime Type Registration

Register custom types for code generation:

```ocaml
Sarek_registry.register_record "point3d"
  ~fields:[
    { field_name = "x"; field_type = "float32"; field_mutable = false };
    { field_name = "y"; field_type = "float32"; field_mutable = false };
    { field_name = "z"; field_type = "float32"; field_mutable = false };
  ]
  ~size:12
```

## Testing

Run spoc package tests:

```bash
make test_spoc
# or
dune build @spoc/runtest
```

Tests are located in each subpackage:
- `spoc/framework/test/` - Framework_sig, Typed_value, Device_type
- `spoc/ir/test/` - Sarek_ir_types, Sarek_ir_pp, Sarek_ir_analysis
- `spoc/registry/test/` - Sarek_registry

## Dependencies

No GPU runtime dependencies. No ctypes dependency.
