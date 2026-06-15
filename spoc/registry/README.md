# spoc_registry - Runtime Type and Function Registry

The `spoc_registry` library provides a runtime registry for types and
functions used in Sarek kernel code generation. It follows the ppx_deriving
pattern where PPX-generated code registers types at module initialization.

## Module Overview

```
spoc_registry/
└── Sarek_registry.ml   # Type/function lookup and registration
[../sarek/core/](../sarek/core/)         # Runtime core using registry data at execution time
[../sarek/ppx/](../sarek/ppx/)           # PPX registers types/functions into the registry
```

## Design Pattern

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          Compile Time (PPX)                              │
│                                                                          │
│   [@@sarek.type]  ──►  PPX generates registration code                  │
│   type point = { x: float32; y: float32 }                               │
│                        │                                                 │
│                        ▼                                                 │
│   let () = Sarek_registry.register_record "point"                       │
│              ~fields:[{field_name="x"; field_type="float32"; ...}; ...]  │
│              ~size:8                                                     │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ Module initialization
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           Runtime (JIT)                                  │
│                                                                          │
│   Code generator queries registry:                                       │
│   - record_fields "point" → [{field_name="x"; ...}; ...]                │
│   - type_device_code "float32" dev → "float" (CUDA) / "float" (OpenCL) │
│   - fun_device_code ~module_path:["Float32"] "sin" dev → "sinf"         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Type Definitions

### Primitive Type Info

```ocaml
type type_info = {
  ti_name : string;
  ti_device : Device_type.t -> string;  (* Device code generator *)
  ti_size : int;                        (* Size in bytes *)
}
```

### Record Type Info

```ocaml
type field_info = {
  field_name : string;
  field_type : string;  (* Type name for lookup *)
  field_mutable : bool;
}

type record_info = {
  ri_name : string;        (* Full name with module path *)
  ri_fields : field_info list;
  ri_size : int;           (* Total size in bytes *)
}
```

### Variant Type Info

```ocaml
type constructor_info = {
  ctor_name : string;
  ctor_arg_type : string option;  (* None for unit constructors *)
}

type variant_info = {
  vi_name : string;
  vi_constructors : constructor_info list;
}
```

### Function Info

```ocaml
type fun_info = {
  fi_name : string;
  fi_arity : int;
  fi_device : Device_type.t -> string;  (* Device code generator *)
  fi_arg_types : string list;
  fi_ret_type : string;
}
```

## API Reference

### Type Registration

```ocaml
(** Register a primitive type *)
val register_type : string -> device:(Device_type.t -> string) -> size:int -> unit

(** Register a record type *)
val register_record : string -> fields:field_info list -> size:int -> unit

(** Register a variant type *)
val register_variant : string -> constructors:constructor_info list -> unit
```

### Type Lookup

```ocaml
(** Find primitive type info *)
val find_type : string -> type_info option

(** Find record type info *)
val find_record : string -> record_info option

(** Find variant type info *)
val find_variant : string -> variant_info option

(** Find record by short name (handles qualified names) *)
val find_record_by_short_name : string -> record_info option

(** Check if name is registered *)
val is_type : string -> bool
val is_record : string -> bool
val is_variant : string -> bool
```

### Field and Constructor Access

```ocaml
(** Get record fields (raises if not found) *)
val record_fields : string -> field_info list

(** Get variant constructors (raises if not found) *)
val variant_constructors : string -> constructor_info list
```

### Device Code Generation

```ocaml
(** Get device code for a type *)
val type_device_code : string -> Device_type.t -> string

(** Example: *)
type_device_code "float32" cuda_dev  (* returns "float" *)
type_device_code "int64" opencl_dev  (* returns "long" *)
```

### Function Registration

```ocaml
(** Register a function *)
val register_fun : ?module_path:string list -> string
                   -> arity:int
                   -> device:(Device_type.t -> string)
                   -> arg_types:string list
                   -> ret_type:string
                   -> unit

(** Find a function *)
val find_fun : ?module_path:string list -> string -> fun_info option

(** Check if function exists *)
val is_fun : ?module_path:string list -> string -> bool

(** Get device code for a function *)
val fun_device_code : ?module_path:string list -> string -> Device_type.t -> string

(** Get device code template (ignores device parameter) *)
val fun_device_template : ?module_path:string list -> string -> string option
```

### Helper Functions

```ocaml
(** Choose between CUDA and OpenCL syntax *)
val cuda_or_opencl : Device_type.t -> string -> string -> string

(** Example: *)
cuda_or_opencl dev "threadIdx.x" "get_local_id(0)"
```

## Built-in Registrations

The following types are registered at startup:

```ocaml
(* Always available *)
"bool" -> { device = "int"; size = 4 }
"unit" -> { device = "void"; size = 0 }

(* Registered by Sarek_stdlib modules when linked *)
"float32" -> { device = "float"; size = 4 }
"float64" -> { device = "double"; size = 8 }
"int32"   -> { device = "int"; size = 4 }
"int64"   -> { device = "long"; size = 8 }
```

## Usage Examples

### PPX-Generated Registration

The `[@@sarek.type]` attribute generates:

```ocaml
(* User code *)
type point = { x : float32; y : float32 } [@@sarek.type]

(* PPX generates *)
let () =
  Sarek_registry.register_record "point"
    ~fields:[
      { field_name = "x"; field_type = "float32"; field_mutable = false };
      { field_name = "y"; field_type = "float32"; field_mutable = false };
    ]
    ~size:8
```

### Code Generator Usage

```ocaml
let generate_struct_def name =
  let fields = Sarek_registry.record_fields name in
  let field_defs = List.map (fun f ->
    let type_code = Sarek_registry.type_device_code f.field_type device in
    Printf.sprintf "  %s %s;" type_code f.field_name
  ) fields in
  Printf.sprintf "typedef struct {\n%s\n} %s;"
    (String.concat "\n" field_defs) name
```

### Intrinsic Function Registration

```ocaml
(* Register Float32.sin intrinsic *)
Sarek_registry.register_fun
  ~module_path:["Float32"] "sin"
  ~arity:1
  ~device:(fun dev ->
    Sarek_registry.cuda_or_opencl dev "sinf" "sin")
  ~arg_types:["float32"]
  ~ret_type:"float32"

(* Use in code generation *)
let call_intrinsic path name dev =
  let code = Sarek_registry.fun_device_code ~module_path:path name dev in
  Printf.sprintf "%s(x)" code
```

### Cross-Module Type Resolution

```ocaml
(* Type defined in module Geometry *)
Sarek_registry.register_record "Geometry.vector3" ~fields:[...] ~size:12

(* Lookup by full name *)
let info = Sarek_registry.find_record "Geometry.vector3"

(* Lookup by short name (for IR where module path may be lost) *)
let info = Sarek_registry.find_record_by_short_name "vector3"
```

## Testing

```bash
dune build @spoc/registry/test/runtest
```

Tests cover:
- Type registration and lookup
- Record field management
- Variant constructor management
- Function registration with module paths
- Device code generation
- Short name resolution
