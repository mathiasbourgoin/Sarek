# spoc_framework - GPU Backend Plugin Interface

The `spoc_framework` library defines the interfaces that GPU backends must
implement. It provides the contract between the Sarek runtime and backend
plugins (CUDA, OpenCL, Vulkan, Native CPU, Interpreter).

## Module Overview

```
spoc_framework/
├── Framework_sig.ml    # BACKEND module type and common types
├── Typed_value.ml      # Type-safe value representation
└── Device_type.ml      # Device type alias (backward compatibility)
[../sarek/core/](../sarek/core/)         # Runtime core that consumes BACKEND plugins
[../sarek/plugins/](../sarek/plugins/)   # Backend implementations (CUDA/OpenCL/Vulkan/Native/Interpreter)
```

## Framework_sig.ml

### Common Types

```ocaml
(** 3D dimensions for kernel launch *)
type dims = { x : int; y : int; z : int }

val dims_1d : int -> dims
val dims_2d : int -> int -> dims
val dims_3d : int -> int -> int -> dims

(** Device capabilities *)
type capabilities = {
  max_threads_per_block : int;
  max_block_dims : int * int * int;
  max_grid_dims : int * int * int;
  shared_mem_per_block : int;
  total_global_mem : int64;
  compute_capability : int * int;  (* CUDA: (major, minor), others: (0, 0) *)
  supports_fp64 : bool;
  supports_atomics : bool;
  warp_size : int;
  max_registers_per_block : int;
  clock_rate_khz : int;
  multiprocessor_count : int;
  is_cpu : bool;  (* True for CPU backends - enables zero-copy *)
}

(** Device representation *)
type device = {
  id : int;          (* Global device ID *)
  backend_id : int;  (* ID within backend *)
  name : string;
  framework : string; (* "CUDA", "OpenCL", "Vulkan", "Native" *)
  capabilities : capabilities;
}

(** Execution model discrimination *)
type execution_model =
  | JIT     (* Runtime compilation: CUDA, OpenCL *)
  | Direct  (* Pre-compiled: Native CPU *)
  | Custom  (* Custom pipeline: Interpreter, LLVM *)

(** Source languages for external kernels *)
type source_lang =
  | CUDA_Source | OpenCL_Source | PTX | SPIR_V | GLSL_Source
```

### BACKEND Module Type

Every GPU backend implements this signature:

```ocaml
module type BACKEND = sig
  val name : string
  val version : int * int * int
  val is_available : unit -> bool
  val execution_model : execution_model

  module Device : sig
    type t
    val init : unit -> unit
    val count : unit -> int
    val get : int -> t
    val name : t -> string
    val capabilities : t -> capabilities
    val set_current : t -> unit
    val synchronize : t -> unit
  end

  module Memory : sig
    type 'a buffer
    val alloc : Device.t -> int -> ('a, 'b) Bigarray.kind -> 'a buffer
    val alloc_custom : Device.t -> size:int -> elem_size:int -> 'a buffer
    val alloc_zero_copy : Device.t -> ('a,'b,Bigarray.c_layout) Bigarray.Array1.t
                          -> ('a,'b) Bigarray.kind -> 'a buffer option
    val free : 'a buffer -> unit
    val host_to_device : src:_ Bigarray.Array1.t -> dst:'a buffer -> unit
    val device_to_host : src:'a buffer -> dst:_ Bigarray.Array1.t -> unit
    val size : 'a buffer -> int
    val is_zero_copy : 'a buffer -> bool
  end

  module Stream : sig
    type t
    val create : Device.t -> t
    val destroy : t -> unit
    val synchronize : t -> unit
    val default : Device.t -> t
  end

  module Event : sig
    type t
    val create : unit -> t
    val destroy : t -> unit
    val record : t -> Stream.t -> unit
    val elapsed : start:t -> stop:t -> float
  end

  module Kernel : sig
    type t
    type args
    val compile : Device.t -> name:string -> source:string -> t
    val compile_cached : Device.t -> name:string -> source:string -> t
    val create_args : unit -> args
    val set_arg_buffer : args -> int -> _ Memory.buffer -> unit
    val set_arg_int32 : args -> int -> int32 -> unit
    val set_arg_float32 : args -> int -> float -> unit
    (* ... *)
    val launch : t -> args:args -> grid:dims -> block:dims
                 -> shared_mem:int -> stream:Stream.t option -> unit
  end

  (** Code generation for JIT backends *)
  val generate_source : ?block:dims -> Sarek_ir_types.kernel -> string option

  (** Direct execution for Direct/Custom backends *)
  val execute_direct :
    native_fn:(block:dims -> grid:dims -> exec_arg array -> unit) option ->
    ir:Sarek_ir_types.kernel option ->
    block:dims -> grid:dims -> exec_arg array -> unit

  (** External kernel execution *)
  val supported_source_langs : source_lang list
  val run_source : source:string -> lang:source_lang -> kernel_name:string
                   -> block:dims -> grid:dims -> shared_mem:int
                   -> run_source_arg list -> unit

  (** Backend-specific kernel args wrapping *)
  val wrap_kargs : Kernel.args -> kargs
  val unwrap_kargs : kargs -> Kernel.args option
end
```

### Extensible kargs Type

The `kargs` type allows type-safe passing of backend-specific kernel arguments:

```ocaml
(* Base extensible type *)
type kargs = ..

(* Each backend extends it *)
type kargs += CUDA_kargs of Cuda_backend.Kernel.args

(* Usage: bind buffer at position 0 *)
let bind_buffer kargs idx =
  match kargs with
  | CUDA_kargs args -> Cuda_backend.Kernel.set_arg_buffer args idx buf
  | OpenCL_kargs args -> Opencl_backend.Kernel.set_arg_buffer args idx buf
  | _ -> failwith "Unknown backend"
```

## Typed_value.ml

Type-safe value representation without `Obj.t`.

### SCALAR_TYPE Interface

```ocaml
module type SCALAR_TYPE = sig
  type t
  val name : string                    (* "float32", "int64", etc. *)
  val size : int                       (* Size in bytes *)
  val to_primitive : t -> primitive    (* Serialize *)
  val of_primitive : primitive -> t    (* Deserialize *)
  val ctype : t Ctypes.typ            (* FFI representation *)
end
```

### Built-in Types

```ocaml
module Int32_type : SCALAR_TYPE with type t = int32
module Int64_type : SCALAR_TYPE with type t = int64
module Float32_type : SCALAR_TYPE with type t = float
module Float64_type : SCALAR_TYPE with type t = float
module Bool_type : SCALAR_TYPE with type t = bool
```

### Type Registry

```ocaml
module Registry : sig
  val register_scalar : (module SCALAR_TYPE) -> unit
  val register_composite : (module COMPOSITE_TYPE) -> unit
  val find_scalar : string -> (module SCALAR_TYPE) option
  val find_composite : string -> (module COMPOSITE_TYPE) option
  val list_scalars : unit -> string list
end
```

### Execution Arguments

```ocaml
type exec_arg =
  | EA_Int32 of int32
  | EA_Int64 of int64
  | EA_Float32 of float
  | EA_Float64 of float
  | EA_Scalar : (module SCALAR_TYPE with type t = 'a) * 'a -> exec_arg
  | EA_Composite : (module COMPOSITE_TYPE with type t = 'a) * 'a -> exec_arg
  | EA_Vec of (module EXEC_VECTOR)

(* Conversions *)
val exec_arg_of_typed_value : typed_value -> exec_arg
val typed_value_of_exec_arg : exec_arg -> typed_value
val type_name_of_exec_arg : exec_arg -> string
```

## Device_type.ml

Backward compatibility alias:

```ocaml
type t = Framework_sig.device = {
  id : int;
  backend_id : int;
  name : string;
  framework : string;
  capabilities : Framework_sig.capabilities;
}
```

New code should use `Framework_sig.device` directly.

## Usage Examples

### Creating a Backend Plugin

```ocaml
module My_backend : Framework_sig.BACKEND = struct
  let name = "MyBackend"
  let version = (1, 0, 0)
  let is_available () =
    (* Check if runtime libraries are available *)
    true

  let execution_model = Framework_sig.JIT

  module Device = struct
    type t = { id : int; (* ... *) }
    let init () = (* Initialize runtime *)
    let count () = (* Return number of devices *)
    let get id = (* Return device by ID *)
    (* ... *)
  end

  module Memory = struct
    type 'a buffer = { ptr : nativeint; size : int }
    let alloc dev n kind = (* Allocate device memory *)
    (* ... *)
  end

  (* ... rest of implementation *)
end
```

### Using Typed Values

```ocaml
(* Create typed argument *)
let arg = EA_Float32 3.14

(* Get type information *)
let name = type_name_of_exec_arg arg  (* "float32" *)

(* Convert to typed_value for storage *)
let tv = typed_value_of_exec_arg arg

(* Convert back *)
let arg' = exec_arg_of_typed_value tv
```

### Launching a Kernel

```ocaml
let launch_kernel device kernel args =
  let grid = Framework_sig.dims_1d 1024 in
  let block = Framework_sig.dims_1d 256 in
  match device.framework with
  | "Native" ->
      (* Use native function directly *)
      Backend.execute_direct ~native_fn:(Some fn) ~ir:None ~block ~grid args
  | _ ->
      (* JIT compile and launch *)
      let source = Option.get (Backend.generate_source kernel) in
      let compiled = Backend.Kernel.compile_cached device
                       ~name:kernel.kern_name ~source in
      Backend.Kernel.launch compiled ~args ~grid ~block ~shared_mem:0 ~stream:None
```

## Testing

```bash
dune build @spoc/framework/test/runtest
```

Tests cover:
- `test_framework_sig.ml` - dims helpers, type construction
- `test_typed_value.ml` - scalar types, registry, conversions
- `test_device_type.ml` - alias verification
