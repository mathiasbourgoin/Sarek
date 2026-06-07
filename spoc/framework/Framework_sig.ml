(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Sarek Framework - Plugin Interface
 *
 * Defines the interface that GPU backend plugins must implement.
 * Plugins (CUDA, OpenCL) register themselves and provide device, memory,
 * stream, event, and kernel management functionality.
 ******************************************************************************)

(** {1 Common Types} *)

(** 3D dimensions for grid and block *)
type dims = {x : int; y : int; z : int}

let dims_1d x = {x; y = 1; z = 1}

let dims_2d x y = {x; y; z = 1}

let dims_3d x y z = {x; y; z}

(** Device capabilities - queried from hardware *)
type capabilities = {
  max_threads_per_block : int;
  max_block_dims : int * int * int;
  max_grid_dims : int * int * int;
  shared_mem_per_block : int;
  total_global_mem : int64;
  compute_capability : int * int;
      (** (major, minor) for CUDA, (0,0) for OpenCL *)
  supports_fp64 : bool;
  supports_atomics : bool;
  warp_size : int;
  max_registers_per_block : int;
  clock_rate_khz : int;
  multiprocessor_count : int;
  is_cpu : bool;  (** True for CPU devices - enables zero-copy optimization *)
}

(** Device representation - SDK layer type shared across all backends *)
type device = {
  id : int;  (** Global device ID (0, 1, 2...) *)
  backend_id : int;  (** ID within the backend (0, 1...) *)
  name : string;  (** Human-readable device name *)
  framework : string;  (** Backend name: "CUDA", "OpenCL", "Vulkan", "Native" *)
  capabilities : capabilities;
}

(** {1 Plugin Module Signature} *)

(** Minimal framework signature for plugin registration. Used by
    Framework_registry for basic plugin management. *)
module type S = sig
  val name : string

  val version : int * int * int

  val is_available : unit -> bool
end

(** Execution model for backends.
    - JIT: Generate source code at runtime, compile with GPU compiler (CUDA,
      OpenCL)
    - Direct: Execute pre-compiled OCaml functions directly (Native CPU)
    - Custom: Full control over compilation pipeline (LLVM, SPIR-V, future) *)
type execution_model = JIT | Direct | Custom

(** Source language for external kernels *)
type source_lang =
  | CUDA_Source  (** CUDA C/C++ source (.cu) *)
  | OpenCL_Source  (** OpenCL C source (.cl) *)
  | PTX  (** NVIDIA PTX assembly *)
  | SPIR_V  (** SPIR-V binary *)
  | GLSL_Source  (** Vulkan GLSL compute shader *)

(** Extensible type for backend-specific kernel arguments. Each backend extends
    this type with its own variant. This allows type-safe passing of kernel args
    across the framework boundary with typed witnesses. *)
type kargs = ..

(** Placeholder kargs for testing - not associated with any backend *)
type kargs += No_kargs

(** Argument type for run_source. Buffer binder receives typed kargs. *)
type run_source_arg =
  | RSA_Buffer of {
      binder : kargs -> int -> unit;  (** Binds buffer to kernel arg *)
      length : int;  (** Vector length for generated kernels *)
    }
  | RSA_Int32 of int32
  | RSA_Int64 of int64
  | RSA_Float32 of float
  | RSA_Float64 of float

(** Convergence behavior of an intrinsic.
    - Uniform: All threads in warp/wavefront compute same value
    - Divergent: Threads may compute different values
    - Sync: Intrinsic contains synchronization (barrier) *)
type convergence = Uniform | Divergent | Sync

(** Intrinsic registry interface for backend-specific intrinsics. Note: The
    actual intrinsic_impl type is defined in each backend's intrinsic registry
    module to avoid circular dependencies with Sarek_ir. *)
module type INTRINSIC_REGISTRY = sig
  (** Backend-specific intrinsic implementation type *)
  type intrinsic_impl

  (** Register an intrinsic by name *)
  val register : string -> intrinsic_impl -> unit

  (** Look up an intrinsic by name *)
  val find : string -> intrinsic_impl option

  (** List all registered intrinsic names *)
  val list_all : unit -> string list
end

(** Re-export typed value types for convenience *)
type exec_arg = Typed_value.exec_arg =
  | EA_Int32 of int32
  | EA_Int64 of int64
  | EA_Float32 of float
  | EA_Float64 of float
  | EA_Scalar :
      (module Typed_value.SCALAR_TYPE with type t = 'a) * 'a
      -> exec_arg
  | EA_Composite :
      (module Typed_value.COMPOSITE_TYPE with type t = 'a) * 'a
      -> exec_arg
  | EA_Vec of (module Typed_value.EXEC_VECTOR)

(** Extended backend signature for Phase 4 unified execution. Adds execution
    model discrimination and IR-based code generation. *)
module type BACKEND = sig
  val name : string

  val version : int * int * int

  val is_available : unit -> bool

  (** Device management *)
  module Device : sig
    type t

    type id = int

    val init : unit -> unit

    val count : unit -> int

    val get : int -> t

    val id : t -> id

    val name : t -> string

    val capabilities : t -> capabilities

    val set_current : t -> unit

    val synchronize : t -> unit
  end

  (** Async execution streams / command queues *)
  module Stream : sig
    type t

    val create : Device.t -> t

    val destroy : t -> unit

    val synchronize : t -> unit

    val default : Device.t -> t
  end

  (** GPU memory allocation and transfer *)
  module Memory : sig
    type 'a buffer

    val alloc : Device.t -> int -> ('a, 'b) Bigarray.kind -> 'a buffer

    (** Allocate buffer for custom types with explicit element size in bytes *)
    val alloc_custom : Device.t -> size:int -> elem_size:int -> 'a buffer

    (** Allocate zero-copy buffer using host bigarray memory directly. For CPU
        OpenCL devices, this avoids memory copies entirely. Returns None if
        zero-copy not supported by this backend. *)
    val alloc_zero_copy :
      Device.t ->
      ('a, 'b, Bigarray.c_layout) Bigarray.Array1.t ->
      ('a, 'b) Bigarray.kind ->
      'a buffer option

    val free : 'a buffer -> unit

    (** {2 Synchronous Transfers} *)

    val host_to_device :
      src:('a, 'b, Bigarray.c_layout) Bigarray.Array1.t -> dst:'a buffer -> unit

    val device_to_host :
      src:'a buffer -> dst:('a, 'b, Bigarray.c_layout) Bigarray.Array1.t -> unit

    (** Transfer from raw pointer to device (for custom types). [src_ptr] is a
        raw machine address (as produced by [raw_address_of_ptr]); a [nativeint]
        is NOT a GC root, so the CALLER must keep the pointed-to allocation live
        across this call. Transfers are synchronous today (a live source on the
        caller's stack suffices); an async backend (e.g. the worker+SAB readback
        planned for the jsoo path) must root the source explicitly. *)
    val host_ptr_to_device :
      src_ptr:nativeint -> byte_size:int -> dst:'a buffer -> unit

    (** Transfer from device to raw pointer (for custom types). Same keep-alive
        contract as {!host_ptr_to_device}: the caller owns [dst_ptr]'s liveness.
    *)
    val device_to_host_ptr :
      src:'a buffer -> dst_ptr:nativeint -> byte_size:int -> unit

    val device_to_device : src:'a buffer -> dst:'a buffer -> unit

    (** {2 Buffer Info} *)

    val size : 'a buffer -> int

    val device_ptr : 'a buffer -> nativeint

    (** Check if buffer uses zero-copy (no transfers needed) *)
    val is_zero_copy : 'a buffer -> bool
  end

  (** Timing events *)
  module Event : sig
    type t

    val create : unit -> t

    val destroy : t -> unit

    val record : t -> Stream.t -> unit

    val synchronize : t -> unit

    val elapsed : start:t -> stop:t -> float
  end

  (** Kernel compilation and execution *)
  module Kernel : sig
    type t

    type args

    val compile : Device.t -> name:string -> source:string -> t

    val compile_cached : Device.t -> name:string -> source:string -> t

    val clear_cache : unit -> unit

    val create_args : unit -> args

    val set_arg_buffer : args -> int -> _ Memory.buffer -> unit

    val set_arg_int32 : args -> int -> int32 -> unit

    val set_arg_int64 : args -> int -> int64 -> unit

    val set_arg_float32 : args -> int -> float -> unit

    val set_arg_float64 : args -> int -> float -> unit

    val set_arg_ptr : args -> int -> nativeint -> unit

    val launch :
      t ->
      args:args ->
      grid:dims ->
      block:dims ->
      shared_mem:int ->
      stream:Stream.t option ->
      unit
  end

  val enable_profiling : unit -> unit

  val disable_profiling : unit -> unit

  (** The execution model this backend uses *)
  val execution_model : execution_model

  (** Generate source code from Sarek IR (for JIT backends). Returns None for
      Direct/Custom backends.
      @param block
        Optional block dimensions (required for Vulkan/GLSL which embeds
        workgroup size in shader) *)
  val generate_source : ?block:dims -> Sarek_ir_types.kernel -> string option

  (** Execute a kernel directly (for Direct/Custom backends). JIT backends
      should raise an error if this is called. The backend chooses which
      component to use:
      - Direct backends (Native): use native_fn
      - Custom backends (Interpreter): use ir

      @param native_fn Pre-compiled OCaml function (from PPX)
      @param ir Sarek IR kernel (for interpretation)
      @param block Block dimensions
      @param grid Grid dimensions
      @param args Kernel arguments - fully typed *)
  val execute_direct :
    native_fn:(block:dims -> grid:dims -> exec_arg array -> unit) option ->
    ir:Sarek_ir_types.kernel option ->
    block:dims ->
    grid:dims ->
    exec_arg array ->
    unit

  (** Backend-specific intrinsic registry *)
  module Intrinsics : INTRINSIC_REGISTRY

  (** {2 External Kernel Execution} *)

  (** List of source languages this backend supports *)
  val supported_source_langs : source_lang list

  (** Execute an external kernel from source code.
      @param source The kernel source code (CUDA/OpenCL/PTX string)
      @param lang The source language
      @param kernel_name The name of the kernel function to execute
      @param block Block dimensions
      @param grid Grid dimensions
      @param shared_mem Shared memory size in bytes
      @param args Kernel arguments as run_source_arg list
      @raise Failure if source language is not supported *)
  val run_source :
    source:string ->
    lang:source_lang ->
    kernel_name:string ->
    block:dims ->
    grid:dims ->
    shared_mem:int ->
    run_source_arg list ->
    unit

  (** {2 Kernel Args Wrapping}
      Type-safe wrapping/unwrapping of backend-specific kernel args into the
      extensible kargs type. Each backend extends kargs with its own variant. *)

  (** Wrap this backend's kernel args into a kargs variant *)
  val wrap_kargs : Kernel.args -> kargs

  (** Unwrap kargs to this backend's kernel args. Returns None if wrong backend.
  *)
  val unwrap_kargs : kargs -> Kernel.args option
end

(** Low-level backend implementation interface shared by the CUDA, OpenCL, and
    Metal plugin-base modules. This is the device/memory/stream/event/kernel
    core that each backend's [*_plugin_base.ml] implements directly on top of
    its FFI bindings; the full {!BACKEND} interface (source generation, direct
    execution, intrinsics, external-kernel execution, kargs wrapping) is
    assembled on top of it in each backend's [*_plugin.ml].

    [args] and [buffer] are abstract: callers construct args via
    {!Kernel.create_args} and the [set_arg_*] functions, never by relying on the
    underlying representation. *)
module type PLUGIN_BASE = sig
  val name : string

  val version : int * int * int

  module Device : sig
    type t

    type id = int

    val init : unit -> unit

    val count : unit -> int

    val get : int -> t

    val id : t -> id

    val name : t -> string

    val capabilities : t -> capabilities

    val set_current : t -> unit

    val get_current_device : unit -> t option

    val synchronize : t -> unit
  end

  module Memory : sig
    type 'a buffer

    val alloc : Device.t -> int -> ('a, 'b) Bigarray.kind -> 'a buffer

    val alloc_custom : Device.t -> size:int -> elem_size:int -> 'a buffer

    val alloc_zero_copy :
      Device.t ->
      ('a, 'b, Bigarray.c_layout) Bigarray.Array1.t ->
      ('a, 'b) Bigarray.kind ->
      'a buffer option

    val is_zero_copy : 'a buffer -> bool

    val free : 'a buffer -> unit

    val host_to_device :
      src:('a, 'b, Bigarray.c_layout) Bigarray.Array1.t -> dst:'a buffer -> unit

    val device_to_host :
      src:'a buffer -> dst:('a, 'b, Bigarray.c_layout) Bigarray.Array1.t -> unit

    val host_ptr_to_device :
      src_ptr:nativeint -> byte_size:int -> dst:'a buffer -> unit

    val device_to_host_ptr :
      src:'a buffer -> dst_ptr:nativeint -> byte_size:int -> unit

    val device_to_device : src:'a buffer -> dst:'a buffer -> unit

    val size : 'a buffer -> int

    val device_ptr : 'a buffer -> nativeint
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

    val synchronize : t -> unit

    val elapsed : start:t -> stop:t -> float
  end

  module Kernel : sig
    type t

    type args

    val compile : Device.t -> name:string -> source:string -> t

    val compile_cached : Device.t -> name:string -> source:string -> t

    val clear_cache : unit -> unit

    val create_args : unit -> args

    val set_arg_buffer : args -> int -> _ Memory.buffer -> unit

    val set_arg_int32 : args -> int -> int32 -> unit

    val set_arg_int64 : args -> int -> int64 -> unit

    val set_arg_float32 : args -> int -> float -> unit

    val set_arg_float64 : args -> int -> float -> unit

    val set_arg_ptr : args -> int -> nativeint -> unit

    val launch :
      t ->
      args:args ->
      grid:dims ->
      block:dims ->
      shared_mem:int ->
      stream:Stream.t option ->
      unit
  end

  val enable_profiling : unit -> unit

  val disable_profiling : unit -> unit

  val is_available : unit -> bool
end
