(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Sarek_cpu_runtime - CPU runtime for generated native kernels

    This module provides the runtime support for kernels compiled to native
    OCaml code by the Sarek PPX. Unlike Sarek_interp which interprets the AST,
    this module is used by generated code that runs at full native speed. *)

(** {1 Float32 Module}

    Re-exported for use in generated native kernels. Provides true float32
    semantics matching GPU behavior. *)
module Float32 = Sarek_float32

(** {1 Execution Mode} *)

(** Execution mode for native kernels *)
type exec_mode =
  | Sequential  (** Single-threaded, barriers are no-ops *)
  | Parallel  (** Spawn domains per kernel launch *)
  | Threadpool  (** Use persistent thread pool (fission mode) *)

(** {1 Thread State} *)

(** Thread state passed to each kernel invocation. Contains thread/block/grid
    indices and dimensions, plus a barrier function.

    All indices are int32 to match GPU semantics (Sarek_stdlib.Gpu uses int32).
*)
type thread_state = {
  thread_idx_x : int32;
  thread_idx_y : int32;
  thread_idx_z : int32;
  block_idx_x : int32;
  block_idx_y : int32;
  block_idx_z : int32;
  block_dim_x : int32;
  block_dim_y : int32;
  block_dim_z : int32;
  grid_dim_x : int32;
  grid_dim_y : int32;
  grid_dim_z : int32;
  barrier : unit -> unit;
}

(** {1 Global Index Helpers} *)

val global_idx_x : thread_state -> int32

val global_idx_y : thread_state -> int32

val global_idx_z : thread_state -> int32

val global_size_x : thread_state -> int32

val global_size_y : thread_state -> int32

val global_size_z : thread_state -> int32

(** {1 Shared Memory} *)

(** Shared memory container for a block. Uses per-type hashtables for type
    safety on common types. *)
type shared_mem

(** Create a new empty shared memory container. *)
val create_shared : unit -> shared_mem

(** {2 Typed Allocators} *)

val alloc_shared_int : shared_mem -> string -> int -> int -> int array

val alloc_shared_float : shared_mem -> string -> int -> float -> float array

val alloc_shared_int32 : shared_mem -> string -> int -> int32 -> int32 array

val alloc_shared_int64 : shared_mem -> string -> int -> int64 -> int64 array

(** {2 Generic Allocator}

    For custom types not covered by typed allocators. *)

val alloc_shared_with_key :
  shared_mem -> 'a Sarek_ir_types.Type_id.t -> string -> int -> 'a -> 'a array

(** {1 Execution Modes} *)

(** Run kernel sequentially. All threads execute in order, barriers are no-ops.
    Good for debugging and correctness testing. The kernel function receives
    thread_state, shared_mem (for the block), and args. *)
val run_sequential :
  block:int * int * int ->
  grid:int * int * int ->
  (thread_state -> shared_mem -> 'a -> unit) ->
  'a ->
  unit

(** Run kernel in parallel. Threads within each block run in parallel using
    OCaml 5 Domains.

    @param has_barriers
      Static barrier metadata from the compiler or caller. [true] uses the BSP
      barrier-capable path; [false] uses the faster barrier-free path. When
      omitted, the runtime uses the sequential barrier-capable path so barrier
      detection never executes user kernel code before launch and worker-pool
      exception behavior is not broadened. *)
val run_parallel :
  ?has_barriers:bool ->
  block:int * int * int ->
  grid:int * int * int ->
  (thread_state -> shared_mem -> 'a -> unit) ->
  'a ->
  unit

(** {1 Fission Mode - Thread Pool Execution}

    For the fission device, kernels are executed by a persistent thread pool.
    This eliminates per-launch Domain spawn/join overhead for workloads with
    many consecutive kernel launches (like odd-even sort with 512 launches).

    Architecture:
    - A persistent pool of N worker domains (one per core) stays alive
    - Each kernel launch distributes work to workers via condition variables
    - Workers execute their portion and signal completion
    - No Domain spawn/join per kernel - just signaling overhead *)

(** Run kernel using the persistent thread pool. Like run_parallel but uses
    pre-created worker domains. Best for workloads with many consecutive kernel
    launches.

    @param has_barriers
      Compile-time flag from PPX indicating barrier usage.
      - false: use thread distribution (faster, more granular)
      - true: use block distribution with BSP barriers *)
val run_threadpool :
  has_barriers:bool ->
  block:int * int * int ->
  grid:int * int * int ->
  (thread_state -> shared_mem -> 'a -> unit) ->
  'a ->
  unit

(** {2 Asynchronous Queue API}

    Multiple queues are supported (like CUDA/OpenCL command queues):
    - Same queue_id: kernels execute in order (serialized)
    - Different queue_id: kernels can run in parallel (one dispatcher per queue)
*)

(** Enqueue a kernel for fission execution on a specific queue. The kernel
    starts executing immediately in the background via thread pool. Kernels on
    the same queue execute in order; different queues run in parallel. Default
    queue_id is 0. *)
val enqueue_fission :
  ?queue_id:int ->
  kernel:
    (mode:exec_mode ->
    block:int * int * int ->
    grid:int * int * int ->
    Spoc_framework.Framework_sig.exec_arg array ->
    unit) ->
  args:Spoc_framework.Framework_sig.exec_arg array ->
  block:int * int * int ->
  grid:int * int * int ->
  unit ->
  unit

(** Wait for a specific queue to complete. *)
val flush_fission_queue : int -> unit

(** Wait for all fission queues to complete. Called by Devices.flush. *)
val flush_fission : unit -> unit

(** {1 Optimized Simple Kernel Runners}

    For kernels that only use global_idx_x/y/z without thread/block dimensions,
    shared memory, or barriers, we can skip the expensive thread_state machinery
    and just pass the global index directly.

    This eliminates:
    - six thread-state field updates per element
    - 6 integer divisions/modulos
    - Function call overhead through thread_state

    These functions are used when the PPX detects Simple1D/2D/3D execution
    strategy. *)

(** Run a simple 1D kernel in parallel - just iterates over global_idx_x.
    @param total_x Total number of elements (block_x * grid_x)
    @param kernel Function taking (gid_x, args) *)
val run_1d_threadpool : total_x:int -> (int32 -> 'a -> unit) -> 'a -> unit

(** Run a simple 2D kernel in parallel - iterates over global_idx_x,
    global_idx_y.
    @param width Total width (block_x * grid_x)
    @param height Total height (block_y * grid_y)
    @param kernel Function taking (gid_x, gid_y, args) *)
val run_2d_threadpool :
  width:int -> height:int -> (int32 -> int32 -> 'a -> unit) -> 'a -> unit

(** Run a simple 3D kernel in parallel - iterates over global_idx_x/y/z.
    @param width Total width
    @param height Total height
    @param depth Total depth
    @param kernel Function taking (gid_x, gid_y, gid_z, args) *)
val run_3d_threadpool :
  width:int ->
  height:int ->
  depth:int ->
  (int32 -> int32 -> int32 -> 'a -> unit) ->
  'a ->
  unit
