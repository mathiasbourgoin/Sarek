(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Sarek_cpu_runtime - CPU runtime for generated native kernels

    This module provides the runtime support for kernels compiled to native
    OCaml code by the Sarek PPX. Unlike Sarek_interp which interprets the AST,
    this module is used by generated code that runs at full native speed. *)

(** Re-export Float32 module for use in generated kernels *)
module Float32 = Sarek_cpu_runtime_types.Float32

(** {1 Re-exports from sub-modules} *)

type exec_mode = Sarek_cpu_runtime_types.exec_mode =
  | Sequential  (** Single-threaded, barriers are no-ops *)
  | Parallel  (** Spawn domains per kernel launch *)
  | Threadpool  (** Use persistent thread pool (fission mode) *)

type thread_state = Sarek_cpu_runtime_types.thread_state = {
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

type shared_mem = Sarek_cpu_runtime_types.shared_mem

let global_idx_x = Sarek_cpu_runtime_types.global_idx_x

let global_idx_y = Sarek_cpu_runtime_types.global_idx_y

let global_idx_z = Sarek_cpu_runtime_types.global_idx_z

let global_size_x = Sarek_cpu_runtime_types.global_size_x

let global_size_y = Sarek_cpu_runtime_types.global_size_y

let global_size_z = Sarek_cpu_runtime_types.global_size_z

let create_shared = Sarek_cpu_runtime_types.create_shared

let alloc_shared_int = Sarek_cpu_runtime_types.alloc_shared_int

let alloc_shared_float = Sarek_cpu_runtime_types.alloc_shared_float

let alloc_shared_int32 = Sarek_cpu_runtime_types.alloc_shared_int32

let alloc_shared_int64 = Sarek_cpu_runtime_types.alloc_shared_int64

let alloc_shared_with_key = Sarek_cpu_runtime_types.alloc_shared_with_key

(** {1 Sequential Execution} *)

let run_sequential = Sarek_cpu_runtime_exec.run_sequential

(** {1 Parallel Execution} *)

(** Run kernel in parallel without barriers - simple work partitioning.
    Distributes all global threads across domains. Optimized for speed. *)
let run_parallel_simple ~block:(bx, by, bz) ~grid:(gx, gy, gz)
    (kernel : thread_state -> shared_mem -> 'a -> unit) (args : 'a) : unit =
  let pool = Sarek_cpu_runtime_pools.get_pool () in
  let num_domains = pool.Sarek_cpu_runtime_pools.DomainPool.num_domains in
  (* Total number of global threads *)
  let threads_per_block = bx * by * bz in
  let total_blocks = gx * gy * gz in
  let total_threads = total_blocks * threads_per_block in
  let threads_per_domain = (total_threads + num_domains - 1) / num_domains in
  (* Pre-convert dimensions to int32 *)
  let bx32 = Int32.of_int bx in
  let by32 = Int32.of_int by in
  let bz32 = Int32.of_int bz in
  let gx32 = Int32.of_int gx in
  let gy32 = Int32.of_int gy in
  let gz32 = Int32.of_int gz in
  (* Shared empty hashtable - barrier-free kernels don't use shared memory *)
  let empty_shared = create_shared () in
  let noop_barrier = fun () -> () in
  (* Pre-compute int32 lookup tables to avoid Int32.of_int in hot loop *)
  let thread_x_table = Array.init bx Int32.of_int in
  let thread_y_table = Array.init by Int32.of_int in
  let thread_z_table = Array.init bz Int32.of_int in
  let block_x_table = Array.init gx Int32.of_int in
  let block_y_table = Array.init gy Int32.of_int in
  let block_z_table = Array.init gz Int32.of_int in
  (* Use Domain.spawn directly for less overhead than pool *)
  let domains =
    Array.init num_domains (fun domain_id ->
        let start_tid = domain_id * threads_per_domain in
        let end_tid =
          min ((domain_id + 1) * threads_per_domain) total_threads
        in
        if start_tid >= total_threads then None
        else
          Some
            (Domain.spawn (fun () ->
                 for global_tid = start_tid to end_tid - 1 do
                   (* Compute block and thread indices from global thread ID *)
                   let block_id = global_tid / threads_per_block in
                   let local_tid =
                     global_tid - (block_id * threads_per_block)
                   in
                   let block_x = block_id mod gx in
                   let block_y = block_id / gx mod gy in
                   let block_z = block_id / (gx * gy) in
                   let thread_x = local_tid mod bx in
                   let thread_y = local_tid / bx mod by in
                   let thread_z = local_tid / (bx * by) in
                   (* Create state for this thread - safe and clean *)
                   let state =
                     {
                       thread_idx_x = thread_x_table.(thread_x);
                       thread_idx_y = thread_y_table.(thread_y);
                       thread_idx_z = thread_z_table.(thread_z);
                       block_idx_x = block_x_table.(block_x);
                       block_idx_y = block_y_table.(block_y);
                       block_idx_z = block_z_table.(block_z);
                       block_dim_x = bx32;
                       block_dim_y = by32;
                       block_dim_z = bz32;
                       grid_dim_x = gx32;
                       grid_dim_y = gy32;
                       grid_dim_z = gz32;
                       barrier = noop_barrier;
                     }
                   in
                   kernel state empty_shared args
                 done)))
  in
  Array.iter (function Some d -> Domain.join d | None -> ()) domains

(** Run kernel in parallel with BSP-style barriers. Distributes blocks across
    domain pool, uses effect-based barriers for thread sync within each block.
*)
let run_parallel_with_barriers ~block:(bx, by, bz) ~grid:(gx, gy, gz)
    (kernel : thread_state -> shared_mem -> 'a -> unit) (args : 'a) : unit =
  let pool = Sarek_cpu_runtime_pools.get_pool () in
  for block_z = 0 to gz - 1 do
    for block_y = 0 to gy - 1 do
      for block_x = 0 to gx - 1 do
        Sarek_cpu_runtime_pools.DomainPool.submit pool (fun () ->
            Sarek_cpu_runtime_exec.run_block_with_barriers
              ~block:(bx, by, bz)
              ~grid:(gx, gy, gz)
              ~block_idx:(block_x, block_y, block_z)
              kernel
              args)
      done
    done
  done ;
  Sarek_cpu_runtime_pools.DomainPool.wait_all pool

(** Run kernel in parallel. Barrier metadata must come from the compiler or
    caller; the runtime must not execute user code to discover it. When metadata
    is unavailable, use the sequential barrier-capable path to preserve
    correctness without broadening worker-pool exception swallowing. *)
let run_parallel ?has_barriers ~block ~grid kernel args =
  match has_barriers with
  | None -> run_sequential ~block ~grid kernel args
  | Some true -> run_parallel_with_barriers ~block ~grid kernel args
  | Some false -> run_parallel_simple ~block ~grid kernel args

(** Run kernel using the persistent thread pool. This is like run_parallel but
    uses the fission thread pool instead of spawning new domains. For use by the
    wrapper in fission mode.

    Uses compile-time barrier detection from PPX to choose optimal distribution:
    - has_barriers=false: distribute threads (more granular, faster)
    - has_barriers=true: distribute blocks (required for BSP semantics) *)
let run_threadpool ~has_barriers ~block:(bx, by, bz) ~grid:(gx, gy, gz)
    (kernel : thread_state -> shared_mem -> 'a -> unit) (args : 'a) : unit =
  if has_barriers then begin
    (* Use ThreadPool for barrier kernels - need BSP semantics *)
    let pool = Sarek_cpu_runtime_pools.get_fission_pool () in
    let wrapped_kernel state shared _obj_args = kernel state shared args in
    Sarek_cpu_runtime_pools.ThreadPool.run_kernel
      pool
      ~block:(bx, by, bz)
      ~grid:(gx, gy, gz)
      ~uses_barriers:true
      wrapped_kernel
      [||]
  end
  else begin
    (* Use ParallelPool for barrier-free kernels - better load balancing *)
    let pool = Sarek_cpu_runtime_pools.get_parallel_pool () in
    let threads_per_block = bx * by * bz in
    let total_blocks = gx * gy * gz in
    let total_threads = total_blocks * threads_per_block in
    let empty_shared = create_shared () in
    (* Pre-compute int32 values for block/thread dimensions *)
    let bx32 = Int32.of_int bx in
    let by32 = Int32.of_int by in
    let bz32 = Int32.of_int bz in
    let gx32 = Int32.of_int gx in
    let gy32 = Int32.of_int gy in
    let gz32 = Int32.of_int gz in
    (* Pre-compute lookup tables *)
    let thread_x_table = Array.init bx Int32.of_int in
    let thread_y_table = Array.init by Int32.of_int in
    let thread_z_table =
      if bz > 1 then Array.init bz Int32.of_int else [|0l|]
    in
    let block_x_table = Array.init gx Int32.of_int in
    let block_y_table = Array.init gy Int32.of_int in
    let block_z_table = if gz > 1 then Array.init gz Int32.of_int else [|0l|] in
    let noop_barrier = fun () -> () in
    Sarek_cpu_runtime_pools.ParallelPool.parallel_for
      pool
      ~total:total_threads
      (fun start end_ ->
        for global_tid = start to end_ - 1 do
          let block_id = global_tid / threads_per_block in
          let local_tid = global_tid - (block_id * threads_per_block) in
          let block_x = block_id mod gx in
          let block_y = block_id / gx mod gy in
          let block_z = block_id / (gx * gy) in
          let thread_x = local_tid mod bx in
          let thread_y = local_tid / bx mod by in
          let thread_z = local_tid / (bx * by) in
          (* Create fresh state for each thread *)
          let state =
            {
              thread_idx_x = thread_x_table.(thread_x);
              thread_idx_y = thread_y_table.(thread_y);
              thread_idx_z = thread_z_table.(thread_z);
              block_idx_x = block_x_table.(block_x);
              block_idx_y = block_y_table.(block_y);
              block_idx_z = block_z_table.(block_z);
              block_dim_x = bx32;
              block_dim_y = by32;
              block_dim_z = bz32;
              grid_dim_x = gx32;
              grid_dim_y = gy32;
              grid_dim_z = gz32;
              barrier = noop_barrier;
            }
          in
          kernel state empty_shared args
        done)
  end

(** {1 Fission Queue API} *)

(** Enqueue a kernel for fission execution on a specific queue. The kernel
    starts executing immediately in the background. Kernels on same queue
    execute in order; different queues run in parallel. The kernel is the
    wrapper function from cpu_kern. *)
let enqueue_fission ?(queue_id = 0) ~kernel ~args ~block ~grid () =
  let q = Sarek_cpu_runtime_pools.get_fission_queue queue_id in
  Sarek_cpu_runtime_pools.LaunchQueue.enqueue q ~kernel ~args ~block ~grid

(** Wait for a specific queue to complete. *)
let flush_fission_queue queue_id =
  Mutex.lock Sarek_cpu_runtime_pools.fission_queues_mutex ;
  let q = Hashtbl.find_opt Sarek_cpu_runtime_pools.fission_queues queue_id in
  Mutex.unlock Sarek_cpu_runtime_pools.fission_queues_mutex ;
  match q with
  | Some q -> Sarek_cpu_runtime_pools.LaunchQueue.flush q
  | None -> () (* Queue doesn't exist = nothing to flush *)

(** Wait for all fission queues to complete. Called by Devices.flush. *)
let flush_fission () =
  (* Get all queues under lock, then flush without lock *)
  Mutex.lock Sarek_cpu_runtime_pools.fission_queues_mutex ;
  let queues =
    Hashtbl.fold
      (fun _ q acc -> q :: acc)
      Sarek_cpu_runtime_pools.fission_queues
      []
  in
  Mutex.unlock Sarek_cpu_runtime_pools.fission_queues_mutex ;
  List.iter Sarek_cpu_runtime_pools.LaunchQueue.flush queues

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

(** Run a simple 1D kernel in parallel - just iterates over global_idx_x. Kernel
    signature: (gid_x:int32 -> args -> unit)

    Uses persistent parallel pool for efficient parallel execution. *)
let run_1d_threadpool ~total_x (kernel : int32 -> 'a -> unit) (args : 'a) : unit
    =
  let pool = Sarek_cpu_runtime_pools.get_parallel_pool () in
  Sarek_cpu_runtime_pools.ParallelPool.parallel_for
    pool
    ~total:total_x
    (fun start end_ ->
      for x = start to end_ - 1 do
        kernel (Int32.of_int x) args
      done)

(** Run a simple 2D kernel in parallel - iterates over global_idx_x,
    global_idx_y. Kernel signature: (gid_x:int32 -> gid_y:int32 -> args -> unit)

    Uses persistent parallel pool. Flattens 2D to 1D for work distribution, then
    unfolds coordinates. This allows fine-grained work stealing which is
    essential for workloads with non-uniform computation (like Mandelbrot where
    center pixels do more work than edge pixels). *)
let run_2d_threadpool ~width ~height (kernel : int32 -> int32 -> 'a -> unit)
    (args : 'a) : unit =
  let pool = Sarek_cpu_runtime_pools.get_parallel_pool () in
  let total = width * height in
  Sarek_cpu_runtime_pools.ParallelPool.parallel_for
    pool
    ~total
    (fun start end_ ->
      for idx = start to end_ - 1 do
        let y = idx / width in
        let x = idx - (y * width) in
        (* Faster than mod *)
        kernel (Int32.of_int x) (Int32.of_int y) args
      done)

(** Run a simple 3D kernel in parallel - iterates over global_idx_x/y/z. Kernel
    signature: (gid_x:int32 -> gid_y:int32 -> gid_z:int32 -> args -> unit)

    Uses persistent parallel pool. Flattens 3D to 1D for work distribution. *)
let run_3d_threadpool ~width ~height ~depth
    (kernel : int32 -> int32 -> int32 -> 'a -> unit) (args : 'a) : unit =
  let pool = Sarek_cpu_runtime_pools.get_parallel_pool () in
  let total = width * height * depth in
  let wh = width * height in
  Sarek_cpu_runtime_pools.ParallelPool.parallel_for
    pool
    ~total
    (fun start end_ ->
      for idx = start to end_ - 1 do
        let z = idx / wh in
        let rem = idx - (z * wh) in
        (* Faster than mod *)
        let y = rem / width in
        let x = rem - (y * width) in
        kernel (Int32.of_int x) (Int32.of_int y) (Int32.of_int z) args
      done)
