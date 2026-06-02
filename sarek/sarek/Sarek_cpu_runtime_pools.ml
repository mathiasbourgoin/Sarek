(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

open Sarek_cpu_runtime_types

(** Global domain pool for parallel execution *)
module DomainPool = struct
  type task = unit -> unit

  type t = {
    num_domains : int;
    task_queue : task Queue.t;
    mutex : Mutex.t;
    cond : Condition.t;
    mutable shutdown : bool;
    domains : unit Domain.t array;
    mutable active_tasks : int;
    mutable first_error : exn option;
    done_cond : Condition.t;
  }

  let worker pool =
    let rec loop () =
      Mutex.lock pool.mutex ;
      while Queue.is_empty pool.task_queue && not pool.shutdown do
        Condition.wait pool.cond pool.mutex
      done ;
      if pool.shutdown && Queue.is_empty pool.task_queue then begin
        Mutex.unlock pool.mutex ;
        ()
      end
      else begin
        let task = Queue.pop pool.task_queue in
        pool.active_tasks <- pool.active_tasks + 1 ;
        Mutex.unlock pool.mutex ;
        let task_error =
          try
            task () ;
            None
          with exn -> Some exn
        in
        Mutex.lock pool.mutex ;
        (match (pool.first_error, task_error) with
        | None, Some exn -> pool.first_error <- Some exn
        | _ -> ()) ;
        pool.active_tasks <- pool.active_tasks - 1 ;
        if pool.active_tasks = 0 && Queue.is_empty pool.task_queue then
          Condition.broadcast pool.done_cond ;
        Mutex.unlock pool.mutex ;
        loop ()
      end
    in
    loop ()

  let create num_domains =
    let pool =
      {
        num_domains;
        task_queue = Queue.create ();
        mutex = Mutex.create ();
        cond = Condition.create ();
        shutdown = false;
        domains = [||];
        active_tasks = 0;
        first_error = None;
        done_cond = Condition.create ();
      }
    in
    let domains =
      Array.init num_domains (fun _ -> Domain.spawn (fun () -> worker pool))
    in
    {pool with domains}

  let submit pool task =
    Mutex.lock pool.mutex ;
    Queue.add task pool.task_queue ;
    Condition.signal pool.cond ;
    Mutex.unlock pool.mutex

  let wait_all pool =
    Mutex.lock pool.mutex ;
    while pool.active_tasks > 0 || not (Queue.is_empty pool.task_queue) do
      Condition.wait pool.done_cond pool.mutex
    done ;
    let first_error = pool.first_error in
    pool.first_error <- None ;
    Mutex.unlock pool.mutex ;
    match first_error with Some exn -> raise exn | None -> ()

  let shutdown pool =
    Mutex.lock pool.mutex ;
    pool.shutdown <- true ;
    Condition.broadcast pool.cond ;
    Mutex.unlock pool.mutex ;
    Array.iter Domain.join pool.domains

  let _ = shutdown (* suppress unused warning - may be used for cleanup *)
end

(** Global pool - lazily initialized *)
let global_pool : DomainPool.t option ref = ref None

let get_pool () =
  match !global_pool with
  | Some pool -> pool
  | None ->
      let num_cores = try Domain.recommended_domain_count () with _ -> 4 in
      let pool = DomainPool.create num_cores in
      global_pool := Some pool ;
      pool

(** {1 Persistent Thread Pool for Fission Mode}

    A pool of worker domains that stay alive across kernel launches. This
    eliminates the Domain spawn/join overhead that kills performance for
    workloads with many consecutive kernel launches (like odd-even sort).

    Design:
    - N worker domains created once, stay alive until program exit
    - Each kernel launch distributes BLOCKS to workers (not individual threads)
    - Each worker processes complete blocks with BSP barrier semantics
    - Workers wait on a condition variable between kernels *)

module ThreadPool = struct
  (** Work item: either thread or block distribution based on barrier usage.
      - No barriers: thread distribution (start_tid/end_tid) - more granular
      - With barriers: block distribution (start_block/end_block) - preserves
        BSP *)
  type work_item = {
    kernel :
      thread_state ->
      shared_mem ->
      Spoc_framework.Framework_sig.exec_arg array ->
      unit;
    args : Spoc_framework.Framework_sig.exec_arg array;
    uses_barriers : bool;  (** Whether kernel uses barriers *)
    (* Thread distribution (uses_barriers=false) *)
    start_tid : int;  (** First global thread ID (inclusive) *)
    end_tid : int;  (** Last global thread ID (exclusive) *)
    (* Block distribution (uses_barriers=true) *)
    start_block : int;  (** First block ID (inclusive) *)
    end_block : int;  (** Last block ID (exclusive) *)
    (* Kernel dimensions *)
    bx : int;
    by : int;
    bz : int;
    gx : int;
    gy : int;
    gz : int;
  }

  type t = {
    num_workers : int;
    mutable workers : unit Domain.t array; (* Mutable to allow in-place init *)
    (* Work distribution *)
    work : work_item option Atomic.t array; (* One slot per worker *)
    (* Synchronization *)
    mutex : Mutex.t;
    work_ready : Condition.t;
    work_done : Condition.t;
    mutable pending_workers : int;
    mutable shutdown : bool;
    mutable generation : int; (* Incremented each kernel launch *)
  }

  (** Run a single block with BSP barriers using Effect.Shallow fibers *)
  let run_block_bsp ~block:(bx, by, bz) ~grid:(gx, gy, gz)
      ~block_idx:(block_x, block_y, block_z)
      (kernel :
        thread_state ->
        shared_mem ->
        Spoc_framework.Framework_sig.exec_arg array ->
        unit) (args : Spoc_framework.Framework_sig.exec_arg array) =
    let num_threads = bx * by * bz in
    let shared = create_shared () in
    (* Thread status: 0 = running, 1 = waiting at barrier, 2 = completed *)
    let status = Array.make num_threads 0 in
    let conts : (unit, unit) Effect.Shallow.continuation option array =
      Array.make num_threads None
    in
    let num_waiting = ref 0 in
    let num_completed = ref 0 in
    (* Pre-compute int32 values *)
    let bx32 = Int32.of_int bx in
    let by32 = Int32.of_int by in
    let bz32 = Int32.of_int bz in
    let gx32 = Int32.of_int gx in
    let gy32 = Int32.of_int gy in
    let gz32 = Int32.of_int gz in
    let block_x32 = Int32.of_int block_x in
    let block_y32 = Int32.of_int block_y in
    let block_z32 = Int32.of_int block_z in
    let thread_x_table = Array.init bx Int32.of_int in
    let thread_y_table = Array.init by Int32.of_int in
    let thread_z_table =
      if bz > 1 then Array.init bz Int32.of_int else [|0l|]
    in
    (* Pre-allocate all thread states *)
    let states =
      Array.init num_threads (fun tid ->
          let thread_x = tid mod bx in
          let thread_y = tid / bx mod by in
          let thread_z = tid / (bx * by) in
          {
            thread_idx_x = thread_x_table.(thread_x);
            thread_idx_y = thread_y_table.(thread_y);
            thread_idx_z = thread_z_table.(thread_z);
            block_idx_x = block_x32;
            block_idx_y = block_y32;
            block_idx_z = block_z32;
            block_dim_x = bx32;
            block_dim_y = by32;
            block_dim_z = bz32;
            grid_dim_x = gx32;
            grid_dim_y = gy32;
            grid_dim_z = gz32;
            barrier = (fun () -> Effect.perform Barrier);
          })
    in
    (* Shallow handler *)
    let handler tid =
      {
        Effect.Shallow.retc =
          (fun () ->
            status.(tid) <- 2 ;
            incr num_completed);
        exnc = raise;
        effc =
          (fun (type a) (eff : a Effect.t) ->
            match eff with
            | Barrier ->
                Some
                  (fun (k : (a, unit) Effect.Shallow.continuation) ->
                    (* Barrier returns unit, so a = unit *)
                    conts.(tid) <- Some k ;
                    status.(tid) <- 1 ;
                    incr num_waiting)
            | _ -> None);
      }
    in
    (* Create fibers for all threads *)
    let fibers =
      Array.init num_threads (fun tid ->
          Effect.Shallow.fiber (fun () -> kernel states.(tid) shared args))
    in
    (* Initial run *)
    for tid = 0 to num_threads - 1 do
      Effect.Shallow.continue_with fibers.(tid) () (handler tid)
    done ;
    (* Superstep loop *)
    while !num_waiting > 0 do
      num_waiting := 0 ;
      for tid = 0 to num_threads - 1 do
        if status.(tid) = 1 then begin
          status.(tid) <- 0 ;
          match conts.(tid) with
          | Some k ->
              conts.(tid) <- None ;
              Effect.Shallow.continue_with k () (handler tid)
          | None -> ()
        end
      done
    done

  (** Run a range of global threads directly - for barrier-free kernels. More
      efficient than block distribution when no barriers are needed. *)
  let run_threads w =
    let threads_per_block = w.bx * w.by * w.bz in
    let empty_shared = create_shared () in
    let noop_barrier = fun () -> () in
    (* Pre-compute int32 values *)
    let bx32 = Int32.of_int w.bx in
    let by32 = Int32.of_int w.by in
    let bz32 = Int32.of_int w.bz in
    let gx32 = Int32.of_int w.gx in
    let gy32 = Int32.of_int w.gy in
    let gz32 = Int32.of_int w.gz in
    (* Pre-compute lookup tables *)
    let thread_x_table = Array.init w.bx Int32.of_int in
    let thread_y_table = Array.init w.by Int32.of_int in
    let thread_z_table =
      if w.bz > 1 then Array.init w.bz Int32.of_int else [|0l|]
    in
    let block_x_table = Array.init w.gx Int32.of_int in
    let block_y_table = Array.init w.gy Int32.of_int in
    let block_z_table =
      if w.gz > 1 then Array.init w.gz Int32.of_int else [|0l|]
    in
    (* Run each global thread in our range *)
    for global_tid = w.start_tid to w.end_tid - 1 do
      let block_id = global_tid / threads_per_block in
      let local_tid = global_tid - (block_id * threads_per_block) in
      let block_x = block_id mod w.gx in
      let block_y = block_id / w.gx mod w.gy in
      let block_z = block_id / (w.gx * w.gy) in
      let thread_x = local_tid mod w.bx in
      let thread_y = local_tid / w.bx mod w.by in
      let thread_z = local_tid / (w.bx * w.by) in
      (* Create fresh state for each thread - safe and clean *)
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
      w.kernel state empty_shared w.args
    done

  (** Worker function - runs in each domain, processes work items *)
  let worker_fn pool worker_id =
    let last_gen = ref 0 in
    let rec loop () =
      (* Wait for new work - generation must change *)
      Mutex.lock pool.mutex ;
      while pool.generation = !last_gen && not pool.shutdown do
        Condition.wait pool.work_ready pool.mutex
      done ;
      if pool.shutdown then begin
        Mutex.unlock pool.mutex ;
        ()
      end
      else begin
        last_gen := pool.generation ;
        Mutex.unlock pool.mutex ;
        (* Get our work item - only signal completion if we had work *)
        let had_work =
          match Atomic.get pool.work.(worker_id) with
          | None -> false
          | Some w ->
              if w.uses_barriers then begin
                (* Block distribution - required for BSP barriers *)
                for block_id = w.start_block to w.end_block - 1 do
                  let block_x = block_id mod w.gx in
                  let block_y = block_id / w.gx mod w.gy in
                  let block_z = block_id / (w.gx * w.gy) in
                  run_block_bsp
                    ~block:(w.bx, w.by, w.bz)
                    ~grid:(w.gx, w.gy, w.gz)
                    ~block_idx:(block_x, block_y, block_z)
                    w.kernel
                    w.args
                done
              end
              else
                (* Thread distribution - more granular, faster *)
                run_threads w ;
              Atomic.set pool.work.(worker_id) None ;
              true
        in
        (* Signal completion only if we had work *)
        if had_work then begin
          Mutex.lock pool.mutex ;
          pool.pending_workers <- pool.pending_workers - 1 ;
          if pool.pending_workers = 0 then Condition.broadcast pool.work_done ;
          Mutex.unlock pool.mutex
        end ;
        loop ()
      end
    in
    loop ()

  (** Create thread pool with N workers *)
  let create num_workers =
    let pool =
      {
        num_workers;
        workers = [||];
        (* Will be mutated in place *)
        work = Array.init num_workers (fun _ -> Atomic.make None);
        mutex = Mutex.create ();
        work_ready = Condition.create ();
        work_done = Condition.create ();
        pending_workers = 0;
        shutdown = false;
        generation = 0;
      }
    in
    (* Spawn workers - they all reference the same pool record *)
    pool.workers <-
      Array.init num_workers (fun i ->
          Domain.spawn (fun () -> worker_fn pool i)) ;
    pool

  (** Submit kernel to thread pool and wait for completion. Distribution
      strategy depends on barrier usage:
      - uses_barriers=false: distribute threads (more granular)
      - uses_barriers=true: distribute blocks (preserves BSP) *)
  let run_kernel pool ~block:(bx, by, bz) ~grid:(gx, gy, gz) ~uses_barriers
      (kernel :
        thread_state ->
        shared_mem ->
        Spoc_framework.Framework_sig.exec_arg array ->
        unit) (args : Spoc_framework.Framework_sig.exec_arg array) =
    let threads_per_block = bx * by * bz in
    let total_blocks = gx * gy * gz in
    let total_threads = total_blocks * threads_per_block in
    Mutex.lock pool.mutex ;
    let active_workers = ref 0 in
    if uses_barriers then begin
      (* Block distribution - required for BSP barriers *)
      let blocks_per_worker =
        (total_blocks + pool.num_workers - 1) / pool.num_workers
      in
      for i = 0 to pool.num_workers - 1 do
        let start_block = i * blocks_per_worker in
        let end_block = min ((i + 1) * blocks_per_worker) total_blocks in
        if start_block < total_blocks then begin
          Atomic.set
            pool.work.(i)
            (Some
               {
                 kernel;
                 args;
                 uses_barriers;
                 start_tid = 0;
                 (* unused *)
                 end_tid = 0;
                 (* unused *)
                 start_block;
                 end_block;
                 bx;
                 by;
                 bz;
                 gx;
                 gy;
                 gz;
               }) ;
          incr active_workers
        end
      done
    end
    else begin
      (* Thread distribution - more granular, faster *)
      let threads_per_worker =
        (total_threads + pool.num_workers - 1) / pool.num_workers
      in
      for i = 0 to pool.num_workers - 1 do
        let start_tid = i * threads_per_worker in
        let end_tid = min ((i + 1) * threads_per_worker) total_threads in
        if start_tid < total_threads then begin
          Atomic.set
            pool.work.(i)
            (Some
               {
                 kernel;
                 args;
                 uses_barriers;
                 start_tid;
                 end_tid;
                 start_block = 0;
                 (* unused *)
                 end_block = 0;
                 (* unused *)
                 bx;
                 by;
                 bz;
                 gx;
                 gy;
                 gz;
               }) ;
          incr active_workers
        end
      done
    end ;
    pool.pending_workers <- !active_workers ;
    pool.generation <- pool.generation + 1 ;
    Condition.broadcast pool.work_ready ;
    (* Wait for completion *)
    while pool.pending_workers > 0 do
      Condition.wait pool.work_done pool.mutex
    done ;
    Mutex.unlock pool.mutex

  let shutdown pool =
    Mutex.lock pool.mutex ;
    pool.shutdown <- true ;
    Condition.broadcast pool.work_ready ;
    Mutex.unlock pool.mutex ;
    Array.iter Domain.join pool.workers

  let _ = shutdown (* Suppress unused warning *)
end

(** Global fission thread pool - lazily initialized *)
let fission_pool : ThreadPool.t option ref = ref None

let fission_pool_mutex = Mutex.create ()

let get_fission_pool () =
  Mutex.lock fission_pool_mutex ;
  let pool =
    match !fission_pool with
    | Some p -> p
    | None ->
        let num_cores = try Domain.recommended_domain_count () with _ -> 4 in
        let p = ThreadPool.create num_cores in
        fission_pool := Some p ;
        p
  in
  Mutex.unlock fission_pool_mutex ;
  pool

(** {1 Simple Parallel Pool for Simple Kernels}

    A persistent thread pool for embarrassingly parallel kernels. Workers are
    spawned once and reused across kernel invocations.

    Uses a simple shared atomic counter for work distribution - each worker
    atomically claims chunks of work. This is simpler and more reliable than
    work-stealing for our use case. *)

module ParallelPool = struct
  (** Default chunk size for work distribution. Smaller = better load balance
      but more overhead. This is tuned for typical GPU-style workloads. *)
  let default_chunk_size = 256

  type t = {
    num_workers : int;
    mutable workers : unit Domain.t array;
    (* Shared work counter - workers atomically fetch-and-add to claim chunks *)
    next_chunk : int Atomic.t;
    total_work : int Atomic.t;
    (* Current chunk size - can be customized per parallel_for call *)
    current_chunk_size : int Atomic.t;
    (* Current work function *)
    work_fn : (int -> int -> unit) Atomic.t;
    (* Synchronization *)
    mutex : Mutex.t;
    work_ready : Condition.t;
    work_done : Condition.t;
    mutable pending_workers : int;
    mutable shutdown : bool;
    mutable generation : int;
  }

  (** Worker function - claims chunks via atomic counter *)
  let worker_fn pool _worker_id =
    let last_gen = ref 0 in

    let rec loop () =
      (* Wait for work *)
      Mutex.lock pool.mutex ;
      while (not pool.shutdown) && pool.generation = !last_gen do
        Condition.wait pool.work_ready pool.mutex
      done ;
      if pool.shutdown then begin
        Mutex.unlock pool.mutex ;
        () (* Exit *)
      end
      else begin
        last_gen := pool.generation ;
        Mutex.unlock pool.mutex ;

        (* Get work function, total, and chunk size *)
        let work_fn = Atomic.get pool.work_fn in
        let total = Atomic.get pool.total_work in
        let chunk_size = Atomic.get pool.current_chunk_size in

        (* Claim and process chunks until done *)
        let rec process_chunks () =
          let start = Atomic.fetch_and_add pool.next_chunk chunk_size in
          if start < total then begin
            let end_ = min (start + chunk_size) total in
            work_fn start end_ ;
            process_chunks ()
          end
        in
        process_chunks () ;

        (* Signal completion *)
        Mutex.lock pool.mutex ;
        pool.pending_workers <- pool.pending_workers - 1 ;
        if pool.pending_workers = 0 then Condition.signal pool.work_done ;
        Mutex.unlock pool.mutex ;

        loop ()
      end
    in
    loop ()

  (** Create a parallel pool with the given number of workers *)
  let create num_workers =
    let pool =
      {
        num_workers;
        workers = [||];
        (* Will be set after workers are spawned *)
        next_chunk = Atomic.make 0;
        total_work = Atomic.make 0;
        current_chunk_size = Atomic.make default_chunk_size;
        work_fn = Atomic.make (fun _ _ -> ());
        mutex = Mutex.create ();
        work_ready = Condition.create ();
        work_done = Condition.create ();
        pending_workers = 0;
        shutdown = false;
        generation = 0;
      }
    in
    (* Spawn worker domains - they reference the same pool record *)
    pool.workers <-
      Array.init num_workers (fun id ->
          Domain.spawn (fun () -> worker_fn pool id)) ;
    pool

  (** Run a parallel_for over the range 0 to total (exclusive) with custom chunk
      size *)
  let parallel_for_chunk pool ~total ~chunk_size (work_fn : int -> int -> unit)
      =
    if total <= 0 then ()
    else begin
      (* Reset counter and set work *)
      Atomic.set pool.next_chunk 0 ;
      Atomic.set pool.total_work total ;
      Atomic.set pool.current_chunk_size chunk_size ;
      Atomic.set pool.work_fn work_fn ;

      (* Wake workers *)
      Mutex.lock pool.mutex ;
      pool.pending_workers <- pool.num_workers ;
      pool.generation <- pool.generation + 1 ;
      Condition.broadcast pool.work_ready ;

      (* Wait for completion *)
      while pool.pending_workers > 0 do
        Condition.wait pool.work_done pool.mutex
      done ;
      Mutex.unlock pool.mutex
    end

  (** Run a parallel_for over the range 0 to total (exclusive) with default
      chunk size *)
  let parallel_for pool ~total work_fn =
    parallel_for_chunk pool ~total ~chunk_size:default_chunk_size work_fn

  let shutdown pool =
    Mutex.lock pool.mutex ;
    pool.shutdown <- true ;
    Condition.broadcast pool.work_ready ;
    Mutex.unlock pool.mutex ;
    Array.iter Domain.join pool.workers

  let _ = shutdown (* Suppress unused warning - available for cleanup *)
end

(** Global parallel pool for simple kernels *)
let parallel_pool : ParallelPool.t option ref = ref None

let parallel_pool_mutex = Mutex.create ()

let get_parallel_pool () =
  match !parallel_pool with
  | Some p -> p
  | None ->
      Mutex.lock parallel_pool_mutex ;
      let p =
        match !parallel_pool with
        | Some p -> p
        | None ->
            let num_workers = max 1 (Domain.recommended_domain_count () - 1) in
            let p = ParallelPool.create num_workers in
            parallel_pool := Some p ;
            p
      in
      Mutex.unlock parallel_pool_mutex ;
      p

(** {1 Fission Queue with Thread Pool Execution}

    Multiple queues are supported (like CUDA/OpenCL command queues):
    - Same queue_id: kernels execute in order (serialized)
    - Different queue_id: kernels can run in parallel (one dispatcher per queue)

    Each queue has a dispatcher domain that pulls kernels and submits them to
    the shared thread pool. The thread pool workers stay alive across all kernel
    launches. *)

module LaunchQueue = struct
  (** A pending kernel launch - uses wrapper signature for compatibility *)
  type launch = {
    kernel :
      mode:exec_mode ->
      block:int * int * int ->
      grid:int * int * int ->
      Spoc_framework.Framework_sig.exec_arg array ->
      unit;
    args : Spoc_framework.Framework_sig.exec_arg array;
    block : int * int * int;
    grid : int * int * int;
  }

  (** Queue state - one per queue_id *)
  type t = {
    queue : launch Queue.t;
    mutex : Mutex.t;
    cond : Condition.t;  (** Signal: new work available *)
    done_cond : Condition.t;  (** Signal: queue empty and idle *)
    mutable dispatcher : unit Domain.t option;
    mutable shutdown : bool;
    mutable pending : int;  (** Number of launches not yet completed *)
  }

  (** Dispatcher loop - pulls kernels and executes via wrapper The wrapper
      internally uses run_threadpool with mode=Threadpool *)
  let dispatcher_loop (q : t) =
    let rec loop () =
      Mutex.lock q.mutex ;
      (* Wait for work or shutdown *)
      while Queue.is_empty q.queue && not q.shutdown do
        Condition.wait q.cond q.mutex
      done ;
      if q.shutdown && Queue.is_empty q.queue then begin
        Mutex.unlock q.mutex ;
        () (* Exit dispatcher *)
      end
      else begin
        let launch = Queue.pop q.queue in
        Mutex.unlock q.mutex ;
        (* Execute via the wrapper - Threadpool mode uses persistent pool *)
        launch.kernel
          ~mode:Threadpool
          ~block:launch.block
          ~grid:launch.grid
          launch.args ;
        (* Mark completion *)
        Mutex.lock q.mutex ;
        q.pending <- q.pending - 1 ;
        if q.pending = 0 then Condition.broadcast q.done_cond ;
        Mutex.unlock q.mutex ;
        loop ()
      end
    in
    loop ()

  (** Create a new launch queue (dispatcher starts on first enqueue) *)
  let create () : t =
    {
      queue = Queue.create ();
      mutex = Mutex.create ();
      cond = Condition.create ();
      done_cond = Condition.create ();
      dispatcher = None;
      shutdown = false;
      pending = 0;
    }

  (** Ensure dispatcher is running *)
  let ensure_dispatcher (q : t) =
    if q.dispatcher = None then
      q.dispatcher <- Some (Domain.spawn (fun () -> dispatcher_loop q))

  (** Enqueue a kernel launch - dispatcher starts processing immediately *)
  let enqueue (q : t) ~kernel ~args ~block ~grid =
    Mutex.lock q.mutex ;
    ensure_dispatcher q ;
    Queue.add {kernel; args; block; grid} q.queue ;
    q.pending <- q.pending + 1 ;
    Condition.signal q.cond ;
    Mutex.unlock q.mutex

  (** Wait for all pending launches to complete (flush/barrier) *)
  let flush (q : t) =
    Mutex.lock q.mutex ;
    while q.pending > 0 do
      Condition.wait q.done_cond q.mutex
    done ;
    Mutex.unlock q.mutex

  (** Shutdown the dispatcher (for cleanup) *)
  let shutdown (q : t) =
    Mutex.lock q.mutex ;
    q.shutdown <- true ;
    Condition.signal q.cond ;
    Mutex.unlock q.mutex ;
    match q.dispatcher with
    | Some d ->
        Domain.join d ;
        q.dispatcher <- None
    | None -> ()

  let _ = shutdown (* Suppress unused warning *)
end

(** Multiple queues indexed by queue_id - allows concurrent execution *)
let fission_queues : (int, LaunchQueue.t) Hashtbl.t = Hashtbl.create 4

let fission_queues_mutex = Mutex.create ()

let get_fission_queue queue_id =
  Mutex.lock fission_queues_mutex ;
  let q =
    match Hashtbl.find_opt fission_queues queue_id with
    | Some q -> q
    | None ->
        let q = LaunchQueue.create () in
        Hashtbl.add fission_queues queue_id q ;
        q
  in
  Mutex.unlock fission_queues_mutex ;
  q
