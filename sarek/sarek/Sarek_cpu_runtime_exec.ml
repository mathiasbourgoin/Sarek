(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

open Sarek_cpu_runtime_types

(** {1 Sequential Execution}

    Runs all threads in sequence with proper BSP barrier synchronization. Uses
    the same effect-based approach as parallel, but without domains. *)

(** Run a block sequentially with BSP-style barrier synchronization. Same
    algorithm as run_block_with_barriers but simpler since no parallelism.

    Optimized version: pre-allocate thread states and reuse effect handlers. *)
let run_block_sequential_bsp ~block:(bx, by, bz) ~grid:(gx, gy, gz)
    ~block_idx:(block_x, block_y, block_z)
    (kernel : thread_state -> shared_mem -> 'a -> unit) (args : 'a) : unit =
  let num_threads = bx * by * bz in
  let shared = create_shared () in

  (* Continuations waiting at barrier *)
  let waiting : (unit, unit) Effect.Deep.continuation option array =
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

  (* Pre-compute thread index lookup tables *)
  let thread_x_table = Array.init bx Int32.of_int in
  let thread_y_table = Array.init by Int32.of_int in
  let thread_z_table = if bz > 1 then Array.init bz Int32.of_int else [|0l|] in

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

  (* Shared effect handler - reused for all threads *)
  let handler tid =
    {
      Effect.Deep.retc = (fun () -> incr num_completed);
      exnc = raise;
      effc =
        (fun (type a) (eff : a Effect.t) ->
          match eff with
          | Barrier ->
              Some
                (fun (k : (a, unit) Effect.Deep.continuation) ->
                  (* Barrier returns unit, so a = unit, k : (unit, unit) continuation *)
                  waiting.(tid) <- Some k ;
                  incr num_waiting)
          | _ -> None);
    }
  in

  (* Run a single thread with effect handler *)
  let run_thread tid =
    Effect.Deep.match_with
      (fun () -> kernel states.(tid) shared args)
      ()
      (handler tid)
  in

  (* Resume a waiting thread *)
  let resume_thread tid =
    match waiting.(tid) with
    | Some k ->
        waiting.(tid) <- None ;
        Effect.Deep.match_with
          (fun () -> Effect.Deep.continue k ())
          ()
          (handler tid)
    | None -> ()
  in

  (* Initial run: start all threads *)
  for tid = 0 to num_threads - 1 do
    run_thread tid
  done ;

  (* Superstep loop: while threads are waiting at barriers *)
  while !num_waiting > 0 do
    let to_resume = !num_waiting in
    num_waiting := 0 ;
    for tid = 0 to num_threads - 1 do
      if Option.is_some waiting.(tid) then resume_thread tid
    done ;
    if !num_waiting = to_resume && !num_completed < num_threads then
      Interp_error.raise_error
        (Interp_error.BSP_deadlock
           {
             message =
               Printf.sprintf
                 "No progress made: %d threads waiting, %d completed (context: \
                  run_block_parallel_bsp)"
                 !num_waiting
                 !num_completed;
           })
  done

let run_sequential ~block:(bx, by, bz) ~grid:(gx, gy, gz)
    (kernel : thread_state -> shared_mem -> 'a -> unit) (args : 'a) : unit =
  for block_z = 0 to gz - 1 do
    for block_y = 0 to gy - 1 do
      for block_x = 0 to gx - 1 do
        run_block_sequential_bsp
          ~block:(bx, by, bz)
          ~grid:(gx, gy, gz)
          ~block_idx:(block_x, block_y, block_z)
          kernel
          args
      done
    done
  done

(** {1 Parallel Execution}

    Uses OCaml 5 Domain parallelism with effects for fibers.
    - Fixed pool of N domains (one per core)
    - Blocks are distributed across domains
    - Threads within a block run as fibers with proper barrier sync

    BSP Model: For barrier-based kernels, we use a superstep execution model: 1.
    Each thread is a fiber that can be suspended at barriers 2. All threads run
    until they hit a barrier (or complete) 3. When all threads have reached the
    barrier, all are resumed 4. Repeat until all threads complete

    The Barrier effect is declared in the sequential section above. *)

(** Run a block with BSP-style barrier synchronization. Each thread is a
    separate fiber. All threads run until barrier, then all resume together.

    Uses Effect.Shallow for better performance - avoids reinstalling handlers on
    every resume, which is significant for BSP execution with many barriers. *)
let run_block_with_barriers ~block:(bx, by, bz) ~grid:(gx, gy, gz)
    ~block_idx:(block_x, block_y, block_z)
    (kernel : thread_state -> shared_mem -> 'a -> unit) (args : 'a) : unit =
  let num_threads = bx * by * bz in
  let shared = create_shared () in

  (* Thread status: 0 = running, 1 = waiting at barrier, 2 = completed *)
  let status = Array.make num_threads 0 in
  (* Shallow continuations stored as option for type safety *)
  let conts : (unit, unit) Effect.Shallow.continuation option array =
    Array.make num_threads None
  in
  let num_waiting = ref 0 in
  let num_completed = ref 0 in
  let first_error = ref None in

  let record_error tid exn =
    if !first_error = None then first_error := Some exn ;
    if status.(tid) <> 2 then begin
      status.(tid) <- 2 ;
      incr num_completed
    end
  in

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

  (* Pre-compute thread index lookup tables *)
  let thread_x_table = Array.init bx Int32.of_int in
  let thread_y_table = Array.init by Int32.of_int in
  let thread_z_table = if bz > 1 then Array.init bz Int32.of_int else [|0l|] in

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

  (* Shallow handler - much lighter weight than Deep handlers *)
  let handler tid =
    {
      Effect.Shallow.retc =
        (fun () ->
          if status.(tid) <> 2 then begin
            status.(tid) <- 2 ;
            incr num_completed
          end);
      exnc = (fun exn -> record_error tid exn);
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
        Effect.Shallow.fiber (fun () ->
            try kernel states.(tid) shared args
            with exn -> record_error tid exn))
  in

  (* Initial run: start all threads with shallow continue_with *)
  for tid = 0 to num_threads - 1 do
    Effect.Shallow.continue_with fibers.(tid) () (handler tid)
  done ;

  (* Superstep loop: while threads are waiting at barriers *)
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
  done ;
  match !first_error with Some exn -> raise exn | None -> ()
