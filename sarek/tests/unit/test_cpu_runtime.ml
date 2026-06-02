(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Unit tests for Sarek_cpu_runtime - CPU native kernel runtime *)

open Sarek.Sarek_cpu_runtime
open Alcotest

(** {1 Thread State Tests} *)

(** Test thread state creation with basic values *)
let test_thread_state_creation () =
  let state =
    {
      thread_idx_x = 5l;
      thread_idx_y = 3l;
      thread_idx_z = 1l;
      block_idx_x = 2l;
      block_idx_y = 1l;
      block_idx_z = 0l;
      block_dim_x = 16l;
      block_dim_y = 8l;
      block_dim_z = 1l;
      grid_dim_x = 4l;
      grid_dim_y = 2l;
      grid_dim_z = 1l;
      barrier = (fun () -> ());
    }
  in
  check int32 "thread_idx_x" 5l state.thread_idx_x ;
  check int32 "thread_idx_y" 3l state.thread_idx_y ;
  check int32 "block_idx_x" 2l state.block_idx_x ;
  check int32 "block_dim_x" 16l state.block_dim_x ;
  check int32 "grid_dim_x" 4l state.grid_dim_x

(** Test thread state with maximum 1D indices *)
let test_thread_state_1d_max () =
  let state =
    {
      thread_idx_x = 1023l;
      thread_idx_y = 0l;
      thread_idx_z = 0l;
      block_idx_x = 255l;
      block_idx_y = 0l;
      block_idx_z = 0l;
      block_dim_x = 1024l;
      block_dim_y = 1l;
      block_dim_z = 1l;
      grid_dim_x = 256l;
      grid_dim_y = 1l;
      grid_dim_z = 1l;
      barrier = (fun () -> ());
    }
  in
  check int32 "thread_idx_x max" 1023l state.thread_idx_x ;
  check int32 "block_idx_x max" 255l state.block_idx_x

(** Test thread state with 3D indices *)
let test_thread_state_3d () =
  let state =
    {
      thread_idx_x = 7l;
      thread_idx_y = 5l;
      thread_idx_z = 3l;
      block_idx_x = 2l;
      block_idx_y = 1l;
      block_idx_z = 1l;
      block_dim_x = 8l;
      block_dim_y = 8l;
      block_dim_z = 4l;
      grid_dim_x = 4l;
      grid_dim_y = 2l;
      grid_dim_z = 2l;
      barrier = (fun () -> ());
    }
  in
  check int32 "thread_idx_z" 3l state.thread_idx_z ;
  check int32 "block_idx_z" 1l state.block_idx_z ;
  check int32 "block_dim_z" 4l state.block_dim_z ;
  check int32 "grid_dim_z" 2l state.grid_dim_z

(** {1 Global Index Calculation Tests} *)

(** Test global_idx_x with simple 1D case *)
let test_global_idx_x_simple () =
  let state =
    {
      thread_idx_x = 5l;
      thread_idx_y = 0l;
      thread_idx_z = 0l;
      block_idx_x = 2l;
      block_idx_y = 0l;
      block_idx_z = 0l;
      block_dim_x = 16l;
      block_dim_y = 1l;
      block_dim_z = 1l;
      grid_dim_x = 4l;
      grid_dim_y = 1l;
      grid_dim_z = 1l;
      barrier = (fun () -> ());
    }
  in
  (* global_x = block_idx * block_dim + thread_idx = 2 * 16 + 5 = 37 *)
  check int32 "global_idx_x" 37l (global_idx_x state)

(** Test global_idx_x with first block *)
let test_global_idx_x_first_block () =
  let state =
    {
      thread_idx_x = 3l;
      thread_idx_y = 0l;
      thread_idx_z = 0l;
      block_idx_x = 0l;
      block_idx_y = 0l;
      block_idx_z = 0l;
      block_dim_x = 8l;
      block_dim_y = 1l;
      block_dim_z = 1l;
      grid_dim_x = 1l;
      grid_dim_y = 1l;
      grid_dim_z = 1l;
      barrier = (fun () -> ());
    }
  in
  (* global_x = 0 * 8 + 3 = 3 *)
  check int32 "global_idx_x first block" 3l (global_idx_x state)

(** Test global_idx_y with 2D case *)
let test_global_idx_y_2d () =
  let state =
    {
      thread_idx_x = 0l;
      thread_idx_y = 7l;
      thread_idx_z = 0l;
      block_idx_x = 0l;
      block_idx_y = 3l;
      block_idx_z = 0l;
      block_dim_x = 16l;
      block_dim_y = 8l;
      block_dim_z = 1l;
      grid_dim_x = 4l;
      grid_dim_y = 4l;
      grid_dim_z = 1l;
      barrier = (fun () -> ());
    }
  in
  (* global_y = 3 * 8 + 7 = 31 *)
  check int32 "global_idx_y" 31l (global_idx_y state)

(** Test global_idx_z with 3D case *)
let test_global_idx_z_3d () =
  let state =
    {
      thread_idx_x = 0l;
      thread_idx_y = 0l;
      thread_idx_z = 2l;
      block_idx_x = 0l;
      block_idx_y = 0l;
      block_idx_z = 1l;
      block_dim_x = 8l;
      block_dim_y = 8l;
      block_dim_z = 4l;
      grid_dim_x = 2l;
      grid_dim_y = 2l;
      grid_dim_z = 2l;
      barrier = (fun () -> ());
    }
  in
  (* global_z = 1 * 4 + 2 = 6 *)
  check int32 "global_idx_z" 6l (global_idx_z state)

(** Test global_size_x calculation *)
let test_global_size_x () =
  let state =
    {
      thread_idx_x = 0l;
      thread_idx_y = 0l;
      thread_idx_z = 0l;
      block_idx_x = 0l;
      block_idx_y = 0l;
      block_idx_z = 0l;
      block_dim_x = 256l;
      block_dim_y = 1l;
      block_dim_z = 1l;
      grid_dim_x = 128l;
      grid_dim_y = 1l;
      grid_dim_z = 1l;
      barrier = (fun () -> ());
    }
  in
  (* global_size_x = grid_dim_x * block_dim_x = 128 * 256 = 32768 *)
  check int32 "global_size_x" 32768l (global_size_x state)

(** Test global_size_y calculation *)
let test_global_size_y () =
  let state =
    {
      thread_idx_x = 0l;
      thread_idx_y = 0l;
      thread_idx_z = 0l;
      block_idx_x = 0l;
      block_idx_y = 0l;
      block_idx_z = 0l;
      block_dim_x = 16l;
      block_dim_y = 16l;
      block_dim_z = 1l;
      grid_dim_x = 4l;
      grid_dim_y = 8l;
      grid_dim_z = 1l;
      barrier = (fun () -> ());
    }
  in
  (* global_size_y = 8 * 16 = 128 *)
  check int32 "global_size_y" 128l (global_size_y state)

(** Test global_size_z calculation *)
let test_global_size_z () =
  let state =
    {
      thread_idx_x = 0l;
      thread_idx_y = 0l;
      thread_idx_z = 0l;
      block_idx_x = 0l;
      block_idx_y = 0l;
      block_idx_z = 0l;
      block_dim_x = 8l;
      block_dim_y = 8l;
      block_dim_z = 4l;
      grid_dim_x = 2l;
      grid_dim_y = 2l;
      grid_dim_z = 3l;
      barrier = (fun () -> ());
    }
  in
  (* global_size_z = 3 * 4 = 12 *)
  check int32 "global_size_z" 12l (global_size_z state)

(** {1 Shared Memory Tests} *)

(** Test creating empty shared memory *)
let test_create_shared () =
  let shared = create_shared () in
  ignore shared ;
  check bool "shared memory created" true true

(** Test allocating int array *)
let test_alloc_shared_int () =
  let shared = create_shared () in
  let arr = alloc_shared_int shared "test_int" 10 0 in
  check int "array length" 10 (Array.length arr) ;
  check int "default value" 0 arr.(0) ;
  arr.(5) <- 42 ;
  check int "written value" 42 arr.(5)

(** Test allocating float array *)
let test_alloc_shared_float () =
  let shared = create_shared () in
  let arr = alloc_shared_float shared "test_float" 8 0.0 in
  check int "array length" 8 (Array.length arr) ;
  check (float 0.0001) "default value" 0.0 arr.(0) ;
  arr.(3) <- 3.14 ;
  check (float 0.0001) "written value" 3.14 arr.(3)

(** Test allocating int32 array *)
let test_alloc_shared_int32 () =
  let shared = create_shared () in
  let arr = alloc_shared_int32 shared "test_i32" 16 0l in
  check int "array length" 16 (Array.length arr) ;
  check int32 "default value" 0l arr.(0) ;
  arr.(10) <- 12345l ;
  check int32 "written value" 12345l arr.(10)

(** Test allocating int64 array *)
let test_alloc_shared_int64 () =
  let shared = create_shared () in
  let arr = alloc_shared_int64 shared "test_i64" 12 0L in
  check int "array length" 12 (Array.length arr) ;
  check int64 "default value" 0L arr.(0) ;
  arr.(7) <- 9876543210L ;
  check int64 "written value" 9876543210L arr.(7)

(** Test reusing same shared array name *)
let test_alloc_shared_reuse () =
  let shared = create_shared () in
  let arr1 = alloc_shared_int shared "same_name" 10 5 in
  arr1.(0) <- 100 ;
  let arr2 = alloc_shared_int shared "same_name" 10 5 in
  (* Should return the same array *)
  check int "reused array value" 100 arr2.(0) ;
  check bool "same array object" true (arr1 == arr2)

(** Test multiple different arrays *)
let test_alloc_shared_multiple () =
  let shared = create_shared () in
  let arr1 = alloc_shared_int shared "array1" 5 1 in
  let arr2 = alloc_shared_int shared "array2" 5 2 in
  let arr3 = alloc_shared_float shared "array3" 5 3.0 in
  check int "array1 default" 1 arr1.(0) ;
  check int "array2 default" 2 arr2.(0) ;
  check (float 0.0001) "array3 default" 3.0 arr3.(0) ;
  arr1.(0) <- 10 ;
  arr2.(0) <- 20 ;
  arr3.(0) <- 30.0 ;
  check int "array1 modified" 10 arr1.(0) ;
  check int "array2 modified" 20 arr2.(0) ;
  check (float 0.0001) "array3 modified" 30.0 arr3.(0)

(** Test generic allocator for custom type *)
type custom_record = {x : int; y : float}

let custom_record_key : custom_record Sarek_ir_types.Type_id.t =
  Sarek_ir_types.Type_id.create ()

type other_record = {z : int32} [@@warning "-69"]

let other_record_key : other_record Sarek_ir_types.Type_id.t =
  Sarek_ir_types.Type_id.create ()

let test_alloc_shared_custom () =
  let shared = create_shared () in
  let default = {x = 0; y = 0.0} in
  let arr = alloc_shared_with_key shared custom_record_key "custom" 4 default in
  check int "array length" 4 (Array.length arr) ;
  check int "default x" 0 arr.(0).x ;
  arr.(2) <- {x = 42; y = 3.14} ;
  check int "custom x" 42 arr.(2).x ;
  check (float 0.0001) "custom y" 3.14 arr.(2).y

let test_alloc_shared_custom_reuse () =
  let shared = create_shared () in
  let default = {x = 0; y = 0.0} in
  let arr1 =
    alloc_shared_with_key shared custom_record_key "custom" 4 default
  in
  arr1.(1) <- {x = 7; y = 2.5} ;
  let arr2 =
    alloc_shared_with_key shared custom_record_key "custom" 4 default
  in
  check int "reused custom x" 7 arr2.(1).x ;
  check (float 0.0001) "reused custom y" 2.5 arr2.(1).y

let test_alloc_shared_custom_type_mismatch () =
  let shared = create_shared () in
  let _ =
    alloc_shared_with_key shared custom_record_key "custom" 4 {x = 0; y = 0.0}
  in
  check_raises
    "same shared name with different custom type"
    (Invalid_argument "alloc_shared: type mismatch for custom")
    (fun () ->
      ignore (alloc_shared_with_key shared other_record_key "custom" 4 {z = 0l}))

(** Test shared memory isolation between blocks *)
let test_shared_memory_isolation () =
  let shared1 = create_shared () in
  let shared2 = create_shared () in
  let arr1 = alloc_shared_int shared1 "data" 5 0 in
  let arr2 = alloc_shared_int shared2 "data" 5 0 in
  arr1.(0) <- 100 ;
  arr2.(0) <- 200 ;
  check int "shared1 value" 100 arr1.(0) ;
  check int "shared2 value" 200 arr2.(0) ;
  check bool "different arrays" true (arr1 != arr2)

(** {1 Sequential Execution Tests} *)

(** Test simple sequential kernel without barriers *)
let test_run_sequential_simple () =
  let results = Array.make 8 0l in
  let kernel state _shared _args =
    let idx = Int32.to_int (global_idx_x state) in
    results.(idx) <- state.thread_idx_x
  in
  run_sequential ~block:(8, 1, 1) ~grid:(1, 1, 1) kernel () ;
  for i = 0 to 7 do
    check
      int32
      (Printf.sprintf "thread %d result" i)
      (Int32.of_int i)
      results.(i)
  done

(** Test sequential kernel with global index calculation *)
let test_run_sequential_global_index () =
  let size = 32 in
  let results = Array.make size 0l in
  let kernel state _shared _args =
    let gid = Int32.to_int (global_idx_x state) in
    results.(gid) <- global_idx_x state
  in
  run_sequential ~block:(16, 1, 1) ~grid:(2, 1, 1) kernel () ;
  for i = 0 to size - 1 do
    check int32 (Printf.sprintf "global_idx %d" i) (Int32.of_int i) results.(i)
  done

(** Test sequential kernel with shared memory *)
let test_run_sequential_with_shared () =
  let results = Array.make 8 0 in
  let kernel state shared _args =
    let tid = Int32.to_int state.thread_idx_x in
    let arr = alloc_shared_int shared "data" 8 0 in
    arr.(tid) <- tid * 10 ;
    (* Read from shared memory *)
    results.(tid) <- arr.(tid)
  in
  run_sequential ~block:(8, 1, 1) ~grid:(1, 1, 1) kernel () ;
  for i = 0 to 7 do
    check int (Printf.sprintf "shared result %d" i) (i * 10) results.(i)
  done

(** Test sequential kernel with 2D block *)
let test_run_sequential_2d () =
  let results = Array.make_matrix 4 4 0l in
  let kernel state _shared _args =
    let x = Int32.to_int state.thread_idx_x in
    let y = Int32.to_int state.thread_idx_y in
    results.(y).(x) <- Int32.add state.thread_idx_x state.thread_idx_y
  in
  run_sequential ~block:(4, 4, 1) ~grid:(1, 1, 1) kernel () ;
  for y = 0 to 3 do
    for x = 0 to 3 do
      check
        int32
        (Printf.sprintf "2d result [%d,%d]" y x)
        (Int32.of_int (x + y))
        results.(y).(x)
    done
  done

(** {1 Parallel Execution Tests} *)

(** Test simple parallel kernel *)
let test_run_parallel_simple () =
  let size = 16 in
  let results = Array.make size 0l in
  let kernel state _shared _args =
    let idx = Int32.to_int (global_idx_x state) in
    results.(idx) <- Int32.mul state.thread_idx_x 2l
  in
  run_parallel ~has_barriers:false ~block:(16, 1, 1) ~grid:(1, 1, 1) kernel () ;
  for i = 0 to size - 1 do
    check
      int32
      (Printf.sprintf "parallel result %d" i)
      (Int32.of_int (i * 2))
      results.(i)
  done

(** Test parallel kernel with shared memory *)
let test_run_parallel_shared () =
  let size = 8 in
  let results = Array.make size 0 in
  let kernel state shared _args =
    let tid = Int32.to_int state.thread_idx_x in
    let arr = alloc_shared_int shared "data" size 0 in
    (* Each thread writes its index *)
    arr.(tid) <- tid + 1 ;
    (* Barrier to ensure all writes complete *)
    state.barrier () ;
    (* Read from next thread (circular) *)
    let next_tid = (tid + 1) mod size in
    results.(tid) <- arr.(next_tid)
  in
  run_parallel ~has_barriers:true ~block:(size, 1, 1) ~grid:(1, 1, 1) kernel () ;
  for i = 0 to size - 1 do
    let expected = ((i + 1) mod size) + 1 in
    check int (Printf.sprintf "parallel shared %d" i) expected results.(i)
  done

(** {1 Barrier Synchronization Tests} *)

(** Test barrier with sequential execution (should be no-op) *)
let test_barrier_sequential () =
  let counter = ref 0 in
  let kernel state _shared _args =
    incr counter ;
    state.barrier () ;
    (* Barrier is no-op in sequential, should just continue *)
    incr counter
  in
  run_sequential ~block:(4, 1, 1) ~grid:(1, 1, 1) kernel () ;
  (* Each of 4 threads increments twice = 8 *)
  check int "counter after barriers" 8 !counter

(** Test barrier synchronization in parallel *)
let test_barrier_parallel_sync () =
  let size = 8 in
  let phase1 = Array.make size false in
  let phase2 = Array.make size false in
  let kernel state _shared _args =
    let tid = Int32.to_int state.thread_idx_x in
    (* Phase 1: mark completion *)
    phase1.(tid) <- true ;
    (* Barrier: all threads must complete phase 1 before phase 2 *)
    state.barrier () ;
    (* Phase 2: check all phase1 are true *)
    let all_done = Array.for_all (fun x -> x) phase1 in
    phase2.(tid) <- all_done
  in
  run_parallel ~block:(size, 1, 1) ~grid:(1, 1, 1) kernel () ;
  (* All threads should see all phase1 completed *)
  for i = 0 to size - 1 do
    check bool (Printf.sprintf "phase1 thread %d" i) true phase1.(i) ;
    check bool (Printf.sprintf "phase2 thread %d" i) true phase2.(i)
  done

(** Test multiple barriers in sequence *)
let test_multiple_barriers () =
  let size = 4 in
  let results = Array.make size 0 in
  let kernel state shared _args =
    let tid = Int32.to_int state.thread_idx_x in
    let arr = alloc_shared_int shared "data" size 0 in
    (* Stage 1 *)
    arr.(tid) <- 1 ;
    state.barrier () ;
    (* Stage 2 *)
    arr.(tid) <- arr.(tid) + 1 ;
    state.barrier () ;
    (* Stage 3 *)
    arr.(tid) <- arr.(tid) + 1 ;
    state.barrier () ;
    (* Final read *)
    results.(tid) <- arr.(tid)
  in
  run_parallel ~block:(size, 1, 1) ~grid:(1, 1, 1) kernel () ;
  for i = 0 to size - 1 do
    check int (Printf.sprintf "multi-barrier %d" i) 3 results.(i)
  done

(** Test barrier metadata fallback does not execute user code for detection *)
let test_barrier_detection_non_mutating () =
  let size = 4 in
  let side_effects = Atomic.make 0 in
  let results = Array.make size 0 in
  let kernel state _shared _args =
    let tid = Int32.to_int state.thread_idx_x in
    let count = Atomic.fetch_and_add side_effects 1 + 1 in
    results.(tid) <- count ;
    state.barrier () ;
    results.(tid) <- results.(tid) + 10
  in
  run_parallel ~block:(size, 1, 1) ~grid:(1, 1, 1) kernel () ;
  check int "side effects occur once per thread" size (Atomic.get side_effects) ;
  for i = 0 to size - 1 do
    check bool (Printf.sprintf "thread %d ran once" i) true (results.(i) > 10)
  done

exception Test_parallel_failure

let test_parallel_default_propagates_worker_exception () =
  let kernel _state _shared _args = raise Test_parallel_failure in
  check_raises
    "worker exception propagates through default barrier-safe path"
    Test_parallel_failure
    (fun () -> run_parallel ~block:(4, 1, 1) ~grid:(1, 1, 1) kernel ())

(** Test that a BSP barrier kernel which raises propagates the exception and
    does not hang (regression for run_block_bsp exnc=raise bug). run_threadpool
    with has_barriers=true exercises run_block_bsp via the fission ThreadPool.
*)
exception Test_bsp_failure

let test_bsp_worker_exception_propagates () =
  let kernel _state _shared _args = raise Test_bsp_failure in
  check_raises
    "BSP worker exception propagates through run_block_bsp"
    Test_bsp_failure
    (fun () ->
      run_threadpool
        ~has_barriers:true
        ~block:(4, 1, 1)
        ~grid:(1, 1, 1)
        kernel
        ())

(** {1 Test Suite} *)

let thread_state_tests =
  [
    ("create basic", `Quick, test_thread_state_creation);
    ("1d max indices", `Quick, test_thread_state_1d_max);
    ("3d indices", `Quick, test_thread_state_3d);
  ]

let global_index_tests =
  [
    ("global_idx_x simple", `Quick, test_global_idx_x_simple);
    ("global_idx_x first block", `Quick, test_global_idx_x_first_block);
    ("global_idx_y 2d", `Quick, test_global_idx_y_2d);
    ("global_idx_z 3d", `Quick, test_global_idx_z_3d);
    ("global_size_x", `Quick, test_global_size_x);
    ("global_size_y", `Quick, test_global_size_y);
    ("global_size_z", `Quick, test_global_size_z);
  ]

let shared_memory_tests =
  [
    ("create shared", `Quick, test_create_shared);
    ("alloc int", `Quick, test_alloc_shared_int);
    ("alloc float", `Quick, test_alloc_shared_float);
    ("alloc int32", `Quick, test_alloc_shared_int32);
    ("alloc int64", `Quick, test_alloc_shared_int64);
    ("reuse array", `Quick, test_alloc_shared_reuse);
    ("multiple arrays", `Quick, test_alloc_shared_multiple);
    ("custom type", `Quick, test_alloc_shared_custom);
    ("custom reuse", `Quick, test_alloc_shared_custom_reuse);
    ("custom mismatch", `Quick, test_alloc_shared_custom_type_mismatch);
    ("isolation", `Quick, test_shared_memory_isolation);
  ]

let sequential_tests =
  [
    ("simple", `Quick, test_run_sequential_simple);
    ("global index", `Quick, test_run_sequential_global_index);
    ("with shared", `Quick, test_run_sequential_with_shared);
    ("2d block", `Quick, test_run_sequential_2d);
  ]

let parallel_tests =
  [
    ("simple", `Quick, test_run_parallel_simple);
    ("with shared", `Quick, test_run_parallel_shared);
  ]

let barrier_tests =
  [
    ("sequential no-op", `Quick, test_barrier_sequential);
    ("parallel sync", `Quick, test_barrier_parallel_sync);
    ("multiple barriers", `Quick, test_multiple_barriers);
    ("non-mutating detection", `Quick, test_barrier_detection_non_mutating);
    ( "default exception propagation",
      `Quick,
      test_parallel_default_propagates_worker_exception );
    ("bsp exception propagation", `Quick, test_bsp_worker_exception_propagates);
  ]

let () =
  Alcotest.run
    "Sarek_cpu_runtime"
    [
      ("Thread State", thread_state_tests);
      ("Global Indices", global_index_tests);
      ("Shared Memory", shared_memory_tests);
      ("Sequential Execution", sequential_tests);
      ("Parallel Execution", parallel_tests);
      ("Barrier Synchronization", barrier_tests);
    ]
