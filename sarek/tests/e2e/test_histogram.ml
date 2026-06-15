(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * E2E test for Sarek PPX - Histogram Computation
 *
 * Tests histogram computation using shared memory, supersteps, and atomics.
 * Histogram is a common pattern for data analysis and image processing.
 ******************************************************************************)

open Sarek
module Std = Sarek_stdlib.Std
module Gpu = Sarek_stdlib.Gpu
module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer
module Benchmarks = Test_helpers.Benchmarks

(* Backends auto-register when linked; Benchmarks.init() ensures initialization *)

(* ========== Pure OCaml baseline ========== *)

let ocaml_histogram input n bins =
  let hist = Array.make bins 0l in
  for i = 0 to n - 1 do
    let bin = Int32.to_int (Int32.rem input.(i) (Int32.of_int bins)) in
    hist.(bin) <- Int32.add hist.(bin) 1l
  done ;
  hist

(* ========== Shared test data ========== *)

let num_bins = 256

let input_data = ref [||]

let init_histogram_data size =
  Random.init 42 ;
  input_data := Array.init size (fun _ -> Int32.of_int (Random.int num_bins))

(* ========== Sarek kernels ========== *)

(* Simple histogram: each thread processes one element *)
let histogram_kernel =
  [%kernel
    fun (input : int32 vector)
        (histogram : int32 vector)
        (n : int32)
        (num_bins : int32) ->
      let open Std in
      let open Gpu in
      let%shared (local_hist : int32) = 256l in
      let tid = thread_idx_x in
      let gid = global_thread_id in
      (* Initialize local histogram *)
      let%superstep init = if tid < num_bins then local_hist.(tid) <- 0l in
      (* Count in local histogram using atomic increment *)
      let%superstep[@divergent] count =
        if gid < n then begin
          let bin = input.(gid) mod num_bins in
          let _old = atomic_add_int32 local_hist bin 1l in
          ()
        end
      in
      (* Merge to global histogram using atomics *)
      let%superstep[@divergent] merge =
        if tid < num_bins then begin
          let _old = atomic_add_global_int32 histogram tid local_hist.(tid) in
          ()
        end
      in
      ()]

(* Strided histogram: each thread processes multiple elements *)
let histogram_strided_kernel =
  [%kernel
    fun (input : int32 vector)
        (histogram : int32 vector)
        (n : int32)
        (num_bins : int32) ->
      let open Std in
      let open Gpu in
      let%shared (local_hist : int32) = 256l in
      let tid = thread_idx_x in
      let gid = global_thread_id in
      let stride = block_dim_x * grid_dim_x in
      (* Initialize local histogram *)
      let%superstep init = if tid < num_bins then local_hist.(tid) <- 0l in
      (* Count with stride using atomic increment *)
      let i = mut gid in
      while i < n do
        let bin = input.(i) mod num_bins in
        let _old = atomic_add_int32 local_hist bin 1l in
        i := i + stride
      done ;
      block_barrier () ;
      (* Merge to global histogram using atomics *)
      if tid < num_bins then begin
        let _old = atomic_add_global_int32 histogram tid local_hist.(tid) in
        ()
      end]

(* ========== Runtime test runners ========== *)

let run_histogram (dev : Device.t) size _block_size =
  let n = size in
  let bins = num_bins in
  let _, kirc = histogram_kernel in
  let ir =
    match kirc.Sarek.Kirc_types.body_ir with
    | Some ir -> ir
    | None -> failwith "No IR"
  in

  let input = Vector.create Vector.int32 n in
  let histogram = Vector.create Vector.int32 bins in

  for i = 0 to n - 1 do
    Vector.set input i !input_data.(i)
  done ;
  for i = 0 to bins - 1 do
    Vector.set histogram i 0l
  done ;

  let block_size = 256 in
  let grid_size = (n + block_size - 1) / block_size in
  let block = Sarek.Execute.dims1d block_size in
  let grid = Sarek.Execute.dims1d grid_size in
  let shared_mem = bins * 4 in

  let t0 = Unix.gettimeofday () in
  Sarek.Execute.run_vectors
    ~device:dev
    ~ir
    ~args:
      [
        Sarek.Execute.Vec input;
        Sarek.Execute.Vec histogram;
        Sarek.Execute.Int32 (Int32.of_int n);
        Sarek.Execute.Int32 (Int32.of_int bins);
      ]
    ~block
    ~grid
    ~shared_mem
    () ;
  Transfer.flush dev ;
  let t1 = Unix.gettimeofday () in

  ((t1 -. t0) *. 1000.0, Vector.to_array histogram)

let run_histogram_strided (dev : Device.t) size _block_size =
  let n = size in
  let bins = num_bins in
  let _, kirc = histogram_strided_kernel in
  let ir =
    match kirc.Sarek.Kirc_types.body_ir with
    | Some ir -> ir
    | None -> failwith "No IR"
  in

  let input = Vector.create Vector.int32 n in
  let histogram = Vector.create Vector.int32 bins in

  for i = 0 to n - 1 do
    Vector.set input i !input_data.(i)
  done ;
  for i = 0 to bins - 1 do
    Vector.set histogram i 0l
  done ;

  let block_size = 256 in
  let grid_size = min 64 ((n + block_size - 1) / block_size) in
  let block = Sarek.Execute.dims1d block_size in
  let grid = Sarek.Execute.dims1d grid_size in
  let shared_mem = bins * 4 in

  let t0 = Unix.gettimeofday () in
  Sarek.Execute.run_vectors
    ~device:dev
    ~ir
    ~args:
      [
        Sarek.Execute.Vec input;
        Sarek.Execute.Vec histogram;
        Sarek.Execute.Int32 (Int32.of_int n);
        Sarek.Execute.Int32 (Int32.of_int bins);
      ]
    ~block
    ~grid
    ~shared_mem
    () ;
  Transfer.flush dev ;
  let t1 = Unix.gettimeofday () in

  ((t1 -. t0) *. 1000.0, Vector.to_array histogram)

(* ========== Main ========== *)

let () =
  Benchmarks.init () ;
  let size = Benchmarks.config.size in
  init_histogram_data size ;

  let baseline size =
    let inp = !input_data in
    ocaml_histogram inp size num_bins
  in

  let verify result expected =
    let bins = Array.length expected in
    let errors = ref 0 in
    for i = 0 to bins - 1 do
      if result.(i) <> expected.(i) then begin
        if !errors < 5 then
          Printf.printf
            "  Bin %d: expected %ld, got %ld\n"
            i
            expected.(i)
            result.(i) ;
        incr errors
      end
    done ;
    !errors = 0
  in

  (* Histogram uses shared memory atomics - interpreter doesn't support it *)
  Benchmarks.run
    ~baseline
    ~verify
    ~filter:Benchmarks.no_interpreter
    "Histogram"
    run_histogram ;
  Benchmarks.run
    ~baseline
    ~verify
    ~filter:Benchmarks.no_interpreter
    "Histogram (strided)"
    run_histogram_strided ;
  Benchmarks.exit ()
