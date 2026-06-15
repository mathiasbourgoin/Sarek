(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * E2E test for Sarek PPX - Matrix Multiplication (runtime)
 *
 * Tests naive and tiled matrix multiplication with shared memory.
 * Matrix multiplication is the canonical GPU compute benchmark.
 *
 * Uses runtime path for all backends (CUDA, OpenCL, Native, Interpreter).
 * Tiled kernel demonstrates shared memory + supersteps (barriers).
 ******************************************************************************)

open Sarek
module Std = Sarek_stdlib.Std
module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer
module Benchmarks = Test_helpers.Benchmarks

(* ========== Pure OCaml baseline ========== *)

let ocaml_matmul a b c m n k =
  for row = 0 to m - 1 do
    for col = 0 to n - 1 do
      let sum = ref 0.0 in
      for i = 0 to k - 1 do
        sum := !sum +. (a.((row * k) + i) *. b.((i * n) + col))
      done ;
      c.((row * n) + col) <- !sum
    done
  done

(* ========== Shared test data ========== *)

let input_a = ref [||]

let input_b = ref [||]

let expected_c = ref [||]

let init_data size =
  let dim = int_of_float (sqrt (float_of_int size)) in
  let m, n, k = (dim, dim, dim) in
  let a = Array.init (m * k) (fun i -> float_of_int (i mod 10) /. 10.0) in
  let b = Array.init (k * n) (fun i -> float_of_int ((i + 1) mod 10) /. 10.0) in
  let c = Array.make (m * n) 0.0 in
  input_a := a ;
  input_b := b ;
  expected_c := c ;
  (* We don't compute baseline here, we let Benchmarks do it *)
  ()

(* ========== Sarek kernels ========== *)

let matmul_naive_kernel =
  [%kernel
    fun (a : float32 vector)
        (b : float32 vector)
        (c : float32 vector)
        (m : int32)
        (n : int32)
        (k : int32) ->
      let open Std in
      let tid = global_thread_id in
      let row = tid / n in
      let col = tid mod n in
      if row < m && col < n then begin
        let sum = mut 0.0 in
        for i = 0 to k - 1l do
          sum := sum +. (a.((row * k) + i) *. b.((i * n) + col))
        done ;
        c.((row * n) + col) <- sum
      end]

let matmul_tiled_kernel =
  [%kernel
    fun (a : float32 vector)
        (b : float32 vector)
        (c : float32 vector)
        (m : int32)
        (n : int32)
        (k : int32) ->
      let%shared (tile_a : float32) = 256l in
      let%shared (tile_b : float32) = 256l in
      let tx = thread_idx_x in
      let ty = thread_idx_y in
      let row = ty + (block_dim_y * block_idx_y) in
      let col = tx + (block_dim_x * block_idx_x) in
      let tile_size = 16l in
      let num_tiles = (k + tile_size - 1l) / tile_size in
      let sum = mut 0.0 in
      for t = 0 to num_tiles - 1l do
        let%superstep load_a =
          let a_col = (t * tile_size) + tx in
          if row < m && a_col < k then
            tile_a.((ty * tile_size) + tx) <- a.((row * k) + a_col)
          else tile_a.((ty * tile_size) + tx) <- 0.0
        in
        let%superstep load_b =
          let b_row = (t * tile_size) + ty in
          if b_row < k && col < n then
            tile_b.((ty * tile_size) + tx) <- b.((b_row * n) + col)
          else tile_b.((ty * tile_size) + tx) <- 0.0
        in
        let%superstep _compute =
          for i = 0 to tile_size - 1l do
            sum :=
              sum
              +. (tile_a.((ty * tile_size) + i) *. tile_b.((i * tile_size) + tx))
          done
        in
        ()
      done ;
      if row < m && col < n then c.((row * n) + col) <- sum]

(* ========== Runners ========== *)

let run_naive dev size _block_size =
  let dim = int_of_float (sqrt (float_of_int size)) in
  let m, n, k = (dim, dim, dim) in
  let inp_a = !input_a in
  let inp_b = !input_b in

  let _, kirc = matmul_naive_kernel in
  let ir =
    match kirc.Sarek.Kirc_types.body_ir with
    | Some ir -> ir
    | None -> failwith "No IR"
  in

  let a = Vector.create Vector.float32 (m * k) in
  let b = Vector.create Vector.float32 (k * n) in
  let c = Vector.create Vector.float32 (m * n) in

  for i = 0 to (m * k) - 1 do
    Vector.set a i inp_a.(i)
  done ;
  for i = 0 to (k * n) - 1 do
    Vector.set b i inp_b.(i)
  done ;
  for i = 0 to (m * n) - 1 do
    Vector.set c i 0.0
  done ;

  let block_sz = 256 in
  let total_elements = m * n in
  let grid_sz = (total_elements + block_sz - 1) / block_sz in
  let block = Sarek.Execute.dims1d block_sz in
  let grid = Sarek.Execute.dims1d grid_sz in

  let t0 = Unix.gettimeofday () in
  Sarek.Execute.run_vectors
    ~device:dev
    ~ir
    ~args:
      [
        Sarek.Execute.Vec a;
        Sarek.Execute.Vec b;
        Sarek.Execute.Vec c;
        Sarek.Execute.Int32 (Int32.of_int m);
        Sarek.Execute.Int32 (Int32.of_int n);
        Sarek.Execute.Int32 (Int32.of_int k);
      ]
    ~block
    ~grid
    () ;
  Transfer.flush dev ;
  let t1 = Unix.gettimeofday () in

  ((t1 -. t0) *. 1000.0, Vector.to_array c)

let run_tiled dev size _block_size =
  if dev.Device.framework = "Interpreter" then
    failwith "Interpreter not supported for tiled kernel" ;

  let dim = int_of_float (sqrt (float_of_int size)) in
  let m, n, k = (dim, dim, dim) in
  let inp_a = !input_a in
  let inp_b = !input_b in

  let _, kirc = matmul_tiled_kernel in
  let ir =
    match kirc.Sarek.Kirc_types.body_ir with
    | Some ir -> ir
    | None -> failwith "No IR"
  in

  let a = Vector.create Vector.float32 (m * k) in
  let b = Vector.create Vector.float32 (k * n) in
  let c = Vector.create Vector.float32 (m * n) in

  for i = 0 to (m * k) - 1 do
    Vector.set a i inp_a.(i)
  done ;
  for i = 0 to (k * n) - 1 do
    Vector.set b i inp_b.(i)
  done ;
  for i = 0 to (m * n) - 1 do
    Vector.set c i 0.0
  done ;

  let tile_sz = 16 in
  let blocks_x = (n + tile_sz - 1) / tile_sz in
  let blocks_y = (m + tile_sz - 1) / tile_sz in
  let block = Sarek.Execute.dims2d tile_sz tile_sz in
  let grid = Sarek.Execute.dims2d blocks_x blocks_y in
  let shared_mem = 2 * tile_sz * tile_sz * 4 in

  let t0 = Unix.gettimeofday () in
  Sarek.Execute.run_vectors
    ~device:dev
    ~ir
    ~args:
      [
        Sarek.Execute.Vec a;
        Sarek.Execute.Vec b;
        Sarek.Execute.Vec c;
        Sarek.Execute.Int32 (Int32.of_int m);
        Sarek.Execute.Int32 (Int32.of_int n);
        Sarek.Execute.Int32 (Int32.of_int k);
      ]
    ~block
    ~grid
    ~shared_mem
    () ;
  Transfer.flush dev ;
  let t1 = Unix.gettimeofday () in

  ((t1 -. t0) *. 1000.0, Vector.to_array c)

(* ========== Main ========== *)

let () =
  Benchmarks.init () ;
  let size = Benchmarks.config.size in
  init_data size ;

  let baseline size =
    let dim = int_of_float (sqrt (float_of_int size)) in
    let m, n, k = (dim, dim, dim) in
    let a = !input_a in
    let b = !input_b in
    let c = Array.make (m * n) 0.0 in
    ocaml_matmul a b c m n k ;
    c
  in

  let verify result expected =
    let size = Array.length expected in
    let errors = ref 0 in
    let check_count = min 100 size in
    for i = 0 to check_count - 1 do
      let diff = abs_float (result.(i) -. expected.(i)) in
      let rel_err =
        if abs_float expected.(i) > 1e-6 then diff /. abs_float expected.(i)
        else diff
      in
      if diff > 0.1 && rel_err > 0.001 then begin
        if !errors < 5 then
          Printf.printf
            "  Mismatch at %d: expected %.6f, got %.6f\n"
            i
            expected.(i)
            result.(i) ;
        incr errors
      end
    done ;
    !errors = 0
  in

  (* Matrix mul is too slow on interpreter - exclude it *)
  Benchmarks.run
    ~baseline
    ~verify
    ~filter:Benchmarks.no_interpreter
    "Naive Matrix Mul"
    run_naive ;
  (* Tiled kernel uses shared memory which interpreter doesn't support *)
  Benchmarks.run
    ~baseline
    ~verify
    ~filter:Benchmarks.no_interpreter
    "Tiled Matrix Mul"
    run_tiled ;
  Benchmarks.exit ()
