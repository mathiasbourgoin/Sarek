(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * E2E test for Sarek PPX - Parallel Reduction
 *
 * Tests tree-based parallel reduction with shared memory and barriers.
 * Reduction is a fundamental parallel primitive for computing sums, min, max.
 * GPU runtime only.
 ******************************************************************************)

open Sarek
module Std = Sarek_stdlib.Std
module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer
module Benchmarks = Test_helpers.Benchmarks

(* ========== Pure OCaml baselines ========== *)

let ocaml_sum arr n =
  let sum = ref 0.0 in
  for i = 0 to n - 1 do
    sum := !sum +. arr.(i)
  done ;
  !sum

let ocaml_max arr n =
  let m = ref arr.(0) in
  for i = 1 to n - 1 do
    if arr.(i) > !m then m := arr.(i)
  done ;
  !m

let ocaml_dot a b n =
  let sum = ref 0.0 in
  for i = 0 to n - 1 do
    sum := !sum +. (a.(i) *. b.(i))
  done ;
  !sum

(* ========== Shared test data ========== *)

let input_sum = ref [||]

let input_max = ref [||]

let input_a = ref [||]

let input_b = ref [||]

let init_data size =
  input_sum := Array.make size 1.0 ;
  input_max := Array.init size (fun i -> float_of_int i) ;
  input_a := Array.init size (fun i -> float_of_int i) ;
  input_b := Array.make size 1.0

(* ========== Sarek kernels ========== *)

let reduce_sum_kernel =
  [%kernel
    fun (input : float32 vector) (output : float32 vector) (n : int32) ->
      let%shared (sdata : float32) = 256l in
      let tid = thread_idx_x in
      let gid = thread_idx_x + (block_dim_x * block_idx_x) in
      let%superstep load =
        if gid < n then sdata.(tid) <- input.(gid) else sdata.(tid) <- 0.0
      in
      let%superstep reduce128 =
        if tid < 128l then sdata.(tid) <- sdata.(tid) +. sdata.(tid + 128l)
      in
      let%superstep reduce64 =
        if tid < 64l then sdata.(tid) <- sdata.(tid) +. sdata.(tid + 64l)
      in
      let%superstep reduce32 =
        if tid < 32l then sdata.(tid) <- sdata.(tid) +. sdata.(tid + 32l)
      in
      let%superstep reduce16 =
        if tid < 16l then sdata.(tid) <- sdata.(tid) +. sdata.(tid + 16l)
      in
      let%superstep reduce8 =
        if tid < 8l then sdata.(tid) <- sdata.(tid) +. sdata.(tid + 8l)
      in
      let%superstep reduce4 =
        if tid < 4l then sdata.(tid) <- sdata.(tid) +. sdata.(tid + 4l)
      in
      let%superstep reduce2 =
        if tid < 2l then sdata.(tid) <- sdata.(tid) +. sdata.(tid + 2l)
      in
      let%superstep reduce1 =
        if tid < 1l then sdata.(tid) <- sdata.(tid) +. sdata.(tid + 1l)
      in
      if tid = 0l then output.(block_idx_x) <- sdata.(0l)]

let reduce_max_kernel =
  [%kernel
    fun (input : float32 vector) (output : float32 vector) (n : int32) ->
      let%shared (sdata : float32) = 256l in
      let tid = thread_idx_x in
      let gid = thread_idx_x + (block_dim_x * block_idx_x) in
      let%superstep load =
        if gid < n then sdata.(tid) <- input.(gid)
        else sdata.(tid) <- -1000000.0
      in
      let%superstep reduce128 =
        if tid < 128l then begin
          let a = sdata.(tid) in
          let b = sdata.(tid + 128l) in
          if b > a then sdata.(tid) <- b
        end
      in
      let%superstep reduce64 =
        if tid < 64l then begin
          let a = sdata.(tid) in
          let b = sdata.(tid + 64l) in
          if b > a then sdata.(tid) <- b
        end
      in
      let%superstep reduce32 =
        if tid < 32l then begin
          let a = sdata.(tid) in
          let b = sdata.(tid + 32l) in
          if b > a then sdata.(tid) <- b
        end
      in
      let%superstep reduce16 =
        if tid < 16l then begin
          let a = sdata.(tid) in
          let b = sdata.(tid + 16l) in
          if b > a then sdata.(tid) <- b
        end
      in
      let%superstep reduce8 =
        if tid < 8l then begin
          let a = sdata.(tid) in
          let b = sdata.(tid + 8l) in
          if b > a then sdata.(tid) <- b
        end
      in
      let%superstep reduce4 =
        if tid < 4l then begin
          let a = sdata.(tid) in
          let b = sdata.(tid + 4l) in
          if b > a then sdata.(tid) <- b
        end
      in
      let%superstep reduce2 =
        if tid < 2l then begin
          let a = sdata.(tid) in
          let b = sdata.(tid + 2l) in
          if b > a then sdata.(tid) <- b
        end
      in
      let%superstep reduce1 =
        if tid < 1l then begin
          let a = sdata.(tid) in
          let b = sdata.(tid + 1l) in
          if b > a then sdata.(tid) <- b
        end
      in
      if tid = 0l then output.(block_idx_x) <- sdata.(0l)]

let dot_product_kernel =
  [%kernel
    fun (a : float32 vector)
        (b : float32 vector)
        (output : float32 vector)
        (n : int32) ->
      let%shared (sdata : float32) = 256l in
      let tid = thread_idx_x in
      let gid = thread_idx_x + (block_dim_x * block_idx_x) in
      let%superstep load =
        if gid < n then sdata.(tid) <- a.(gid) *. b.(gid)
        else sdata.(tid) <- 0.0
      in
      let%superstep reduce128 =
        if tid < 128l then sdata.(tid) <- sdata.(tid) +. sdata.(tid + 128l)
      in
      let%superstep reduce64 =
        if tid < 64l then sdata.(tid) <- sdata.(tid) +. sdata.(tid + 64l)
      in
      let%superstep reduce32 =
        if tid < 32l then sdata.(tid) <- sdata.(tid) +. sdata.(tid + 32l)
      in
      let%superstep reduce16 =
        if tid < 16l then sdata.(tid) <- sdata.(tid) +. sdata.(tid + 16l)
      in
      let%superstep reduce8 =
        if tid < 8l then sdata.(tid) <- sdata.(tid) +. sdata.(tid + 8l)
      in
      let%superstep reduce4 =
        if tid < 4l then sdata.(tid) <- sdata.(tid) +. sdata.(tid + 4l)
      in
      let%superstep reduce2 =
        if tid < 2l then sdata.(tid) <- sdata.(tid) +. sdata.(tid + 2l)
      in
      let%superstep reduce1 =
        if tid < 1l then sdata.(tid) <- sdata.(tid) +. sdata.(tid + 1l)
      in
      if tid = 0l then output.(block_idx_x) <- sdata.(0l)]

(* ========== Runners ========== *)

let run_sum dev size _block_size =
  if dev.Device.framework = "Interpreter" then
    failwith "Interpreter not supported for reduction kernel" ;

  let n = size in
  (* Kernel expects block_size=256 because of hardcoded reduction steps *)
  let block_sz = 256 in
  let num_blocks = (n + block_sz - 1) / block_sz in
  let inp = !input_sum in

  let _, kirc = reduce_sum_kernel in
  let ir =
    match kirc.Sarek.Kirc_types.body_ir with
    | Some ir -> ir
    | None -> failwith "No IR"
  in

  let input = Vector.create Vector.float32 n in
  let output = Vector.create Vector.float32 num_blocks in

  for i = 0 to n - 1 do
    Vector.set input i inp.(i)
  done ;
  for i = 0 to num_blocks - 1 do
    Vector.set output i 0.0
  done ;

  let block = Sarek.Execute.dims1d block_sz in
  let grid = Sarek.Execute.dims1d num_blocks in

  let t0 = Unix.gettimeofday () in
  Sarek.Execute.run_vectors
    ~device:dev
    ~ir
    ~args:
      [
        Sarek.Execute.Vec input;
        Sarek.Execute.Vec output;
        Sarek.Execute.Int32 (Int32.of_int n);
      ]
    ~block
    ~grid
    () ;
  Transfer.flush dev ;
  let t1 = Unix.gettimeofday () in

  let partial_sums = Vector.to_array output in
  let total = Array.fold_left ( +. ) 0.0 partial_sums in
  ((t1 -. t0) *. 1000.0, total)

let run_max dev size _block_size =
  if dev.Device.framework = "Interpreter" then
    failwith "Interpreter not supported for reduction kernel" ;

  let n = size in
  (* Kernel expects block_size=256 *)
  let block_sz = 256 in
  let num_blocks = (n + block_sz - 1) / block_sz in
  let inp = !input_max in

  let _, kirc = reduce_max_kernel in
  let ir =
    match kirc.Sarek.Kirc_types.body_ir with
    | Some ir -> ir
    | None -> failwith "No IR"
  in

  let input = Vector.create Vector.float32 n in
  let output = Vector.create Vector.float32 num_blocks in

  for i = 0 to n - 1 do
    Vector.set input i inp.(i)
  done ;
  for i = 0 to num_blocks - 1 do
    Vector.set output i (-1000000.0)
  done ;

  let block = Sarek.Execute.dims1d block_sz in
  let grid = Sarek.Execute.dims1d num_blocks in

  let t0 = Unix.gettimeofday () in
  Sarek.Execute.run_vectors
    ~device:dev
    ~ir
    ~args:
      [
        Sarek.Execute.Vec input;
        Sarek.Execute.Vec output;
        Sarek.Execute.Int32 (Int32.of_int n);
      ]
    ~block
    ~grid
    () ;
  Transfer.flush dev ;
  let t1 = Unix.gettimeofday () in

  let partial_maxs = Vector.to_array output in
  let total_max = Array.fold_left max (-1000000.0) partial_maxs in
  ((t1 -. t0) *. 1000.0, total_max)

let run_dot dev size _block_size =
  if dev.Device.framework = "Interpreter" then
    failwith "Interpreter not supported for reduction kernel" ;

  let n = size in
  (* Kernel expects block_size=256 *)
  let block_sz = 256 in
  let num_blocks = (n + block_sz - 1) / block_sz in
  let inp_a = !input_a in
  let inp_b = !input_b in

  let _, kirc = dot_product_kernel in
  let ir =
    match kirc.Sarek.Kirc_types.body_ir with
    | Some ir -> ir
    | None -> failwith "No IR"
  in

  let a = Vector.create Vector.float32 n in
  let b = Vector.create Vector.float32 n in
  let output = Vector.create Vector.float32 num_blocks in

  for i = 0 to n - 1 do
    Vector.set a i inp_a.(i) ;
    Vector.set b i inp_b.(i)
  done ;
  for i = 0 to num_blocks - 1 do
    Vector.set output i 0.0
  done ;

  let block = Sarek.Execute.dims1d block_sz in
  let grid = Sarek.Execute.dims1d num_blocks in

  let t0 = Unix.gettimeofday () in
  Sarek.Execute.run_vectors
    ~device:dev
    ~ir
    ~args:
      [
        Sarek.Execute.Vec a;
        Sarek.Execute.Vec b;
        Sarek.Execute.Vec output;
        Sarek.Execute.Int32 (Int32.of_int n);
      ]
    ~block
    ~grid
    () ;
  Transfer.flush dev ;
  let t1 = Unix.gettimeofday () in

  let partial_sums = Vector.to_array output in
  let total = Array.fold_left ( +. ) 0.0 partial_sums in
  ((t1 -. t0) *. 1000.0, total)

(* ========== Main ========== *)

let () =
  Benchmarks.init () ;
  let size = Benchmarks.config.size in
  init_data size ;

  let verify_float name tol result expected =
    let diff = abs_float (result -. expected) in
    let ok = diff < tol in
    if not ok then
      Printf.printf
        "  %s mismatch: expected %.6f, got %.6f (diff %.6f, tol %.6f)\n"
        name
        expected
        result
        diff
        tol ;
    ok
  in

  (* Sum *)
  let baseline_sum size = ocaml_sum !input_sum size in
  (* Sum of 1.0s: relative error might be needed for large N due to float32 precision *)
  (* For N=4M, expected is 4194304.0. Float32 precision is ~0.25 at this magnitude. *)
  (* But accumulation error grows. Let's allow 1.0 or relative error. *)
  let verify_sum result expected =
    let tol = max 1.0 (expected *. 0.0001) in
    verify_float "Sum" tol result expected
  in
  (* Exclude interpreter - too slow for large reductions *)
  Benchmarks.run
    ~baseline:baseline_sum
    ~verify:verify_sum
    ~filter:Benchmarks.no_interpreter
    "Reduce Sum"
    run_sum ;

  (* Max *)
  let baseline_max size = ocaml_max !input_max size in
  (* Max should be exact for integers < 2^24 *)
  Benchmarks.run
    ~filter:Benchmarks.no_interpreter
    ~baseline:baseline_max
    ~verify:(verify_float "Max" 0.1)
    "Reduce Max"
    run_max ;

  (* Dot *)
  let baseline_dot size = ocaml_dot !input_a !input_b size in
  (* Dot product grows as N^2. For N=4M, result is ~8e12. *)
  (* Float32 has 7 sig figs. 8e12 has precision ~1e6. *)
  (* We need a very loose tolerance or relative error. *)
  let verify_dot result expected =
    let tol = max 1.0 (expected *. 0.001) in
    (* 0.1% error *)
    verify_float "Dot" tol result expected
  in
  Benchmarks.run
    ~baseline:baseline_dot
    ~verify:verify_dot
    ~filter:Benchmarks.no_interpreter
    "Dot Product"
    run_dot ;
  Benchmarks.exit ()
