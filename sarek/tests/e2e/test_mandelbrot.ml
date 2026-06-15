(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * E2E test for Sarek PPX - Mandelbrot set computation
 *
 * Tests iterative computation with complex arithmetic.
 * Mandelbrot is a classic GPU benchmark with high arithmetic intensity.
 *
 * GPU runtime only.
 ******************************************************************************)

module Std = Sarek_stdlib.Std
module Benchmarks = Test_helpers.Benchmarks

(* runtime module aliases *)
module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer

(* Force backend registration *)

let max_iter = ref 256

(* ========== Pure OCaml baselines ========== *)

let ocaml_mandelbrot output width height max_iter =
  for py = 0 to height - 1 do
    for px = 0 to width - 1 do
      let x0 = (4.0 *. float_of_int px /. float_of_int width) -. 2.5 in
      let y0 = (3.0 *. float_of_int py /. float_of_int height) -. 1.5 in
      let x = ref 0.0 in
      let y = ref 0.0 in
      let iter = ref 0 in
      while (!x *. !x) +. (!y *. !y) <= 4.0 && !iter < max_iter do
        let xtemp = (!x *. !x) -. (!y *. !y) +. x0 in
        y := (2.0 *. !x *. !y) +. y0 ;
        x := xtemp ;
        incr iter
      done ;
      output.((py * width) + px) <- Int32.of_int !iter
    done
  done

let ocaml_julia output width height max_iter =
  let c_re = -0.8 in
  let c_im = 0.156 in
  for py = 0 to height - 1 do
    for px = 0 to width - 1 do
      let x = ref ((4.0 *. float_of_int px /. float_of_int width) -. 2.0) in
      let y = ref ((3.0 *. float_of_int py /. float_of_int height) -. 1.5) in
      let iter = ref 0 in
      while (!x *. !x) +. (!y *. !y) <= 4.0 && !iter < max_iter do
        let xtemp = (!x *. !x) -. (!y *. !y) +. c_re in
        y := (2.0 *. !x *. !y) +. c_im ;
        x := xtemp ;
        incr iter
      done ;
      output.((py * width) + px) <- Int32.of_int !iter
    done
  done

(* ========== Shared test data ========== *)

let expected_mandelbrot = ref [||]

let init_mandelbrot_data size =
  let dim = int_of_float (sqrt (float_of_int size)) in
  let n = dim * dim in
  let output = Array.make n 0l in
  expected_mandelbrot := output

(* ========== Mandelbrot kernel ========== *)

let mandelbrot_kernel =
  [%kernel
    fun (output : int32 vector)
        (width : int32)
        (height : int32)
        (max_iter : int32) ->
      let open Std in
      let px = global_idx_x in
      let py = global_idx_y in
      if px < width && py < height then begin
        let x0 = (4.0 *. (float px /. float width)) -. 2.5 in
        let y0 = (3.0 *. (float py /. float height)) -. 1.5 in
        let x = mut 0.0 in
        let y = mut 0.0 in
        let iter = mut 0l in
        while (x *. x) +. (y *. y) <= 4.0 && iter < max_iter do
          let xtemp = (x *. x) -. (y *. y) +. x0 in
          y := (2.0 *. x *. y) +. y0 ;
          x := xtemp ;
          iter := iter + 1l
        done ;
        output.((py * width) + px) <- iter
      end]

let run_mandelbrot_test (dev : Device.t) size _block_size =
  let _, kirc = mandelbrot_kernel in
  let ir =
    match kirc.Sarek.Kirc_types.body_ir with
    | Some ir -> ir
    | None -> failwith "Mandelbrot kernel has no IR"
  in

  let dim = int_of_float (sqrt (float_of_int size)) in
  let width = dim in
  let height = dim in
  let n = width * height in

  let output = Vector.create Vector.int32 n in

  for i = 0 to n - 1 do
    Vector.set output i 0l
  done ;

  let block_size = 16 in
  let blocks_x = (width + block_size - 1) / block_size in
  let blocks_y = (height + block_size - 1) / block_size in
  let block = Sarek.Execute.dims2d block_size block_size in
  let grid = Sarek.Execute.dims2d blocks_x blocks_y in

  (* Warm up: run once to trigger JIT compilation (cached for subsequent runs) *)
  Sarek.Execute.run_vectors
    ~device:dev
    ~ir
    ~args:
      [
        Sarek.Execute.Vec output;
        Sarek.Execute.Int width;
        Sarek.Execute.Int height;
        Sarek.Execute.Int !max_iter;
      ]
    ~block
    ~grid
    () ;
  Transfer.flush dev ;

  (* Reset output for timed run *)
  for i = 0 to n - 1 do
    Vector.set output i 0l
  done ;

  let t0 = Unix.gettimeofday () in
  Sarek.Execute.run_vectors
    ~device:dev
    ~ir
    ~args:
      [
        Sarek.Execute.Vec output;
        Sarek.Execute.Int width;
        Sarek.Execute.Int height;
        Sarek.Execute.Int !max_iter;
      ]
    ~block
    ~grid
    () ;
  Transfer.flush dev ;
  let t1 = Unix.gettimeofday () in
  let time_ms = (t1 -. t0) *. 1000.0 in

  (time_ms, Vector.to_array output)

(* ========== Tail-recursive Mandelbrot ========== *)

let mandelbrot_tailrec_kernel =
  [%kernel
    (* Tail-recursive iteration using module function *)
    let open Std in
    let rec iterate (x : float32) (y : float32) (x0 : float32) (y0 : float32)
        (iter : int32) (max_iter : int32) : int32 =
      if (x *. x) +. (y *. y) > 4.0 || iter >= max_iter then iter
      else
        let xtemp = (x *. x) -. (y *. y) +. x0 in
        let ynew = (2.0 *. x *. y) +. y0 in
        iterate xtemp ynew x0 y0 (iter + 1l) max_iter
    in

    fun (output : int32 vector)
        (width : int32)
        (height : int32)
        (max_iter : int32)
      ->
      let px = global_idx_x in
      let py = global_idx_y in
      if px < width && py < height then begin
        let x0 = (4.0 *. (float px /. float width)) -. 2.5 in
        let y0 = (3.0 *. (float py /. float height)) -. 1.5 in
        let iter = iterate 0.0 0.0 x0 y0 0l max_iter in
        output.((py * width) + px) <- iter
      end]

let run_mandelbrot_tailrec_test (dev : Device.t) size _block_size =
  let _, kirc = mandelbrot_tailrec_kernel in
  let ir =
    match kirc.Sarek.Kirc_types.body_ir with
    | Some ir -> ir
    | None -> failwith "Mandelbrot tailrec kernel has no IR"
  in

  let dim = int_of_float (sqrt (float_of_int size)) in
  let width = dim in
  let height = dim in
  let n = width * height in

  let output = Vector.create Vector.int32 n in

  for i = 0 to n - 1 do
    Vector.set output i 0l
  done ;

  let block_size = 16 in
  let blocks_x = (width + block_size - 1) / block_size in
  let blocks_y = (height + block_size - 1) / block_size in
  let block = Sarek.Execute.dims2d block_size block_size in
  let grid = Sarek.Execute.dims2d blocks_x blocks_y in

  (* Warm up: run once to trigger JIT compilation *)
  Sarek.Execute.run_vectors
    ~device:dev
    ~ir
    ~args:
      [
        Sarek.Execute.Vec output;
        Sarek.Execute.Int width;
        Sarek.Execute.Int height;
        Sarek.Execute.Int !max_iter;
      ]
    ~block
    ~grid
    () ;
  Transfer.flush dev ;

  (* Reset output for timed run *)
  for i = 0 to n - 1 do
    Vector.set output i 0l
  done ;

  let t0 = Unix.gettimeofday () in
  Sarek.Execute.run_vectors
    ~device:dev
    ~ir
    ~args:
      [
        Sarek.Execute.Vec output;
        Sarek.Execute.Int width;
        Sarek.Execute.Int height;
        Sarek.Execute.Int !max_iter;
      ]
    ~block
    ~grid
    () ;
  Transfer.flush dev ;
  let t1 = Unix.gettimeofday () in
  let time_ms = (t1 -. t0) *. 1000.0 in

  (time_ms, Vector.to_array output)

(* ========== Julia Set ========== *)

let julia_c_re = -0.8

let julia_c_im = 0.156

let julia_kernel =
  [%kernel
    fun (output : int32 vector) (width : int) (height : int) (max_iter : int) ->
      let open Std in
      let px = global_idx_x in
      let py = global_idx_y in
      if px < width && py < height then begin
        let x = mut ((4.0 *. (float px /. float width)) -. 2.0) in
        let y = mut ((3.0 *. (float py /. float height)) -. 1.5) in
        let c_re = -0.8 in
        let c_im = 0.156 in
        let iter = mut 0l in
        while (x *. x) +. (y *. y) <= 4.0 && iter < max_iter do
          let xtemp = (x *. x) -. (y *. y) +. c_re in
          y := (2.0 *. x *. y) +. c_im ;
          x := xtemp ;
          iter := iter + 1l
        done ;
        output.((py * width) + px) <- iter
      end]

let run_julia_test (dev : Device.t) size _block_size =
  let _, kirc = julia_kernel in
  let ir =
    match kirc.Sarek.Kirc_types.body_ir with
    | Some ir -> ir
    | None -> failwith "Julia kernel has no IR"
  in

  let dim = int_of_float (sqrt (float_of_int size)) in
  let width = dim in
  let height = dim in
  let n = width * height in

  let output = Vector.create Vector.int32 n in

  let block_size = 16 in
  let blocks_x = (width + block_size - 1) / block_size in
  let blocks_y = (height + block_size - 1) / block_size in
  let block = Sarek.Execute.dims2d block_size block_size in
  let grid = Sarek.Execute.dims2d blocks_x blocks_y in

  (* Warm up *)
  Sarek.Execute.run_vectors
    ~device:dev
    ~ir
    ~args:
      [
        Sarek.Execute.Vec output;
        Sarek.Execute.Int width;
        Sarek.Execute.Int height;
        Sarek.Execute.Int !max_iter;
      ]
    ~block
    ~grid
    () ;
  Transfer.flush dev ;

  let t0 = Unix.gettimeofday () in
  Sarek.Execute.run_vectors
    ~device:dev
    ~ir
    ~args:
      [
        Sarek.Execute.Vec output;
        Sarek.Execute.Int width;
        Sarek.Execute.Int height;
        Sarek.Execute.Int !max_iter;
      ]
    ~block
    ~grid
    () ;
  Transfer.flush dev ;
  let t1 = Unix.gettimeofday () in
  let time_ms = (t1 -. t0) *. 1000.0 in

  (time_ms, Vector.to_array output)

let () =
  Benchmarks.init () ;
  let size = Benchmarks.config.size in
  init_mandelbrot_data size ;

  let baseline size =
    let dim = int_of_float (sqrt (float_of_int size)) in
    let n = dim * dim in
    let output = Array.make n 0l in
    ocaml_mandelbrot output dim dim !max_iter ;
    output
  in

  let verify result expected =
    let n = Array.length expected in
    let errors = ref 0 in
    for i = 0 to min 1000 (n - 1) do
      if result.(i) <> expected.(i) then begin
        if !errors < 5 then
          Printf.printf
            "Mismatch at %d: expected %ld, got %ld\n"
            i
            expected.(i)
            result.(i) ;
        incr errors
      end
    done ;
    !errors = 0
  in

  let verify_fuzzy result expected =
    let n = Array.length expected in
    let errors = ref 0 in
    for i = 0 to min 1000 (n - 1) do
      let diff = abs (Int32.to_int result.(i) - Int32.to_int expected.(i)) in
      (* Allow significant deviation due to fp32 vs fp64 divergence *)
      if diff > 50 then begin
        if !errors < 5 then
          Printf.printf
            "Mismatch at %d: expected %ld, got %ld (diff %d)\n"
            i
            expected.(i)
            result.(i)
            diff ;
        incr errors
      end
    done ;
    !errors = 0
  in

  (* Mandelbrot is way too slow on interpreter (134s) - exclude it *)
  Benchmarks.run
    ~baseline
    ~verify
    ~filter:Benchmarks.no_interpreter
    "Mandelbrot"
    run_mandelbrot_test ;
  Benchmarks.run
    ~baseline
    ~verify
    ~filter:Benchmarks.no_interpreter
    "Mandelbrot (Tail Rec)"
    run_mandelbrot_tailrec_test ;
  let baseline_julia size =
    let dim = int_of_float (sqrt (float_of_int size)) in
    let n = dim * dim in
    let output = Array.make n 0l in
    ocaml_julia output dim dim !max_iter ;
    output
  in

  Benchmarks.run
    ~baseline:baseline_julia
    ~verify:verify_fuzzy
    ~filter:Benchmarks.no_interpreter
    "Julia Set"
    run_julia_test ;
  Benchmarks.exit ()
