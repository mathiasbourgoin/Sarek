(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * E2E test for Sarek PPX - Matrix Transpose
 *
 * Tests naive matrix transpose with 1D kernel.
 * Transpose is a memory-bound operation that benefits from coalescing.
 *
 * Also tests polymorphic transpose using a polymorphic helper function
 * monomorphized at different types: int32, float32, float64, and point3d.
 ******************************************************************************)

[@@@warning "-32"]

open Sarek
module Std = Sarek_stdlib.Std

(* Module aliases *)
module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer
module Benchmarks = Test_helpers.Benchmarks

(* Backends auto-register when linked; Benchmarks.init() ensures initialization *)

(* ========== Custom type for polymorphism test ========== *)

type float32 = float

type point3d = {x : float32; y : float32; z : float32} [@@sarek.type]

(* ========== Pure OCaml baselines ========== *)

let ocaml_transpose_float input output width height =
  for row = 0 to height - 1 do
    for col = 0 to width - 1 do
      let in_idx = (row * width) + col in
      let out_idx = (col * height) + row in
      output.(out_idx) <- input.(in_idx)
    done
  done

let ocaml_transpose_int32 input output width height =
  for row = 0 to height - 1 do
    for col = 0 to width - 1 do
      let in_idx = (row * width) + col in
      let out_idx = (col * height) + row in
      output.(out_idx) <- input.(in_idx)
    done
  done

(* ========== Shared test data ========== *)

let input_data_float32 = ref [||]

let input_data_float64 = ref [||]

let input_data_int32 = ref [||]

let input_data_point3d = ref [||]

let init_transpose_data size =
  let dim = Int32.to_int (Int32.of_float (sqrt (float_of_int size))) in
  let n = dim * dim in
  input_data_float32 := Array.init n (fun i -> float_of_int i) ;
  input_data_float64 := Array.init n (fun i -> float_of_int i) ;
  input_data_int32 := Array.init n (fun i -> Int32.of_int i) ;
  input_data_point3d :=
    Array.init n (fun i ->
        let f = float_of_int i in
        (f, f *. 2.0, f *. 3.0))

(* ========== Module-level polymorphic transpose function ========== *)

(* Type alias for vector - needed for [@sarek.module] outside of kernel context.
   Note: We use 'a array as the OCaml-level type to make the syntax type-check,
   but the PPX treats it as a GPU vector. *)
type 'a vector = 'a array

(* Define the polymorphic transpose helper at module level, shared by all kernels.
   This is monomorphized separately for each concrete type at the call sites.
   Using Array.get/Array.set syntax to satisfy OCaml's typechecker. *)
let[@sarek.module] do_transpose (input : 'a vector) (output : 'a vector)
    (width : int) (height : int) (tid : int) : unit =
  let n = width * height in
  if tid < n then begin
    let col = tid mod width in
    let row = tid / width in
    let in_idx = (row * width) + col in
    let out_idx = (col * height) + row in
    output.(out_idx) <- input.(in_idx)
  end

(* ========== Sarek kernels using the shared polymorphic transpose ========== *)

(* Polymorphic transpose - monomorphized at int32 *)
let transpose_int32_kernel =
  [%kernel
    fun (input : int32 vector)
        (output : int32 vector)
        (width : int32)
        (height : int32) ->
      let open Std in
      do_transpose input output width height global_thread_id]

(* Polymorphic transpose - monomorphized at float32 *)
let transpose_float32_kernel =
  [%kernel
    fun (input : float32 vector)
        (output : float32 vector)
        (width : int32)
        (height : int32) ->
      let open Std in
      do_transpose input output width height global_thread_id]

(* Polymorphic transpose - monomorphized at float64 *)
let transpose_float64_kernel =
  [%kernel
    fun (input : float64 vector)
        (output : float64 vector)
        (width : int32)
        (height : int32) ->
      let open Std in
      do_transpose input output width height global_thread_id]

(* Polymorphic transpose - monomorphized at point3d record *)
let transpose_point3d_kernel =
  [%kernel
    fun (input : point3d vector)
        (output : point3d vector)
        (width : int32)
        (height : int32) ->
      let open Std in
      do_transpose input output width height global_thread_id]

(* ========== Runtime test runners ========== *)

(* Int32 transpose *)
let run_transpose_int32 (dev : Device.t) size _block_size =
  let dim = Int32.to_int (Int32.of_float (sqrt (float_of_int size))) in
  let n = dim * dim in
  let _, kirc = transpose_int32_kernel in
  let ir =
    match kirc.Sarek.Kirc_types.body_ir with
    | Some ir -> ir
    | None -> failwith "No IR"
  in

  let input = Vector.create Vector.int32 n in
  let output = Vector.create Vector.int32 n in

  for i = 0 to n - 1 do
    Vector.set input i !input_data_int32.(i) ;
    Vector.set output i 0l
  done ;

  let block_size = 256 in
  let grid_size = (n + block_size - 1) / block_size in
  let block = Sarek.Execute.dims1d block_size in
  let grid = Sarek.Execute.dims1d grid_size in

  let t0 = Unix.gettimeofday () in
  Sarek.Execute.run_vectors
    ~device:dev
    ~ir
    ~args:
      [
        Sarek.Execute.Vec input;
        Sarek.Execute.Vec output;
        Sarek.Execute.Int dim;
        Sarek.Execute.Int dim;
      ]
    ~block
    ~grid
    () ;
  Transfer.flush dev ;
  let t1 = Unix.gettimeofday () in

  ((t1 -. t0) *. 1000.0, Vector.to_array output)

(* Float32 transpose *)
let run_transpose_float32 (dev : Device.t) size _block_size =
  let dim = Int32.to_int (Int32.of_float (sqrt (float_of_int size))) in
  let n = dim * dim in
  let _, kirc = transpose_float32_kernel in
  let ir =
    match kirc.Sarek.Kirc_types.body_ir with
    | Some ir -> ir
    | None -> failwith "No IR"
  in

  let input = Vector.create Vector.float32 n in
  let output = Vector.create Vector.float32 n in

  for i = 0 to n - 1 do
    Vector.set input i !input_data_float32.(i) ;
    Vector.set output i 0.0
  done ;

  let block_size = 256 in
  let grid_size = (n + block_size - 1) / block_size in
  let block = Sarek.Execute.dims1d block_size in
  let grid = Sarek.Execute.dims1d grid_size in

  let t0 = Unix.gettimeofday () in
  Sarek.Execute.run_vectors
    ~device:dev
    ~ir
    ~args:
      [
        Sarek.Execute.Vec input;
        Sarek.Execute.Vec output;
        Sarek.Execute.Int dim;
        Sarek.Execute.Int dim;
      ]
    ~block
    ~grid
    () ;
  Transfer.flush dev ;
  let t1 = Unix.gettimeofday () in

  ((t1 -. t0) *. 1000.0, Vector.to_array output)

(* Float64 transpose *)
let run_transpose_float64 (dev : Device.t) size _block_size =
  let dim = Int32.to_int (Int32.of_float (sqrt (float_of_int size))) in
  let n = dim * dim in
  let _, kirc = transpose_float64_kernel in
  let ir =
    match kirc.Sarek.Kirc_types.body_ir with
    | Some ir -> ir
    | None -> failwith "No IR"
  in

  let input = Vector.create Vector.float64 n in
  let output = Vector.create Vector.float64 n in

  for i = 0 to n - 1 do
    Vector.set input i !input_data_float64.(i) ;
    Vector.set output i 0.0
  done ;

  let block_size = 256 in
  let grid_size = (n + block_size - 1) / block_size in
  let block = Sarek.Execute.dims1d block_size in
  let grid = Sarek.Execute.dims1d grid_size in

  let t0 = Unix.gettimeofday () in
  Sarek.Execute.run_vectors
    ~device:dev
    ~ir
    ~args:
      [
        Sarek.Execute.Vec input;
        Sarek.Execute.Vec output;
        Sarek.Execute.Int dim;
        Sarek.Execute.Int dim;
      ]
    ~block
    ~grid
    () ;
  Transfer.flush dev ;
  let t1 = Unix.gettimeofday () in

  ((t1 -. t0) *. 1000.0, Vector.to_array output)

(* Point3D transpose *)
let run_transpose_point3d (dev : Device.t) size _block_size =
  let dim = Int32.to_int (Int32.of_float (sqrt (float_of_int size))) in
  let n = dim * dim in
  let _, kirc = transpose_point3d_kernel in
  let ir =
    match kirc.Sarek.Kirc_types.body_ir with
    | Some ir -> ir
    | None -> failwith "No IR"
  in

  let input = Vector.create_custom point3d_custom n in
  let output = Vector.create_custom point3d_custom n in

  for i = 0 to n - 1 do
    let px, py, pz = !input_data_point3d.(i) in
    Vector.set input i {x = px; y = py; z = pz} ;
    Vector.set output i {x = 0.0; y = 0.0; z = 0.0}
  done ;

  let block_size = 256 in
  let grid_size = (n + block_size - 1) / block_size in
  let block = Sarek.Execute.dims1d block_size in
  let grid = Sarek.Execute.dims1d grid_size in

  let t0 = Unix.gettimeofday () in
  Sarek.Execute.run_vectors
    ~device:dev
    ~ir
    ~args:
      [
        Sarek.Execute.Vec input;
        Sarek.Execute.Vec output;
        Sarek.Execute.Int dim;
        Sarek.Execute.Int dim;
      ]
    ~block
    ~grid
    () ;
  Transfer.flush dev ;
  let t1 = Unix.gettimeofday () in

  (* Extract point3d values as tuples *)
  let result =
    Array.init n (fun i ->
        let p = Vector.get output i in
        (p.x, p.y, p.z))
  in
  ((t1 -. t0) *. 1000.0, result)

(* ========== Main ========== *)

let () =
  Benchmarks.init () ;
  let size = Benchmarks.config.size in
  init_transpose_data size ;

  (* Int32 baseline and verify *)
  let baseline_int32 size =
    let dim = Int32.to_int (Int32.of_float (sqrt (float_of_int size))) in
    let n = dim * dim in
    let inp = !input_data_int32 in
    let out = Array.make n 0l in
    ocaml_transpose_int32 inp out dim dim ;
    out
  in
  let verify_int32 result expected =
    let n = Array.length expected in
    let errors = ref 0 in
    for i = 0 to n - 1 do
      if result.(i) <> expected.(i) then begin
        if !errors < 5 then
          Printf.printf
            "  Mismatch at %d: expected %ld, got %ld\n"
            i
            expected.(i)
            result.(i) ;
        incr errors
      end
    done ;
    !errors = 0
  in

  (* Float32 baseline and verify *)
  let baseline_float32 size =
    let dim = Int32.to_int (Int32.of_float (sqrt (float_of_int size))) in
    let n = dim * dim in
    let inp = !input_data_float32 in
    let out = Array.make n 0.0 in
    ocaml_transpose_float inp out dim dim ;
    out
  in
  let verify_float32 result expected =
    let n = Array.length expected in
    let errors = ref 0 in
    let tolerance = 0.001 in
    for i = 0 to n - 1 do
      let diff = abs_float (result.(i) -. expected.(i)) in
      if diff > tolerance then begin
        if !errors < 5 then
          Printf.printf
            "  Mismatch at %d: expected %.2f, got %.2f\n"
            i
            expected.(i)
            result.(i) ;
        incr errors
      end
    done ;
    !errors = 0
  in

  (* Float64 baseline and verify *)
  let baseline_float64 size =
    let dim = Int32.to_int (Int32.of_float (sqrt (float_of_int size))) in
    let n = dim * dim in
    let inp = !input_data_float64 in
    let out = Array.make n 0.0 in
    ocaml_transpose_float inp out dim dim ;
    out
  in
  let verify_float64 = verify_float32 in

  (* Point3D baseline and verify *)
  let baseline_point3d size =
    let dim = Int32.to_int (Int32.of_float (sqrt (float_of_int size))) in
    let n = dim * dim in
    let inp = !input_data_point3d in
    let out = Array.make n (0.0, 0.0, 0.0) in
    for row = 0 to dim - 1 do
      for col = 0 to dim - 1 do
        let in_idx = (row * dim) + col in
        let out_idx = (col * dim) + row in
        out.(out_idx) <- inp.(in_idx)
      done
    done ;
    out
  in
  let verify_point3d result expected =
    let n = Array.length expected in
    let errors = ref 0 in
    let tolerance = 0.001 in
    for i = 0 to n - 1 do
      let rx, ry, rz = result.(i) in
      let ex, ey, ez = expected.(i) in
      let diff =
        abs_float (rx -. ex) +. abs_float (ry -. ey) +. abs_float (rz -. ez)
      in
      if diff > tolerance then begin
        if !errors < 5 then
          Printf.printf
            "  Mismatch at %d: expected (%.2f,%.2f,%.2f), got (%.2f,%.2f,%.2f)\n"
            i
            ex
            ey
            ez
            rx
            ry
            rz ;
        incr errors
      end
    done ;
    !errors = 0
  in

  (* Transpose is slow on interpreter (10-19s) - exclude it *)
  Benchmarks.run
    ~baseline:baseline_int32
    ~verify:verify_int32
    ~filter:Benchmarks.no_interpreter
    "Transpose<int32>"
    run_transpose_int32 ;
  Benchmarks.run
    ~baseline:baseline_float32
    ~verify:verify_float32
    ~filter:Benchmarks.no_interpreter
    "Transpose<float32>"
    run_transpose_float32 ;
  (* Float64 needs both fp64 support AND no interpreter *)
  let fp64_no_interp d = Device.allows_fp64 d && Benchmarks.no_interpreter d in
  Benchmarks.run
    ~baseline:baseline_float64
    ~verify:verify_float64
    ~filter:fp64_no_interp
    "Transpose<float64>"
    run_transpose_float64 ;
  (* Point3d transpose is very slow on Native (16s) and Interpreter (28-45s) *)
  Benchmarks.run
    ~baseline:baseline_point3d
    ~verify:verify_point3d
    ~filter:Benchmarks.jit_only
    "Transpose<point3d>"
    run_transpose_point3d ;
  Benchmarks.exit ()
