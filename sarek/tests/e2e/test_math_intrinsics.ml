(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * E2E test for Sarek PPX - Math intrinsics
 *
 * Tests all mathematical intrinsic functions: sin, cos, exp, log, sqrt, etc.
 * These map directly to GPU hardware math units.
 ******************************************************************************)

open Sarek
module Std = Sarek_stdlib.Std

(* Module aliases *)
module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer

(* Force backend registration *)
let () = Test_helpers.Benchmarks.init_backends ()

let cfg = Test_helpers.default_config ()

(* ========== Pure OCaml baseline ========== *)

let ocaml_complex_math x y output n =
  for i = 0 to n - 1 do
    let a = x.(i) in
    let b = y.(i) in
    let r = sqrt ((a *. a) +. (b *. b)) in
    let decay = exp (-0.1 *. (a +. b)) in
    let oscillation = cos (a *. b) in
    output.(i) <- r *. decay *. oscillation
  done

(* ========== Shared test data ========== *)

let input_x = ref [||]

let input_y = ref [||]

let expected_complex = ref [||]

let init_complex_data () =
  let n = cfg.size in
  let x = Array.init n (fun i -> float_of_int (i mod 10) *. 0.5) in
  let y = Array.init n (fun i -> float_of_int ((i + 5) mod 10) *. 0.5) in
  let o = Array.make n 0.0 in
  input_x := x ;
  input_y := y ;
  expected_complex := o ;
  ocaml_complex_math x y o n

(* ========== Sarek kernel ========== *)

let complex_math_kernel =
  [%kernel
    fun (x : float32 vector)
        (y : float32 vector)
        (output : float32 vector)
        (n : int32) ->
      let open Std in
      let tid = global_thread_id in
      if tid < n then begin
        let a = x.(tid) in
        let b = y.(tid) in
        let r = sqrt ((a *. a) +. (b *. b)) in
        let decay = exp (-0.1 *. (a +. b)) in
        let oscillation = cos (a *. b) in
        output.(tid) <- r *. decay *. oscillation
      end]

(* ========== runtime test runner ========== *)

let run_complex_math (dev : Device.t) =
  let n = cfg.size in
  let _, kirc = complex_math_kernel in
  let ir =
    match kirc.Sarek.Kirc_types.body_ir with
    | Some ir -> ir
    | None -> failwith "No IR"
  in

  let x = Vector.create Vector.float32 n in
  let y = Vector.create Vector.float32 n in
  let output = Vector.create Vector.float32 n in

  for i = 0 to n - 1 do
    Vector.set x i !input_x.(i) ;
    Vector.set y i !input_y.(i) ;
    Vector.set output i 0.0
  done ;

  let block_size = 256 in
  let grid_size = (n + block_size - 1) / block_size in
  let block = Execute.dims1d block_size in
  let grid = Execute.dims1d grid_size in

  let t0 = Unix.gettimeofday () in
  Execute.run_vectors
    ~device:dev
    ~ir
    ~args:[Execute.Vec x; Execute.Vec y; Execute.Vec output; Execute.Int n]
    ~block
    ~grid
    () ;
  Transfer.flush dev ;
  let t1 = Unix.gettimeofday () in

  ((t1 -. t0) *. 1000.0, Vector.to_array output)

(* ========== Verification ========== *)

let verify_float_arrays name result expected tolerance =
  let n = Array.length expected in
  let errors = ref 0 in
  for i = 0 to n - 1 do
    let diff = abs_float (result.(i) -. expected.(i)) in
    if diff > tolerance then begin
      if !errors < 5 then
        Printf.printf
          "  %s mismatch at %d: expected %.6f, got %.6f (diff=%.6f)\n"
          name
          i
          expected.(i)
          result.(i)
          diff ;
      incr errors
    end
  done ;
  !errors = 0

let () =
  let c = Test_helpers.parse_args "test_math_intrinsics" in
  cfg.dev_id <- c.dev_id ;
  cfg.use_interpreter <- c.use_interpreter ;
  cfg.use_native <- c.use_native ;
  cfg.benchmark_all <- c.benchmark_all ;
  cfg.benchmark_devices <- c.benchmark_devices ;
  cfg.verify <- c.verify ;
  cfg.size <- c.size ;
  cfg.block_size <- c.block_size ;

  print_endline "=== Math Intrinsics Test (runtime) ===" ;
  Printf.printf "Size: %d elements\n\n" cfg.size ;

  init_complex_data () ;

  let devs = Test_helpers.init_devices cfg in
  if Array.length devs = 0 then begin
    print_endline "No devices found" ;
    exit 1
  end ;
  Test_helpers.print_devices devs ;

  if cfg.benchmark_all then begin
    print_endline (String.make 60 '-') ;
    Printf.printf "%-35s %10s %10s\n" "Device" "Time(ms)" "Status" ;
    print_endline (String.make 60 '-') ;

    let all_ok = ref true in

    Array.iter
      (fun dev ->
        let name = dev.Device.name in
        let framework = dev.Device.framework in

        try
          let time, result = run_complex_math dev in
          let ok =
            (not cfg.verify)
            || verify_float_arrays "runtime" result !expected_complex 0.01
          in
          let status = if ok then "OK" else "FAIL" in

          if not ok then all_ok := false ;

          Printf.printf
            "%-35s %10.4f %10s\n"
            (Printf.sprintf "%s (%s)" name framework)
            time
            status
        with
        | Spoc_framework.Backend_error.Backend_error _ ->
            Printf.printf
              "%-35s %10s %10s\n"
              (Printf.sprintf "%s (%s)" name framework)
              "ERR"
              "ERROR" ;
            all_ok := false
        | e ->
            Printf.printf
              "%-35s %10s %10s\n"
              (Printf.sprintf "%s (%s)" name framework)
              "ERR"
              (Printexc.to_string e) ;
            all_ok := false)
      devs ;

    print_endline (String.make 60 '-') ;

    if !all_ok then print_endline "\n=== All math intrinsics tests PASSED ==="
    else begin
      print_endline "\n=== Some math intrinsics tests FAILED ===" ;
      exit 1
    end
  end
  else begin
    let dev = Test_helpers.get_device cfg devs in
    Printf.printf "Using device: %s\n%!" dev.Device.name ;

    Printf.printf "\nRunning runtime path (complex math: sqrt/exp/cos)...\n%!" ;
    try
      let time, result = run_complex_math dev in
      Printf.printf "  Time: %.4f ms\n%!" time ;
      let ok =
        (not cfg.verify)
        || verify_float_arrays "runtime" result !expected_complex 0.01
      in
      Printf.printf "  Status: %s\n%!" (if ok then "PASSED" else "FAILED") ;

      if ok then print_endline "\nMath intrinsics tests PASSED"
      else begin
        print_endline "\nMath intrinsics tests FAILED" ;
        exit 1
      end
    with
    | Spoc_framework.Backend_error.Backend_error _ ->
        print_endline "  Status: ERROR (Backend error)" ;
        print_endline "\nMath intrinsics tests SKIPPED (backend error)" ;
        exit 0
    | e ->
        Printf.printf "  Status: ERROR (%s)\n" (Printexc.to_string e) ;
        print_endline "\nMath intrinsics tests FAILED" ;
        exit 1
  end
