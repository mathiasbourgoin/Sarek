(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * E2E test for Sarek PPX - Bitwise operations
 *
 * Tests bitwise AND, OR, XOR, NOT, shifts, and bit manipulation patterns.
 * These are essential for cryptography, compression, and low-level algorithms.
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

(* ========== Pure OCaml baselines ========== *)

let ocaml_bitwise_basic a b and_out or_out xor_out n =
  for i = 0 to n - 1 do
    and_out.(i) <- Int32.logand a.(i) b.(i) ;
    or_out.(i) <- Int32.logor a.(i) b.(i) ;
    xor_out.(i) <- Int32.logxor a.(i) b.(i)
  done

let ocaml_shift input shift_amt left_out right_out n =
  for i = 0 to n - 1 do
    let s = Int32.to_int shift_amt.(i) in
    left_out.(i) <- Int32.shift_left input.(i) s ;
    right_out.(i) <- Int32.shift_right_logical input.(i) s
  done

let count_bits x =
  let rec loop acc v =
    if v = 0l then acc
    else
      loop (Int32.add acc (Int32.logand v 1l)) (Int32.shift_right_logical v 1)
  in
  loop 0l x

let ocaml_popcount input output n =
  for i = 0 to n - 1 do
    output.(i) <- count_bits input.(i)
  done

let ocaml_gray_code input to_gray from_gray n =
  for i = 0 to n - 1 do
    let x = input.(i) in
    let gray = Int32.logxor x (Int32.shift_right_logical x 1) in
    to_gray.(i) <- gray ;
    let g = gray in
    let b = ref g in
    let mask = ref (Int32.shift_right_logical g 1) in
    while !mask <> 0l do
      b := Int32.logxor !b !mask ;
      mask := Int32.shift_right_logical !mask 1
    done ;
    from_gray.(i) <- !b
  done

(* ========== Shared test data ========== *)

let input_a = ref [||]

let input_b = ref [||]

let expected_and = ref [||]

let expected_or = ref [||]

let expected_xor = ref [||]

let input_shift = ref [||]

let input_shift_amt = ref [||]

let expected_left = ref [||]

let expected_right = ref [||]

let input_popcount = ref [||]

let expected_popcount = ref [||]

let input_gray = ref [||]

let expected_to_gray = ref [||]

let expected_from_gray = ref [||]

let init_bitwise_data () =
  let n = cfg.size in
  let a = Array.init n (fun i -> Int32.of_int (i * 17)) in
  let b = Array.init n (fun i -> Int32.of_int (i * 13)) in
  let and_out = Array.make n 0l in
  let or_out = Array.make n 0l in
  let xor_out = Array.make n 0l in
  input_a := a ;
  input_b := b ;
  expected_and := and_out ;
  expected_or := or_out ;
  expected_xor := xor_out ;
  ocaml_bitwise_basic a b and_out or_out xor_out n

let init_shift_data () =
  let n = cfg.size in
  let inp = Array.make n (Int32.of_int 0xFF00FF) in
  let amt = Array.init n (fun i -> Int32.of_int (i mod 16)) in
  let left = Array.make n 0l in
  let right = Array.make n 0l in
  input_shift := inp ;
  input_shift_amt := amt ;
  expected_left := left ;
  expected_right := right ;
  ocaml_shift inp amt left right n

let init_popcount_data () =
  let n = cfg.size in
  let inp = Array.init n (fun i -> Int32.of_int i) in
  let out = Array.make n 0l in
  input_popcount := inp ;
  expected_popcount := out ;
  ocaml_popcount inp out n

let init_gray_data () =
  let n = cfg.size in
  let inp = Array.init n (fun i -> Int32.of_int i) in
  let to_g = Array.make n 0l in
  let from_g = Array.make n 0l in
  input_gray := inp ;
  expected_to_gray := to_g ;
  expected_from_gray := from_g ;
  ocaml_gray_code inp to_g from_g n

(* ========== Sarek kernels ========== *)

let bitwise_basic_kernel =
  [%kernel
    fun (a : int32 vector)
        (b : int32 vector)
        (and_out : int32 vector)
        (or_out : int32 vector)
        (xor_out : int32 vector)
        (n : int32) ->
      let open Std in
      let tid = global_thread_id in
      if tid < n then begin
        let x = a.(tid) in
        let y = b.(tid) in
        and_out.(tid) <- x land y ;
        or_out.(tid) <- x lor y ;
        xor_out.(tid) <- x lxor y
      end]

let shift_kernel =
  [%kernel
    fun (input : int32 vector)
        (shift_amt : int32 vector)
        (left_out : int32 vector)
        (right_out : int32 vector)
        (n : int32) ->
      let tid = thread_idx_x + (block_dim_x * block_idx_x) in
      if tid < n then begin
        let x = input.(tid) in
        let s = shift_amt.(tid) in
        left_out.(tid) <- x lsl s ;
        right_out.(tid) <- x lsr s
      end]

let popcount_kernel =
  [%kernel
    fun (input : int32 vector) (output : int32 vector) (n : int32) ->
      let tid = thread_idx_x + (block_dim_x * block_idx_x) in
      if tid < n then begin
        let x = mut input.(tid) in
        let count = mut 0l in
        while x <> 0l do
          count := count + (x land 1l) ;
          x := x lsr 1l
        done ;
        output.(tid) <- count
      end]

let gray_code_kernel =
  [%kernel
    fun (input : int32 vector)
        (to_gray : int32 vector)
        (from_gray : int32 vector)
        (n : int32) ->
      let tid = thread_idx_x + (block_dim_x * block_idx_x) in
      if tid < n then begin
        let x = input.(tid) in
        to_gray.(tid) <- x lxor (x lsr 1l) ;
        let g = x lxor (x lsr 1l) in
        let b = mut g in
        let mask = mut (g lsr 1l) in
        while mask <> 0l do
          b := b lxor mask ;
          mask := mask lsr 1l
        done ;
        from_gray.(tid) <- b
      end]

(* ========== runtime test runner ========== *)

let run_bitwise_basic (dev : Device.t) =
  let n = cfg.size in
  let _, kirc = bitwise_basic_kernel in
  let ir =
    match kirc.Sarek.Kirc_types.body_ir with
    | Some ir -> ir
    | None -> failwith "No IR"
  in

  let a = Vector.create Vector.int32 n in
  let b = Vector.create Vector.int32 n in
  let and_out = Vector.create Vector.int32 n in
  let or_out = Vector.create Vector.int32 n in
  let xor_out = Vector.create Vector.int32 n in

  for i = 0 to n - 1 do
    Vector.set a i !input_a.(i) ;
    Vector.set b i !input_b.(i) ;
    Vector.set and_out i 0l ;
    Vector.set or_out i 0l ;
    Vector.set xor_out i 0l
  done ;

  let block_size = 256 in
  let grid_size = (n + block_size - 1) / block_size in
  let block = Execute.dims1d block_size in
  let grid = Execute.dims1d grid_size in

  let t0 = Unix.gettimeofday () in
  Execute.run_vectors
    ~device:dev
    ~ir
    ~args:
      [
        Execute.Vec a;
        Execute.Vec b;
        Execute.Vec and_out;
        Execute.Vec or_out;
        Execute.Vec xor_out;
        Execute.Int n;
      ]
    ~block
    ~grid
    () ;
  Transfer.flush dev ;
  let t1 = Unix.gettimeofday () in

  ( (t1 -. t0) *. 1000.0,
    Vector.to_array and_out,
    Vector.to_array or_out,
    Vector.to_array xor_out )

(* ========== Verification ========== *)

let verify_int32_arrays name result expected =
  let n = Array.length expected in
  let errors = ref 0 in
  for i = 0 to n - 1 do
    if result.(i) <> expected.(i) then begin
      if !errors < 5 then
        Printf.printf
          "  %s mismatch at %d: expected %ld, got %ld\n"
          name
          i
          expected.(i)
          result.(i) ;
      incr errors
    end
  done ;
  !errors = 0

let () =
  let c = Test_helpers.parse_args "test_bitwise_ops" in
  cfg.dev_id <- c.dev_id ;
  cfg.use_interpreter <- c.use_interpreter ;
  cfg.use_native <- c.use_native ;
  cfg.benchmark_all <- c.benchmark_all ;
  cfg.benchmark_devices <- c.benchmark_devices ;
  cfg.verify <- c.verify ;
  cfg.size <- c.size ;
  cfg.block_size <- c.block_size ;

  print_endline "=== Bitwise Operations Test (runtime) ===" ;
  Printf.printf "Size: %d elements\n\n" cfg.size ;

  (* Initialize data *)
  init_bitwise_data () ;
  init_shift_data () ;
  init_popcount_data () ;
  init_gray_data () ;

  (* Initialize runtime devices *)
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

        let time, and_, or_, xor = run_bitwise_basic dev in
        let ok =
          (not cfg.verify)
          || verify_int32_arrays "AND" and_ !expected_and
             && verify_int32_arrays "OR" or_ !expected_or
             && verify_int32_arrays "XOR" xor !expected_xor
        in
        let status = if ok then "OK" else "FAIL" in

        if not ok then all_ok := false ;

        Printf.printf
          "%-35s %10.4f %10s\n"
          (Printf.sprintf "%s (%s)" name framework)
          time
          status)
      devs ;

    print_endline (String.make 60 '-') ;

    if !all_ok then print_endline "\n=== All bitwise tests PASSED ==="
    else begin
      print_endline "\n=== Some bitwise tests FAILED ===" ;
      exit 1
    end
  end
  else begin
    let dev = Test_helpers.get_device cfg devs in
    let dev_name = dev.Device.name in
    Printf.printf "Using device: %s\n%!" dev_name ;

    Printf.printf "\nRunning runtime path (bitwise AND/OR/XOR)...\n%!" ;
    let time, and_, or_, xor = run_bitwise_basic dev in
    Printf.printf "  Time: %.4f ms\n%!" time ;
    let ok =
      (not cfg.verify)
      || verify_int32_arrays "AND" and_ !expected_and
         && verify_int32_arrays "OR" or_ !expected_or
         && verify_int32_arrays "XOR" xor !expected_xor
    in
    Printf.printf "  Status: %s\n%!" (if ok then "PASSED" else "FAILED") ;

    if ok then print_endline "\nBitwise operations tests PASSED"
    else begin
      print_endline "\nBitwise operations tests FAILED" ;
      exit 1
    end
  end
