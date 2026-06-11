(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * E2E test for Sarek PPX - Discrete Fourier Transform
 *
 * This test verifies that kernels compiled with the PPX can generate valid
 * GPU code and execute correctly via the GPU runtime.
 ******************************************************************************)

open Sarek
module Std = Sarek_stdlib.Std
module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer
module Benchmarks = Test_helpers.Benchmarks

let n = 16

let vector_dft =
  [%kernel
    fun (a : float32 vector)
        (b : float32 vector)
        (c : float32 vector)
        (d : float32 vector)
        (size : int32)
        (n : int32) ->
      let open Std in
      let tid = global_thread_id in
      if tid < n then (
        let pi = float 4l *. atan (float 1l) in
        let value = float 2l *. pi *. float tid /. float n in
        let sum_a = mut 0.0 in
        let sum_b = mut 0.0 in
        for i = 0 to size - 1l do
          let angle = value *. float i in
          sum_a := sum_a +. (a.(i) *. cos angle) ;
          sum_b := sum_b -. (b.(i) *. sin angle)
        done ;
        c.(tid) <- sum_a ;
        d.(tid) <- sum_b)]

let compute_expected size =
  let a = Array.init size (fun i -> float_of_int i) in
  let b = Array.init size (fun i -> float_of_int i) in
  let c = Array.make n 0.0 in
  let d = Array.make n 0.0 in
  let const = 2.0 *. Float.pi /. float n in
  for t = 0 to n - 1 do
    let sum_a = ref 0.0 in
    let sum_b = ref 0.0 in
    for k = 0 to size - 1 do
      let angle = const *. float_of_int t *. float_of_int k in
      sum_a := !sum_a +. (a.(k) *. cos angle) ;
      sum_b := !sum_b -. (b.(k) *. sin angle)
    done ;
    c.(t) <- !sum_a ;
    d.(t) <- !sum_b
  done ;
  (c, d)

(* Relative tolerance: |result - expected| / (|expected| + 1) <= epsilon.
   The +1 guard prevents division-by-zero near zero.
   Float32 trig on large angles (t=14,15 at size=1024 reach ~5600 rad) loses
   ~3e-4 rad of precision per term, compounding to ~2% over 1024 terms.
   Absolute tolerance is wrong here — use relative with 5e-2.
   The Nyquist bin (t=8) has expected imag=0; float32 π error makes
   sin(π*1023) ≈ 0.05 instead of 0, so the absolute floor matters too. *)
let find_mismatch label result expected size epsilon =
  let errors = ref 0 in
  for i = 0 to size - 1 do
    let diff = abs_float (result.(i) -. expected.(i)) in
    let scale = abs_float expected.(i) +. 1.0 in
    if diff /. scale > epsilon then begin
      if !errors < 5 then
        Printf.printf
          "  %s mismatch at %d: expected %.4f, got %.4f (rel err %.2e)\n"
          label
          i
          expected.(i)
          result.(i)
          (diff /. scale) ;
      incr errors
    end
  done ;
  if !errors > 5 then
    Printf.printf "  ... and %d more %s mismatches\n" (!errors - 5) label ;
  !errors = 0

let verify_results result expected =
  let res_a, res_b = result in
  let exp_a, exp_b = expected in
  let size = Array.length exp_a in
  (* 12% covers Vulkan's larger sin/cos deviation on the Nyquist bin (t=8,
     expected imag=0) where the float32 π error reaches ~11.4% absolute.
     OpenCL's worst case is ~5%, so this gives margin for both backends. *)
  let epsilon = 0.12 in
  find_mismatch "real" res_a exp_a size epsilon
  && find_mismatch "imag" res_b exp_b size epsilon

let run_test dev size block_size =
  let _, kirc = vector_dft in
  let ir =
    match kirc.Sarek.Kirc_types.body_ir with
    | Some ir -> ir
    | None -> failwith "No IR"
  in

  let a = Vector.create Vector.float32 size in
  let b = Vector.create Vector.float32 size in
  let c = Vector.create Vector.float32 n in
  let d = Vector.create Vector.float32 n in

  for i = 0 to size - 1 do
    Vector.set a i (float_of_int i) ;
    Vector.set b i (float_of_int i)
  done ;
  for i = 0 to n - 1 do
    Vector.set c i (-999.0) ;
    Vector.set d i (-999.0)
  done ;

  let block_sz = block_size in
  (* Grid covers the n output threads, not the input size. *)
  let grid_sz = (n + block_sz - 1) / block_sz in
  let block = Execute.dims1d block_sz in
  let grid = Execute.dims1d grid_sz in

  Execute.run_vectors
    ~device:dev
    ~ir
    ~args:[Vec a; Vec b; Vec c; Vec d; Int size; Int n]
    ~block
    ~grid
    () ;
  Transfer.flush dev ;

  let t0 = Unix.gettimeofday () in
  Execute.run_vectors
    ~device:dev
    ~ir
    ~args:[Vec a; Vec b; Vec c; Vec d; Int size; Int n]
    ~block
    ~grid
    () ;
  Transfer.flush dev ;
  let t1 = Unix.gettimeofday () in

  let res_a = Vector.to_array c in
  let res_b = Vector.to_array d in
  ((t1 -. t0) *. 1000.0, (res_a, res_b))

let () =
  Benchmarks.run
    ~baseline:compute_expected
    ~verify:verify_results
    ~filter:Benchmarks.no_interpreter
    "Vector DFT Test"
    run_test ;
  Benchmarks.exit ()
