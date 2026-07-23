(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * E2E test for float64 arithmetic written directly in the [%kernel] DSL.
 *
 * This is the end-to-end proof for the "G" float64-literal suffix
 * (Sarek_parse.ml): a real Sarek kernel written in ordinary DSL syntax -
 * float64 literals, +. -. *. /. comparisons, `mut`, a `while` loop and one
 * float64 intrinsic (Float64.sqrt) - compiles, lowers and runs, producing
 * results that match an OCaml binary64 reference.
 *
 * Every float literal in the kernel carries the `G` suffix, so it types as
 * float64 (not the float32 GPU default). Before the suffix existed there was
 * NO way to write an f64 literal in the DSL: `mut 0.0` was float32 and
 * `x *. 4.0` at an f64 `x` failed to unify. This test would not even compile
 * without that change - it is the regression guard for the whole path.
 *
 * The kernel is a Mandelbrot-style escape iteration in double precision. The
 * native/interpreter device runs the SAME OCaml float64 arithmetic as the
 * reference, so its results are compared exactly; other devices are compared
 * within an fp64 tolerance and reported (a per-device mismatch is a report
 * line, only the native gate fails the test).
 *
 * Run with: dune exec sarek/tests/e2e/test_float64_kernel_arith.exe
 ******************************************************************************)

[@@@warning "-33"]

open Sarek
module Std = Sarek_stdlib.Std
module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer

(* Initialise every available backend (Native + Interpreter always, plus any
   GPU backend that is present and not disabled by the SPOC_DISABLE env vars).
   Running under ZLUDA (LD_LIBRARY_PATH pointing at a ZLUDA CUDA runtime)
   surfaces the CUDA device here. *)
let () = Test_helpers.Benchmarks.init_backends ()

let max_iter = 1000l

(* The float64 kernel, written entirely in DSL syntax. Every float literal uses
   the `G` suffix -> float64. Exercises: f64 `mut` locals (x, y) initialised
   from an f64 literal, f64 +. -. *. /. , an f64 comparison (<=), and the
   Float64.sqrt intrinsic on an f64 value. `cx`/`cy` are f64 vector inputs. *)
let f64_mandelbrot_kernel =
  [%kernel
    fun (mag_out : float64 vector)
        (iter_out : int32 vector)
        (cx : float64 vector)
        (cy : float64 vector)
        (n : int32) ->
      let open Std in
      let open Sarek_float64 in
      let tid = global_thread_id in
      if tid < n then begin
        let x = mut 0.0G in
        let y = mut 0.0G in
        let iter = mut 0l in
        while (x *. x) +. (y *. y) <=. 4.0G && iter < 1000l do
          let xtemp = (x *. x) -. (y *. y) +. cx.(tid) in
          y := (2.0G *. x *. y) +. cy.(tid) ;
          x := xtemp ;
          iter := iter + 1l
        done ;
        iter_out.(tid) <- iter ;
        mag_out.(tid) <- Float64.sqrt ((x *. x) +. (y *. y))
      end]

(* OCaml binary64 reference: byte-identical arithmetic to the kernel body. *)
let ocaml_reference cx cy =
  let x = ref 0.0 in
  let y = ref 0.0 in
  let iter = ref 0l in
  while (!x *. !x) +. (!y *. !y) <= 4.0 && Int32.compare !iter max_iter < 0 do
    let xtemp = (!x *. !x) -. (!y *. !y) +. cx in
    y := (2.0 *. !x *. !y) +. cy ;
    x := xtemp ;
    iter := Int32.add !iter 1l
  done ;
  (!iter, Stdlib.sqrt ((!x *. !x) +. (!y *. !y)))

(* A spread of points: some inside the set (never escape, hit max_iter), some
   outside (escape quickly). Gives non-trivial iteration counts and magnitudes. *)
let n = 64

let cx_of i = (3.5 *. float_of_int i /. float_of_int n) -. 2.5

let cy_of i = (2.0 *. float_of_int (i mod 8) /. 8.0) -. 1.0

let run_on_device (dev : Device.t) ir =
  let mag = Vector.create Vector.float64 n in
  let iters = Vector.create Vector.int32 n in
  let cx = Vector.create Vector.float64 n in
  let cy = Vector.create Vector.float64 n in
  for i = 0 to n - 1 do
    Vector.set cx i (cx_of i) ;
    Vector.set cy i (cy_of i) ;
    Vector.set mag i 0.0 ;
    Vector.set iters i 0l
  done ;
  Execute.run_vectors
    ~device:dev
    ~ir
    ~args:
      [
        Execute.Vec mag;
        Execute.Vec iters;
        Execute.Vec cx;
        Execute.Vec cy;
        Execute.Int n;
      ]
    ~block:(Execute.dims1d n)
    ~grid:(Execute.dims1d 1)
    () ;
  Transfer.flush dev ;
  (Vector.to_array iters, Vector.to_array mag)

(* Compare device output to the reference. [exact] gates on bit-for-bit match of
   iteration counts (native only); otherwise a small fp64 tolerance is used. *)
let verify ~exact got_iters got_mag =
  let iter_tol = if exact then 0 else 2 in
  let mag_tol = if exact then 0.0 else 1e-9 in
  let bad = ref 0 in
  for i = 0 to n - 1 do
    let ref_iter, ref_mag = ocaml_reference (cx_of i) (cy_of i) in
    let di = abs (Int32.to_int got_iters.(i) - Int32.to_int ref_iter) in
    let dm = Stdlib.abs_float (got_mag.(i) -. ref_mag) in
    if di > iter_tol || dm > mag_tol then begin
      if !bad < 5 then
        Printf.printf
          "    mismatch @%d: iter got=%ld ref=%ld | mag got=%.15g ref=%.15g\n%!"
          i
          got_iters.(i)
          ref_iter
          got_mag.(i)
          ref_mag ;
      incr bad
    end
  done ;
  !bad

let is_native (dev : Device.t) =
  dev.Device.framework = "Native" || dev.Device.framework = "Interpreter"

let () =
  let _, kirc = f64_mandelbrot_kernel in
  let ir =
    match kirc.Kirc_types.body_ir with
    | Some ir -> ir
    | None -> failwith "f64 mandelbrot kernel has no IR"
  in
  (* The lowered IR must actually use float64 - if the `G` literals had silently
     stayed float32 the whole kernel would collapse to the f32 path. *)
  if not (Sarek_ir_analysis.kernel_uses_float64 ir) then begin
    print_endline
      "test_float64_kernel_arith: FAILED - kernel_uses_float64 = false (the G \
       literals did not produce a float64 kernel)" ;
    exit 1
  end ;
  let devs = Device.init () in
  Printf.printf "=== float64 DSL kernel arithmetic (G-suffix literals) ===\n%!" ;
  let native_ok = ref false in
  let native_failed = ref false in
  Array.iter
    (fun (dev : Device.t) ->
      let native = is_native dev in
      (* Non-native devices need real fp64 support; skip the rest. *)
      if native || Device.allows_fp64 dev then begin
        try
          let got_iters, got_mag = run_on_device dev ir in
          let bad = verify ~exact:native got_iters got_mag in
          Printf.printf
            "  %-10s %-24s %s (%d/%d ok)\n%!"
            dev.Device.framework
            dev.Device.name
            (if bad = 0 then "PASS" else "FAIL")
            (n - bad)
            n ;
          if native then begin
            native_ok := true ;
            if bad <> 0 then native_failed := true
          end
        with e ->
          Printf.printf
            "  %-10s %-24s ERROR (%s)\n%!"
            dev.Device.framework
            dev.Device.name
            (Printexc.to_string e) ;
          if native then native_failed := true
      end)
    devs ;
  if not !native_ok then begin
    print_endline
      "test_float64_kernel_arith: FAILED - no native/interpreter device ran \
       the kernel" ;
    exit 1
  end ;
  if !native_failed then begin
    print_endline "test_float64_kernel_arith: FAILED" ;
    exit 1
  end ;
  print_endline "test_float64_kernel_arith: PASSED"
