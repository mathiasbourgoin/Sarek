(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * E2E test for Sarek PPX - Prefix Sum (Scan)
 *
 * Tests inclusive prefix sum operations with shared memory and supersteps.
 * Scan is a fundamental parallel primitive for many algorithms.
 * GPU runtime only.
 ******************************************************************************)

(* Module aliases *)
module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer
module Benchmarks = Test_helpers.Benchmarks

(* Backends auto-register when linked; Benchmarks.init() ensures initialization *)

let cfg = Test_helpers.default_config ()

(* ========== Pure OCaml baselines ========== *)

let ocaml_inclusive_scan input output n =
  if n > 0 then begin
    output.(0) <- input.(0) ;
    for i = 1 to n - 1 do
      output.(i) <- Int32.add output.(i - 1) input.(i)
    done
  end

(* ========== Shared test data ========== *)

let input_ones = ref [||]

let input_varying = ref [||]

let init_scan_data size =
  let n = min size 256 in
  input_ones := Array.make n 1l ;
  input_varying := Array.init n (fun i -> Int32.of_int (i + 1))

(* ========== Sarek kernels ========== *)

(** Inclusive scan within a block using Hillis-Steele algorithm. *)
let inclusive_scan_kernel =
  [%kernel
    fun (input : int32 vector) (output : int32 vector) (n : int32) ->
      let%shared (temp : int32) = 512l in
      let%shared (temp2 : int32) = 512l in
      let tid = thread_idx_x in
      let gid = thread_idx_x + (block_dim_x * block_idx_x) in
      let%superstep load =
        if gid < n then temp.(tid) <- input.(gid) else temp.(tid) <- 0l
      in
      let%superstep step1 =
        let v = temp.(tid) in
        let add = if tid >= 1l then temp.(tid - 1l) else 0l in
        temp2.(tid) <- v + add
      in
      let%superstep step2 =
        let v = temp2.(tid) in
        let add = if tid >= 2l then temp2.(tid - 2l) else 0l in
        temp.(tid) <- v + add
      in
      let%superstep step4 =
        let v = temp.(tid) in
        let add = if tid >= 4l then temp.(tid - 4l) else 0l in
        temp2.(tid) <- v + add
      in
      let%superstep step8 =
        let v = temp2.(tid) in
        let add = if tid >= 8l then temp2.(tid - 8l) else 0l in
        temp.(tid) <- v + add
      in
      let%superstep step16 =
        let v = temp.(tid) in
        let add = if tid >= 16l then temp.(tid - 16l) else 0l in
        temp2.(tid) <- v + add
      in
      let%superstep step32 =
        let v = temp2.(tid) in
        let add = if tid >= 32l then temp2.(tid - 32l) else 0l in
        temp.(tid) <- v + add
      in
      let%superstep step64 =
        let v = temp.(tid) in
        let add = if tid >= 64l then temp.(tid - 64l) else 0l in
        temp2.(tid) <- v + add
      in
      let%superstep step128 =
        let v = temp2.(tid) in
        let add = if tid >= 128l then temp2.(tid - 128l) else 0l in
        temp.(tid) <- v + add
      in
      if gid < n then output.(gid) <- temp.(tid)]

(* ========== runtime test runners ========== *)

let run_inclusive_scan (dev : Device.t) inp size =
  let n = min size 256 in
  let _, kirc = inclusive_scan_kernel in
  let ir =
    match kirc.Sarek.Kirc_types.body_ir with
    | Some ir -> ir
    | None -> failwith "No IR"
  in

  let input = Vector.create Vector.int32 n in
  let output = Vector.create Vector.int32 n in

  for i = 0 to n - 1 do
    Vector.set input i inp.(i) ;
    Vector.set output i 0l
  done ;

  let block_size = 256 in
  let block = Sarek.Execute.dims1d block_size in
  let grid = Sarek.Execute.dims1d 1 in

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
  let time_ms = (t1 -. t0) *. 1000.0 in

  (time_ms, Vector.to_array output)

(* ========== Main ========== *)

let () =
  Benchmarks.init () ;
  let size = Benchmarks.config.size in
  init_scan_data size ;

  let verify result expected =
    let n = Array.length expected in
    let errors = ref 0 in
    for i = 0 to n - 1 do
      if result.(i) <> expected.(i) then begin
        if !errors < 10 then
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

  (* Scan with ones *)
  let baseline_ones size =
    let n = min size 256 in
    let out = Array.make n 0l in
    ocaml_inclusive_scan !input_ones out n ;
    out
  in
  let run_ones dev size _block_size = run_inclusive_scan dev !input_ones size in
  (* Scan uses shared memory - interpreter doesn't support it *)
  Benchmarks.run
    ~filter:Benchmarks.no_interpreter
    ~baseline:baseline_ones
    ~verify
    "Inclusive Scan (all ones)"
    run_ones ;

  (* Scan with varying values *)
  let baseline_varying size =
    let n = min size 256 in
    let out = Array.make n 0l in
    ocaml_inclusive_scan !input_varying out n ;
    out
  in
  let run_varying dev size _block_size =
    run_inclusive_scan dev !input_varying size
  in
  Benchmarks.run
    ~filter:Benchmarks.no_interpreter
    ~baseline:baseline_varying
    ~verify
    "Inclusive Scan (varying)"
    run_varying ;
  Benchmarks.exit ()
