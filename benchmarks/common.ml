(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Common utilities for benchmarks *)

open Spoc_core

(** Statistical functions for timing data *)

let mean values =
  if Array.length values = 0 then 0.0
  else
    let sum = Array.fold_left ( +. ) 0.0 values in
    sum /. float_of_int (Array.length values)

let stddev values =
  if Array.length values <= 1 then 0.0
  else
    let m = mean values in
    let sum_sq_diff =
      Array.fold_left (fun acc v -> acc +. ((v -. m) *. (v -. m))) 0.0 values
    in
    sqrt (sum_sq_diff /. float_of_int (Array.length values - 1))

let median values =
  if Array.length values = 0 then 0.0
  else
    let sorted = Array.copy values in
    Array.sort Float.compare sorted ;
    let n = Array.length sorted in
    if n mod 2 = 0 then (sorted.((n / 2) - 1) +. sorted.(n / 2)) /. 2.0
    else sorted.(n / 2)

let min values =
  if Array.length values = 0 then 0.0
  else Array.fold_left Float.min infinity values

let max values =
  if Array.length values = 0 then 0.0
  else Array.fold_left Float.max neg_infinity values

let percentile values p =
  if Array.length values = 0 then 0.0
  else
    let sorted = Array.copy values in
    Array.sort Float.compare sorted ;
    let idx = int_of_float (p *. float_of_int (Array.length sorted)) in
    let idx = Int.max 0 (Int.min (Array.length sorted - 1) idx) in
    sorted.(idx)

(** Timing utilities *)

let time_ms f =
  let t0 = Unix.gettimeofday () in
  let result = f () in
  let t1 = Unix.gettimeofday () in
  let time_ms = (t1 -. t0) *. 1000.0 in
  (time_ms, result)

let time_iterations ~warmup ~iterations f =
  (* Warmup *)
  for _ = 1 to warmup do
    let _ = f () in
    ()
  done ;

  (* Benchmark iterations *)
  let times = Array.init iterations (fun _ -> fst (time_ms f)) in
  times

(** Result type for benchmarks *)

type 'a result = {
  iterations : float array; (* Individual iteration times in ms *)
  mean_ms : float;
  stddev_ms : float;
  median_ms : float;
  min_ms : float;
  max_ms : float;
  value : 'a; (* The computed result for verification *)
}

let make_result iterations value =
  {
    iterations;
    mean_ms = mean iterations;
    stddev_ms = stddev iterations;
    median_ms = median iterations;
    min_ms = min iterations;
    max_ms = max iterations;
    value;
  }

(** Array comparison for verification *)

let arrays_equal ~epsilon a b =
  if Array.length a <> Array.length b then false
  else
    let errors = ref 0 in
    Array.iteri
      (fun i va -> if abs_float (va -. b.(i)) > epsilon then incr errors)
      a ;
    !errors = 0

let arrays_equal_verbose ~epsilon a b =
  if Array.length a <> Array.length b then (
    Printf.printf
      "Array length mismatch: %d vs %d\n"
      (Array.length a)
      (Array.length b) ;
    false)
  else
    let errors = ref 0 in
    Array.iteri
      (fun i va ->
        let diff = abs_float (va -. b.(i)) in
        if diff > epsilon then (
          if !errors < 10 then
            Printf.printf
              "  Mismatch at %d: %.6f vs %.6f (diff=%.6e)\n"
              i
              va
              b.(i)
              diff ;
          incr errors))
      a ;
    if !errors > 0 then
      Printf.printf "Total errors: %d / %d\n" !errors (Array.length a) ;
    !errors = 0

(** Git commit hash for reproducibility *)

let get_git_commit () =
  try
    let ic = Unix.open_process_in "git rev-parse HEAD 2>/dev/null" in
    let commit = input_line ic in
    let _ = Unix.close_process_in ic in
    Some (String.trim commit)
  with _ -> None

(** Timestamp *)

let get_timestamp () =
  let open Unix in
  let tm = gmtime (time ()) in
  Printf.sprintf
    "%04d-%02d-%02dT%02d:%02d:%02dZ"
    (tm.tm_year + 1900)
    (tm.tm_mon + 1)
    tm.tm_mday
    tm.tm_hour
    tm.tm_min
    tm.tm_sec

(** Safe filename from timestamp *)

let timestamp_filename () =
  let open Unix in
  let tm = gmtime (time ()) in
  Printf.sprintf
    "%04d-%02d-%02dT%02d-%02d-%02d"
    (tm.tm_year + 1900)
    (tm.tm_mon + 1)
    tm.tm_mday
    tm.tm_hour
    tm.tm_min
    tm.tm_sec

(** GPU-aware benchmarking with proper synchronization

    Handles the complete benchmark protocol: 1. First run to trigger compilation
    2. Warmup iterations 3. Synchronized timing of kernel execution only

    @param dev Device to benchmark on
    @param warmup Number of warmup iterations (after compilation)
    @param iterations Number of benchmark iterations
    @param kernel_fn Function that executes the kernel (not including transfers)
    @return Array of execution times in milliseconds *)
let benchmark_kernel_on_device ~dev ~warmup ~iterations kernel_fn =
  (* Ensure device is ready *)
  Device.synchronize dev ;

  (* First run to trigger kernel compilation *)
  kernel_fn () ;
  Device.synchronize dev ;

  (* Warmup runs with compiled kernel *)
  for _ = 1 to warmup do
    kernel_fn () ;
    Device.synchronize dev
  done ;

  (* Benchmark - time only kernel execution *)
  Array.init iterations (fun _ ->
      Device.synchronize dev ;
      (* Ensure previous work is done *)
      let t0 = Unix.gettimeofday () in
      kernel_fn () ;
      Device.synchronize dev ;
      (* Wait for kernel completion *)
      let t1 = Unix.gettimeofday () in
      (t1 -. t0) *. 1000.0)
