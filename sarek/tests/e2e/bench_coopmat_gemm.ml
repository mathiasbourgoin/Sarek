(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Time of Sarek_gemm.Coopmat.run (f16 x f16 + f32, default tiling) on every
 * device of one framework, at an aligned size: the number to compare between
 * two builds of the same tree (main versus a branch). Information only.
 *
 *   COOPMAT_BENCH_FRAMEWORK=CUDA/PTX (default) | CUDA/C | HIP ...
 *   COOPMAT_BENCH_SIZE=2048          m = n = k (a multiple of 64)
 *   COOPMAT_BENCH_REPS=30            timed runs after 5 warm-up runs
 *
 * Each run is followed by a flush, so the time is the whole GEMM; the median
 * and the best of the timed runs are printed in milliseconds and TFLOP/s
 * (2 m n k flops). Run it on an otherwise idle machine, alternate the builds,
 * and compare medians.
 ******************************************************************************)

open Coopmat_hip_support
module Gemm = Sarek_gemm.Coopmat

let () =
  Backend_hip.init () ;
  Backend_cuda_c.register () ;
  Test_helpers.Benchmarks.init_backends ()

let env_int name default =
  match Sys.getenv_opt name with Some s -> int_of_string s | None -> default

let framework =
  Option.value (Sys.getenv_opt "COOPMAT_BENCH_FRAMEWORK") ~default:"CUDA/PTX"

let size = env_int "COOPMAT_BENCH_SIZE" 2048

let reps = env_int "COOPMAT_BENCH_REPS" 30

let time_ms f =
  let t0 = Unix.gettimeofday () in
  f () ;
  (Unix.gettimeofday () -. t0) *. 1e3

let bench dev =
  let m = size and n = size and k = size in
  let a = f16_vec (Array.make (m * k) 1.0) in
  let b = f16_vec (Array.make (k * n) 1.0) in
  let c = f32_vec (Array.make (m * n) 0.0) in
  let d = f32_vec (Array.make (m * n) 0.0) in
  let once () =
    Gemm.run ~device:dev ~a ~b ~c ~d ~m ~n ~k () ;
    Transfer.flush dev
  in
  for _ = 1 to 5 do
    once ()
  done ;
  let times = List.init reps (fun _ -> time_ms once) in
  let sorted = List.sort compare times in
  let median = List.nth sorted (reps / 2) and best = List.hd sorted in
  let tflops ms = 2.0 *. float_of_int (m * n * k) /. (ms *. 1e-3) /. 1e12 in
  Printf.printf
    "%s [%s] %dx%dx%d: median %.3f ms (%.2f TFLOP/s), best %.3f ms (%.2f \
     TFLOP/s), %d runs\n\
     %!"
    dev.Device.name
    dev.Device.framework
    m
    n
    k
    median
    (tflops median)
    best
    (tflops best)
    reps

let () =
  C.accept_relaxed_accumulation ~reason:"bench_coopmat_gemm: timing only" ;
  let devices =
    Device.init () |> Array.to_list
    |> List.filter (fun d -> d.Device.framework = framework)
  in
  if devices = [] then Printf.printf "[SKIP] no %s device\n" framework
  else List.iter bench devices
