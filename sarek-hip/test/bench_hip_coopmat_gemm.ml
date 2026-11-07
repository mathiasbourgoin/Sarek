(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* f16 GEMM on the RDNA3 matrix units through SCoopmat (Sarek_gemm.Coopmat)
   against the scalar f32 Sarek_gemm.sgemm_tiled_kernel, at downstream LLM prefill
   shapes: M = 256 tokens, K = 3072, N in {1024, 3072, 12288}
   (specs/coopmat-hip-wmma.md). rocBLAS is measured by
   tools/probes/rocblas_gemm_f16.cpp with the same method. Each tiling runs
   with B row-major ([k][n], a strided fragment gather on gfx11) and B stored
   transposed ([n][k], [~b_layout:Col_major], contiguous fragment loads).

   Method: GPU events on the default stream around a batch of [batch]
   back-to-back launches (so host enqueue gaps between launches are hidden by
   the queue), [samples] batches per round, [rounds] rounds; per-launch time is
   batch time / batch. Reported: median and min-max over all samples, in
   microseconds and TFLOP/s (2*M*N*K flops). The GPU is shared with other
   users, so the spread is part of the result. Information only: no threshold.

   Usage: bench_hip_coopmat_gemm.exe [rounds] [samples] [batch] *)

open Sarek
module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer
module Profiling = Spoc_core.Profiling
module Gemm = Sarek_gemm.Coopmat

let () = Sarek_hip.Hip_plugin.init ()

let arg i default =
  if Array.length Sys.argv > i then int_of_string Sys.argv.(i) else default

let rounds = arg 1 3

let samples = arg 2 10

let batch = arg 3 10

let median xs =
  let a = Array.of_list xs in
  Array.sort compare a ;
  let n = Array.length a in
  if n mod 2 = 1 then a.(n / 2) else (a.((n / 2) - 1) +. a.(n / 2)) /. 2.0

(** Per-launch microseconds: [rounds * samples] batches of [batch] launches. *)
let measure dev launch =
  launch () ;
  Transfer.flush dev ;
  let timer = Profiling.create_timer dev in
  let per_round =
    List.init rounds (fun _ ->
        List.init samples (fun _ ->
            Profiling.timer_start timer ;
            for _ = 1 to batch do
              launch ()
            done ;
            Profiling.timer_stop timer ;
            Profiling.timer_elapsed_ms timer *. 1000.0 /. float_of_int batch))
  in
  Profiling.timer_destroy timer ;
  per_round

let report ~name ~m ~n ~k per_round =
  let all = List.concat per_round in
  let tflops us =
    2.0 *. float m *. float n *. float k /. (us *. 1e-6) /. 1e12
  in
  let lo = List.fold_left min infinity all
  and hi = List.fold_left max 0.0 all in
  let med = median all in
  Printf.printf
    "%-28s M=%d N=%-5d K=%d  median %9.1f us (%6.2f TFLOP/s)  min %9.1f max \
     %9.1f us  round medians %s\n\
     %!"
    name
    m
    n
    k
    med
    (tflops med)
    lo
    hi
    (String.concat
       " / "
       (List.map (fun r -> Printf.sprintf "%.1f" (median r)) per_round))

let f16_vec len =
  let v = Vector.create Vector.float16 len in
  for i = 0 to len - 1 do
    Vector.set v i (float_of_int ((i * 7919 mod 17) - 8) /. 16.0)
  done ;
  v

let f32_vec len =
  let v = Vector.create Vector.float32 len in
  for i = 0 to len - 1 do
    Vector.set v i (float_of_int ((i * 104729 mod 13) - 6) /. 16.0)
  done ;
  v

let bench_coopmat dev ~m ~n ~k ~b_layout tiling =
  let a = f16_vec (m * k) and b = f16_vec (k * n) in
  let c = f32_vec (m * n) and d = f32_vec (m * n) in
  let launch () =
    Gemm.run ~device:dev ~tiling ~b_layout ~a ~b ~c ~d ~m ~n ~k ()
  in
  report
    ~name:
      (Printf.sprintf
         "coopmat f16 %dx%d/w%d%s"
         tiling.Gemm.frags_m
         tiling.frags_n
         tiling.waves
         (match b_layout with
         | Sarek_coopmat.Row_major -> ""
         | Sarek_coopmat.Col_major -> " B^T"))
    ~m
    ~n
    ~k
    (measure dev launch)

let scalar_ir =
  match (snd Sarek_gemm.sgemm_tiled_kernel).Sarek.Kirc_types.body_ir with
  | Some ir -> ir
  | None -> failwith "Sarek_gemm.sgemm_tiled_kernel has no IR"

let bench_scalar dev ~m ~n ~k =
  let a = f32_vec (m * k) and b = f32_vec (k * n) and c = f32_vec (m * n) in
  let i x = Execute.Int32 (Int32.of_int x) in
  let launch () =
    Execute.run_vectors
      ~device:dev
      ~ir:scalar_ir
      ~args:
        Execute.[Vec a; Vec b; Vec c; i m; i n; i k; Float32 1.0; Float32 0.0]
      ~block:(Sarek_gemm.Host.block ())
      ~grid:(Sarek_gemm.Host.grid ~m ~n)
      ()
  in
  report ~name:"scalar f32 Sarek_gemm" ~m ~n ~k (measure dev launch)

let tilings =
  Gemm.
    [
      {frags_m = 1; frags_n = 1; waves = 4};
      {frags_m = 2; frags_n = 2; waves = 4};
      {frags_m = 2; frags_n = 2; waves = 2};
      {frags_m = 4; frags_n = 2; waves = 2};
      {frags_m = 2; frags_n = 4; waves = 2};
      {frags_m = 4; frags_n = 4; waves = 1};
    ]

let () =
  match
    List.filter
      (fun d ->
        d.Device.framework = "HIP"
        && Sarek_capability.permits
             (Sarek_coopmat.verdict
                ~support:
                  d.Device.capabilities.Spoc_framework.Framework_sig.coopmat
                Gemm.config))
      (Array.to_list (Device.init ()))
  with
  | [] -> print_endline "[SKIP] no gfx11 HIP device"
  | dev :: _ ->
      Sarek_coopmat.accept_relaxed_accumulation
        ~reason:"benchmark: throughput, not accuracy, is measured here" ;
      Printf.printf
        "device: %s; rounds=%d samples=%d batch=%d\n%!"
        dev.Device.name
        rounds
        samples
        batch ;
      List.iter
        (fun n ->
          let m = 256 and k = 3072 in
          List.iter
            (fun b_layout ->
              List.iter (bench_coopmat dev ~m ~n ~k ~b_layout) tilings)
            [Sarek_coopmat.Row_major; Sarek_coopmat.Col_major] ;
          bench_scalar dev ~m ~n ~k)
        [1024; 3072; 12288]
