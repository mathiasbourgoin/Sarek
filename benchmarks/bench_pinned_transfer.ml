(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Tier 1a benchmark: pinned (page-locked) vs pageable host memory.
 *
 * Measures H2D and D2H bandwidth for host<->device transfers at several sizes,
 * using the *blocking* cuMemcpyHtoD/DtoH path (the same path every SPOC
 * transfer uses today). Pinned host memory lets the driver DMA directly
 * instead of staging through an internal pageable bounce buffer, so it is the
 * cheapest transfer-bandwidth lever and the prerequisite for async overlap.
 *
 * CUDA-backend only (runs under ZLUDA on the RX 7900 XTX):
 *   LD_LIBRARY_PATH=$HOME/opt/zluda \
 *     dune exec --root . benchmarks/bench_pinned_transfer.exe
 *
 * The pinned host-memory APIs (cuMemAllocHost / cuMemHostRegister) are probed
 * for support at startup. When the active driver does not implement them
 * (ZLUDA v7-preview.3 returns CUDA_ERROR_OTHER for both), the benchmark still
 * reports pageable bandwidth and states that the pinned comparison is
 * unavailable on this driver. Exits 0 (skips cleanly) with no CUDA device.
 ******************************************************************************)

open Ctypes
module Cuda = Sarek_cuda.Cuda_api
module Berr = Sarek_backend_error.Backend_error

let iters = 30

let warmup = 5

(* Transfer sizes in bytes. *)
let sizes_mb = [1; 4; 16; 64; 256]

let bytes_of_mb mb = mb * 1024 * 1024

(* Median of a float array (robust against the occasional scheduling spike). *)
let median arr =
  let a = Array.copy arr in
  Array.sort Float.compare a ;
  let n = Array.length a in
  if n = 0 then 0.0
  else if n mod 2 = 1 then a.(n / 2)
  else (a.((n / 2) - 1) +. a.(n / 2)) /. 2.0

let time_median f =
  for _ = 1 to warmup do
    f ()
  done ;
  let samples =
    Array.init iters (fun _ ->
        let t0 = Unix.gettimeofday () in
        f () ;
        (Unix.gettimeofday () -. t0) *. 1000.0)
  in
  median samples

(* GB/s from milliseconds for [bytes] moved. *)
let gbps bytes ms =
  if ms <= 0.0 then 0.0 else float_of_int bytes /. (ms /. 1000.0) /. 1.0e9

(* Does the active driver implement pinned host memory? A single tiny
   cuMemAllocHost tells us; ZLUDA returns CUDA_ERROR_OTHER. *)
let pinned_supported () =
  try
    let p = Cuda.Memory.alloc_host 4096 in
    Cuda.Memory.free_host p ;
    true
  with Berr.Backend_error _ -> false

type row = {
  mb : int;
  h2d : float;
  d2h : float;
  h2d_pin : float option;
  d2h_pin : float option;
}

let bench_size dev ~pinned mb =
  let bytes = bytes_of_mb mb in
  let dbuf = Cuda.Memory.alloc_custom dev ~size:bytes ~elem_size:1 in
  let page = allocate_n uint8_t ~count:bytes in
  let page_ptr = to_voidp page in
  let h2d_fn src_ptr () =
    Cuda.Memory.host_ptr_to_device ~src_ptr ~byte_size:bytes ~dst:dbuf ;
    Cuda.Device.synchronize dev
  in
  let d2h_fn dst_ptr () =
    Cuda.Memory.device_to_host_ptr ~src:dbuf ~dst_ptr ~byte_size:bytes ;
    Cuda.Device.synchronize dev
  in
  let h2d = gbps bytes (time_median (h2d_fn page_ptr)) in
  let d2h = gbps bytes (time_median (d2h_fn page_ptr)) in
  let h2d_pin, d2h_pin =
    if not pinned then (None, None)
    else begin
      let pin = Cuda.Memory.alloc_host bytes in
      let hp = gbps bytes (time_median (h2d_fn pin.Cuda.Memory.host_ptr)) in
      let dp = gbps bytes (time_median (d2h_fn pin.Cuda.Memory.host_ptr)) in
      Cuda.Memory.free_host pin ;
      (Some hp, Some dp)
    end
  in
  Cuda.Memory.free dbuf ;
  {mb; h2d; d2h; h2d_pin; d2h_pin}

let print_table ~pinned rows =
  if pinned then begin
    Printf.printf
      "\n%-8s | %-18s | %-18s | %-9s\n"
      "size"
      "H2D GB/s (pg/pin)"
      "D2H GB/s (pg/pin)"
      "speedup" ;
    Printf.printf "%s\n" (String.make 63 '-') ;
    List.iter
      (fun r ->
        let hp = Option.value ~default:0.0 r.h2d_pin in
        let dp = Option.value ~default:0.0 r.d2h_pin in
        Printf.printf
          "%5dMB  | %7.2f / %7.2f  | %7.2f / %7.2f  | %.2fx/%.2fx\n"
          r.mb
          r.h2d
          hp
          r.d2h
          dp
          (hp /. r.h2d)
          (dp /. r.d2h))
      rows
  end
  else begin
    Printf.printf "\n%-8s | %-12s | %-12s\n" "size" "H2D GB/s" "D2H GB/s" ;
    Printf.printf "%s\n" (String.make 38 '-') ;
    List.iter
      (fun r -> Printf.printf "%5dMB  | %10.2f | %10.2f\n" r.mb r.h2d r.d2h)
      rows
  end

let () =
  Cuda.Device.init () ;
  if Cuda.Device.count () = 0 then (
    print_endline "bench_pinned_transfer: no CUDA device present - SKIPPED" ;
    exit 0) ;
  let dev = Cuda.Device.get 0 in
  Cuda.Device.set_current dev ;
  Printf.printf
    "=== Pinned vs pageable host-memory transfer (device: %s) ===\n%!"
    dev.Cuda.Device.name ;
  let pinned = pinned_supported () in
  if not pinned then
    Printf.printf
      "NOTE: this driver does not implement pinned host memory \
       (cuMemAllocHost/cuMemHostRegister return CUDA_ERROR_OTHER; observed on \
       ZLUDA v7-preview.3). Reporting pageable bandwidth only; the pinned \
       comparison requires a driver with pinned-memory support.\n\
       %!" ;
  let rows = List.map (bench_size dev ~pinned) sizes_mb in
  print_table ~pinned rows
