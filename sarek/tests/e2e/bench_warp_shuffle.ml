(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Micro-benchmark (docs/plans/warp-shuffle-intake-2026-10-17.md, AC-9): the
 * 8-lane float32 sum of a downstream gated-delta-net reduction, as a xor tree of
 * three shuffles against shared memory with two barriers per reduction.
 *
 * Each thread runs [iters] dependent reductions (the sum feeds the next), so
 * the loop measures the reduction and not the launch. The two kernels add the
 * same 8 values in a different order, so the results are close, not equal; the
 * value is only printed to keep the work alive. Information only: no gate.
 * Alternating A/B pairs, median reported, with the load average read before
 * and after. Run by hand on one device:
 *   HIP_VISIBLE_DEVICES=0 dune exec sarek/tests/e2e/bench_warp_shuffle.exe
 ******************************************************************************)

[@@@warning "-33"]

open Sarek
module Std = Sarek_stdlib.Std
module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer

let () =
  Test_helpers.Benchmarks.init_backends () ;
  Backend_hip.init ()

let iters = 4096

let shuffle_kernel =
  [%kernel
    fun (inp : float32 vector) (out : float32 vector) (n : int32) ->
      let open Std in
      let g = (block_idx_x * block_dim_x) + thread_idx_x in
      let acc = mut inp.(g) in
      for _i = 1 to n do
        let s1 = acc +. warp_shuffle_xor acc 4l in
        let s2 = s1 +. warp_shuffle_xor s1 2l in
        acc := (s2 +. warp_shuffle_xor s2 1l) *. 0.125
      done ;
      out.(g) <- acc]

let shared_kernel =
  [%kernel
    fun (inp : float32 vector) (out : float32 vector) (n : int32) ->
      let open Std in
      let%shared (sh : float32) = 256l in
      let t = thread_idx_x in
      let g = (block_idx_x * block_dim_x) + t in
      let base = t land 0xFFFFFFF8l in
      let acc = mut inp.(g) in
      for _i = 1 to n do
        sh.(t) <- acc ;
        block_barrier () ;
        let s =
          sh.(base)
          +. sh.(base + 1l)
          +. sh.(base + 2l)
          +. sh.(base + 3l)
          +. sh.(base + 4l)
          +. sh.(base + 5l)
          +. sh.(base + 6l)
          +. sh.(base + 7l)
        in
        block_barrier () ;
        acc := s *. 0.125
      done ;
      out.(g) <- acc]

let ir_of (_, kirc) =
  match kirc.Kirc_types.body_ir with
  | Some ir -> ir
  | None -> failwith "kernel has no IR"

let block = 256

let blocks = 4096

let n = block * blocks

let time dev ir inp out =
  let t0 = Unix.gettimeofday () in
  Execute.run_vectors
    ~device:dev
    ~ir
    ~args:[Execute.Vec inp; Execute.Vec out; Execute.Int iters]
    ~block:(Execute.dims1d block)
    ~grid:(Execute.dims1d blocks)
    () ;
  Transfer.flush dev ;
  Unix.gettimeofday () -. t0

let median l =
  let a = Array.of_list (List.sort compare l) in
  a.(Array.length a / 2)

let load () =
  try
    let ic = open_in "/proc/loadavg" in
    let l = input_line ic in
    close_in ic ;
    l
  with _ -> "unknown"

let bench dev =
  let inp = Vector.create Vector.float32 n in
  let out = Vector.create Vector.float32 n in
  for i = 0 to n - 1 do
    Vector.set inp i (1.0 +. (float_of_int (i mod 7) *. 0.25)) ;
    Vector.set out i 0.0
  done ;
  let shuffle_ir = ir_of shuffle_kernel and shared_ir = ir_of shared_kernel in
  ignore (time dev shuffle_ir inp out) ;
  ignore (time dev shared_ir inp out) ;
  let load_before = load () in
  let pairs =
    List.init 7 (fun _ ->
        let a = time dev shuffle_ir inp out in
        let b = time dev shared_ir inp out in
        (a, b))
  in
  let sh = median (List.map fst pairs) and sm = median (List.map snd pairs) in
  let checksum = Vector.get out 0 in
  Printf.printf
    "%s | %s | %d reductions/thread, %d threads | shuffle %.2f ms, shared+2 \
     barriers %.2f ms, ratio %.2fx | load before: %s, after: %s | out[0]=%g\n\
     %!"
    dev.Device.framework
    dev.Device.name
    iters
    n
    (sh *. 1000.)
    (sm *. 1000.)
    (sm /. sh)
    load_before
    (load ())
    checksum

let () =
  Array.iter
    (fun (dev : Device.t) ->
      if dev.Device.framework = "HIP" || dev.Device.framework = "CUDA/PTX" then
        match bench dev with
        | () -> ()
        | exception e ->
            Printf.printf
              "%s | %s | FAILED: %s\n%!"
              dev.Device.framework
              dev.Device.name
              (Printexc.to_string e))
    (Device.init ())
