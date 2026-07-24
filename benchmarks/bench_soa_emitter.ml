(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Tier 1b benchmark: device-side SoA EMITTER vs AoS, single-field access.
 *
 * Unlike bench_soa_aos (which compares an AoS record kernel against a
 * hand-written scalar kernel over a separate leaf vector), this benchmarks the
 * SAME custom-vector kernel IR compiled two ways by the Tier 1b emitter:
 *   - AoS: Execute.run_vectors, single packed base pointer, 32B-strided read of
 *     field f0 (1/8 bus efficiency on an 8-field record).
 *   - SoA: Sarek_ir_ptx.generate ~soa_params:["pts"] → 8 per-leaf base pointers;
 *     the f0 read becomes a fully coalesced scalar load of f0's own contiguous
 *     buffer. Launched via run_source ~inject_lengths:false with the 8 leaf
 *     buffers (only f0 is dereferenced; the rest are unused ABI slots).
 *
 * This isolates the emitter's coalescing win. Reports median kernel time
 * (transfers amortised across [iters]) + effective single-field bandwidth.
 * CUDA/PTX device required for the SoA leg.
 *
 *   LD_LIBRARY_PATH=$HOME/opt/zluda PATH=/opt/cuda/bin:$PATH \
 *     dune exec --root . benchmarks/bench_soa_emitter.exe
 ******************************************************************************)

module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer
module Soa = Spoc_core.Soa
open Sarek_codegen

type ('a, 'b) vector = ('a, 'b) Vector.t

type float32 = float

type wide = {
  f0 : float32;
  f1 : float32;
  f2 : float32;
  f3 : float32;
  f4 : float32;
  f5 : float32;
  f6 : float32;
  f7 : float32;
}
[@@sarek.type]

(* One custom-vector kernel, reading a single field — compiled AoS and SoA. *)
let kernel =
  snd
    [%kernel
      fun (pts : wide vector) (out : float32 vector) (n : int32) ->
        let tid = thread_idx_x + (block_idx_x * block_dim_x) in
        if tid < n then
          let p = pts.(tid) in
          out.(tid) <- p.f0]

let ir_of kirc =
  match kirc.Sarek.Kirc_types.body_ir with
  | Some ir -> ir
  | None -> failwith "kernel has no IR"

let iters = 50

let warmup = 10

let median arr =
  let a = Array.copy arr in
  Array.sort Float.compare a ;
  let n = Array.length a in
  if n = 0 then 0.0 else a.(n / 2)

let fields =
  Sarek_ir_types.
    [
      ("f0", TFloat32);
      ("f1", TFloat32);
      ("f2", TFloat32);
      ("f3", TFloat32);
      ("f4", TFloat32);
      ("f5", TFloat32);
      ("f6", TFloat32);
      ("f7", TFloat32);
    ]

let plan = Soa.plan ~name:"wide" fields

let is_ptx (dev : Device.t) = dev.Device.framework = "CUDA/PTX"

(* Median wall time (ms) of [launch] over [iters], after [warmup] launches. *)
let time launch =
  for _ = 1 to warmup do
    launch ()
  done ;
  median
    (Array.init iters (fun _ ->
         let t0 = Unix.gettimeofday () in
         launch () ;
         (Unix.gettimeofday () -. t0) *. 1000.0))

let run dev n =
  let threads = 256 in
  let block = Sarek.Execute.dims1d threads in
  let grid = Sarek.Execute.dims1d ((n + threads - 1) / threads) in
  let ir = ir_of kernel in
  (* AoS source. *)
  let pts = Vector.create_custom wide_custom n in
  for i = 0 to n - 1 do
    let v = float_of_int i in
    Vector.set
      pts
      i
      {f0 = v; f1 = v; f2 = v; f3 = v; f4 = v; f5 = v; f6 = v; f7 = v}
  done ;
  let out_a = Vector.create Vector.float32 n in
  let t_aos =
    time (fun () ->
        Sarek.Execute.run_vectors
          ~device:dev
          ~ir
          ~args:
            [
              Sarek.Execute.Vec pts; Sarek.Execute.Vec out_a; Sarek.Execute.Int n;
            ]
          ~block
          ~grid
          () ;
        Transfer.flush dev)
  in
  (* SoA emitter: 8 per-leaf buffers (only f0 read), driven through the emitted
     N-pointer ABI. *)
  let leaves = Array.init 8 (fun _ -> Vector.create Vector.float32 n) in
  Soa.scatter
    plan
    ~aos:(Vector.to_ctypes_ptr pts)
    ~length:n
    ~leaves:(Array.map Vector.to_ctypes_ptr leaves) ;
  let out_s = Vector.create Vector.float32 n in
  let ptx = Sarek_ir_ptx.generate ~soa_params:["pts"] ir in
  let len = Sarek.Execute.Int32 (Int32.of_int n) in
  let soa_args =
    Array.to_list (Array.map (fun v -> Sarek.Execute.Vec v) leaves)
    @ [len; Sarek.Execute.Vec out_s; len; Sarek.Execute.Int n]
  in
  let t_soa =
    time (fun () ->
        Sarek.Execute.run_source
          ~device:dev
          ~source:ptx
          ~lang:Sarek.Execute.PTX
          ~kernel_name:ir.Sarek_ir_types.kern_name
          ~block
          ~grid
          ~inject_lengths:false
          soa_args ;
        Transfer.flush dev)
  in
  (t_aos, t_soa)

(* n elements read (4B) + n written (4B) of the single field. *)
let field_gbps n ms =
  if ms <= 0.0 then 0.0 else float_of_int (n * 4 * 2) /. (ms /. 1000.0) /. 1.0e9

let () =
  Benchmark_backends.Backend_loader.init () ;
  let devs = Device.all () in
  match Array.find_opt is_ptx devs with
  | None ->
      print_endline
        "bench_soa_emitter: no CUDA/PTX device (SoA is PTX-only) - SKIPPED \
         (set LD_LIBRARY_PATH=$HOME/opt/zluda)" ;
      exit 0
  | Some dev ->
      Printf.printf
        "=== SoA-emitter vs AoS single-field copy (device: %s, 8-field record, \
         32B stride) ===\n\
         %!"
        dev.Device.name ;
      Printf.printf
        "%-10s | %-14s | %-14s | %-8s\n"
        "N"
        "AoS ms (GB/s)"
        "SoA ms (GB/s)"
        "speedup" ;
      Printf.printf "%s\n" (String.make 54 '-') ;
      List.iter
        (fun n ->
          let t_aos, t_soa = run dev n in
          Printf.printf
            "%-10d | %6.3f (%5.1f) | %6.3f (%5.1f) | %.2fx\n%!"
            n
            t_aos
            (field_gbps n t_aos)
            t_soa
            (field_gbps n t_soa)
            (if t_soa > 0.0 then t_aos /. t_soa else 0.0))
        [1 lsl 18; 1 lsl 20; 1 lsl 22; 1 lsl 24]
