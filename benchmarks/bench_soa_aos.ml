(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Tier 1a benchmark: AoS vs SoA memory-bound single-field access.
 *
 * A wide 8-field record ([wide], 32-byte stride) is read one field at a time
 * by a copy kernel — the textbook uncoalesced case. AoS stores the record
 * packed, so consecutive threads' reads of field [f0] are 32 bytes apart
 * (1/8 bus efficiency); SoA stores [f0] in its own contiguous float32 buffer,
 * restoring full coalescing. The SoA input is produced from the AoS buffer via
 * the host transpose (Spoc_core.Soa.scatter), then fed to a plain scalar
 * kernel.
 *
 * Reports median kernel time (excludes the one-time H2D upload; transfers are
 * amortised across [iters]) and effective single-field bandwidth for both
 * layouts. GPU device preferred.
 *
 *   LD_LIBRARY_PATH=$HOME/opt/zluda \
 *     dune exec --root . benchmarks/bench_soa_aos.exe
 ******************************************************************************)

module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer
module Soa = Spoc_core.Soa

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

(* AoS: bind the record, use one field. *)
let aos_kernel =
  snd
    [%kernel
      fun (pts : wide vector) (out : float32 vector) (n : int32) ->
        let tid = thread_idx_x + (block_idx_x * block_dim_x) in
        if tid < n then
          let p = pts.(tid) in
          out.(tid) <- p.f0]

(* SoA: read the field's own contiguous array. *)
let soa_kernel =
  snd
    [%kernel
      fun (f0 : float32 vector) (out : float32 vector) (n : int32) ->
        let tid = thread_idx_x + (block_idx_x * block_dim_x) in
        if tid < n then out.(tid) <- f0.(tid)]

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

let plan =
  Soa.plan
    ~name:"wide"
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

let time_kernel dev ~ir ~args ~block ~grid =
  for _ = 1 to warmup do
    Sarek.Execute.run_vectors ~device:dev ~ir ~args ~block ~grid () ;
    Transfer.flush dev
  done ;
  let samples =
    Array.init iters (fun _ ->
        let t0 = Unix.gettimeofday () in
        Sarek.Execute.run_vectors ~device:dev ~ir ~args ~block ~grid () ;
        Transfer.flush dev ;
        (Unix.gettimeofday () -. t0) *. 1000.0)
  in
  median samples

let run dev n =
  let threads = 256 in
  let grid_x = (n + threads - 1) / threads in
  let block = Sarek.Execute.dims1d threads in
  let grid = Sarek.Execute.dims1d grid_x in
  (* AoS source. *)
  let pts = Vector.create_custom wide_custom n in
  for i = 0 to n - 1 do
    let v = float_of_int i in
    Vector.set
      pts
      i
      {f0 = v; f1 = v; f2 = v; f3 = v; f4 = v; f5 = v; f6 = v; f7 = v}
  done ;
  (* SoA: scatter, keep only the f0 leaf for the read. *)
  let leaves = Array.init 8 (fun _ -> Vector.create Vector.float32 n) in
  Soa.scatter
    plan
    ~aos:(Vector.to_ctypes_ptr pts)
    ~length:n
    ~leaves:(Array.map Vector.to_ctypes_ptr leaves) ;
  let out_a = Vector.create Vector.float32 n in
  let out_s = Vector.create Vector.float32 n in
  let t_aos =
    time_kernel
      dev
      ~ir:(ir_of aos_kernel)
      ~args:
        [Sarek.Execute.Vec pts; Sarek.Execute.Vec out_a; Sarek.Execute.Int n]
      ~block
      ~grid
  in
  let t_soa =
    time_kernel
      dev
      ~ir:(ir_of soa_kernel)
      ~args:
        [
          Sarek.Execute.Vec leaves.(0);
          Sarek.Execute.Vec out_s;
          Sarek.Execute.Int n;
        ]
      ~block
      ~grid
  in
  (t_aos, t_soa)

(* Effective single-field bandwidth: n elements read (4B) + n written (4B). *)
let field_gbps n ms =
  if ms <= 0.0 then 0.0 else float_of_int (n * 4 * 2) /. (ms /. 1000.0) /. 1.0e9

let () =
  Benchmark_backends.Backend_loader.init () ;
  let devs = Device.all () in
  if Array.length devs = 0 then (
    print_endline "bench_soa_aos: no device - SKIPPED" ;
    exit 0) ;
  let dev =
    match Array.find_opt Device.is_gpu devs with
    | Some d -> d
    | None -> devs.(0)
  in
  Printf.printf
    "=== AoS vs SoA single-field copy (device: %s, 8-field record, 32B stride) \
     ===\n\
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
