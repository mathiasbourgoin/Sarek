(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Tier 1a SoA correctness: AoS and SoA produce identical values on every
 * device.
 *
 * A [point3d] custom (AoS) vector is filled with source data. The host-side
 * SoA transpose (Spoc_core.Soa.scatter) derives three contiguous float32 leaf
 * vectors (xs/ys/zs) from that AoS buffer. Two kernels compute the same
 * per-element reduction x+y+z:
 *   - AoS kernel reads the whole record  [let p = pts.(tid) in p.x+p.y+p.z]
 *   - SoA kernel reads three scalar arrays [xs.(tid)+ys.(tid)+zs.(tid)]
 * The two device results must match each other and a pure-OCaml reference, on
 * every available device (Native/Interpreter always, plus any GPU backend).
 *
 * This exercises the SoA storage plan + transpose + (scalar) transfer path
 * end-to-end. Device-side SoA addressing of a single custom vector value is
 * the Tier 1b emitter handoff and is not exercised here.
 ******************************************************************************)

module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer
module Soa = Spoc_core.Soa
module Benchmarks = Test_helpers.Benchmarks

type ('a, 'b) vector = ('a, 'b) Vector.t

type float32 = float

type point3d = {x : float32; y : float32; z : float32} [@@sarek.type]

let aos_kernel =
  snd
    [%kernel
      fun (pts : point3d vector) (out : float32 vector) (n : int32) ->
        let tid = thread_idx_x + (block_idx_x * block_dim_x) in
        if tid < n then
          let p = pts.(tid) in
          out.(tid) <- p.x +. p.y +. p.z]

let soa_kernel =
  snd
    [%kernel
      fun (xs : float32 vector)
          (ys : float32 vector)
          (zs : float32 vector)
          (out : float32 vector)
          (n : int32) ->
        let tid = thread_idx_x + (block_idx_x * block_dim_x) in
        if tid < n then out.(tid) <- xs.(tid) +. ys.(tid) +. zs.(tid)]

let ir_of kirc =
  match kirc.Sarek.Kirc_types.body_ir with
  | Some ir -> ir
  | None -> failwith "kernel has no IR"

let run_device dev n =
  let threads = min 128 n in
  let grid_x = (n + threads - 1) / threads in
  let block = Sarek.Execute.dims1d threads in
  let grid = Sarek.Execute.dims1d grid_x in
  (* Source data + AoS vector. *)
  let src = Vector.create_custom point3d_custom n in
  for i = 0 to n - 1 do
    Vector.set
      src
      i
      {
        x = float_of_int i;
        y = (float_of_int i *. 0.5) +. 1.0;
        z = float_of_int (n - i);
      }
  done ;
  (* SoA leaves derived from the AoS buffer via the host transpose. *)
  let plan =
    Soa.plan
      ~name:"point3d"
      Sarek_ir_types.[("x", TFloat32); ("y", TFloat32); ("z", TFloat32)]
  in
  let xs = Vector.create Vector.float32 n in
  let ys = Vector.create Vector.float32 n in
  let zs = Vector.create Vector.float32 n in
  Soa.scatter
    plan
    ~aos:(Vector.to_ctypes_ptr src)
    ~length:n
    ~leaves:
      [|
        Vector.to_ctypes_ptr xs;
        Vector.to_ctypes_ptr ys;
        Vector.to_ctypes_ptr zs;
      |] ;
  (* AoS kernel. *)
  let out_aos = Vector.create Vector.float32 n in
  Sarek.Execute.run_vectors
    ~device:dev
    ~ir:(ir_of aos_kernel)
    ~args:
      [Sarek.Execute.Vec src; Sarek.Execute.Vec out_aos; Sarek.Execute.Int n]
    ~block
    ~grid
    () ;
  Transfer.flush dev ;
  (* SoA kernel. *)
  let out_soa = Vector.create Vector.float32 n in
  Sarek.Execute.run_vectors
    ~device:dev
    ~ir:(ir_of soa_kernel)
    ~args:
      [
        Sarek.Execute.Vec xs;
        Sarek.Execute.Vec ys;
        Sarek.Execute.Vec zs;
        Sarek.Execute.Vec out_soa;
        Sarek.Execute.Int n;
      ]
    ~block
    ~grid
    () ;
  Transfer.flush dev ;
  (out_aos, out_soa, src)

let () =
  Benchmarks.init () ;
  let n = 1024 in
  let devs = Device.all () in
  if Array.length devs = 0 then (
    print_endline "test_soa_aos_equiv: no device - SKIPPED" ;
    exit 0) ;
  let any_failure = ref false in
  Array.iter
    (fun dev ->
      Printf.printf
        "SoA/AoS equiv [%s] %s: %!"
        dev.Device.framework
        dev.Device.name ;
      try
        let out_aos, out_soa, src = run_device dev n in
        let ok = ref true in
        for i = 0 to n - 1 do
          let p = Vector.get src i in
          let reference = p.x +. p.y +. p.z in
          let a = Vector.get out_aos i in
          let s = Vector.get out_soa i in
          if abs_float (a -. s) > 1e-4 || abs_float (a -. reference) > 1e-3 then (
            ok := false ;
            if i < 5 then
              Printf.printf
                "\n  Mismatch at %d: aos=%f soa=%f ref=%f%!"
                i
                a
                s
                reference)
        done ;
        if !ok then print_endline "PASSED"
        else (
          any_failure := true ;
          print_endline "FAILED")
      with e ->
        any_failure := true ;
        Printf.printf "FAIL (%s)\n%!" (Printexc.to_string e))
    devs ;
  if !any_failure then exit 1
