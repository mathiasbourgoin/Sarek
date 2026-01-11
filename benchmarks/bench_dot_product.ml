(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Dot Product Benchmark - Map-reduce combination *)

open Benchmark_common
open Benchmark_backends
module Device = Spoc_core.Device
module Vector = Spoc_core.Vector

type config = {
  sizes : int list;
  block_size : int;
  iterations : int;
  warmup : int;
  output_dir : string;
  device_filter : Device.t -> bool;
}

let default_config =
  {
    sizes = [1_000_000; 10_000_000; 50_000_000; 100_000_000];
    block_size = 256;
    iterations = 20;
    warmup = 5;
    output_dir = "results";
    device_filter =
      (fun dev ->
        dev.Device.framework <> "Native"
        && dev.Device.framework <> "Interpreter");
  }

let dot_product_kernel =
  [%kernel
    fun (a : float32 vector)
        (b : float32 vector)
        (output : float32 vector)
        (n : int32) ->
      let%shared (sdata : float32) = 256l in
      let tid = thread_idx_x in
      let gid = thread_idx_x + (block_dim_x * block_idx_x) in
      let%superstep load =
        if gid < n then sdata.(tid) <- a.(gid) *. b.(gid)
        else sdata.(tid) <- 0.0
      in
      let%superstep reduce128 =
        if tid < 128l then sdata.(tid) <- sdata.(tid) +. sdata.(tid + 128l)
      in
      let%superstep reduce64 =
        if tid < 64l then sdata.(tid) <- sdata.(tid) +. sdata.(tid + 64l)
      in
      let%superstep reduce32 =
        if tid < 32l then sdata.(tid) <- sdata.(tid) +. sdata.(tid + 32l)
      in
      let%superstep reduce16 =
        if tid < 16l then sdata.(tid) <- sdata.(tid) +. sdata.(tid + 16l)
      in
      let%superstep reduce8 =
        if tid < 8l then sdata.(tid) <- sdata.(tid) +. sdata.(tid + 8l)
      in
      let%superstep reduce4 =
        if tid < 4l then sdata.(tid) <- sdata.(tid) +. sdata.(tid + 4l)
      in
      let%superstep reduce2 =
        if tid < 2l then sdata.(tid) <- sdata.(tid) +. sdata.(tid + 2l)
      in
      let%superstep reduce1 =
        if tid < 1l then sdata.(tid) <- sdata.(tid) +. sdata.(tid + 1l)
      in
      if tid = 0l then output.(block_idx_x) <- sdata.(0l)]
[@@warning "-33"]

let benchmark_device dev size config =
  Printf.printf
    "  Device: %s (%s), Size: %d\n"
    dev.Device.name
    dev.Device.framework
    size ;
  let n = size in
  let a = Array.init n (fun i -> float_of_int (i mod 100) /. 100.0) in
  let b = Array.init n (fun i -> float_of_int ((i + 1) mod 100) /. 100.0) in
  let expected = ref 0.0 in
  for i = 0 to n - 1 do
    expected := !expected +. (a.(i) *. b.(i))
  done ;
  let _, kirc = dot_product_kernel in
  let ir =
    match kirc.Sarek.Kirc_types.body_ir with
    | Some ir -> ir
    | None -> failwith "No IR"
  in
  let va = Vector.create Vector.float32 n in
  let vb = Vector.create Vector.float32 n in
  for i = 0 to n - 1 do
    Vector.set va i a.(i) ;
    Vector.set vb i b.(i)
  done ;
  let block_sz = config.block_size in
  let grid_sz = (n + block_sz - 1) / block_sz in
  let vout = Vector.create Vector.float32 grid_sz in
  let block = Sarek.Execute.dims1d block_sz in
  let grid = Sarek.Execute.dims1d grid_sz in
  let shared_mem = block_sz * 4 in
  for _ = 1 to config.warmup do
    Sarek.Execute.run_vectors
      ~device:dev
      ~ir
      ~args:
        [
          Sarek.Execute.Vec va;
          Sarek.Execute.Vec vb;
          Sarek.Execute.Vec vout;
          Sarek.Execute.Int32 (Int32.of_int n);
        ]
      ~block
      ~grid
      ~shared_mem
      () ;
    Spoc_core.Transfer.flush dev
  done ;
  let times = ref [] in
  for _ = 1 to config.iterations do
    let t0 = Unix.gettimeofday () in
    Sarek.Execute.run_vectors
      ~device:dev
      ~ir
      ~args:
        [
          Sarek.Execute.Vec va;
          Sarek.Execute.Vec vb;
          Sarek.Execute.Vec vout;
          Sarek.Execute.Int32 (Int32.of_int n);
        ]
      ~block
      ~grid
      ~shared_mem
      () ;
    Spoc_core.Transfer.flush dev ;
    let t1 = Unix.gettimeofday () in
    times := (t1 -. t0) :: !times
  done ;
  let partial = Vector.to_array vout in
  let result = Array.fold_left ( +. ) 0.0 partial in
  let verified = abs_float (result -. !expected) < 0.001 *. !expected in
  let times_array = Array.of_list !times in
  let median_ms = Common.median times_array in
  let bandwidth_gb_s = 2.0 *. float_of_int n *. 4.0 /. (median_ms *. 1e9) in
  Printf.printf
    "  Median: %.3f ms, BW: %.3f GB/s, Verified: %s\n"
    median_ms
    bandwidth_gb_s
    (if verified then "✓" else "✗") ;
  Output.
    {
      device_id = dev.Device.id;
      device_name = dev.Device.name;
      framework = dev.Device.framework;
      iterations = times_array;
      mean_ms = Common.mean times_array;
      stddev_ms = Common.stddev times_array;
      median_ms;
      min_ms = Common.min times_array;
      max_ms = Common.max times_array;
      throughput = Some bandwidth_gb_s;
      verified = Some verified;
    }

let run_benchmark config =
  Printf.printf "Dot Product Benchmark\n=====================\n\n" ;
  Backend_loader.init () ;
  let devices =
    Device.init () |> Array.to_list
    |> List.filter config.device_filter
    |> Array.of_list
  in
  if Array.length devices = 0 then (
    Printf.eprintf "No devices\n" ;
    exit 1) ;
  if not (Sys.file_exists config.output_dir) then
    Unix.mkdir config.output_dir 0o755 ;
  let system_info = System_info.collect devices in
  let git_commit = Common.get_git_commit () in
  List.iter
    (fun size ->
      Printf.printf "\n--- Size: %d ---\n" size ;
      let results =
        Array.to_list devices
        |> List.mapi (fun device_id dev ->
            try
              let r = benchmark_device dev size config in
              {r with Output.device_id}
            with e ->
              Printf.eprintf "Error: %s\n" (Printexc.to_string e) ;
              Output.
                {
                  device_id;
                  device_name = dev.Device.name;
                  framework = dev.Device.framework;
                  iterations = [||];
                  mean_ms = 0.0;
                  stddev_ms = 0.0;
                  median_ms = 0.0;
                  min_ms = 0.0;
                  max_ms = 0.0;
                  throughput = None;
                  verified = Some false;
                })
      in
      let result =
        Output.
          {
            params =
              {
                name = "dot_product";
                size;
                block_size = config.block_size;
                iterations = config.iterations;
                warmup = config.warmup;
              };
            timestamp = Common.get_timestamp ();
            git_commit;
            system = system_info;
            results;
          }
      in
      let filename =
        Output.make_filename
          ~output_dir:config.output_dir
          ~benchmark_name:"dot_product"
          ~size
      in
      Output.write_json filename result ;
      Printf.printf "Written: %s\n" filename)
    config.sizes

let () =
  let config = ref default_config in
  Arg.parse [] (fun _ -> ()) "Dot product benchmark" ;
  run_benchmark !config
