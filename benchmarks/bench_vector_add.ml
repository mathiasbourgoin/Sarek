(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Vector Addition Benchmark

    Benchmarks element-wise vector addition: C[i] = A[i] + B[i]. This is a
    memory-bound operation, good for testing memory bandwidth. *)

open Sarek
open Benchmark_common
open Benchmark_backends
module Std = Sarek_stdlib.Std
module Device = Spoc_core.Device
module Vector = Spoc_core.Vector

(** Configuration *)
type config = {
  sizes : int list;
  iterations : int;
  warmup : int;
  output_dir : string;
  device_filter : Device.t -> bool;
}

let default_config =
  {
    sizes = [1_000_000; 10_000_000; 50_000_000; 100_000_000];
    iterations = 20;
    warmup = 5;
    output_dir = "results";
    device_filter =
      (fun dev ->
        (* By default, exclude slow CPU backends (Native, Interpreter) *)
        dev.Device.framework <> "Native"
        && dev.Device.framework <> "Interpreter");
  }

(** Pure OCaml baseline *)
let ocaml_vector_add a b c n =
  for i = 0 to n - 1 do
    c.(i) <- a.(i) +. b.(i)
  done

(** Vector addition kernel *)
let vector_add_kernel =
  [%kernel
    fun (a : float32 vector)
        (b : float32 vector)
        (c : float32 vector)
        (n : int32) ->
      let open Std in
      let tid = global_thread_id in
      if tid < n then c.(tid) <- a.(tid) +. b.(tid)]
[@@warning "-33"]

(** Run benchmark for one device and size *)
let benchmark_device dev size config =
  Printf.printf
    "  Device: %s (%s), Size: %d\n"
    dev.Device.name
    dev.Device.framework
    size ;
  flush stdout ;

  (* Initialize vectors *)
  let a = Array.init size (fun i -> float_of_int i *. 0.001) in
  let b = Array.init size (fun i -> float_of_int i *. 0.002) in

  (* Create device vectors *)
  let vec_a = Vector.create Vector.float32 size in
  let vec_b = Vector.create Vector.float32 size in
  let vec_c = Vector.create Vector.float32 size in

  Array.iteri (fun i x -> Vector.set vec_a i x) a ;
  Array.iteri (fun i x -> Vector.set vec_b i x) b ;

  (* Get kernel IR *)
  let _, kirc = vector_add_kernel in
  let ir =
    match kirc.Sarek.Kirc_types.body_ir with
    | Some ir -> ir
    | None -> failwith "No IR"
  in

  (* Setup execution *)
  let block = Execute.dims1d 256 in
  let grid = Execute.dims1d ((size + 255) / 256) in

  (* Ensure device is ready and kernel is compiled *)
  Device.synchronize dev ;

  (* Warmup runs to ensure kernel is compiled and cached *)
  for _ = 1 to config.warmup do
    Execute.run_vectors
      ~device:dev
      ~ir
      ~args:[Vec vec_a; Vec vec_b; Vec vec_c; Int size]
      ~block
      ~grid
      () ;
    Device.synchronize dev
  done ;

  (* Benchmark - time only kernel execution, not transfers *)
  let times =
    Array.init config.iterations (fun _ ->
        Device.synchronize dev ;
        (* Ensure previous work is done *)
        let t0 = Unix.gettimeofday () in
        Execute.run_vectors
          ~device:dev
          ~ir
          ~args:[Vec vec_a; Vec vec_b; Vec vec_c; Int size]
          ~block
          ~grid
          () ;
        Device.synchronize dev ;
        (* Wait for kernel completion *)
        let t1 = Unix.gettimeofday () in
        (t1 -. t0) *. 1000.0)
  in

  (* Get results for verification - this happens outside timing *)
  let c = Vector.to_array vec_c in

  (* Verify *)
  let expected = Array.make size 0.0 in
  ocaml_vector_add a b expected size ;
  let verified = Common.arrays_equal ~epsilon:0.001 c expected in

  (* Calculate bandwidth (GB/s)
     We read 2 vectors and write 1 vector = 3 * size * 4 bytes *)
  let bytes = float_of_int (3 * size * 4) in
  let min_time_s = Common.min times /. 1000.0 in
  let bandwidth_gbps = bytes /. min_time_s /. 1e9 in

  Printf.printf
    "    Min time: %.3f ms, Bandwidth: %.3f GB/s\n"
    (Common.min times)
    bandwidth_gbps ;
  flush stdout ;

  (* Create result *)
  Output.
    {
      device_id = 0;
      (* Will be set properly by caller *)
      device_name = dev.Device.name;
      framework = dev.Device.framework;
      iterations = times;
      mean_ms = Common.mean times;
      stddev_ms = Common.stddev times;
      median_ms = Common.median times;
      min_ms = Common.min times;
      max_ms = Common.max times;
      throughput = Some bandwidth_gbps;
      verified = Some verified;
    }

(** Run full benchmark suite *)
let run config =
  Printf.printf "=== Vector Addition Benchmark ===\n" ;
  Printf.printf "Iterations: %d, Warmup: %d\n" config.iterations config.warmup ;
  Printf.printf
    "Sizes: %s\n"
    (String.concat ", " (List.map string_of_int config.sizes)) ;
  flush stdout ;

  (* Initialize backends *)
  Backend_loader.init () ;

  (* Initialize devices *)
  let devices = Device.init () in
  let devices =
    Array.to_list devices |> List.filter config.device_filter |> Array.of_list
  in

  if Array.length devices = 0 then (
    Printf.eprintf "No devices available\n" ;
    exit 1) ;

  Printf.printf "Devices: %d\n" (Array.length devices) ;
  Array.iter
    (fun dev ->
      Printf.printf "  - %s (%s)\n" dev.Device.name dev.Device.framework)
    devices ;

  (* Create output directory *)
  if not (Sys.file_exists config.output_dir) then
    Unix.mkdir config.output_dir 0o755 ;

  (* Collect system info once *)
  let system_info = System_info.collect devices in
  let git_commit = Common.get_git_commit () in

  (* Run benchmarks for each size *)
  List.iter
    (fun size ->
      Printf.printf "\n--- Size: %d ---\n" size ;

      (* Benchmark all devices *)
      let results =
        Array.to_list devices
        |> List.mapi (fun device_id dev ->
            try
              let r = benchmark_device dev size config in
              {r with Output.device_id}
            with e ->
              Printf.eprintf
                "Error on %s: %s\n"
                dev.Device.name
                (Printexc.to_string e) ;
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

      (* Create benchmark result *)
      let result =
        Output.
          {
            params =
              {
                name = "vector_add";
                size;
                block_size = 0;
                (* Not applicable for vector add *)
                iterations = config.iterations;
                warmup = config.warmup;
              };
            timestamp = Common.get_timestamp ();
            git_commit;
            system = system_info;
            results;
          }
      in

      (* Output to JSON *)
      let filename =
        Output.make_filename
          ~output_dir:config.output_dir
          ~benchmark_name:"vector_add"
          ~size
      in
      Output.write_json filename result ;
      Printf.printf "Written: %s\n" filename ;
      flush stdout)
    config.sizes

(** Command line interface *)
let () =
  let sizes = ref [] in
  let iterations = ref 20 in
  let warmup = ref 5 in
  let output_dir = ref "results" in
  let include_cpu = ref false in

  let usage =
    "Usage: bench_vector_add [options]\n\
     Benchmark vector addition across devices\n"
  in

  let specs =
    [
      ( "--sizes",
        Arg.String
          (fun s ->
            sizes := List.map int_of_string (String.split_on_char ',' s)),
        "Comma-separated list of vector sizes (default: \
         1000000,10000000,50000000,100000000)" );
      ("--iterations", Arg.Set_int iterations, "Number of iterations per size");
      ("--warmup", Arg.Set_int warmup, "Number of warmup iterations");
      ("--output", Arg.Set_string output_dir, "Output directory for results");
      ( "--include-cpu",
        Arg.Set include_cpu,
        "Include Native/Interpreter backends" );
    ]
  in

  Arg.parse specs (fun _ -> ()) usage ;

  let config =
    {
      sizes = (if !sizes = [] then default_config.sizes else !sizes);
      iterations = !iterations;
      warmup = !warmup;
      output_dir = !output_dir;
      device_filter =
        (if !include_cpu then fun _ -> true else default_config.device_filter);
    }
  in

  run config
