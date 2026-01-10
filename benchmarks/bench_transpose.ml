(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Matrix Transpose Benchmark

    Measures memory access pattern performance for matrix transpose operations.
    Transpose is a memory-bound operation that benefits from coalesced access.

    Demonstrates:
    - Memory coalescing patterns
    - Non-contiguous write patterns
    - Pure memory bandwidth (no compute)
    - Naive vs optimized access patterns *)

[@@@warning "-32-33-34"]

module Std = Sarek_stdlib.Std
module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Execute = Sarek.Execute

(* Benchmark infrastructure *)
open Benchmark_common
open Benchmark_backends

(** Benchmark configuration *)
type config = {
  sizes : int list;
  iterations : int;
  warmup : int;
  block_size : int;
  output_dir : string;
  device_filter : Device.t -> bool;
}

let default_config =
  {
    sizes = [256; 512; 1024; 2048];
    iterations = 20;
    warmup = 5;
    block_size = 256;
    output_dir = "results";
    device_filter =
      (fun dev ->
        (* By default, exclude slow CPU backends (Native, Interpreter) *)
        dev.Device.framework <> "Native"
        && dev.Device.framework <> "Interpreter");
  }

type float32 = float

(** Pure OCaml baseline *)
let cpu_transpose input output width height =
  for row = 0 to height - 1 do
    for col = 0 to width - 1 do
      let in_idx = (row * width) + col in
      let out_idx = (col * height) + row in
      output.(out_idx) <- input.(in_idx)
    done
  done

(* Type alias for vector - needed for [@sarek.module] outside of kernel context *)
type 'a vector = 'a array

(* Polymorphic transpose helper that gets monomorphized at the call site *)
let[@sarek.module] do_transpose (input : 'a vector) (output : 'a vector)
    (width : int) (height : int) (tid : int) : unit =
  let n = width * height in
  if tid < n then begin
    let col = tid mod width in
    let row = tid / width in
    let in_idx = (row * width) + col in
    let out_idx = (col * height) + row in
    output.(out_idx) <- input.(in_idx)
  end

(* Transpose kernel monomorphized at float32 *)
let transpose_kernel =
  [%kernel
    fun (input : float32 vector)
        (output : float32 vector)
        (width : int32)
        (height : int32) ->
      let open Std in
      do_transpose input output width height global_thread_id]

(** Run benchmark for one device and size *)
let benchmark_device dev size config =
  Printf.printf
    "  Device: %s (%s), Size: %dx%d\n"
    dev.Device.name
    dev.Device.framework
    size
    size ;
  flush stdout ;

  let m = size in
  let n = m * m in

  (* Prepare host data *)
  let input_data = Array.init n (fun i -> float_of_int i) in

  (* Get kernel IR *)
  let _, kirc = transpose_kernel in
  let ir =
    match kirc.Sarek.Kirc_types.body_ir with
    | Some ir -> ir
    | None -> failwith "No IR"
  in

  (* Initialize vectors for benchmark *)
  let init () =
    let input = Vector.create Vector.float32 n in
    let output = Vector.create Vector.float32 n in
    for i = 0 to n - 1 do
      Vector.set input i input_data.(i) ;
      Vector.set output i 0.0
    done ;
    (input, output)
  in

  (* Compute: run kernel *)
  let compute (input, output) =
    let block_size = 256 in
    let grid_size = (n + block_size - 1) / block_size in
    let block = Execute.dims1d block_size in
    let grid = Execute.dims1d grid_size in
    Execute.run_vectors
      ~device:dev
      ~ir
      ~args:
        [Execute.Vec input; Execute.Vec output; Execute.Int m; Execute.Int m]
      ~block
      ~grid
      ()
  in

  (* Verify: check correctness *)
  let verify (_input, output) =
    let result = Vector.to_array output in
    let expected = Array.make n 0.0 in
    cpu_transpose input_data expected m m ;
    let tolerance = 0.001 in
    let errors = ref 0 in
    for i = 0 to n - 1 do
      let diff = abs_float (result.(i) -. expected.(i)) in
      if diff > tolerance then begin
        if !errors < 5 then
          Printf.printf
            "  Mismatch at %d: expected %.2f, got %.2f\n"
            i
            expected.(i)
            result.(i) ;
        incr errors
      end
    done ;
    Printf.printf "  Verification: %s\n" (if !errors = 0 then "✓" else "✗") ;
    flush stdout ;
    !errors = 0
  in

  (* Run benchmark *)
  let times, verified =
    Common.benchmark_gpu
      ~dev
      ~warmup:config.warmup
      ~iterations:config.iterations
      ~init
      ~compute
      ~verify
  in

  (* Calculate statistics and bandwidth *)
  let mean_ms = Common.mean times in
  let stddev_ms = Common.stddev times in
  let bandwidth_gb_s =
    let bytes = float_of_int n *. 4.0 *. 2.0 in
    (* read + write, 4 bytes/float *)
    let time_s = mean_ms /. 1000.0 in
    bytes /. time_s /. 1e9
  in

  Printf.printf
    "  Mean: %.3f ms, Stddev: %.3f ms, Bandwidth: %.2f GB/s\n"
    mean_ms
    stddev_ms
    bandwidth_gb_s ;
  flush stdout ;

  Output.
    {
      device_id = 0;
      device_name = dev.Device.name;
      framework = dev.Device.framework;
      iterations = times;
      mean_ms;
      stddev_ms;
      median_ms = Common.median times;
      min_ms = Common.min times;
      max_ms = Common.max times;
      throughput = Some bandwidth_gb_s;
      verified = Some verified;
    }

(** Main entry point *)
let () =
  let config = ref default_config in

  (* Parse command line *)
  let specs =
    [
      ( "--sizes",
        Arg.String
          (fun s ->
            config :=
              {
                !config with
                sizes = List.map int_of_string (String.split_on_char ',' s);
              }),
        "Comma-separated list of matrix sizes (default: 256,512,1024,2048)" );
      ( "--iterations",
        Arg.Int (fun n -> config := {!config with iterations = n}),
        "Number of benchmark iterations (default: 20)" );
      ( "--warmup",
        Arg.Int (fun n -> config := {!config with warmup = n}),
        "Number of warmup iterations (default: 5)" );
      ( "--output",
        Arg.String (fun s -> config := {!config with output_dir = s}),
        "Output directory (default: results)" );
      ( "--all-backends",
        Arg.Unit
          (fun () -> config := {!config with device_filter = (fun _ -> true)}),
        "Include all backends (Native, Interpreter)" );
    ]
  in
  Arg.parse specs (fun _ -> ()) "Matrix transpose benchmark" ;

  Printf.printf "Matrix Transpose Benchmark\n" ;
  Printf.printf "==========================\n" ;
  Printf.printf
    "Sizes: %s\n"
    (String.concat ", " (List.map string_of_int !config.sizes)) ;
  Printf.printf "Iterations: %d, Warmup: %d\n" !config.iterations !config.warmup ;
  Printf.printf "Output: %s/\n" !config.output_dir ;
  flush stdout ;

  (* Initialize backends *)
  Backend_loader.init () ;

  (* Initialize devices *)
  let devices = Device.init () in
  if Array.length devices = 0 then begin
    Printf.eprintf "Error: No compute devices found\n" ;
    exit 1
  end ;

  Printf.printf "\nAvailable devices:\n" ;
  Array.iter
    (fun dev ->
      Printf.printf "  - %s (%s)\n" dev.Device.name dev.Device.framework)
    devices ;
  flush stdout ;

  (* Filter devices *)
  let devices =
    Array.to_list devices |> List.filter !config.device_filter |> Array.of_list
  in

  if Array.length devices = 0 then begin
    Printf.eprintf "Error: No devices match filter\n" ;
    exit 1
  end ;

  (* Create output directory *)
  if not (Sys.file_exists !config.output_dir) then
    Unix.mkdir !config.output_dir 0o755 ;

  (* Collect system info once *)
  let system_info = System_info.collect devices in
  let git_commit = Common.get_git_commit () in

  (* Run benchmarks for each size *)
  List.iter
    (fun size ->
      Printf.printf "\n--- Size: %dx%d ---\n" size size ;

      (* Benchmark all devices *)
      let results =
        Array.to_list devices
        |> List.mapi (fun device_id dev ->
            try
              let r = benchmark_device dev size !config in
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
                name = "transpose_naive";
                size;
                block_size = !config.block_size;
                iterations = !config.iterations;
                warmup = !config.warmup;
              };
            timestamp = Common.get_timestamp ();
            git_commit;
            system = system_info;
            results;
          }
      in

      (* Write JSON output *)
      let filename =
        Output.make_filename
          ~output_dir:!config.output_dir
          ~benchmark_name:"transpose_naive"
          ~size
      in
      Output.write_json filename result ;
      Printf.printf "\nResults written to: %s\n" filename ;
      flush stdout)
    !config.sizes
