(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Matrix Transpose Benchmark (Tiled with Shared Memory)

    Measures performance of optimized matrix transpose using shared memory
    tiling to achieve coalesced memory access patterns.

    Demonstrates:
    - Shared memory usage for performance optimization
    - Coalesced memory reads and writes
    - Thread block synchronization with barriers
    - 4-6x speedup over naive transpose

    Algorithm:
    - Uses 16x16 tiles loaded into shared memory
    - Reads from input in coalesced pattern (row-major)
    - Transposes within shared memory
    - Writes to output in coalesced pattern (row-major after transpose)
    - Adds padding to shared memory to avoid bank conflicts *)

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
  tile_size : int;
  output_dir : string;
  device_filter : Device.t -> bool;
}

let default_config =
  {
    sizes = [256; 512; 1024; 2048; 4096; 8192];
    iterations = 20;
    warmup = 5;
    tile_size = 16;
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

(** Tiled transpose kernel with shared memory

    Uses 16x16 tiles to achieve coalesced access: 1. Load tile from input
    (coalesced reads, row-major) 2. Synchronize threads 3. Write tile to output
    transposed (coalesced writes, row-major)

    Note: We add +1 padding to shared memory to avoid bank conflicts *)
let transpose_tiled_kernel =
  [%kernel
    fun (input : float32 vector)
        (output : float32 vector)
        (width : int32)
        (height : int32) ->
      (* Shared memory tile with padding to avoid bank conflicts
         Size: 16x17 = 272 elements (17 to avoid bank conflicts) *)
      let%shared (tile : float32) = 272l in
      let tile_size = 16l in
      let tx = thread_idx_x in
      let ty = thread_idx_y in
      (* Calculate global position for this thread block *)
      let block_col = block_idx_x * tile_size in
      let block_row = block_idx_y * tile_size in
      (* Global position for reading (input coordinates) *)
      let read_col = block_col + tx in
      let read_row = block_row + ty in
      (* Global position for writing (output transposed coordinates) *)
      let write_col = block_row + tx in
      let write_row = block_col + ty in
      (* Load tile from input in coalesced manner (row-major) *)
      let%superstep load =
        if read_row < height && read_col < width then
          (* Note: +1 stride to avoid bank conflicts *)
          tile.((ty * (tile_size + 1l)) + tx) <-
            input.((read_row * width) + read_col)
      in
      (* Write transposed tile to output in coalesced manner *)
      let%superstep store =
        if write_row < height && write_col < width then
          (* Read from transposed position in tile *)
          output.((write_row * width) + write_col) <-
            tile.((tx * (tile_size + 1l)) + ty)
      in
      ()]

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
  let _, kirc = transpose_tiled_kernel in
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

  (* Compute: run kernel with 2D thread blocks *)
  let compute (input, output) =
    let tile_size = config.tile_size in
    let grid_x = (m + tile_size - 1) / tile_size in
    let grid_y = (m + tile_size - 1) / tile_size in
    (* 2D thread blocks: tile_size x tile_size threads *)
    let block = Execute.dims2d tile_size tile_size in
    let grid = Execute.dims2d grid_x grid_y in
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
    (* Dynamic tolerance: allow for float32 precision limits
       For values beyond 2^24, float32 has ULP > 1
       At 8192x8192 (67M elements), ULP ~= 4 *)
    let tolerance = if n > 16_777_216 then 10.0 else 0.001 in
    let errors = ref 0 in
    for i = 0 to n - 1 do
      let diff = abs_float (result.(i) -. expected.(i)) in
      if diff > tolerance then begin
        if !errors < 5 then
          Printf.printf
            "  Mismatch at %d: expected %.2f, got %.2f (diff: %.2f)\n"
            i
            expected.(i)
            result.(i)
            diff ;
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
      ( "--tile-size",
        Arg.Int (fun n -> config := {!config with tile_size = n}),
        "Tile size for shared memory (default: 16)" );
      ( "--output",
        Arg.String (fun s -> config := {!config with output_dir = s}),
        "Output directory (default: results)" );
      ( "--all-backends",
        Arg.Unit
          (fun () -> config := {!config with device_filter = (fun _ -> true)}),
        "Include all backends (Native, Interpreter)" );
    ]
  in
  Arg.parse specs (fun _ -> ()) "Tiled matrix transpose benchmark" ;

  Printf.printf "Matrix Transpose Benchmark (Tiled with Shared Memory)\n" ;
  Printf.printf "=====================================================\n" ;
  Printf.printf
    "Sizes: %s\n"
    (String.concat ", " (List.map string_of_int !config.sizes)) ;
  Printf.printf
    "Iterations: %d, Warmup: %d, Tile: %dx%d\n"
    !config.iterations
    !config.warmup
    !config.tile_size
    !config.tile_size ;
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
                name = "transpose_tiled";
                size;
                block_size = !config.tile_size * !config.tile_size;
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
          ~benchmark_name:"transpose_tiled"
          ~size
      in
      Output.write_json filename result ;
      Printf.printf "\nResults written to: %s\n" filename ;
      flush stdout)
    !config.sizes
