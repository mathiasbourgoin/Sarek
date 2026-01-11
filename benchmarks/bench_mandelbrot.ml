(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Mandelbrot Set Benchmark

    Benchmarks fractal generation using iterative complex arithmetic. Generates
    beautiful visualizations while testing arithmetic intensity. *)

open Sarek
open Benchmark_common
open Benchmark_backends
module Std = Sarek_stdlib.Std
module Device = Spoc_core.Device
module Vector = Spoc_core.Vector

(** Configuration *)
type config = {
  sizes : int list;
  max_iterations : int;
  iterations : int;
  warmup : int;
  output_dir : string;
  generate_images : bool;
  device_filter : Device.t -> bool;
}

let default_config =
  {
    sizes = [512; 1024; 2048; 4096];
    max_iterations = 256;
    iterations = 20;
    warmup = 5;
    output_dir = "results";
    generate_images = true;
    device_filter =
      (fun dev ->
        (* Exclude slow CPU backends *)
        dev.Device.framework <> "Native"
        && dev.Device.framework <> "Interpreter");
  }

(** Pure OCaml baseline *)
let ocaml_mandelbrot output width height max_iter =
  for py = 0 to height - 1 do
    for px = 0 to width - 1 do
      let x0 = (4.0 *. float_of_int px /. float_of_int width) -. 2.5 in
      let y0 = (3.0 *. float_of_int py /. float_of_int height) -. 1.5 in
      let x = ref 0.0 in
      let y = ref 0.0 in
      let iter = ref 0 in
      while (!x *. !x) +. (!y *. !y) <= 4.0 && !iter < max_iter do
        let xtemp = (!x *. !x) -. (!y *. !y) +. x0 in
        y := (2.0 *. !x *. !y) +. y0 ;
        x := xtemp ;
        incr iter
      done ;
      output.((py * width) + px) <- Int32.of_int !iter
    done
  done

(** Mandelbrot kernel *)
let mandelbrot_kernel =
  [%kernel
    fun (output : int32 vector)
        (width : int32)
        (height : int32)
        (max_iter : int32) ->
      let open Std in
      let px = global_idx_x in
      let py = global_idx_y in
      if px < width && py < height then begin
        let x0 = (4.0 *. (float px /. float width)) -. 2.5 in
        let y0 = (3.0 *. (float py /. float height)) -. 1.5 in
        let x = mut 0.0 in
        let y = mut 0.0 in
        let iter = mut 0l in
        while (x *. x) +. (y *. y) <= 4.0 && iter < max_iter do
          let xtemp = (x *. x) -. (y *. y) +. x0 in
          y := (2.0 *. x *. y) +. y0 ;
          x := xtemp ;
          iter := iter + 1l
        done ;
        output.((py * width) + px) <- iter
      end]
[@@warning "-33"]

(** Generate PPM image from iteration data *)
let save_ppm filename data width height max_iter =
  let oc = open_out filename in
  Printf.fprintf oc "P3\n%d %d\n255\n" width height ;
  for py = 0 to height - 1 do
    for px = 0 to width - 1 do
      let iter = Int32.to_int data.((py * width) + px) in
      (* Color mapping: smooth gradient *)
      let t = float_of_int iter /. float_of_int max_iter in
      let r = int_of_float (9.0 *. (1.0 -. t) *. t *. t *. t *. 255.0) in
      let g =
        int_of_float (15.0 *. (1.0 -. t) *. (1.0 -. t) *. t *. t *. 255.0)
      in
      let b =
        int_of_float
          (8.5 *. (1.0 -. t) *. (1.0 -. t) *. (1.0 -. t) *. t *. 255.0)
      in
      Printf.fprintf oc "%d %d %d " r g b
    done ;
    Printf.fprintf oc "\n"
  done ;
  close_out oc ;
  Printf.printf "Saved image: %s\n" filename

(** Run benchmark for one device and size *)
let benchmark_device dev size config =
  Printf.printf
    "  Device: %s (%s), Size: %dx%d\n"
    dev.Device.name
    dev.Device.framework
    size
    size ;
  flush stdout ;

  let width = size in
  let height = size in
  let n = width * height in

  (* Prepare expected output for verification *)
  let expected = Array.make n 0l in
  ocaml_mandelbrot expected width height config.max_iterations ;

  (* Get kernel IR *)
  let _, kirc = mandelbrot_kernel in
  let ir =
    match kirc.Sarek.Kirc_types.body_ir with
    | Some ir -> ir
    | None -> failwith "No IR"
  in

  (* Setup execution parameters *)
  let block_size = 16 in
  let blocks_x = (width + block_size - 1) / block_size in
  let blocks_y = (height + block_size - 1) / block_size in
  let block = Execute.dims2d block_size block_size in
  let grid = Execute.dims2d blocks_x blocks_y in

  (* Benchmark using the new API *)
  let init () =
    let vec_output = Vector.create Vector.int32 n in
    for i = 0 to n - 1 do
      Vector.set vec_output i 0l
    done ;
    vec_output
  in

  let compute vec_output =
    Execute.run_vectors
      ~device:dev
      ~ir
      ~args:[Vec vec_output; Int width; Int height; Int config.max_iterations]
      ~block
      ~grid
      ()
  in

  let verify vec_output =
    let result_array = Vector.to_array vec_output in
    (* Fuzzy verification: allow some divergence due to floating point *)
    let errors = ref 0 in
    for i = 0 to min 1000 (n - 1) do
      let diff =
        abs (Int32.to_int result_array.(i) - Int32.to_int expected.(i))
      in
      if diff > 10 then incr errors
    done ;
    let verified = !errors = 0 in

    (* Generate image if requested *)
    (if config.generate_images then
       let safe_name =
         String.map (fun c -> if c = ' ' then '_' else c) dev.Device.name
       in
       let img_filename =
         Printf.sprintf
           "%s/mandelbrot_%s_%s_%dx%d.ppm"
           config.output_dir
           safe_name
           dev.Device.framework
           width
           height
       in
       save_ppm img_filename result_array width height config.max_iterations) ;

    verified
  in

  let times, verified =
    Common.benchmark_gpu
      ~dev
      ~warmup:config.warmup
      ~iterations:config.iterations
      ~init
      ~compute
      ~verify
  in

  (* Compute throughput (pixels per second) *)
  let pixels = float_of_int (width * height) in
  let mean_s = Common.mean times /. 1000.0 in
  let mpixels_per_sec = pixels /. mean_s /. 1e6 in

  (* Create result *)
  Output.
    {
      device_id = 0;
      device_name = dev.Device.name;
      framework = dev.Device.framework;
      iterations = times;
      mean_ms = Common.mean times;
      stddev_ms = Common.stddev times;
      median_ms = Common.median times;
      min_ms = Common.min times;
      max_ms = Common.max times;
      throughput = Some mpixels_per_sec;
      verified = Some verified;
    }

(** Run full benchmark suite *)
let run config =
  Printf.printf "=== Mandelbrot Set Benchmark ===\n" ;
  Printf.printf
    "Iterations: %d, Warmup: %d, Max Iter: %d\n"
    config.iterations
    config.warmup
    config.max_iterations ;
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
      Printf.printf "\n--- Size: %dx%d ---\n" size size ;

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
                name = "mandelbrot";
                size = size * size;
                block_size = 16;
                iterations = config.iterations;
                warmup = config.warmup;
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
          ~output_dir:config.output_dir
          ~benchmark_name:"mandelbrot"
          ~size:(size * size)
      in
      Output.write_json filename result ;
      Printf.printf "Written: %s\n" filename ;

      (* Also append to CSV *)
      let csv_filename = Filename.concat config.output_dir "mandelbrot.csv" in
      Output.append_csv csv_filename result)
    config.sizes ;

  Printf.printf "\n=== Benchmark Complete ===\n"

(** Main entry point *)
let () =
  let config = ref default_config in

  let speclist =
    [
      ( "--sizes",
        Arg.String
          (fun s ->
            config :=
              {
                !config with
                sizes = String.split_on_char ',' s |> List.map int_of_string;
              }),
        "Comma-separated list of sizes (e.g., 512,1024,2048)" );
      ( "--max-iter",
        Arg.Int (fun n -> config := {!config with max_iterations = n}),
        "Maximum iterations for Mandelbrot calculation" );
      ( "--iterations",
        Arg.Int (fun n -> config := {!config with iterations = n}),
        "Number of benchmark iterations" );
      ( "--warmup",
        Arg.Int (fun n -> config := {!config with warmup = n}),
        "Number of warmup iterations" );
      ( "--output",
        Arg.String (fun s -> config := {!config with output_dir = s}),
        "Output directory for results" );
      ( "--no-images",
        Arg.Unit (fun () -> config := {!config with generate_images = false}),
        "Don't generate PPM images" );
      ( "--all-devices",
        Arg.Unit
          (fun () -> config := {!config with device_filter = (fun _ -> true)}),
        "Benchmark all devices including slow CPU backends" );
    ]
  in

  Arg.parse speclist (fun _ -> ()) "Mandelbrot set benchmark" ;
  run !config
