(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Generate Backend Source Code from Benchmarks

    This tool extracts the kernel IR from each benchmark and generates the
    corresponding backend code (CUDA, OpenCL, Vulkan GLSL, Metal).

    Output is saved as markdown files that can be included in documentation. *)

module Std = Sarek_stdlib.Std

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

(** Vector copy kernel *)
let vector_copy_kernel =
  [%kernel
    fun (a : float32 vector) (b : float32 vector) (n : int32) ->
      let open Std in
      let tid = global_thread_id in
      if tid < n then b.(tid) <- a.(tid)]
[@@warning "-33"]

(** STREAM Triad kernel *)
let stream_triad_kernel =
  [%kernel
    fun (a : float32 vector)
        (b : float32 vector)
        (c : float32 vector)
        (scalar : float32)
        (n : int32) ->
      let open Std in
      let tid = global_thread_id in
      if tid < n then a.(tid) <- b.(tid) +. (c.(tid) *. scalar)]
[@@warning "-33"]

(** Matrix multiplication kernel (naive) *)
let matrix_mul_kernel =
  [%kernel
    fun (a : float32 vector)
        (b : float32 vector)
        (c : float32 vector)
        (m : int32)
        (n : int32)
        (k : int32) ->
      let open Std in
      let tid = global_thread_id in
      let row = tid / n in
      let col = tid mod n in
      if row < m && col < n then begin
        let sum = mut 0.0 in
        for i = 0 to k - 1l do
          sum := sum +. (a.((row * k) + i) *. b.((i * n) + col))
        done ;
        c.((row * n) + col) <- sum
      end]
[@@warning "-33"]

(** Matrix multiplication kernel (tiled with shared memory) *)
let matrix_mul_tiled_kernel =
  [%kernel
    fun (a : float32 vector)
        (b : float32 vector)
        (c : float32 vector)
        (m : int32)
        (n : int32)
        (k : int32) ->
      let%shared (tile_a : float32) = 256l in
      let%shared (tile_b : float32) = 256l in
      let tx = thread_idx_x in
      let ty = thread_idx_y in
      let row = ty + (block_dim_y * block_idx_y) in
      let col = tx + (block_dim_x * block_idx_x) in
      let tile_size = 16l in
      let num_tiles = (k + tile_size - 1l) / tile_size in
      let sum = mut 0.0 in
      for t = 0 to num_tiles - 1l do
        let%superstep load_a =
          let a_col = (t * tile_size) + tx in
          if row < m && a_col < k then
            tile_a.((ty * tile_size) + tx) <- a.((row * k) + a_col)
          else tile_a.((ty * tile_size) + tx) <- 0.0
        in
        let%superstep load_b =
          let b_row = (t * tile_size) + ty in
          if b_row < k && col < n then
            tile_b.((ty * tile_size) + tx) <- b.((b_row * n) + col)
          else tile_b.((ty * tile_size) + tx) <- 0.0
        in
        let%superstep _compute =
          for i = 0 to tile_size - 1l do
            sum :=
              sum
              +. (tile_a.((ty * tile_size) + i) *. tile_b.((i * tile_size) + tx))
          done
        in
        ()
      done ;
      if row < m && col < n then c.((row * n) + col) <- sum]
[@@warning "-33"]

(** Reduction kernel *)
let reduction_kernel =
  [%kernel
    fun (input : float32 vector) (output : float32 vector) (n : int32) ->
      let%shared (sdata : float32) = 256l in
      let tid = thread_idx_x in
      let gid = thread_idx_x + (block_dim_x * block_idx_x) in
      let%superstep load =
        if gid < n then sdata.(tid) <- input.(gid) else sdata.(tid) <- 0.0
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
      let%superstep write =
        if tid = 0l then output.(block_idx_x) <- sdata.(0l)
      in
      ()]
[@@warning "-33"]

(** Max reduction kernel *)
let reduction_max_kernel =
  [%kernel
    fun (input : float32 vector) (output : float32 vector) (n : int32) ->
      let%shared (sdata : float32) = 256l in
      let tid = thread_idx_x in
      let gid = thread_idx_x + (block_dim_x * block_idx_x) in
      let%superstep load =
        if gid < n then sdata.(tid) <- input.(gid)
        else sdata.(tid) <- -1000000.0
      in
      let%superstep reduce128 =
        if tid < 128l then begin
          let a = sdata.(tid) in
          let b = sdata.(tid + 128l) in
          if b > a then sdata.(tid) <- b
        end
      in
      let%superstep reduce64 =
        if tid < 64l then begin
          let a = sdata.(tid) in
          let b = sdata.(tid + 64l) in
          if b > a then sdata.(tid) <- b
        end
      in
      let%superstep reduce32 =
        if tid < 32l then begin
          let a = sdata.(tid) in
          let b = sdata.(tid + 32l) in
          if b > a then sdata.(tid) <- b
        end
      in
      let%superstep reduce16 =
        if tid < 16l then begin
          let a = sdata.(tid) in
          let b = sdata.(tid + 16l) in
          if b > a then sdata.(tid) <- b
        end
      in
      let%superstep reduce8 =
        if tid < 8l then begin
          let a = sdata.(tid) in
          let b = sdata.(tid + 8l) in
          if b > a then sdata.(tid) <- b
        end
      in
      let%superstep reduce4 =
        if tid < 4l then begin
          let a = sdata.(tid) in
          let b = sdata.(tid + 4l) in
          if b > a then sdata.(tid) <- b
        end
      in
      let%superstep reduce2 =
        if tid < 2l then begin
          let a = sdata.(tid) in
          let b = sdata.(tid + 2l) in
          if b > a then sdata.(tid) <- b
        end
      in
      let%superstep reduce1 =
        if tid < 1l then begin
          let a = sdata.(tid) in
          let b = sdata.(tid + 1l) in
          if b > a then sdata.(tid) <- b
        end
      in
      let%superstep write =
        if tid = 0l then output.(block_idx_x) <- sdata.(0l)
      in
      ()]
[@@warning "-33"]

(** Transpose kernel (naive) *)
let transpose_naive_kernel =
  [%kernel
    fun (input : float32 vector)
        (output : float32 vector)
        (width : int32)
        (height : int32) ->
      let open Std in
      let tid = global_thread_id in
      let n = width * height in
      if tid < n then begin
        let col = tid mod width in
        let row = tid / width in
        let in_idx = (row * width) + col in
        let out_idx = (col * height) + row in
        output.(out_idx) <- input.(in_idx)
      end]
[@@warning "-33"]

(** Transpose kernel (tiled with shared memory) *)
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
      (* Load tile into shared memory *)
      let%superstep load =
        if read_col < width && read_row < height then
          let read_idx = (read_row * width) + read_col in
          let tile_idx = (ty * 17l) + tx in
          (* 17 = tile_size + 1 for padding *)
          tile.(tile_idx) <- input.(read_idx)
      in
      (* Global position for writing (output coordinates - transposed) *)
      let write_col = block_row + tx in
      (* Swap block indices *)
      let write_row = block_col + ty in
      (* Write transposed tile from shared memory *)
      let%superstep store =
        if write_col < height && write_row < width then
          let tile_idx = (tx * 17l) + ty in
          (* Read transposed from tile *)
          let write_idx = (write_row * height) + write_col in
          output.(write_idx) <- tile.(tile_idx)
      in
      ()]
[@@warning "-33"]

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

(** Generate backend code for a kernel *)
let generate_backend_code kernel_name kernel_func output_dir =
  Printf.printf "Generating backend code for: %s\n" kernel_name ;

  (* Extract IR *)
  let _, kirc = kernel_func in
  let ir =
    match kirc.Sarek.Kirc_types.body_ir with
    | Some ir -> ir
    | None -> failwith ("No IR for kernel: " ^ kernel_name)
  in

  (* Create output directory *)
  if not (Sys.file_exists output_dir) then Unix.mkdir output_dir 0o755 ;

  let md_file = Filename.concat output_dir (kernel_name ^ "_generated.md") in
  let oc = open_out md_file in

  (* Write markdown header *)
  Printf.fprintf oc "# Generated Backend Code: %s\n\n" kernel_name ;
  Printf.fprintf oc "This file is auto-generated. Do not edit manually.\n\n" ;
  Printf.fprintf
    oc
    "Generated on: %s\n\n"
    (let t = Unix.time () in
     let tm = Unix.localtime t in
     Printf.sprintf
       "%04d-%02d-%02d %02d:%02d:%02d"
       (tm.Unix.tm_year + 1900)
       (tm.Unix.tm_mon + 1)
       tm.Unix.tm_mday
       tm.Unix.tm_hour
       tm.Unix.tm_min
       tm.Unix.tm_sec) ;

  (* Generate CUDA code *)
  (try
     let cuda_code = Sarek_cuda.Sarek_ir_cuda.generate ir in
     Printf.fprintf oc "## CUDA C\n\n" ;
     Printf.fprintf oc "```cuda\n%s```\n\n" cuda_code ;
     Printf.printf "  ✓ CUDA C generated\n"
   with e ->
     Printf.fprintf oc "## CUDA C\n\n" ;
     Printf.fprintf
       oc
       "*Error generating CUDA code: %s*\n\n"
       (Printexc.to_string e) ;
     Printf.printf "  ✗ CUDA C failed: %s\n" (Printexc.to_string e)) ;

  (* Generate OpenCL code *)
  (try
     let opencl_code = Sarek_opencl.Sarek_ir_opencl.generate ir in
     Printf.fprintf oc "## OpenCL C\n\n" ;
     Printf.fprintf oc "```opencl\n%s```\n\n" opencl_code ;
     Printf.printf "  ✓ OpenCL C generated\n"
   with e ->
     Printf.fprintf oc "## OpenCL C\n\n" ;
     Printf.fprintf
       oc
       "*Error generating OpenCL code: %s*\n\n"
       (Printexc.to_string e) ;
     Printf.printf "  ✗ OpenCL C failed: %s\n" (Printexc.to_string e)) ;

  (* Generate Vulkan GLSL code *)
  (try
     let glsl_code = Sarek_vulkan.Sarek_ir_glsl.generate ir in
     Printf.fprintf oc "## Vulkan GLSL\n\n" ;
     Printf.fprintf oc "```glsl\n%s```\n\n" glsl_code ;
     Printf.printf "  ✓ Vulkan GLSL generated\n"
   with e ->
     Printf.fprintf oc "## Vulkan GLSL\n\n" ;
     Printf.fprintf
       oc
       "*Error generating Vulkan code: %s*\n\n"
       (Printexc.to_string e) ;
     Printf.printf "  ✗ Vulkan GLSL failed: %s\n" (Printexc.to_string e)) ;

  (* Generate Metal code *)
  (try
     let metal_code = Sarek_metal.Sarek_ir_metal.generate ir in
     Printf.fprintf oc "## Metal\n\n" ;
     Printf.fprintf oc "```metal\n%s```\n\n" metal_code ;
     Printf.printf "  ✓ Metal generated\n"
   with e ->
     Printf.fprintf oc "## Metal\n\n" ;
     Printf.fprintf
       oc
       "*Error generating Metal code: %s*\n\n"
       (Printexc.to_string e) ;
     Printf.printf "  ✗ Metal failed: %s\n" (Printexc.to_string e)) ;

  close_out oc ;
  Printf.printf "Generated: %s\n\n" md_file

let () =
  let output_dir = ref "benchmarks/descriptions/generated" in

  Arg.parse
    [
      ( "--output",
        Arg.Set_string output_dir,
        "Output directory (default: benchmarks/descriptions/generated)" );
    ]
    (fun _ -> ())
    "Generate backend source code for benchmark kernels" ;

  Printf.printf "=== Generating Backend Source Code ===\n\n" ;

  (* Generate for each benchmark kernel *)
  generate_backend_code "vector_add" vector_add_kernel !output_dir ;
  generate_backend_code "vector_copy" vector_copy_kernel !output_dir ;
  generate_backend_code "stream_triad" stream_triad_kernel !output_dir ;
  generate_backend_code "matrix_mul" matrix_mul_kernel !output_dir ;
  generate_backend_code "matrix_mul_tiled" matrix_mul_tiled_kernel !output_dir ;
  generate_backend_code "reduction" reduction_kernel !output_dir ;
  generate_backend_code "reduction_max" reduction_max_kernel !output_dir ;
  generate_backend_code "transpose_naive" transpose_naive_kernel !output_dir ;
  generate_backend_code "transpose_tiled" transpose_tiled_kernel !output_dir ;
  generate_backend_code "mandelbrot" mandelbrot_kernel !output_dir ;

  Printf.printf "=== Generation Complete ===\n"
