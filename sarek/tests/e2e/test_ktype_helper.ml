(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * E2E test: ktype record + helper function executed on device.
 * Uses GPU runtime only.
 ******************************************************************************)

(* runtime module aliases *)
module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer

(* Type alias for kernel parameter annotations *)
type ('a, 'b) vector = ('a, 'b) Vector.t

(* Force backend registration *)
let () =
  Sarek_cuda.Cuda_plugin.init () ;
  Sarek_opencl.Opencl_plugin.init ()

let () =
  let length_kernel =
    [%kernel
      let module Types = struct
        type point = {x : float32; y : float32}
      end in
      let make_point (x : float32) (y : float32) : point = {x; y} in
      fun (xv : float32 vector)
          (yv : float32 vector)
          (dst : float32 vector)
          (n : int32)
        ->
        let tid = thread_idx_x + (block_idx_x * block_dim_x) in
        if tid < n then
          let p = make_point xv.(tid) yv.(tid) in
          dst.(tid) <- sqrt ((p.x *. p.x) +. (p.y *. p.y))]
  in

  let _, kirc = length_kernel in
  print_endline "=== ktype helper IR ===" ;
  (match kirc.Sarek.Kirc_types.body_ir with
  | Some ir -> Sarek_ir_pp.print_kernel ir
  | None -> print_endline "(No IR available)") ;
  print_endline "=======================" ;

  (* Run with GPU runtime *)
  let devs =
    Device.init ~frameworks:["CUDA"; "OpenCL"; "Native"; "Interpreter"] ()
  in
  if Array.length devs = 0 then (
    print_endline "No device found - IR generation test passed" ;
    exit 0) ;
  let dev = devs.(0) in
  Printf.printf "Using device: %s\n%!" dev.Device.name ;

  let n = 256 in

  (match kirc.Sarek.Kirc_types.body_ir with
  | None -> print_endline "No IR - SKIPPED"
  | Some ir ->
      let xv = Vector.create Vector.float32 n in
      let yv = Vector.create Vector.float32 n in
      let dst = Vector.create Vector.float32 n in
      for i = 0 to n - 1 do
        Vector.set xv i (float_of_int i) ;
        Vector.set yv i (float_of_int (n - i)) ;
        Vector.set dst i 0.0
      done ;

      let threads = 128 in
      let grid_x = (n + threads - 1) / threads in
      let block = Sarek.Execute.dims1d threads in
      let grid = Sarek.Execute.dims1d grid_x in

      Sarek.Execute.run_vectors
        ~device:dev
        ~ir
        ~args:
          [
            Sarek.Execute.Vec xv;
            Sarek.Execute.Vec yv;
            Sarek.Execute.Vec dst;
            Sarek.Execute.Int n;
          ]
        ~block
        ~grid
        () ;
      Transfer.flush dev ;

      (* Verify results *)
      let ok = ref true in
      for i = 0 to n - 1 do
        let x = Vector.get xv i in
        let y = Vector.get yv i in
        let expected = sqrt ((x *. x) +. (y *. y)) in
        let got = Vector.get dst i in
        if abs_float (got -. expected) > 1e-3 then (
          ok := false ;
          Printf.printf "Mismatch at %d: got %f expected %f\n%!" i got expected)
      done ;
      if !ok then print_endline "Execution check PASSED"
      else (
        print_endline "Execution check FAILED" ;
        exit 1)) ;
  print_endline "test_ktype_helper PASSED"
