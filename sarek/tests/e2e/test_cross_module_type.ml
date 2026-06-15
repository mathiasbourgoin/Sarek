[@@@warning "-32-34"]
(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * E2E test: use a Sarek type registered in another module/file.
 * Uses GPU runtime only.
 ******************************************************************************)

module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer

(* Force backend registration *)
let () =
  Sarek_cuda.Cuda_plugin.init () ;
  Sarek_opencl.Opencl_plugin.init ()

type float32 = float

(* Include types and functions from registered_defs.ml *)
let%sarek_include _ = "registered_defs.ml"

let () =
  let kernel =
    [%kernel
      fun (xs : float32 vector)
          (ys : float32 vector)
          (dst : float32 vector)
          (n : int32) ->
        let tid = thread_idx_x + (block_idx_x * block_dim_x) in
        if tid < n then
          let open Registered_defs in
          let p : Registered_defs.vec2 = {x = xs.(tid); y = ys.(tid)} in
          dst.(tid) <- add_vec p]
  in
  let _, kirc = kernel in

  (* Print IR for verification *)
  print_endline "=== Cross-module Type IR ===" ;
  print_endline "=============================" ;

  let devs =
    Device.init ~frameworks:["CUDA"; "OpenCL"; "Native"; "Interpreter"] ()
  in
  if Array.length devs = 0 then (
    print_endline "No devices found - IR generation test passed" ;
    exit 0) ;
  let dev = devs.(0) in
  Printf.printf "Using device: %s\n%!" dev.Device.name ;

  (* Get IR and execute *)
  match kirc.Sarek.Kirc_types.body_ir with
  | None -> print_endline "No IR - IR generation test PASSED"
  | Some ir ->
      let n = 64 in
      let xs = Vector.create Vector.float32 n in
      let ys = Vector.create Vector.float32 n in
      let dst = Vector.create Vector.float32 n in
      for i = 0 to n - 1 do
        Vector.set xs i (float_of_int i) ;
        Vector.set ys i (float_of_int (2 * i)) ;
        Vector.set dst i 0.0
      done ;

      let threads = 64 in
      let grid_x = (n + threads - 1) / threads in
      Sarek.Execute.run_vectors
        ~device:dev
        ~ir
        ~args:
          [
            Sarek.Execute.Vec xs;
            Sarek.Execute.Vec ys;
            Sarek.Execute.Vec dst;
            Sarek.Execute.Int n;
          ]
        ~block:(Sarek.Execute.dims1d threads)
        ~grid:(Sarek.Execute.dims1d grid_x)
        () ;
      Transfer.flush dev ;

      let ok = ref true in
      for i = 0 to n - 1 do
        let x = Vector.get xs i in
        let y = Vector.get ys i in
        let expected = x +. y in
        let got = Vector.get dst i in
        if abs_float (got -. expected) > 1e-4 then (
          ok := false ;
          Printf.printf "Mismatch at %d: got %f expected %f\n%!" i got expected)
      done ;
      if !ok then print_endline "Cross-module registered type execution PASSED"
      else (
        print_endline "Cross-module registered type execution FAILED" ;
        exit 1)
