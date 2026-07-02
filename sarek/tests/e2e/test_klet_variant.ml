(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * E2E test for Sarek PPX with variant type and helper function.
 * Uses GPU runtime only.
 ******************************************************************************)

(* runtime module aliases *)
module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer

(* Force backend registration *)
let () =
  Sarek_cuda.Cuda_plugin.init () ;
  Sarek_opencl.Opencl_plugin.init ()

let () =
  let dispatch =
    [%kernel
      let module Types = struct
        type shape = Circle of float32 | Square of float32
      end in
      let area (s : shape) : float32 =
        match s with Circle r -> 3.14 *. r *. r | Square x -> x *. x
      in
      fun (src : float32 vector) (dst : float32 vector) (n : int32) ->
        let tid = thread_idx_x in
        if tid < n then
          let s =
            if src.(tid) > 0.0 then Circle src.(tid)
            else Square (0.0 -. src.(tid))
          in
          dst.(tid) <- area s]
  in

  (* Get IR *)
  let _, kirc = dispatch in
  print_endline "=== Variant helper IR ===" ;
  (match kirc.Sarek.Kirc_types.body_ir with
  | Some ir -> Sarek_ir_pp.print_kernel ir
  | None -> print_endline "(No IR available)") ;
  print_endline "=========================" ;

  (* Run with GPU runtime *)
  let devs =
    Device.init ~frameworks:["CUDA"; "OpenCL"; "Native"; "Interpreter"] ()
  in
  if Array.length devs = 0 then begin
    print_endline "No device found - IR generation test passed" ;
    exit 0
  end ;
  let dev = devs.(0) in
  Printf.printf "Using device: %s\n%!" dev.Device.name ;
  match kirc.Sarek.Kirc_types.body_ir with
  | None ->
      print_endline "No IR - SKIPPED" ;
      exit 0
  | Some ir ->
      let n = 64 in
      let src = Vector.create Vector.float32 n in
      let dst = Vector.create Vector.float32 n in
      for i = 0 to n - 1 do
        (* Alternate between "circle" (positive radius) and "square"
           (negative side, sign is the dispatch tag). *)
        Vector.set
          src
          i
          (if i mod 2 = 0 then float_of_int (i + 1) else -.float_of_int (i + 1)) ;
        Vector.set dst i 0.0
      done ;
      let threads = min 64 n in
      let grid_x = (n + threads - 1) / threads in
      Sarek.Execute.run_vectors
        ~device:dev
        ~block:(Sarek.Execute.dims1d threads)
        ~grid:(Sarek.Execute.dims1d grid_x)
        ~ir
        ~args:
          [
            Sarek.Execute.Vec src;
            Sarek.Execute.Vec dst;
            Sarek.Execute.Int32 (Int32.of_int n);
          ]
        () ;
      Transfer.flush dev ;
      let ok = ref true in
      for i = 0 to n - 1 do
        let x = Vector.get src i in
        let expected = if x > 0.0 then 3.14 *. x *. x else x *. x in
        let got = Vector.get dst i in
        if abs_float (got -. expected) > 1e-2 then begin
          ok := false ;
          if i < 5 then
            Printf.printf
              "  Mismatch at %d: got %f expected %f\n%!"
              i
              got
              expected
        end
      done ;
      if !ok then print_endline "test_klet_variant PASSED"
      else begin
        print_endline "test_klet_variant FAILED: area mismatch" ;
        exit 1
      end
