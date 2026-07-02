(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * E2E test for Sarek PPX with helper function (klet-style) in the payload.
 * Runs on Native/Interpreter when available (falls back to whatever device
 * Device.init enumerates first otherwise); plain float32 vectors only, no
 * custom types, so GPU backends are not documented as unreliable here.
 ******************************************************************************)

(* runtime module aliases *)
module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer

(* Force backend registration. Also register the always-available
   Native/Interpreter plugins - the previous version of this test only
   called Sarek_cuda.Cuda_plugin.init/Sarek_opencl.Opencl_plugin.init,
   which never registered Native/Interpreter at all, so the
   Interpreter/Native device preference below could never actually find
   them (see briefs/make-tests-actually-run-impl-notes.md). *)
let () =
  Sarek_cuda.Cuda_plugin.init () ;
  Sarek_opencl.Opencl_plugin.init () ;
  Sarek_native.Native_plugin.init () ;
  Sarek_interpreter.Interpreter_plugin.init ()

let () =
  let scale_add =
    [%kernel
      let add_scale (x : float32) (y : float32) : float32 = x +. (2.0 *. y) in
      fun (src : float32 vector) (dst : float32 vector) (n : int32) ->
        let tid = thread_idx_x + (block_dim_x * block_idx_x) in
        if tid < n then dst.(tid) <- add_scale src.(tid) 3.0]
  in
  let _native, kirc = scale_add in
  print_endline "=== klet-style helper IR ===" ;
  print_endline "=============================" ;
  let devs =
    Device.init ~frameworks:["CUDA"; "OpenCL"; "Native"; "Interpreter"] ()
  in
  if Array.length devs = 0 then (
    print_endline "No device found - IR generation test passed" ;
    exit 0) ;
  let dev =
    match Array.find_opt (fun d -> d.Device.framework = "Interpreter") devs with
    | Some d -> d
    | None -> (
        match Array.find_opt (fun d -> d.Device.framework = "Native") devs with
        | Some d -> d
        | None -> devs.(0))
  in
  Printf.printf "Using device: %s\n%!" dev.Device.name ;

  let n = 64 in
  let src = Vector.create Vector.float32 n in
  let dst = Vector.create Vector.float32 n in
  for i = 0 to n - 1 do
    Vector.set src i (float_of_int i) ;
    Vector.set dst i 0.0
  done ;

  let threads = min 64 n in
  let grid_x = (n + threads - 1) / threads in
  (try
     let ir =
       match kirc.Sarek.Kirc_types.body_ir with
       | Some ir -> ir
       | None -> failwith "Kernel has no IR"
     in
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
       (* add_scale x 3.0 = x + 2.0 * 3.0 *)
       let expected = x +. (2.0 *. 3.0) in
       let got = Vector.get dst i in
       if abs_float (got -. expected) > 1e-3 then begin
         ok := false ;
         if i < 5 then
           Printf.printf
             "  Mismatch at %d: got %f expected %f\n%!"
             i
             got
             expected
       end
     done ;
     if !ok then print_endline "Helper function codegen PASSED"
     else begin
       print_endline "Helper function codegen FAILED: value mismatch" ;
       exit 1
     end
   with e ->
     Printf.printf "Codegen failed: %s\n%!" (Printexc.to_string e) ;
     exit 1) ;
  ()
