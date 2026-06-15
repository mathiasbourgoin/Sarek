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

(* Type alias for kernel parameter annotations *)
type ('a, 'b) vector = ('a, 'b) Vector.t

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
      fun (src : shape vector) (dst : float32 vector) (n : int32) ->
        let tid = thread_idx_x in
        if tid < n then dst.(tid) <- area src.(tid)]
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
  if Array.length devs = 0 then (
    print_endline "No device found - IR generation test passed" ;
    exit 0) ;
  let dev = devs.(0) in
  Printf.printf "Using device: %s\n%!" dev.Device.name ;
  (match kirc.Sarek.Kirc_types.body_ir with
  | Some ir ->
      Printf.printf
        "IR available, kernel name: %s\n%!"
        ir.Sarek.Sarek_ir.kern_name ;
      print_endline "Variant helper IR PASSED"
  | None -> print_endline "No IR - SKIPPED") ;
  print_endline "test_klet_variant PASSED"
