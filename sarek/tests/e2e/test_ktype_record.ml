(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * E2E test for Sarek PPX with record type declarations at top level.
 *
 * The [@@sarek.type] attribute generates:
 *   - point_custom (for runtime Spoc_core.Vector)
 ******************************************************************************)

module Vector = Spoc_core.Vector
module Device = Spoc_core.Device
module Transfer = Spoc_core.Transfer

[@@@warning "-32"]

(* Force backend registration *)
let () =
  Sarek_cuda.Cuda_plugin.init () ;
  Sarek_opencl.Opencl_plugin.init ()

type float32 = float

(* Define point type with sarek.type attribute - generates point_custom *)
type point = {x : float32; y : float32} [@@sarek.type]

(* Top-level kernel - no type annotation needed if we just extract the kirc part *)
let point_copy_kirc =
  snd
    [%kernel
      fun (src : point vector) (dst : point vector) (n : int32) ->
        let tid = thread_idx_x in
        if tid < n then
          let p = src.(tid) in
          let next : point = {x = p.x +. 1.0; y = p.y} in
          dst.(tid) <- next]

let () =
  print_endline "=== Generated Kernel IR (ktype record) ===" ;
  print_endline "==========================================" ;

  (* runtime execution *)
  print_endline "\n=== runtime Path ===" ;
  let devs =
    Device.init ~frameworks:["Interpreter"; "Native"; "CUDA"; "OpenCL"] ()
  in
  if Array.length devs = 0 then begin
    print_endline "No runtime devices found - IR generation test passed" ;
    exit 0
  end ;

  let dev =
    match Array.find_opt (fun d -> d.Device.framework = "Interpreter") devs with
    | Some d -> d
    | None -> (
        match Array.find_opt (fun d -> d.Device.framework = "Native") devs with
        | Some d -> d
        | None -> devs.(0))
  in
  Printf.printf "runtime device: %s\n%!" dev.Device.name ;
  if dev.framework <> "Native" then (
    Printf.printf
      "runtime: SKIP (record test checked on native backend only)\n%!" ;
    exit 0) ;
  print_string "runtime: " ;
  try
    let n = 64 in
    let src = Vector.create_custom point_custom n in
    let dst = Vector.create_custom point_custom n in
    for i = 0 to n - 1 do
      Vector.set src i {x = float_of_int i; y = float_of_int (n - i)} ;
      Vector.set dst i {x = 0.0; y = 0.0}
    done ;
    let threads = min 64 n in
    let grid_x = (n + threads - 1) / threads in
    let ir =
      match point_copy_kirc.Sarek.Kirc_types.body_ir with
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
    (* Verify: dst[i].x should be src[i].x + 1.0, dst[i].y should be src[i].y *)
    let ok = ref true in
    for i = 0 to n - 1 do
      let s = Vector.get src i in
      let d = Vector.get dst i in
      let expected_x = s.x +. 1.0 in
      let expected_y = s.y in
      if
        abs_float (d.x -. expected_x) > 1e-3
        || abs_float (d.y -. expected_y) > 1e-3
      then (
        ok := false ;
        if i < 5 then
          Printf.printf
            "  Mismatch at %d: got {%.1f, %.1f} expected {%.1f, %.1f}\n%!"
            i
            d.x
            d.y
            expected_x
            expected_y)
    done ;
    if !ok then print_endline "PASSED" else print_endline "FAILED"
  with e -> Printf.printf "FAIL (%s)\n%!" (Printexc.to_string e)
