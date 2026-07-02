(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Test kernel using external Geometry_lib types via convention *)
open Sarek_geometry
module Vector = Spoc_core.Vector
module Device = Spoc_core.Device
module Transfer = Spoc_core.Transfer

(* Force backend registration *)
let () =
  Sarek_native.Native_plugin.init () ;
  Sarek_interpreter.Interpreter_plugin.init () ;
  Sarek_cuda.Cuda_plugin.init () ;
  Sarek_opencl.Opencl_plugin.init ()

(* This kernel takes Geometry_lib.point vectors and computes distance to origin *)
let () =
  (* Version that should type correctly:
     - Uses Geometry_lib.point from external library
     - Accesses fields x, y
     - Computes sqrt(x^2 + y^2) *)
  let distance_to_origin_kernel =
    [%kernel
      fun (points : Geometry_lib.point vector)
          (distances : float32 vector)
          (n : int32) ->
        let tid = thread_idx_x + (block_dim_x * block_idx_x) in
        if tid < n then
          let p = points.(tid) in
          let x = p.x in
          let y = p.y in
          distances.(tid) <- sqrt ((x *. x) +. (y *. y))]
  in

  let _native, kirc = distance_to_origin_kernel in
  print_endline "=== Distance to origin kernel IR ===" ;
  print_endline "=====================================" ;

  let devs =
    Device.init ~frameworks:["Interpreter"; "Native"; "CUDA"; "OpenCL"] ()
  in
  if Array.length devs = 0 then begin
    print_endline "No device found - IR generation test passed" ;
    exit 0
  end ;
  let dev =
    match Array.find_opt (fun d -> d.Device.framework = "Native") devs with
    | Some d -> d
    | None -> devs.(0)
  in
  Printf.printf "Using device: %s\n%!" dev.Device.name ;

  let n = 64 in
  let points = Vector.create_custom Geometry_lib.point_custom n in
  let distances = Vector.create Vector.float32 n in
  for i = 0 to n - 1 do
    Vector.set
      points
      i
      {Geometry_lib.x = float_of_int i; y = float_of_int (n - i)} ;
    Vector.set distances i 0.0
  done ;

  let threads = min 64 n in
  let grid_x = (n + threads - 1) / threads in
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
        Sarek.Execute.Vec points;
        Sarek.Execute.Vec distances;
        Sarek.Execute.Int32 (Int32.of_int n);
      ]
    () ;
  Transfer.flush dev ;

  let ok = ref true in
  for i = 0 to n - 1 do
    let p = Vector.get points i in
    let expected = sqrt ((p.Geometry_lib.x *. p.x) +. (p.y *. p.y)) in
    let got = Vector.get distances i in
    if abs_float (got -. expected) > 1e-3 then begin
      ok := false ;
      if i < 5 then
        Printf.printf "  Mismatch at %d: got %f expected %f\n%!" i got expected
    end
  done ;
  if !ok then print_endline "Convention kernel test PASSED"
  else begin
    print_endline "Convention kernel test FAILED: distance mismatch" ;
    exit 1
  end
