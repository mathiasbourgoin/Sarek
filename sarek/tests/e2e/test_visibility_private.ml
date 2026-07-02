(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

type float32 = float

module Vector = Spoc_core.Vector
module Device = Spoc_core.Device
module Transfer = Spoc_core.Transfer

(* Force backend registration *)
let () =
  Sarek_native.Native_plugin.init () ;
  Sarek_interpreter.Interpreter_plugin.init () ;
  Sarek_cuda.Cuda_plugin.init () ;
  Sarek_opencl.Opencl_plugin.init ()

(* Test that public functions are accessible in kernels.
   Private functions (marked with sarek.module_private) should not be
   visible to external modules - this is enforced by the registry. *)
let () =
  let kernel =
    [%kernel
      fun (xs : float32 vector)
          (ys : float32 vector)
          (dst : float32 vector)
          (n : int32) ->
        let tid = thread_idx_x + (block_idx_x * block_dim_x) in
        if tid < n then dst.(tid) <- Visibility_lib.public_add xs.(tid) ys.(tid)]
  in

  let _native, kirc = kernel in
  print_endline "=== Visibility kernel IR ===" ;
  print_endline "===========================" ;

  let devs =
    Device.init ~frameworks:["Interpreter"; "Native"; "CUDA"; "OpenCL"] ()
  in
  if Array.length devs = 0 then begin
    print_endline "No device found - IR generation test passed" ;
    exit 0
  end ;
  (* Run on the Native backend: it evaluates the kernel closure directly
     (no textual codegen), so it exercises the actual public_add function
     rather than a generated-source rendering of the module-qualified call.
     Fall back to whatever device is available if Native isn't registered. *)
  let dev =
    match Array.find_opt (fun d -> d.Device.framework = "Native") devs with
    | Some d -> d
    | None -> (
        match
          Array.find_opt (fun d -> d.Device.framework = "Interpreter") devs
        with
        | Some d -> d
        | None -> devs.(0))
  in
  Printf.printf "Using device: %s\n%!" dev.Device.name ;

  let n = 64 in
  let xs = Vector.create Vector.float32 n in
  let ys = Vector.create Vector.float32 n in
  let dst = Vector.create Vector.float32 n in
  for i = 0 to n - 1 do
    Vector.set xs i (float_of_int i) ;
    Vector.set ys i (float_of_int (n - i)) ;
    Vector.set dst i 0.0
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
        Sarek.Execute.Vec xs;
        Sarek.Execute.Vec ys;
        Sarek.Execute.Vec dst;
        Sarek.Execute.Int32 (Int32.of_int n);
      ]
    () ;
  Transfer.flush dev ;

  let ok = ref true in
  for i = 0 to n - 1 do
    let x = Vector.get xs i in
    let y = Vector.get ys i in
    let expected = x +. y in
    let got = Vector.get dst i in
    if abs_float (got -. expected) > 1e-3 then begin
      ok := false ;
      if i < 5 then
        Printf.printf "  Mismatch at %d: got %f expected %f\n%!" i got expected
    end
  done ;
  if !ok then
    print_endline "Visibility test PASSED (public_add accessible in kernel)"
  else begin
    print_endline "Visibility test FAILED: public_add result mismatch" ;
    exit 1
  end
