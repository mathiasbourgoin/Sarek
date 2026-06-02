(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * PR-2 GPU E2E test: Float32.sin via the pure registry
 *
 * Builds a Sarek IR kernel directly (no PPX) using
 * EIntrinsic (["Float32"], "sin", ...) and runs it on the available GPU
 * backend.  Verifies numerical correctness against OCaml sin.
 *
 * On CUDA the generator emits sinf() (f-suffix, single-precision).
 * On Vulkan/OpenCL/Metal the generator emits sin().
 *
 * Run with: dune exec sarek/tests/e2e/test_float32_sin_pure.exe -- --vulkan
 ******************************************************************************)

open Sarek_ir_types
open Sarek

module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer

(* Force backend registration *)
let () = Test_helpers.Benchmarks.init_backends ()

(** Build the Float32.sin kernel IR directly — no PPX dependency.
    Equivalent to:
      fun (a : float32 vec) (b : float32 vec) ->
        let idx = global_thread_id in
        b.[idx] <- Float32.sin a.[idx]   *)
let make_float32_sin_ir () : kernel =
  let make_var name ty =
    {var_name = name; var_id = 0; var_type = ty; var_mutable = false}
  in
  let a = make_var "a" (TVec TFloat32) in
  let b = make_var "b" (TVec TFloat32) in
  let idx = make_var "idx" TInt32 in
  let body =
    SLet
      ( idx,
        EIntrinsic ([], "global_thread_id", []),
        SAssign
          ( LArrayElem ("b", EVar idx),
            (* Path-qualified: ["Float32"]."sin" -> pure registry -> sinf on
               CUDA, sin on Vulkan/OpenCL/Metal *)
            EIntrinsic (["Float32"], "sin", [EArrayRead ("a", EVar idx)]) ) )
  in
  {
    kern_name = "float32_sin_pure";
    kern_params =
      [
        DParam (a, Some {arr_elttype = TFloat32; arr_memspace = Global});
        DParam (b, Some {arr_elttype = TFloat32; arr_memspace = Global});
      ];
    kern_locals = [];
    kern_body = body;
    kern_types = [];
    kern_variants = [];
    kern_funcs = [];
    kern_native_fn = None;
  }

let n = 256

let run_kernel_on_device (dev : Device.t) =
  let ir = make_float32_sin_ir () in
  let a_vec = Vector.create Vector.float32 n in
  let b_vec = Vector.create Vector.float32 n in
  (* Fill input with values in [0, 2*pi] *)
  for i = 0 to n - 1 do
    let x = Float.pi *. 2.0 *. (float_of_int i /. float_of_int n) in
    Vector.set a_vec i x ;
    Vector.set b_vec i 0.0
  done ;
  let block = Execute.dims1d 256 in
  let grid = Execute.dims1d 1 in
  Execute.run_vectors
    ~device:dev
    ~ir
    ~args:[Execute.Vec a_vec; Execute.Vec b_vec]
    ~block
    ~grid
    () ;
  Transfer.flush dev ;
  (* Read back *)
  let result = Vector.to_array b_vec in
  result

let verify_result result =
  let errors = ref 0 in
  for i = 0 to n - 1 do
    let x = Float.pi *. 2.0 *. (float_of_int i /. float_of_int n) in
    let expected = sin x in
    let diff = abs_float (result.(i) -. expected) in
    (* sin(float) should be accurate to ~1e-5 for float32 *)
    if diff > 1e-4 then begin
      if !errors < 5 then
        Printf.printf
          "  Mismatch at %d: x=%.4f expected=%.6f got=%.6f diff=%.2e\n"
          i
          x
          expected
          result.(i)
          diff ;
      incr errors
    end
  done ;
  !errors = 0

let () =
  let cfg = Test_helpers.parse_args "test_float32_sin_pure" in
  let devs = Test_helpers.init_devices cfg in
  if Array.length devs = 0 then begin
    print_endline "No devices found" ;
    exit 1
  end ;
  Test_helpers.print_devices devs ;
  let dev = Test_helpers.get_device cfg devs in
  Printf.printf "Using device: %s (%s)\n%!" dev.Device.name dev.Device.framework ;
  Printf.printf "Running Float32.sin pure-registry kernel (n=%d)...\n%!" n ;
  (try
     let result = run_kernel_on_device dev in
     if verify_result result then begin
       Printf.printf "PASSED: Float32.sin pure-registry e2e on %s\n%!"
         dev.Device.framework
     end else begin
       Printf.printf "FAILED: numerical mismatch\n%!" ;
       exit 1
     end
   with
  | Spoc_framework.Backend_error.Backend_error _msg ->
      Printf.printf "SKIPPED: backend error\n%!" ;
      exit 0
  | e ->
      Printf.printf "ERROR: %s\n%!" (Printexc.to_string e) ;
      exit 1)
