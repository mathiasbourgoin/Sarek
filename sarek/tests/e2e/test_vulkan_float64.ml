(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * E2E regression test: float64 (double-precision) Sarek kernel on Vulkan.
 *
 * Vulkan's GLSL codegen emits `double` for TFloat64 (Sarek_ir_glsl.ml,
 * glsl_type_of_elttype) but the GLSL header used to always be a bare
 * `#version 450` with no `#extension GL_ARB_gpu_shader_fp64 : require`
 * (Sarek_ir_glsl.ml, glsl_header), and the Vulkan logical device used to be
 * created with `pEnabledFeatures = null` (Vulkan_api_device.ml, get), so the
 * `shaderFloat64` physical-device feature was never enabled. A SPIR-V shader
 * using the Float64 capability without shaderFloat64 enabled is a Vulkan
 * spec violation.
 *
 * This test builds a Sarek IR kernel directly (no PPX, same shape as
 * test_float64_math_intrinsics.ml) computing
 *   dst[i] = a[i] * b[i] + c[i]
 * entirely in float64, runs it on a real Vulkan device, and checks the
 * result against a host-computed float64 reference.
 *
 * Device-filtered to Vulkan only; skips cleanly (prints [SKIP], exits 0) if
 * no Vulkan device is available - see [main] below.
 *
 * Run with: dune exec sarek/tests/e2e/test_vulkan_float64.exe
 ******************************************************************************)

open Sarek_ir_types
module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer
module Execute = Sarek_execute.Execute

let () = Sarek_vulkan.Vulkan_plugin.init ()

let n = 4096

(** dst[i] = a[i] * b[i] + c[i], all float64. *)
let make_fma_ir () : kernel =
  let a =
    {var_name = "a"; var_id = 0; var_type = TVec TFloat64; var_mutable = false}
  in
  let b =
    {var_name = "b"; var_id = 1; var_type = TVec TFloat64; var_mutable = false}
  in
  let c =
    {var_name = "c"; var_id = 2; var_type = TVec TFloat64; var_mutable = false}
  in
  let dst =
    {
      var_name = "dst";
      var_id = 3;
      var_type = TVec TFloat64;
      var_mutable = false;
    }
  in
  let idx =
    {var_name = "idx"; var_id = 4; var_type = TInt32; var_mutable = false}
  in
  let body =
    SLet
      ( idx,
        EIntrinsic ([], "global_thread_id", []),
        SAssign
          ( LArrayElem ("dst", EVar idx),
            EBinop
              ( Add,
                EBinop
                  (Mul, EArrayRead ("a", EVar idx), EArrayRead ("b", EVar idx)),
                EArrayRead ("c", EVar idx) ) ) )
  in
  {
    kern_name = "float64_fma_vulkan";
    kern_params =
      [
        DParam (a, Some {arr_elttype = TFloat64; arr_memspace = Global});
        DParam (b, Some {arr_elttype = TFloat64; arr_memspace = Global});
        DParam (c, Some {arr_elttype = TFloat64; arr_memspace = Global});
        DParam (dst, Some {arr_elttype = TFloat64; arr_memspace = Global});
      ];
    kern_locals = [];
    kern_body = body;
    kern_types = [];
    kern_variants = [];
    kern_funcs = [];
    kern_native_fn = None;
  }

let find_vulkan_device () =
  let vulkan_devices = Device.by_framework "Vulkan" in
  if Array.length vulkan_devices > 0 then Some vulkan_devices.(0) else None

let run_test (dev : Device.t) =
  let ir = make_fma_ir () in
  let a_vec = Vector.create Vector.float64 n in
  let b_vec = Vector.create Vector.float64 n in
  let c_vec = Vector.create Vector.float64 n in
  let dst_vec = Vector.create Vector.float64 n in
  (* Values chosen to require true double precision: differences on the
     order of 1e-12 must survive the multiply-add, which float32 could not
     represent. *)
  for i = 0 to n - 1 do
    let x = 1.0 +. (float_of_int i *. 1e-12) in
    Vector.set a_vec i x ;
    Vector.set b_vec i x ;
    Vector.set c_vec i (float_of_int i *. 1e-13) ;
    Vector.set dst_vec i 0.0
  done ;
  let block = Execute.dims1d 256 in
  let grid = Execute.dims1d ((n + 255) / 256) in
  Execute.run_vectors
    ~device:dev
    ~ir
    ~args:
      [
        Execute.Vec a_vec;
        Execute.Vec b_vec;
        Execute.Vec c_vec;
        Execute.Vec dst_vec;
      ]
    ~block
    ~grid
    () ;
  Transfer.flush dev ;
  Vector.to_array dst_vec

let verify result =
  let errors = ref 0 in
  for i = 0 to n - 1 do
    let x = 1.0 +. (float_of_int i *. 1e-12) in
    let c = float_of_int i *. 1e-13 in
    let expected = (x *. x) +. c in
    let diff = abs_float (result.(i) -. expected) in
    (* Tight tolerance appropriate for double precision: 1e-15 relative. *)
    if diff > (1e-15 *. abs_float expected) +. 1e-15 then begin
      if !errors < 5 then
        Printf.printf
          "  Mismatch at %d: expected %.17g, got %.17g (diff %.3e)\n"
          i
          expected
          result.(i)
          diff ;
      incr errors
    end
  done ;
  !errors = 0

let () =
  match find_vulkan_device () with
  | None ->
      Printf.printf
        "[SKIP] No Vulkan device available - skipping float64 Vulkan e2e test\n\
         %!"
  | Some dev ->
      if not (Device.allows_fp64 dev) then begin
        Printf.printf
          "[SKIP] Vulkan device %s does not report fp64 support\n%!"
          dev.Device.name
      end
      else begin
        Printf.printf
          "Running float64 FMA kernel on Vulkan device: %s\n%!"
          dev.Device.name ;
        match run_test dev with
        | result ->
            if verify result then
              Printf.printf
                "[PASS] Vulkan float64 FMA kernel: %d elements match host \
                 double-precision reference\n\
                 %!"
                n
            else begin
              Printf.printf
                "[FAIL] Vulkan float64 FMA kernel produced wrong results\n%!" ;
              exit 1
            end
        | exception e ->
            Printf.printf
              "[FAIL] Vulkan float64 FMA kernel raised: %s\n%!"
              (Printexc.to_string e) ;
            exit 1
      end
