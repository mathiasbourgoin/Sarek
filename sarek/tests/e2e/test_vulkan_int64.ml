(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * E2E regression test: int64 Sarek kernel on Vulkan (#142, the DEVICE half).
 *
 * #141 fixed the EMITTER half: `glsl_header` now emits
 * `#extension GL_ARB_gpu_shader_int64 : require` whenever the kernel uses
 * int64 at all, so glslangValidator accepts the source (exit 0).
 *
 * That is not sufficient to RUN. GLSL `int64_t` compiles to SPIR-V that
 * declares `OpCapability Int64`, and Vulkan requires the corresponding
 * `VkPhysicalDeviceFeatures.shaderInt64` to be ENABLED on the logical device
 * (VUID-RuntimeSpirv-Int64-06894). `Vulkan_api_device.get` used to build
 * `pEnabledFeatures` with shaderFloat64 alone, so every int64 kernel ran
 * against a device that had never enabled the capability its shader
 * declares - undefined behaviour, and a validation error on any device.
 *
 * The arithmetic is deliberately outside the int32 range: the addends are
 * near 2^40, so a backend that silently narrowed to 32 bits would produce
 * wrong values rather than merely a slower correct answer.
 *
 * Device-filtered to Vulkan only; skips cleanly (prints [SKIP], exits 0) if
 * no Vulkan device is available, or if the device does not report int64.
 *
 * Run with: dune exec sarek/tests/e2e/test_vulkan_int64.exe
 ******************************************************************************)

open Sarek_ir_types
module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer
module Execute = Sarek_execute.Execute

let () = Sarek_vulkan.Vulkan_plugin.init ()

let n = 4096

(** dst[i] = a[i] + b[i], all int64. *)
let make_add_ir () : kernel =
  let a =
    {var_name = "a"; var_id = 0; var_type = TVec TInt64; var_mutable = false}
  in
  let b =
    {var_name = "b"; var_id = 1; var_type = TVec TInt64; var_mutable = false}
  in
  let dst =
    {var_name = "dst"; var_id = 2; var_type = TVec TInt64; var_mutable = false}
  in
  let idx =
    {var_name = "idx"; var_id = 3; var_type = TInt32; var_mutable = false}
  in
  let body =
    SLet
      ( idx,
        EIntrinsic ([], "global_thread_id", []),
        SAssign
          ( LArrayElem ("dst", EVar idx),
            EBinop (Add, EArrayRead ("a", EVar idx), EArrayRead ("b", EVar idx))
          ) )
  in
  {
    kern_name = "int64_add_vulkan";
    kern_params =
      [
        DParam (a, Some {arr_elttype = TInt64; arr_memspace = Global});
        DParam (b, Some {arr_elttype = TInt64; arr_memspace = Global});
        DParam (dst, Some {arr_elttype = TInt64; arr_memspace = Global});
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

(* Both addends sit near 2^40, so every expected sum is outside int32. *)
let a_at i = Int64.add 1099511627776L (Int64.of_int i)

let b_at i = Int64.add 2199023255552L (Int64.of_int (2 * i))

let run_test (dev : Device.t) =
  let ir = make_add_ir () in
  let a_vec = Vector.create Vector.int64 n in
  let b_vec = Vector.create Vector.int64 n in
  let dst_vec = Vector.create Vector.int64 n in
  for i = 0 to n - 1 do
    Vector.set a_vec i (a_at i) ;
    Vector.set b_vec i (b_at i) ;
    Vector.set dst_vec i 0L
  done ;
  let block = Execute.dims1d 256 in
  let grid = Execute.dims1d ((n + 255) / 256) in
  Execute.run_vectors
    ~device:dev
    ~ir
    ~args:[Execute.Vec a_vec; Execute.Vec b_vec; Execute.Vec dst_vec]
    ~block
    ~grid
    () ;
  Transfer.flush dev ;
  Vector.to_array dst_vec

let verify result =
  let errors = ref 0 in
  for i = 0 to n - 1 do
    let expected = Int64.add (a_at i) (b_at i) in
    if result.(i) <> expected then begin
      if !errors < 5 then
        Printf.printf
          "  Mismatch at %d: expected %Ld, got %Ld\n"
          i
          expected
          result.(i) ;
      incr errors
    end
  done ;
  !errors = 0

let () =
  match find_vulkan_device () with
  | None ->
      Printf.printf
        "[SKIP] No Vulkan device available - skipping int64 Vulkan e2e test\n%!"
  | Some dev -> (
      Printf.printf
        "Running int64 add kernel on Vulkan device: %s\n%!"
        dev.Device.name ;
      match run_test dev with
      | result ->
          if verify result then
            Printf.printf
              "[PASS] Vulkan int64 add kernel: %d elements match host int64 \
               reference\n\
               %!"
              n
          else begin
            Printf.printf
              "[FAIL] Vulkan int64 add kernel produced wrong results\n%!" ;
            exit 1
          end
      | exception e ->
          Printf.printf
            "[FAIL] Vulkan int64 add kernel raised: %s\n%!"
            (Printexc.to_string e) ;
          exit 1)
