(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * E2E regression test: recursion + vector-parameter helper on Vulkan.
 *
 * A helper that takes a vector ([TVec]) parameter cannot be emitted as a real
 * GLSL/WGSL function (a storage buffer is not a valid function argument). Before
 * the vector-helper inlining pass (Sarek_ir_inline_vec) the pipeline returned
 * [Ok] while emitting an invalid shader — the helper referenced a stripped
 * parameter that no longer existed. This test builds a Sarek IR kernel with a
 * loop-form vector-reduction helper (exactly the shape the tail-recursion
 * elimination pass produces from a tail-recursive reduction) directly and runs
 * it on a real Vulkan device, checking the result against a host reference.
 *
 *   out[tid] = sum_range(data, tid + 1)      where
 *   sum_range(arr, n) = arr[0] + arr[1] + ... + arr[n-1]
 *
 * With data[i] = 1.0 the reference is out[tid] = tid + 1 (exact in f32 for the
 * small n used here).
 *
 * Device-filtered to Vulkan only; skips cleanly (prints [SKIP], exits 0) if no
 * Vulkan device is available.
 *
 * Run with: dune exec sarek/tests/e2e/test_vulkan_recursion_vector.exe
 ******************************************************************************)

open Sarek_ir_types
module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer
module Execute = Sarek_execute.Execute

let () = Sarek_vulkan.Vulkan_plugin.init ()

let n = 256

(** Loop-form vector-reduction helper: [sum_range(arr, len)] returns the sum of
    [arr.(0 .. len-1)]. This is the shape the tail-recursion elimination pass
    emits from a tail-recursive accumulator reduction — a bounded [while] loop
    over the vector parameter, with the accumulator returned. The [arr]
    parameter is a [TVec], which is what the inlining pass must resolve. *)
let sum_range_helper () =
  let arr =
    {
      var_name = "arr";
      var_id = 10;
      var_type = TVec TFloat32;
      var_mutable = false;
    }
  in
  let len =
    {var_name = "len"; var_id = 11; var_type = TInt32; var_mutable = false}
  in
  let i =
    {var_name = "__i"; var_id = 12; var_type = TInt32; var_mutable = true}
  in
  let acc =
    {var_name = "_result"; var_id = 13; var_type = TFloat32; var_mutable = true}
  in
  let body =
    SLetMut
      ( i,
        EConst (CInt32 0l),
        SLetMut
          ( acc,
            EConst (CFloat32 0.0),
            SSeq
              [
                SWhile
                  ( EBinop (Lt, EVar i, EVar len),
                    SSeq
                      [
                        SAssign
                          ( LVar acc,
                            EBinop (Add, EVar acc, EArrayRead ("arr", EVar i))
                          );
                        SAssign
                          (LVar i, EBinop (Add, EVar i, EConst (CInt32 1l)));
                      ] );
                SReturn (EVar acc);
              ] ) )
  in
  {
    hf_name = "sum_range";
    hf_params = [arr; len];
    hf_ret_type = TFloat32;
    hf_body = body;
  }

(** out[tid] = sum_range(data, tid + 1). *)
let make_ir () : kernel =
  let data =
    {
      var_name = "data";
      var_id = 0;
      var_type = TVec TFloat32;
      var_mutable = false;
    }
  in
  let out =
    {
      var_name = "out";
      var_id = 1;
      var_type = TVec TFloat32;
      var_mutable = false;
    }
  in
  let tid =
    {var_name = "tid"; var_id = 2; var_type = TInt32; var_mutable = false}
  in
  let body =
    SLet
      ( tid,
        EIntrinsic ([], "global_thread_id", []),
        SIf
          ( EBinop (Lt, EVar tid, EConst (CInt32 (Int32.of_int n))),
            SAssign
              ( LArrayElem ("out", EVar tid),
                EApp
                  ( EVar
                      {
                        var_name = "sum_range";
                        var_id = 3;
                        var_type = TFloat32;
                        var_mutable = false;
                      },
                    [EVar data; EBinop (Add, EVar tid, EConst (CInt32 1l))] ) ),
            None ) )
  in
  {
    default_kernel with
    kern_name = "recursion_vector_vulkan";
    kern_params =
      [
        DParam (data, Some {arr_elttype = TFloat32; arr_memspace = Global});
        DParam (out, Some {arr_elttype = TFloat32; arr_memspace = Global});
      ];
    kern_body = body;
    kern_funcs = [sum_range_helper ()];
  }

let find_vulkan_device () =
  let vulkan_devices = Device.by_framework "Vulkan" in
  if Array.length vulkan_devices > 0 then Some vulkan_devices.(0) else None

let run_test (dev : Device.t) =
  let ir = make_ir () in
  let data = Vector.create Vector.float32 n in
  let out = Vector.create Vector.float32 n in
  for i = 0 to n - 1 do
    Vector.set data i 1.0 ;
    Vector.set out i 0.0
  done ;
  let block = Execute.dims1d 256 in
  let grid = Execute.dims1d ((n + 255) / 256) in
  Execute.run_vectors
    ~device:dev
    ~ir
    ~args:[Execute.Vec data; Execute.Vec out]
    ~block
    ~grid
    () ;
  Transfer.flush dev ;
  Vector.to_array out

(* Host reference: with data.(i) = 1.0, out.(tid) = tid + 1. *)
let verify result =
  let errors = ref 0 in
  for i = 0 to n - 1 do
    let expected = float_of_int (i + 1) in
    if abs_float (result.(i) -. expected) > 1e-3 then begin
      if !errors < 5 then
        Printf.printf
          "  Mismatch at %d: expected %.1f, got %.6f\n"
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
        "[SKIP] No Vulkan device available - skipping recursion+vector Vulkan \
         e2e test\n\
         %!"
  | Some dev -> (
      Printf.printf
        "Running recursion+vector kernel on Vulkan device: %s\n%!"
        dev.Device.name ;
      match run_test dev with
      | result ->
          if verify result then
            Printf.printf
              "[PASS] Vulkan recursion+vector kernel: %d elements match host \
               reference\n\
               %!"
              n
          else begin
            Printf.printf
              "[FAIL] Vulkan recursion+vector kernel produced wrong results\n%!" ;
            exit 1
          end
      | exception e ->
          Printf.printf
            "[FAIL] Vulkan recursion+vector kernel raised: %s\n%!"
            (Printexc.to_string e) ;
          exit 1)
