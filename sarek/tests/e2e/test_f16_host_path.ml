(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * f16 HOST path, on every backend present (#57 slice 1 review, MF2).
 *
 * The Ctypes workaround for f16 landed on HIP only. Everywhere else the host
 * path went through [Ctypes_static.sizeof (Ctypes.typ_of_bigarray_kind kind)]
 * for allocation and [Ctypes.bigarray_start] for transfers, and BOTH raise
 * Failure "Unsupported bigarray kind" for Bigarray.Float16 against ctypes
 * 0.24.0 — ctypes' kind GADT has no Float16 arm. So an f16 vector died with an
 * opaque ctypes error on CUDA, OpenCL, Vulkan and Metal before any device work
 * happened, instead of reaching the deliberate slice-2 codegen diagnostic.
 *
 * What is asserted here, per enumerated device:
 *
 *   1. ALLOCATION + H2D + D2H of an f16 vector must NOT raise a raw ctypes
 *      Failure. This is the host path and it is element-agnostic beyond byte
 *      size, so it works on every backend.
 *   2. The data must survive the round trip at binary16 precision.
 *   3. Running an f16 KERNEL on a slice-2 backend must fail with that backend's
 *      LOCATED slice-2 diagnostic — never a ctypes Failure.
 *
 * Self-skips per backend: whatever devices Device.init () reports is what gets
 * exercised, so this is CPU-safe and belongs on the default runtest alias.
 ******************************************************************************)

module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer

let () = Test_helpers.Benchmarks.init_backends ()

(* Values chosen so binary16 rounding is visible: 3.14159 -> 3.140625, and
   65504 is the largest finite binary16. *)
let samples = [|3.14159; 1.0; 0.5; -2.5; 0.1; 65504.0; 0.0; -0.0|]

let n = Array.length samples

let expected = Array.map Sarek_interp.Sarek_float16.to_float16 samples

let is_ctypes_kind_failure e =
  match e with
  | Failure msg ->
      (* The exact shipped symptom. *)
      let needle = "Unsupported bigarray kind" in
      let nl = String.length needle and hl = String.length msg in
      let rec go i =
        i + nl <= hl && (String.sub msg i nl = needle || go (i + 1))
      in
      go 0
  | _ -> false

let failures = ref 0

let report label ok detail =
  if not ok then incr failures ;
  Printf.printf "    %-34s : %s%s\n" label (if ok then "OK" else "FAIL") detail

(* ------------------------------------------------------------------ *)
(* 1+2. Host allocation and round trip                                *)
(* ------------------------------------------------------------------ *)

let test_round_trip dev =
  let label = Printf.sprintf "%s host f16 round-trip" dev.Device.framework in
  match
    let v = Vector.create Vector.float16 n in
    Array.iteri (fun i x -> Vector.set v i x) samples ;
    (* Force a real H2D then D2H through this device.

       [~force:true] is REQUIRED and is the whole point of this test. After
       [to_device] the vector's location is [Both], and [Transfer.to_cpu]
       treats [Both] as already-coherent (`| Both _ -> force`), so the default
       [~force:false] performs NO device-to-host copy at all — the comparison
       below would then re-read the untouched host array and pass no matter
       what the backend's f16 D2H path did. *)
    Transfer.to_device v dev ;
    Transfer.to_cpu ~force:true v ;
    Transfer.flush dev ;
    Vector.to_array v
  with
  | got -> (
      let bad = ref None in
      Array.iteri
        (fun i g ->
          let e = expected.(i) in
          (* BIT equality: [samples] deliberately contains -0.0, and OCaml's
             [=] says [-0.0 = 0.0], so [=] could not tell whether the round
             trip preserved the sign of zero — the one thing that sample is
             there to check. *)
          let same =
            Int64.bits_of_float g = Int64.bits_of_float e || (g <> g && e <> e)
          in
          if !bad = None && not same then bad := Some (i, g, e))
        got ;
      match !bad with
      | None -> report label true ""
      | Some (i, g, e) ->
          report
            label
            false
            (Printf.sprintf " (lane %d: got %.9g expected %.9g)" i g e))
  | exception e when is_ctypes_kind_failure e ->
      report
        label
        false
        (Printf.sprintf
           " — RAW CTYPES FAILURE (this is MF2): %s"
           (Printexc.to_string e))
  | exception e ->
      report label false (Printf.sprintf " — %s" (Printexc.to_string e))

(* ------------------------------------------------------------------ *)
(* 3. An f16 KERNEL on a slice-2 backend: located diagnostic, not     *)
(*    a ctypes Failure                                                *)
(* ------------------------------------------------------------------ *)

let f16_scale =
  [%kernel
    fun (out : float16 vector) (inp : float16 vector) ->
      let open Sarek_stdlib.Std in
      let tid = global_thread_id in
      out.(tid) <- float16_of_float32 (float32_of_float16 inp.(tid) *. 2.0)]

(* The MANDATED usage shape from the f16 design doc — the conversion bound to a
   `let` rather than nested inside the store (#57 slice 1 review, MF4c). This
   failed to COMPILE with `Unbound module "Gpu"`: the intrinsic-existence witness
   emitted an unqualified [Gpu.float32_of_float16] for core primitives, naming a
   module that is not in scope at the expansion site (and a function that does
   not exist in Gpu.ml at all). Its mere presence in this file is the regression
   test; it is also executed below so the shape is not just type-checked. *)
let f16_scale_let_shape =
  [%kernel
    fun (out : float16 vector) (inp : float16 vector) ->
      let open Sarek_stdlib.Std in
      let tid = global_thread_id in
      let x = float32_of_float16 inp.(tid) in
      out.(tid) <- float16_of_float32 (x *. 2.0)]

let ir_of (_, kirc) =
  match kirc.Sarek.Kirc_types.body_ir with
  | Some ir -> ir
  | None -> failwith "no IR"

(* Frameworks that implement f16 device-side in slice 1. Everything else must
   raise its own slice-2 diagnostic. *)
let f16_capable = ["HIP"; "CUDA"; "Native"; "Interpreter"]

let test_kernel ?(shape = f16_scale) ?(suffix = "") dev =
  let fw = dev.Device.framework in
  let label = Printf.sprintf "%s f16 kernel verdict%s" fw suffix in
  let inp = Vector.create Vector.float16 n in
  Array.iteri (fun i x -> Vector.set inp i x) samples ;
  let out = Vector.create Vector.float16 n in
  match
    Sarek.Execute.run_vectors
      ~device:dev
      ~ir:(ir_of shape)
      ~args:[Vec out; Vec inp]
      ~block:(Sarek.Execute.dims1d n)
      ~grid:(Sarek.Execute.dims1d 1)
      () ;
    Transfer.flush dev
  with
  | () ->
      if List.mem fw f16_capable then report label true " (ran)"
      else
        report
          label
          false
          " — a slice-2 backend accepted an f16 kernel (silent-wrong)"
  | exception e when is_ctypes_kind_failure e ->
      report
        label
        false
        (Printf.sprintf
           " — RAW CTYPES FAILURE instead of the slice-2 diagnostic (MF2): %s"
           (Printexc.to_string e))
  | exception e ->
      if List.mem fw f16_capable then
        report label false (Printf.sprintf " — %s" (Printexc.to_string e))
      else begin
        (* Must be a proper backend diagnostic mentioning float16. *)
        let msg = Printexc.to_string e in
        let mentions_f16 =
          let needle = "float16" in
          let nl = String.length needle and hl = String.length msg in
          let rec go i =
            i + nl <= hl && (String.sub msg i nl = needle || go (i + 1))
          in
          go 0
        in
        if mentions_f16 then report label true " (slice-2 diagnostic)"
        else
          report
            label
            false
            (Printf.sprintf " — diagnostic does not name float16: %s" msg)
      end

(* ------------------------------------------------------------------ *)
(* 4. Memory.alloc / set_arg_buffer consistency (CodeRabbit, #290)     *)
(* ------------------------------------------------------------------ *)

(* [Memory.alloc] accepted [Bigarray.Float16] on the two CPU backends while
   their [Kernel.set_arg_buffer] silently omitted it from the allowed
   kind/storage match, so an f16 buffer could be allocated and then not bound.
   Nothing reached it — the live f16 launch path is Vector-based, not
   Buffer-based — which is exactly why it needed a test rather than an
   argument. This binds a real f16 buffer through the public
   [Kernel.set_arg_buffer] entry point, which is the path that was broken.

   Restricted to the two backends whose accessors were changed; the device
   backends route buffers through their own drivers. *)
let test_buffer_bind dev =
  let fw = dev.Device.framework in
  if fw = "Native" || fw = "Interpreter" then begin
    let label = Printf.sprintf "%s f16 buffer binds as arg" fw in
    match
      let args = Spoc_core.Kernel.create_args dev in
      let buf = Spoc_core.Memory.alloc dev n Bigarray.Float16 in
      Spoc_core.Kernel.set_arg_buffer args 0 buf ;
      (* Control: the f32 buffer that always worked must still work, so a
         blanket breakage cannot read as an f16 pass. *)
      let buf32 = Spoc_core.Memory.alloc dev n Bigarray.Float32 in
      Spoc_core.Kernel.set_arg_buffer args 1 buf32 ;
      Spoc_core.Memory.free buf ;
      Spoc_core.Memory.free buf32
    with
    | () -> report label true ""
    | exception e ->
        report label false (Printf.sprintf " — %s" (Printexc.to_string e))
  end

(* ------------------------------------------------------------------ *)

let () =
  Printf.printf "test_f16_host_path (#57 slice 1 review, MF2)\n" ;
  let devices = Device.init () in
  if Array.length devices = 0 then
    print_endline "    [SKIP] no devices enumerated"
  else
    Array.iter
      (fun dev ->
        Printf.printf
          "  device %d: %s (%s)\n"
          dev.Device.id
          dev.Device.name
          dev.Device.framework ;
        test_round_trip dev ;
        test_buffer_bind dev ;
        test_kernel dev ;
        (* MF4c: the documented `let`-bound conversion shape, executed. *)
        test_kernel ~shape:f16_scale_let_shape ~suffix:" (let shape)" dev)
      devices ;
  if !failures = 0 then print_endline "test_f16_host_path PASSED"
  else begin
    Printf.printf "test_f16_host_path FAILED (%d)\n" !failures ;
    exit 1
  end
