(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * E2E test: a custom [@@sarek.type] record with float64 fields, used as a
 * VECTOR, with real float64 ARITHMETIC on the fields inside the kernel.
 *
 * This closes the coverage gap left open by test_ktype_mixed_align, which only
 * COPIES the float64 field verbatim ("Float64 arithmetic is out of scope for
 * the ABI test") and by test_soa_emitter_equiv, whose f64 record leg runs on
 * PTX only. The custom-vector float64 path was never exercised on OpenCL with
 * arithmetic before this test.
 *
 * Diagnosis (2026-07-24, RX 7900 XTX + Ryzen 7950X, rusticl ICD): the OpenCL
 * source generator DOES emit `#pragma OPENCL EXTENSION cl_khr_fp64 : enable`
 * for records whose fields are float64 (Sarek_ir_analysis.kernel_uses_float64
 * recurses through TRecord fields, and the record is always present in
 * ir.kern_types). The layout (L8 aligned aggregate ABI) and the kernel-arg
 * marshalling are also correct: the round-trip below matches the OCaml binary64
 * reference bit-closely on every fp64-capable device. The historical "OpenCL
 * errors on f64 custom vectors" symptom is a DEVICE-CAPABILITY gate: the
 * rusticl ICD only advertises cl_khr_fp64 when RUSTICL_FEATURES=fp64 is set, so
 * without it the program build legitimately fails with
 *   "error: use of type 'double' requires cl_khr_fp64 support".
 * The runtest rule for this test sets RUSTICL_FEATURES=fp64 so the OpenCL leg
 * is actually exercised here; devices that still do not report fp64 are skipped
 * (never failed), exactly like test_ktype_mixed_align.
 ******************************************************************************)

module Vector = Spoc_core.Vector
module Device = Spoc_core.Device
module Transfer = Spoc_core.Transfer

[@@@warning "-32"]

let () =
  Sarek_native.Native_plugin.init () ;
  Sarek_interpreter.Interpreter_plugin.init () ;
  Sarek_cuda.Cuda_plugin.init () ;
  Sarek_opencl.Opencl_plugin.init ()

type float64 = float

(* All-float64 record (natural alignment 8, stride 24). *)
type vec3 = {x : float64; y : float64; z : float64} [@@sarek.type]

(* Float64 arithmetic on every field: multiply, add, subtract, all in double
   precision. If the pragma were missing the program would not build; if the
   layout or marshalling were wrong the field values would come back wrong. *)
let scale_kirc =
  snd
    [%kernel
      fun (src : vec3 vector) (dst : vec3 vector) (n : int32) ->
        let tid = thread_idx_x in
        if tid < n then
          let p = src.(tid) in
          let next : vec3 =
            {x = (p.x *. 2.0) +. p.z; y = (p.y -. 0.5) *. p.x; z = p.z +. 1.25}
          in
          dst.(tid) <- next]

(* OCaml binary64 reference: identical arithmetic to the kernel body. *)
let ref_x x _y z = (x *. 2.0) +. z

let ref_y x y _z = (y -. 0.5) *. x

let ref_z _x _y z = z +. 1.25

let is_cpu (dev : Device.t) =
  dev.Device.framework = "Native" || dev.Device.framework = "Interpreter"

let () =
  print_endline "=== ktype record {f64;f64;f64} float64 arithmetic ===" ;
  let devs =
    Device.init ~frameworks:["Interpreter"; "Native"; "CUDA"; "OpenCL"] ()
  in
  if Array.length devs = 0 then begin
    print_endline "No runtime devices found - IR generation test passed" ;
    exit 0
  end ;
  let ir =
    match scale_kirc.Sarek.Kirc_types.body_ir with
    | Some ir -> ir
    | None -> failwith "Kernel has no IR"
  in
  (* Guard: the kernel must actually lower to a float64 kernel (the pragma
     decision keys on exactly this predicate). *)
  if not (Sarek_ir_analysis.kernel_uses_float64 ir) then begin
    print_endline
      "FAILED - kernel_uses_float64 = false (float64 record field not detected)" ;
    exit 1
  end ;
  let any_failure = ref false in
  Array.iter
    (fun (dev : Device.t) ->
      Printf.printf "runtime [%s] %s: %!" dev.Device.framework dev.Device.name ;
      if (not (is_cpu dev)) && not (Device.allows_fp64 dev) then
        print_endline "SKIP (no device fp64 support)"
      else
        try
          let n = 64 in
          let src = Vector.create_custom vec3_custom n in
          let dst = Vector.create_custom vec3_custom n in
          for i = 0 to n - 1 do
            Vector.set
              src
              i
              {
                x = float_of_int i +. 0.25;
                y = float_of_int (n - i) -. 0.5;
                z = (float_of_int i *. 0.5) +. 0.125;
              } ;
            Vector.set dst i {x = 0.0; y = 0.0; z = 0.0}
          done ;
          let threads = min 64 n in
          let grid_x = (n + threads - 1) / threads in
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
            let s = Vector.get src i in
            let d = Vector.get dst i in
            if
              abs_float (d.x -. ref_x s.x s.y s.z) > 1e-9
              || abs_float (d.y -. ref_y s.x s.y s.z) > 1e-9
              || abs_float (d.z -. ref_z s.x s.y s.z) > 1e-9
            then begin
              ok := false ;
              if i < 5 then
                Printf.printf
                  "\n\
                  \  mismatch @%d: got {%.6f,%.6f,%.6f} expected \
                   {%.6f,%.6f,%.6f}%!"
                  i
                  d.x
                  d.y
                  d.z
                  (ref_x s.x s.y s.z)
                  (ref_y s.x s.y s.z)
                  (ref_z s.x s.y s.z)
            end
          done ;
          (* Route through the shared fp64 classifier (audit #52 / F4). This
             kernel does only +, -, * on fp64 (no div/sqrt), so it never touches
             the rusticl transcendental KNOWN-ISSUE: [transcendental=false]
             keeps it a strict PASS/FAIL, exactly the pre-refactor behaviour. *)
          (* [`Known_issue] is unreachable here (transcendental:false). *)
          begin match
            Test_helpers.classify_fp64_result
              ~framework:dev.Device.framework
              ~device:dev.Device.name
              ~within_tol:!ok
              ~transcendental:false
              ~exact_ok:true
              ~max_rel:0.0
              ~non_finite:false
              ~label:""
              ()
          with
          | `Pass -> print_endline "PASSED"
          | `Known_issue s -> print_endline s
          | `Fail ->
              any_failure := true ;
              print_endline "FAILED"
          end
        with e ->
          any_failure := true ;
          Printf.printf "FAIL (%s)\n%!" (Printexc.to_string e))
    devs ;
  if !any_failure then exit 1
