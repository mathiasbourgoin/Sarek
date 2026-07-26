(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * The OpenCL build-option guard (#136).
 *
 * Sibling of sarek-cuda/test/test_cuda_fp_conformance.ml, and shaped the same
 * way: the screening half is PURE, so it runs on a host with no OpenCL ICD and
 * no device at all, and only the end-to-end half needs hardware.
 *
 * EVERY CHECK HERE WAS PROVED RED BY MUTATING THE THING IT CHECKS. The table
 * of mutations and the message each produced is in
 * docs/fp-contraction-policy.md §10; if you add a case, add its mutation there
 * too. A guard test that has never been seen to fail is not evidence.
 *
 * The two anti-vacuity controls are the important part of this file:
 *
 *   - [assembled_string_contains_the_option_under_test] — the gate could
 *     "pass" while Sarek assembled an option string that never reaches
 *     clBuildProgram. This asserts the option is IN the string the build path
 *     produces, so the accept/reject cases cannot be proving something about a
 *     value nobody uses.
 *   - [device_fp_config_query_is_live] — the capability gate is OFF whenever
 *     CL_DEVICE_SINGLE_FP_CONFIG lacks the bit. If the query silently returned
 *     0 (wrong ctypes width, wrong enum value, a driver returning nothing),
 *     the gate would be permanently off and every "correctly gated off" case
 *     in this file would pass while checking nothing. This asserts the query
 *     returns a NON-ZERO config on a real device, which is the only thing that
 *     distinguishes "the device says no" from "we never asked".
 ******************************************************************************)

open Sarek_opencl

(* ------------------------------------------------------------------ *)
(* Pure: screening a caller's option string                            *)
(* ------------------------------------------------------------------ *)

let refused opts =
  match Opencl_fp.check_fp_conformance opts with
  | () -> false
  | exception Opencl_fp.Fp_conformance_violation _ -> true

let must_refuse opts () =
  if not (refused opts) then
    Alcotest.failf
      "%S relaxes float semantics and must be refused on the OpenCL path; the \
       guard accepted it"
      opts

let must_accept opts () =
  if refused opts then
    Alcotest.failf
      "%S is legitimate and must be accepted; the guard refused it"
      opts

(* The relaxing options, each named individually rather than in a loop, so a
   failure names the option rather than an index. *)
let rejection_cases =
  [
    "-cl-fast-relaxed-math";
    "-cl-unsafe-math-optimizations";
    "-cl-finite-math-only";
    "-cl-no-signed-zeros";
    "-cl-mad-enable";
    "-cl-denorms-are-zero";
    "-cl-single-precision-constant";
  ]

let acceptance_cases =
  [
    "";
    "-cl-opt-disable";
    "-I/usr/include";
    "-D SAREK_TEST=1";
    "-cl-std=CL1.2";
    "-Werror";
  ]

(* Rejection must survive being embedded in a realistic option string, and must
   match a WHOLE token: a longer option that merely starts with a refused name
   is a different option and must not be caught by accident. *)
let test_embedded_in_a_longer_string () =
  must_refuse "-I/usr/include -cl-fast-relaxed-math -D N=4" () ;
  must_refuse "-D N=4 -cl-mad-enable" () ;
  must_accept "-cl-mad-enable-that-is-not-a-real-option" () ;
  must_accept "-D FLAG=-cl-fast-relaxed-math-lookalike" ()

(* ------------------------------------------------------------------ *)
(* Pure: the capability gate                                           *)
(* ------------------------------------------------------------------ *)

(* CL_FP_INF_NAN | CL_FP_ROUND_TO_NEAREST — the value BOTH devices on the
   machine this was written on actually report (0x6, measured 2026-07-26,
   rusticl/radeonsi 26.1.4-arch3.1). No CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT. *)
let fp_config_without_cr = 0x6L

let fp_config_with_cr =
  Int64.logor fp_config_without_cr Opencl_fp.cl_fp_correctly_rounded_divide_sqrt

let has_cr_flag s =
  let re = Str.regexp_string "-cl-fp32-correctly-rounded-divide-sqrt" in
  try
    ignore (Str.search_forward re s 0) ;
    true
  with Not_found -> false

let test_gate_on_when_device_advertises () =
  let opts =
    Opencl_fp.build_options ~single_fp_config:fp_config_with_cr ~caller:""
  in
  if not (has_cr_flag opts) then
    Alcotest.failf
      "device advertises CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT, so \
       -cl-fp32-correctly-rounded-divide-sqrt must be requested; got %S"
      opts

let test_gate_off_when_device_does_not_advertise () =
  let opts =
    Opencl_fp.build_options ~single_fp_config:fp_config_without_cr ~caller:""
  in
  if has_cr_flag opts then
    Alcotest.failf
      "device does NOT advertise CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT, so \
       passing -cl-fp32-correctly-rounded-divide-sqrt is an error \
       (CL_INVALID_BUILD_OPTIONS) on a conformant implementation and every \
       kernel build would fail; got %S"
      opts

(* ANTI-VACUITY CONTROL. Without this, the two cases above could both hold of a
   build_options that returns "" unconditionally. *)
let test_assembled_string_contains_the_option_under_test () =
  let opts =
    Opencl_fp.build_options ~single_fp_config:fp_config_with_cr ~caller:""
  in
  if String.trim opts = "" then
    Alcotest.fail
      "the assembled option string is empty, so the accept/reject cases in \
       this file prove nothing about what reaches clBuildProgram" ;
  if not (has_cr_flag opts) then
    Alcotest.failf
      "the assembled option string no longer contains the option under test, \
       so this check proves nothing; got %S"
      opts

let test_caller_options_are_preserved () =
  let opts =
    Opencl_fp.build_options
      ~single_fp_config:fp_config_with_cr
      ~caller:"-I/usr/include -D N=4"
  in
  let contains sub =
    try
      ignore (Str.search_forward (Str.regexp_string sub) opts 0) ;
      true
    with Not_found -> false
  in
  if not (contains "-I/usr/include" && contains "-D" && contains "N=4") then
    Alcotest.failf
      "a legitimate caller option was dropped while assembling the build \
       string; got %S"
      opts ;
  if not (has_cr_flag opts) then
    Alcotest.failf
      "Sarek's own conformance option was dropped once a caller supplied \
       options; got %S"
      opts

let test_violation_raises_before_any_assembly () =
  match
    Opencl_fp.build_options
      ~single_fp_config:fp_config_with_cr
      ~caller:"-cl-fast-relaxed-math"
  with
  | s ->
      Alcotest.failf
        "build_options assembled %S from a relaxing caller option; the guard \
         must raise instead, on the path that actually reaches clBuildProgram"
        s
  | exception Opencl_fp.Fp_conformance_violation _ -> ()

(* ------------------------------------------------------------------ *)
(* Device-backed                                                       *)
(* ------------------------------------------------------------------ *)

let with_device name f () =
  if not (Opencl_api.is_available ()) then begin
    Printf.printf "[SKIP] no OpenCL device available - skipping %s\n%!" name ;
    Alcotest.skip ()
  end
  else begin
    Opencl_api.Device.init () ;
    f (Opencl_api.Device.get 0)
  end

(* ANTI-VACUITY CONTROL — see the header. *)
let test_device_fp_config_query_is_live device =
  let cfg = device.Opencl_api.Device.single_fp_config in
  if cfg = 0L then
    Alcotest.failf
      "CL_DEVICE_SINGLE_FP_CONFIG read as 0 on %s. Every conformant OpenCL \
       device sets at least CL_FP_ROUND_TO_NEAREST and CL_FP_INF_NAN, so 0 \
       means the query is broken, not that the device is austere - and a \
       broken query silently disables the capability gate for every device, \
       making the gating tests in this file vacuous."
      device.Opencl_api.Device.name ;
  Printf.printf
    "[info] %s: CL_DEVICE_SINGLE_FP_CONFIG = 0x%Lx (correctly-rounded \
     divide/sqrt: %s; denormals: %s)\n\
     %!"
    device.Opencl_api.Device.name
    cfg
    (if Opencl_fp.has_bit cfg Opencl_fp.cl_fp_correctly_rounded_divide_sqrt then
       "yes"
     else "NO")
    (if Opencl_fp.has_bit cfg Opencl_fp.cl_fp_denorm then "yes" else "NO")

let trivial_source =
  {|
__kernel void k(__global float *out, __global const float *in) {
    int i = get_global_id(0);
    out[i] = sqrt(in[i]) + in[i] / 3.0f;
}
|}

(* The change must not break the thing it is protecting: whatever option string
   the gate produces for the LOCAL device has to be one clBuildProgram accepts.
   This is the regression that an ungated, unconditional
   -cl-fp32-correctly-rounded-divide-sqrt would cause on a conformant driver. *)
let test_real_build_still_succeeds device =
  let context = Opencl_api.Context.create device in
  let program = Opencl_api.Program.create_from_source context trivial_source in
  Opencl_api.Program.build program () ;
  let (_ : Opencl_api.Kernel.t) = Opencl_api.Kernel.create program "k" in
  ()

let test_real_build_refuses_relaxing_option device =
  let context = Opencl_api.Context.create device in
  let program = Opencl_api.Program.create_from_source context trivial_source in
  match
    Opencl_api.Program.build program ~options:"-cl-fast-relaxed-math" ()
  with
  | () ->
      Alcotest.fail
        "Program.build accepted -cl-fast-relaxed-math and compiled; the guard \
         did not fire on the real entry point"
  | exception Opencl_fp.Fp_conformance_violation _ -> ()

let () =
  Alcotest.run
    "Opencl_fp_conformance"
    [
      ( "rejects relaxing options",
        List.map
          (fun o -> Alcotest.test_case o `Quick (must_refuse o))
          rejection_cases );
      ( "accepts legitimate options",
        List.map
          (fun o ->
            Alcotest.test_case
              (if o = "" then "(empty)" else o)
              `Quick
              (must_accept o))
          acceptance_cases );
      ( "token matching",
        [
          Alcotest.test_case
            "refused options are matched as whole tokens"
            `Quick
            test_embedded_in_a_longer_string;
        ] );
      ( "capability gate",
        [
          Alcotest.test_case
            "on when the device advertises the capability"
            `Quick
            test_gate_on_when_device_advertises;
          Alcotest.test_case
            "off when it does not"
            `Quick
            test_gate_off_when_device_does_not_advertise;
          Alcotest.test_case
            "assembled string contains the option under test (anti-vacuity)"
            `Quick
            test_assembled_string_contains_the_option_under_test;
          Alcotest.test_case
            "caller options survive assembly"
            `Quick
            test_caller_options_are_preserved;
          Alcotest.test_case
            "a relaxing caller option raises instead of assembling"
            `Quick
            test_violation_raises_before_any_assembly;
        ] );
      ( "device",
        [
          Alcotest.test_case
            "CL_DEVICE_SINGLE_FP_CONFIG query is live (anti-vacuity)"
            `Quick
            (with_device "fp-config query" test_device_fp_config_query_is_live);
          Alcotest.test_case
            "a real build still succeeds under the new option string"
            `Quick
            (with_device "real build" test_real_build_still_succeeds);
          Alcotest.test_case
            "a real build refuses a relaxing option"
            `Quick
            (with_device
               "real build refusal"
               test_real_build_refuses_relaxing_option);
        ] );
    ]
