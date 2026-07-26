(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Negative controls for the Metal validation gate (#139).

    The metal_validation_sweep in test_codegen_golden.ml is the gate; this file
    is the proof that the gate can go red, and it matters more here than for the
    other backends: layer 2 ([xcrun metal]) cannot run on Linux at all, so layer
    1 is the only thing standing between this project and another year of
    committed Metal that has never compiled. A check that has never been
    observed failing is not evidence of anything.

    The first case below is the ACTUAL committed golden from before the fix,
    pasted verbatim. It is the string an Apple M4 (macOS 15.6.1, Apple clang 17)
    refused, and it is here so the detector stays pinned to the real defect
    rather than to a synthetic approximation of it. *)

module Addr = Metal_gate.Metal_addrspace
module Compile = Metal_gate.Metal_compile

(* The record_kernel golden as it stood before #139 was fixed. Apple clang 17
   rejected it; so must layer 1. *)
let historical_record_kernel =
  "#include <metal_stdlib>\n\
   using namespace metal;\n\
   #pragma METAL fp contract(off)\n\n\
   typedef struct {\n\
  \  float x;\n\
  \  float y;\n\
   } Point2;\n\n\
   kernel void record_kernel(constant Point2* &pts [[buffer(0)]], constant int \
   &sarek_pts_length [[buffer(1)]],\n\
   uint3 __metal_gid [[thread_position_in_grid]]) {\n\
  \  int idx = __metal_gid.x;\n\
  \  Point2 p = pts[idx];\n\
   }\n"

(* The same kernel with the fix applied: `device T*`, no reference. *)
let fixed_record_kernel =
  "#include <metal_stdlib>\n\
   using namespace metal;\n\
   #pragma METAL fp contract(off)\n\n\
   typedef struct {\n\
  \  float x;\n\
  \  float y;\n\
   } Point2;\n\n\
   kernel void record_kernel(device Point2* pts [[buffer(0)]], constant int \
   &sarek_pts_length [[buffer(1)]],\n\
   uint3 __metal_gid [[thread_position_in_grid]]) {\n\
  \  int idx = __metal_gid.x;\n\
  \  Point2 p = pts[idx];\n\
   }\n"

let test_historical_defect_is_red () =
  match Addr.offences historical_record_kernel with
  | [] ->
      Alcotest.fail
        "layer 1 accepted `constant Point2* &pts`, the exact parameter an \
         Apple M4 refused and the reason #139 exists. The gate is not a gate."
  | [o] ->
      Alcotest.(check bool)
        "names the offending parameter"
        true
        (o.Addr.param = "constant Point2* &pts [[buffer(0)]]")
  | os ->
      Alcotest.failf
        "expected exactly one offence, got %d:\n%s"
        (List.length os)
        (String.concat "\n" (List.map Addr.describe os))

let test_fixed_shape_is_green () =
  match Addr.offences fixed_record_kernel with
  | [] -> ()
  | os ->
      Alcotest.failf
        "layer 1 rejected the FIXED signature, so it would block the fix it \
         exists to force:\n\
         %s"
        (String.concat "\n" (List.map Addr.describe os))

(* A pointer with no address space at all — the other half of MSL 3.2 §4.2.
   Sarek does not currently emit this shape; the control is here so that if a
   future arm does, the gate says so instead of shrugging. *)
let test_unqualified_pointer_is_red () =
  let src =
    "#include <metal_stdlib>\n\
     kernel void k(float* a [[buffer(0)]],\n\
     uint3 gid [[thread_position_in_grid]]) { a[gid.x] = 1.0f; }\n"
  in
  match Addr.offences src with
  | [] ->
      Alcotest.fail
        "layer 1 accepted `float* a` with no address space; Metal has no \
         default address space (MSL 3.2 §4.2)."
  | _ -> ()

(* Scalars and thread-position builtins must NOT trip the check: `constant int
   &n` is a reference to a scalar, which is the correct Metal spelling, and
   `uint3 gid` is not a pointer. A gate that fires on those is noise and would
   be turned off. *)
let test_scalar_and_builtin_params_are_green () =
  let src =
    "#include <metal_stdlib>\n\
     kernel void k(device float* a [[buffer(0)]], constant int &n [[buffer(1)]],\n\
     threadgroup float* scratch [[threadgroup(0)]],\n\
     uint3 gid [[thread_position_in_grid]]) { a[gid.x] = float(n); }\n"
  in
  match Addr.offences src with
  | [] -> ()
  | os ->
      Alcotest.failf
        "layer 1 fired on well-formed Metal parameters:\n%s"
        (String.concat "\n" (List.map Addr.describe os))

(* Anti-vacuity: a signature parser that silently returns None turns every case
   above into a free pass. *)
let test_signature_is_actually_found () =
  match Addr.kernel_signature fixed_record_kernel with
  | None ->
      Alcotest.fail
        "kernel_signature found no `kernel void ...(...)`, so every offence \
         check above inspected an empty parameter list and asserted nothing"
  | Some params ->
      Alcotest.(check bool)
        "the parameter list is non-empty"
        true
        (String.trim params <> "")

(* Layer 2's availability is a positive control, not `command -v xcrun`. On
   Linux it must report unavailable WITH a reason; on macOS with Xcode it must
   report available. Either way the reason string must be non-empty when
   unavailable, because that string is the only thing distinguishing an honest
   skip from a silent one. *)
let test_compile_layer_states_its_availability () =
  if Compile.available () then
    Alcotest.(check string)
      "available => no reason"
      ""
      (Compile.why_unavailable ())
  else
    Alcotest.(check bool)
      "unavailable => a stated reason"
      true
      (String.length (Compile.why_unavailable ()) > 0)

let () =
  Alcotest.run
    "metal_gate"
    [
      ( "addrspace",
        [
          Alcotest.test_case
            "the pre-#139 golden is REJECTED (red proof)"
            `Quick
            test_historical_defect_is_red;
          Alcotest.test_case
            "the fixed signature is accepted"
            `Quick
            test_fixed_shape_is_green;
          Alcotest.test_case
            "an unqualified pointer is rejected"
            `Quick
            test_unqualified_pointer_is_red;
          Alcotest.test_case
            "scalars, threadgroup buffers and builtins are accepted"
            `Quick
            test_scalar_and_builtin_params_are_green;
          Alcotest.test_case
            "the signature parser finds a non-empty parameter list"
            `Quick
            test_signature_is_actually_found;
        ] );
      ( "compile",
        [
          Alcotest.test_case
            "availability is stated, never silent"
            `Quick
            test_compile_layer_states_its_availability;
        ] );
    ]
