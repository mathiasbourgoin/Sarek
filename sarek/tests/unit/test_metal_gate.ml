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

(* Layer 2's own red path.

   The sweep runs layer 1 first, and layer 1 raises, so on a Mac the compile
   layer is only ever observed SUCCEEDING. A gate that has only ever been seen
   passing is the thing this file exists to rule out — so drive it red directly,
   with a defect layer 1 is structurally incapable of seeing.

   The kernel below has a perfectly well-formed signature (layer 1 is silent on
   it) and an undeclared identifier in the BODY. That asymmetry is the argument
   for keeping both layers: layer 1 covers the address-space class from Linux,
   layer 2 covers everything else and only on macOS. *)
let body_defect_kernel =
  "#include <metal_stdlib>\n\
   using namespace metal;\n\
   kernel void k(device float* a [[buffer(0)]],\n\
   uint3 gid [[thread_position_in_grid]]) {\n\
  \  a[gid.x] = no_such_identifier_here;\n\
   }\n"

let test_compile_layer_goes_red_on_a_body_defect () =
  if not (Compile.available ()) then begin
    Printf.printf "  SKIP: %s\n%!" (Compile.why_unavailable ()) ;
    Alcotest.skip ()
  end
  else begin
    (* Positive control first, so a red result below is attributable to the
       kernel and not to a driver that rejects everything. *)
    (match Compile.run_metal Compile.probe with
    | Ok () -> ()
    | Error e ->
        Alcotest.failf
          "the Metal driver rejected its own probe kernel, so nothing it says \
           about ours means anything:\n\
           %s"
          e) ;
    (* Layer 1 must be SILENT here — otherwise this case would be proving layer
       1 again rather than layer 2. *)
    (match Addr.offences body_defect_kernel with
    | [] -> ()
    | os ->
        Alcotest.failf
          "layer 1 fired on the body-defect kernel, so this case no longer \
           isolates layer 2:\n\
           %s"
          (String.concat "\n" (List.map Addr.describe os))) ;
    match Compile.run_metal body_defect_kernel with
    | Ok () ->
        Alcotest.fail
          "the Metal driver ACCEPTED a kernel using an undeclared identifier. \
           Layer 2 is not compiling what it is given."
    | Error e ->
        Alcotest.(check bool)
          "the failure names the undeclared identifier"
          true
          (try
             ignore
               (Str.search_forward
                  (Str.regexp_string "no_such_identifier_here")
                  e
                  0) ;
             true
           with Not_found -> false)
  end

(* Parameter order (CodeRabbit, #316). [split_params] used to reverse the list,
   so offences were reported bottom-up. Detection never depended on order — each
   parameter is inspected on its own, so no permutation can hide one — but a
   diagnostic listing parameters in an order the reader cannot find in the
   source wastes the time this gate exists to save. Pinned here so it cannot
   silently flip back, and so the claim "order does not affect detection" is
   checked rather than asserted: the two-offence kernel below must report BOTH,
   in source order. *)
let two_offence_kernel =
  "#include <metal_stdlib>\n\
   kernel void k(constant Point2* &first [[buffer(0)]],\n\
   device float* ok [[buffer(1)]],\n\
   float* second [[buffer(2)]],\n\
   uint3 gid [[thread_position_in_grid]]) { }\n"

let test_offences_are_reported_in_source_order () =
  match Addr.offences two_offence_kernel with
  | [a; b] ->
      Alcotest.(check bool)
        "first offending parameter is the one written first"
        true
        (a.Addr.param = "constant Point2* &first [[buffer(0)]]") ;
      Alcotest.(check bool)
        "second offending parameter is the one written second"
        true
        (b.Addr.param = "float* second [[buffer(2)]]")
  | os ->
      Alcotest.failf
        "expected exactly 2 offences (the well-formed `device float* ok` must \
         not be one), got %d:\n\
         %s"
        (List.length os)
        (String.concat "\n" (List.map Addr.describe os))

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
          Alcotest.test_case
            "offences are reported in source order"
            `Quick
            test_offences_are_reported_in_source_order;
        ] );
      ( "compile",
        [
          Alcotest.test_case
            "availability is stated, never silent"
            `Quick
            test_compile_layer_states_its_availability;
          Alcotest.test_case
            "red on a body defect layer 1 cannot see"
            `Quick
            test_compile_layer_goes_red_on_a_body_defect;
        ] );
    ]
