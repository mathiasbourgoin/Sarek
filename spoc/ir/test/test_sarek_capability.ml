(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Unit tests for Sarek_capability (#64 slice 1)
 *
 * The verdict algebra, the static/dynamic split, the diagnostic rendering, and
 * the whole-kernel refusal composer.
 ******************************************************************************)

open Sarek_ir_types
open Sarek_capability

(* The neighbouring tests in this directory use bare [assert], which `-noassert`
   compiles out — a build flag away from a suite that passes without checking
   anything. These checks raise unconditionally instead. *)
let check name cond =
  if not cond then failwith ("capability check failed: " ^ name)

let check_string name ~expected ~actual =
  if expected <> actual then
    failwith
      (Printf.sprintf
         "capability check failed: %s\n  expected: %s\n  actual:   %s"
         name
         expected
         actual)

(** {1 The verdict algebra} *)

let sample_cap =
  {
    cap_name = "widget";
    cap_kind = Backend_structural;
    cap_why = "the target has no widget";
    cap_evidence = Quoted "the spec says so";
    cap_remedy = None;
  }

(* THE safety property of the module. A two-valued model forces an unprobed
   device into a bucket, and every `not unsupported` spelling puts it in
   "permitted". If this test ever goes green with [Unknown] permitting, the
   capability model has become the thing it was built to prevent. *)
let test_permits () =
  check "Available permits" (permits Available) ;
  check "Unavailable does not permit" (not (permits (Unavailable sample_cap))) ;
  check
    "Unknown does not permit"
    (not (permits (Unknown "probe failed: no device context"))) ;
  print_endline "  permits: Unknown does not permit: OK"

let test_first_refusal () =
  check
    "all-available yields no refusal"
    (first_refusal [Available; Available] = None) ;
  (* The interesting case: an [Unknown] sitting between permitting verdicts must
     still be caught. This is what a `List.find_opt (function Unavailable _ ->
     true | _ -> false)` written by hand would miss. *)
  (match first_refusal [Available; Unknown "no probe"; Available] with
  | Some (Unknown r) ->
      check_string "unknown reason" ~expected:"no probe" ~actual:r
  | _ -> failwith "first_refusal must surface an Unknown between Availables") ;
  (match first_refusal [Available; Unavailable sample_cap] with
  | Some (Unavailable c) ->
      check_string "refused cap" ~expected:"widget" ~actual:c.cap_name
  | _ -> failwith "first_refusal must surface an Unavailable") ;
  check "empty list permits" (first_refusal [] = None) ;
  print_endline "  first_refusal: OK"

(** {1 The static/dynamic split} *)

let test_kind_needs_device () =
  (* This predicate is what a later slice consults to know which capabilities
     still need a launch gate, so the split is asserted rather than assumed. *)
  check "structural is static" (not (kind_needs_device Backend_structural)) ;
  check "policy is static" (not (kind_needs_device Policy)) ;
  check "device-optional needs a device" (kind_needs_device Device_optional) ;
  check "host-toolchain needs a probe" (kind_needs_device Host_toolchain) ;
  check
    "toolchain-semantic needs a measurement"
    (kind_needs_device Toolchain_semantic) ;
  check "flag-legality needs a device bit" (kind_needs_device Flag_legality) ;
  print_endline "  kind_needs_device: OK"

let test_kind_names () =
  check_string
    "structural"
    ~expected:"backend-structural"
    ~actual:(kind_name Backend_structural) ;
  check_string
    "toolchain-semantic"
    ~expected:"toolchain-semantic"
    ~actual:(kind_name Toolchain_semantic) ;
  check_string
    "flag-legality"
    ~expected:"flag-legality"
    ~actual:(kind_name Flag_legality) ;
  print_endline "  kind_name: OK"

let test_evidence_provenance () =
  (* Measured-vs-quoted must survive into the message: the fp-contraction work
     turned on being able to tell "we observed this on named hardware" from
     "a vendor document asserts it". *)
  check_string
    "measured"
    ~expected:"measured"
    ~actual:(evidence_provenance (Measured "x")) ;
  check_string
    "quoted"
    ~expected:"quoted"
    ~actual:(evidence_provenance (Quoted "x")) ;
  check_string
    "by construction"
    ~expected:"by construction"
    ~actual:(evidence_provenance (By_construction "x")) ;
  check_string "text" ~expected:"x" ~actual:(evidence_text (Measured "x")) ;
  print_endline "  evidence: OK"

(** {1 Rendering} *)

let contains haystack needle =
  let nl = String.length needle and hl = String.length haystack in
  let rec go i =
    i + nl <= hl && (String.sub haystack i nl = needle || go (i + 1))
  in
  nl = 0 || go 0

(* A diagnostic that does not name the capability and the target is the failure
   mode #64 exists to remove, so it is asserted directly rather than through a
   golden string (which would drift into asserting punctuation). *)
let test_explain () =
  let msg = explain ~target:"Metal" float64_absent_metal in
  check "names the target" (contains msg "Metal") ;
  check "names the capability" (contains msg "float64") ;
  check "names the kind" (contains msg "backend-structural") ;
  check "names the provenance" (contains msg "quoted") ;
  check "gives the remedy" (contains msg "Sarek_real64") ;
  (* Negative control: [explain] must not be a function that says "Metal" no
     matter what it is handed. *)
  let other = explain ~target:"WGSL" {sample_cap with cap_name = "float16"} in
  check "other target named" (contains other "WGSL") ;
  check "other capability named" (contains other "float16") ;
  check "no stray Metal" (not (contains other "Metal")) ;
  print_endline "  explain: OK"

(** {1 The whole-kernel refusal composer} *)

let mk_kernel elt : kernel =
  let v =
    {var_name = "x"; var_id = 0; var_type = TVec elt; var_mutable = false}
  in
  {
    default_kernel with
    kern_name = "test";
    kern_params = [DParam (v, Some {arr_elttype = elt; arr_memspace = Global})];
  }

exception Refused of string

let test_refuse_if_used () =
  let refuse =
    refuse_if_used
      ~raise_:(fun r -> raise (Refused r))
      ~target:"Metal"
      float64_absent_metal
      Sarek_ir_analysis.Float64
  in
  (* Red: a kernel that genuinely requests the missing capability is refused,
     with a message that names it. *)
  (match refuse (mk_kernel TFloat64) with
  | () -> failwith "an f64 kernel must be refused"
  | exception Refused msg ->
      check "refusal names float64" (contains msg "float64") ;
      check "refusal names Metal" (contains msg "Metal")) ;
  (* Positive control. Without this, a composer that raised unconditionally
     would pass the test above — the gate must DISCRIMINATE, not merely fire. *)
  (match refuse (mk_kernel TFloat32) with
  | () -> ()
  | exception Refused _ ->
      failwith "an f32 kernel must NOT be refused (gate fires unconditionally)") ;
  print_endline "  refuse_if_used: fires on f64, silent on f32: OK"

(* #142, the device half. Three properties, each with the control that stops it
   passing vacuously. *)
let test_device_verdict () =
  let open Sarek_ir_analysis in
  (* 1. An unprobed device refuses. This is the safety property restated at the
     one call site that will carry real device data, and it is the direction
     the #142 defect failed in: nothing could describe int64, so int64 was
     never refused. *)
  (match device_verdict ~provided:None Int64 with
  | Unknown why ->
      check "unprobed device names the feature" (contains why "int64") ;
      check "unprobed device does not permit" (not (permits (Unknown why)))
  | Available | Unavailable _ ->
      failwith "an unprobed device must yield Unknown, not a decision") ;
  (* 2. A device that provides the feature permits it. Positive control: without
     this, [device_verdict] could refuse everything and pass (1) and (3). *)
  check
    "provided feature permits"
    (permits (device_verdict ~provided:(Some [Float64; Int64]) Int64)) ;
  (* 3. The widths are INDEPENDENT. A device list containing only Float64 must
     refuse Int64 — the exact confusion the old [supports_fp64] bool forced,
     where the only available answer to "does it do int64" was the fp64 one. *)
  (match device_verdict ~provided:(Some [Float64]) Int64 with
  | Unavailable cap ->
      check "refusal names int64" (cap.cap_name = "int64") ;
      check "device gap is Device_optional" (cap.cap_kind = Device_optional) ;
      check "Device_optional needs a device" (kind_needs_device cap.cap_kind) ;
      let msg = explain ~target:"Fake GPU" cap in
      check "explain names the target" (contains msg "Fake GPU") ;
      check "explain names shaderInt64" (contains msg "shaderInt64")
  | Available -> failwith "an fp64-only device must NOT permit int64"
  | Unknown _ -> failwith "a probed device must decide, not return Unknown") ;
  (* 4. And symmetrically, so (3) cannot be passing because Int64 is hardcoded
     unavailable. *)
  (match device_verdict ~provided:(Some [Int64]) Float64 with
  | Unavailable cap ->
      check "symmetric refusal names float64" (cap.cap_name = "float64")
  | Available | Unknown _ ->
      failwith "an int64-only device must NOT permit float64") ;
  print_endline "  device_verdict: Unknown refuses, widths independent: OK"

let () =
  print_endline "Testing Sarek_capability (#64 slice 1)..." ;
  test_device_verdict () ;
  test_permits () ;
  test_first_refusal () ;
  test_kind_needs_device () ;
  test_kind_names () ;
  test_evidence_provenance () ;
  test_explain () ;
  test_refuse_if_used () ;
  print_endline "All Sarek_capability tests passed!"
