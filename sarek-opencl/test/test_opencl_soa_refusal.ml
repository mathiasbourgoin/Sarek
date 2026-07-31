(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * backlog-214 - the OpenCL backend REFUSES a non-empty [~soa_params].
 *
 * [Framework_sig.generate_source] offers [?soa_params] to every backend. This
 * one bound it away as [?soa_params:_] and returned its ordinary packed-AoS
 * source: one pointer plus one length per vector parameter. The launch side
 * expands an SoA-dispatched vector into N [RSA_Buffer]s plus one
 * [RSA_Vector_Length], so AoS source under an SoA argument list is never a
 * compile error. How badly it then fails is a property of this
 * backend's binding layer, not of codegen: [Opencl_plugin_base] derives its
 * expected argument count from the indices the caller bound
 * ([Kernel_args.count]), so it has no independent count to disagree with. This
 * test does not claim which symptom follows - it claims the emitter should not
 * have accepted the request.
 *
 * [Sarek_ir_opencl] has no SoA lowering, so the request cannot be honoured and is
 * refused. Both polarities are pinned here, because the refusal has its own
 * failure mode - refusing the EMPTY list would break every ordinary launch,
 * and the ["CUDA/PTX"] caller-side gate passes [[]] on this backend on every
 * single launch:
 *
 *   1. a non-empty list raises, and the message names this framework and the
 *      parameter that was requested;
 *   2. [~soa_params:[]] and an omitted [?soa_params] both still produce source,
 *      byte-identical to each other.
 ******************************************************************************)

open Sarek_ir_types
module Backend = Sarek_opencl.Opencl_plugin.Backend
module Backend_error = Sarek_backend_error.Backend_error

let make_var name ty =
  {var_name = name; var_id = 0; var_type = ty; var_mutable = false}

(* Minimal kernel: c.[0] <- a.[0]. No intrinsic, so codegen is not what decides
   the outcome of either case below. *)
let probe_kernel () =
  let a = make_var "a" (TVec TFloat32) in
  let c = make_var "c" (TVec TFloat32) in
  {
    default_kernel with
    kern_name = "soa_refusal_probe";
    kern_params =
      [
        DParam (a, Some {arr_elttype = TFloat32; arr_memspace = Global});
        DParam (c, Some {arr_elttype = TFloat32; arr_memspace = Global});
      ];
    kern_body =
      SAssign
        ( LArrayElem ("c", EConst (CInt32 0l)),
          EArrayRead ("a", EConst (CInt32 0l)) );
  }

(* Dependency-light substring check (mirror of the codegen_golden helper). *)
let string_contains ~haystack ~needle =
  let hl = String.length haystack and nl = String.length needle in
  let rec loop i =
    if i + nl > hl then false
    else if String.sub haystack i nl = needle then true
    else loop (i + 1)
  in
  nl = 0 || loop 0

(* --- polarity 1: a non-empty list is refused, by name --- *)

let test_opencl_refuses_nonempty_soa_params () =
  match Backend.generate_source ~soa_params:["a"] (probe_kernel ()) with
  | Some src ->
      Alcotest.failf
        "expected a refusal, got %d bytes of AoS source"
        (String.length src)
  | None ->
      Alcotest.fail
        "expected a refusal, got None (request swallowed, not refused)"
  | exception
      Backend_error.Backend_error
        (Backend_error.Plugin
           {
             backend;
             error = Backend_error.Feature_not_supported {feature; backend = _};
           }) ->
      Alcotest.(check string)
        "the refusal names this framework"
        "OpenCL"
        backend ;
      Alcotest.(check bool)
        (Printf.sprintf "refusal %S names the requested parameter" feature)
        true
        (string_contains ~haystack:feature ~needle:"'a'") ;
      Alcotest.(check bool)
        (Printf.sprintf "refusal %S says what was refused" feature)
        true
        (string_contains ~haystack:feature ~needle:"Structure-of-Arrays")

(* The rendered message is what a user sees, and the framework name reaches it
   only through the [backend] field checked above - so render it and look. *)
let test_opencl_refusal_renders_the_framework () =
  match Backend.generate_source ~soa_params:["a"] (probe_kernel ()) with
  | _ -> Alcotest.fail "expected a refusal, generate_source did not raise"
  | exception Backend_error.Backend_error err ->
      let msg = Backend_error.to_string err in
      Alcotest.(check bool)
        (Printf.sprintf "rendered message %S names OpenCL" msg)
        true
        (string_contains ~haystack:msg ~needle:"OpenCL")

(* --- polarity 2: the empty list is the untouched fast path --- *)

let test_opencl_empty_soa_params_still_generates () =
  let k = probe_kernel () in
  match
    (Backend.generate_source k, Backend.generate_source ~soa_params:[] k)
  with
  | Some omitted, Some explicit_empty ->
      Alcotest.(check bool)
        "AoS source is non-empty"
        true
        (String.length omitted > 0) ;
      Alcotest.(check string)
        "~soa_params:[] is byte-identical to omitting the argument"
        omitted
        explicit_empty
  | None, _ | _, None ->
      Alcotest.fail "None for an AoS kernel: the empty-list fast path regressed"

let () =
  Alcotest.run
    "OpenCL SoA refusal (backlog-214)"
    [
      ( "soa_params",
        [
          Alcotest.test_case
            "non-empty list is refused, naming framework and parameter"
            `Quick
            test_opencl_refuses_nonempty_soa_params;
          Alcotest.test_case
            "rendered refusal names the framework"
            `Quick
            test_opencl_refusal_renders_the_framework;
          Alcotest.test_case
            "empty list still generates AoS source"
            `Quick
            test_opencl_empty_soa_params_still_generates;
        ] );
    ]
