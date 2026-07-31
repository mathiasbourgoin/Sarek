(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * backlog-214 - the Vulkan backend REFUSES a non-empty [~soa_params].
 *
 * [Framework_sig.generate_source] offers [?soa_params] to every backend. This
 * one bound it away as [?soa_params:_] and returned its ordinary packed-AoS
 * source. A GLSL compute entry point has no parameter list at all, so "packed
 * AoS" here means one SSBO binding per vector with its length in the
 * push-constant block ([Sarek_ir_glsl.gen_push_constants],
 * [glsl_array_length_name]) - NOT the pointer-plus-length parameter shape the
 * C-family backends emit. The launch side expands an SoA-dispatched vector into
 * N [RSA_Buffer]s plus one [RSA_Vector_Length], so AoS source under an SoA
 * argument list is never a compile error.
 *
 * What it IS on Vulkan specifically, stated rather than assumed: this backend
 * carries a second, source-derived count -
 * [Vulkan_api_kernel.validate_buffer_indices], whose [expected_count] is read
 * from the GLSL [binding = N] declarations - so an SoA argument list against
 * AoS GLSL is caught late, as a buffer-count rejection naming the count the
 * shader declares and the count that arrived (the exact numbers depend on how
 * many vector parameters the kernel has, so they are not quoted here). It is a
 * named rejection that says nothing about SoA. That is a better outcome than
 * CUDA/C, HIP and Metal have, where nothing checks the arity at all, and it is
 * still the wrong diagnostic at the wrong layer. Refusing in the emitter makes
 * it early and named.
 *
 * [Sarek_ir_glsl] has no SoA lowering to select, so it is refused. (A
 * single-leaf record would in fact bind correctly, N = 1 making the two
 * argument lists the same shape - it is refused anyway, because that is a
 * coincidence of the leaf count and not a property of the emitter. See
 * [Backend_error.reject_soa_params].)
 *
 * Both polarities are pinned here, because the refusal has its own failure
 * mode - refusing the EMPTY list would break every ordinary launch,
 * and the ["CUDA/PTX"] caller-side gate passes [[]] on this backend on every
 * single launch:
 *
 *   1. a non-empty list raises, and the message names this framework and the
 *      parameter that was requested - checked for a scalar-element vector AND
 *      for a flat two-leaf record, the shape a real SoA launch names, so a
 *      refusal conditional on eligibility could not pass for a refusal;
 *   2. [~soa_params:[]] and an omitted [?soa_params] both still produce source,
 *      byte-identical to each other.
 ******************************************************************************)

open Sarek_ir_types
module Backend = Sarek_vulkan.Vulkan_plugin.Backend
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

let test_vulkan_refuses_nonempty_soa_params () =
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
        "Vulkan"
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
let test_vulkan_refusal_renders_the_framework () =
  match Backend.generate_source ~soa_params:["a"] (probe_kernel ()) with
  | _ -> Alcotest.fail "expected a refusal, generate_source did not raise"
  | exception Backend_error.Backend_error err ->
      let msg = Backend_error.to_string err in
      Alcotest.(check bool)
        (Printf.sprintf "rendered message %S names Vulkan" msg)
        true
        (string_contains ~haystack:msg ~needle:"Vulkan")

(* The refusal must precede codegen, not follow it. In all five backends that
   is structural - it is the first statement of [generate_source] - but
   structure is not evidence, so it is executed once, here, on the backend that
   already has a known-unsupported intrinsic pinned by
   test_vulkan_error_observability.ml. [warp_shuffle] has no GLSL lowering, so
   codegen on this kernel raises [Codegen (Unknown_intrinsic _)]; getting the
   SoA refusal instead is what proves nothing was generated before the request
   was rejected. Reachable only here: the other four backends'
   unsupported-intrinsic sets are their own, and duplicating this case five
   times would assert the same one ordering. *)
let test_vulkan_refusal_precedes_codegen () =
  let a = make_var "a" (TVec TFloat32) in
  let c = make_var "c" (TVec TFloat32) in
  let arg = EArrayRead ("a", EConst (CInt32 0l)) in
  let uncompilable =
    {
      default_kernel with
      kern_name = "soa_refusal_order_probe";
      kern_params =
        [
          DParam (a, Some {arr_elttype = TFloat32; arr_memspace = Global});
          DParam (c, Some {arr_elttype = TFloat32; arr_memspace = Global});
        ];
      kern_body =
        SAssign
          ( LArrayElem ("c", EConst (CInt32 0l)),
            EIntrinsic ([], "warp_shuffle", [arg; arg]) );
    }
  in
  (* Control: without the SoA request this kernel really is uncompilable, so the
     case below cannot pass by the intrinsic having quietly become supported. *)
  (match Backend.generate_source uncompilable with
  | _ ->
      Alcotest.fail
        "control failed: warp_shuffle now compiles to GLSL, so this kernel no \
         longer distinguishes refusal-before-codegen from refusal-after"
  | exception Backend_error.Backend_error (Backend_error.Codegen {error = _; _})
    ->
      ()) ;
  match Backend.generate_source ~soa_params:["a"] uncompilable with
  | _ -> Alcotest.fail "expected a refusal, generate_source did not raise"
  | exception
      Backend_error.Backend_error
        (Backend_error.Plugin {error = Backend_error.Feature_not_supported _; _})
    ->
      ()
  | exception Backend_error.Backend_error (Backend_error.Codegen {error = _; _})
    ->
      Alcotest.fail
        "codegen ran before the SoA request was refused - the refusal is not \
         the first thing generate_source does"

(* The scalar-element probe above is a name the PTX emitter would itself reject
   as SoA-ineligible, so on its own it cannot distinguish "refuses any non-empty
   list" from "refuses only ineligible names while still ignoring the eligible
   ones" - and the second is the pre-fix behaviour with a coat of paint. This
   case is the shape a real SoA launch actually names: a flat two-leaf record,
   which [Soa.plan] accepts and the PTX emitter does lower. It must be refused
   here too, and by the SoA refusal rather than by anything the emitter has to
   say about record vectors. *)
let test_vulkan_refuses_a_record_vector () =
  let pt = TRecord ("pt2", [("x", TFloat32); ("y", TFloat32)]) in
  let pts = make_var "pts" (TVec pt) in
  let out = make_var "out" (TVec TFloat32) in
  let k =
    {
      default_kernel with
      kern_name = "soa_refusal_record_probe";
      kern_params =
        [
          DParam (pts, Some {arr_elttype = pt; arr_memspace = Global});
          DParam (out, Some {arr_elttype = TFloat32; arr_memspace = Global});
        ];
      kern_body =
        SAssign
          ( LArrayElem ("out", EConst (CInt32 0l)),
            ERecordField (EArrayRead ("pts", EConst (CInt32 0l)), "x") );
    }
  in
  match Backend.generate_source ~soa_params:["pts"] k with
  | Some src ->
      Alcotest.failf
        "expected a refusal for a record vector, got %d bytes of AoS source"
        (String.length src)
  | None -> Alcotest.fail "expected a refusal for a record vector, got None"
  | exception
      Backend_error.Backend_error
        (Backend_error.Plugin
           {backend; error = Backend_error.Feature_not_supported {feature; _}})
    ->
      Alcotest.(check string)
        "the refusal names this framework"
        "Vulkan"
        backend ;
      Alcotest.(check bool)
        (Printf.sprintf "refusal %S names the requested parameter" feature)
        true
        (string_contains ~haystack:feature ~needle:"'pts'")

(* --- polarity 2: the empty list is the untouched fast path --- *)

let test_vulkan_empty_soa_params_still_generates () =
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
    "Vulkan SoA refusal (backlog-214)"
    [
      ( "soa_params",
        [
          Alcotest.test_case
            "non-empty list is refused, naming framework and parameter"
            `Quick
            test_vulkan_refuses_nonempty_soa_params;
          Alcotest.test_case
            "rendered refusal names the framework"
            `Quick
            test_vulkan_refusal_renders_the_framework;
          Alcotest.test_case
            "the refusal precedes codegen"
            `Quick
            test_vulkan_refusal_precedes_codegen;
          Alcotest.test_case
            "a record vector - the realistic shape - is refused too"
            `Quick
            test_vulkan_refuses_a_record_vector;
          Alcotest.test_case
            "empty list still generates AoS source"
            `Quick
            test_vulkan_empty_soa_params_still_generates;
        ] );
    ]
