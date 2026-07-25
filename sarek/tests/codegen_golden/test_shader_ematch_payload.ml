(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * IR-level unit test for the match-EXPRESSION payload fail-loud guard
 * (task [ematch-payload-fail-loud]; #73).
 *
 * A match-EXPRESSION ([EMatch]) lowers to a nested ternary (GLSL) / [select()]
 * (WGSL) that evaluates each case body in the enclosing scope and has nowhere
 * to declare a constructor payload. A case body that USES a payload binder
 * therefore emitted an undefined identifier (glslang: "undefined identifier")
 * or silently read a same-named in-scope variable. Both shader backends now
 * fail loud with a located [Unsupported_construct] instead.
 *
 * Guarantees pinned here:
 *
 * 1. NEGATIVE (red-on-mutation) — a match-expression case that binds a
 *    constructor payload AND uses it in the body raises
 *    [Backend_error (Codegen {error = Unsupported_construct
 *    {construct = "match-expression payload binding"; _}})] on both GLSL and
 *    WGSL. Without the guard, generation succeeds and emits invalid /
 *    silent-wrong code, so this test goes red if the guard is removed.
 *
 * 2. POSITIVE — a tag-only case ([OptSome] with no binder), a wildcard-payload
 *    case ([OptSome _], binder never referenced), and a [PWild] catch-all still
 *    generate successfully and emit the tag-dispatch lowering ([.tag ==] for
 *    GLSL, [select(...)] for WGSL). These forms are NOT affected by the guard.
 ******************************************************************************)

open Sarek_ir_types
module Glsl = Sarek_codegen.Sarek_ir_glsl
module Wgsl = Sarek_codegen.Sarek_ir_wgsl
module Backend_error = Sarek_backend_error.Backend_error

let make_var ?(mut = false) name ty =
  {var_name = name; var_id = 0; var_type = ty; var_mutable = mut}

let opt_constrs = [("OptNone", []); ("OptSome", [TFloat32])]

let opt_type = TVariant ("Opt", opt_constrs)

(* A kernel whose body assigns a match-EXPRESSION result to out.[idx]:
     out.[idx] <- (match opt.[idx] with <cases>)
   The [cases] are supplied by each test so the same skeleton exercises the
   payload-using (negative) and tag-only / wildcard (positive) shapes. *)
let kernel_with_cases cases =
  let opt = make_var "opt" (TVec opt_type) in
  let out = make_var "out" (TVec TFloat32) in
  let idx = make_var "idx" TInt32 in
  let body =
    SLet
      ( idx,
        EIntrinsic ([], "global_thread_id", []),
        SAssign
          ( LArrayElem ("out", EVar idx),
            EMatch (EArrayRead ("opt", EVar idx), cases) ) )
  in
  {
    kern_name = "ematch_probe";
    kern_params =
      [
        DParam (opt, Some {arr_elttype = opt_type; arr_memspace = Global});
        DParam (out, Some {arr_elttype = TFloat32; arr_memspace = Global});
      ];
    kern_locals = [];
    kern_body = body;
    kern_types = [];
    kern_variants = [("Opt", opt_constrs)];
    kern_funcs = [];
    kern_native_fn = None;
  }

let y = make_var "y" TFloat32

(* Payload BINDER USED in the body — the unsupported shape. *)
let payload_used_cases =
  [
    (PConstr ("OptSome", ["y"]), EBinop (Add, EVar y, EConst (CFloat32 1.0)));
    (PConstr ("OptNone", []), EConst (CFloat32 0.0));
  ]

(* Tag-only: no binder at all. *)
let tag_only_cases =
  [
    (PConstr ("OptSome", []), EConst (CFloat32 1.0));
    (PConstr ("OptNone", []), EConst (CFloat32 0.0));
  ]

(* Wildcard payload [OptSome _]: binds the throwaway name ["_"] which never
   appears as a referenced identifier, so it stays supported. *)
let wildcard_payload_cases =
  [
    (PConstr ("OptSome", ["_"]), EConst (CFloat32 1.0));
    (PConstr ("OptNone", []), EConst (CFloat32 0.0));
  ]

(* PWild catch-all arm. *)
let pwild_cases =
  [
    (PConstr ("OptSome", []), EConst (CFloat32 1.0));
    (PWild, EConst (CFloat32 0.0));
  ]

let gen_glsl k = Glsl.generate_with_types ~types:[] k

let gen_wgsl k = Wgsl.generate_with_types ~types:[] k

(* --- 1. Negative: payload-using match-expression fails loud --- *)

let check_fails_loud ~backend_label ~expected_backend generate () =
  match generate (kernel_with_cases payload_used_cases) with
  | (_ : string) ->
      Alcotest.failf
        "%s: expected Unsupported_construct for a payload-using match \
         expression, but generation succeeded (silent-wrong / undefined code)"
        backend_label
  | exception
      Backend_error.Backend_error
        (Backend_error.Codegen
           {backend; error = Backend_error.Unsupported_construct {construct; _}})
    ->
      Alcotest.(check string)
        (backend_label ^ ": backend tag")
        expected_backend
        backend ;
      Alcotest.(check string)
        (backend_label ^ ": names the construct")
        "match-expression payload binding"
        construct

(* --- 2. Positive: tag-only / wildcard / PWild still generate --- *)

let string_contains ~haystack ~needle =
  let hl = String.length haystack and nl = String.length needle in
  let rec loop i =
    if i + nl > hl then false
    else if String.sub haystack i nl = needle then true
    else loop (i + 1)
  in
  nl = 0 || loop 0

let check_glsl_supported ~label cases () =
  let glsl = gen_glsl (kernel_with_cases cases) in
  Alcotest.(check bool)
    (label ^ ": GLSL emits the .tag dispatch")
    true
    (string_contains ~haystack:glsl ~needle:".tag == OptSome")

let check_wgsl_supported ~label cases () =
  let wgsl = gen_wgsl (kernel_with_cases cases) in
  Alcotest.(check bool)
    (label ^ ": WGSL emits a select() dispatch")
    true
    (string_contains ~haystack:wgsl ~needle:"select(")

let () =
  let open Alcotest in
  run
    "shader EMatch payload fail-loud"
    [
      ( "fail-loud-negative",
        [
          test_case
            "GLSL raises Unsupported_construct"
            `Quick
            (check_fails_loud
               ~backend_label:"GLSL"
               ~expected_backend:"Vulkan"
               gen_glsl);
          test_case
            "WGSL raises Unsupported_construct"
            `Quick
            (check_fails_loud
               ~backend_label:"WGSL"
               ~expected_backend:"WebGPU"
               gen_wgsl);
        ] );
      ( "supported-positive",
        [
          test_case
            "GLSL tag-only"
            `Quick
            (check_glsl_supported ~label:"tag-only" tag_only_cases);
          test_case
            "GLSL wildcard payload"
            `Quick
            (check_glsl_supported
               ~label:"wildcard-payload"
               wildcard_payload_cases);
          test_case
            "GLSL PWild"
            `Quick
            (check_glsl_supported ~label:"pwild" pwild_cases);
          test_case
            "WGSL tag-only"
            `Quick
            (check_wgsl_supported ~label:"tag-only" tag_only_cases);
          test_case
            "WGSL wildcard payload"
            `Quick
            (check_wgsl_supported
               ~label:"wildcard-payload"
               wildcard_payload_cases);
          test_case
            "WGSL PWild"
            `Quick
            (check_wgsl_supported ~label:"pwild" pwild_cases);
        ] );
    ]
