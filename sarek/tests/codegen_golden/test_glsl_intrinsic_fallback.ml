(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * IR-level unit test for the GLSL intrinsic-dispatch fallback (task
 * [glsl-ffi-fallback]; class opened by #246 abs_float / #256 copysign).
 *
 * Two guarantees are pinned here:
 *
 * 1. NEGATIVE — an intrinsic with no GLSL lowering must raise a located
 *    [Backend_error (Codegen {backend = "Vulkan"; Unknown_intrinsic {name}})]
 *    naming the intrinsic, NOT emit the raw OCaml path [full_name(...)] that
 *    glslang rejects cryptically ("vector swizzle too long"). This covers every
 *    fall-through name in the brief inventory (warp/subgroup ops, atomic-int
 *    variants, [_f64] primitives, int bit-ops, memory fences).
 *
 * 2. POSITIVE — [log10], which IS present in the pure-registry float32/float64
 *    tables (so it "resolves" but to the non-existent GLSL builtin [log10]).
 *    The f32 spellings (unqualified and [Float32]) are polyfilled to
 *    [(log(x) / log(10.0))]; the [Float64] spelling has no GLSL f64 builtin at
 *    all and is lowered to the software helper family ([sarek_f64_log10], built
 *    over [sarek_f64_log]) — see [Sarek_ir_softmath] / [Sarek_ir_glsl].
 ******************************************************************************)

open Sarek_ir_types
module Glsl = Sarek_codegen.Sarek_ir_glsl
module Backend_error = Sarek_backend_error.Backend_error

let make_var name ty =
  {var_name = name; var_id = 0; var_type = ty; var_mutable = false}

(* A minimal kernel: c.[idx] <- <intr>(a.[idx]) with a float32 in/out pair.
   [idx] is a bare literal so the body itself never introduces another
   intrinsic that could mask the one under test. *)
let kernel_calling ~path ~name ~arity =
  let a = make_var "a" (TVec TFloat32) in
  let c = make_var "c" (TVec TFloat32) in
  let arg = EArrayRead ("a", EConst (CInt32 0l)) in
  let args = List.init arity (fun _ -> arg) in
  {
    kern_name = "fallback_probe";
    kern_params =
      [
        DParam (a, Some {arr_elttype = TFloat32; arr_memspace = Global});
        DParam (c, Some {arr_elttype = TFloat32; arr_memspace = Global});
      ];
    kern_locals = [];
    kern_body =
      SAssign
        (LArrayElem ("c", EConst (CInt32 0l)), EIntrinsic (path, name, args));
    kern_types = [];
    kern_variants = [];
    kern_funcs = [];
    kern_native_fn = None;
  }

let generate k = Glsl.generate_with_types ~types:[] k

(* --- 1. Negative: unhandled intrinsics raise a located error --- *)

(* (path, name, arity) — a representative slice of the fall-through inventory. *)
let unhandled =
  [
    ([], "warp_shuffle", 2);
    ([], "warp_vote_all", 1);
    ([], "atomic_cas_int32", 3);
    ([], "atomic_xor_int32", 2);
    ([], "clz_int32", 1);
    ([], "popcount_int32", 1);
    ([], "abs_int32", 1);
    ([], "memory_fence_block", 0);
    ([], "sin_f64", 1);
    ([], "min_int32", 2);
  ]

let test_unhandled_raises (path, name, arity) () =
  let full_name =
    match path with [] -> name | _ -> String.concat "." path ^ "." ^ name
  in
  match generate (kernel_calling ~path ~name ~arity) with
  | (_ : string) ->
      Alcotest.failf
        "expected Unknown_intrinsic for %S but generation succeeded"
        full_name
  | exception
      Backend_error.Backend_error
        (Backend_error.Codegen
           {backend; error = Backend_error.Unknown_intrinsic {name = got}}) ->
      Alcotest.(check string) "backend is Vulkan" "Vulkan" backend ;
      Alcotest.(check string) "error names the intrinsic" full_name got

(* --- 2. Positive: log10 is polyfilled, never emitted as a raw name --- *)

(* Dependency-free substring test (avoids pulling in the [str] library). *)
let string_contains ~haystack ~needle =
  let hl = String.length haystack and nl = String.length needle in
  let rec loop i =
    if i + nl > hl then false
    else if String.sub haystack i nl = needle then true
    else loop (i + 1)
  in
  nl = 0 || loop 0

(* f32 route (unqualified / Float32): [log10] is polyfilled to
   [(log(x) / log(10.0))] with the un-suffixed (single-precision) divisor. *)
let test_log10 ~path ~label () =
  let glsl = generate (kernel_calling ~path ~name:"log10" ~arity:1) in
  let contains needle = string_contains ~haystack:glsl ~needle in
  Alcotest.(check bool)
    (label ^ ": emits (log(x) / log(10.0)) polyfill")
    true
    (contains "log(" && contains "/ log(10.0)") ;
  Alcotest.(check bool)
    (label ^ ": f32 divisor carries no lf suffix")
    false
    (contains "10.0lf") ;
  Alcotest.(check bool)
    (label ^ ": no raw log10( token")
    false
    (contains "log10(")

(* Float64 route: no GLSL f64 builtin exists, so [log10] lowers to the software
   helper family — a call to [sarek_f64_log10], defined over [sarek_f64_log],
   under the int64 extension — never a bare double-typed [log10]/[log] builtin. *)
let test_log10_f64 () =
  let glsl =
    generate (kernel_calling ~path:["Float64"] ~name:"log10" ~arity:1)
  in
  let contains needle = string_contains ~haystack:glsl ~needle in
  Alcotest.(check bool)
    "Float64.log10: calls the sarek_f64_log10 helper"
    true
    (contains "sarek_f64_log10(") ;
  Alcotest.(check bool)
    "Float64.log10: defines the helper"
    true
    (contains "double sarek_f64_log10(") ;
  Alcotest.(check bool)
    "Float64.log10: routes through sarek_f64_log"
    true
    (contains "sarek_f64_log(") ;
  Alcotest.(check bool)
    "Float64.log10: emits the int64 extension"
    true
    (contains "GL_ARB_gpu_shader_int64") ;
  (* No reserved double-underscore identifier leaks into GLSL. *)
  Alcotest.(check bool)
    "Float64.log10: no __ reserved identifier"
    false
    (contains "__sarek")

let () =
  let open Alcotest in
  run
    "GLSL intrinsic fallback"
    [
      ( "unhandled-intrinsic-errors",
        List.map
          (fun ((_, name, _) as spec) ->
            test_case name `Quick (test_unhandled_raises spec))
          unhandled );
      ( "log10-polyfill",
        [
          test_case "unqualified" `Quick (test_log10 ~path:[] ~label:"log10");
          test_case
            "Float32-qualified"
            `Quick
            (test_log10 ~path:["Float32"] ~label:"Float32.log10");
          test_case "Float64-qualified" `Quick test_log10_f64;
        ] );
    ]
