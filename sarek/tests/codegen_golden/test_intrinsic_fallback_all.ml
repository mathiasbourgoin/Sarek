(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * IR-level unit test for the unified intrinsic-dispatch fall-through on the
 * four non-GLSL source backends (WGSL/WebGPU, Metal, CUDA, OpenCL).
 *
 * This pins the audit #48 fix delivered by the #49 unification: an intrinsic
 * with no backend lowering (and no FFI-registry template) must now RAISE a
 * located [Backend_error (Codegen {backend; Unknown_intrinsic {name}})] on
 * EVERY backend, exactly as GLSL already did after #259. Before #49 these four
 * backends silently emitted the raw OCaml path [warp_shuffle(...)] — a call to
 * a non-existent device function — and the pipeline returned [Ok], yielding
 * invalid device code that only failed (cryptically) at the driver.
 *
 * GLSL's own fall-through is already covered by [test_glsl_intrinsic_fallback].
 ******************************************************************************)

open Sarek_ir_types
module Backend_error = Sarek_backend_error.Backend_error
module Wgsl = Sarek_codegen.Sarek_ir_wgsl
module Metal = Sarek_codegen.Sarek_ir_metal
module Cuda = Sarek_codegen.Sarek_ir_cuda
module Opencl = Sarek_codegen.Sarek_ir_opencl

let make_var name ty =
  {var_name = name; var_id = 0; var_type = ty; var_mutable = false}

(* A minimal kernel: c.[0] <- <intr>(a.[0], a.[0]) with a float32 in/out pair.
   The indices are bare literals so the body never introduces another intrinsic
   that could mask the one under test. *)
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

(* Each backend's IR-to-source entry point, uniformly [kernel -> string]. *)
let backends =
  [
    ("WGSL", "WebGPU", fun k -> Wgsl.generate_with_types ~types:[] k);
    ("Metal", "Metal", fun k -> Metal.generate_with_types ~types:[] k);
    ("CUDA", "CUDA", fun k -> Cuda.generate_with_types ~types:[] k);
    ("OpenCL", "OpenCL", fun k -> Opencl.generate_with_types ~types:[] k);
  ]

(* An intrinsic with no lowering on any backend and no FFI-registry template. *)
let unknown_name = "warp_shuffle"

let test_backend_raises (label, expected_backend, generate) () =
  let k = kernel_calling ~path:[] ~name:unknown_name ~arity:2 in
  match generate k with
  | (_ : string) ->
      Alcotest.failf
        "%s: expected Unknown_intrinsic for %S but generation succeeded"
        label
        unknown_name
  | exception
      Backend_error.Backend_error
        (Backend_error.Codegen
           {backend; error = Backend_error.Unknown_intrinsic {name = got}}) ->
      Alcotest.(check string)
        (label ^ ": backend name")
        expected_backend
        backend ;
      Alcotest.(check string)
        (label ^ ": error names the intrinsic")
        unknown_name
        got

let () =
  let open Alcotest in
  run
    "intrinsic fallback (all backends)"
    [
      ( "unknown-intrinsic-raises",
        List.map
          (fun ((label, _, _) as spec) ->
            test_case label `Quick (test_backend_raises spec))
          backends );
    ]
