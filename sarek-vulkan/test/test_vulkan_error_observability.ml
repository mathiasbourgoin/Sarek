(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Regression test for the Vulkan error-observability finding (PR #259 review).
 *
 * Before the fix, [Vulkan_plugin.Backend.generate_source] wrapped codegen in
 * [try Some (...) with _ -> None], swallowing the located
 * [Backend_error (Codegen {backend = "Vulkan"; Unknown_intrinsic {name}})]
 * into a bare [None]. Execute.ml then raised the generic
 * "generate_source returned None (kernel may use unsupported IR nodes)" and the
 * offending intrinsic NAME was lost.
 *
 * Two guarantees are pinned here:
 *
 * 1. [Backend.generate_source] on a kernel that uses an intrinsic with no GLSL
 *    lowering must PROPAGATE the located [Backend_error], not return [None].
 *    The error's rendered message must name the intrinsic.
 *
 * 2. [Printexc.to_string] on a raw [Backend_error] must render the full
 *    human-readable message (via the registered printer), not the opaque
 *    [Backend_error(_)] constructor — so any generic stringify path
 *    (e.g. Sarek_transpile's [Internal_error (Printexc.to_string exn)]) keeps
 *    the intrinsic name.
 ******************************************************************************)

open Sarek_ir_types
module Backend = Sarek_vulkan.Vulkan_plugin.Backend
module Backend_error = Sarek_backend_error.Backend_error

let make_var name ty =
  {var_name = name; var_id = 0; var_type = ty; var_mutable = false}

(* Minimal kernel: c.[0] <- <intr>(a.[0], a.[0]) with a float32 in/out pair. *)
let kernel_calling ~name ~arity =
  let a = make_var "a" (TVec TFloat32) in
  let c = make_var "c" (TVec TFloat32) in
  let arg = EArrayRead ("a", EConst (CInt32 0l)) in
  let args = List.init arity (fun _ -> arg) in
  {
    kern_name = "observability_probe";
    kern_params =
      [
        DParam (a, Some {arr_elttype = TFloat32; arr_memspace = Global});
        DParam (c, Some {arr_elttype = TFloat32; arr_memspace = Global});
      ];
    kern_locals = [];
    kern_body =
      SAssign (LArrayElem ("c", EConst (CInt32 0l)), EIntrinsic ([], name, args));
    kern_types = [];
    kern_variants = [];
    kern_funcs = [];
    kern_native_fn = None;
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

(* --- 1. generate_source propagates the located error (never swallows) --- *)

let test_generate_source_propagates () =
  let ir = kernel_calling ~name:"warp_shuffle" ~arity:2 in
  match Backend.generate_source ir with
  | Some (_ : string) ->
      Alcotest.fail
        "expected Backend_error for unsupported intrinsic, but generate_source \
         returned Some _"
  | None ->
      Alcotest.fail
        "generate_source returned None — the located Backend_error was \
         swallowed (observability regression)"
  | exception
      Backend_error.Backend_error
        (Backend_error.Codegen
           {backend; error = Backend_error.Unknown_intrinsic {name}}) ->
      Alcotest.(check string) "backend is Vulkan" "Vulkan" backend ;
      Alcotest.(check string) "error names the intrinsic" "warp_shuffle" name

(* --- 2. the rendered message (to_string) names the intrinsic --- *)

let test_message_contains_name () =
  let ir = kernel_calling ~name:"warp_shuffle" ~arity:2 in
  match Backend.generate_source ir with
  | _ -> Alcotest.fail "expected Backend_error, generation did not raise"
  | exception Backend_error.Backend_error err ->
      let msg = Backend_error.to_string err in
      Alcotest.(check bool)
        (Printf.sprintf "to_string message %S contains 'warp_shuffle'" msg)
        true
        (string_contains ~haystack:msg ~needle:"warp_shuffle")

(* --- 3. Printexc printer renders the message, not the opaque ctor --- *)

let test_printexc_printer () =
  let err =
    Backend_error.Codegen
      {
        backend = "Vulkan";
        error = Backend_error.Unknown_intrinsic {name = "warp_shuffle"};
      }
  in
  let rendered = Printexc.to_string (Backend_error.Backend_error err) in
  Alcotest.(check bool)
    (Printf.sprintf "Printexc.to_string %S names the intrinsic" rendered)
    true
    (string_contains ~haystack:rendered ~needle:"warp_shuffle") ;
  Alcotest.(check bool)
    "Printexc.to_string is not the opaque constructor"
    false
    (string_contains ~haystack:rendered ~needle:"Backend_error(_)")

(* --- 4. a well-formed kernel still compiles (no behavior change) --- *)

let test_supported_kernel_ok () =
  let a = make_var "a" (TVec TFloat32) in
  let c = make_var "c" (TVec TFloat32) in
  let ir =
    {
      kern_name = "identity_probe";
      kern_params =
        [
          DParam (a, Some {arr_elttype = TFloat32; arr_memspace = Global});
          DParam (c, Some {arr_elttype = TFloat32; arr_memspace = Global});
        ];
      kern_locals = [];
      kern_body =
        SAssign
          ( LArrayElem ("c", EConst (CInt32 0l)),
            EArrayRead ("a", EConst (CInt32 0l)) );
      kern_types = [];
      kern_variants = [];
      kern_funcs = [];
      kern_native_fn = None;
    }
  in
  match Backend.generate_source ir with
  | Some (_ : string) -> ()
  | None -> Alcotest.fail "supported kernel unexpectedly returned None"

let () =
  let open Alcotest in
  run
    "Vulkan error observability"
    [
      ( "generate_source",
        [
          test_case
            "propagates located error (no swallow)"
            `Quick
            test_generate_source_propagates;
          test_case "message names intrinsic" `Quick test_message_contains_name;
          test_case
            "supported kernel still compiles"
            `Quick
            test_supported_kernel_ok;
        ] );
      ( "printexc-printer",
        [
          test_case
            "renders message not opaque ctor"
            `Quick
            test_printexc_printer;
        ] );
    ]
