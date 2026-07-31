(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * backlog-214, the other polarity of the change: the CUDA/PTX backend must NOT
 * acquire the refusal its five siblings just did.
 *
 * CUDA/PTX is the one backend whose emitter lowers a named vector parameter to
 * N per-leaf base pointers plus one shared length, so [~soa_params] is not a
 * request it cannot serve - it is the reason the parameter exists. A refusal
 * added there would delete working functionality, which is the overshoot a
 * narrowing correction invites.
 *
 * The check has to be indirect. Naming a SCALAR-element vector in
 * [~soa_params] is rejected by the PTX emitter itself (SoA v1 is flat records
 * only), so this kernel does raise - but it must raise
 * [Ptx_codegen_error], the emitter's OWN rejection, which is only reachable if
 * the plugin threaded [~soa_params] into codegen. If the plugin had grown the
 * backlog-214 refusal, the exception would be [Backend_error] and codegen
 * would never have been entered. Distinguishing those two exceptions is
 * therefore exactly the question "did the refusal land here too".
 *
 * The record-typed SoA path itself is covered, positively and by execution, in
 * sarek/tests/unit/test_ptx_snapshot.ml; this file is not that test.
 ******************************************************************************)

open Sarek_ir_types
module Backend = Sarek_cuda.Cuda_ptx_plugin.Backend
module Backend_error = Sarek_backend_error.Backend_error

let make_var name ty =
  {var_name = name; var_id = 0; var_type = ty; var_mutable = false}

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

let test_ptx_soa_params_reaches_the_emitter () =
  match Backend.generate_source ~soa_params:["a"] (probe_kernel ()) with
  | _ ->
      Alcotest.fail
        "expected the PTX emitter's own rejection of a scalar-element SoA \
         parameter; generate_source did not raise"
  | exception Sarek_codegen.Sarek_ir_ptx_types.Ptx_codegen_error _ -> ()
  | exception Backend_error.Backend_error err ->
      Alcotest.failf
        "CUDA/PTX raised a plugin-level refusal (%s) - ~soa_params no longer \
         reaches the emitter that implements it"
        (Backend_error.to_string err)

let test_ptx_empty_soa_params_still_generates () =
  let k = probe_kernel () in
  match
    (Backend.generate_source k, Backend.generate_source ~soa_params:[] k)
  with
  | Some omitted, Some explicit_empty ->
      Alcotest.(check bool)
        "PTX source is non-empty"
        true
        (String.length omitted > 0) ;
      Alcotest.(check string)
        "~soa_params:[] is byte-identical to omitting the argument"
        omitted
        explicit_empty
  | None, _ | _, None ->
      Alcotest.fail "generate_source returned None for an AoS kernel"

let () =
  Alcotest.run
    "CUDA/PTX keeps SoA (backlog-214)"
    [
      ( "soa_params",
        [
          Alcotest.test_case
            "~soa_params reaches the PTX emitter, unrefused by the plugin"
            `Quick
            test_ptx_soa_params_reaches_the_emitter;
          Alcotest.test_case
            "empty list still generates PTX"
            `Quick
            test_ptx_empty_soa_params_still_generates;
        ] );
    ]
