(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * backlog-194 — every C-family emitter must REFUSE Ir.ETuple, not print a
 * brace list.
 *
 * Why this test exists as well as the frontend ones. The PPX no longer builds
 * Ir.ETuple: Sarek_lower_ir's TETuple arm raises, and the aggregate-equality
 * refusal removed the only source route that reached it. That makes the five
 * C-family arms unreachable FROM SOURCE — which is exactly the condition under
 * which a bad emitter arm rots unnoticed. Ir.ETuple is a public constructor of
 * spoc/ir/Sarek_ir_types.ml, so IR built directly (a snapshot, a unit test, a
 * library consumer) can still carry it, and before this change all five arms
 * printed
 *
 *   {a, b}
 *
 * which is not an expression in C. Measured on 97a062a2 through the OpenCL
 * emitter and clang -x cl -cl-std=CL1.2: "expected ';' after expression",
 * "expected ')'", "statement requires expression of scalar type". Emitting
 * illegal device source and leaving the vendor compiler to notice is the
 * failure mode this project has spent the week removing; the PTX emitter
 * already refused, and now the other five do too.
 *
 * Device-independent by construction: no device is created, no kernel is run,
 * only the pure source generators are called. It therefore means the same
 * thing on a GPU-less CI runner as it does here.
 ******************************************************************************)

open Sarek_ir_types
open Sarek_codegen

(** Local substring test — the sibling f16 test carries the same four lines
    rather than pulling in a dependency for one call. *)
let contains ~needle haystack =
  let nl = String.length needle and hl = String.length haystack in
  let rec go i =
    i + nl <= hl && (String.sub haystack i nl = needle || go (i + 1))
  in
  nl = 0 || go 0

let mk_kernel () =
  let v =
    {var_name = "out"; var_id = 0; var_type = TInt32; var_mutable = true}
  in
  {
    default_kernel with
    kern_name = "etuple_probe";
    kern_params = [DParam (v, None)];
    kern_body = SExpr (ETuple [EConst (CInt32 1l); EConst (CInt32 2l)]);
  }

(** Assert the emitter raises the shared Codegen_error with construct "ETuple".

    The exception SHAPE is matched, never [_]: an unrelated Not_found would
    otherwise read as "correctly refused", which is the vacuous-gate shape this
    repository keeps finding. The [backend] tag is checked too, so a copy-paste
    of one backend's arm into another is caught rather than passing on the
    strength of the construct name alone. *)
let expect_refused ~backend ~tag name f =
  match f () with
  | (_ : string) ->
      Alcotest.failf
        "%s: ETuple was EMITTED, not refused — the brace-list arm is back \
         (backlog-194)"
        name
  | exception
      Sarek_backend_error.Backend_error.Backend_error
        (Sarek_backend_error.Backend_error.Codegen
           {
             backend = actual_tag;
             error =
               Sarek_backend_error.Backend_error.Unsupported_construct
                 {construct; reason};
           }) ->
      Alcotest.(check string) (name ^ ": backend tag") tag actual_tag ;
      Alcotest.(check string) (name ^ ": construct") "ETuple" construct ;
      (* Substring, not equality: the reason carries the explanation and the
         backlog reference, and pinning it whole would make every reword a test
         edit. What must not drift is that the diagnostic names the node. *)
      if not (contains ~needle:"tuple value reached the emitter" reason) then
        Alcotest.failf
          "%s: refused, but the reason does not describe the node: %s"
          name
          reason ;
      ignore backend
  | exception e ->
      Alcotest.failf
        "%s: refused with the WRONG exception (expected Codegen_error): %s"
        name
        (Printexc.to_string e)

let cases =
  [
    ( "OpenCL",
      "OpenCL",
      fun k -> Sarek_ir_opencl.generate_with_types ~types:[] k );
    ("CUDA", "CUDA", fun k -> Sarek_ir_cuda.generate_with_types ~types:[] k);
    ("Metal", "Metal", fun k -> Sarek_ir_metal.generate_with_types ~types:[] k);
    (* GLSL's Backend_error tag is "Vulkan" — that is the framework name. *)
    ("GLSL", "Vulkan", fun k -> Sarek_ir_glsl.generate_with_types ~types:[] k);
    ("WGSL", "WebGPU", fun k -> Sarek_ir_wgsl.generate_with_types ~types:[] k);
  ]

let tests =
  List.map
    (fun (name, tag, gen) ->
      Alcotest.test_case (name ^ " refuses ETuple") `Quick (fun () ->
          expect_refused ~backend:name ~tag name (fun () -> gen (mk_kernel ()))))
    cases

let () = Alcotest.run "etuple-backend-refusal" [("backlog-194", tests)]
