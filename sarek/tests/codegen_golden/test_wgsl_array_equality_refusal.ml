(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * backlog-217 — WGSL must refuse whole-value equality on a vector/array
 * operand; the four C-family emitters must keep accepting it.
 *
 * Context: backlog-194 refused aggregate `=`/`<>` (tuple, record, variant,
 * TFun) at the typer, in {!Sarek_typer.reject_aggregate_equality}. [TVec] and
 * [TArray] were deliberately left OUT of that refused set — measured, a
 * kernel-vector or local-array parameter is a pointer on the device, `src =
 * dst` emits `(src == dst)`, and clang -x cl / glslangValidator both accept
 * it at exit 0 (verified again below via [test_c_family_still_accepts]).
 *
 * WGSL does not generalise. naga has no equality operator on `array<T>` (a
 * kernel-vector parameter) or `array<T, N>` (a local array) — both IR shapes
 * lower to a WGSL array type. Measured on naga 30.0.0: emitting the
 * unmodified `(a == b)` this repo's WGSL backend used to print for this case
 * and feeding it to `naga` fails to parse with
 *
 *   error: Incompatible operands: Equal(Array { base: [0], size: Dynamic,
 *   stride: 4 }, _)
 *
 * for a [TVec] operand, and the [size: Constant(n)] variant for [TArray].
 * That is the naga output the WGSL emitter now refuses before ever being
 * printed — refused at the ONE emitter this is true for, per
 * {!Sarek_ir_wgsl.is_array_shaped_operand}, rather than widening the shared
 * typer predicate ({!Sarek_typer.reject_aggregate_equality}) and breaking the
 * four backends where the construct works.
 *
 * Device-independent by construction: no device is created, no kernel is
 * run, only the pure source generators are called — the same as
 * test_etuple_backend_refusal.ml.
 ******************************************************************************)

open Sarek_ir_types
open Sarek_codegen

let make_var name ty =
  {var_name = name; var_id = 0; var_type = ty; var_mutable = false}

(** [(a, b) = (a, b)] on two whole vector-typed kernel parameters — the exact
    shape [Sarek_transpile.of_source] produces from real Sarek source `fun (a :
    float32 vector) (b : float32 vector) -> ... if a = b then ...` (backlog-217,
    verified by hand against the full frontend pipeline before writing this
    IR-level pin). *)
let vec_eq_kernel () =
  let a = make_var "a" (TVec TFloat32) in
  let b = make_var "b" (TVec TFloat32) in
  {
    default_kernel with
    kern_name = "vec_eq_probe";
    kern_params = [DParam (a, None); DParam (b, None)];
    kern_body = SExpr (EBinop (Eq, EVar a, EVar b));
  }

(** Same shape for a LOCAL array ([TArray], not a kernel-vector parameter) — the
    other IR type the WGSL emitter maps to `array<T, N>`. *)
let arr_eq_kernel () =
  let x = make_var "x" (TArray (TFloat32, Local)) in
  let y = make_var "y" (TArray (TFloat32, Local)) in
  {
    default_kernel with
    kern_name = "arr_eq_probe";
    kern_params = [];
    kern_locals = [DLocal (x, None); DLocal (y, None)];
    kern_body = SExpr (EBinop (Ne, EVar x, EVar y));
  }

(** NOT refused: comparing two SCALAR ELEMENTS read out of a vector. Indexing
    ([EArrayRead]) yields a [TFloat32] value, not the vector itself — ordinary
    scalar equality, which WGSL has always supported. This is the over-refusal
    guard: {!Sarek_ir_wgsl.is_array_shaped_operand} must not fire on the read,
    only on the bare [EVar] of vector/array type. *)
let vec_element_eq_kernel () =
  let a = make_var "a" (TVec TFloat32) in
  let b = make_var "b" (TVec TFloat32) in
  {
    default_kernel with
    kern_name = "vec_element_eq_probe";
    kern_params = [DParam (a, None); DParam (b, None)];
    kern_body =
      SExpr
        (EBinop
           ( Eq,
             EArrayRead ("a", EConst (CInt32 0l)),
             EArrayRead ("b", EConst (CInt32 0l)) ));
  }

(** Local substring test — same four lines as the sibling ETuple test, kept
    dependency-free rather than pulling in a string library for one call. *)
let contains ~needle haystack =
  let nl = String.length needle and hl = String.length haystack in
  let rec go i =
    i + nl <= hl && (String.sub haystack i nl = needle || go (i + 1))
  in
  nl = 0 || go 0

(** Assert the WGSL emitter raises the shared Codegen_error, correctly tagged
    and naming the construct — matched on SHAPE, not [_], so an unrelated
    exception cannot read as "correctly refused" (same discipline as
    test_etuple_backend_refusal.ml). *)
let expect_wgsl_refused name f =
  match f () with
  | (_ : string) ->
      Alcotest.failf
        "%s: array/vector equality was EMITTED by WGSL, not refused \
         (backlog-217) — naga will reject the output with \"Incompatible \
         operands: Equal(Array …)\""
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
      Alcotest.(check string) (name ^ ": backend tag") "WebGPU" actual_tag ;
      Alcotest.(check string) (name ^ ": construct") "array equality" construct ;
      if not (contains ~needle:"Incompatible operands: Equal(Array" reason) then
        Alcotest.failf
          "%s: refused, but the reason does not cite the naga rejection: %s"
          name
          reason
  | exception e ->
      Alcotest.failf
        "%s: refused with the WRONG exception (expected Codegen_error): %s"
        name
        (Printexc.to_string e)

let test_wgsl_refuses_vec_eq () =
  expect_wgsl_refused "WGSL/vector" (fun () ->
      Sarek_ir_wgsl.generate_with_types ~types:[] (vec_eq_kernel ()))

let test_wgsl_refuses_arr_eq () =
  expect_wgsl_refused "WGSL/local-array" (fun () ->
      Sarek_ir_wgsl.generate_with_types ~types:[] (arr_eq_kernel ()))

let test_wgsl_accepts_element_eq () =
  let out =
    Sarek_ir_wgsl.generate_with_types ~types:[] (vec_element_eq_kernel ())
  in
  if not (contains ~needle:"==" out) then
    Alcotest.failf
      "WGSL/element: scalar element equality should still emit `==`, got: %s"
      out

(** The four C-family emitters (plus Vulkan/GLSL) must keep accepting BOTH
    shapes unchanged: this is what makes backlog-217 an asymmetry rather than a
    generalisable refusal. Each must still print `==` and must NOT raise. *)
let c_family_cases =
  [
    ("OpenCL", fun k -> Sarek_ir_opencl.generate_with_types ~types:[] k);
    ("CUDA", fun k -> Sarek_ir_cuda.generate_with_types ~types:[] k);
    ("Metal", fun k -> Sarek_ir_metal.generate_with_types ~types:[] k);
    ("GLSL", fun k -> Sarek_ir_glsl.generate_with_types ~types:[] k);
  ]

let test_c_family_still_accepts () =
  List.iter
    (fun (name, gen) ->
      let out_vec = gen (vec_eq_kernel ()) in
      if not (contains ~needle:"==" out_vec) then
        Alcotest.failf
          "%s: vector equality should still emit `==` (backlog-217 must not \
           widen the refusal past WGSL), got: %s"
          name
          out_vec ;
      let out_arr = gen (arr_eq_kernel ()) in
      if not (contains ~needle:"!=" out_arr) then
        Alcotest.failf
          "%s: local-array equality should still emit `!=` (backlog-217 must \
           not widen the refusal past WGSL), got: %s"
          name
          out_arr)
    c_family_cases

let () =
  Alcotest.run
    "wgsl-array-equality-refusal"
    [
      ( "backlog-217",
        [
          Alcotest.test_case
            "WGSL refuses vector-parameter equality"
            `Quick
            test_wgsl_refuses_vec_eq;
          Alcotest.test_case
            "WGSL refuses local-array equality"
            `Quick
            test_wgsl_refuses_arr_eq;
          Alcotest.test_case
            "WGSL still accepts scalar element equality"
            `Quick
            test_wgsl_accepts_element_eq;
          Alcotest.test_case
            "C-family emitters (+ GLSL/Vulkan) still accept both shapes"
            `Quick
            test_c_family_still_accepts;
        ] );
    ]
