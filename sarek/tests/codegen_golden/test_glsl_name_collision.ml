(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * IR-level unit test for the GLSL collision-safe helper-name machinery
 * (PR #255 [sarek_smod] / PR #256 [sarek_copysign]).
 *
 * HISTORY: this replaces the former e2e test [test_glsl_mod_name_collision.ml],
 * whose premise - a *user* kernel parameter literally named [sarek_smod] - is
 * now rejected at PPX elaboration by the reserved-prefix policy (see the
 * negative test [test_reserved_prefix_param.ml]). A user can no longer author
 * the collision through the front end.
 *
 * The collision-safe name computation is kept as defense-in-depth and is
 * exercised here directly at the IR level, where no reserved-prefix policy
 * applies (the IR is post-elaboration, and generated names legitimately start
 * with [sarek_]). We build [Sarek_ir_types.kernel] values whose param / helper
 * names already occupy the default helper name and assert the machinery falls
 * back to a fresh, non-colliding name - the exact guarantee PR #255/#256 add.
 ******************************************************************************)

open Sarek_ir_types
module Glsl = Sarek_codegen.Sarek_ir_glsl

let make_var name ty =
  {var_name = name; var_id = 0; var_type = ty; var_mutable = false}

let kernel_with ~params ~funcs =
  {
    kern_name = "collision_probe";
    kern_params = params;
    kern_locals = [];
    kern_body = SEmpty;
    kern_types = [];
    kern_variants = [];
    kern_funcs = funcs;
    kern_native_fn = None;
  }

(* No collision: the default base name is returned verbatim. *)
let test_no_collision () =
  let k =
    kernel_with
      ~params:
        [
          DParam
            ( make_var "a" (TVec TInt32),
              Some {arr_elttype = TInt32; arr_memspace = Global} );
          DParam (make_var "divisor" TInt32, None);
        ]
      ~funcs:[]
  in
  Alcotest.(check string)
    "smod base name uncontested"
    "sarek_smod"
    (Glsl.compute_smod_name k) ;
  Alcotest.(check string)
    "copysign base name uncontested"
    "sarek_copysign"
    (Glsl.compute_copysign_name k)

(* A scalar param occupying [sarek_smod] forces the suffix fallback. This is the
   scenario the old e2e test drove through the PPX; it is only constructible at
   the IR level now. *)
let test_param_collision () =
  let k =
    kernel_with
      ~params:
        [
          DParam
            ( make_var "a" (TVec TInt32),
              Some {arr_elttype = TInt32; arr_memspace = Global} );
          DParam (make_var "sarek_smod" TInt32, None);
        ]
      ~funcs:[]
  in
  Alcotest.(check string)
    "smod name avoids colliding param"
    "sarek_smod_1"
    (Glsl.compute_smod_name k)

(* A helper function occupying [sarek_smod] also forces the fallback. *)
let test_helper_collision () =
  let helper =
    {
      hf_name = "sarek_smod";
      hf_params = [make_var "x" TInt32];
      hf_ret_type = TInt32;
      hf_body = SEmpty;
    }
  in
  let k = kernel_with ~params:[] ~funcs:[helper] in
  Alcotest.(check string)
    "smod name avoids colliding helper"
    "sarek_smod_1"
    (Glsl.compute_smod_name k)

(* Consecutive occupied slots [sarek_smod], [sarek_smod_1] fall through to the
   next free suffix. *)
let test_cascading_collision () =
  let k =
    kernel_with
      ~params:
        [
          DParam (make_var "sarek_smod" TInt32, None);
          DParam (make_var "sarek_smod_1" TInt32, None);
        ]
      ~funcs:[]
  in
  Alcotest.(check string)
    "smod name skips to first free suffix"
    "sarek_smod_2"
    (Glsl.compute_smod_name k)

let () =
  let open Alcotest in
  run
    "GLSL collision-safe names"
    [
      ( "compute_collision_safe_name",
        [
          test_case "no collision" `Quick test_no_collision;
          test_case "param collision" `Quick test_param_collision;
          test_case "helper collision" `Quick test_helper_collision;
          test_case "cascading collision" `Quick test_cascading_collision;
        ] );
    ]
