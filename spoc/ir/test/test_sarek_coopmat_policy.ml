(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* The accuracy policy of docs/design/f16-relaxed-accuracy.md §6.1 as
   implemented by Sarek_coopmat.policy_verdict (specs/coopmat-hip-wmma.md).

   Its own executable because the opt-in is process-wide and one-way: the
   "before" cases must run in a process that has not opted in, and Alcotest
   runs the cases below in order. *)

open Sarek_coopmat

let cfg a b c r =
  {
    cfg_shape = {m = 16; n = 16; k = 16};
    cfg_a = a;
    cfg_b = b;
    cfg_c = c;
    cfg_result = r;
    cfg_saturating = false;
    cfg_scope = Subgroup;
  }

let f16_f32 = cfg Float16 Float16 Float32 Float32

let f16_f16 = cfg Float16 Float16 Float16 Float16

let f16_s32 = cfg Float16 Float16 Sint32 Sint32

let u8_s32 = cfg Uint8 Sint8 Sint32 Sint32

let available v = v = Sarek_capability.Available

let policy_refusal v =
  match v with
  | Sarek_capability.Unavailable cap ->
      cap.Sarek_capability.cap_kind = Sarek_capability.Policy
  | _ -> false

let remedy_names_call v =
  match v with
  | Sarek_capability.Unavailable {Sarek_capability.cap_remedy = Some r; _} ->
      Astring.String.is_infix ~affix:"accept_relaxed_accumulation" r
  | _ -> false

let test_before () =
  Alcotest.(check (option string))
    "no opt-in yet"
    None
    (relaxed_accumulation_reason ()) ;
  Alcotest.(check bool)
    "integer: available"
    true
    (available (policy_verdict u8_s32)) ;
  let v = policy_verdict f16_f32 in
  Alcotest.(check bool) "f16*f16+f32: Policy refusal" true (policy_refusal v) ;
  Alcotest.(check bool) "the refusal names the call" true (remedy_names_call v) ;
  Alcotest.(check bool)
    "f16 accumulate: refused"
    true
    (policy_refusal (policy_verdict f16_f16)) ;
  Alcotest.(check bool)
    "float operands into an integer accumulator: refused, although regime says \
     Strict"
    true
    (regime f16_s32 = Strict && policy_refusal (policy_verdict f16_s32))

let test_empty_reason () =
  List.iter
    (fun reason ->
      Alcotest.check_raises
        "blank reason"
        (Invalid_argument
           "Sarek_coopmat.accept_relaxed_accumulation: the reason must not be \
            empty (it is the record that the bounded deviation was understood)")
        (fun () -> accept_relaxed_accumulation ~reason))
    [""; "   "] ;
  Alcotest.(check (option string))
    "a refused call records nothing"
    None
    (relaxed_accumulation_reason ())

let test_after () =
  accept_relaxed_accumulation ~reason:"first" ;
  accept_relaxed_accumulation ~reason:"second" ;
  Alcotest.(check (option string))
    "first reason kept"
    (Some "first")
    (relaxed_accumulation_reason ()) ;
  Alcotest.(check bool)
    "f16*f16+f32: available"
    true
    (available (policy_verdict f16_f32)) ;
  Alcotest.(check bool)
    "f16 accumulate: still refused (no derived bound)"
    true
    (policy_refusal (policy_verdict f16_f16)) ;
  Alcotest.(check bool)
    "saturating f16: still refused"
    true
    (policy_refusal (policy_verdict {f16_f32 with cfg_saturating = true}))

let test_device_evidence () =
  let support =
    {
      ds_configs = [];
      ds_robust_buffer_access = false;
      ds_subgroup_size = 32;
      ds_advertised_count = 0;
      ds_scale_add = false;
      ds_evidence = "HIP: probe rule";
    }
  in
  match verdict ~support:(Some support) u8_s32 with
  | Sarek_capability.Unavailable cap ->
      Alcotest.(check bool)
        "the refusal quotes the probe's evidence"
        true
        (Astring.String.is_infix
           ~affix:"HIP: probe rule"
           cap.Sarek_capability.cap_why)
  | _ -> Alcotest.fail "an empty probed list must refuse"

(* specs/matrix-pattern-rewrite.md: operand rounding is accepted per
   precision, never with a blank reason, and bf16 x bf16 + f32 -> f32 shares the
   accumulation policy of f16. *)
let test_operand_rounding () =
  let bf16 = {f16_f32 with cfg_a = BFloat16; cfg_b = BFloat16} in
  (* Runs after [test_after]: the relaxed accumulation is accepted. *)
  Alcotest.(check bool)
    "bf16 refused before an operand-rounding opt-in"
    true
    (policy_refusal (policy_verdict bf16)) ;
  Alcotest.check_raises
    "blank reason"
    (Invalid_argument
       "Sarek_coopmat.accept_operand_rounding: the reason must not be empty \
        (it is the record that the operand rounding was understood)")
    (fun () -> accept_operand_rounding ~precision:Operands_bf16 ~reason:" ") ;
  accept_operand_rounding ~precision:Operands_f16 ~reason:"f16" ;
  Alcotest.(check bool)
    "f16 rounding does not admit bf16"
    true
    (policy_refusal (policy_verdict bf16)) ;
  accept_operand_rounding ~precision:Operands_bf16x3 ~reason:"split" ;
  Alcotest.(check (option string))
    "bf16x3 recorded"
    (Some "split")
    (operand_rounding_reason Operands_bf16x3) ;
  Alcotest.(check (option string))
    "bf16 not implied"
    None
    (operand_rounding_reason Operands_bf16) ;
  Alcotest.(check bool)
    "bf16 admitted after bf16x3 rounding"
    true
    (available (policy_verdict bf16))

(* The opt-in for the automatic float rewrite is independent of the accuracy
   opt-ins: accepting those (done by the earlier cases) does not set it, and it
   does not change the accuracy policy. *)
let test_automatic_float_rewrite () =
  Alcotest.(check (option string))
    "not implied by the accuracy opt-ins"
    None
    (automatic_float_rewrite_reason ()) ;
  List.iter
    (fun reason ->
      Alcotest.check_raises
        "blank reason refused"
        (Invalid_argument
           "Sarek_coopmat.accept_automatic_float_rewrite: the reason must not \
            be empty (it is the record that rewriting the program's own float \
            kernels was understood)")
        (fun () -> accept_automatic_float_rewrite ~reason))
    [""; "   "] ;
  Alcotest.(check (option string))
    "a refused call records nothing"
    None
    (automatic_float_rewrite_reason ()) ;
  accept_automatic_float_rewrite ~reason:"first" ;
  accept_automatic_float_rewrite ~reason:"second" ;
  Alcotest.(check (option string))
    "first reason kept"
    (Some "first")
    (automatic_float_rewrite_reason ()) ;
  Alcotest.(check bool)
    "f16 with f16 accumulator stays refused"
    true
    (policy_refusal (policy_verdict f16_f16))

let () =
  Alcotest.run
    "Sarek_coopmat policy"
    [
      ( "opt-in",
        [
          Alcotest.test_case "before the opt-in" `Quick test_before;
          Alcotest.test_case "blank reason refused" `Quick test_empty_reason;
          Alcotest.test_case "after the opt-in" `Quick test_after;
          Alcotest.test_case "operand rounding" `Quick test_operand_rounding;
          Alcotest.test_case
            "automatic float rewrite opt-in"
            `Quick
            test_automatic_float_rewrite;
          Alcotest.test_case
            "device refusal evidence"
            `Quick
            test_device_evidence;
        ] );
    ]
