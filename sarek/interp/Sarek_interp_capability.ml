(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** See Sarek_interp_capability.mli for what this claims and why. *)

let device_features =
  [
    Sarek_ir_analysis.Float64;
    Sarek_ir_analysis.Int64;
    Sarek_ir_analysis.Float16;
    (* 8-bit storage: an element is carried as the int32 its read yields, and
       every store narrows (Sarek_ir_int8.normalise). *)
    Sarek_ir_analysis.Int8;
    (* The interpreter simulates the warp: Sarek_warp.simulated_width lanes
       exchanging through shared memory. *)
    Sarek_ir_analysis.Warp_shuffle;
  ]

(* The interpreter holds the whole matrix in every invocation, so the shape is
   unconstrained and no device list applies: the accuracy policy alone decides.
   Integer configurations are Strict and always evaluated. The one float
   configuration with a derived bound (f16*f16+f32->f32) is evaluated in a FIXED
   order once the program has accepted the relaxed accumulation; it is then one
   admissible member of the bounded set, not "the" answer (see
   Sarek_ir_interp_eval's [exec_coopmat]). *)
let coopmat_verdict ?(budget = Sarek_policy.budget ~kernels:[])
    (cfg : Sarek_coopmat.config) : Sarek_capability.verdict =
  Sarek_coopmat.policy_verdict_for budget cfg

(* [Coopmat] is a [Sarek_ir_analysis.feature] but it is NOT a width, and
   judging it by list membership the way the other three are judged is wrong in
   both directions: absent from [device_features] it refuses every cooperative
   matrix including the integer ones this evaluator computes, and present it
   permits the float ones it refuses. It is decided per CONFIGURATION, by
   {!coopmat_verdict}. [Execute.check_device_capabilities] draws the same line —
   its [gated] list is widths only and its coopmat verdicts are a separate
   list — and this partition is written as an exhaustive match rather than a
   filter so that a fifth feature is a compile error here, not a silently
   ungated capability. *)
let is_width_feature = function
  | Sarek_ir_analysis.Float64 | Sarek_ir_analysis.Float16
  | Sarek_ir_analysis.Int64 | Sarek_ir_analysis.Int8
  | Sarek_ir_analysis.Warp_shuffle ->
      true
  | Sarek_ir_analysis.Coopmat -> false

let first_refusal (ir : Sarek_ir_types.kernel) : Sarek_capability.verdict option
    =
  let provided = Some device_features in
  let required =
    List.filter
      (fun f -> is_width_feature f && Sarek_ir_analysis.kernel_uses f ir)
      Sarek_ir_analysis.all_features
  in
  Sarek_capability.first_refusal
    (List.map (Sarek_capability.device_verdict ~provided) required
    @ List.map
        (coopmat_verdict
           ~budget:(Sarek_policy.budget ~kernels:[ir.Sarek_ir_types.kern_name]))
        (Sarek_ir_analysis.kernel_coopmat_configs ir)
    (* CM_scale_add is Strict, so the interpreter evaluates every one whose
       operands are what the statement is defined on, and refuses the others
       for the reason the device gate gives. *)
    @ List.filter_map
        (function
          | Sarek_ir_types.CM_scale_add {dst_frag; src_frag; _} ->
              Some
                (Sarek_coopmat.scale_add_operands_verdict ~dst_frag ~src_frag)
          | Sarek_ir_types.CM_decl _ | Sarek_ir_types.CM_load _
          | Sarek_ir_types.CM_store _ | Sarek_ir_types.CM_muladd _ ->
              None)
        (Sarek_ir_analysis.kernel_coopmat_ops ir))
