(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** See Sarek_interp_capability.mli for what this claims and why. *)

let device_features =
  [
    Sarek_ir_analysis.Float64; Sarek_ir_analysis.Int64; Sarek_ir_analysis.Float16;
  ]

let float_coopmat_refused (c : Sarek_coopmat.component_type) :
    Sarek_capability.t =
  {
    Sarek_capability.cap_name =
      "coopmat-" ^ Sarek_coopmat.component_name c ^ "-accumulation";
    cap_kind = Sarek_capability.Policy;
    cap_why =
      "the interpreter is a strict oracle and float cooperative-matrix \
       accumulation has no single right answer to be strict about: \
       SPV_KHR_cooperative_matrix leaves the order of the k+1 additions to the \
       implementation";
    cap_evidence =
      Sarek_capability.Quoted
        "SPV_KHR_cooperative_matrix, cooperative-matrix design document 5.1";
    cap_remedy =
      Some
        "use an integer configuration, whose accumulation the specification \
         states is exact at the precision of the result type, or compare the \
         float path across devices without an interpreter oracle";
  }

(* Not [List.exists] over an enumerated config list: the interpreter holds the
   whole matrix in every invocation, so the shape is unconstrained and the set
   is infinite. Only the component types decide. *)
let coopmat_verdict (cfg : Sarek_coopmat.config) : Sarek_capability.verdict =
  let components =
    [
      cfg.Sarek_coopmat.cfg_a;
      cfg.Sarek_coopmat.cfg_b;
      cfg.Sarek_coopmat.cfg_c;
      cfg.Sarek_coopmat.cfg_result;
    ]
  in
  match
    List.find_opt
      (fun c -> not (Sarek_coopmat.component_is_integer c))
      components
  with
  | Some c -> Sarek_capability.Unavailable (float_coopmat_refused c)
  | None -> Sarek_capability.Available

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
  | Sarek_ir_analysis.Int64 ->
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
    @ List.map coopmat_verdict (Sarek_ir_analysis.kernel_coopmat_configs ir))
