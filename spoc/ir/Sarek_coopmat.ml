(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Cooperative matrix: the DEVICE-facing half (backlog-62 slice 2).

    The vocabulary lives in {!Sarek_coopmat_types}, which this module re-exports
    wholesale — see there for why the split exists. What is left here is exactly
    the part that consults {!Sarek_capability}: given a device's probed support,
    is a requested configuration permitted? *)

include Sarek_coopmat_types

let device_lacks_config ?(evidence = "") cfg =
  {
    Sarek_capability.cap_name = "cooperative-matrix";
    (* Device_optional, per docs/design/capability-model.md §2: the backend can
       spell the instruction, a given device may not provide it. The local box
       has one device that does and one that does not under the SAME driver
       (Vulkan: radv; HIP: ROCm), which is why a driver-keyed or backend-keyed
       refusal would be wrong here. *)
    cap_kind = Sarek_capability.Device_optional;
    cap_why =
      Printf.sprintf
        "the device offers no cooperative-matrix configuration matching %s%s"
        (config_name cfg)
        (if evidence = "" then "" else " (" ^ evidence ^ ")");
    cap_evidence =
      Sarek_capability.Measured
        "Vulkan: AMD Radeon RX 7900 XTX (RADV NAVI31) advertises \
         VK_KHR_cooperative_matrix revision 2 with 14 configurations; the AMD \
         Ryzen 9 7950X iGPU (RADV RAPHAEL_MENDOCINO) does not, under the same \
         radv / Mesa 26.1.4-arch3.1. HIP: the RX 7900 XTX (gfx1100) executes \
         the gfx11 WMMA instructions and the iGPU (gfx1036, RDNA2) has none, \
         under ROCm 7.2.4 (specs/coopmat-hip-wmma.md)";
    cap_remedy =
      Some
        "Use an ordinary multiply-accumulate kernel, or select a device that \
         offers this cooperative-matrix configuration.";
  }

let verdict ~support cfg =
  match support with
  | None ->
      Sarek_capability.Unknown
        "device cooperative-matrix support was not probed, so no configuration \
         can be confirmed"
  | Some s ->
      if
        List.exists
          (fun advertised ->
            config_matches
              ~shape:cfg.cfg_shape
              ~a:cfg.cfg_a
              ~b:cfg.cfg_b
              ~c:cfg.cfg_c
              ~result:cfg.cfg_result
              advertised
            && advertised.cfg_saturating = cfg.cfg_saturating
            && advertised.cfg_scope = cfg.cfg_scope)
          s.ds_configs
      then Sarek_capability.Available
      else
        Sarek_capability.Unavailable
          (device_lacks_config ~evidence:s.ds_evidence cfg)

(* ------------------------------------------------------------------------ *)
(* The accuracy policy: docs/design/f16-relaxed-accuracy.md §6.1             *)
(* ------------------------------------------------------------------------ *)

(* The opt-ins are grants of {!Sarek_policy}, which keeps them one-way and
   process-wide (an [Atomic] there: launches run on several domains and the
   value is read on every capability check; there is deliberately no way to
   withdraw an opt-in, so a capability check that passed, and that Execute
   memoises, can never become stale). The first non-empty reason wins. These
   functions keep their historical contract of raising on a blank reason. *)
let accept_process grant ~message ~reason =
  match Sarek_policy.grant Sarek_policy.Process grant ~reason with
  | Ok () -> ()
  (* Process scope cannot be refused as [Default_kernel_name]. *)
  | Error (Sarek_policy.Blank_reason | Sarek_policy.Default_kernel_name) ->
      invalid_arg message

let accept_relaxed_accumulation ~reason =
  accept_process
    (Sarek_policy.Deviation Sarek_policy.Accumulation_order)
    ~reason
    ~message:
      "Sarek_coopmat.accept_relaxed_accumulation: the reason must not be empty \
       (it is the record that the bounded deviation was understood)"

let relaxed_accumulation_reason () =
  Sarek_policy.reason (Sarek_policy.Deviation Sarek_policy.Accumulation_order)

type operand_precision = Sarek_policy.operand_precision =
  | Operands_f16
  | Operands_bf16
  | Operands_bf16x3

let operand_precision_name = function
  | Operands_f16 -> "f16"
  | Operands_bf16 -> "bf16"
  | Operands_bf16x3 -> "bf16x3"

let rounding_grant precision =
  Sarek_policy.Deviation (Sarek_policy.Operand_rounding precision)

let accept_operand_rounding ~precision ~reason =
  accept_process
    (rounding_grant precision)
    ~reason
    ~message:
      "Sarek_coopmat.accept_operand_rounding: the reason must not be empty (it \
       is the record that the operand rounding was understood)"

let operand_rounding_reason precision =
  Sarek_policy.reason (rounding_grant precision)

(* The opt-in for the AUTOMATIC float matrix rewrite (Matrix_specialise). Same
   contract as the accuracy opt-ins above; it grants no accuracy deviation by
   itself, only the permission to apply the accepted ones to a program's own
   kernels. *)
let automatic_float_grant =
  Sarek_policy.Consent Sarek_policy.Automatic_float_rewrite

let accept_automatic_float_rewrite ~reason =
  accept_process
    automatic_float_grant
    ~reason
    ~message:
      "Sarek_coopmat.accept_automatic_float_rewrite: the reason must not be \
       empty (it is the record that rewriting the program's own float kernels \
       was understood)"

let automatic_float_rewrite_reason () =
  Sarek_policy.reason automatic_float_grant

(* The float configurations a bound has been derived for: f16 x f16 + f32 ->
   f32 (§5.2) and bf16 x bf16 + f32 -> f32 (specs/matrix-pattern-rewrite.md:
   the same derivation, since a product of two 8-bit-significand values is
   exact in binary32 too). Only the order of the additions is free. *)
let has_derived_bound cfg =
  let narrow = function Float16 | BFloat16 -> true | _ -> false in
  narrow cfg.cfg_a && cfg.cfg_a = cfg.cfg_b && cfg.cfg_c = Float32
  && cfg.cfg_result = Float32 && not cfg.cfg_saturating

let relaxed_not_accepted cfg =
  {
    Sarek_capability.cap_name = "coopmat-relaxed-accumulation";
    cap_kind = Sarek_capability.Policy;
    cap_why =
      Printf.sprintf
        "%s accumulates in floating point in an order the hardware chooses, so \
         its result is only BOUNDED (|D - exact| <= gamma_n * (sum|A*B| + \
         |C|)), not bit-identical to the interpreter, and %s"
        (config_name cfg)
        (if
           Sarek_policy.revoked
             (Sarek_policy.Deviation Sarek_policy.Accumulation_order)
         then "accumulation-order is revoked by SAREK_OPT_DISABLE"
         else "the program has not accepted that");
    cap_evidence =
      Sarek_capability.Quoted
        "docs/design/f16-relaxed-accuracy.md §5 (the bound) and §6.1 (a \
         bounded deviation needs an explicit opt-in), decided 2026-07-27";
    cap_remedy =
      Some
        "Call Sarek_coopmat.accept_relaxed_accumulation ~reason:\"...\" once \
         before launching, or use an integer configuration (exact).";
  }

let no_derived_bound cfg =
  {
    Sarek_capability.cap_name = "coopmat-float-accumulation";
    cap_kind = Sarek_capability.Policy;
    cap_why =
      Printf.sprintf
        "%s has no derived error bound: only f16*f16+f32->f32 and \
         bf16*bf16+f32->f32 have one (an f16 result rounds every partial sum \
         to binary16; a float operand with an integer accumulator has no \
         defined meaning)"
        (config_name cfg);
    cap_evidence =
      Sarek_capability.Quoted "docs/design/f16-relaxed-accuracy.md §5.4";
    cap_remedy =
      Some "Accumulate in f32 (f16*f16+f32->f32 or bf16*bf16+f32->f32).";
  }

(* A bf16 operand is a float32 buffer rounded on load: a deviation beyond the
   accumulation order, so it also needs the operand-rounding opt-in. *)
let bf16_rounding_accepted budget =
  Sarek_policy.Budget.permits budget (rounding_grant Operands_bf16)
  || Sarek_policy.Budget.permits budget (rounding_grant Operands_bf16x3)

let rounding_not_accepted cfg =
  {
    Sarek_capability.cap_name = "coopmat-operand-rounding";
    cap_kind = Sarek_capability.Policy;
    cap_why =
      Printf.sprintf
        "%s rounds float32 operands to bfloat16 on load, which changes results \
         beyond the accumulation order, and %s"
        (config_name cfg)
        (match
           List.filter
             Sarek_policy.revoked
             [rounding_grant Operands_bf16; rounding_grant Operands_bf16x3]
         with
        | [] -> "the program has not accepted that"
        | [g] -> Sarek_policy.grant_name g ^ " is revoked by SAREK_OPT_DISABLE"
        | gs ->
            String.concat " and " (List.map Sarek_policy.grant_name gs)
            ^ " are revoked by SAREK_OPT_DISABLE");
    cap_evidence =
      Sarek_capability.Quoted
        "specs/matrix-pattern-rewrite.md (operand rounding is its own opt-in)";
    cap_remedy =
      Some
        "Call Sarek_coopmat.accept_operand_rounding ~precision:Operands_bf16 \
         (or Operands_bf16x3) ~reason:\"...\" once before launching.";
  }

(* Keyed on ALL FOUR component types, not on {!regime}: [regime] reads only
   the addend and the result, so a float-operand configuration with an integer
   accumulator would come out [Strict] — a configuration no device offers and
   whose meaning (convert a float operand to an integer?) nothing defines. *)
let policy_verdict_for budget cfg =
  let all_integer =
    List.for_all
      component_is_integer
      [cfg.cfg_a; cfg.cfg_b; cfg.cfg_c; cfg.cfg_result]
  in
  if all_integer then Sarek_capability.Available
  else if not (has_derived_bound cfg) then
    Sarek_capability.Unavailable (no_derived_bound cfg)
  else if
    not
      (Sarek_policy.Budget.permits
         budget
         (Sarek_policy.Deviation Sarek_policy.Accumulation_order))
  then Sarek_capability.Unavailable (relaxed_not_accepted cfg)
  else if cfg.cfg_a = BFloat16 && not (bf16_rounding_accepted budget) then
    Sarek_capability.Unavailable (rounding_not_accepted cfg)
  else Sarek_capability.Available

(* The grants an [Available] float verdict rests on (none for an integer
   configuration). *)
let grants_behind budget cfg =
  let all_integer =
    List.for_all
      component_is_integer
      [cfg.cfg_a; cfg.cfg_b; cfg.cfg_c; cfg.cfg_result]
  in
  if all_integer then []
  else
    let rounding =
      if cfg.cfg_a <> BFloat16 then []
      else
        match
          List.find_opt
            (Sarek_policy.Budget.permits budget)
            [rounding_grant Operands_bf16; rounding_grant Operands_bf16x3]
        with
        | Some g -> [g]
        | None -> []
    in
    Sarek_policy.Deviation Sarek_policy.Accumulation_order :: rounding

let policy_decision_for budget cfg =
  match policy_verdict_for budget cfg with
  | Sarek_capability.Unavailable cap ->
      Sarek_policy.Refused cap.Sarek_capability.cap_name
  | Sarek_capability.Unknown why -> Sarek_policy.Refused why
  | Sarek_capability.Available -> (
      match grants_behind budget cfg with
      | [] -> Sarek_policy.Applied Sarek_policy.Exact_by_construction
      | grants ->
          Sarek_policy.Applied
            (Sarek_policy.Granted
               {
                 grants;
                 reason =
                   Sarek_policy.join_reasons
                     (List.filter_map
                        (Sarek_policy.Budget.reason budget)
                        grants);
                 lowering = None;
               }))

let policy_verdict cfg =
  policy_verdict_for (Sarek_policy.budget ~kernels:[]) cfg

(* ------------------------------------------------------------------------ *)
(* Fragments without a multiply-add                                         *)
(* ------------------------------------------------------------------------ *)

let fragment_name f =
  Printf.sprintf
    "%s %s fragment of %s (%s scope)"
    (component_name f.frag_component)
    (use_name f.frag_use)
    (shape_name f.frag_shape)
    (scope_name f.frag_scope)

let device_lacks_fragment ?(evidence = "") f =
  {
    Sarek_capability.cap_name = "cooperative-matrix";
    cap_kind = Sarek_capability.Device_optional;
    cap_why =
      Printf.sprintf
        "the device offers no cooperative-matrix configuration with a %s%s"
        (fragment_name f)
        (if evidence = "" then "" else " (" ^ evidence ^ ")");
    cap_evidence =
      Sarek_capability.Quoted
        "a fragment is the per-lane layout of one operand of an advertised \
         configuration; a device that offers none has no layout to load it \
         with (specs/coopmat-hip-wmma.md)";
    cap_remedy =
      Some "Select a device that offers a configuration with this fragment.";
  }

let fragment_verdict ~support f =
  match support with
  | None ->
      Sarek_capability.Unknown
        "device cooperative-matrix support was not probed, so no fragment \
         layout can be confirmed"
  | Some s ->
      let offered cfg =
        let a, b, c, r = fragments_of_config cfg in
        List.mem f [a; b; c; r]
      in
      if List.exists offered s.ds_configs then Sarek_capability.Available
      else
        Sarek_capability.Unavailable
          (device_lacks_fragment ~evidence:s.ds_evidence f)

(* ------------------------------------------------------------------------ *)
(* Scaled accumulation                                                      *)
(* ------------------------------------------------------------------------ *)

let scale_add_design = "docs/design/coopmat-scaled-accumulate.md"

let scale_add_refusal ~name ~why ~remedy =
  Sarek_capability.Unavailable
    {
      Sarek_capability.cap_name = name;
      cap_kind = Sarek_capability.Backend_structural;
      cap_why = why;
      cap_evidence = Sarek_capability.Quoted (scale_add_design ^ " §2.2");
      cap_remedy = Some remedy;
    }

let scale_add_operands_verdict ~dst_frag ~src_frag =
  if dst_frag.frag_use <> Accumulator || dst_frag.frag_component <> Float32 then
    scale_add_refusal
      ~name:"coopmat-scale-add-destination"
      ~why:
        (Printf.sprintf
           "CM_scale_add accumulates into an f32 accumulator fragment, but its \
            destination is a %s"
           (fragment_name dst_frag))
      ~remedy:"Declare the destination as an f32 accumulator fragment."
  else if src_frag.frag_use = Accumulator && src_frag.frag_component = Uint32
  then
    scale_add_refusal
      ~name:"coopmat-scale-add-u32-source"
      ~why:
        "CM_scale_add converts an s32 or f32 source to f32; a u32 source is \
         refused, because its values above 2^31 are not s32 values and no \
         conversion for them is defined"
      ~remedy:
        "Accumulate into an s32 fragment (the same bits for a 32-bit wrapping \
         sum) and scale that."
  else if
    src_frag.frag_use <> Accumulator
    || not
         (src_frag.frag_component = Sint32 || src_frag.frag_component = Float32)
  then
    scale_add_refusal
      ~name:"coopmat-scale-add-source"
      ~why:
        (Printf.sprintf
           "CM_scale_add scales an s32 or f32 accumulator fragment, but its \
            source is a %s"
           (fragment_name src_frag))
      ~remedy:"Scale an s32 or f32 accumulator fragment."
  else if
    src_frag.frag_shape <> dst_frag.frag_shape
    || src_frag.frag_scope <> dst_frag.frag_scope
  then
    scale_add_refusal
      ~name:"coopmat-scale-add-shape"
      ~why:
        (Printf.sprintf
           "CM_scale_add needs a source and a destination of the same shape \
            and scope, but they are a %s and a %s"
           (fragment_name src_frag)
           (fragment_name dst_frag))
      ~remedy:"Use fragments of one shape and scope."
  else Sarek_capability.Available

let device_lacks_scale_add ~evidence =
  {
    Sarek_capability.cap_name = "coopmat-scale-add";
    cap_kind = Sarek_capability.Device_optional;
    cap_why =
      Printf.sprintf
        "the device's backend does not lower CM_scale_add (scaled accumulation \
         of a cooperative-matrix fragment) on this device (%s)"
        evidence;
    cap_evidence =
      Sarek_capability.Quoted
        (scale_add_design
       ^ " §3 and §6: offered on HIP gfx1100 only; CUDA, Vulkan, Metal and the \
          backends without cooperative matrices refuse it");
    cap_remedy =
      Some
        "Run the kernel on a HIP gfx1100 device or on the interpreter, or \
         store the fragment and apply the scales with scalar code.";
  }

let scale_add_verdict ~support ~dst_frag ~src_frag =
  match scale_add_operands_verdict ~dst_frag ~src_frag with
  | (Sarek_capability.Unavailable _ | Sarek_capability.Unknown _) as v -> v
  | Sarek_capability.Available -> (
      match support with
      | None ->
          Sarek_capability.Unknown
            "device cooperative-matrix support was not probed, so whether its \
             backend lowers CM_scale_add is unknown"
      | Some s ->
          if s.ds_scale_add then Sarek_capability.Available
          else
            Sarek_capability.Unavailable
              (device_lacks_scale_add ~evidence:s.ds_evidence))
