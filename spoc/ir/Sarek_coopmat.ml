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

let device_lacks_config cfg =
  {
    Sarek_capability.cap_name = "cooperative-matrix";
    (* Device_optional, per docs/design/capability-model.md §2: the backend can
       spell the instruction, a given device may not provide it. The local box
       has one device that does and one that does not under the SAME driver,
       which is why a driver-keyed or backend-keyed refusal would be wrong
       here. *)
    cap_kind = Sarek_capability.Device_optional;
    cap_why =
      Printf.sprintf
        "the device advertises no cooperative-matrix configuration matching %s \
         (VK_KHR_cooperative_matrix absent, its cooperativeMatrix feature \
         false, or no advertised configuration with these dimensions and \
         component types)"
        (config_name cfg);
    cap_evidence =
      Sarek_capability.Measured
        "AMD Radeon RX 7900 XTX (RADV NAVI31) advertises \
         VK_KHR_cooperative_matrix revision 2 with 14 configurations; the AMD \
         Ryzen 9 7950X iGPU (RADV RAPHAEL_MENDOCINO) does not advertise the \
         extension and reports cooperativeMatrix = false, under the same radv \
         / Mesa 26.1.4-arch3.1 / Vulkan 1.4.354";
    cap_remedy =
      Some
        "Use an ordinary multiply-accumulate kernel, or select a device that \
         advertises this cooperative-matrix configuration.";
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
      else Sarek_capability.Unavailable (device_lacks_config cfg)
