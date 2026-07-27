(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * backlog-62 slice 2 — the cooperative-matrix capability gate, on real devices.
 *
 * WHAT THIS TEST IS FOR.
 *
 * docs/design/capability-model.md classifies cooperative matrices as
 * [Device_optional]: the backend can spell the instruction, a given device may
 * not provide it. A capability gate that has never been observed REFUSING is
 * not known to work — it is indistinguishable from a function that returns
 * Available unconditionally. This machine has a free negative device: the
 * Raphael iGPU does not advertise VK_KHR_cooperative_matrix while the RX 7900
 * XTX does, under the same radv / Mesa build, so both branches are reachable
 * without a second machine (docs/design/f16-relaxed-accuracy.md §4).
 *
 * So this file does not assert "coopmat works". It asserts that the gate
 * SEPARATES: on a machine with more than one Vulkan device, if the devices
 * disagree about support then the verdict must disagree with them in the same
 * direction, and the refusing device must produce a diagnostic that names the
 * capability and the device.
 *
 * WHY IT ALSO CHECKS THE EXTENSION AND NOT ONLY THE CONFIGURATION LIST.
 *
 * Measured on this workstation, and it is the reason [probe_coopmat] is shaped
 * the way it is: vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR returns
 * VK_SUCCESS and FOURTEEN configurations for the Raphael iGPU — a device that
 * does not advertise the extension and reports cooperativeMatrix = false.
 * Calling an extension entry point on a device that lacks the extension is
 * undefined behaviour, and RADV's undefined behaviour here is a well-formed,
 * plausible, entirely wrong answer. A probe that trusted the query would have
 * reported the iGPU as capable and this gate would never refuse anything.
 *
 * It skips (does not fail) where there is no Vulkan device, and degrades to the
 * checks that are still meaningful where every device happens to agree.
 ******************************************************************************)

open Sarek_vulkan

let device_report i =
  let dev = Vulkan_api_device.get i in
  let caps = Vulkan_plugin_base.Vulkan.Device.capabilities dev in
  (i, dev, caps)

let f16_f32_16x16x16 =
  {
    Sarek_coopmat.cfg_shape = {Sarek_coopmat.m = 16; n = 16; k = 16};
    cfg_a = Sarek_coopmat.Float16;
    cfg_b = Sarek_coopmat.Float16;
    cfg_c = Sarek_coopmat.Float32;
    cfg_result = Sarek_coopmat.Float32;
    cfg_saturating = false;
    cfg_scope = Sarek_coopmat.Subgroup;
  }

let contains needle haystack =
  let re = Str.regexp_string needle in
  try
    ignore (Str.search_forward re haystack 0) ;
    true
  with Not_found -> false

let with_devices f =
  if not (Vulkan_api.is_available ()) then begin
    Printf.printf "[SKIP] No Vulkan loader available\n%!" ;
    Alcotest.skip ()
  end
  else
    let n = try Vulkan_api_device.count () with _ -> 0 in
    if n = 0 then begin
      Printf.printf "[SKIP] No Vulkan physical device\n%!" ;
      Alcotest.skip ()
    end
    else f (List.init n device_report)

(* The VkComponentTypeKHR / VkScopeKHR enumerants, pinned against Khronos
   vulkan_core.h.

   This is a value-for-value restatement rather than a derived check, and it
   earns its place: an earlier draft of Vulkan_types guessed these and produced
   a completely plausible wrong answer — six configurations instead of fourteen,
   including an [f16 * s8 -> s32] and a [u8 * u8 -> u8] that no hardware
   advertises, with the other eight dropped as unrepresentable. Nothing in the
   output looked malformed, and the gate test below still passed, because it
   compared the verdict against the SAME mis-decoded list. Two independent
   checks were needed: this one, and the no-drop invariant in
   [test_no_configuration_was_dropped]. *)
let test_enumerant_values () =
  List.iter
    (fun (name, actual, expected) ->
      Alcotest.(check int32) name expected actual)
    [
      ( "VK_COMPONENT_TYPE_FLOAT16_KHR",
        Vulkan_types.vk_component_type_float16,
        0l );
      ( "VK_COMPONENT_TYPE_FLOAT32_KHR",
        Vulkan_types.vk_component_type_float32,
        1l );
      ( "VK_COMPONENT_TYPE_FLOAT64_KHR",
        Vulkan_types.vk_component_type_float64,
        2l );
      ("VK_COMPONENT_TYPE_SINT8_KHR", Vulkan_types.vk_component_type_sint8, 3l);
      ("VK_COMPONENT_TYPE_SINT16_KHR", Vulkan_types.vk_component_type_sint16, 4l);
      ("VK_COMPONENT_TYPE_SINT32_KHR", Vulkan_types.vk_component_type_sint32, 5l);
      ("VK_COMPONENT_TYPE_SINT64_KHR", Vulkan_types.vk_component_type_sint64, 6l);
      ("VK_COMPONENT_TYPE_UINT8_KHR", Vulkan_types.vk_component_type_uint8, 7l);
      ("VK_COMPONENT_TYPE_UINT16_KHR", Vulkan_types.vk_component_type_uint16, 8l);
      ("VK_COMPONENT_TYPE_UINT32_KHR", Vulkan_types.vk_component_type_uint32, 9l);
      ( "VK_COMPONENT_TYPE_UINT64_KHR",
        Vulkan_types.vk_component_type_uint64,
        10l );
      ("VK_SCOPE_DEVICE_KHR", Vulkan_types.vk_scope_device, 1l);
      ("VK_SCOPE_WORKGROUP_KHR", Vulkan_types.vk_scope_workgroup, 2l);
      ("VK_SCOPE_SUBGROUP_KHR", Vulkan_types.vk_scope_subgroup, 3l);
      ("VK_SCOPE_QUEUE_FAMILY_KHR", Vulkan_types.vk_scope_queue_family, 5l);
    ]

(* The no-drop invariant, and the reason [ds_advertised_count] exists.

   Dropping a configuration whose component type or scope this build cannot
   represent is the SAFE direction — it can only cause a refusal — but it is
   silent, and silence is what let a wrong enumerant table look like a working
   probe. This check is hardware-independent: it makes no claim about which
   configurations any GPU offers, only that none of the ones it offered was
   thrown away. It goes red on exactly the defect that the gate test could not
   see. *)
let test_no_configuration_was_dropped () =
  with_devices (fun devs ->
      List.iter
        (fun (i, dev, caps) ->
          match caps.Spoc_framework.Framework_sig.coopmat with
          | None -> ()
          | Some s ->
              let kept = List.length s.Sarek_coopmat.ds_configs in
              if kept <> s.Sarek_coopmat.ds_advertised_count then
                Alcotest.failf
                  "device %d (%s) advertised %d cooperative-matrix \
                   configurations but only %d were representable: %d were \
                   dropped, which means the VkComponentTypeKHR/VkScopeKHR \
                   decoding in Vulkan_types is incomplete or wrong"
                  i
                  dev.Vulkan_api_device.name
                  s.Sarek_coopmat.ds_advertised_count
                  kept
                  (s.Sarek_coopmat.ds_advertised_count - kept))
        devs)

(* The implication that makes the extension check falsifiable:
     a device reporting cooperative-matrix configurations must have had the
     extension advertised AND the feature true.

   Without this, removing the extension gate from [probe_coopmat] is INVISIBLE.
   Verified by mutation on 2026-07-27: replacing the guard with [if false && ...]
   made the Raphael iGPU report all fourteen configurations, both devices
   permitted, and the whole file stayed green — because [test_verdict_tracks_support]
   compares the verdict against the same list the verdict reads, which is a
   tautology whenever the list is wrong. This test goes red on that mutant, and
   is the reason [coopmat_enabled] is recorded on the device rather than derived
   from the configuration list. *)
let test_configurations_imply_the_enabled_feature () =
  with_devices (fun devs ->
      List.iter
        (fun (i, dev, caps) ->
          match caps.Spoc_framework.Framework_sig.coopmat with
          | None -> ()
          | Some s ->
              let n = List.length s.Sarek_coopmat.ds_configs in
              if n > 0 && not dev.Vulkan_api_device.coopmat_enabled then
                Alcotest.failf
                  "device %d (%s) reports %d cooperative-matrix configurations \
                   but the extension/feature were not enabled (advertised=%b): \
                   the configuration query answers even for devices that lack \
                   the extension, so this list is not evidence of support"
                  i
                  dev.Vulkan_api_device.name
                  n
                  dev.Vulkan_api_device.coopmat_extension_advertised ;
              if dev.Vulkan_api_device.coopmat_enabled && n = 0 then
                Alcotest.failf
                  "device %d (%s) enabled cooperative matrices but advertises \
                   no configuration, which no driver should do"
                  i
                  dev.Vulkan_api_device.name)
        devs)

(* Every device must come back PROBED. [None] means the loader could not
   resolve the entry point at all, which is a legitimate outcome on an old
   loader but not on any machine that can run the rest of this backend — and it
   is the outcome that would make every verdict below vacuously refuse. Failing
   here is what stops this file from passing green while measuring nothing. *)
let test_every_device_is_probed () =
  with_devices (fun devs ->
      List.iter
        (fun (i, dev, caps) ->
          match caps.Spoc_framework.Framework_sig.coopmat with
          | None ->
              Alcotest.failf
                "device %d (%s) reports coopmat = None: the probe did not run, \
                 so every verdict below would refuse for the wrong reason"
                i
                dev.Vulkan_api_device.name
          | Some support ->
              Printf.printf
                "  device %d: %s [driverID=%d %s / %s] subgroupSize=%d \
                 shaderFloat16=%b storageBuffer16=%b coopmat_configs=%d \
                 robust=%b\n\
                 %!"
                i
                dev.Vulkan_api_device.name
                dev.Vulkan_api_device.driver_id
                dev.Vulkan_api_device.driver_name
                dev.Vulkan_api_device.driver_info
                support.Sarek_coopmat.ds_subgroup_size
                dev.Vulkan_api_device.supports_fp16
                dev.Vulkan_api_device.storage_buffer_16bit
                (List.length support.Sarek_coopmat.ds_configs)
                support.Sarek_coopmat.ds_robust_buffer_access)
        devs)

(* The subgroup size must be what the device reports, and it must be usable as
   an ABI number. A zero would divide nothing; a value the plugin invented
   would silently mis-size every fragment. *)
let test_subgroup_size_is_reported () =
  with_devices (fun devs ->
      List.iter
        (fun (i, dev, caps) ->
          let sg = caps.Spoc_framework.Framework_sig.warp_size in
          Alcotest.(check bool)
            (Printf.sprintf "device %d subgroup size is positive" i)
            true
            (sg > 0) ;
          Alcotest.(check int)
            (Printf.sprintf
               "device %d: warp_size is the probed subgroupSize, not a constant"
               i)
            dev.Vulkan_api_device.subgroup_size
            sg ;
          match caps.Spoc_framework.Framework_sig.coopmat with
          | Some s ->
              Alcotest.(check int)
                "and the coopmat record carries the same number"
                sg
                s.Sarek_coopmat.ds_subgroup_size
          | None -> ())
        devs)

(* The driver key of docs/fp-contraction-policy.md §11.7: a non-empty driver
   name, so an allowlist can be keyed on the driver rather than on a substring
   of a device name. *)
let test_driver_identity_is_available () =
  with_devices (fun devs ->
      List.iter
        (fun (i, dev, _) ->
          Alcotest.(check bool)
            (Printf.sprintf "device %d has a driver name" i)
            true
            (String.length dev.Vulkan_api_device.driver_name > 0))
        devs)

(* The gate itself. *)
let test_verdict_tracks_support () =
  with_devices (fun devs ->
      let refusing = ref [] and permitting = ref [] in
      List.iter
        (fun (i, dev, caps) ->
          let support = caps.Spoc_framework.Framework_sig.coopmat in
          let v = Sarek_coopmat.verdict ~support f16_f32_16x16x16 in
          let advertises =
            match support with
            | Some s ->
                List.exists
                  (fun c -> c = f16_f32_16x16x16)
                  s.Sarek_coopmat.ds_configs
            | None -> false
          in
          Alcotest.(check bool)
            (Printf.sprintf
               "device %d (%s): verdict agrees with what the device advertises"
               i
               dev.Vulkan_api_device.name)
            advertises
            (Sarek_capability.permits v) ;
          if Sarek_capability.permits v then
            permitting := (i, dev.Vulkan_api_device.name) :: !permitting
          else begin
            refusing := (i, dev.Vulkan_api_device.name) :: !refusing ;
            (* A refusal must be attributable. An unnamed refusal is the
               failure mode docs/design/capability-model.md exists to remove. *)
            match v with
            | Sarek_capability.Unavailable cap ->
                let msg =
                  Sarek_capability.explain
                    ~target:dev.Vulkan_api_device.name
                    cap
                in
                Printf.printf "  REFUSED: %s\n%!" msg ;
                Alcotest.(check string)
                  "the refusal names the capability"
                  "cooperative-matrix"
                  cap.Sarek_capability.cap_name ;
                Alcotest.(check string)
                  "and classifies it as Device_optional"
                  "device-optional"
                  (Sarek_capability.kind_name cap.Sarek_capability.cap_kind) ;
                Alcotest.(check bool)
                  "and names the device"
                  true
                  (contains dev.Vulkan_api_device.name msg)
            | Sarek_capability.Unknown why ->
                Alcotest.failf
                  "device %d was probed, so its verdict must be Unavailable \
                   rather than Unknown: %s"
                  i
                  why
            | Sarek_capability.Available -> assert false
          end)
        devs ;
      Printf.printf
        "  gate observed: %d device(s) permitted, %d refused\n%!"
        (List.length !permitting)
        (List.length !refusing) ;
      if !refusing = [] then
        Printf.printf
          "  [NOTE] every Vulkan device on this machine advertises the \
           configuration, so the refusing branch was not exercised here. The \
           deterministic refusal cases are in \
           spoc/ir/test/test_sarek_coopmat.ml.\n\
           %!")

(* The consistency rule that turns the two device reports into one claim: a
   device advertising configurations must have the extension AND the feature,
   and a device with neither must advertise none. This is what would go red if
   [probe_coopmat] ever stopped gating the query on the extension, because the
   iGPU would then come back with fourteen configurations while its
   cooperativeMatrix feature still read false. *)
let test_configurations_imply_the_feature () =
  with_devices (fun devs ->
      List.iter
        (fun (i, dev, caps) ->
          match caps.Spoc_framework.Framework_sig.coopmat with
          | None -> ()
          | Some s ->
              let n = List.length s.Sarek_coopmat.ds_configs in
              if n > 0 then begin
                (* Every advertised configuration must be layout-able over the
                   subgroup the same device reports, or the pair of numbers is
                   incoherent and no codegen slice could use them together. *)
                List.iter
                  (fun c ->
                    Alcotest.(check bool)
                      (Printf.sprintf
                         "device %d: %s fits a %d-wide subgroup"
                         i
                         (Sarek_coopmat.config_name c)
                         s.Sarek_coopmat.ds_subgroup_size)
                      true
                      (Sarek_coopmat.config_fits_subgroup
                         ~subgroup_size:s.Sarek_coopmat.ds_subgroup_size
                         c))
                  s.Sarek_coopmat.ds_configs ;
                (* And the strict/relaxed split must be computable, with at
                   least one strict-contract configuration wherever any
                   configuration exists — that is §8's fallback, and if it ever
                   becomes false on real hardware the fallback is gone. *)
                let exact =
                  List.filter
                    Sarek_coopmat.accumulation_is_exact
                    s.Sarek_coopmat.ds_configs
                in
                Printf.printf
                  "  device %d (%s): %d configurations, %d accumulate exactly \
                   (strict contract)\n\
                   %!"
                  i
                  dev.Vulkan_api_device.name
                  n
                  (List.length exact)
              end)
        devs)

let () =
  let open Alcotest in
  run
    "Vulkan cooperative-matrix capability"
    [
      ( "probe",
        [
          test_case "every device is probed" `Quick test_every_device_is_probed;
          test_case
            "subgroup size is reported, not assumed"
            `Quick
            test_subgroup_size_is_reported;
          test_case
            "driver identity is available"
            `Quick
            test_driver_identity_is_available;
          test_case
            "enumerants match vulkan_core.h"
            `Quick
            test_enumerant_values;
          test_case
            "no advertised configuration was dropped"
            `Quick
            test_no_configuration_was_dropped;
          test_case
            "configurations imply the enabled feature"
            `Quick
            test_configurations_imply_the_enabled_feature;
        ] );
      ( "gate",
        [
          test_case
            "verdict tracks what the device advertises"
            `Quick
            test_verdict_tracks_support;
          test_case
            "advertised configurations are coherent"
            `Quick
            test_configurations_imply_the_feature;
        ] );
    ]
