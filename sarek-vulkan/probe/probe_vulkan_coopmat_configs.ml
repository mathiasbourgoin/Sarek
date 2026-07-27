(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * backlog-62 slice 3 — dump every cooperative-matrix configuration each local
 * Vulkan device advertises, through Sarek's own probe rather than a C program.
 *
 * An executable and not a test, for the reason sarek-vulkan/probe/dune already
 * gives: this MEASURES a driver. The gate that must stay green on any runner is
 * sarek-vulkan/test/test_vulkan_coopmat_capability.ml.
 *
 * It exists because the C probe in tools/probes/vulkan_coopmat_probe.c needs
 * Vulkan headers new enough to declare VkComponentTypeKHR, and the newest
 * headers on this workstation predate the KHR promotion. Sarek's ctypes probe
 * pins the enumerants by value and does not, so it can answer the question the
 * C probe cannot be built to ask here.
 ******************************************************************************)

open Sarek_vulkan

let () =
  if not (Vulkan_api.is_available ()) then (
    print_endline "no Vulkan device available" ;
    exit 0) ;
  Vulkan_api_device.init () ;
  let n = Vulkan_api_device.count () in
  Printf.printf "%d Vulkan device(s)\n\n" n ;
  for i = 0 to n - 1 do
    let dev = Vulkan_api_device.get i in
    let caps = Vulkan_plugin_base.Vulkan.Device.capabilities dev in
    Printf.printf "device %d: %s\n" i dev.Vulkan_api_device.name ;
    match caps.Spoc_framework.Framework_sig.coopmat with
    | None -> Printf.printf "  coopmat: UNPROBED\n\n"
    | Some ds ->
        Printf.printf
          "  subgroup size %d, robustBufferAccess %b, advertised %d, \
           represented %d\n"
          ds.Sarek_coopmat.ds_subgroup_size
          ds.Sarek_coopmat.ds_robust_buffer_access
          ds.Sarek_coopmat.ds_advertised_count
          (List.length ds.Sarek_coopmat.ds_configs) ;
        List.iteri
          (fun j cfg ->
            Printf.printf
              "  [%2d] %-44s exact=%b regime=%s\n"
              j
              (Sarek_coopmat.config_name cfg)
              (Sarek_coopmat.accumulation_is_exact cfg)
              (Sarek_coopmat.regime_name (Sarek_coopmat.regime cfg)))
          ds.Sarek_coopmat.ds_configs ;
        print_newline ()
  done
