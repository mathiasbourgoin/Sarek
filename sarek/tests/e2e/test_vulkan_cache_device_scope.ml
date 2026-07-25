(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Regression: the Vulkan kernel cache must be evictable PER DEVICE (#90).
 *
 * Vulkan used to be the worst case of the mixed eviction scope #86 left behind.
 * Its cache key carries the backend-local device index like everyone else's,
 * but it passed no [~device_id] to [Guarded_cache.find_or_build], so the entry
 * was never grouped by device and [Guarded_cache.evict_device] could not reach
 * it - and [Vulkan_api_device.destroy] fired no notification either. Per-device
 * eviction was therefore inexpressible at BOTH layers, which is what #90
 * records. Everything the cache's [destroy] releases (pipeline, layout,
 * descriptor pool and set layout, shader module) belongs to the VkDevice that
 * [destroy] is about to take down, so the entries outlived their VkDevice and
 * would be handed to the next lookup for a recreated index.
 *
 * This is the Vulkan half of test_backend_cache_scope.ml and is a SEPARATE
 * EXECUTABLE for a driver reason, not a stylistic one: on this host (Mesa,
 * radeonsi OpenCL + RADV Vulkan) creating a Vulkan compute pipeline in a
 * process that has already enumerated OpenCL segfaults inside
 * libvulkan_radeon. Pre-existing and unrelated to the cache work -
 * test_static_tag_erasure SIGSEGVs identically on origin/main and passes with
 * OCL_ICD_VENDORS pointed at an empty directory - but it means the two
 * backends cannot share a test binary.
 *
 * Skips - as a distinct Alcotest [SKIP] row, not a green [OK] - when no Vulkan
 * device is enumerated. A Vulkan device that IS present and fails to compile
 * the probe is a failure, never a skip.
 ******************************************************************************)

let source =
  {|#version 450

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(std430, set=0, binding = 0) buffer BufferA {
    float a[];
};

void main() {
    a[gl_GlobalInvocationID.x] = 7.0;
}
|}

(* Registers the Vulkan backend WITHOUT going through
   Test_helpers.Benchmarks.init_backends, which would pull in OpenCL - see the
   header for why that is not survivable in this process. *)
let () = Sarek_vulkan.Vulkan_plugin.init ()

let test_vulkan_cache_is_evictable_per_device () =
  match Spoc_framework_registry.Framework_registry.find_backend "Vulkan" with
  | None ->
      Printf.printf "[SKIP] the Vulkan backend is not registered\n%!" ;
      Alcotest.skip ()
  | Some (module B : Spoc_framework.Framework_sig.BACKEND) ->
      if not (B.is_available ()) then begin
        Printf.printf "[SKIP] no usable Vulkan driver on this host\n%!" ;
        Alcotest.skip ()
      end ;
      B.Device.init () ;
      if B.Device.count () = 0 then begin
        Printf.printf "[SKIP] no Vulkan device enumerated here\n%!" ;
        Alcotest.skip ()
      end ;
      let dev = B.Device.get 0 in
      let index = B.Device.id dev in
      Printf.printf
        "Vulkan: using backend-local device %d (%s)\n%!"
        index
        (B.Device.name dev) ;
      let k1 = B.Kernel.compile_cached dev ~name:"scope_probe" ~source in
      let k2 = B.Kernel.compile_cached dev ~name:"scope_probe" ~source in
      if not (k1 == k2) then
        Alcotest.failf
          "two compile_cached calls with the same key returned different \
           values - the Vulkan cache is not hitting at all, so this test \
           cannot tell eviction from a permanent miss" ;
      (* Backend-local indices collide across backends (OpenCL 0, Vulkan 0, HIP
         0 all exist), and evict_device does not merely drop memoization - it
         aborts in-flight builds for that index. A foreign teardown must not
         touch this entry. *)
      Spoc_framework.Cache_hooks.notify_device_destroy ~backend:"HIP" index ;
      let k3 = B.Kernel.compile_cached dev ~name:"scope_probe" ~source in
      if not (k3 == k1) then
        Alcotest.failf
          "a HIP device-%d teardown evicted the Vulkan entry for the same \
           index - the listener matches on the backend-local index alone"
          index ;
      Spoc_framework.Cache_hooks.notify_device_destroy ~backend:"Vulkan" index ;
      let k4 = B.Kernel.compile_cached dev ~name:"scope_probe" ~source in
      if k4 == k1 then
        Alcotest.failf
          "the Vulkan device-%d teardown did NOT evict the backend cache \
           entry. Guarded_cache.evict_device can only reach entries installed \
           with ~device_id, and Vulkan's find_or_build does not pass it - so \
           the pipeline, layouts, descriptor pool and shader module for a \
           destroyed VkDevice survive it (#90)"
          index ;
      Printf.printf
        "Vulkan: device-%d teardown evicted the cache entry, a foreign \
         backend's did not\n\
         %!"
        index

(* The other half of #90: it is [Vulkan_api_device.destroy] that must FIRE the
   notification, and nothing did before. Driven through the real destroy path
   rather than a hand-fired notify, because the ordering is the property -
   everything the cache releases belongs to the VkDevice being destroyed, so the
   eviction has to happen while it is still alive.

   Observable without reaching into the cache: [destroy] empties Vulkan's
   [device_cache], so [get 0] afterwards builds a NEW VkDevice under the same
   index. A surviving cache entry would then be served to it, carrying a
   pipeline belonging to the destroyed VkDevice. Physical identity distinguishes
   the two.

   Runs after the case above and destroys the device, so it is declared last. *)
module VDev = Sarek_vulkan.Vulkan_api.Device

let test_device_destroy_evicts_the_cache () =
  match Spoc_framework_registry.Framework_registry.find_backend "Vulkan" with
  | None ->
      Printf.printf "[SKIP] the Vulkan backend is not registered\n%!" ;
      Alcotest.skip ()
  | Some (module B : Spoc_framework.Framework_sig.BACKEND) ->
      if not (B.is_available ()) then begin
        Printf.printf "[SKIP] no usable Vulkan driver on this host\n%!" ;
        Alcotest.skip ()
      end ;
      B.Device.init () ;
      if B.Device.count () = 0 then begin
        Printf.printf "[SKIP] no Vulkan device enumerated here\n%!" ;
        Alcotest.skip ()
      end ;
      let dev = VDev.get 0 in
      let k1 =
        Sarek_vulkan.Vulkan_api.Kernel.compile_cached
          dev
          ~name:"scope_probe"
          ~source
      in
      if
        not
          (Sarek_vulkan.Vulkan_api.Kernel.compile_cached
             dev
             ~name:"scope_probe"
             ~source
          == k1)
      then
        Alcotest.failf
          "the Vulkan cache is not hitting before the destroy, so this test \
           cannot tell eviction from a permanent miss" ;
      VDev.destroy dev ;
      let dev' = VDev.get 0 in
      let k2 =
        Sarek_vulkan.Vulkan_api.Kernel.compile_cached
          dev'
          ~name:"scope_probe"
          ~source
      in
      if k2 == k1 then
        Alcotest.failf
          "Vulkan_api_device.destroy left the kernel cache entry in place: the \
           recreated device 0 was handed a pipeline, layouts, descriptor pool \
           and shader module belonging to the VkDevice that was just destroyed \
           (#90 - destroy fires no Cache_hooks.notify_device_destroy)" ;
      Printf.printf "Vulkan: Vulkan_api_device.destroy evicted the cache\n%!"

let () =
  Alcotest.run
    "Vulkan cache device scope"
    [
      ( "per-device eviction",
        [
          Alcotest.test_case
            "Vulkan kernel cache is evictable per device"
            `Quick
            test_vulkan_cache_is_evictable_per_device;
          Alcotest.test_case
            "Vulkan_api_device.destroy fires the notification"
            `Quick
            test_device_destroy_evicts_the_cache;
        ] );
    ]
