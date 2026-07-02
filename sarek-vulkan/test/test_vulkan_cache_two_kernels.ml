(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Hardware-verified non-vacuous test for the compile-cache fix (S3a item 5).
 *
 * Before this change, both Vulkan_api_kernel's in-memory pipeline cache
 * (~314-320) and the on-disk Framework_cache SPIR-V key (~99-104) were keyed
 * on device + source digest only, omitting the kernel/entry name. GLSL
 * compute shaders can only expose a single SPIR-V entry point per compiled
 * module (glslangValidator requires the GLSL-level function to be literally
 * named "main"; the [-e] flag only renames the SPIR-V-level entry symbol),
 * so two DIFFERENT kernel names compiled from the SAME GLSL source text
 * (byte-identical - a scenario that can legitimately arise from Sarek
 * codegen when two distinct kernel definitions produce structurally
 * identical bodies) previously collided under one cache key: the second
 * [compile_cached] call would silently return the first kernel's pipeline.
 *
 * This test drives the real [Vulkan_api_kernel.compile_cached] entry point
 * twice with identical source but different names, executes both compiled
 * pipelines on the real device, and asserts they are independent cache
 * entries. Requires a real Vulkan device; skips (does not fail) when none is
 * present so the suite stays green in CI environments without a GPU.
 ******************************************************************************)

open Sarek_vulkan
module Device = Vulkan_api_device
module Memory = Vulkan_api_memory
module Kernel = Vulkan_api_kernel

(* GLSL requires the entry function to be literally named "main"; the kernel
   "name" passed to compile_cached is purely a cache/label distinguisher
   here, which is exactly the scenario the fix targets: two different names,
   byte-identical source. *)
let shared_source =
  {|
#version 450
layout(local_size_x = 1) in;
layout(std430, binding = 0) buffer Buf { float data[]; };
void main() {
    data[0] = 1.0;
}
|}

let compile_and_read_result device ~name =
  let compiled = Kernel.compile_cached device ~name ~source:shared_source in
  let buf = Memory.alloc device 1 Bigarray.float32 in
  let args = Kernel.create_args () in
  Kernel.set_arg_buffer args 0 buf ;
  let dims = Spoc_framework.Framework_sig.dims_1d 1 in
  Kernel.launch compiled ~args ~grid:dims ~block:dims ~shared_mem:0 ~stream:None ;
  Device.synchronize device ;
  let result = Bigarray.Array1.create Bigarray.float32 Bigarray.c_layout 1 in
  Memory.device_to_host ~src:buf ~dst:result ;
  Memory.free buf ;
  (compiled, result.{0})

let test_two_kernel_names_one_source_do_not_alias () =
  if not (Vulkan_api.is_available ()) then
    Printf.printf
      "[SKIP] No Vulkan device available - skipping hardware cache test\n%!"
  else begin
    Device.init () ;
    let device = Device.get 0 in
    let compiled_a, result_a =
      compile_and_read_result device ~name:"kernel_a"
    in
    let compiled_b, result_b =
      compile_and_read_result device ~name:"kernel_b"
    in
    (* This is the regression the fix targets: before it, the cache key was
       device + source digest only, so [compile_cached device ~name:"kernel_b"
       ~source:shared_source] would hit the entry already inserted under
       "kernel_a" and return that SAME physical record - [compiled_a] and
       [compiled_b] would be [==] and [compiled_b.name] would read
       "kernel_a". After the fix, the name is part of the key, so each name
       gets its own compiled pipeline even though the GLSL text is
       byte-identical. *)
    Alcotest.(check bool)
      "kernel_a and kernel_b get distinct pipeline objects, not an aliased \
       cache hit"
      true
      (compiled_a != compiled_b) ;
    Alcotest.(check string)
      "compiled_a keeps its own name"
      "kernel_a"
      compiled_a.Kernel.name ;
    Alcotest.(check string)
      "compiled_b keeps its own name, not aliased to kernel_a's"
      "kernel_b"
      compiled_b.Kernel.name ;
    (* Both bodies are identical GLSL ("main" cannot vary by name), so both
       correctly write 1.0 - this proves the fix does not break the real
       compile -> cache -> launch pipeline end-to-end on hardware. *)
    Alcotest.(check (float 0.0001))
      "kernel_a executes correctly through the cache"
      1.0
      result_a ;
    Alcotest.(check (float 0.0001))
      "kernel_b executes correctly through its own distinct cache entry"
      1.0
      result_b
  end

let () =
  Alcotest.run
    "Vulkan_cache_two_kernels"
    [
      ( "compile_cache",
        [
          Alcotest.test_case
            "two kernel names sharing one GLSL source do not alias"
            `Quick
            test_two_kernel_names_one_source_do_not_alias;
        ] );
    ]
