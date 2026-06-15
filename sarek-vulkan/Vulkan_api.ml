(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

open Ctypes
open Vulkan_bindings

exception Vk_result_error = Vulkan_api_base.Vk_result_error

let memcpy = Vulkan_api_base.memcpy

let u32 = Vulkan_api_base.u32

let check = Vulkan_api_base.check

let compile_glsl_to_spirv_cli = Vulkan_api_base.compile_glsl_to_spirv_cli

let compile_glsl_to_spirv = Vulkan_api_base.compile_glsl_to_spirv

let glslang_available = Vulkan_api_base.glslang_available

module Device = Vulkan_api_device
module Memory = Vulkan_api_memory
module Stream = Vulkan_api_stream
module Event = Vulkan_api_event
module Kernel = Vulkan_api_kernel

(** {1 Utility Functions} *)

let vulkan_version () =
  let ver = allocate uint32_t (Unsigned.UInt32.of_int 0) in
  let _ = vkEnumerateInstanceVersion ver in
  let v = Unsigned.UInt32.to_int !@ver in
  (v lsr 22, (v lsr 12) land 0x3FF, v land 0xFFF)

let is_available () =
  if not (Vulkan_bindings.is_available ()) then false
  else if (not (glslang_available ())) && not (Shaderc.is_available ()) then begin
    Spoc_core.Log.debug
      Spoc_core.Log.Device
      "Vulkan: neither glslangValidator nor libshaderc found" ;
    false
  end
  else
    try
      Device.init () ;
      Device.count () > 0
    with _ -> false
