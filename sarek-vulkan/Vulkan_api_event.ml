(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

open Ctypes
open Vulkan_types
open Vulkan_bindings
open Vulkan_api_base
module Device = Vulkan_api_device

type t = {fence : vk_fence; device : Device.t}

let create_with_device device =
  let fence_info = make vk_fence_create_info in
  setf fence_info fence_create_sType (u32 vk_structure_type_fence_create_info) ;
  setf fence_info fence_create_pNext null ;
  setf fence_info fence_create_flags (Unsigned.UInt32.of_int 0) ;

  let fence = allocate vk_fence vk_null_handle in
  check
    "vkCreateFence"
    (vkCreateFence device.Device.device (addr fence_info) null fence) ;
  {fence = !@fence; device}

let destroy event = vkDestroyFence event.device.Device.device event.fence null

let record _event _stream = ()
(* Fences work differently in Vulkan - submit with fence *)

let synchronize event =
  let fence_ptr = allocate vk_fence event.fence in
  check
    "vkWaitForFences"
    (vkWaitForFences
       event.device.Device.device
       (Unsigned.UInt32.of_int 1)
       fence_ptr
       vk_true
       (Unsigned.UInt64.of_int64 Int64.max_int)) ;
  ignore fence_ptr

let elapsed ~start:_ ~stop:_ = 0.0
(* Would need timestamp queries for real timing *)
