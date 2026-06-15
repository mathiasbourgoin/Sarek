(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Vulkan API - Ctypes Bindings
 *
 * Direct FFI bindings to Vulkan API via ctypes-foreign.
 * All bindings are lazy - they only dlopen the library when first used.
 * This allows the module to be linked even on systems without Vulkan.
 ******************************************************************************)

open Ctypes
open Foreign
open Vulkan_types

(** {1 Library Loading} *)

(** Load Vulkan library dynamically (lazy) *)
let vulkan_lib : Dl.library option Lazy.t =
  lazy
    (try
       Some
         (Dl.dlopen
            ~filename:"libvulkan.so.1"
            ~flags:[Dl.RTLD_LAZY; Dl.RTLD_GLOBAL])
     with _ -> (
       try
         Some
           (Dl.dlopen
              ~filename:"libvulkan.so"
              ~flags:[Dl.RTLD_LAZY; Dl.RTLD_GLOBAL])
       with _ -> (
         try
           Some
             (Dl.dlopen
                ~filename:"libvulkan.1.dylib"
                ~flags:[Dl.RTLD_LAZY; Dl.RTLD_GLOBAL])
         with _ -> (
           try
             Some
               (Dl.dlopen
                  ~filename:"vulkan-1.dll"
                  ~flags:[Dl.RTLD_LAZY; Dl.RTLD_GLOBAL])
           with _ -> None))))

(** Check if Vulkan library is available *)
let is_available () =
  match Lazy.force vulkan_lib with Some _ -> true | None -> false

(** Get Vulkan library, raising if not available *)
let get_vulkan_lib () =
  match Lazy.force vulkan_lib with
  | Some lib -> lib
  | None ->
      Vulkan_error.raise_error (Vulkan_error.library_not_found "vulkan" [])

(** Create a lazy foreign binding to Vulkan API *)
let foreign_vk_lazy name typ = lazy (foreign ~from:(get_vulkan_lib ()) name typ)

(** {1 Instance Functions} *)

let vkCreateInstance_lazy =
  foreign_vk_lazy
    "vkCreateInstance"
    (ptr vk_instance_create_info
    @-> ptr void @-> ptr vk_instance_ptr @-> returning vk_result)

let vkCreateInstance info alloc inst =
  Lazy.force vkCreateInstance_lazy info alloc inst

let vkDestroyInstance_lazy =
  foreign_vk_lazy
    "vkDestroyInstance"
    (vk_instance_ptr @-> ptr void @-> returning void)

let vkDestroyInstance inst alloc = Lazy.force vkDestroyInstance_lazy inst alloc

let vkEnumeratePhysicalDevices_lazy =
  foreign_vk_lazy
    "vkEnumeratePhysicalDevices"
    (vk_instance_ptr @-> ptr uint32_t @-> ptr vk_physical_device_ptr
   @-> returning vk_result)

let vkEnumeratePhysicalDevices inst count devs =
  Lazy.force vkEnumeratePhysicalDevices_lazy inst count devs

(** {1 Physical Device Functions} *)

let vkGetPhysicalDeviceProperties_lazy =
  foreign_vk_lazy
    "vkGetPhysicalDeviceProperties"
    (vk_physical_device_ptr
    @-> ptr vk_physical_device_properties
    @-> returning void)

let vkGetPhysicalDeviceProperties dev props =
  Lazy.force vkGetPhysicalDeviceProperties_lazy dev props

let vkGetPhysicalDeviceQueueFamilyProperties_lazy =
  foreign_vk_lazy
    "vkGetPhysicalDeviceQueueFamilyProperties"
    (vk_physical_device_ptr @-> ptr uint32_t
    @-> ptr vk_queue_family_properties
    @-> returning void)

let vkGetPhysicalDeviceQueueFamilyProperties dev count props =
  Lazy.force vkGetPhysicalDeviceQueueFamilyProperties_lazy dev count props

let vkGetPhysicalDeviceMemoryProperties_lazy =
  foreign_vk_lazy
    "vkGetPhysicalDeviceMemoryProperties"
    (vk_physical_device_ptr
    @-> ptr vk_physical_device_memory_properties
    @-> returning void)

let vkGetPhysicalDeviceMemoryProperties dev props =
  Lazy.force vkGetPhysicalDeviceMemoryProperties_lazy dev props

(** {1 Device Functions} *)

let vkCreateDevice_lazy =
  foreign_vk_lazy
    "vkCreateDevice"
    (vk_physical_device_ptr @-> ptr vk_device_create_info @-> ptr void
   @-> ptr vk_device_ptr @-> returning vk_result)

let vkCreateDevice phys_dev info alloc dev =
  Lazy.force vkCreateDevice_lazy phys_dev info alloc dev

let vkDestroyDevice_lazy =
  foreign_vk_lazy
    "vkDestroyDevice"
    (vk_device_ptr @-> ptr void @-> returning void)

let vkDestroyDevice dev alloc = Lazy.force vkDestroyDevice_lazy dev alloc

let vkGetDeviceQueue_lazy =
  foreign_vk_lazy
    "vkGetDeviceQueue"
    (vk_device_ptr @-> uint32_t @-> uint32_t @-> ptr vk_queue_ptr
   @-> returning void)

let vkGetDeviceQueue dev family idx queue =
  Lazy.force vkGetDeviceQueue_lazy dev family idx queue

let vkDeviceWaitIdle_lazy =
  foreign_vk_lazy "vkDeviceWaitIdle" (vk_device_ptr @-> returning vk_result)

let vkDeviceWaitIdle dev = Lazy.force vkDeviceWaitIdle_lazy dev

(** {1 Memory Functions} *)

let vkAllocateMemory_lazy =
  foreign_vk_lazy
    "vkAllocateMemory"
    (vk_device_ptr
    @-> ptr vk_memory_allocate_info
    @-> ptr void @-> ptr vk_device_memory @-> returning vk_result)

let vkAllocateMemory dev info alloc mem =
  Lazy.force vkAllocateMemory_lazy dev info alloc mem

let vkFreeMemory_lazy =
  foreign_vk_lazy
    "vkFreeMemory"
    (vk_device_ptr @-> vk_device_memory @-> ptr void @-> returning void)

let vkFreeMemory dev mem alloc = Lazy.force vkFreeMemory_lazy dev mem alloc

let vkMapMemory_lazy =
  foreign_vk_lazy
    "vkMapMemory"
    (vk_device_ptr @-> vk_device_memory @-> vk_device_size @-> vk_device_size
   @-> vk_flags
    @-> ptr (ptr void)
    @-> returning vk_result)

let vkMapMemory dev mem offset size flags data =
  Lazy.force vkMapMemory_lazy dev mem offset size flags data

let vkUnmapMemory_lazy =
  foreign_vk_lazy
    "vkUnmapMemory"
    (vk_device_ptr @-> vk_device_memory @-> returning void)

let vkUnmapMemory dev mem = Lazy.force vkUnmapMemory_lazy dev mem

(** {1 Buffer Functions} *)

let vkCreateBuffer_lazy =
  foreign_vk_lazy
    "vkCreateBuffer"
    (vk_device_ptr @-> ptr vk_buffer_create_info @-> ptr void @-> ptr vk_buffer
   @-> returning vk_result)

let vkCreateBuffer dev info alloc buf =
  Lazy.force vkCreateBuffer_lazy dev info alloc buf

let vkDestroyBuffer_lazy =
  foreign_vk_lazy
    "vkDestroyBuffer"
    (vk_device_ptr @-> vk_buffer @-> ptr void @-> returning void)

let vkDestroyBuffer dev buf alloc =
  Lazy.force vkDestroyBuffer_lazy dev buf alloc

let vkGetBufferMemoryRequirements_lazy =
  foreign_vk_lazy
    "vkGetBufferMemoryRequirements"
    (vk_device_ptr @-> vk_buffer @-> ptr vk_memory_requirements
   @-> returning void)

let vkGetBufferMemoryRequirements dev buf reqs =
  Lazy.force vkGetBufferMemoryRequirements_lazy dev buf reqs

let vkBindBufferMemory_lazy =
  foreign_vk_lazy
    "vkBindBufferMemory"
    (vk_device_ptr @-> vk_buffer @-> vk_device_memory @-> vk_device_size
   @-> returning vk_result)

let vkBindBufferMemory dev buf mem offset =
  Lazy.force vkBindBufferMemory_lazy dev buf mem offset

(** {1 Shader Module Functions} *)

let vkCreateShaderModule_lazy =
  foreign_vk_lazy
    "vkCreateShaderModule"
    (vk_device_ptr
    @-> ptr vk_shader_module_create_info
    @-> ptr void @-> ptr vk_shader_module @-> returning vk_result)

let vkCreateShaderModule dev info alloc module_ =
  Lazy.force vkCreateShaderModule_lazy dev info alloc module_

let vkDestroyShaderModule_lazy =
  foreign_vk_lazy
    "vkDestroyShaderModule"
    (vk_device_ptr @-> vk_shader_module @-> ptr void @-> returning void)

let vkDestroyShaderModule dev module_ alloc =
  Lazy.force vkDestroyShaderModule_lazy dev module_ alloc

(** {1 Pipeline Functions} *)

let vkCreateComputePipelines_lazy =
  foreign_vk_lazy
    "vkCreateComputePipelines"
    (vk_device_ptr @-> vk_pipeline_cache @-> uint32_t
    @-> ptr vk_compute_pipeline_create_info
    @-> ptr void @-> ptr vk_pipeline @-> returning vk_result)

let vkCreateComputePipelines dev cache count infos alloc pipelines =
  Lazy.force vkCreateComputePipelines_lazy dev cache count infos alloc pipelines

let vkDestroyPipeline_lazy =
  foreign_vk_lazy
    "vkDestroyPipeline"
    (vk_device_ptr @-> vk_pipeline @-> ptr void @-> returning void)

let vkDestroyPipeline dev pipeline alloc =
  Lazy.force vkDestroyPipeline_lazy dev pipeline alloc

let vkCreatePipelineLayout_lazy =
  foreign_vk_lazy
    "vkCreatePipelineLayout"
    (vk_device_ptr
    @-> ptr vk_pipeline_layout_create_info
    @-> ptr void @-> ptr vk_pipeline_layout @-> returning vk_result)

let vkCreatePipelineLayout dev info alloc layout =
  Lazy.force vkCreatePipelineLayout_lazy dev info alloc layout

let vkDestroyPipelineLayout_lazy =
  foreign_vk_lazy
    "vkDestroyPipelineLayout"
    (vk_device_ptr @-> vk_pipeline_layout @-> ptr void @-> returning void)

let vkDestroyPipelineLayout dev layout alloc =
  Lazy.force vkDestroyPipelineLayout_lazy dev layout alloc

(** {1 Descriptor Functions} *)

let vkCreateDescriptorSetLayout_lazy =
  foreign_vk_lazy
    "vkCreateDescriptorSetLayout"
    (vk_device_ptr
    @-> ptr vk_descriptor_set_layout_create_info
    @-> ptr void
    @-> ptr vk_descriptor_set_layout
    @-> returning vk_result)

let vkCreateDescriptorSetLayout dev info alloc layout =
  Lazy.force vkCreateDescriptorSetLayout_lazy dev info alloc layout

let vkDestroyDescriptorSetLayout_lazy =
  foreign_vk_lazy
    "vkDestroyDescriptorSetLayout"
    (vk_device_ptr @-> vk_descriptor_set_layout @-> ptr void @-> returning void)

let vkDestroyDescriptorSetLayout dev layout alloc =
  Lazy.force vkDestroyDescriptorSetLayout_lazy dev layout alloc

let vkCreateDescriptorPool_lazy =
  foreign_vk_lazy
    "vkCreateDescriptorPool"
    (vk_device_ptr
    @-> ptr vk_descriptor_pool_create_info
    @-> ptr void @-> ptr vk_descriptor_pool @-> returning vk_result)

let vkCreateDescriptorPool dev info alloc pool =
  Lazy.force vkCreateDescriptorPool_lazy dev info alloc pool

let vkDestroyDescriptorPool_lazy =
  foreign_vk_lazy
    "vkDestroyDescriptorPool"
    (vk_device_ptr @-> vk_descriptor_pool @-> ptr void @-> returning void)

let vkDestroyDescriptorPool dev pool alloc =
  Lazy.force vkDestroyDescriptorPool_lazy dev pool alloc

let vkAllocateDescriptorSets_lazy =
  foreign_vk_lazy
    "vkAllocateDescriptorSets"
    (vk_device_ptr
    @-> ptr vk_descriptor_set_allocate_info
    @-> ptr vk_descriptor_set @-> returning vk_result)

let vkAllocateDescriptorSets dev info sets =
  Lazy.force vkAllocateDescriptorSets_lazy dev info sets

let vkUpdateDescriptorSets_lazy =
  foreign_vk_lazy
    "vkUpdateDescriptorSets"
    (vk_device_ptr @-> uint32_t
    @-> ptr vk_write_descriptor_set
    @-> uint32_t @-> ptr void @-> returning void)

let vkUpdateDescriptorSets dev write_count writes copy_count copies =
  Lazy.force
    vkUpdateDescriptorSets_lazy
    dev
    write_count
    writes
    copy_count
    copies

(** {1 Command Pool/Buffer Functions} *)

let vkCreateCommandPool_lazy =
  foreign_vk_lazy
    "vkCreateCommandPool"
    (vk_device_ptr
    @-> ptr vk_command_pool_create_info
    @-> ptr void @-> ptr vk_command_pool @-> returning vk_result)

let vkCreateCommandPool dev info alloc pool =
  Lazy.force vkCreateCommandPool_lazy dev info alloc pool

let vkDestroyCommandPool_lazy =
  foreign_vk_lazy
    "vkDestroyCommandPool"
    (vk_device_ptr @-> vk_command_pool @-> ptr void @-> returning void)

let vkDestroyCommandPool dev pool alloc =
  Lazy.force vkDestroyCommandPool_lazy dev pool alloc

let vkAllocateCommandBuffers_lazy =
  foreign_vk_lazy
    "vkAllocateCommandBuffers"
    (vk_device_ptr
    @-> ptr vk_command_buffer_allocate_info
    @-> ptr vk_command_buffer_ptr @-> returning vk_result)

let vkAllocateCommandBuffers dev info bufs =
  Lazy.force vkAllocateCommandBuffers_lazy dev info bufs

let vkFreeCommandBuffers_lazy =
  foreign_vk_lazy
    "vkFreeCommandBuffers"
    (vk_device_ptr @-> vk_command_pool @-> uint32_t
   @-> ptr vk_command_buffer_ptr @-> returning void)

let vkFreeCommandBuffers dev pool count bufs =
  Lazy.force vkFreeCommandBuffers_lazy dev pool count bufs

let vkBeginCommandBuffer_lazy =
  foreign_vk_lazy
    "vkBeginCommandBuffer"
    (vk_command_buffer_ptr
    @-> ptr vk_command_buffer_begin_info
    @-> returning vk_result)

let vkBeginCommandBuffer buf info =
  Lazy.force vkBeginCommandBuffer_lazy buf info

let vkEndCommandBuffer_lazy =
  foreign_vk_lazy
    "vkEndCommandBuffer"
    (vk_command_buffer_ptr @-> returning vk_result)

let vkEndCommandBuffer buf = Lazy.force vkEndCommandBuffer_lazy buf

let vkResetCommandBuffer_lazy =
  foreign_vk_lazy
    "vkResetCommandBuffer"
    (vk_command_buffer_ptr @-> vk_flags @-> returning vk_result)

let vkResetCommandBuffer buf flags =
  Lazy.force vkResetCommandBuffer_lazy buf flags

(** {1 Command Recording Functions} *)

let vkCmdBindPipeline_lazy =
  foreign_vk_lazy
    "vkCmdBindPipeline"
    (vk_command_buffer_ptr @-> uint32_t @-> vk_pipeline @-> returning void)

let vkCmdBindPipeline buf bind_point pipeline =
  Lazy.force vkCmdBindPipeline_lazy buf bind_point pipeline

let vkCmdBindDescriptorSets_lazy =
  foreign_vk_lazy
    "vkCmdBindDescriptorSets"
    (vk_command_buffer_ptr @-> uint32_t @-> vk_pipeline_layout @-> uint32_t
   @-> uint32_t @-> ptr vk_descriptor_set @-> uint32_t @-> ptr uint32_t
   @-> returning void)

let vkCmdBindDescriptorSets buf bind_point layout first count sets dyn_count
    dyn_offsets =
  Lazy.force
    vkCmdBindDescriptorSets_lazy
    buf
    bind_point
    layout
    first
    count
    sets
    dyn_count
    dyn_offsets

let vkCmdPushConstants_lazy =
  foreign_vk_lazy
    "vkCmdPushConstants"
    (vk_command_buffer_ptr @-> vk_pipeline_layout @-> uint32_t @-> uint32_t
   @-> uint32_t @-> ptr void @-> returning void)

let vkCmdPushConstants buf layout stage_flags offset size values =
  Lazy.force vkCmdPushConstants_lazy buf layout stage_flags offset size values

let vkCmdDispatch_lazy =
  foreign_vk_lazy
    "vkCmdDispatch"
    (vk_command_buffer_ptr @-> uint32_t @-> uint32_t @-> uint32_t
   @-> returning void)

let vkCmdDispatch buf gx gy gz = Lazy.force vkCmdDispatch_lazy buf gx gy gz

let vkCmdFillBuffer_lazy =
  foreign_vk_lazy
    "vkCmdFillBuffer"
    (vk_command_buffer_ptr @-> vk_buffer @-> vk_device_size @-> vk_device_size
   @-> uint32_t @-> returning void)

let vkCmdFillBuffer buf dst_buf dst_offset size data =
  Lazy.force vkCmdFillBuffer_lazy buf dst_buf dst_offset size data

let vkCmdPipelineBarrier_lazy =
  foreign_vk_lazy
    "vkCmdPipelineBarrier"
    (vk_command_buffer_ptr @-> vk_flags @-> vk_flags @-> vk_flags @-> uint32_t
   @-> ptr void @-> uint32_t
    @-> ptr vk_buffer_memory_barrier
    @-> uint32_t @-> ptr void @-> returning void)

let vkCmdPipelineBarrier buf src_stage dst_stage dep_flags mem_barrier_count
    mem_barriers buf_barrier_count buf_barriers img_barrier_count img_barriers =
  Lazy.force
    vkCmdPipelineBarrier_lazy
    buf
    src_stage
    dst_stage
    dep_flags
    mem_barrier_count
    mem_barriers
    buf_barrier_count
    buf_barriers
    img_barrier_count
    img_barriers

let vkCmdCopyBuffer_lazy =
  foreign_vk_lazy
    "vkCmdCopyBuffer"
    (vk_command_buffer_ptr @-> vk_buffer @-> vk_buffer @-> uint32_t
   @-> ptr vk_buffer_copy @-> returning void)

let vkCmdCopyBuffer buf src dst count regions =
  Lazy.force vkCmdCopyBuffer_lazy buf src dst count regions

(** {1 Fence Functions} *)

let vkCreateFence_lazy =
  foreign_vk_lazy
    "vkCreateFence"
    (vk_device_ptr @-> ptr vk_fence_create_info @-> ptr void @-> ptr vk_fence
   @-> returning vk_result)

let vkCreateFence dev info alloc fence =
  Lazy.force vkCreateFence_lazy dev info alloc fence

let vkDestroyFence_lazy =
  foreign_vk_lazy
    "vkDestroyFence"
    (vk_device_ptr @-> vk_fence @-> ptr void @-> returning void)

let vkDestroyFence dev fence alloc =
  Lazy.force vkDestroyFence_lazy dev fence alloc

let vkWaitForFences_lazy =
  foreign_vk_lazy
    "vkWaitForFences"
    (vk_device_ptr @-> uint32_t @-> ptr vk_fence @-> vk_bool32 @-> uint64_t
   @-> returning vk_result)

let vkWaitForFences dev count fences wait_all timeout =
  Lazy.force vkWaitForFences_lazy dev count fences wait_all timeout

let vkResetFences_lazy =
  foreign_vk_lazy
    "vkResetFences"
    (vk_device_ptr @-> uint32_t @-> ptr vk_fence @-> returning vk_result)

let vkResetFences dev count fences =
  Lazy.force vkResetFences_lazy dev count fences

(** {1 Queue Functions} *)

let vkQueueSubmit_lazy =
  foreign_vk_lazy
    "vkQueueSubmit"
    (vk_queue_ptr @-> uint32_t @-> ptr vk_submit_info @-> vk_fence
   @-> returning vk_result)

let vkQueueSubmit queue count submits fence =
  Lazy.force vkQueueSubmit_lazy queue count submits fence

let vkQueueWaitIdle_lazy =
  foreign_vk_lazy "vkQueueWaitIdle" (vk_queue_ptr @-> returning vk_result)

let vkQueueWaitIdle queue = Lazy.force vkQueueWaitIdle_lazy queue

(** {1 Version Query} *)

let vkEnumerateInstanceVersion_lazy =
  lazy
    (try
       Some
         (foreign
            ~from:(get_vulkan_lib ())
            "vkEnumerateInstanceVersion"
            (ptr uint32_t @-> returning vk_result))
     with _ -> None)

let vkEnumerateInstanceVersion p =
  match Lazy.force vkEnumerateInstanceVersion_lazy with
  | Some f -> f p
  | None ->
      (* Vulkan 1.0 - function doesn't exist, return 1.0.0 *)
      p <-@ Unsigned.UInt32.of_int ((1 lsl 22) lor (0 lsl 12) lor 0) ;
      VK_SUCCESS
