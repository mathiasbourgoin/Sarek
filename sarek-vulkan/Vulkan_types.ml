(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Vulkan API - Ctypes Type Definitions
 *
 * Pure OCaml bindings to Vulkan API using ctypes.
 * No C stubs required - all FFI via ctypes-foreign.
 *
 * This module defines only the subset of Vulkan needed for compute shaders:
 * - Instance/Device management
 * - Memory allocation
 * - Command buffers and queues
 * - Compute pipelines (no graphics)
 * - Descriptor sets for buffer binding
 ******************************************************************************)

open Ctypes

(** {1 Basic Types} *)

(** Vulkan handles are 64-bit on 64-bit platforms *)
type vk_handle = Unsigned.uint64

let vk_handle : vk_handle typ = uint64_t

(** Non-dispatchable handles (pointers on 64-bit, uint64 on 32-bit) *)
type vk_non_dispatchable_handle = Unsigned.uint64

let vk_non_dispatchable_handle : vk_non_dispatchable_handle typ = uint64_t

(** VkDeviceSize for memory sizes *)
type vk_device_size = Unsigned.uint64

let vk_device_size : vk_device_size typ = uint64_t

(** VkFlags for bitmasks *)
type vk_flags = Unsigned.uint32

let vk_flags : vk_flags typ = uint32_t

(** VkBool32 *)
type vk_bool32 = Unsigned.uint32

let vk_bool32 : vk_bool32 typ = uint32_t

let vk_true = Unsigned.UInt32.of_int 1

let vk_false = Unsigned.UInt32.of_int 0

(** {1 Opaque Handle Types} *)

(** Dispatchable handles (pointer-sized) *)
type vk_instance

let vk_instance : vk_instance structure typ = structure "VkInstance_T"

let vk_instance_ptr = ptr vk_instance

type vk_physical_device

let vk_physical_device : vk_physical_device structure typ =
  structure "VkPhysicalDevice_T"

let vk_physical_device_ptr = ptr vk_physical_device

type vk_device

let vk_device : vk_device structure typ = structure "VkDevice_T"

let vk_device_ptr = ptr vk_device

type vk_queue

let vk_queue : vk_queue structure typ = structure "VkQueue_T"

let vk_queue_ptr = ptr vk_queue

type vk_command_buffer

let vk_command_buffer : vk_command_buffer structure typ =
  structure "VkCommandBuffer_T"

let vk_command_buffer_ptr = ptr vk_command_buffer

(** Non-dispatchable handles *)
type vk_buffer = vk_non_dispatchable_handle

let vk_buffer : vk_buffer typ = vk_non_dispatchable_handle

type vk_device_memory = vk_non_dispatchable_handle

let vk_device_memory : vk_device_memory typ = vk_non_dispatchable_handle

type vk_shader_module = vk_non_dispatchable_handle

let vk_shader_module : vk_shader_module typ = vk_non_dispatchable_handle

type vk_pipeline = vk_non_dispatchable_handle

let vk_pipeline : vk_pipeline typ = vk_non_dispatchable_handle

type vk_pipeline_layout = vk_non_dispatchable_handle

let vk_pipeline_layout : vk_pipeline_layout typ = vk_non_dispatchable_handle

type vk_descriptor_set_layout = vk_non_dispatchable_handle

let vk_descriptor_set_layout : vk_descriptor_set_layout typ =
  vk_non_dispatchable_handle

type vk_descriptor_pool = vk_non_dispatchable_handle

let vk_descriptor_pool : vk_descriptor_pool typ = vk_non_dispatchable_handle

type vk_descriptor_set = vk_non_dispatchable_handle

let vk_descriptor_set : vk_descriptor_set typ = vk_non_dispatchable_handle

type vk_command_pool = vk_non_dispatchable_handle

let vk_command_pool : vk_command_pool typ = vk_non_dispatchable_handle

type vk_fence = vk_non_dispatchable_handle

let vk_fence : vk_fence typ = vk_non_dispatchable_handle

type vk_semaphore = vk_non_dispatchable_handle

let vk_semaphore : vk_semaphore typ = vk_non_dispatchable_handle

type vk_event = vk_non_dispatchable_handle

let vk_event : vk_event typ = vk_non_dispatchable_handle

type vk_pipeline_cache = vk_non_dispatchable_handle

let vk_pipeline_cache : vk_pipeline_cache typ = vk_non_dispatchable_handle

(** {1 Result Codes} *)

type vk_result =
  | VK_SUCCESS
  | VK_NOT_READY
  | VK_TIMEOUT
  | VK_EVENT_SET
  | VK_EVENT_RESET
  | VK_INCOMPLETE
  | VK_ERROR_OUT_OF_HOST_MEMORY
  | VK_ERROR_OUT_OF_DEVICE_MEMORY
  | VK_ERROR_INITIALIZATION_FAILED
  | VK_ERROR_DEVICE_LOST
  | VK_ERROR_MEMORY_MAP_FAILED
  | VK_ERROR_LAYER_NOT_PRESENT
  | VK_ERROR_EXTENSION_NOT_PRESENT
  | VK_ERROR_FEATURE_NOT_PRESENT
  | VK_ERROR_INCOMPATIBLE_DRIVER
  | VK_ERROR_TOO_MANY_OBJECTS
  | VK_ERROR_FORMAT_NOT_SUPPORTED
  | VK_ERROR_FRAGMENTED_POOL
  | VK_ERROR_UNKNOWN of int

let vk_result_of_int = function
  | 0 -> VK_SUCCESS
  | 1 -> VK_NOT_READY
  | 2 -> VK_TIMEOUT
  | 3 -> VK_EVENT_SET
  | 4 -> VK_EVENT_RESET
  | 5 -> VK_INCOMPLETE
  | -1 -> VK_ERROR_OUT_OF_HOST_MEMORY
  | -2 -> VK_ERROR_OUT_OF_DEVICE_MEMORY
  | -3 -> VK_ERROR_INITIALIZATION_FAILED
  | -4 -> VK_ERROR_DEVICE_LOST
  | -5 -> VK_ERROR_MEMORY_MAP_FAILED
  | -6 -> VK_ERROR_LAYER_NOT_PRESENT
  | -7 -> VK_ERROR_EXTENSION_NOT_PRESENT
  | -8 -> VK_ERROR_FEATURE_NOT_PRESENT
  | -9 -> VK_ERROR_INCOMPATIBLE_DRIVER
  | -10 -> VK_ERROR_TOO_MANY_OBJECTS
  | -11 -> VK_ERROR_FORMAT_NOT_SUPPORTED
  | -12 -> VK_ERROR_FRAGMENTED_POOL
  | n -> VK_ERROR_UNKNOWN n

let int_of_vk_result = function
  | VK_SUCCESS -> 0
  | VK_NOT_READY -> 1
  | VK_TIMEOUT -> 2
  | VK_EVENT_SET -> 3
  | VK_EVENT_RESET -> 4
  | VK_INCOMPLETE -> 5
  | VK_ERROR_OUT_OF_HOST_MEMORY -> -1
  | VK_ERROR_OUT_OF_DEVICE_MEMORY -> -2
  | VK_ERROR_INITIALIZATION_FAILED -> -3
  | VK_ERROR_DEVICE_LOST -> -4
  | VK_ERROR_MEMORY_MAP_FAILED -> -5
  | VK_ERROR_LAYER_NOT_PRESENT -> -6
  | VK_ERROR_EXTENSION_NOT_PRESENT -> -7
  | VK_ERROR_FEATURE_NOT_PRESENT -> -8
  | VK_ERROR_INCOMPATIBLE_DRIVER -> -9
  | VK_ERROR_TOO_MANY_OBJECTS -> -10
  | VK_ERROR_FORMAT_NOT_SUPPORTED -> -11
  | VK_ERROR_FRAGMENTED_POOL -> -12
  | VK_ERROR_UNKNOWN n -> n

let vk_result : vk_result typ =
  view ~read:vk_result_of_int ~write:int_of_vk_result int

let string_of_vk_result = function
  | VK_SUCCESS -> "VK_SUCCESS"
  | VK_NOT_READY -> "VK_NOT_READY"
  | VK_TIMEOUT -> "VK_TIMEOUT"
  | VK_EVENT_SET -> "VK_EVENT_SET"
  | VK_EVENT_RESET -> "VK_EVENT_RESET"
  | VK_INCOMPLETE -> "VK_INCOMPLETE"
  | VK_ERROR_OUT_OF_HOST_MEMORY -> "VK_ERROR_OUT_OF_HOST_MEMORY"
  | VK_ERROR_OUT_OF_DEVICE_MEMORY -> "VK_ERROR_OUT_OF_DEVICE_MEMORY"
  | VK_ERROR_INITIALIZATION_FAILED -> "VK_ERROR_INITIALIZATION_FAILED"
  | VK_ERROR_DEVICE_LOST -> "VK_ERROR_DEVICE_LOST"
  | VK_ERROR_MEMORY_MAP_FAILED -> "VK_ERROR_MEMORY_MAP_FAILED"
  | VK_ERROR_LAYER_NOT_PRESENT -> "VK_ERROR_LAYER_NOT_PRESENT"
  | VK_ERROR_EXTENSION_NOT_PRESENT -> "VK_ERROR_EXTENSION_NOT_PRESENT"
  | VK_ERROR_FEATURE_NOT_PRESENT -> "VK_ERROR_FEATURE_NOT_PRESENT"
  | VK_ERROR_INCOMPATIBLE_DRIVER -> "VK_ERROR_INCOMPATIBLE_DRIVER"
  | VK_ERROR_TOO_MANY_OBJECTS -> "VK_ERROR_TOO_MANY_OBJECTS"
  | VK_ERROR_FORMAT_NOT_SUPPORTED -> "VK_ERROR_FORMAT_NOT_SUPPORTED"
  | VK_ERROR_FRAGMENTED_POOL -> "VK_ERROR_FRAGMENTED_POOL"
  | VK_ERROR_UNKNOWN n -> Printf.sprintf "VK_ERROR_UNKNOWN(%d)" n

(** {1 Structure Types} *)

type vk_structure_type = int

(* Only the ones we need for compute *)
let vk_structure_type_application_info = 0

let vk_structure_type_instance_create_info = 1

let vk_structure_type_device_queue_create_info = 2

let vk_structure_type_device_create_info = 3

let vk_structure_type_submit_info = 4

let vk_structure_type_memory_allocate_info = 5

let vk_structure_type_buffer_create_info = 12

let vk_structure_type_shader_module_create_info = 16

let vk_structure_type_compute_pipeline_create_info = 29

let vk_structure_type_pipeline_layout_create_info = 30

let vk_structure_type_descriptor_set_layout_create_info = 32

let vk_structure_type_descriptor_pool_create_info = 33

let vk_structure_type_descriptor_set_allocate_info = 34

let vk_structure_type_write_descriptor_set = 35

let vk_structure_type_fence_create_info = 8

let vk_structure_type_command_pool_create_info = 39

let vk_structure_type_command_buffer_allocate_info = 40

let vk_structure_type_command_buffer_begin_info = 42

let vk_structure_type_pipeline_shader_stage_create_info = 18

(** {1 Queue Family Properties} *)

let vk_queue_compute_bit = 0x00000002

let vk_queue_transfer_bit = 0x00000004

(** {1 Memory Property Flags} *)

let vk_memory_property_device_local_bit = 0x00000001

let vk_memory_property_host_visible_bit = 0x00000002

let vk_memory_property_host_coherent_bit = 0x00000004

let vk_memory_property_host_cached_bit = 0x00000008

(** {1 Buffer Usage Flags} *)

let vk_buffer_usage_transfer_src_bit = 0x00000001

let vk_buffer_usage_transfer_dst_bit = 0x00000002

let vk_buffer_usage_storage_buffer_bit = 0x00000020

(** {1 Descriptor Types} *)

let vk_descriptor_type_storage_buffer = 7

(** {1 Shader Stage Flags} *)

let vk_shader_stage_compute_bit = 0x00000020

(** {1 Pipeline Stage Flags} *)

let vk_pipeline_stage_host_bit = 0x00004000

let vk_pipeline_stage_compute_shader_bit = 0x00000800

(** {1 Access Flags} *)

let vk_access_host_write_bit = 0x00004000

let vk_access_host_read_bit = 0x00008000

let vk_access_shader_read_bit = 0x00000020

let vk_access_shader_write_bit = 0x00000040

(** {1 Misc Constants} *)

let vk_queue_family_ignored = 0xFFFFFFFF

let vk_structure_type_buffer_memory_barrier = 44

(** {1 Pipeline Bind Points} *)

let vk_pipeline_bind_point_compute = 1

(** {1 Command Buffer Level} *)

let vk_command_buffer_level_primary = 0

(** {1 Command Buffer Usage Flags} *)

let vk_command_buffer_usage_one_time_submit_bit = 0x00000001

(** {1 Fence Create Flags} *)

let vk_fence_create_signaled_bit = 0x00000001

(** {1 Structures} *)

(* VkApplicationInfo *)
type vk_application_info

let vk_application_info : vk_application_info structure typ =
  structure "VkApplicationInfo"

let app_info_sType = field vk_application_info "sType" uint32_t

let app_info_pNext = field vk_application_info "pNext" (ptr void)

let app_info_pApplicationName =
  field vk_application_info "pApplicationName" (ptr char)

let app_info_applicationVersion =
  field vk_application_info "applicationVersion" uint32_t

let app_info_pEngineName = field vk_application_info "pEngineName" (ptr char)

let app_info_engineVersion = field vk_application_info "engineVersion" uint32_t

let app_info_apiVersion = field vk_application_info "apiVersion" uint32_t

let () = seal vk_application_info

(* VkInstanceCreateInfo *)
type vk_instance_create_info

let vk_instance_create_info : vk_instance_create_info structure typ =
  structure "VkInstanceCreateInfo"

let inst_create_sType = field vk_instance_create_info "sType" uint32_t

let inst_create_pNext = field vk_instance_create_info "pNext" (ptr void)

let inst_create_flags = field vk_instance_create_info "flags" vk_flags

let inst_create_pApplicationInfo =
  field vk_instance_create_info "pApplicationInfo" (ptr vk_application_info)

let inst_create_enabledLayerCount =
  field vk_instance_create_info "enabledLayerCount" uint32_t

let inst_create_ppEnabledLayerNames =
  field vk_instance_create_info "ppEnabledLayerNames" (ptr string)

let inst_create_enabledExtensionCount =
  field vk_instance_create_info "enabledExtensionCount" uint32_t

let inst_create_ppEnabledExtensionNames =
  field vk_instance_create_info "ppEnabledExtensionNames" (ptr string)

let () = seal vk_instance_create_info

(* VkPhysicalDeviceProperties *)
type vk_physical_device_properties

let vk_physical_device_properties : vk_physical_device_properties structure typ
    =
  structure "VkPhysicalDeviceProperties"

let phys_props_apiVersion =
  field vk_physical_device_properties "apiVersion" uint32_t

let phys_props_driverVersion =
  field vk_physical_device_properties "driverVersion" uint32_t

let phys_props_vendorID =
  field vk_physical_device_properties "vendorID" uint32_t

let phys_props_deviceID =
  field vk_physical_device_properties "deviceID" uint32_t

let phys_props_deviceType =
  field vk_physical_device_properties "deviceType" uint32_t

let phys_props_deviceName =
  field vk_physical_device_properties "deviceName" (array 256 char)

(* Skip pipelineCacheUUID and limits/sparseProperties for now - just padding *)
let phys_props_padding =
  field vk_physical_device_properties "padding" (array 1024 char)

let () = seal vk_physical_device_properties

(* VkPhysicalDeviceFeatures - exactly 55 VkBool32 (uint32_t) fields in the
   fixed order defined by the Vulkan spec (vulkan_core.h). We only care
   about a couple of them (shaderFloat64 at 0-based index 39, shaderInt64
   at index 40), so the struct is modelled as a single fixed-size array of
   uint32_t rather than 55 named fields - same "care about a slice, pad the
   rest" approach as vk_physical_device_properties above. Field index
   derivation (0-based, verified against Khronos vulkan_core.h - the 40th
   field 1-based): robustBufferAccess=0 ... shaderClipDistance=37,
   shaderCullDistance=38, shaderFloat64=39, shaderInt64=40, shaderInt16=41,
   ... inheritedQueries=54. *)
type vk_physical_device_features

let vk_physical_device_features_field_count = 55

let shader_float64_field_index = 39

let shader_int64_field_index = 40

let vk_physical_device_features : vk_physical_device_features structure typ =
  structure "VkPhysicalDeviceFeatures"

let phys_features_bools =
  field
    vk_physical_device_features
    "bools"
    (array vk_physical_device_features_field_count uint32_t)

let () = seal vk_physical_device_features

(* VkQueueFamilyProperties *)
type vk_queue_family_properties

let vk_queue_family_properties : vk_queue_family_properties structure typ =
  structure "VkQueueFamilyProperties"

let queue_family_queueFlags =
  field vk_queue_family_properties "queueFlags" vk_flags

let queue_family_queueCount =
  field vk_queue_family_properties "queueCount" uint32_t

let queue_family_timestampValidBits =
  field vk_queue_family_properties "timestampValidBits" uint32_t

(* VkExtent3D inline *)
let queue_family_minImageTransferGranularity_width =
  field vk_queue_family_properties "minImageTransferGranularity_width" uint32_t

let queue_family_minImageTransferGranularity_height =
  field vk_queue_family_properties "minImageTransferGranularity_height" uint32_t

let queue_family_minImageTransferGranularity_depth =
  field vk_queue_family_properties "minImageTransferGranularity_depth" uint32_t

let () = seal vk_queue_family_properties

(* VkDeviceQueueCreateInfo *)
type vk_device_queue_create_info

let vk_device_queue_create_info : vk_device_queue_create_info structure typ =
  structure "VkDeviceQueueCreateInfo"

let dev_queue_create_sType = field vk_device_queue_create_info "sType" uint32_t

let dev_queue_create_pNext =
  field vk_device_queue_create_info "pNext" (ptr void)

let dev_queue_create_flags = field vk_device_queue_create_info "flags" vk_flags

let dev_queue_create_queueFamilyIndex =
  field vk_device_queue_create_info "queueFamilyIndex" uint32_t

let dev_queue_create_queueCount =
  field vk_device_queue_create_info "queueCount" uint32_t

let dev_queue_create_pQueuePriorities =
  field vk_device_queue_create_info "pQueuePriorities" (ptr float)

let () = seal vk_device_queue_create_info

(* VkDeviceCreateInfo *)
type vk_device_create_info

let vk_device_create_info : vk_device_create_info structure typ =
  structure "VkDeviceCreateInfo"

let dev_create_sType = field vk_device_create_info "sType" uint32_t

let dev_create_pNext = field vk_device_create_info "pNext" (ptr void)

let dev_create_flags = field vk_device_create_info "flags" vk_flags

let dev_create_queueCreateInfoCount =
  field vk_device_create_info "queueCreateInfoCount" uint32_t

let dev_create_pQueueCreateInfos =
  field
    vk_device_create_info
    "pQueueCreateInfos"
    (ptr vk_device_queue_create_info)

let dev_create_enabledLayerCount =
  field vk_device_create_info "enabledLayerCount" uint32_t

let dev_create_ppEnabledLayerNames =
  field vk_device_create_info "ppEnabledLayerNames" (ptr string)

let dev_create_enabledExtensionCount =
  field vk_device_create_info "enabledExtensionCount" uint32_t

let dev_create_ppEnabledExtensionNames =
  field vk_device_create_info "ppEnabledExtensionNames" (ptr string)

let dev_create_pEnabledFeatures =
  field vk_device_create_info "pEnabledFeatures" (ptr void)

let () = seal vk_device_create_info

(* VkMemoryAllocateInfo *)
type vk_memory_allocate_info

let vk_memory_allocate_info : vk_memory_allocate_info structure typ =
  structure "VkMemoryAllocateInfo"

let mem_alloc_sType = field vk_memory_allocate_info "sType" uint32_t

let mem_alloc_pNext = field vk_memory_allocate_info "pNext" (ptr void)

let mem_alloc_allocationSize =
  field vk_memory_allocate_info "allocationSize" vk_device_size

let mem_alloc_memoryTypeIndex =
  field vk_memory_allocate_info "memoryTypeIndex" uint32_t

let () = seal vk_memory_allocate_info

(* VkBufferCreateInfo *)
type vk_buffer_create_info

let vk_buffer_create_info : vk_buffer_create_info structure typ =
  structure "VkBufferCreateInfo"

let buf_create_sType = field vk_buffer_create_info "sType" uint32_t

let buf_create_pNext = field vk_buffer_create_info "pNext" (ptr void)

let buf_create_flags = field vk_buffer_create_info "flags" vk_flags

let buf_create_size = field vk_buffer_create_info "size" vk_device_size

let buf_create_usage = field vk_buffer_create_info "usage" vk_flags

let buf_create_sharingMode = field vk_buffer_create_info "sharingMode" uint32_t

let buf_create_queueFamilyIndexCount =
  field vk_buffer_create_info "queueFamilyIndexCount" uint32_t

let buf_create_pQueueFamilyIndices =
  field vk_buffer_create_info "pQueueFamilyIndices" (ptr uint32_t)

let () = seal vk_buffer_create_info

(* VkMemoryRequirements *)
type vk_memory_requirements

let vk_memory_requirements : vk_memory_requirements structure typ =
  structure "VkMemoryRequirements"

let mem_req_size = field vk_memory_requirements "size" vk_device_size

let mem_req_alignment = field vk_memory_requirements "alignment" vk_device_size

let mem_req_memoryTypeBits =
  field vk_memory_requirements "memoryTypeBits" uint32_t

let () = seal vk_memory_requirements

(* VkPhysicalDeviceMemoryProperties *)
type vk_memory_type

let vk_memory_type : vk_memory_type structure typ = structure "VkMemoryType"

let mem_type_propertyFlags = field vk_memory_type "propertyFlags" vk_flags

let mem_type_heapIndex = field vk_memory_type "heapIndex" uint32_t

let () = seal vk_memory_type

type vk_memory_heap

let vk_memory_heap : vk_memory_heap structure typ = structure "VkMemoryHeap"

let mem_heap_size = field vk_memory_heap "size" vk_device_size

let mem_heap_flags = field vk_memory_heap "flags" vk_flags

let () = seal vk_memory_heap

type vk_physical_device_memory_properties

let vk_physical_device_memory_properties :
    vk_physical_device_memory_properties structure typ =
  structure "VkPhysicalDeviceMemoryProperties"

let mem_props_memoryTypeCount =
  field vk_physical_device_memory_properties "memoryTypeCount" uint32_t

let mem_props_memoryTypes =
  field
    vk_physical_device_memory_properties
    "memoryTypes"
    (array 32 vk_memory_type)

let mem_props_memoryHeapCount =
  field vk_physical_device_memory_properties "memoryHeapCount" uint32_t

let mem_props_memoryHeaps =
  field
    vk_physical_device_memory_properties
    "memoryHeaps"
    (array 16 vk_memory_heap)

let () = seal vk_physical_device_memory_properties

(* VkShaderModuleCreateInfo *)
type vk_shader_module_create_info

let vk_shader_module_create_info : vk_shader_module_create_info structure typ =
  structure "VkShaderModuleCreateInfo"

let shader_create_sType = field vk_shader_module_create_info "sType" uint32_t

let shader_create_pNext = field vk_shader_module_create_info "pNext" (ptr void)

let shader_create_flags = field vk_shader_module_create_info "flags" vk_flags

let shader_create_codeSize =
  field vk_shader_module_create_info "codeSize" size_t

let shader_create_pCode =
  field vk_shader_module_create_info "pCode" (ptr uint32_t)

let () = seal vk_shader_module_create_info

(* VkCommandPoolCreateInfo *)
type vk_command_pool_create_info

let vk_command_pool_create_info : vk_command_pool_create_info structure typ =
  structure "VkCommandPoolCreateInfo"

let cmd_pool_create_sType = field vk_command_pool_create_info "sType" uint32_t

let cmd_pool_create_pNext = field vk_command_pool_create_info "pNext" (ptr void)

let cmd_pool_create_flags = field vk_command_pool_create_info "flags" vk_flags

let cmd_pool_create_queueFamilyIndex =
  field vk_command_pool_create_info "queueFamilyIndex" uint32_t

let () = seal vk_command_pool_create_info

(* VkCommandBufferAllocateInfo *)
type vk_command_buffer_allocate_info

let vk_command_buffer_allocate_info :
    vk_command_buffer_allocate_info structure typ =
  structure "VkCommandBufferAllocateInfo"

let cmd_buf_alloc_sType = field vk_command_buffer_allocate_info "sType" uint32_t

let cmd_buf_alloc_pNext =
  field vk_command_buffer_allocate_info "pNext" (ptr void)

let cmd_buf_alloc_commandPool =
  field vk_command_buffer_allocate_info "commandPool" vk_command_pool

let cmd_buf_alloc_level = field vk_command_buffer_allocate_info "level" uint32_t

let cmd_buf_alloc_commandBufferCount =
  field vk_command_buffer_allocate_info "commandBufferCount" uint32_t

let () = seal vk_command_buffer_allocate_info

(* VkFenceCreateInfo *)
type vk_fence_create_info

let vk_fence_create_info : vk_fence_create_info structure typ =
  structure "VkFenceCreateInfo"

let fence_create_sType = field vk_fence_create_info "sType" uint32_t

let fence_create_pNext = field vk_fence_create_info "pNext" (ptr void)

let fence_create_flags = field vk_fence_create_info "flags" vk_flags

let () = seal vk_fence_create_info

(* VkSubmitInfo *)
type vk_submit_info

let vk_submit_info : vk_submit_info structure typ = structure "VkSubmitInfo"

let submit_sType = field vk_submit_info "sType" uint32_t

let submit_pNext = field vk_submit_info "pNext" (ptr void)

let submit_waitSemaphoreCount =
  field vk_submit_info "waitSemaphoreCount" uint32_t

let submit_pWaitSemaphores =
  field vk_submit_info "pWaitSemaphores" (ptr vk_semaphore)

let submit_pWaitDstStageMask =
  field vk_submit_info "pWaitDstStageMask" (ptr vk_flags)

let submit_commandBufferCount =
  field vk_submit_info "commandBufferCount" uint32_t

let submit_pCommandBuffers =
  field vk_submit_info "pCommandBuffers" (ptr vk_command_buffer_ptr)
(* ptr VkCommandBuffer = VkCommandBuffer* - pointer to array of command buffers *)

let submit_signalSemaphoreCount =
  field vk_submit_info "signalSemaphoreCount" uint32_t

let submit_pSignalSemaphores =
  field vk_submit_info "pSignalSemaphores" (ptr vk_semaphore)

let () = seal vk_submit_info

(** VkDescriptorSetLayoutBinding *)
type vk_descriptor_set_layout_binding

let vk_descriptor_set_layout_binding :
    vk_descriptor_set_layout_binding structure typ =
  structure "VkDescriptorSetLayoutBinding"

let dsl_binding_binding =
  field vk_descriptor_set_layout_binding "binding" uint32_t

let dsl_binding_descriptorType =
  field vk_descriptor_set_layout_binding "descriptorType" uint32_t

let dsl_binding_descriptorCount =
  field vk_descriptor_set_layout_binding "descriptorCount" uint32_t

let dsl_binding_stageFlags =
  field vk_descriptor_set_layout_binding "stageFlags" vk_flags

let dsl_binding_pImmutableSamplers =
  field vk_descriptor_set_layout_binding "pImmutableSamplers" (ptr void)

let () = seal vk_descriptor_set_layout_binding

(** VkDescriptorSetLayoutCreateInfo *)
type vk_descriptor_set_layout_create_info

let vk_descriptor_set_layout_create_info :
    vk_descriptor_set_layout_create_info structure typ =
  structure "VkDescriptorSetLayoutCreateInfo"

let dsl_create_sType =
  field vk_descriptor_set_layout_create_info "sType" uint32_t

let dsl_create_pNext =
  field vk_descriptor_set_layout_create_info "pNext" (ptr void)

let dsl_create_flags =
  field vk_descriptor_set_layout_create_info "flags" vk_flags

let dsl_create_bindingCount =
  field vk_descriptor_set_layout_create_info "bindingCount" uint32_t

let dsl_create_pBindings =
  field
    vk_descriptor_set_layout_create_info
    "pBindings"
    (ptr vk_descriptor_set_layout_binding)

let () = seal vk_descriptor_set_layout_create_info

(** VkPushConstantRange *)
type vk_push_constant_range

let vk_push_constant_range : vk_push_constant_range structure typ =
  structure "VkPushConstantRange"

let push_const_stageFlags = field vk_push_constant_range "stageFlags" uint32_t

let push_const_offset = field vk_push_constant_range "offset" uint32_t

let push_const_size = field vk_push_constant_range "size" uint32_t

let () = seal vk_push_constant_range

(** VkBufferMemoryBarrier *)
type vk_buffer_memory_barrier

let vk_buffer_memory_barrier : vk_buffer_memory_barrier structure typ =
  structure "VkBufferMemoryBarrier"

let buf_barrier_sType = field vk_buffer_memory_barrier "sType" uint32_t

let buf_barrier_pNext = field vk_buffer_memory_barrier "pNext" (ptr void)

let buf_barrier_srcAccessMask =
  field vk_buffer_memory_barrier "srcAccessMask" vk_flags

let buf_barrier_dstAccessMask =
  field vk_buffer_memory_barrier "dstAccessMask" vk_flags

let buf_barrier_srcQueueFamilyIndex =
  field vk_buffer_memory_barrier "srcQueueFamilyIndex" uint32_t

let buf_barrier_dstQueueFamilyIndex =
  field vk_buffer_memory_barrier "dstQueueFamilyIndex" uint32_t

let buf_barrier_buffer = field vk_buffer_memory_barrier "buffer" vk_buffer

let buf_barrier_offset = field vk_buffer_memory_barrier "offset" vk_device_size

let buf_barrier_size = field vk_buffer_memory_barrier "size" vk_device_size

let () = seal vk_buffer_memory_barrier

(** VkPipelineLayoutCreateInfo *)
type vk_pipeline_layout_create_info

let vk_pipeline_layout_create_info :
    vk_pipeline_layout_create_info structure typ =
  structure "VkPipelineLayoutCreateInfo"

let pl_create_sType = field vk_pipeline_layout_create_info "sType" uint32_t

let pl_create_pNext = field vk_pipeline_layout_create_info "pNext" (ptr void)

let pl_create_flags = field vk_pipeline_layout_create_info "flags" vk_flags

let pl_create_setLayoutCount =
  field vk_pipeline_layout_create_info "setLayoutCount" uint32_t

let pl_create_pSetLayouts =
  field
    vk_pipeline_layout_create_info
    "pSetLayouts"
    (ptr vk_descriptor_set_layout)

let pl_create_pushConstantRangeCount =
  field vk_pipeline_layout_create_info "pushConstantRangeCount" uint32_t

let pl_create_pPushConstantRanges =
  field
    vk_pipeline_layout_create_info
    "pPushConstantRanges"
    (ptr vk_push_constant_range)

let () = seal vk_pipeline_layout_create_info

(** VkPipelineShaderStageCreateInfo *)
type vk_pipeline_shader_stage_create_info

let vk_pipeline_shader_stage_create_info :
    vk_pipeline_shader_stage_create_info structure typ =
  structure "VkPipelineShaderStageCreateInfo"

let shader_stage_sType =
  field vk_pipeline_shader_stage_create_info "sType" uint32_t

let shader_stage_pNext =
  field vk_pipeline_shader_stage_create_info "pNext" (ptr void)

let shader_stage_flags =
  field vk_pipeline_shader_stage_create_info "flags" vk_flags

let shader_stage_stage =
  field vk_pipeline_shader_stage_create_info "stage" vk_flags

let shader_stage_module =
  field vk_pipeline_shader_stage_create_info "module" vk_shader_module

(* [ptr char], not the [string] view: writing an OCaml string through the
   [string] view allocates an anonymous C buffer whose only GC root is the fat
   pointer that [setf] immediately discards, leaving pName dangling. Callers
   must hold the [CArray.t] themselves and keep it alive past the FFI call. *)
let shader_stage_pName =
  field vk_pipeline_shader_stage_create_info "pName" (ptr char)

let shader_stage_pSpecializationInfo =
  field vk_pipeline_shader_stage_create_info "pSpecializationInfo" (ptr void)

let () = seal vk_pipeline_shader_stage_create_info

(** VkComputePipelineCreateInfo *)
type vk_compute_pipeline_create_info

let vk_compute_pipeline_create_info :
    vk_compute_pipeline_create_info structure typ =
  structure "VkComputePipelineCreateInfo"

let compute_pipe_sType = field vk_compute_pipeline_create_info "sType" uint32_t

let compute_pipe_pNext =
  field vk_compute_pipeline_create_info "pNext" (ptr void)

let compute_pipe_flags = field vk_compute_pipeline_create_info "flags" vk_flags

let compute_pipe_stage =
  field
    vk_compute_pipeline_create_info
    "stage"
    vk_pipeline_shader_stage_create_info

let compute_pipe_layout =
  field vk_compute_pipeline_create_info "layout" vk_pipeline_layout

let compute_pipe_basePipelineHandle =
  field vk_compute_pipeline_create_info "basePipelineHandle" vk_pipeline

let compute_pipe_basePipelineIndex =
  field vk_compute_pipeline_create_info "basePipelineIndex" int32_t

let () = seal vk_compute_pipeline_create_info

(** VkDescriptorPoolSize *)
type vk_descriptor_pool_size

let vk_descriptor_pool_size : vk_descriptor_pool_size structure typ =
  structure "VkDescriptorPoolSize"

let pool_size_type = field vk_descriptor_pool_size "type" uint32_t

let pool_size_descriptorCount =
  field vk_descriptor_pool_size "descriptorCount" uint32_t

let () = seal vk_descriptor_pool_size

(** VkDescriptorPoolCreateInfo *)
type vk_descriptor_pool_create_info

let vk_descriptor_pool_create_info :
    vk_descriptor_pool_create_info structure typ =
  structure "VkDescriptorPoolCreateInfo"

let desc_pool_sType = field vk_descriptor_pool_create_info "sType" uint32_t

let desc_pool_pNext = field vk_descriptor_pool_create_info "pNext" (ptr void)

let desc_pool_flags = field vk_descriptor_pool_create_info "flags" vk_flags

let desc_pool_maxSets = field vk_descriptor_pool_create_info "maxSets" uint32_t

let desc_pool_poolSizeCount =
  field vk_descriptor_pool_create_info "poolSizeCount" uint32_t

let desc_pool_pPoolSizes =
  field
    vk_descriptor_pool_create_info
    "pPoolSizes"
    (ptr vk_descriptor_pool_size)

let () = seal vk_descriptor_pool_create_info

(** VkDescriptorSetAllocateInfo *)
type vk_descriptor_set_allocate_info

let vk_descriptor_set_allocate_info :
    vk_descriptor_set_allocate_info structure typ =
  structure "VkDescriptorSetAllocateInfo"

let desc_set_alloc_sType =
  field vk_descriptor_set_allocate_info "sType" uint32_t

let desc_set_alloc_pNext =
  field vk_descriptor_set_allocate_info "pNext" (ptr void)

let desc_set_alloc_descriptorPool =
  field vk_descriptor_set_allocate_info "descriptorPool" vk_descriptor_pool

let desc_set_alloc_descriptorSetCount =
  field vk_descriptor_set_allocate_info "descriptorSetCount" uint32_t

let desc_set_alloc_pSetLayouts =
  field
    vk_descriptor_set_allocate_info
    "pSetLayouts"
    (ptr vk_descriptor_set_layout)

let () = seal vk_descriptor_set_allocate_info

(** VkDescriptorBufferInfo *)
type vk_descriptor_buffer_info

let vk_descriptor_buffer_info : vk_descriptor_buffer_info structure typ =
  structure "VkDescriptorBufferInfo"

let desc_buf_buffer = field vk_descriptor_buffer_info "buffer" vk_buffer

let desc_buf_offset = field vk_descriptor_buffer_info "offset" vk_device_size

let desc_buf_range = field vk_descriptor_buffer_info "range" vk_device_size

let () = seal vk_descriptor_buffer_info

(** VkWriteDescriptorSet *)
type vk_write_descriptor_set

let vk_write_descriptor_set : vk_write_descriptor_set structure typ =
  structure "VkWriteDescriptorSet"

let write_desc_sType = field vk_write_descriptor_set "sType" uint32_t

let write_desc_pNext = field vk_write_descriptor_set "pNext" (ptr void)

let write_desc_dstSet = field vk_write_descriptor_set "dstSet" vk_descriptor_set

let write_desc_dstBinding = field vk_write_descriptor_set "dstBinding" uint32_t

let write_desc_dstArrayElement =
  field vk_write_descriptor_set "dstArrayElement" uint32_t

let write_desc_descriptorCount =
  field vk_write_descriptor_set "descriptorCount" uint32_t

let write_desc_descriptorType =
  field vk_write_descriptor_set "descriptorType" uint32_t

let write_desc_pImageInfo =
  field vk_write_descriptor_set "pImageInfo" (ptr void)

let write_desc_pBufferInfo =
  field vk_write_descriptor_set "pBufferInfo" (ptr vk_descriptor_buffer_info)

let write_desc_pTexelBufferView =
  field vk_write_descriptor_set "pTexelBufferView" (ptr void)

let () = seal vk_write_descriptor_set

(** VkCommandBufferBeginInfo *)
type vk_command_buffer_begin_info

let vk_command_buffer_begin_info : vk_command_buffer_begin_info structure typ =
  structure "VkCommandBufferBeginInfo"

let cmd_buf_begin_sType = field vk_command_buffer_begin_info "sType" uint32_t

let cmd_buf_begin_pNext = field vk_command_buffer_begin_info "pNext" (ptr void)

let cmd_buf_begin_flags = field vk_command_buffer_begin_info "flags" vk_flags

let cmd_buf_begin_pInheritanceInfo =
  field vk_command_buffer_begin_info "pInheritanceInfo" (ptr void)

let () = seal vk_command_buffer_begin_info

(** Null handle constant *)
let vk_null_handle = Unsigned.UInt64.zero

(** VkBufferCopy *)
type vk_buffer_copy

let vk_buffer_copy : vk_buffer_copy structure typ = structure "VkBufferCopy"

let buffer_copy_srcOffset = field vk_buffer_copy "srcOffset" vk_device_size

let buffer_copy_dstOffset = field vk_buffer_copy "dstOffset" vk_device_size

let buffer_copy_size = field vk_buffer_copy "size" vk_device_size

let () = seal vk_buffer_copy

(** {1 Extended feature and property queries (backlog-62 slice 2)}

    Everything below is reached through the [pNext] chains of
    [VkPhysicalDeviceFeatures2] / [VkPhysicalDeviceProperties2], which is the
    only way to query [shaderFloat16], [storageBuffer16BitAccess],
    [cooperativeMatrix], the driver identity and the subgroup size. Both entry
    points are core Vulkan 1.1, and the instance this backend creates requests
    API 1.2, so neither needs an instance extension.

    Two rules the structs below obey, and both were arrived at the hard way:

    - Every struct is ZEROED and given its [sType] before the query. A driver
      that does not recognise an [sType] leaves the struct untouched, so an
      uninitialised struct would be read as "feature present" on exactly the
      devices that lack it.
    - A struct chained into a query must be at least as large as the driver
      believes it to be. [vk_physical_device_properties] above is a TRUNCATED
      model (it stops after [deviceName] and pads), so it cannot be embedded in
      [VkPhysicalDeviceProperties2] by value; the properties2 wrapper below
      carries an opaque blob sized past the real struct instead. *)

let vk_structure_type_physical_device_features_2 = 1000059000

let vk_structure_type_physical_device_properties_2 = 1000059001

let vk_structure_type_physical_device_shader_float16_int8_features = 1000082000

let vk_structure_type_physical_device_16bit_storage_features = 1000083000

(** backlog-62 slice 3. The three enumerants below are what the INTEGER
    cooperative-matrix path needs and the f16 path did not.

    Measured, not guessed: [glslc --target-env=vulkan1.3] on a 16x16x16
    [u8 x u8 -> s32] [coopMatMulAdd] emits exactly [OpCapability Int8],
    [OpCapability StorageBuffer8BitAccess], [OpCapability VulkanMemoryModel] and
    [OpCapability CooperativeMatrixKHR]. Slice 2 enables only the last. The
    first three are conditional on [shaderInt8], [storageBuffer8BitAccess] and
    [vulkanMemoryModel] being ENABLED on the logical device, exactly as #142's
    [shaderInt64] was, and with the same failure mode: RADV computes the right
    answer anyway and the violation is visible only under
    VK_LAYER_KHRONOS_validation. *)

let vk_structure_type_physical_device_8bit_storage_features = 1000177000

let vk_structure_type_physical_device_vulkan_memory_model_features = 1000211000

let vk_structure_type_physical_device_subgroup_properties = 1000094000

let vk_structure_type_physical_device_driver_properties = 1000196000

let vk_structure_type_physical_device_cooperative_matrix_features_khr =
  1000506000

let vk_structure_type_cooperative_matrix_properties_khr = 1000506001

(** [VK_KHR_cooperative_matrix]. The extension NAME is what gates the whole
    path: see [Vulkan_api_device.probe_coopmat] for the measurement that makes
    checking it mandatory rather than tidy. *)
let vk_khr_cooperative_matrix_extension_name = "VK_KHR_cooperative_matrix"

let vk_khr_shader_float16_int8_extension_name = "VK_KHR_shader_float16_int8"

let vk_khr_16bit_storage_extension_name = "VK_KHR_16bit_storage"

let vk_khr_8bit_storage_extension_name = "VK_KHR_8bit_storage"

let vk_khr_vulkan_memory_model_extension_name = "VK_KHR_vulkan_memory_model"

(** VkExtensionProperties: [char extensionName[256]] then [uint32 specVersion].
*)
type vk_extension_properties

let vk_extension_properties : vk_extension_properties structure typ =
  structure "VkExtensionProperties"

let ext_props_extensionName =
  field vk_extension_properties "extensionName" (array 256 char)

let ext_props_specVersion = field vk_extension_properties "specVersion" uint32_t

let () = seal vk_extension_properties

(** VkPhysicalDeviceFeatures2. [features] is embedded BY VALUE, and that is safe
    here only because {!vk_physical_device_features} is modelled exactly — all
    55 [VkBool32] and nothing omitted — unlike the properties struct. *)
type vk_physical_device_features_2

let vk_physical_device_features_2 : vk_physical_device_features_2 structure typ
    =
  structure "VkPhysicalDeviceFeatures2"

let features2_sType = field vk_physical_device_features_2 "sType" uint32_t

let features2_pNext = field vk_physical_device_features_2 "pNext" (ptr void)

let features2_features =
  field vk_physical_device_features_2 "features" vk_physical_device_features

let () = seal vk_physical_device_features_2

(** VkPhysicalDeviceProperties2, with the embedded [VkPhysicalDeviceProperties]
    modelled as an opaque blob.

    The blob must be at least [sizeof(VkPhysicalDeviceProperties)], because the
    driver writes the whole thing. That is 4 x uint32 + deviceType +
    deviceName[256] + pipelineCacheUUID[16] + VkPhysicalDeviceLimits (504) +
    VkPhysicalDeviceSparseProperties (20) ~= 824 bytes; 1024 is comfortably past
    it and costs nothing, since exactly one of these is made per device query.
    An [array _ char] has alignment 1, and the blob's offset (16, after a 4-byte
    [sType] and an 8-aligned [pNext]) is already 8-aligned, so the layout
    matches the C struct without a named-field model of the limits. *)
type vk_physical_device_properties_2

let vk_physical_device_properties_2_blob_bytes = 1024

let vk_physical_device_properties_2 :
    vk_physical_device_properties_2 structure typ =
  structure "VkPhysicalDeviceProperties2"

let properties2_sType = field vk_physical_device_properties_2 "sType" uint32_t

let properties2_pNext = field vk_physical_device_properties_2 "pNext" (ptr void)

let properties2_properties =
  field
    vk_physical_device_properties_2
    "properties"
    (array vk_physical_device_properties_2_blob_bytes char)

let () = seal vk_physical_device_properties_2

(** VkPhysicalDeviceShaderFloat16Int8Features *)
type vk_physical_device_shader_float16_int8_features

let vk_physical_device_shader_float16_int8_features :
    vk_physical_device_shader_float16_int8_features structure typ =
  structure "VkPhysicalDeviceShaderFloat16Int8Features"

let f16i8_sType =
  field vk_physical_device_shader_float16_int8_features "sType" uint32_t

let f16i8_pNext =
  field vk_physical_device_shader_float16_int8_features "pNext" (ptr void)

let f16i8_shaderFloat16 =
  field vk_physical_device_shader_float16_int8_features "shaderFloat16" uint32_t

let f16i8_shaderInt8 =
  field vk_physical_device_shader_float16_int8_features "shaderInt8" uint32_t

let () = seal vk_physical_device_shader_float16_int8_features

(** VkPhysicalDevice16BitStorageFeatures *)
type vk_physical_device_16bit_storage_features

let vk_physical_device_16bit_storage_features :
    vk_physical_device_16bit_storage_features structure typ =
  structure "VkPhysicalDevice16BitStorageFeatures"

let storage16_sType =
  field vk_physical_device_16bit_storage_features "sType" uint32_t

let storage16_pNext =
  field vk_physical_device_16bit_storage_features "pNext" (ptr void)

let storage16_storageBuffer16BitAccess =
  field
    vk_physical_device_16bit_storage_features
    "storageBuffer16BitAccess"
    uint32_t

let storage16_uniformAndStorageBuffer16BitAccess =
  field
    vk_physical_device_16bit_storage_features
    "uniformAndStorageBuffer16BitAccess"
    uint32_t

let storage16_storagePushConstant16 =
  field
    vk_physical_device_16bit_storage_features
    "storagePushConstant16"
    uint32_t

let storage16_storageInputOutput16 =
  field
    vk_physical_device_16bit_storage_features
    "storageInputOutput16"
    uint32_t

let () = seal vk_physical_device_16bit_storage_features

(** VkPhysicalDevice8BitStorageFeatures — backlog-62 slice 3.

    Field order and count are pinned against [vulkan_core.h]; the struct is
    passed by ADDRESS in a pNext chain, so a short or misordered layout is read
    past its end by the driver rather than diagnosed. Three feature booleans,
    the last two of which Sarek never requests. *)
type vk_physical_device_8bit_storage_features

let vk_physical_device_8bit_storage_features :
    vk_physical_device_8bit_storage_features structure typ =
  structure "VkPhysicalDevice8BitStorageFeatures"

let storage8_sType =
  field vk_physical_device_8bit_storage_features "sType" uint32_t

let storage8_pNext =
  field vk_physical_device_8bit_storage_features "pNext" (ptr void)

let storage8_storageBuffer8BitAccess =
  field
    vk_physical_device_8bit_storage_features
    "storageBuffer8BitAccess"
    uint32_t

let storage8_uniformAndStorageBuffer8BitAccess =
  field
    vk_physical_device_8bit_storage_features
    "uniformAndStorageBuffer8BitAccess"
    uint32_t

let storage8_storagePushConstant8 =
  field vk_physical_device_8bit_storage_features "storagePushConstant8" uint32_t

let () = seal vk_physical_device_8bit_storage_features

(** VkPhysicalDeviceVulkanMemoryModelFeatures — backlog-62 slice 3.

    [GL_KHR_memory_scope_semantics] is a hard prerequisite of
    [GL_KHR_cooperative_matrix] in glslang, and it lowers to
    [OpCapability VulkanMemoryModel]. Without [vulkanMemoryModel] enabled on the
    logical device, a cooperative-matrix shader is a specification violation
    however correct its results look. *)
type vk_physical_device_vulkan_memory_model_features

let vk_physical_device_vulkan_memory_model_features :
    vk_physical_device_vulkan_memory_model_features structure typ =
  structure "VkPhysicalDeviceVulkanMemoryModelFeatures"

let memmodel_sType =
  field vk_physical_device_vulkan_memory_model_features "sType" uint32_t

let memmodel_pNext =
  field vk_physical_device_vulkan_memory_model_features "pNext" (ptr void)

let memmodel_vulkanMemoryModel =
  field
    vk_physical_device_vulkan_memory_model_features
    "vulkanMemoryModel"
    uint32_t

let memmodel_vulkanMemoryModelDeviceScope =
  field
    vk_physical_device_vulkan_memory_model_features
    "vulkanMemoryModelDeviceScope"
    uint32_t

let memmodel_vulkanMemoryModelAvailabilityVisibilityChains =
  field
    vk_physical_device_vulkan_memory_model_features
    "vulkanMemoryModelAvailabilityVisibilityChains"
    uint32_t

let () = seal vk_physical_device_vulkan_memory_model_features

(** VkPhysicalDeviceCooperativeMatrixFeaturesKHR *)
type vk_physical_device_cooperative_matrix_features

let vk_physical_device_cooperative_matrix_features :
    vk_physical_device_cooperative_matrix_features structure typ =
  structure "VkPhysicalDeviceCooperativeMatrixFeaturesKHR"

let coopmat_feat_sType =
  field vk_physical_device_cooperative_matrix_features "sType" uint32_t

let coopmat_feat_pNext =
  field vk_physical_device_cooperative_matrix_features "pNext" (ptr void)

let coopmat_feat_cooperativeMatrix =
  field
    vk_physical_device_cooperative_matrix_features
    "cooperativeMatrix"
    uint32_t

let coopmat_feat_robustBufferAccess =
  field
    vk_physical_device_cooperative_matrix_features
    "cooperativeMatrixRobustBufferAccess"
    uint32_t

let () = seal vk_physical_device_cooperative_matrix_features

(** VkPhysicalDeviceSubgroupProperties. [subgroupSize] is the one field this
    backend reads, and it replaces a hard-coded 32 in [Vulkan_plugin_base]. *)
type vk_physical_device_subgroup_properties

let vk_physical_device_subgroup_properties :
    vk_physical_device_subgroup_properties structure typ =
  structure "VkPhysicalDeviceSubgroupProperties"

let subgroup_props_sType =
  field vk_physical_device_subgroup_properties "sType" uint32_t

let subgroup_props_pNext =
  field vk_physical_device_subgroup_properties "pNext" (ptr void)

let subgroup_props_subgroupSize =
  field vk_physical_device_subgroup_properties "subgroupSize" uint32_t

let subgroup_props_supportedStages =
  field vk_physical_device_subgroup_properties "supportedStages" uint32_t

let subgroup_props_supportedOperations =
  field vk_physical_device_subgroup_properties "supportedOperations" uint32_t

let subgroup_props_quadOperationsInAllStages =
  field
    vk_physical_device_subgroup_properties
    "quadOperationsInAllStages"
    uint32_t

let () = seal vk_physical_device_subgroup_properties

(** VkPhysicalDeviceDriverProperties. This is the driver KEY that
    docs/fp-contraction-policy.md §11.7 asks for: [driverID] is an enumerant (3
    = [VK_DRIVER_ID_MESA_RADV]), and [driverInfo] carries the Mesa version
    string, so an allowlist can be keyed on a driver rather than on a substring
    of a device name. *)
type vk_physical_device_driver_properties

let vk_physical_device_driver_properties :
    vk_physical_device_driver_properties structure typ =
  structure "VkPhysicalDeviceDriverProperties"

let driver_props_sType =
  field vk_physical_device_driver_properties "sType" uint32_t

let driver_props_pNext =
  field vk_physical_device_driver_properties "pNext" (ptr void)

let driver_props_driverID =
  field vk_physical_device_driver_properties "driverID" uint32_t

let driver_props_driverName =
  field vk_physical_device_driver_properties "driverName" (array 256 char)

let driver_props_driverInfo =
  field vk_physical_device_driver_properties "driverInfo" (array 256 char)

let driver_props_conformanceVersion =
  field
    vk_physical_device_driver_properties
    "conformanceVersion"
    (array 4 uint8_t)

let () = seal vk_physical_device_driver_properties

(** VkCooperativeMatrixPropertiesKHR — one advertised configuration.

    [AType]/[BType]/[CType]/[ResultType] are [VkComponentTypeKHR] and [scope] is
    [VkScopeKHR]; both are C enums, hence [int32_t]. For the enumerant values,
    read the constants below — NOT a restatement here.

    There was a restatement here, and it was wrong (f16 = 3, f32 = 4, s8 = 7,
    s32 = 9, u8 = 0, u32 = 1: the correct values rotated), in the one file whose
    entire narrative is a wrong enumerant table producing plausible garbage. It
    survived because [test_vulkan_coopmat_enumerants] pins the CONSTANTS and
    nothing pins the PROSE — the same asymmetry, one layer up. Hence no second
    copy of the numbers: a comment that cannot disagree with the code is the
    only kind worth having here. *)
type vk_cooperative_matrix_properties

let vk_cooperative_matrix_properties :
    vk_cooperative_matrix_properties structure typ =
  structure "VkCooperativeMatrixPropertiesKHR"

let coopmat_props_sType =
  field vk_cooperative_matrix_properties "sType" uint32_t

let coopmat_props_pNext =
  field vk_cooperative_matrix_properties "pNext" (ptr void)

let coopmat_props_MSize =
  field vk_cooperative_matrix_properties "MSize" uint32_t

let coopmat_props_NSize =
  field vk_cooperative_matrix_properties "NSize" uint32_t

let coopmat_props_KSize =
  field vk_cooperative_matrix_properties "KSize" uint32_t

let coopmat_props_AType = field vk_cooperative_matrix_properties "AType" int32_t

let coopmat_props_BType = field vk_cooperative_matrix_properties "BType" int32_t

let coopmat_props_CType = field vk_cooperative_matrix_properties "CType" int32_t

let coopmat_props_ResultType =
  field vk_cooperative_matrix_properties "ResultType" int32_t

let coopmat_props_saturatingAccumulation =
  field vk_cooperative_matrix_properties "saturatingAccumulation" uint32_t

let coopmat_props_scope = field vk_cooperative_matrix_properties "scope" int32_t

let () = seal vk_cooperative_matrix_properties

(** {2 VkComponentTypeKHR and VkScopeKHR enumerants}

    Verbatim from Khronos [vulkan_core.h]. These are NOT in a memorable order —
    the float types come first and the integer types are interleaved by width
    rather than by signedness — and an earlier draft of this file guessed them
    wrong in a way that produced entirely plausible garbage: [f16 * s8 -> s32]
    and [u8 * u8 -> u8] configurations that no hardware advertises, with eight
    of the fourteen real ones silently dropped as unrepresentable. Nothing about
    the resulting values looked malformed. They are pinned by
    [test_vulkan_coopmat_enumerants] against the same source. *)

let vk_component_type_float16 = 0l

let vk_component_type_float32 = 1l

let vk_component_type_float64 = 2l

let vk_component_type_sint8 = 3l

let vk_component_type_sint16 = 4l

let vk_component_type_sint32 = 5l

let vk_component_type_sint64 = 6l

let vk_component_type_uint8 = 7l

let vk_component_type_uint16 = 8l

let vk_component_type_uint32 = 9l

let vk_component_type_uint64 = 10l

let vk_scope_device = 1l

let vk_scope_workgroup = 2l

let vk_scope_subgroup = 3l

let vk_scope_queue_family = 5l
