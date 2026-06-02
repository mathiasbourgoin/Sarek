(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

open Ctypes
open Vulkan_types
open Vulkan_bindings
open Vulkan_api_base
module Device = Vulkan_api_device

type 'a buffer = {
  buffer : vk_buffer;
  memory : vk_device_memory;
  size : int;
  elem_size : int;
  device : Device.t;
  mutable mapped_ptr : unit Ctypes.ptr option; (* Persistent mapping *)
}

(** Find suitable memory type *)
let find_memory_type dev type_filter properties =
  let props = dev.Device.memory_properties in
  let count = Unsigned.UInt32.to_int (getf props mem_props_memoryTypeCount) in
  let types_arr = getf props mem_props_memoryTypes in
  let rec find i =
    if i >= count then None
    else if type_filter land (1 lsl i) <> 0 then
      let mem_type = CArray.get types_arr i in
      let flags = getf mem_type mem_type_propertyFlags in
      if Unsigned.UInt32.to_int flags land properties = properties then Some i
      else find (i + 1)
    else find (i + 1)
  in
  find 0

(* VK_WHOLE_SIZE = ~0ULL - map entire memory range *)
let vk_whole_size = Unsigned.UInt64.max_int

(* Helper to allocate a command buffer, record commands, and submit *)
let run_single_time_command device record_fn =
  let alloc_info = make vk_command_buffer_allocate_info in
  setf
    alloc_info
    cmd_buf_alloc_sType
    (u32 vk_structure_type_command_buffer_allocate_info) ;
  setf alloc_info cmd_buf_alloc_level (u32 vk_command_buffer_level_primary) ;
  setf alloc_info cmd_buf_alloc_commandPool device.Device.command_pool ;
  setf alloc_info cmd_buf_alloc_commandBufferCount (Unsigned.UInt32.of_int 1) ;

  let cmd_buf_ptr =
    allocate vk_command_buffer_ptr (from_voidp vk_command_buffer null)
  in
  check
    "vkAllocateCommandBuffers"
    (vkAllocateCommandBuffers
       device.Device.device
       (addr alloc_info)
       cmd_buf_ptr) ;
  let cmd_buf = !@cmd_buf_ptr in

  let free_cmd_buf () =
    vkFreeCommandBuffers
      device.Device.device
      device.Device.command_pool
      (Unsigned.UInt32.of_int 1)
      cmd_buf_ptr
  in

  Fun.protect ~finally:free_cmd_buf (fun () ->
      let begin_info = make vk_command_buffer_begin_info in
      setf
        begin_info
        cmd_buf_begin_sType
        (u32 vk_structure_type_command_buffer_begin_info) ;
      setf
        begin_info
        cmd_buf_begin_flags
        (Unsigned.UInt32.of_int vk_command_buffer_usage_one_time_submit_bit) ;

      check
        "vkBeginCommandBuffer"
        (vkBeginCommandBuffer cmd_buf (addr begin_info)) ;

      record_fn cmd_buf ;

      check "vkEndCommandBuffer" (vkEndCommandBuffer cmd_buf) ;

      let submit_info = make vk_submit_info in
      setf submit_info submit_sType (u32 vk_structure_type_submit_info) ;
      setf submit_info submit_commandBufferCount (Unsigned.UInt32.of_int 1) ;
      setf submit_info submit_pCommandBuffers cmd_buf_ptr ;

      check
        "vkQueueSubmit"
        (vkQueueSubmit
           device.Device.compute_queue
           (Unsigned.UInt32.of_int 1)
           (addr submit_info)
           vk_null_handle) ;
      check "vkQueueWaitIdle" (vkQueueWaitIdle device.Device.compute_queue))

let create_staging_buffer device size usage =
  let buf_info = make vk_buffer_create_info in
  setf buf_info buf_create_sType (u32 vk_structure_type_buffer_create_info) ;
  setf buf_info buf_create_size (Unsigned.UInt64.of_int size) ;
  setf buf_info buf_create_usage (u32 usage) ;
  setf buf_info buf_create_sharingMode (u32 0) ;
  (* VK_SHARING_MODE_EXCLUSIVE *)
  setf buf_info buf_create_queueFamilyIndexCount (Unsigned.UInt32.of_int 0) ;
  setf buf_info buf_create_pQueueFamilyIndices (from_voidp uint32_t null) ;

  let buffer = allocate vk_buffer vk_null_handle in
  check
    "vkCreateBuffer (staging)"
    (vkCreateBuffer device.Device.device (addr buf_info) null buffer) ;

  (* Track allocated resources for cleanup on partial failure *)
  let memory = ref vk_null_handle in
  let cleanup () =
    if !memory <> vk_null_handle then
      vkFreeMemory device.Device.device !memory null ;
    vkDestroyBuffer device.Device.device !@buffer null
  in

  try
    let mem_reqs = make vk_memory_requirements in
    vkGetBufferMemoryRequirements device.Device.device !@buffer (addr mem_reqs) ;

    let mem_type_bits =
      Unsigned.UInt32.to_int (getf mem_reqs mem_req_memoryTypeBits)
    in

    let mem_type_idx =
      match
        find_memory_type
          device
          mem_type_bits
          (vk_memory_property_host_visible_bit
         lor vk_memory_property_host_coherent_bit)
      with
      | Some idx -> idx
      | None ->
          Vulkan_error.raise_error
            (Vulkan_error.context_error
               "memory allocation"
               "Failed to find HOST_VISIBLE | HOST_COHERENT memory for staging \
                buffer")
    in

    let alloc_info = make vk_memory_allocate_info in
    setf alloc_info mem_alloc_sType (u32 vk_structure_type_memory_allocate_info) ;
    setf alloc_info mem_alloc_pNext null ;
    setf alloc_info mem_alloc_allocationSize (getf mem_reqs mem_req_size) ;
    setf
      alloc_info
      mem_alloc_memoryTypeIndex
      (Unsigned.UInt32.of_int mem_type_idx) ;

    let mem_ptr = allocate vk_device_memory vk_null_handle in
    check
      "vkAllocateMemory (staging)"
      (vkAllocateMemory device.Device.device (addr alloc_info) null mem_ptr) ;
    memory := !@mem_ptr ;

    check
      "vkBindBufferMemory (staging)"
      (vkBindBufferMemory
         device.Device.device
         !@buffer
         !memory
         (Unsigned.UInt64.of_int 0)) ;

    let data_ptr = allocate (ptr void) null in
    check
      "vkMapMemory (staging)"
      (vkMapMemory
         device.Device.device
         !memory
         (Unsigned.UInt64.of_int 0)
         vk_whole_size
         (Unsigned.UInt32.of_int 0)
         data_ptr) ;

    (!@buffer, !memory, !@data_ptr)
  with e ->
    cleanup () ;
    raise e

let alloc device size kind =
  let elem_size = Ctypes_static.sizeof (Ctypes.typ_of_bigarray_kind kind) in
  let byte_size = size * elem_size in

  (* Create buffer *)
  let buf_info = make vk_buffer_create_info in
  setf buf_info buf_create_sType (u32 vk_structure_type_buffer_create_info) ;
  setf buf_info buf_create_pNext null ;
  setf buf_info buf_create_flags (Unsigned.UInt32.of_int 0) ;
  setf buf_info buf_create_size (Unsigned.UInt64.of_int byte_size) ;
  setf
    buf_info
    buf_create_usage
    (Unsigned.UInt32.of_int
       (vk_buffer_usage_storage_buffer_bit lor vk_buffer_usage_transfer_src_bit
      lor vk_buffer_usage_transfer_dst_bit)) ;
  setf buf_info buf_create_sharingMode (u32 0) ;
  (* VK_SHARING_MODE_EXCLUSIVE *)
  setf buf_info buf_create_queueFamilyIndexCount (Unsigned.UInt32.of_int 0) ;
  setf buf_info buf_create_pQueueFamilyIndices (from_voidp uint32_t null) ;

  let buffer = allocate vk_buffer vk_null_handle in
  check
    "vkCreateBuffer"
    (vkCreateBuffer device.Device.device (addr buf_info) null buffer) ;

  (* Get memory requirements *)
  let mem_reqs = make vk_memory_requirements in
  vkGetBufferMemoryRequirements device.Device.device !@buffer (addr mem_reqs) ;

  let mem_type_bits =
    Unsigned.UInt32.to_int (getf mem_reqs mem_req_memoryTypeBits)
  in

  (* Memory allocation strategy:
     1. Try DEVICE_LOCAL + HOST_VISIBLE + HOST_COHERENT (best for integrated GPUs)
     2. Try DEVICE_LOCAL only (discrete GPU VRAM, requires staging buffers)
     3. Fallback to HOST_VISIBLE + HOST_COHERENT (system RAM) *)
  let mem_type_idx, is_mappable =
    match
      find_memory_type
        device
        mem_type_bits
        (vk_memory_property_device_local_bit
       lor vk_memory_property_host_visible_bit
       lor vk_memory_property_host_coherent_bit)
    with
    | Some idx -> (idx, true) (* Best case: fast + mappable *)
    | None -> (
        match
          find_memory_type
            device
            mem_type_bits
            vk_memory_property_device_local_bit
        with
        | Some idx -> (idx, false) (* Discrete GPU VRAM *)
        | None -> (
            (* Fallback to HOST_VISIBLE | HOST_COHERENT (system RAM) *)
            match
              find_memory_type
                device
                mem_type_bits
                (vk_memory_property_host_visible_bit
               lor vk_memory_property_host_coherent_bit)
            with
            | Some idx -> (idx, true)
            | None ->
                Vulkan_error.raise_error
                  (Vulkan_error.context_error
                     "memory allocation"
                     "Failed to find suitable memory type")))
  in

  (* Track allocated resources for cleanup on partial failure *)
  let memory_ref = ref vk_null_handle in
  let cleanup () =
    if !memory_ref <> vk_null_handle then
      vkFreeMemory device.Device.device !memory_ref null ;
    vkDestroyBuffer device.Device.device !@buffer null
  in

  try
    (* Allocate memory *)
    let alloc_info = make vk_memory_allocate_info in
    setf alloc_info mem_alloc_sType (u32 vk_structure_type_memory_allocate_info) ;
    setf alloc_info mem_alloc_pNext null ;
    setf alloc_info mem_alloc_allocationSize (getf mem_reqs mem_req_size) ;
    setf
      alloc_info
      mem_alloc_memoryTypeIndex
      (Unsigned.UInt32.of_int mem_type_idx) ;

    let memory = allocate vk_device_memory vk_null_handle in
    check
      "vkAllocateMemory"
      (vkAllocateMemory device.Device.device (addr alloc_info) null memory) ;
    memory_ref := !@memory ;

    (* Bind memory to buffer *)
    check
      "vkBindBufferMemory"
      (vkBindBufferMemory
         device.Device.device
         !@buffer
         !@memory
         (Unsigned.UInt64.of_int 0)) ;

    (* Map memory persistently only if mappable (HOST_VISIBLE) *)
    let mapped_ptr =
      if is_mappable then (
        let data_ptr = allocate (ptr void) null in
        check
          "vkMapMemory (persistent)"
          (vkMapMemory
             device.Device.device
             !@memory
             (Unsigned.UInt64.of_int 0)
             vk_whole_size
             (Unsigned.UInt32.of_int 0)
             data_ptr) ;
        Some !@data_ptr)
      else None
    in

    {buffer = !@buffer; memory = !@memory; size; elem_size; device; mapped_ptr}
  with e ->
    cleanup () ;
    raise e

let alloc_custom device ~size ~elem_size =
  let byte_size = size * elem_size in

  let buf_info = make vk_buffer_create_info in
  setf buf_info buf_create_sType (u32 vk_structure_type_buffer_create_info) ;
  setf buf_info buf_create_pNext null ;
  setf buf_info buf_create_flags (Unsigned.UInt32.of_int 0) ;
  setf buf_info buf_create_size (Unsigned.UInt64.of_int byte_size) ;
  setf
    buf_info
    buf_create_usage
    (Unsigned.UInt32.of_int
       (vk_buffer_usage_storage_buffer_bit lor vk_buffer_usage_transfer_src_bit
      lor vk_buffer_usage_transfer_dst_bit)) ;
  setf buf_info buf_create_sharingMode (u32 0) ;
  setf buf_info buf_create_queueFamilyIndexCount (Unsigned.UInt32.of_int 0) ;
  setf buf_info buf_create_pQueueFamilyIndices (from_voidp uint32_t null) ;

  let buffer = allocate vk_buffer vk_null_handle in
  check
    "vkCreateBuffer"
    (vkCreateBuffer device.Device.device (addr buf_info) null buffer) ;

  let mem_reqs = make vk_memory_requirements in
  vkGetBufferMemoryRequirements device.Device.device !@buffer (addr mem_reqs) ;

  let mem_type_bits =
    Unsigned.UInt32.to_int (getf mem_reqs mem_req_memoryTypeBits)
  in
  let mem_type_idx =
    match
      find_memory_type
        device
        mem_type_bits
        (vk_memory_property_host_visible_bit
       lor vk_memory_property_host_coherent_bit)
    with
    | Some idx -> idx
    | None ->
        Vulkan_error.raise_error
          (Vulkan_error.context_error
             "memory allocation"
             "Failed to find HOST_VISIBLE | HOST_COHERENT memory")
  in

  let alloc_info = make vk_memory_allocate_info in
  setf alloc_info mem_alloc_sType (u32 vk_structure_type_memory_allocate_info) ;
  setf alloc_info mem_alloc_pNext null ;
  setf alloc_info mem_alloc_allocationSize (getf mem_reqs mem_req_size) ;
  setf
    alloc_info
    mem_alloc_memoryTypeIndex
    (Unsigned.UInt32.of_int mem_type_idx) ;

  (* Track allocated resources for cleanup on partial failure *)
  let memory_ref = ref vk_null_handle in
  let cleanup () =
    if !memory_ref <> vk_null_handle then
      vkFreeMemory device.Device.device !memory_ref null ;
    vkDestroyBuffer device.Device.device !@buffer null
  in

  try
    let memory = allocate vk_device_memory vk_null_handle in
    check
      "vkAllocateMemory"
      (vkAllocateMemory device.Device.device (addr alloc_info) null memory) ;
    memory_ref := !@memory ;

    check
      "vkBindBufferMemory"
      (vkBindBufferMemory
         device.Device.device
         !@buffer
         !@memory
         (Unsigned.UInt64.of_int 0)) ;

    (* Map memory persistently *)
    let data_ptr = allocate (ptr void) null in
    check
      "vkMapMemory (persistent custom)"
      (vkMapMemory
         device.Device.device
         !@memory
         (Unsigned.UInt64.of_int 0)
         vk_whole_size
         (Unsigned.UInt32.of_int 0)
         data_ptr) ;

    {
      buffer = !@buffer;
      memory = !@memory;
      size;
      elem_size;
      device;
      mapped_ptr = Some !@data_ptr;
    }
  with e ->
    cleanup () ;
    raise e

let free buf =
  (match buf.mapped_ptr with
  | Some _ -> vkUnmapMemory buf.device.Device.device buf.memory
  | None -> ()) ;
  vkDestroyBuffer buf.device.Device.device buf.buffer null ;
  vkFreeMemory buf.device.Device.device buf.memory null

(** Vulkan doesn't expose device pointers like CUDA. Return 0 as placeholder.
    Binding uses the buffer handle directly via set_arg_buffer. *)
let device_ptr _buf = Nativeint.zero

(** Vulkan always uses explicit transfers (vkMapMemory/memcpy), never zero-copy
*)
let is_zero_copy _buf = false

let host_to_device ~src ~dst =
  let bytes = Bigarray.Array1.size_in_bytes src in
  match dst.mapped_ptr with
  | Some p ->
      let src_ptr = bigarray_start array1 src |> to_voidp in
      let _ = memcpy p src_ptr (Unsigned.Size_t.of_int bytes) in
      ()
  | None ->
      (* Staging buffer transfer *)
      let staging_buf, staging_mem, staging_ptr =
        create_staging_buffer dst.device bytes vk_buffer_usage_transfer_src_bit
      in
      let free_staging () =
        vkDestroyBuffer dst.device.Device.device staging_buf null ;
        vkFreeMemory dst.device.Device.device staging_mem null
      in
      Fun.protect ~finally:free_staging (fun () ->
          let src_ptr = bigarray_start array1 src |> to_voidp in
          let _ = memcpy staging_ptr src_ptr (Unsigned.Size_t.of_int bytes) in

          run_single_time_command dst.device (fun cmd_buf ->
              let region = make vk_buffer_copy in
              setf region buffer_copy_srcOffset (Unsigned.UInt64.of_int 0) ;
              setf region buffer_copy_dstOffset (Unsigned.UInt64.of_int 0) ;
              setf region buffer_copy_size (Unsigned.UInt64.of_int bytes) ;

              vkCmdCopyBuffer
                cmd_buf
                staging_buf
                dst.buffer
                (Unsigned.UInt32.of_int 1)
                (addr region)))

let device_to_host ~src ~dst =
  let bytes = Bigarray.Array1.size_in_bytes dst in
  match src.mapped_ptr with
  | Some p ->
      let dst_ptr = bigarray_start array1 dst |> to_voidp in
      let _ = memcpy dst_ptr p (Unsigned.Size_t.of_int bytes) in
      ()
  | None ->
      (* Staging buffer transfer *)
      let staging_buf, staging_mem, staging_ptr =
        create_staging_buffer src.device bytes vk_buffer_usage_transfer_dst_bit
      in
      let free_staging () =
        vkDestroyBuffer src.device.Device.device staging_buf null ;
        vkFreeMemory src.device.Device.device staging_mem null
      in
      Fun.protect ~finally:free_staging (fun () ->
          run_single_time_command src.device (fun cmd_buf ->
              let region = make vk_buffer_copy in
              setf region buffer_copy_srcOffset (Unsigned.UInt64.of_int 0) ;
              setf region buffer_copy_dstOffset (Unsigned.UInt64.of_int 0) ;
              setf region buffer_copy_size (Unsigned.UInt64.of_int bytes) ;

              vkCmdCopyBuffer
                cmd_buf
                src.buffer
                staging_buf
                (Unsigned.UInt32.of_int 1)
                (addr region)) ;

          let dst_ptr = bigarray_start array1 dst |> to_voidp in
          let _ = memcpy dst_ptr staging_ptr (Unsigned.Size_t.of_int bytes) in
          ())

let host_ptr_to_device ~src_ptr ~byte_size ~dst =
  match dst.mapped_ptr with
  | Some p ->
      let _ = memcpy p src_ptr (Unsigned.Size_t.of_int byte_size) in
      ()
  | None ->
      let staging_buf, staging_mem, staging_ptr =
        create_staging_buffer
          dst.device
          byte_size
          vk_buffer_usage_transfer_src_bit
      in
      let free_staging () =
        vkDestroyBuffer dst.device.Device.device staging_buf null ;
        vkFreeMemory dst.device.Device.device staging_mem null
      in
      Fun.protect ~finally:free_staging (fun () ->
          let _ =
            memcpy staging_ptr src_ptr (Unsigned.Size_t.of_int byte_size)
          in

          run_single_time_command dst.device (fun cmd_buf ->
              let region = make vk_buffer_copy in
              setf region buffer_copy_srcOffset (Unsigned.UInt64.of_int 0) ;
              setf region buffer_copy_dstOffset (Unsigned.UInt64.of_int 0) ;
              setf region buffer_copy_size (Unsigned.UInt64.of_int byte_size) ;

              vkCmdCopyBuffer
                cmd_buf
                staging_buf
                dst.buffer
                (Unsigned.UInt32.of_int 1)
                (addr region)))

let device_to_host_ptr ~src ~dst_ptr ~byte_size =
  match src.mapped_ptr with
  | Some p ->
      let _ = memcpy dst_ptr p (Unsigned.Size_t.of_int byte_size) in
      ()
  | None ->
      let staging_buf, staging_mem, staging_ptr =
        create_staging_buffer
          src.device
          byte_size
          vk_buffer_usage_transfer_dst_bit
      in
      let free_staging () =
        vkDestroyBuffer src.device.Device.device staging_buf null ;
        vkFreeMemory src.device.Device.device staging_mem null
      in
      Fun.protect ~finally:free_staging (fun () ->
          run_single_time_command src.device (fun cmd_buf ->
              let region = make vk_buffer_copy in
              setf region buffer_copy_srcOffset (Unsigned.UInt64.of_int 0) ;
              setf region buffer_copy_dstOffset (Unsigned.UInt64.of_int 0) ;
              setf region buffer_copy_size (Unsigned.UInt64.of_int byte_size) ;

              vkCmdCopyBuffer
                cmd_buf
                src.buffer
                staging_buf
                (Unsigned.UInt32.of_int 1)
                (addr region)) ;

          let _ =
            memcpy dst_ptr staging_ptr (Unsigned.Size_t.of_int byte_size)
          in
          ())

let device_to_device ~src:_ ~dst:_ =
  Vulkan_error.raise_error
    (Vulkan_error.feature_not_supported "device_to_device transfer")

let memset _buf _value =
  Vulkan_error.raise_error (Vulkan_error.feature_not_supported "memset")
