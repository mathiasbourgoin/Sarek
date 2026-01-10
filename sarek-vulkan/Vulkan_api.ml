(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Vulkan API - High-Level Wrappers
 *
 * Provides a safe, OCaml-friendly interface to Vulkan compute functionality.
 * Handles error checking, resource management, and type conversions.
 *
 * Note: This module focuses on compute shaders only (no graphics).
 ******************************************************************************)

open Ctypes
open Vulkan_types
open Vulkan_bindings
open Spoc_framework_registry

(** memcpy from libc for memory transfers *)
let memcpy dst src size =
  let memcpy_c =
    Foreign.foreign
      "memcpy"
      (ptr void @-> ptr void @-> size_t @-> returning (ptr void))
  in
  memcpy_c dst src size

(** Helper to convert OCaml int to Unsigned.UInt32.t *)
let u32 = Unsigned.UInt32.of_int

(** {1 Exceptions} *)

exception Vulkan_error of vk_result * string

(** Check Vulkan result and raise exception on error *)
let check (ctx : string) (result : vk_result) : unit =
  match result with
  | VK_SUCCESS -> ()
  | err ->
      Spoc_core.Log.errorf
        Spoc_core.Log.Device
        "[Vulkan] %s failed with %s"
        ctx
        (string_of_vk_result err) ;
      raise (Vulkan_error (err, ctx))

(** {1 SPIR-V Compilation} *)

(** Compile GLSL to SPIR-V using glslangValidator. Requires glslangValidator in
    PATH. *)
let compile_glsl_to_spirv_cli ~(entry_point : string) (glsl_source : string) :
    string =
  (* Write GLSL to temp file *)
  let glsl_file = Filename.temp_file "sarek_" ".comp" in
  let spirv_file = Filename.temp_file "sarek_" ".spv" in
  let oc = open_out glsl_file in
  output_string oc glsl_source ;
  close_out oc ;

  (* Compile with glslangValidator *)
  (* NOTE: Don't use --target-env vulkan1.1 - it changes storage classes from
     Uniform to StorageBuffer which may cause issues with some drivers *)
  let _entry_point = entry_point in
  let cmd =
    Printf.sprintf
      "glslangValidator -V -S comp -o %s %s 2>&1"
      spirv_file
      glsl_file
  in
  Spoc_core.Log.debugf
    Spoc_core.Log.Device
    "[Vulkan] Compiling GLSL to SPIR-V: %s"
    cmd ;
  let ic = Unix.open_process_in cmd in
  let output = Buffer.create 256 in
  (try
     while true do
       Buffer.add_string output (input_line ic ^ "\n")
     done
   with End_of_file -> ()) ;
  let status = Unix.close_process_in ic in

  (* Check result *)
  (match status with
  | Unix.WEXITED 0 -> (
      Spoc_core.Log.debugf
        Spoc_core.Log.Device
        "[Vulkan] SPIR-V compilation succeeded, file size: %d bytes"
        Unix.((stat spirv_file).st_size) ;
      (* Save both SPIR-V and GLSL for debugging if logging is enabled *)
      if Spoc_core.Log.is_enabled Spoc_core.Log.Device then
        let debug_spirv = "/tmp/sarek_debug.spv" in
        let debug_glsl = "/tmp/sarek_debug.comp" in
        try
          let cmd =
            Printf.sprintf
              "cp %s %s && cp %s %s"
              spirv_file
              debug_spirv
              glsl_file
              debug_glsl
          in
          ignore (Sys.command cmd) ;
          Spoc_core.Log.debugf
            Spoc_core.Log.Device
            "[Vulkan] Saved SPIR-V to %s and GLSL to %s for debugging"
            debug_spirv
            debug_glsl
        with _ -> ())
  | _ ->
      (try Unix.unlink spirv_file with _ -> ()) ;
      Vulkan_error.raise_error
        (Vulkan_error.compilation_failed
           ""
           (Printf.sprintf
              "glslangValidator failed:\n%s"
              (Buffer.contents output)))) ;

  (* Clean up GLSL file *)
  (try Unix.unlink glsl_file with _ -> ()) ;

  (* Read SPIR-V binary *)
  let ic = open_in_bin spirv_file in
  let size = in_channel_length ic in
  let spirv = really_input_string ic size in
  close_in ic ;

  (* Clean up SPIR-V file *)
  (try Unix.unlink spirv_file with _ -> ()) ;

  spirv

(** Compile GLSL to SPIR-V using Shaderc if available, otherwise fallback to CLI
*)
let compile_glsl_to_spirv ~(entry_point : string) (glsl_source : string) :
    string =
  if Shaderc.is_available () then begin
    Spoc_core.Log.debug
      Spoc_core.Log.Device
      "[Vulkan] Compiling with libshaderc" ;
    try Shaderc.compile_glsl_to_spirv ~entry_point glsl_source
    with e ->
      Spoc_core.Log.errorf
        Spoc_core.Log.Device
        "[Vulkan] libshaderc failed: %s"
        (Printexc.to_string e) ;
      compile_glsl_to_spirv_cli ~entry_point glsl_source
  end
  else begin
    Spoc_core.Log.debug
      Spoc_core.Log.Device
      "[Vulkan] libshaderc not found, using glslangValidator" ;
    compile_glsl_to_spirv_cli ~entry_point glsl_source
  end

(** Check if glslangValidator is available *)
let glslang_available () : bool =
  try
    let ic = Unix.open_process_in "glslangValidator --version 2>&1" in
    let _ = input_line ic in
    let status = Unix.close_process_in ic in
    match status with Unix.WEXITED 0 -> true | _ -> false
  with _ -> false

(** {1 Device Management} *)

module Device = struct
  type t = {
    id : int;
    physical_device : vk_physical_device structure ptr;
    device : vk_device structure ptr;
    compute_queue : vk_queue structure ptr;
    queue_family : int;
    instance : vk_instance structure ptr;
    name : string;
    api_version : int * int * int;
    memory_properties : vk_physical_device_memory_properties structure;
    command_pool : vk_command_pool;
  }

  let instance_ref : vk_instance structure ptr option ref = ref None

  let initialized = ref false

  (* Cache for logical devices to ensure we don't create multiple vk_device handles
     for the same physical device, which would prevent sharing resources. *)
  let device_cache : (int, t) Hashtbl.t = Hashtbl.create 4

  (** Calculate total device memory from memory heaps

      Sums all memory heaps that have VK_MEMORY_HEAP_DEVICE_LOCAL_BIT set. This
      gives us the actual GPU memory for discrete GPUs, or the largest
      device-accessible heap for integrated GPUs (which may be shared system
      RAM).

      VK_MEMORY_HEAP_DEVICE_LOCAL_BIT = 0x00000001 per Vulkan spec. *)
  let get_total_device_memory
      (props : vk_physical_device_memory_properties structure) : int64 =
    let heap_count =
      Unsigned.UInt32.to_int (getf props mem_props_memoryHeapCount)
    in
    let heaps_arr = getf props mem_props_memoryHeaps in
    let vk_memory_heap_device_local_bit = 0x00000001 in

    let total = ref 0L in
    for i = 0 to heap_count - 1 do
      let heap = CArray.get heaps_arr i in
      let size = Unsigned.UInt64.to_int64 (getf heap mem_heap_size) in
      let flags = Unsigned.UInt32.to_int (getf heap mem_heap_flags) in

      (* Include heap if it has DEVICE_LOCAL_BIT set *)
      if flags land vk_memory_heap_device_local_bit <> 0 then
        total := Int64.add !total size
    done ;
    !total

  let init () =
    if not !initialized then begin
      if not (is_available ()) then
        Vulkan_error.raise_error (Vulkan_error.library_not_found "vulkan" []) ;
      initialized := true
    end

  (** Create Vulkan instance (shared among all devices) *)
  let get_or_create_instance () =
    match !instance_ref with
    | Some inst -> inst
    | None ->
        (* Application info *)
        let app_info = make vk_application_info in
        setf app_info app_info_sType (u32 vk_structure_type_application_info) ;
        setf app_info app_info_pNext null ;
        setf app_info app_info_pApplicationName (Some "Sarek") ;
        setf app_info app_info_applicationVersion (Unsigned.UInt32.of_int 1) ;
        setf app_info app_info_pEngineName (Some "SPOC") ;
        setf app_info app_info_engineVersion (Unsigned.UInt32.of_int 1) ;
        (* Vulkan 1.2 *)
        setf
          app_info
          app_info_apiVersion
          (Unsigned.UInt32.of_int ((1 lsl 22) lor (2 lsl 12) lor 0)) ;

        (* Instance create info *)
        let create_info = make vk_instance_create_info in
        setf
          create_info
          inst_create_sType
          (u32 vk_structure_type_instance_create_info) ;
        setf create_info inst_create_pNext null ;
        setf create_info inst_create_flags (Unsigned.UInt32.of_int 0) ;
        setf create_info inst_create_pApplicationInfo (addr app_info) ;
        setf
          create_info
          inst_create_enabledLayerCount
          (Unsigned.UInt32.of_int 0) ;
        setf
          create_info
          inst_create_ppEnabledLayerNames
          (from_voidp string null) ;
        setf
          create_info
          inst_create_enabledExtensionCount
          (Unsigned.UInt32.of_int 0) ;
        setf
          create_info
          inst_create_ppEnabledExtensionNames
          (from_voidp string null) ;

        let inst = allocate vk_instance_ptr (from_voidp vk_instance null) in
        check "vkCreateInstance" (vkCreateInstance (addr create_info) null inst) ;
        instance_ref := Some !@inst ;
        !@inst

  let count () =
    init () ;
    let inst = get_or_create_instance () in
    let n = allocate uint32_t (Unsigned.UInt32.of_int 0) in
    check
      "vkEnumeratePhysicalDevices"
      (vkEnumeratePhysicalDevices
         inst
         n
         (from_voidp vk_physical_device_ptr null)) ;
    Unsigned.UInt32.to_int !@n

  (** Find compute queue family index *)
  let find_compute_queue_family phys_dev =
    let count = allocate uint32_t (Unsigned.UInt32.of_int 0) in
    vkGetPhysicalDeviceQueueFamilyProperties
      phys_dev
      count
      (from_voidp vk_queue_family_properties null) ;
    let n = Unsigned.UInt32.to_int !@count in
    let props = CArray.make vk_queue_family_properties n in
    vkGetPhysicalDeviceQueueFamilyProperties phys_dev count (CArray.start props) ;
    (* Find first queue with compute support *)
    let rec find i =
      if i >= n then
        Vulkan_error.raise_error
          (Vulkan_error.context_error
             "queue family selection"
             "no compute queue family found")
      else
        let qf = CArray.get props i in
        let flags = getf qf queue_family_queueFlags in
        if Unsigned.UInt32.to_int flags land vk_queue_compute_bit <> 0 then i
        else find (i + 1)
    in
    find 0

  let get idx =
    match Hashtbl.find_opt device_cache idx with
    | Some dev -> dev
    | None ->
        init () ;
        let inst = get_or_create_instance () in

        (* Get physical device *)
        let count = allocate uint32_t (Unsigned.UInt32.of_int 0) in
        check
          "vkEnumeratePhysicalDevices"
          (vkEnumeratePhysicalDevices
             inst
             count
             (from_voidp vk_physical_device_ptr null)) ;
        let n = Unsigned.UInt32.to_int !@count in
        if idx >= n then
          Vulkan_error.raise_error (Vulkan_error.device_not_found idx n) ;

        let phys_devs = CArray.make vk_physical_device_ptr n in
        check
          "vkEnumeratePhysicalDevices"
          (vkEnumeratePhysicalDevices inst count (CArray.start phys_devs)) ;
        let phys_dev = CArray.get phys_devs idx in

        (* Get properties *)
        let props = make vk_physical_device_properties in
        vkGetPhysicalDeviceProperties phys_dev (addr props) ;
        let name_arr = getf props phys_props_deviceName in
        let name_chars = CArray.to_list name_arr in
        let name =
          String.init
            (min
               255
               (let rec find_nul i =
                  if i >= 255 then 255
                  else if List.nth name_chars i = '\000' then i
                  else find_nul (i + 1)
                in
                find_nul 0))
            (fun i -> List.nth name_chars i)
        in

        let api_ver =
          Unsigned.UInt32.to_int (getf props phys_props_apiVersion)
        in
        let api_major = api_ver lsr 22 in
        let api_minor = (api_ver lsr 12) land 0x3FF in
        let api_patch = api_ver land 0xFFF in

        (* Get memory properties *)
        let mem_props = make vk_physical_device_memory_properties in
        vkGetPhysicalDeviceMemoryProperties phys_dev (addr mem_props) ;

        (* Find compute queue family *)
        let queue_family = find_compute_queue_family phys_dev in

        (* Create logical device with compute queue *)
        let queue_priority = allocate float 1.0 in
        let queue_create_info = make vk_device_queue_create_info in
        setf
          queue_create_info
          dev_queue_create_sType
          (u32 vk_structure_type_device_queue_create_info) ;
        setf queue_create_info dev_queue_create_pNext null ;
        setf queue_create_info dev_queue_create_flags (Unsigned.UInt32.of_int 0) ;
        setf
          queue_create_info
          dev_queue_create_queueFamilyIndex
          (Unsigned.UInt32.of_int queue_family) ;
        setf
          queue_create_info
          dev_queue_create_queueCount
          (Unsigned.UInt32.of_int 1) ;
        setf queue_create_info dev_queue_create_pQueuePriorities queue_priority ;

        let dev_create_info = make vk_device_create_info in
        setf
          dev_create_info
          dev_create_sType
          (u32 vk_structure_type_device_create_info) ;
        setf dev_create_info dev_create_pNext null ;
        setf dev_create_info dev_create_flags (Unsigned.UInt32.of_int 0) ;
        setf
          dev_create_info
          dev_create_queueCreateInfoCount
          (Unsigned.UInt32.of_int 1) ;
        setf
          dev_create_info
          dev_create_pQueueCreateInfos
          (addr queue_create_info) ;
        setf
          dev_create_info
          dev_create_enabledLayerCount
          (Unsigned.UInt32.of_int 0) ;
        setf
          dev_create_info
          dev_create_ppEnabledLayerNames
          (from_voidp string null) ;
        setf
          dev_create_info
          dev_create_enabledExtensionCount
          (Unsigned.UInt32.of_int 0) ;
        setf
          dev_create_info
          dev_create_ppEnabledExtensionNames
          (from_voidp string null) ;
        setf dev_create_info dev_create_pEnabledFeatures null ;

        let device = allocate vk_device_ptr (from_voidp vk_device null) in
        check
          "vkCreateDevice"
          (vkCreateDevice phys_dev (addr dev_create_info) null device) ;

        (* Get compute queue *)
        let queue = allocate vk_queue_ptr (from_voidp vk_queue null) in
        vkGetDeviceQueue
          !@device
          (Unsigned.UInt32.of_int queue_family)
          (Unsigned.UInt32.of_int 0)
          queue ;

        (* Create persistent command pool *)
        let pool_info = make vk_command_pool_create_info in
        setf
          pool_info
          cmd_pool_create_sType
          (u32 vk_structure_type_command_pool_create_info) ;
        setf pool_info cmd_pool_create_pNext null ;
        setf pool_info cmd_pool_create_flags (Unsigned.UInt32.of_int 0x02) ;
        (* VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT *)
        setf
          pool_info
          cmd_pool_create_queueFamilyIndex
          (Unsigned.UInt32.of_int queue_family) ;

        let pool = allocate vk_command_pool vk_null_handle in
        check
          "vkCreateCommandPool"
          (vkCreateCommandPool !@device (addr pool_info) null pool) ;

        Spoc_core.Log.debugf
          Spoc_core.Log.Device
          "Vulkan device %d: %s (API %d.%d.%d)"
          idx
          name
          api_major
          api_minor
          api_patch ;

        let dev =
          {
            id = idx;
            physical_device = phys_dev;
            device = !@device;
            compute_queue = !@queue;
            queue_family;
            instance = inst;
            name;
            api_version = (api_major, api_minor, api_patch);
            memory_properties = mem_props;
            command_pool = !@pool;
          }
        in
        Hashtbl.add device_cache idx dev ;
        dev

  let set_current _dev = ()
  (* Vulkan doesn't have a global "current device" concept *)

  let synchronize dev = check "vkDeviceWaitIdle" (vkDeviceWaitIdle dev.device)

  let destroy dev =
    vkDestroyCommandPool dev.device dev.command_pool null ;
    vkDestroyDevice dev.device null
end

(** {1 Memory Management} *)

module Memory = struct
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
      if i >= count then
        Vulkan_error.raise_error
          (Vulkan_error.memory_allocation_failed
             0L
             "no suitable memory type found")
      else if type_filter land (1 lsl i) <> 0 then
        let mem_type = CArray.get types_arr i in
        let flags = getf mem_type mem_type_propertyFlags in
        if Unsigned.UInt32.to_int flags land properties = properties then i
        else find (i + 1)
      else find (i + 1)
    in
    find 0

  (* VK_WHOLE_SIZE = ~0ULL - map entire memory range *)
  let vk_whole_size = Unsigned.UInt64.max_int

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
         (vk_buffer_usage_storage_buffer_bit
        lor vk_buffer_usage_transfer_src_bit
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
    let mem_type_idx =
      find_memory_type
        device
        mem_type_bits
        (vk_memory_property_host_visible_bit
       lor vk_memory_property_host_coherent_bit)
    in

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

    (* Bind memory to buffer *)
    check
      "vkBindBufferMemory"
      (vkBindBufferMemory
         device.Device.device
         !@buffer
         !@memory
         (Unsigned.UInt64.of_int 0)) ;

    (* Map memory persistently like C example does *)
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

    {
      buffer = !@buffer;
      memory = !@memory;
      size;
      elem_size;
      device;
      mapped_ptr = Some !@data_ptr;
    }

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
         (vk_buffer_usage_storage_buffer_bit
        lor vk_buffer_usage_transfer_src_bit
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
      find_memory_type
        device
        mem_type_bits
        (vk_memory_property_host_visible_bit
       lor vk_memory_property_host_coherent_bit)
    in

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

  let free buf =
    (match buf.mapped_ptr with
    | Some _ -> vkUnmapMemory buf.device.Device.device buf.memory
    | None -> ()) ;
    vkDestroyBuffer buf.device.Device.device buf.buffer null ;
    vkFreeMemory buf.device.Device.device buf.memory null

  (** Vulkan doesn't expose device pointers like CUDA. Return 0 as placeholder.
      Binding uses the buffer handle directly via set_arg_buffer. *)
  let device_ptr _buf = Nativeint.zero

  (** Vulkan always uses explicit transfers (vkMapMemory/memcpy), never
      zero-copy *)
  let is_zero_copy _buf = false

  let host_to_device ~src ~dst =
    let bytes = Bigarray.Array1.size_in_bytes src in
    let data =
      match dst.mapped_ptr with
      | Some p -> p
      | None ->
          let data = allocate (ptr void) null in
          check
            "vkMapMemory"
            (vkMapMemory
               dst.device.Device.device
               dst.memory
               (Unsigned.UInt64.of_int 0)
               vk_whole_size
               (Unsigned.UInt32.of_int 0)
               data) ;
          !@data
    in
    let src_ptr = bigarray_start array1 src |> to_voidp in
    let _ = memcpy data src_ptr (Unsigned.Size_t.of_int bytes) in
    ()

  let device_to_host ~src ~dst =
    let bytes = Bigarray.Array1.size_in_bytes dst in
    let data =
      match src.mapped_ptr with
      | Some p -> p
      | None ->
          let data = allocate (ptr void) null in
          check
            "vkMapMemory"
            (vkMapMemory
               src.device.Device.device
               src.memory
               (Unsigned.UInt64.of_int 0)
               vk_whole_size
               (Unsigned.UInt32.of_int 0)
               data) ;
          !@data
    in
    let dst_ptr = bigarray_start array1 dst |> to_voidp in
    let _ = memcpy dst_ptr data (Unsigned.Size_t.of_int bytes) in
    ()

  let host_ptr_to_device ~src_ptr ~byte_size ~dst =
    (* Use persistent mapping if available *)
    let data =
      match dst.mapped_ptr with
      | Some p -> p
      | None ->
          let data = allocate (ptr void) null in
          check
            "vkMapMemory"
            (vkMapMemory
               dst.device.Device.device
               dst.memory
               (Unsigned.UInt64.of_int 0)
               vk_whole_size
               (Unsigned.UInt32.of_int 0)
               data) ;
          !@data
    in
    let _ = memcpy data src_ptr (Unsigned.Size_t.of_int byte_size) in
    ()

  let device_to_host_ptr ~src ~dst_ptr ~byte_size =
    (* Use persistent mapping if available *)
    let data =
      match src.mapped_ptr with
      | Some p -> p
      | None ->
          let data = allocate (ptr void) null in
          check
            "vkMapMemory"
            (vkMapMemory
               src.device.Device.device
               src.memory
               (Unsigned.UInt64.of_int 0)
               vk_whole_size
               (Unsigned.UInt32.of_int 0)
               data) ;
          !@data
    in
    let _ = memcpy dst_ptr data (Unsigned.Size_t.of_int byte_size) in
    ()

  let device_to_device ~src:_ ~dst:_ =
    Vulkan_error.raise_error
      (Vulkan_error.feature_not_supported "device_to_device transfer")

  let memset _buf _value =
    Vulkan_error.raise_error (Vulkan_error.feature_not_supported "memset")
end

(** {1 Stream Management} *)

module Stream = struct
  type t = {
    command_pool : vk_command_pool;
    command_buffer : vk_command_buffer structure ptr;
    fence : vk_fence;
    device : Device.t;
  }

  let create device =
    (* Create command pool *)
    let pool_info = make vk_command_pool_create_info in
    setf
      pool_info
      cmd_pool_create_sType
      (u32 vk_structure_type_command_pool_create_info) ;
    setf pool_info cmd_pool_create_pNext null ;
    setf pool_info cmd_pool_create_flags (Unsigned.UInt32.of_int 0x02) ;
    (* VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT *)
    setf
      pool_info
      cmd_pool_create_queueFamilyIndex
      (Unsigned.UInt32.of_int device.Device.queue_family) ;

    let pool = allocate vk_command_pool vk_null_handle in
    check
      "vkCreateCommandPool"
      (vkCreateCommandPool device.Device.device (addr pool_info) null pool) ;

    (* Allocate command buffer *)
    let alloc_info = make vk_command_buffer_allocate_info in
    setf
      alloc_info
      cmd_buf_alloc_sType
      (u32 vk_structure_type_command_buffer_allocate_info) ;
    setf alloc_info cmd_buf_alloc_pNext null ;
    setf alloc_info cmd_buf_alloc_commandPool !@pool ;
    setf alloc_info cmd_buf_alloc_level (u32 vk_command_buffer_level_primary) ;
    setf alloc_info cmd_buf_alloc_commandBufferCount (Unsigned.UInt32.of_int 1) ;

    let cmd_buf =
      allocate vk_command_buffer_ptr (from_voidp vk_command_buffer null)
    in
    check
      "vkAllocateCommandBuffers"
      (vkAllocateCommandBuffers device.Device.device (addr alloc_info) cmd_buf) ;

    (* Create fence in signaled state so first vkWaitForFences succeeds *)
    let fence_info = make vk_fence_create_info in
    setf fence_info fence_create_sType (u32 vk_structure_type_fence_create_info) ;
    setf fence_info fence_create_pNext null ;
    setf fence_info fence_create_flags (u32 vk_fence_create_signaled_bit) ;

    let fence = allocate vk_fence vk_null_handle in
    check
      "vkCreateFence"
      (vkCreateFence device.Device.device (addr fence_info) null fence) ;

    {command_pool = !@pool; command_buffer = !@cmd_buf; fence = !@fence; device}

  let destroy stream =
    vkDestroyFence stream.device.Device.device stream.fence null ;
    vkDestroyCommandPool stream.device.Device.device stream.command_pool null

  let synchronize stream =
    let fence_ptr = allocate vk_fence stream.fence in
    check
      "vkWaitForFences"
      (vkWaitForFences
         stream.device.Device.device
         (Unsigned.UInt32.of_int 1)
         fence_ptr
         vk_true
         (Unsigned.UInt64.of_int64 Int64.max_int)) ;
    ignore fence_ptr

  let default_streams : (int, t) Hashtbl.t = Hashtbl.create 4

  let default device =
    match Hashtbl.find_opt default_streams device.Device.id with
    | Some s -> s
    | None ->
        let s = create device in
        Hashtbl.add default_streams device.Device.id s ;
        s
end

(** {1 Event Management} *)

module Event = struct
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
end

(** {1 Kernel Management} *)

module Kernel = struct
  type t = {
    shader_module : vk_shader_module;
    pipeline : vk_pipeline;
    pipeline_layout : vk_pipeline_layout;
    descriptor_set_layout : vk_descriptor_set_layout;
    descriptor_pool : vk_descriptor_pool;
    descriptor_set : vk_descriptor_set;
    name : string;
    num_bindings : int;
    device : Device.t;
  }

  type arg =
    | ArgBuffer : _ Memory.buffer -> arg
    | ArgInt32 : int32 -> arg
    | ArgInt64 : int64 -> arg
    | ArgFloat32 : float -> arg
    | ArgFloat64 : float -> arg
    | ArgPtr : nativeint -> arg

  (** Existential wrapper to hide buffer type parameter *)
  type any_buffer = AnyBuf : 'a Memory.buffer -> any_buffer

  type args = {
    mutable bindings : (int * any_buffer) list;
    mutable descriptor_set : vk_descriptor_set;
    mutable push_constants : bytes option; (* Raw bytes for push constants *)
    mutable push_constant_offset : int;
        (* Current offset in push constant block *)
    mutable buffer_binding : int; (* Next available buffer binding index *)
  }

  (* Compilation cache *)
  let cache : (string, t) Hashtbl.t = Hashtbl.create 16

  (** Create shader module from SPIR-V *)
  let create_shader_module device spirv =
    let code_size = String.length spirv in
    (* SPIR-V must be 4-byte aligned *)
    if code_size mod 4 <> 0 then
      Vulkan_error.raise_error
        (Vulkan_error.module_load_failed
           code_size
           "SPIR-V size must be multiple of 4") ;

    (* Convert string to uint32 array *)
    let num_words = code_size / 4 in
    let code = CArray.make uint32_t num_words in
    for i = 0 to num_words - 1 do
      let b0 = Char.code spirv.[i * 4] in
      let b1 = Char.code spirv.[(i * 4) + 1] in
      let b2 = Char.code spirv.[(i * 4) + 2] in
      let b3 = Char.code spirv.[(i * 4) + 3] in
      let word = b0 lor (b1 lsl 8) lor (b2 lsl 16) lor (b3 lsl 24) in
      CArray.set code i (Unsigned.UInt32.of_int word)
    done ;

    let create_info = make vk_shader_module_create_info in
    setf
      create_info
      shader_create_sType
      (u32 vk_structure_type_shader_module_create_info) ;
    setf create_info shader_create_pNext null ;
    setf create_info shader_create_flags (Unsigned.UInt32.of_int 0) ;
    setf create_info shader_create_codeSize (Unsigned.Size_t.of_int code_size) ;
    setf create_info shader_create_pCode (CArray.start code) ;

    let shader_module = allocate vk_shader_module vk_null_handle in
    check
      "vkCreateShaderModule"
      (vkCreateShaderModule
         device.Device.device
         (addr create_info)
         null
         shader_module) ;
    !@shader_module

  (** Compile GLSL source to compute pipeline *)
  let compile device ~name ~source =
    (* 1. Check cache for SPIR-V *)
    let driver_version =
      let maj, min, patch = device.Device.api_version in
      Printf.sprintf "%d.%d.%d" maj min patch
    in
    let cache_key =
      Framework_cache.compute_key
        ~dev_name:device.Device.name
        ~driver_version
        ~source
    in

    let spirv =
      match Framework_cache.get ~key:cache_key with
      | Some data ->
          Spoc_core.Log.debugf
            Spoc_core.Log.Device
            "[Vulkan] Cache hit for kernel %s"
            name ;
          data
      | None ->
          Spoc_core.Log.debugf
            Spoc_core.Log.Device
            "[Vulkan] Cache miss for kernel %s, compiling..."
            name ;
          let data = compile_glsl_to_spirv ~entry_point:name source in
          Framework_cache.put ~key:cache_key ~data ;
          data
    in

    (* Create shader module *)
    let shader_module = create_shader_module device spirv in

    (* Count buffer bindings from source (look for "binding = N") *)
    let num_bindings =
      let count = ref 0 in
      let binding_re = Str.regexp "binding *= *[0-9]+" in
      let _ =
        try
          let pos = ref 0 in
          while true do
            let _ = Str.search_forward binding_re source !pos in
            incr count ;
            pos := Str.match_end ()
          done
        with Not_found -> ()
      in
      max 1 !count
    in

    (* Create descriptor set layout *)
    let bindings = CArray.make vk_descriptor_set_layout_binding num_bindings in
    for i = 0 to num_bindings - 1 do
      let binding = make vk_descriptor_set_layout_binding in
      setf binding dsl_binding_binding (Unsigned.UInt32.of_int i) ;
      setf
        binding
        dsl_binding_descriptorType
        (u32 vk_descriptor_type_storage_buffer) ;
      setf binding dsl_binding_descriptorCount (Unsigned.UInt32.of_int 1) ;
      setf
        binding
        dsl_binding_stageFlags
        (Unsigned.UInt32.of_int vk_shader_stage_compute_bit) ;
      setf binding dsl_binding_pImmutableSamplers null ;
      CArray.set bindings i binding
    done ;

    let dsl_create_info = make vk_descriptor_set_layout_create_info in
    setf
      dsl_create_info
      dsl_create_sType
      (u32 vk_structure_type_descriptor_set_layout_create_info) ;
    setf dsl_create_info dsl_create_pNext null ;
    setf dsl_create_info dsl_create_flags (Unsigned.UInt32.of_int 0) ;
    setf
      dsl_create_info
      dsl_create_bindingCount
      (Unsigned.UInt32.of_int num_bindings) ;
    setf dsl_create_info dsl_create_pBindings (CArray.start bindings) ;

    let dsl = allocate vk_descriptor_set_layout vk_null_handle in
    check
      "vkCreateDescriptorSetLayout"
      (vkCreateDescriptorSetLayout
         device.Device.device
         (addr dsl_create_info)
         null
         dsl) ;

    (* Create pipeline layout *)
    let pl_create_info = make vk_pipeline_layout_create_info in
    setf
      pl_create_info
      pl_create_sType
      (u32 vk_structure_type_pipeline_layout_create_info) ;
    setf pl_create_info pl_create_pNext null ;
    setf pl_create_info pl_create_flags (Unsigned.UInt32.of_int 0) ;
    setf pl_create_info pl_create_setLayoutCount (Unsigned.UInt32.of_int 1) ;
    setf pl_create_info pl_create_pSetLayouts dsl ;

    (* Add push constant range for scalar parameters *)
    let push_constant_range = make vk_push_constant_range in
    setf
      push_constant_range
      push_const_stageFlags
      (Unsigned.UInt32.of_int vk_shader_stage_compute_bit) ;
    setf push_constant_range push_const_offset (Unsigned.UInt32.of_int 0) ;
    setf push_constant_range push_const_size (Unsigned.UInt32.of_int 128) ;

    (* 128 bytes max for push constants *)
    setf
      pl_create_info
      pl_create_pushConstantRangeCount
      (Unsigned.UInt32.of_int 1) ;
    setf pl_create_info pl_create_pPushConstantRanges (addr push_constant_range) ;

    let pipeline_layout = allocate vk_pipeline_layout vk_null_handle in
    check
      "vkCreatePipelineLayout"
      (vkCreatePipelineLayout
         device.Device.device
         (addr pl_create_info)
         null
         pipeline_layout) ;

    (* Create compute pipeline *)
    let stage_info = make vk_pipeline_shader_stage_create_info in
    setf
      stage_info
      shader_stage_sType
      (u32 vk_structure_type_pipeline_shader_stage_create_info) ;
    setf stage_info shader_stage_pNext null ;
    setf stage_info shader_stage_flags (Unsigned.UInt32.of_int 0) ;
    setf
      stage_info
      shader_stage_stage
      (Unsigned.UInt32.of_int vk_shader_stage_compute_bit) ;
    setf stage_info shader_stage_module shader_module ;
    setf stage_info shader_stage_pName "main" ;
    setf stage_info shader_stage_pSpecializationInfo null ;

    let pipeline_info = make vk_compute_pipeline_create_info in
    setf
      pipeline_info
      compute_pipe_sType
      (u32 vk_structure_type_compute_pipeline_create_info) ;
    setf pipeline_info compute_pipe_pNext null ;
    setf pipeline_info compute_pipe_flags (Unsigned.UInt32.of_int 0) ;
    setf pipeline_info compute_pipe_stage stage_info ;
    setf pipeline_info compute_pipe_layout !@pipeline_layout ;
    setf pipeline_info compute_pipe_basePipelineHandle vk_null_handle ;
    setf pipeline_info compute_pipe_basePipelineIndex (Int32.of_int (-1)) ;

    let pipeline = allocate vk_pipeline vk_null_handle in
    let result =
      vkCreateComputePipelines
        device.Device.device
        vk_null_handle
        (Unsigned.UInt32.of_int 1)
        (addr pipeline_info)
        null
        pipeline
    in
    check "vkCreateComputePipelines" result ;

    (* Create descriptor pool *)
    let pool_size = make vk_descriptor_pool_size in
    setf pool_size pool_size_type (u32 vk_descriptor_type_storage_buffer) ;
    setf
      pool_size
      pool_size_descriptorCount
      (Unsigned.UInt32.of_int (num_bindings * 10)) ;

    let pool_info = make vk_descriptor_pool_create_info in
    setf
      pool_info
      desc_pool_sType
      (u32 vk_structure_type_descriptor_pool_create_info) ;
    setf pool_info desc_pool_pNext null ;
    setf pool_info desc_pool_flags (Unsigned.UInt32.of_int 0) ;
    setf pool_info desc_pool_maxSets (Unsigned.UInt32.of_int 10) ;
    setf pool_info desc_pool_poolSizeCount (Unsigned.UInt32.of_int 1) ;
    setf pool_info desc_pool_pPoolSizes (addr pool_size) ;

    let pool = allocate vk_descriptor_pool vk_null_handle in
    check
      "vkCreateDescriptorPool"
      (vkCreateDescriptorPool device.Device.device (addr pool_info) null pool) ;

    (* Allocate persistent descriptor set *)
    let ds_ai = make vk_descriptor_set_allocate_info in
    setf
      ds_ai
      desc_set_alloc_sType
      (u32 vk_structure_type_descriptor_set_allocate_info) ;
    setf ds_ai desc_set_alloc_pNext null ;
    setf ds_ai desc_set_alloc_descriptorPool !@pool ;
    setf ds_ai desc_set_alloc_descriptorSetCount (u32 1) ;
    (* Keep this allocation alive - it's passed by pointer to Vulkan *)
    let dsl_ptr = allocate vk_descriptor_set_layout !@dsl in
    setf ds_ai desc_set_alloc_pSetLayouts dsl_ptr ;

    let desc_set = allocate vk_descriptor_set vk_null_handle in
    check
      "vkAllocateDescriptorSets"
      (vkAllocateDescriptorSets device.Device.device (addr ds_ai) desc_set) ;
    ignore dsl_ptr ;
    {
      shader_module;
      pipeline = !@pipeline;
      pipeline_layout = !@pipeline_layout;
      descriptor_set_layout = !@dsl;
      descriptor_pool = !@pool;
      descriptor_set = !@desc_set;
      name;
      num_bindings;
      device;
    }

  let compile_cached device ~name ~source =
    let key =
      Printf.sprintf
        "%d:%s"
        device.Device.id
        (Digest.string source |> Digest.to_hex)
    in
    match Hashtbl.find_opt cache key with
    | Some k -> k
    | None ->
        let k = compile device ~name ~source in
        Hashtbl.add cache key k ;
        k

  let clear_cache () =
    Hashtbl.iter
      (fun _ k ->
        vkDestroyPipeline k.device.Device.device k.pipeline null ;
        vkDestroyPipelineLayout k.device.Device.device k.pipeline_layout null ;
        vkDestroyDescriptorPool k.device.Device.device k.descriptor_pool null ;
        vkDestroyDescriptorSetLayout
          k.device.Device.device
          k.descriptor_set_layout
          null ;
        vkDestroyShaderModule k.device.Device.device k.shader_module null)
      cache ;
    Hashtbl.clear cache

  let create_args () =
    {
      bindings = [];
      descriptor_set = vk_null_handle;
      push_constants = None;
      push_constant_offset = 0;
      buffer_binding = 0;
    }

  let set_arg_buffer args _idx buf =
    let binding = args.buffer_binding in
    args.bindings <- (binding, AnyBuf buf) :: args.bindings ;
    args.buffer_binding <- binding + 1

  let ensure_push_constants args =
    match args.push_constants with
    | Some pc -> pc
    | None ->
        (* Vulkan guarantees at least 128 bytes of push constants.
           This accommodates vector lengths + scalar arguments. *)
        let pc = Bytes.create 128 in
        args.push_constants <- Some pc ;
        pc

  let set_arg_int32 args _idx n =
    let pc = ensure_push_constants args in
    let offset = args.push_constant_offset in
    Bytes.set_int32_le pc offset n ;
    args.push_constant_offset <- offset + 4

  let set_arg_int64 args _idx n =
    let pc = ensure_push_constants args in
    let offset = args.push_constant_offset in
    Bytes.set_int64_le pc offset n ;
    args.push_constant_offset <- offset + 8

  let set_arg_float32 args _idx f =
    let pc = ensure_push_constants args in
    let offset = args.push_constant_offset in
    Bytes.set_int32_le pc offset (Int32.bits_of_float f) ;
    args.push_constant_offset <- offset + 4

  let set_arg_float64 args _idx f =
    let pc = ensure_push_constants args in
    let offset = args.push_constant_offset in
    Bytes.set_int64_le pc offset (Int64.bits_of_float f) ;
    args.push_constant_offset <- offset + 8

  let set_arg_ptr _args _idx _p =
    Vulkan_error.raise_error
      (Vulkan_error.feature_not_supported "raw pointer kernel arguments")

  let launch kernel ~args ~(grid : Spoc_framework.Framework_sig.dims)
      ~(block : Spoc_framework.Framework_sig.dims) ~shared_mem:_ ~stream =
    ignore block ;
    (* Vulkan doesn't use block size in dispatch, only grid *)
    let device = kernel.device in
    let u32 = Unsigned.UInt32.of_int in
    let u64 = Unsigned.UInt64.of_int in

    (* Helper to prevent GC from collecting Ctypes allocations.
       Unlike OpenCL which copies values, Vulkan reads from pointers during calls,
       so we must keep allocations alive through the entire function scope. *)
    let keep = Sys.opaque_identity in

    try
      (* 1. Get Stream (Command Buffer + Fence) *)
      let s = match stream with Some s -> s | None -> Stream.default device in
      let cmd_buf = s.Stream.command_buffer in
      let fence = s.Stream.fence in

      (* 2. Update Descriptor Set (reuse persistent set) *)
      let desc_set = kernel.descriptor_set in
      let num_bindings = List.length args.bindings in
      let writes = CArray.make vk_write_descriptor_set num_bindings in
      let buf_infos = CArray.make vk_descriptor_buffer_info num_bindings in

      List.iteri
        (fun i (binding_idx, any_buf) ->
          let buf_handle, buf_size, buf_elem_size =
            match any_buf with
            | AnyBuf buf -> (buf.buffer, buf.size, buf.elem_size)
          in

          let buf_info = CArray.get buf_infos i in
          setf buf_info desc_buf_buffer buf_handle ;
          setf buf_info desc_buf_offset (u64 0) ;
          setf buf_info desc_buf_range (u64 (buf_size * buf_elem_size)) ;

          let write = CArray.get writes i in
          setf
            write
            write_desc_sType
            (u32 vk_structure_type_write_descriptor_set) ;
          setf write write_desc_pNext null ;
          setf write write_desc_dstSet desc_set ;
          setf write write_desc_dstBinding (u32 binding_idx) ;
          setf write write_desc_dstArrayElement (u32 0) ;
          setf write write_desc_descriptorCount (u32 1) ;
          setf
            write
            write_desc_descriptorType
            (u32 vk_descriptor_type_storage_buffer) ;
          setf write write_desc_pImageInfo null ;
          setf write write_desc_pBufferInfo (addr buf_info) ;
          setf write write_desc_pTexelBufferView null)
        args.bindings ;

      if num_bindings > 0 then
        vkUpdateDescriptorSets
          device.Device.device
          (u32 num_bindings)
          (CArray.start writes)
          (u32 0)
          null ;
      ignore (keep writes) ;
      ignore (keep buf_infos) ;

      (* 3. Wait for any previous work to complete before reusing command buffer.
            This is critical: vkBeginCommandBuffer on an in-flight buffer is UB. *)
      let fence_ptr = allocate vk_fence fence in
      check
        "vkWaitForFences (pre-record)"
        (vkWaitForFences
           device.Device.device
           (u32 1)
           fence_ptr
           vk_true
           (Unsigned.UInt64.of_int64 Int64.max_int)) ;

      (* Reset fence after waiting, before recording new commands *)
      check
        "vkResetFences (pre-record)"
        (vkResetFences device.Device.device (u32 1) fence_ptr) ;
      ignore (keep fence_ptr) ;

      (* 4. Record Command Buffer *)
      let begin_info = make vk_command_buffer_begin_info in
      setf
        begin_info
        cmd_buf_begin_sType
        (u32 vk_structure_type_command_buffer_begin_info) ;
      setf begin_info cmd_buf_begin_pNext null ;
      setf
        begin_info
        cmd_buf_begin_flags
        (u32 vk_command_buffer_usage_one_time_submit_bit) ;
      setf begin_info cmd_buf_begin_pInheritanceInfo null ;

      check
        "vkBeginCommandBuffer"
        (vkBeginCommandBuffer cmd_buf (addr begin_info)) ;
      ignore (keep begin_info) ;

      vkCmdBindPipeline
        cmd_buf
        (u32 vk_pipeline_bind_point_compute)
        kernel.pipeline ;
      let desc_set_ptr = allocate vk_descriptor_set desc_set in
      vkCmdBindDescriptorSets
        cmd_buf
        (u32 vk_pipeline_bind_point_compute)
        kernel.pipeline_layout
        (u32 0)
        (u32 1)
        desc_set_ptr
        (u32 0)
        (from_voidp uint32_t null) ;
      ignore (keep desc_set_ptr) ;

      (* Push constants *)
      (match args.push_constants with
      | Some pc ->
          let len = Bytes.length pc in
          let pc_ptr = Ctypes.allocate_n Ctypes.char ~count:len in
          for i = 0 to len - 1 do
            pc_ptr +@ i <-@ Bytes.get pc i
          done ;
          vkCmdPushConstants
            cmd_buf
            kernel.pipeline_layout
            (u32 vk_shader_stage_compute_bit)
            (u32 0)
            (u32 len)
            (Ctypes.to_voidp pc_ptr) ;
          ignore (keep pc_ptr)
      | None -> ()) ;

      vkCmdDispatch cmd_buf (u32 grid.x) (u32 grid.y) (u32 grid.z) ;

      check "vkEndCommandBuffer" (vkEndCommandBuffer cmd_buf) ;

      (* 5. Submit *)
      let cmd_buf_ptr = allocate vk_command_buffer_ptr cmd_buf in
      let submit_info = make vk_submit_info in
      setf submit_info submit_sType (u32 vk_structure_type_submit_info) ;
      setf submit_info submit_pNext null ;
      setf submit_info submit_waitSemaphoreCount (u32 0) ;
      setf submit_info submit_pWaitSemaphores (from_voidp vk_semaphore null) ;
      setf submit_info submit_pWaitDstStageMask (from_voidp vk_flags null) ;
      setf submit_info submit_commandBufferCount (u32 1) ;
      setf submit_info submit_pCommandBuffers cmd_buf_ptr ;
      setf submit_info submit_signalSemaphoreCount (u32 0) ;
      setf submit_info submit_pSignalSemaphores (from_voidp vk_semaphore null) ;

      check
        "vkQueueSubmit"
        (vkQueueSubmit
           device.Device.compute_queue
           (u32 1)
           (addr submit_info)
           fence) ;
      ignore (keep cmd_buf_ptr) ;
      ignore (keep submit_info) ;

      (* 6. Wait for completion *)
      check
        "vkWaitForFences"
        (vkWaitForFences
           device.Device.device
           (u32 1)
           fence_ptr
           vk_true
           (Unsigned.UInt64.of_int64 Int64.max_int))
    with e ->
      Spoc_core.Log.errorf
        Spoc_core.Log.Device
        "[Vulkan] launch() EXCEPTION: %s"
        (Printexc.to_string e) ;
      (* Printexc.print_backtrace stderr ; *)
      raise e
end

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
