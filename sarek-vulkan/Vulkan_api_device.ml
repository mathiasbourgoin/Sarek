(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

open Ctypes
open Vulkan_types
open Vulkan_bindings
open Vulkan_api_base

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
    device-accessible heap for integrated GPUs (which may be shared system RAM).

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
      setf create_info inst_create_enabledLayerCount (Unsigned.UInt32.of_int 0) ;
      setf create_info inst_create_ppEnabledLayerNames (from_voidp string null) ;
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
    (vkEnumeratePhysicalDevices inst n (from_voidp vk_physical_device_ptr null)) ;
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

      let api_ver = Unsigned.UInt32.to_int (getf props phys_props_apiVersion) in
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
      setf dev_create_info dev_create_pQueueCreateInfos (addr queue_create_info) ;
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
  Hashtbl.remove device_cache dev.id ;
  vkDestroyCommandPool dev.device dev.command_pool null ;
  vkDestroyDevice dev.device null
