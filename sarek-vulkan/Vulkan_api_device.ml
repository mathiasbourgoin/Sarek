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
  supports_fp64 : bool;
      (** Physical-device [shaderFloat64] feature, queried via
          [vkGetPhysicalDeviceFeatures] and mirrored into the logical device's
          [pEnabledFeatures] at creation time (see [get] below). *)
  supports_int64 : bool;
      (** Physical-device [shaderInt64] feature, on the same query-and-mirror
          path as {!supports_fp64}. Both halves matter and only the mirroring
          half is easy to forget: a kernel whose SPIR-V declares
          [OpCapability Int64] is legal only against a logical device that
          ENABLED the feature, so querying it without requesting it would still
          be the #142 defect. *)
  supports_fp16 : bool;
      (** [VkPhysicalDeviceShaderFloat16Int8Features.shaderFloat16], queried
          through the [VkPhysicalDeviceFeatures2] chain and REQUESTED in
          [VkDeviceCreateInfo.pNext] (backlog-62 slice 2). It is not in core
          [VkPhysicalDeviceFeatures], which is why it could not be plumbed on
          the #142 path.

          docs/fp-contraction-policy.md §7(b) records that RADV accepts f16
          shaders today without this feature enabled. That makes the existing
          f16 tripwire a measurement of an un-enabled path — fine for measuring
          a driver, not fine for shipping — and it is precisely the "supported
          but never requested" shape of #332. Enabling it here does NOT lift any
          f16 refusal; that is slice 3. *)
  storage_buffer_16bit : bool;
      (** [VkPhysicalDevice16BitStorageFeatures.storageBuffer16BitAccess],
          queried and requested on the same chain. Required before a shader may
          declare 16-bit types in a storage buffer, which every f16 or
          cooperative-matrix kernel that reads its operands from memory must do.
      *)
  driver_id : int;
      (** [VkPhysicalDeviceDriverProperties.driverID]. 3 is
          [VK_DRIVER_ID_MESA_RADV]. This is the driver KEY that
          docs/fp-contraction-policy.md §11.7 and the [is_anv_device] comment in
          [Test_helpers] both ask for: an allowlist keyed on a device-NAME
          substring would match a future non-Mesa driver on the same silicon. *)
  driver_name : string;  (** e.g. ["radv"]. *)
  driver_info : string;  (** e.g. ["Mesa 26.1.4-arch3.1"]. *)
  subgroup_size : int;
      (** [VkPhysicalDeviceSubgroupProperties.subgroupSize] — the real one, not
          a constant. Measured 64 on the RX 7900 XTX under radv / Mesa
          26.1.4-arch3.1, where [Vulkan_plugin_base] reported a hard-coded 32. A
          cooperative-matrix fragment is distributed across exactly this many
          invocations, so the wrong value is a wrong ABI.

          {b Guaranteed positive}: it is [fallback_subgroup_size] rather than
          the driver's zero when the query came back unwritten, so a consumer
          may divide by it. {!subgroup_size_probed} says which it is. *)
  subgroup_size_probed : bool;
      (** Whether {!subgroup_size} is a measurement rather than the fallback.
          Both local devices are probed; a [false] here is a test failure, not a
          degradation to be tolerated. *)
  coopmat_extension_advertised : bool;
      (** Whether [vkEnumerateDeviceExtensionProperties] listed
          [VK_KHR_cooperative_matrix] for this physical device. *)
  coopmat_enabled : bool;
      (** Whether the extension was advertised AND
          [VkPhysicalDeviceCooperativeMatrixFeaturesKHR.cooperativeMatrix] was
          true, so the extension and the feature were both REQUESTED at
          [vkCreateDevice].

          Recorded separately from {!coopmat} rather than derived from it,
          because deriving it is exactly the mistake this pair exists to catch:
          [ds_configs] comes from a query that answers even for devices without
          the extension, so a non-empty list is NOT evidence that the device
          supports anything. The implication
          [ds_configs <> [] ==> coopmat_enabled] is asserted by
          [test_vulkan_coopmat_capability], and it is what goes red if the
          extension check in [probe_coopmat] is ever removed — the gate test
          alone does not, because it compares the verdict against the same list.
      *)
  coopmat : Sarek_coopmat.device_support option;
      (** Cooperative-matrix support, [None] when it could not be probed at all
          (loader too old to resolve the entry point). See [probe_coopmat] for
          why an empty list and [None] must stay distinguishable, and for the
          measurement that makes the extension check load-bearing. *)
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
      (* Held explicitly rather than written through ctypes' [string_opt] view:
         that view allocates an anonymous C buffer rooted only by the fat
         pointer [setf] discards, so the name would dangle by the time the
         loader reads it in vkCreateInstance. *)
      let app_name = CArray.of_string "Sarek" in
      let engine_name = CArray.of_string "SPOC" in
      setf app_info app_info_sType (u32 vk_structure_type_application_info) ;
      setf app_info app_info_pNext null ;
      setf app_info app_info_pApplicationName (CArray.start app_name) ;
      setf app_info app_info_applicationVersion (Unsigned.UInt32.of_int 1) ;
      setf app_info app_info_pEngineName (CArray.start engine_name) ;
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
      (* [create_info] holds bare addresses into these; keep them reachable
         until the driver has finished reading them. *)
      ignore (Sys.opaque_identity app_info) ;
      ignore (Sys.opaque_identity app_name) ;
      ignore (Sys.opaque_identity engine_name) ;
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

(** {1 Extended feature and property probes (backlog-62 slice 2)} *)

(** Read a NUL-terminated fixed-size C char array into an OCaml string. *)
let string_of_char_array arr =
  let n = CArray.length arr in
  let buf = Buffer.create 64 in
  (try
     for i = 0 to n - 1 do
       let c = CArray.get arr i in
       if c = '\000' then raise Exit else Buffer.add_char buf c
     done
   with Exit -> ()) ;
  Buffer.contents buf

let u32_is_true v = Unsigned.UInt32.to_int v <> 0

(** Zero a struct's bytes before a [pNext] query.

    A driver that does not recognise an [sType] leaves the struct alone, so
    without this an unrecognised feature struct would be read out of
    uninitialised memory — and the failure direction is "feature present", on
    exactly the devices that lack it. [ctypes]' [make] does not zero. *)
let zero_struct (type a) (typ : a structure typ) (s : a structure) =
  let bytes = sizeof typ in
  let p = coerce (ptr typ) (ptr char) (addr s) in
  for i = 0 to bytes - 1 do
    p +@ i <-@ '\000'
  done

(** The device extension names a physical device advertises. *)
let device_extension_names phys_dev =
  let count = allocate uint32_t (Unsigned.UInt32.of_int 0) in
  check
    "vkEnumerateDeviceExtensionProperties"
    (vkEnumerateDeviceExtensionProperties
       phys_dev
       (from_voidp char null)
       count
       (from_voidp vk_extension_properties null)) ;
  let n = Unsigned.UInt32.to_int !@count in
  if n = 0 then []
  else begin
    let arr = CArray.make vk_extension_properties n in
    check
      "vkEnumerateDeviceExtensionProperties"
      (vkEnumerateDeviceExtensionProperties
         phys_dev
         (from_voidp char null)
         count
         (CArray.start arr)) ;
    List.init n (fun i ->
        string_of_char_array (getf (CArray.get arr i) ext_props_extensionName))
  end

type extended_features = {
  ef_shader_float16 : bool;
  ef_storage_buffer_16bit : bool;
  ef_cooperative_matrix : bool;
  ef_coopmat_robust_buffer_access : bool;
}

(** Query the three extension feature structs in one [VkPhysicalDeviceFeatures2]
    chain.

    Chaining a struct whose extension the device does not advertise is harmless
    — the driver skips an [sType] it does not know, and {!zero_struct}
    guarantees the fields then read false. What is NOT harmless is calling an
    extension's own entry point on such a device; see {!probe_coopmat}. *)
let query_extended_features phys_dev =
  let coopmat_f = make vk_physical_device_cooperative_matrix_features in
  zero_struct vk_physical_device_cooperative_matrix_features coopmat_f ;
  setf
    coopmat_f
    coopmat_feat_sType
    (u32 vk_structure_type_physical_device_cooperative_matrix_features_khr) ;
  setf coopmat_f coopmat_feat_pNext null ;

  let storage16 = make vk_physical_device_16bit_storage_features in
  zero_struct vk_physical_device_16bit_storage_features storage16 ;
  setf
    storage16
    storage16_sType
    (u32 vk_structure_type_physical_device_16bit_storage_features) ;
  setf storage16 storage16_pNext (to_voidp (addr coopmat_f)) ;

  let f16i8 = make vk_physical_device_shader_float16_int8_features in
  zero_struct vk_physical_device_shader_float16_int8_features f16i8 ;
  setf
    f16i8
    f16i8_sType
    (u32 vk_structure_type_physical_device_shader_float16_int8_features) ;
  setf f16i8 f16i8_pNext (to_voidp (addr storage16)) ;

  let features2 = make vk_physical_device_features_2 in
  zero_struct vk_physical_device_features_2 features2 ;
  setf
    features2
    features2_sType
    (u32 vk_structure_type_physical_device_features_2) ;
  setf features2 features2_pNext (to_voidp (addr f16i8)) ;

  vkGetPhysicalDeviceFeatures2 phys_dev (addr features2) ;
  (* The chain holds bare addresses into all four. *)
  ignore (Sys.opaque_identity features2) ;
  ignore (Sys.opaque_identity f16i8) ;
  ignore (Sys.opaque_identity storage16) ;
  ignore (Sys.opaque_identity coopmat_f) ;
  {
    ef_shader_float16 = u32_is_true (getf f16i8 f16i8_shaderFloat16);
    ef_storage_buffer_16bit =
      u32_is_true (getf storage16 storage16_storageBuffer16BitAccess);
    ef_cooperative_matrix =
      u32_is_true (getf coopmat_f coopmat_feat_cooperativeMatrix);
    ef_coopmat_robust_buffer_access =
      u32_is_true (getf coopmat_f coopmat_feat_robustBufferAccess);
  }

type extended_properties = {
  ep_driver_id : int;
  ep_driver_name : string;
  ep_driver_info : string;
  ep_subgroup_size : int;
      (** Always positive — see {!fallback_subgroup_size}. *)
  ep_subgroup_size_probed : bool;
      (** [false] when the driver left [subgroupSize] at the zero {!zero_struct}
          wrote, so {!ep_subgroup_size} is the fallback rather than a
          measurement. *)
}

(** Used only when [VkPhysicalDeviceSubgroupProperties] came back unwritten.

    [zero_struct] zeroes the struct before the query, so a driver that does not
    recognise the [sType] leaves [subgroupSize = 0] — and zero flowing into
    [warp_size] is worse than the wrong-but-usable 32 that preceded this work,
    because any consumer that divides by it faults instead of merely being
    wrong. 32 is that historical value, kept as the fallback for continuity and
    for nothing else.

    It is WRONG on both devices this project measures on, which report 64 (RX
    7900 XTX / RADV NAVI31 and the Ryzen 9 7950X iGPU / RADV RAPHAEL_MENDOCINO,
    radv / Mesa 26.1.4-arch3.1). So it must never become the silent normal:
    [ep_subgroup_size_probed] records which of the two a caller is holding, and
    [test_vulkan_coopmat_capability] asserts that every local device is PROBED —
    the fallback going live here is a test failure, not a quiet degradation. *)
let fallback_subgroup_size = 32

(** Driver identity and subgroup size, through one [VkPhysicalDeviceProperties2]
    chain. Both are core Vulkan 1.1 property structs, so no extension gate. *)
let query_extended_properties phys_dev =
  let subgroup = make vk_physical_device_subgroup_properties in
  zero_struct vk_physical_device_subgroup_properties subgroup ;
  setf
    subgroup
    subgroup_props_sType
    (u32 vk_structure_type_physical_device_subgroup_properties) ;
  setf subgroup subgroup_props_pNext null ;

  let driver = make vk_physical_device_driver_properties in
  zero_struct vk_physical_device_driver_properties driver ;
  setf
    driver
    driver_props_sType
    (u32 vk_structure_type_physical_device_driver_properties) ;
  setf driver driver_props_pNext (to_voidp (addr subgroup)) ;

  let props2 = make vk_physical_device_properties_2 in
  zero_struct vk_physical_device_properties_2 props2 ;
  setf
    props2
    properties2_sType
    (u32 vk_structure_type_physical_device_properties_2) ;
  setf props2 properties2_pNext (to_voidp (addr driver)) ;

  vkGetPhysicalDeviceProperties2 phys_dev (addr props2) ;
  ignore (Sys.opaque_identity props2) ;
  ignore (Sys.opaque_identity driver) ;
  ignore (Sys.opaque_identity subgroup) ;
  let reported_subgroup_size =
    Unsigned.UInt32.to_int (getf subgroup subgroup_props_subgroupSize)
  in
  {
    ep_driver_id = Unsigned.UInt32.to_int (getf driver driver_props_driverID);
    ep_driver_name = string_of_char_array (getf driver driver_props_driverName);
    ep_driver_info = string_of_char_array (getf driver driver_props_driverInfo);
    ep_subgroup_size =
      (if reported_subgroup_size > 0 then reported_subgroup_size
       else fallback_subgroup_size);
    ep_subgroup_size_probed = reported_subgroup_size > 0;
  }

(** {2 Cooperative-matrix configuration enumeration} *)

let component_type_of_enum (v : int32) : Sarek_coopmat.component_type option =
  if v = vk_component_type_float16 then Some Sarek_coopmat.Float16
  else if v = vk_component_type_float32 then Some Sarek_coopmat.Float32
  else if v = vk_component_type_uint8 then Some Sarek_coopmat.Uint8
  else if v = vk_component_type_sint8 then Some Sarek_coopmat.Sint8
  else if v = vk_component_type_uint32 then Some Sarek_coopmat.Uint32
  else if v = vk_component_type_sint32 then Some Sarek_coopmat.Sint32
  else None

let scope_of_enum (v : int32) : Sarek_coopmat.scope option =
  if v = vk_scope_subgroup then Some Sarek_coopmat.Subgroup
  else if v = vk_scope_workgroup then Some Sarek_coopmat.Workgroup
  else if v = vk_scope_device then Some Sarek_coopmat.Device_scope
  else if v = vk_scope_queue_family then Some Sarek_coopmat.Queue_family
  else None

(** Probe cooperative-matrix support for one physical device.

    {b The extension check is load-bearing and this is measured, not tidy.}
    [vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR] is an instance-level
    entry point that dispatches on the physical device handle, and on this
    workstation RADV answers it for BOTH local devices: the AMD Ryzen 9 7950X
    iGPU (RADV RAPHAEL_MENDOCINO), which does not advertise
    [VK_KHR_cooperative_matrix] and reports [cooperativeMatrix = false], still
    returns [VK_SUCCESS] and fourteen configurations — the same fourteen as the
    RX 7900 XTX. Calling an extension entry point on a device that does not
    support the extension is undefined behaviour, and here the undefined
    behaviour is a plausible, well-formed, entirely wrong answer.

    So a probe that populated the configuration list from the query alone would
    report the iGPU as fully cooperative-matrix capable, and every gate
    downstream would say [Available] for a device that cannot execute the
    instruction. The order below — extension advertised, THEN feature true, THEN
    query — is what makes the gate able to refuse.

    [None] is returned only when the loader cannot resolve the entry point at
    all; that is "not probed", and it refuses. A device that is probed and has
    nothing returns [Some] with an empty list, which is a different fact. *)
let probe_coopmat ~instance ~phys_dev ~extensions
    ~(features : extended_features) ~subgroup_size =
  let advertised =
    List.mem vk_khr_cooperative_matrix_extension_name extensions
  in
  if not (advertised && features.ef_cooperative_matrix) then
    Some
      {
        Sarek_coopmat.ds_configs = [];
        ds_robust_buffer_access = false;
        ds_subgroup_size = subgroup_size;
        ds_advertised_count = 0;
      }
  else
    match get_physical_device_cooperative_matrix_properties instance with
    | None -> None
    | Some query ->
        let count = allocate uint32_t (Unsigned.UInt32.of_int 0) in
        check
          "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR"
          (query
             phys_dev
             count
             (from_voidp vk_cooperative_matrix_properties null)) ;
        let n = Unsigned.UInt32.to_int !@count in
        let configs =
          if n = 0 then []
          else begin
            let arr = CArray.make vk_cooperative_matrix_properties n in
            for i = 0 to n - 1 do
              let e = CArray.get arr i in
              zero_struct vk_cooperative_matrix_properties e ;
              setf
                e
                coopmat_props_sType
                (u32 vk_structure_type_cooperative_matrix_properties_khr) ;
              setf e coopmat_props_pNext null
            done ;
            check
              "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR"
              (query phys_dev count (CArray.start arr)) ;
            List.filter_map
              (fun i ->
                let e = CArray.get arr i in
                let u f = Unsigned.UInt32.to_int (getf e f) in
                (* A configuration naming a component type or a scope this
                   build cannot represent is DROPPED, not approximated. The
                   list is only ever read to decide what may run, so dropping
                   an entry can only cause a refusal — the safe direction —
                   whereas mapping an unknown enumerant onto a neighbouring one
                   would admit an operation with different semantics. *)
                match
                  ( component_type_of_enum (getf e coopmat_props_AType),
                    component_type_of_enum (getf e coopmat_props_BType),
                    component_type_of_enum (getf e coopmat_props_CType),
                    component_type_of_enum (getf e coopmat_props_ResultType),
                    scope_of_enum (getf e coopmat_props_scope) )
                with
                | Some a, Some b, Some c, Some result, Some scope ->
                    Some
                      {
                        Sarek_coopmat.cfg_shape =
                          {
                            Sarek_coopmat.m = u coopmat_props_MSize;
                            n = u coopmat_props_NSize;
                            k = u coopmat_props_KSize;
                          };
                        cfg_a = a;
                        cfg_b = b;
                        cfg_c = c;
                        cfg_result = result;
                        cfg_saturating =
                          u coopmat_props_saturatingAccumulation <> 0;
                        cfg_scope = scope;
                      }
                | _ ->
                    Spoc_core.Log.debugf
                      Spoc_core.Log.Device
                      "Vulkan coopmat: dropping configuration %d (component \
                       type or scope not representable)"
                      i ;
                    None)
              (List.init n (fun i -> i))
          end
        in
        Some
          {
            Sarek_coopmat.ds_configs = configs;
            ds_robust_buffer_access = features.ef_coopmat_robust_buffer_access;
            ds_subgroup_size = subgroup_size;
            ds_advertised_count = n;
          }

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

      (* Query supported physical-device features so we only ever request a
         feature the hardware actually reports - requesting an unsupported
         feature in pEnabledFeatures fails vkCreateDevice outright. *)
      let supported_features = make vk_physical_device_features in
      vkGetPhysicalDeviceFeatures phys_dev (addr supported_features) ;
      let supported_bools = getf supported_features phys_features_bools in
      let feature_supported index =
        Unsigned.UInt32.to_int (CArray.get supported_bools index) <> 0
      in
      let supports_fp64 = feature_supported shader_float64_field_index in
      (* #142: [shader_int64_field_index] has existed in Vulkan_types since the
         features struct was modelled, and nothing read it. GLSL [int64_t]
         compiles to SPIR-V declaring [OpCapability Int64], which Vulkan makes
         conditional on this feature being ENABLED on the logical device
         (VUID-VkShaderModuleCreateInfo-pCode-08740) - not merely supported by
         the physical one. *)
      let supports_int64 = feature_supported shader_int64_field_index in

      (* backlog-62 slice 2. Everything below this line is queried through the
         Features2 / Properties2 pNext chains, which is the only way to reach
         shaderFloat16, storageBuffer16BitAccess, cooperativeMatrix, the driver
         identity and the subgroup size. *)
      let extensions = device_extension_names phys_dev in
      let ext = query_extended_features phys_dev in
      let props2 = query_extended_properties phys_dev in
      let coopmat =
        probe_coopmat
          ~instance:inst
          ~phys_dev
          ~extensions
          ~features:ext
          ~subgroup_size:props2.ep_subgroup_size
      in
      let has_ext name = List.mem name extensions in
      (* #332's lesson, applied — and applied in BOTH directions, which the
         first draft of this block got wrong.

         A feature must be REQUESTED, not merely supported: that is #332. But
         requiring an extension STRING for a feature that has since been
         promoted to core is the same error arriving from the other side, and
         it silently stops requesting a feature the device does support.
         [VK_KHR_16bit_storage] is core in Vulkan 1.1 and
         [VK_KHR_shader_float16_int8] is core in Vulkan 1.2; a device at those
         versions need not advertise the extension string at all, and on such a
         device an [advertised && supported] conjunction reads false and the
         feature is never enabled. Both local devices happen to advertise both
         strings, so the defect is invisible here on the hardware alone.

         Measured by simulation instead, 2026-07-27: hiding the two PROMOTED
         extension strings from [has_ext] — i.e. presenting an otherwise
         identical device that supports them only as core — the previous
         conjunction reported [shaderFloat16=false storageBuffer16=false] on a
         device that fully supports both, while the form below reports both
         true and vkCreateDevice still succeeds. The cooperative-matrix arm is
         unaffected in either run, which is the control: it is a real extension
         and its string requirement is correct.

         So each promoted feature is gated on the feature bit plus EITHER the
         core version that promoted it OR the extension string, and only
         [VK_KHR_cooperative_matrix] — which is a real extension, promoted to
         nothing — keeps an unconditional string requirement.

         The version that counts is the EFFECTIVE one: an application may not
         use a core feature above the [apiVersion] its instance requested, and
         [get_or_create_instance] above requests 1.2. Both promotions are at or
         below that, so nothing here is reachable-but-refused; the [min] is
         there so that lowering the instance version cannot silently start
         requesting features the application is not entitled to. *)
      let instance_api_version = (1, 2) in
      let effective_api_version =
        min (api_major, api_minor) instance_api_version
      in
      let api_at_least v = effective_api_version >= v in
      let want_fp16 =
        ext.ef_shader_float16
        && (api_at_least (1, 2)
           || has_ext vk_khr_shader_float16_int8_extension_name)
      in
      let want_storage16 =
        ext.ef_storage_buffer_16bit
        && (api_at_least (1, 1) || has_ext vk_khr_16bit_storage_extension_name)
      in
      let want_coopmat =
        ext.ef_cooperative_matrix
        && has_ext vk_khr_cooperative_matrix_extension_name
      in

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
      (* Device extensions, backlog-62 slice 2. Requested only when advertised;
         the list is empty on a device that advertises none, which restores the
         previous (count = 0, names = NULL) call exactly.

         The name buffers are held in [ext_name_arrays] and kept alive past
         vkCreateDevice explicitly. [CArray.of_string] is used rather than
         ctypes' [string] view for the same reason the instance's application
         name is: that view allocates an anonymous buffer rooted only by a fat
         pointer [setf] discards, so the string would dangle before the loader
         read it. *)
      let requested_extensions =
        List.filter_map
          (fun (wanted, name) ->
            (* [has_ext] here and not in [want_*]: a promoted feature may be
               wanted via its core version on a device that does not advertise
               the string, and requesting an unadvertised extension fails
               vkCreateDevice. The feature struct is still chained in either
               case, which is what actually enables the feature. *)
            if wanted && has_ext name then Some name else None)
          [
            (want_fp16, vk_khr_shader_float16_int8_extension_name);
            (want_storage16, vk_khr_16bit_storage_extension_name);
            (want_coopmat, vk_khr_cooperative_matrix_extension_name);
          ]
      in
      let ext_name_arrays =
        List.map (fun n -> CArray.of_string n) requested_extensions
      in
      let ext_name_ptrs =
        CArray.of_list (ptr char) (List.map CArray.start ext_name_arrays)
      in
      setf
        dev_create_info
        dev_create_enabledExtensionCount
        (Unsigned.UInt32.of_int (List.length requested_extensions)) ;
      setf
        dev_create_info
        dev_create_ppEnabledExtensionNames
        (if requested_extensions = [] then from_voidp string null
         else coerce (ptr (ptr char)) (ptr string) (CArray.start ext_name_ptrs)) ;

      (* The feature pNext chain. Only structs whose feature is both supported
         and wanted are chained, and each has exactly the requested field set —
         cooperativeMatrixRobustBufferAccess is deliberately NOT requested, as
         it carries a further requirement on core robustBufferAccess; the probe
         RECORDS whether the device offers it without asking for it.

         pEnabledFeatures below stays non-NULL and keeps carrying the core
         shaderFloat64 / shaderInt64 request. That is legal: the "pEnabledFeatures
         must be NULL" rule applies only when VkPhysicalDeviceFeatures2 itself
         is in the chain, and it is not. *)
      let coopmat_enable =
        make vk_physical_device_cooperative_matrix_features
      in
      zero_struct vk_physical_device_cooperative_matrix_features coopmat_enable ;
      setf
        coopmat_enable
        coopmat_feat_sType
        (u32 vk_structure_type_physical_device_cooperative_matrix_features_khr) ;
      setf coopmat_enable coopmat_feat_pNext null ;
      setf
        coopmat_enable
        coopmat_feat_cooperativeMatrix
        (Unsigned.UInt32.of_int 1) ;

      let storage16_enable = make vk_physical_device_16bit_storage_features in
      zero_struct vk_physical_device_16bit_storage_features storage16_enable ;
      setf
        storage16_enable
        storage16_sType
        (u32 vk_structure_type_physical_device_16bit_storage_features) ;
      setf
        storage16_enable
        storage16_storageBuffer16BitAccess
        (Unsigned.UInt32.of_int 1) ;

      let f16_enable = make vk_physical_device_shader_float16_int8_features in
      zero_struct vk_physical_device_shader_float16_int8_features f16_enable ;
      setf
        f16_enable
        f16i8_sType
        (u32 vk_structure_type_physical_device_shader_float16_int8_features) ;
      setf f16_enable f16i8_shaderFloat16 (Unsigned.UInt32.of_int 1) ;

      let chain_head =
        List.fold_left
          (fun next (wanted, set_pnext, self) ->
            if wanted then begin
              set_pnext next ;
              self
            end
            else next)
          null
          [
            ( want_coopmat,
              (fun p -> setf coopmat_enable coopmat_feat_pNext p),
              to_voidp (addr coopmat_enable) );
            ( want_storage16,
              (fun p -> setf storage16_enable storage16_pNext p),
              to_voidp (addr storage16_enable) );
            ( want_fp16,
              (fun p -> setf f16_enable f16i8_pNext p),
              to_voidp (addr f16_enable) );
          ]
      in
      setf dev_create_info dev_create_pNext chain_head ;
      (* Request every wide-type feature the physical device actually reports,
         and only those - requesting an unsupported feature fails
         vkCreateDevice outright. All other features are left at their default
         (false), matching the original pEnabledFeatures = null behaviour.

         #142: this used to request shaderFloat64 ALONE, which is how an int64
         kernel became a spec violation on hardware that fully supports int64.
         The emitter half (#141) declares GL_ARB_gpu_shader_int64 so the source
         validates, and glslang lowers it to OpCapability Int64 - but the
         capability is only legal when the DEVICE has enabled shaderInt64, and
         nothing here ever did. Measured on an RX 7900 XTX (RADV, Mesa
         26.1.4-arch3.1): VK_LAYER_KHRONOS_validation reports
         VUID-VkShaderModuleCreateInfo-pCode-08740 at vkCreateShaderModule,
         while the kernel still returns CORRECT results - so the defect is
         silent undefined behaviour on this driver, not a visible failure, and
         no test that only checks results could have caught it.

         Iterating over a request table rather than open-coding a second [if]
         is deliberate: the previous shape had one branch per feature, and the
         way it went wrong was a feature with no branch at all.

         NOTE: hoisting this binding out of the [if] does NOT by itself keep
         [addr enabled_features] valid across [vkCreateDevice] - in OCaml a
         value dies after its last USE, not at the end of its scope, so the
         GC may reclaim it the moment [setf] has copied the bare address in.
         The explicit keep-alives after the call are what make it safe. *)
      let requested_features =
        List.filter_map
          (fun (supported, index) -> if supported then Some index else None)
          [
            (supports_fp64, shader_float64_field_index);
            (supports_int64, shader_int64_field_index);
          ]
      in
      let enabled_features = make vk_physical_device_features in
      if requested_features <> [] then begin
        let enabled_bools = getf enabled_features phys_features_bools in
        for i = 0 to vk_physical_device_features_field_count - 1 do
          CArray.set enabled_bools i (Unsigned.UInt32.of_int 0)
        done ;
        List.iter
          (fun index ->
            CArray.set enabled_bools index (Unsigned.UInt32.of_int 1))
          requested_features ;
        setf
          dev_create_info
          dev_create_pEnabledFeatures
          (to_voidp (addr enabled_features))
      end
      else setf dev_create_info dev_create_pEnabledFeatures null ;

      let device = allocate vk_device_ptr (from_voidp vk_device null) in
      check
        "vkCreateDevice"
        (vkCreateDevice phys_dev (addr dev_create_info) null device) ;
      (* [dev_create_info] holds bare addresses into all of these, and in OCaml
         a value dies after its last USE rather than at end of scope. *)
      ignore (Sys.opaque_identity enabled_features) ;
      ignore (Sys.opaque_identity queue_create_info) ;
      ignore (Sys.opaque_identity queue_priority) ;
      ignore (Sys.opaque_identity ext_name_arrays) ;
      ignore (Sys.opaque_identity ext_name_ptrs) ;
      ignore (Sys.opaque_identity coopmat_enable) ;
      ignore (Sys.opaque_identity storage16_enable) ;
      ignore (Sys.opaque_identity f16_enable) ;

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
          supports_fp64;
          supports_int64;
          (* Reported as the ENABLED state, not the supported one. On a device
             where the extension is absent these are false even if some other
             path could have reached the feature — which is what makes the
             value safe to hand to a capability gate. *)
          supports_fp16 = want_fp16;
          storage_buffer_16bit = want_storage16;
          driver_id = props2.ep_driver_id;
          driver_name = props2.ep_driver_name;
          driver_info = props2.ep_driver_info;
          subgroup_size = props2.ep_subgroup_size;
          subgroup_size_probed = props2.ep_subgroup_size_probed;
          coopmat_extension_advertised =
            has_ext vk_khr_cooperative_matrix_extension_name;
          coopmat_enabled = want_coopmat;
          coopmat;
        }
      in
      Hashtbl.add device_cache idx dev ;
      dev

let set_current _dev = ()
(* Vulkan doesn't have a global "current device" concept *)

let synchronize dev = check "vkDeviceWaitIdle" (vkDeviceWaitIdle dev.device)

(* Notify the layers above this backend BEFORE anything is released (#90):
   [Vulkan_api_kernel]'s cache holds pipelines, layouts, descriptor pools and
   shader modules created from [dev.device], and [Sarek.Runtime]'s outer memo
   holds closures over those. Destroying the VkDevice first would leave both
   tables referencing dead objects, to be served to the next lookup for a
   recreated index.

   [notify_device_destroy] re-raises the first failing listener, and
   [device_cache] has already been emptied, so letting it escape here would
   leave the device unreachable with its VkDevice still alive. Capture it,
   finish the teardown, re-raise at the end — the discipline Cache_hooks.mli
   imposes on this path, and the shape Cuda_api/Hip_api use. *)
let destroy dev =
  Hashtbl.remove device_cache dev.id ;
  let listener_exn =
    match
      Spoc_framework.Cache_hooks.notify_device_destroy ~backend:"Vulkan" dev.id
    with
    | () -> None
    | exception e -> Some e
  in
  vkDestroyCommandPool dev.device dev.command_pool null ;
  vkDestroyDevice dev.device null ;
  Option.iter raise listener_exn
