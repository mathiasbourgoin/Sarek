(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * HIP API - High-Level Wrappers
 *
 * Safe, OCaml-friendly interface to HIP. HIP analog of Cuda_api. Uses HIP's
 * runtime device model (hipSetDevice) instead of the deprecated hipCtx* API;
 * module load/launch go through the driver-style hipModule* calls. Kernels are
 * JIT-compiled with hiprtc to a finalized code object and loaded via
 * hipModuleLoadData (one fewer stage than the CUDA NVRTC->ptxas->module path).
 ******************************************************************************)

open Ctypes
open Hip_types
open Hip_bindings

(** {1 Constants} *)

let max_device_name_length = 256

(** {1 Error checking} *)

let check (ctx : string) (result : hip_result) : unit =
  Hip_error.check
    ~is_success:(fun r -> r = HIP_SUCCESS)
    ~to_string:string_of_hip_result
    ctx
    result

(** Hooks invoked with a device id right before its device is torn down (kernel
    cache eviction), registered by [Kernel] to avoid a circular dependency. *)
let device_destroy_hooks : (int -> unit) list ref = ref []

(** hipModuleLaunchKernel is asynchronous and the HIP stack snapshots the kernel
    parameters at dispatch time, not at the call site (see the CUDA sibling's
    note - it explicitly calls out the HIP behaviour). Freeing the arg cells
    when [Kernel.launch] returns would be a use-after-free once the GC reclaims
    them. We retain them until the stream that ran the launch is drained, keyed
    by [(device_id, stream_key)]. Values are kept as [Obj.t] for liveness only
    and never inspected. A mutex guards the list against concurrent domains. *)
let pending_kernargs : (int * nativeint * Obj.t) list ref = ref []

let pending_lock = Mutex.create ()

let retain_kernargs device_id stream_key keepalive =
  Mutex.protect pending_lock (fun () ->
      pending_kernargs :=
        (device_id, stream_key, keepalive) :: !pending_kernargs)

let retire_stream device_id stream_key =
  Mutex.protect pending_lock (fun () ->
      pending_kernargs :=
        List.filter
          (fun (id, sk, _) -> id <> device_id || sk <> stream_key)
          !pending_kernargs)

let retire_device device_id =
  Mutex.protect pending_lock (fun () ->
      pending_kernargs :=
        List.filter (fun (id, _, _) -> id <> device_id) !pending_kernargs)

let stream_key_of_ptr (handle : _ ptr) = raw_address_of_ptr (to_voidp handle)

let default_stream_key = 0n

(** {1 Device Management} *)

module Device = struct
  type t = {
    id : int;
    handle : hip_device;
    name : string;
    total_mem : int64;
    compute_capability : int * int;
    max_threads_per_block : int;
    max_block_dims : int * int * int;
    max_grid_dims : int * int * int;
    shared_mem_per_block : int;
    warp_size : int;
    multiprocessor_count : int;
  }

  let initialized = ref false

  let device_cache : (int, t) Hashtbl.t = Hashtbl.create 4

  let init () =
    if not !initialized then begin
      check "hipInit" (hipInit (Unsigned.UInt.of_int 0)) ;
      initialized := true
    end

  let count () =
    init () ;
    let n = allocate int 0 in
    check "hipGetDeviceCount" (hipGetDeviceCount n) ;
    !@n

  let get_attribute dev attr =
    let v = allocate int 0 in
    check
      "hipDeviceGetAttribute"
      (hipDeviceGetAttribute v (int_of_device_attribute attr) dev) ;
    !@v

  (* Domain-local current-device tracker: hipSetDevice is per-OS-thread and
     OCaml 5 domains are threads, so each domain tracks its own current device
     (mirrors the CUDA backend). *)
  let current_id_key = Domain.DLS.new_key (fun () -> -1)

  let current_id () = Domain.DLS.get current_id_key

  let set_current_id id = Domain.DLS.set current_id_key id

  let create_device idx =
    init () ;
    let dev = allocate hip_device 0 in
    check "hipDeviceGet" (hipDeviceGet dev idx) ;
    let handle = !@dev in

    let name_buf = allocate_n char ~count:max_device_name_length in
    check
      "hipDeviceGetName"
      (hipDeviceGetName name_buf max_device_name_length handle) ;
    let name = string_from_ptr name_buf ~length:(max_device_name_length - 1) in
    let name =
      String.sub
        name
        0
        (try String.index name '\000'
         with Not_found -> max_device_name_length - 1)
    in

    let mem = allocate size_t Unsigned.Size_t.zero in
    check "hipDeviceTotalMem" (hipDeviceTotalMem mem handle) ;
    let total_mem = Unsigned.Size_t.to_int64 !@mem in

    let major = allocate int 0 and minor = allocate int 0 in
    check
      "hipDeviceComputeCapability"
      (hipDeviceComputeCapability major minor handle) ;
    let major = !@major and minor = !@minor in

    let max_threads =
      get_attribute handle HIP_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK
    in
    let max_block_x =
      get_attribute handle HIP_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X
    in
    let max_block_y =
      get_attribute handle HIP_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y
    in
    let max_block_z =
      get_attribute handle HIP_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z
    in
    let max_grid_x = get_attribute handle HIP_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X in
    let max_grid_y = get_attribute handle HIP_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y in
    let max_grid_z = get_attribute handle HIP_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z in
    let shared_mem =
      get_attribute handle HIP_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK
    in
    let warp = get_attribute handle HIP_DEVICE_ATTRIBUTE_WARP_SIZE in
    let mp_count =
      get_attribute handle HIP_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT
    in

    let dev =
      {
        id = idx;
        handle;
        name;
        total_mem;
        compute_capability = (major, minor);
        max_threads_per_block = max_threads;
        max_block_dims = (max_block_x, max_block_y, max_block_z);
        max_grid_dims = (max_grid_x, max_grid_y, max_grid_z);
        shared_mem_per_block = shared_mem;
        warp_size = warp;
        multiprocessor_count = mp_count;
      }
    in
    Spoc_core.Log.debugf
      Spoc_core.Log.Device
      "HIP device %d: %s (cc %d.%d, %Ld MB, warp %d)"
      idx
      name
      major
      minor
      (Int64.div total_mem (Int64.of_int (1024 * 1024)))
      warp ;
    dev

  let get idx =
    match Hashtbl.find_opt device_cache idx with
    | Some dev -> dev
    | None ->
        let dev = create_device idx in
        Hashtbl.add device_cache idx dev ;
        dev

  let set_current dev =
    set_current_id dev.id ;
    check "hipSetDevice" (hipSetDevice dev.id)

  let synchronize dev =
    set_current dev ;
    check "hipDeviceSynchronize" (hipDeviceSynchronize ()) ;
    retire_device dev.id

  let destroy dev =
    (* Evict cache + run kernel-cache eviction hooks while the device is still
       selected, mirroring the CUDA backend. HIP has no explicit context to
       destroy under the runtime model, so there is no hipCtxDestroy analog. *)
    Hashtbl.remove device_cache dev.id ;
    (* Notify layers above this backend BEFORE the local hooks unload modules:
       they memoize values that close over the handles those hooks release. *)
    Spoc_framework.Cache_hooks.notify_device_destroy dev.id ;
    List.iter (fun hook -> hook dev.id) !device_destroy_hooks ;
    retire_device dev.id
end

(** {1 Memory Management} *)

module Memory = struct
  type 'a buffer = {
    ptr : hip_deviceptr;
    size : int;
    elem_size : int;
    device : Device.t;
  }

  let alloc device size kind =
    Device.set_current device ;
    (* NOT Ctypes_static.sizeof (Ctypes.typ_of_bigarray_kind kind): ctypes has
       no Float16 kind and raises "Unsupported bigarray kind" there. Spoc_core's
       pure table knows f16 is 2 bytes. See Spoc_core.Memory.bigarray_void_ptr
       for the full explanation. *)
    let elem_size = Spoc_core.Memory.bigarray_elem_size kind in
    let bytes = Unsigned.Size_t.of_int (size * elem_size) in
    let pp = allocate (ptr void) null in
    check "hipMalloc" (hipMalloc pp bytes) ;
    {ptr = !@pp; size; elem_size; device}

  let alloc_custom device ~size ~elem_size =
    Device.set_current device ;
    let bytes = Unsigned.Size_t.of_int (size * elem_size) in
    let pp = allocate (ptr void) null in
    check "hipMalloc (custom)" (hipMalloc pp bytes) ;
    {ptr = !@pp; size; elem_size; device}

  let free buf =
    Device.set_current buf.device ;
    check "hipFree" (hipFree buf.ptr)

  (* The [Sys.opaque_identity] here is the same keep-alive the sibling
     [load_module_from_code_object] below applies to its code bigarray: the
     bigarray must stay reachable until hipMemcpy has consumed its address.
     [Spoc_core.Memory.bigarray_void_ptr] now returns a MANAGED pointer for
     every kind including f16, so the pointer value itself roots the array —
     but the explicit keepalive is kept because it is cheap and because the
     ctypes view here could be inlined away in a future refactor. *)
  let host_to_device ~src ~dst =
    Device.set_current dst.device ;
    let src_ptr = Spoc_core.Memory.bigarray_void_ptr src in
    let bytes = Unsigned.Size_t.of_int (Bigarray.Array1.size_in_bytes src) in
    check "hipMemcpyHtoD" (hipMemcpyHtoD dst.ptr src_ptr bytes) ;
    ignore (Sys.opaque_identity src) ;
    retire_stream dst.device.id default_stream_key

  let device_to_host ~src ~dst =
    Device.set_current src.device ;
    let dst_ptr = Spoc_core.Memory.bigarray_void_ptr dst in
    let bytes = Unsigned.Size_t.of_int (Bigarray.Array1.size_in_bytes dst) in
    check "hipMemcpyDtoH" (hipMemcpyDtoH dst_ptr src.ptr bytes) ;
    ignore (Sys.opaque_identity dst) ;
    retire_stream src.device.id default_stream_key

  let host_ptr_to_device ~src_ptr ~byte_size ~dst =
    Device.set_current dst.device ;
    let bytes = Unsigned.Size_t.of_int byte_size in
    check "hipMemcpyHtoD (ptr)" (hipMemcpyHtoD dst.ptr src_ptr bytes) ;
    retire_stream dst.device.id default_stream_key

  let device_to_host_ptr ~src ~dst_ptr ~byte_size =
    Device.set_current src.device ;
    let bytes = Unsigned.Size_t.of_int byte_size in
    check "hipMemcpyDtoH (ptr)" (hipMemcpyDtoH dst_ptr src.ptr bytes) ;
    retire_stream src.device.id default_stream_key

  let device_to_device ~src ~dst =
    Device.set_current src.device ;
    let bytes = Unsigned.Size_t.of_int (src.size * src.elem_size) in
    check "hipMemcpyDtoD" (hipMemcpyDtoD dst.ptr src.ptr bytes) ;
    retire_stream src.device.id default_stream_key

  let memset buf value =
    Device.set_current buf.device ;
    let bytes = Unsigned.Size_t.of_int (buf.size * buf.elem_size) in
    check
      "hipMemsetD8"
      (hipMemsetD8 buf.ptr (Unsigned.UChar.of_int value) bytes)
end

(** {1 Stream Management} *)

module Stream = struct
  type t = {handle : hip_stream structure ptr; device : Device.t}

  let create device =
    Device.set_current device ;
    let stream = allocate hip_stream_ptr (from_voidp hip_stream null) in
    check "hipStreamCreate" (hipStreamCreate stream) ;
    {handle = !@stream; device}

  let destroy stream =
    Device.set_current stream.device ;
    check "hipStreamDestroy" (hipStreamDestroy stream.handle)

  let synchronize stream =
    check "hipStreamSynchronize" (hipStreamSynchronize stream.handle) ;
    retire_stream stream.device.id (stream_key_of_ptr stream.handle)

  let default device = {handle = from_voidp hip_stream null; device}
end

(** {1 Event Management} *)

module Event = struct
  type t = {handle : hip_event structure ptr}

  let create () =
    let event = allocate hip_event_ptr (from_voidp hip_event null) in
    check "hipEventCreate" (hipEventCreate event) ;
    {handle = !@event}

  let destroy event = check "hipEventDestroy" (hipEventDestroy event.handle)

  let record event stream =
    check "hipEventRecord" (hipEventRecord event.handle stream.Stream.handle)

  let synchronize event =
    check "hipEventSynchronize" (hipEventSynchronize event.handle)

  let elapsed ~start ~stop =
    let ms = allocate float 0.0 in
    check
      "hipEventElapsedTime"
      (hipEventElapsedTime ms start.handle stop.handle) ;
    !@ms
end

(** {1 Kernel Management} *)

module Kernel = struct
  type t = {
    module_ : hip_module structure ptr;
    function_ : hip_function structure ptr;
    name : string;
  }

  type arg =
    | ArgBuffer : _ Memory.buffer -> arg
    | ArgInt32 : int32 -> arg
    | ArgInt64 : int64 -> arg
    | ArgFloat32 : float -> arg
    | ArgFloat64 : float -> arg
    | ArgPtr : nativeint -> arg

  (* Compilation cache. Guarded against concurrent multi-domain access by
     [Spoc_framework.Guarded_cache]: cache lookup/insert, per-device key
     tracking, eviction and clearing are all atomic, while the hiprtc compile
     runs outside the lock. Keys are grouped by device id (via
     [find_or_build ~device_id]) so a device destroy/recreate cycle can evict
     exactly its own stale module/function handles. *)
  let cache : (string, t) Spoc_framework.Guarded_cache.t =
    Spoc_framework.Guarded_cache.create
      ~destroy:(fun k -> ignore (hipModuleUnload k.module_))
      ()

  let evict_device device_id =
    Spoc_framework.Guarded_cache.evict_device cache device_id

  let () = device_destroy_hooks := evict_device :: !device_destroy_hooks

  (* Load a finalized HIP code object (binary ELF bytes) into a module and
     resolve [name]. The device must already be selected. The code object is
     self-describing (its size is in the ELF header), so hipModuleLoadData
     needs only the base pointer. *)
  let load_module_from_code_object ~name code =
    let module_ = allocate hip_module_ptr (from_voidp hip_module null) in
    let len = String.length code in
    let code_ba = Bigarray.Array1.create Bigarray.char Bigarray.c_layout len in
    for i = 0 to len - 1 do
      Bigarray.Array1.set code_ba i code.[i]
    done ;
    let code_ptr = bigarray_start array1 code_ba |> to_voidp in
    let load_result = hipModuleLoadData module_ code_ptr in
    ignore (Sys.opaque_identity code_ba) ;
    (match load_result with
    | HIP_SUCCESS ->
        Spoc_core.Log.debug
          Spoc_core.Log.Kernel
          "HIP code object load succeeded"
    | err ->
        Spoc_core.Log.errorf
          Spoc_core.Log.Kernel
          "hipModuleLoadData failed: %s"
          (string_of_hip_result err) ;
        Hip_error.raise_error
          (Hip_error.module_load_failed
             len
             (Printf.sprintf "hipModuleLoadData: %s" (string_of_hip_result err)))) ;
    let func = allocate hip_function_ptr (from_voidp hip_function null) in
    check "hipModuleGetFunction" (hipModuleGetFunction func !@module_ name) ;
    {module_ = !@module_; function_ = !@func; name}

  let compile device ~name ~source =
    Device.set_current device ;
    Spoc_core.Log.debugf
      Spoc_core.Log.Kernel
      "HIP compile: kernel='%s' device=%d (current-device arch)"
      name
      device.Device.id ;
    (* No explicit --offload-arch: hiprtc targets the just-selected device,
       which is correct for every gfx target incl. the integrated gfx1036. *)
    let code = Hip_rtc.compile_to_code_object ~name source in
    Spoc_core.Log.debugf
      Spoc_core.Log.Kernel
      "HIP code object generated (%d bytes)"
      (String.length code) ;
    load_module_from_code_object ~name code

  (* Like [compile] but threads extra hiprtc options (e.g. include dirs such as
     "-I/opt/rocm/include" for rocWMMA headers, or preprocessor defines).
     Targets the current device (no explicit --offload-arch). *)
  let compile_with_options device ~name ~source ~options =
    Device.set_current device ;
    let code = Hip_rtc.compile_to_code_object ~name ~options source in
    Spoc_core.Log.debugf
      Spoc_core.Log.Kernel
      "HIP code object generated (%d bytes, %d extra opts)"
      (String.length code)
      (List.length options) ;
    load_module_from_code_object ~name code

  let with_cache device ~name ~source build =
    let key =
      Spoc_framework.Compile_cache.make_key
        ~device:(string_of_int device.Device.id)
        ~name
        ~source
        ()
    in
    Spoc_framework.Guarded_cache.find_or_build
      cache
      ~key
      ~device_id:device.Device.id
      build

  let compile_cached device ~name ~source =
    with_cache device ~name ~source (fun () -> compile device ~name ~source)

  let clear_cache () = Spoc_framework.Guarded_cache.clear cache

  type ctype_ref = CTypeRef : 'a typ * 'a ptr -> ctype_ref

  let launch kernel ~args ~grid ~block ~shared_mem ~stream =
    let params = CArray.make (ptr void) (List.length args) in
    let refs : ctype_ref list ref = ref [] in

    List.iteri
      (fun i arg ->
        let ptr =
          match arg with
          | ArgBuffer buf ->
              let v = allocate (ptr void) buf.Memory.ptr in
              refs := CTypeRef (Ctypes.ptr void, v) :: !refs ;
              to_voidp v
          | ArgInt32 n ->
              let v = allocate int32_t n in
              refs := CTypeRef (int32_t, v) :: !refs ;
              to_voidp v
          | ArgInt64 n ->
              let v = allocate int64_t n in
              refs := CTypeRef (int64_t, v) :: !refs ;
              to_voidp v
          | ArgFloat32 f ->
              let v = allocate float f in
              refs := CTypeRef (float, v) :: !refs ;
              to_voidp v
          | ArgFloat64 f ->
              let v = allocate double f in
              refs := CTypeRef (double, v) :: !refs ;
              to_voidp v
          | ArgPtr p ->
              let v = allocate nativeint p in
              refs := CTypeRef (nativeint, v) :: !refs ;
              to_voidp v
        in
        CArray.set params i ptr)
      args ;

    let stream_ptr =
      match stream with
      | Some s -> s.Stream.handle
      | None -> from_voidp hip_stream null
    in

    let gx, gy, gz = grid in
    let bx, by, bz = block in

    check
      "hipModuleLaunchKernel"
      (hipModuleLaunchKernel
         kernel.function_
         (Unsigned.UInt.of_int gx)
         (Unsigned.UInt.of_int gy)
         (Unsigned.UInt.of_int gz)
         (Unsigned.UInt.of_int bx)
         (Unsigned.UInt.of_int by)
         (Unsigned.UInt.of_int bz)
         (Unsigned.UInt.of_int shared_mem)
         stream_ptr
         (CArray.start params)
         (from_voidp (ptr void) null)) ;
    let device_id, stream_key =
      match stream with
      | Some s -> (s.Stream.device.id, stream_key_of_ptr s.Stream.handle)
      | None -> (Device.current_id (), default_stream_key)
    in
    retain_kernargs device_id stream_key (Obj.repr (params, !refs))
end

(** {1 Utility Functions} *)

let runtime_version () =
  let v = allocate int 0 in
  check "hipRuntimeGetVersion" (hipRuntimeGetVersion v) ;
  !@v

(** Available if libamdhip64 loads, libhiprtc loads, and at least one device is
    present. hiprtc is required because this backend is JIT-only (HIP C++ ->
    code object at runtime). *)
let is_available () =
  if not (Hip_bindings.is_available () && Hip_rtc.is_available ()) then false
  else
    try
      Device.init () ;
      Device.count () > 0
    with _ -> false

let memory_info device =
  Device.set_current device ;
  let free = allocate size_t Unsigned.Size_t.zero in
  let total = allocate size_t Unsigned.Size_t.zero in
  check "hipMemGetInfo" (hipMemGetInfo free total) ;
  (Unsigned.Size_t.to_int64 !@free, Unsigned.Size_t.to_int64 !@total)
