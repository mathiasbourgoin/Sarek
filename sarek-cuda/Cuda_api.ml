(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * CUDA API - High-Level Wrappers
 *
 * Provides a safe, OCaml-friendly interface to CUDA functionality.
 * Handles error checking, resource management, and type conversions.
 ******************************************************************************)

open Ctypes
open Cuda_types
open Cuda_bindings

(** {1 Constants} *)

(** Maximum device name length in characters *)
let max_device_name_length = 256

(** Maximum PTX header preview length for error messages *)
let max_ptx_header_preview = 200

(** {1 Exceptions} *)

(** Deprecated: no longer raised. [check] below now raises the canonical
    {!Cuda_error.Cuda_error} (a [Backend_error] alias) via the shared
    {!Sarek_backend_error.Backend_error.Make.check} funnel, so every handler in
    the codebase can catch one exception shape across backends. This declaration
    is kept only so that out-of-tree code matching on
    [Cuda_api.Cuda_error (code, ctx)] still compiles (this library is
    opam-published). *)
exception
  Cuda_error of cu_result * string
      [@ocaml.deprecated
        "no longer raised; Cuda_api.check now raises Cuda_error.Cuda_error \
         (Backend_error) - catch that instead"]

(** Check CUDA result and raise a canonical {!Backend_error} on failure. *)
let check (ctx : string) (result : cu_result) : unit =
  Cuda_error.check
    ~is_success:(fun r -> r = CUDA_SUCCESS)
    ~to_string:string_of_cu_result
    ctx
    result

(** Hooks invoked with a device id right before its context is destroyed.
    [Kernel] registers an eviction hook here (after [Kernel.cache] is defined
    below) so that [Device.destroy] can retire per-device compiled kernels
    without a circular module dependency. *)
let device_destroy_hooks : (int -> unit) list ref = ref []

(** {1 Device Management} *)

module Device = struct
  type t = {
    id : int;
    handle : cu_device;
    context : cu_context structure ptr;
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

  (* Device cache - reuse the same device/context to keep kernel handles valid *)
  let device_cache : (int, t) Hashtbl.t = Hashtbl.create 4

  let init () =
    if not !initialized then begin
      check "cuInit" (cuInit 0) ;
      initialized := true
    end

  let count () =
    init () ;
    let n = allocate int 0 in
    check "cuDeviceGetCount" (cuDeviceGetCount n) ;
    !@n

  let get_attribute dev attr =
    let v = allocate int 0 in
    check
      "cuDeviceGetAttribute"
      (cuDeviceGetAttribute v (int_of_device_attribute attr) dev) ;
    !@v

  (* Create a new device with context - internal, use get for cached version *)
  let create_device idx =
    init () ;
    let dev = allocate cu_device 0 in
    check "cuDeviceGet" (cuDeviceGet dev idx) ;
    let handle = !@dev in

    (* Get name *)
    let name_buf = allocate_n char ~count:max_device_name_length in
    check
      "cuDeviceGetName"
      (cuDeviceGetName name_buf max_device_name_length handle) ;
    let name = string_from_ptr name_buf ~length:(max_device_name_length - 1) in
    let name =
      String.sub
        name
        0
        (try String.index name '\000'
         with Not_found -> max_device_name_length - 1)
    in

    (* Get total memory *)
    let mem = allocate size_t Unsigned.Size_t.zero in
    check "cuDeviceTotalMem" (cuDeviceTotalMem mem handle) ;
    let total_mem = Unsigned.Size_t.to_int64 !@mem in

    (* Get attributes *)
    let major =
      get_attribute handle CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR
    in
    let minor =
      get_attribute handle CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR
    in
    let max_threads =
      get_attribute handle CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK
    in
    let max_block_x =
      get_attribute handle CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X
    in
    let max_block_y =
      get_attribute handle CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y
    in
    let max_block_z =
      get_attribute handle CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z
    in
    let max_grid_x = get_attribute handle CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X in
    let max_grid_y = get_attribute handle CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y in
    let max_grid_z = get_attribute handle CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z in
    let shared_mem =
      get_attribute handle CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK
    in
    let warp = get_attribute handle CU_DEVICE_ATTRIBUTE_WARP_SIZE in
    let mp_count =
      get_attribute handle CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT
    in

    (* Create context *)
    let ctx = allocate cu_context_ptr (from_voidp cu_context null) in
    check
      "cuCtxCreate"
      (cuCtxCreate ctx (Unsigned.UInt.of_int cu_ctx_sched_auto) handle) ;

    let dev =
      {
        id = idx;
        handle;
        context = !@ctx;
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
      "CUDA device %d: %s (cc %d.%d, %Ld MB)"
      idx
      name
      major
      minor
      (Int64.div total_mem (Int64.of_int (1024 * 1024))) ;
    dev

  (* Get a device, reusing cached context to keep kernel handles valid *)
  let get idx =
    match Hashtbl.find_opt device_cache idx with
    | Some dev -> dev
    | None ->
        let dev = create_device idx in
        Hashtbl.add device_cache idx dev ;
        dev

  let set_current dev = check "cuCtxSetCurrent" (cuCtxSetCurrent dev.context)

  let synchronize dev =
    set_current dev ;
    check "cuCtxSynchronize" (cuCtxSynchronize ())

  let destroy dev =
    (* Evict from device_cache first: leaving a stale entry means a later
       [get idx] returns a handle whose context has already been destroyed
       (mirrors the Vulkan fix in Vulkan_api_device.ml). Also run the
       registered destroy hooks (Kernel.cache eviction) while the context
       is still current, so stale module/function handles for this device
       can't be returned by [Kernel.compile_cached] after the context is
       recreated. *)
    Hashtbl.remove device_cache dev.id ;
    List.iter (fun hook -> hook dev.id) !device_destroy_hooks ;
    check "cuCtxDestroy" (cuCtxDestroy dev.context)
end

(** {1 Memory Management} *)

module Memory = struct
  type 'a buffer = {
    ptr : cu_deviceptr;
    size : int;
    elem_size : int;
    device : Device.t;
  }

  let alloc device size kind =
    Device.set_current device ;
    let elem_size = Ctypes_static.sizeof (Ctypes.typ_of_bigarray_kind kind) in
    let bytes = Unsigned.Size_t.of_int (size * elem_size) in
    let ptr = allocate cu_deviceptr Unsigned.UInt64.zero in
    check "cuMemAlloc" (cuMemAlloc ptr bytes) ;
    {ptr = !@ptr; size; elem_size; device}

  (** Allocate buffer for custom types with explicit element size in bytes *)
  let alloc_custom device ~size ~elem_size =
    Device.set_current device ;
    let bytes = Unsigned.Size_t.of_int (size * elem_size) in
    let ptr = allocate cu_deviceptr Unsigned.UInt64.zero in
    check "cuMemAlloc (custom)" (cuMemAlloc ptr bytes) ;
    {ptr = !@ptr; size; elem_size; device}

  let free buf =
    Device.set_current buf.device ;
    check "cuMemFree" (cuMemFree buf.ptr)

  let host_to_device ~src ~dst =
    Device.set_current dst.device ;
    let src_ptr = bigarray_start array1 src |> to_voidp in
    let bytes = Unsigned.Size_t.of_int (Bigarray.Array1.size_in_bytes src) in
    check "cuMemcpyHtoD" (cuMemcpyHtoD dst.ptr src_ptr bytes)

  let device_to_host ~src ~dst =
    Device.set_current src.device ;
    let dst_ptr = bigarray_start array1 dst |> to_voidp in
    let bytes = Unsigned.Size_t.of_int (Bigarray.Array1.size_in_bytes dst) in
    check "cuMemcpyDtoH" (cuMemcpyDtoH dst_ptr src.ptr bytes)

  (** Transfer from raw pointer to device buffer (for custom types) *)
  let host_ptr_to_device ~src_ptr ~byte_size ~dst =
    Device.set_current dst.device ;
    let bytes = Unsigned.Size_t.of_int byte_size in
    check "cuMemcpyHtoD (ptr)" (cuMemcpyHtoD dst.ptr src_ptr bytes)

  (** Transfer from device buffer to raw pointer (for custom types) *)
  let device_to_host_ptr ~src ~dst_ptr ~byte_size =
    Device.set_current src.device ;
    let bytes = Unsigned.Size_t.of_int byte_size in
    check "cuMemcpyDtoH (ptr)" (cuMemcpyDtoH dst_ptr src.ptr bytes)

  let device_to_device ~src ~dst =
    Device.set_current src.device ;
    let bytes = Unsigned.Size_t.of_int (src.size * src.elem_size) in
    check "cuMemcpyDtoD" (cuMemcpyDtoD dst.ptr src.ptr bytes)

  let memset buf value =
    Device.set_current buf.device ;
    let bytes = Unsigned.Size_t.of_int (buf.size * buf.elem_size) in
    check "cuMemsetD8" (cuMemsetD8 buf.ptr (Unsigned.UChar.of_int value) bytes)
end

(** {1 Stream Management} *)

module Stream = struct
  type t = {handle : cu_stream structure ptr; device : Device.t}

  let create device =
    Device.set_current device ;
    let stream = allocate cu_stream_ptr (from_voidp cu_stream null) in
    check
      "cuStreamCreate"
      (cuStreamCreate stream (Unsigned.UInt.of_int cu_stream_default)) ;
    {handle = !@stream; device}

  let destroy stream =
    Device.set_current stream.device ;
    check "cuStreamDestroy" (cuStreamDestroy stream.handle)

  let synchronize stream =
    check "cuStreamSynchronize" (cuStreamSynchronize stream.handle)

  let default device = {handle = from_voidp cu_stream null; device}
end

(** {1 Event Management} *)

module Event = struct
  type t = {handle : cu_event structure ptr}

  let create () =
    let event = allocate cu_event_ptr (from_voidp cu_event null) in
    check
      "cuEventCreate"
      (cuEventCreate event (Unsigned.UInt.of_int cu_event_default)) ;
    {handle = !@event}

  let destroy event = check "cuEventDestroy" (cuEventDestroy event.handle)

  let record event stream =
    check "cuEventRecord" (cuEventRecord event.handle stream.Stream.handle)

  let synchronize event =
    check "cuEventSynchronize" (cuEventSynchronize event.handle)

  let elapsed ~start ~stop =
    let ms = allocate float 0.0 in
    check "cuEventElapsedTime" (cuEventElapsedTime ms start.handle stop.handle) ;
    !@ms
end

(** {1 Kernel Management} *)

module Kernel = struct
  type t = {
    module_ : cu_module structure ptr;
    function_ : cu_function structure ptr;
    name : string;
  }

  type arg =
    | ArgBuffer : _ Memory.buffer -> arg
    | ArgInt32 : int32 -> arg
    | ArgInt64 : int64 -> arg
    | ArgFloat32 : float -> arg
    | ArgFloat64 : float -> arg
    | ArgPtr : nativeint -> arg

  (* Compilation cache *)
  let cache : (string, t) Hashtbl.t = Hashtbl.create 16

  (* Cache keys grouped by device id, so a device destroy/recreate cycle can
     evict exactly its own stale module/function handles from [cache]
     without needing to reverse the (digested, opaque) cache key. *)
  let keys_by_device : (int, string list ref) Hashtbl.t = Hashtbl.create 16

  let record_key_for_device device_id key =
    match Hashtbl.find_opt keys_by_device device_id with
    | Some keys -> keys := key :: !keys
    | None -> Hashtbl.add keys_by_device device_id (ref [key])

  (* Evict every cached kernel compiled for [device_id]. Registered as a
     [device_destroy_hooks] callback below so [Device.destroy] retires
     these handles before the underlying CUDA context is destroyed. *)
  let evict_device device_id =
    match Hashtbl.find_opt keys_by_device device_id with
    | None -> ()
    | Some keys ->
        List.iter
          (fun key ->
            match Hashtbl.find_opt cache key with
            | None -> ()
            | Some k ->
                let _ = cuModuleUnload k.module_ in
                Hashtbl.remove cache key)
          !keys ;
        Hashtbl.remove keys_by_device device_id

  let () = device_destroy_hooks := evict_device :: !device_destroy_hooks

  (* Replace the .target directive in a PTX string to match the given SM version.
     This makes a static PTX string portable: PTX written for sm_86 loads fine
     on sm_61 as long as it doesn't use sm_86-specific instructions. *)
  let with_sm_target ~major ~minor ptx =
    let prefix = ".target " in
    let plen = String.length prefix in
    String.split_on_char '\n' ptx
    |> List.map (fun line ->
        if String.length line >= plen && String.sub line 0 plen = prefix then
          Printf.sprintf "%ssm_%d%d" prefix major minor
        else line)
    |> String.concat "\n"

  (* Load a PTX string into a CUDA module and retrieve [name] as a function.
     The device context must already be current when this is called. *)
  let load_module_from_ptx ~name ptx =
    let module_ = allocate cu_module_ptr (from_voidp cu_module null) in
    let ptx_len = String.length ptx in
    let ptx_ba =
      Bigarray.Array1.create Bigarray.char Bigarray.c_layout (ptx_len + 1)
    in
    for i = 0 to ptx_len - 1 do
      Bigarray.Array1.set ptx_ba i ptx.[i]
    done ;
    Bigarray.Array1.set ptx_ba ptx_len '\000' ;
    let ptx_ptr = bigarray_start array1 ptx_ba |> to_voidp in
    let opt_arr =
      CArray.of_list int [int_of_jit_option CU_JIT_TARGET_FROM_CUCONTEXT]
    in
    let opt_vals = CArray.of_list (ptr void) [from_voidp void null] in
    let load_result =
      cuModuleLoadDataEx
        module_
        ptx_ptr
        (Unsigned.UInt.of_int (CArray.length opt_arr))
        (CArray.start opt_arr)
        (CArray.start opt_vals)
    in
    let load_result =
      match load_result with
      | CUDA_SUCCESS -> load_result
      | _ -> cuModuleLoadData module_ ptx_ptr
    in
    ignore (Sys.opaque_identity ptx_ba) ;
    (match load_result with
    | CUDA_SUCCESS ->
        Spoc_core.Log.debug Spoc_core.Log.Kernel "PTX module load succeeded"
    | err ->
        let ptx_header =
          String.sub ptx 0 (min max_ptx_header_preview (String.length ptx))
        in
        Spoc_core.Log.errorf
          Spoc_core.Log.Kernel
          "cuModuleLoadData failed: %s\nPTX header: %s"
          (string_of_cu_result err)
          ptx_header ;
        Cuda_error.raise_error
          (Cuda_error.module_load_failed
             (String.length ptx)
             (Printf.sprintf "cuModuleLoadData: %s" (string_of_cu_result err)))) ;
    let func = allocate cu_function_ptr (from_voidp cu_function null) in
    check "cuModuleGetFunction" (cuModuleGetFunction func !@module_ name) ;
    {module_ = !@module_; function_ = !@func; name}

  let compile device ~name ~source =
    Device.set_current device ;

    (* Compile to PTX - clamp architecture to what NVRTC likely supports.
       CUDA 13.x NVRTC supports up to compute_90. Newer devices will use
       the highest supported arch and rely on driver JIT. *)
    let major, minor = device.Device.compute_capability in
    let cc_num = (major * 10) + minor in
    let arch =
      if cc_num >= 90 then "compute_90"
        (* Clamp to compute_90 for Hopper and newer *)
      else Printf.sprintf "compute_%d%d" major minor
    in
    Spoc_core.Log.debugf
      Spoc_core.Log.Kernel
      "CUDA compile: kernel='%s' arch=%s (cc %d.%d) device=%d"
      name
      arch
      major
      minor
      device.Device.id ;
    let ptx = Cuda_nvrtc.compile_to_ptx ~name ~arch source in
    Spoc_core.Log.debugf
      Spoc_core.Log.Kernel
      "CUDA PTX generated (%d bytes)"
      (String.length ptx) ;
    load_module_from_ptx ~name ptx

  (** Load a pre-assembled PTX string directly, bypassing NVRTC. The .target
      directive in the PTX is automatically rewritten to match the device's
      actual SM, so a PTX built for sm_86 loads cleanly on sm_61 as long as it
      uses no sm_86-specific instructions. *)
  let load_from_ptx device ~name ~ptx =
    Device.set_current device ;
    let major, minor = device.Device.compute_capability in
    let ptx = with_sm_target ~major ~minor ptx in
    Spoc_core.Log.debugf
      Spoc_core.Log.Kernel
      "PTX load_from_ptx: kernel='%s' sm_%d%d (%d bytes)"
      name
      major
      minor
      (String.length ptx) ;
    load_module_from_ptx ~name ptx

  (** Load a pre-assembled PTX string using the already-current CUDA context.
      The caller must have already set the device context via
      Device.set_current. *)
  let load_from_ptx_current ~name ~ptx =
    Spoc_core.Log.debugf
      Spoc_core.Log.Kernel
      "PTX load_from_ptx_current: kernel='%s' (%d bytes)"
      name
      (String.length ptx) ;
    load_module_from_ptx ~name ptx

  (* Shared memoization for compiled/loaded kernels. The cache key must
     include device ID and the kernel name - a source file may define more
     than one kernel, and a resolved kernel handle for one name must never be
     returned for another (see Compile_cache.mli). [record_key_for_device]
     keeps the device-destroy eviction hook working for every entry. *)
  let with_cache device ~name ~source build =
    let key =
      Spoc_framework.Compile_cache.make_key
        ~device:(string_of_int device.Device.id)
        ~name
        ~source
        ()
    in
    match Hashtbl.find_opt cache key with
    | Some k -> k
    | None ->
        let k = build () in
        Hashtbl.add cache key k ;
        record_key_for_device device.Device.id key ;
        k

  (** Cached variant of [load_from_ptx] — same cache as [compile_cached].
      Without it, every launch reloads (and re-JITs) the PTX module, which
      dominates kernel time on drivers that compile at module-load (NVIDIA JIT,
      ZLUDA). *)
  let load_from_ptx_cached device ~name ~ptx =
    with_cache device ~name ~source:ptx (fun () ->
        load_from_ptx device ~name ~ptx)

  let compile_cached device ~name ~source =
    with_cache device ~name ~source (fun () -> compile device ~name ~source)

  let clear_cache () =
    Hashtbl.iter
      (fun _ k ->
        let _ = cuModuleUnload k.module_ in
        ())
      cache ;
    Hashtbl.clear cache ;
    Hashtbl.clear keys_by_device

  (** Existential wrapper for keeping Ctypes-allocated values alive during FFI
      calls *)
  type ctype_ref = CTypeRef : 'a typ * 'a ptr -> ctype_ref

  let launch kernel ~args ~grid ~block ~shared_mem ~stream =
    (* Build parameter array *)
    let params = CArray.make (ptr void) (List.length args) in
    let refs : ctype_ref list ref = ref [] in
    (* Keep references alive *)

    List.iteri
      (fun i arg ->
        let ptr =
          match arg with
          | ArgBuffer buf ->
              let v = allocate cu_deviceptr buf.Memory.ptr in
              refs := CTypeRef (cu_deviceptr, v) :: !refs ;
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
      | None -> from_voidp cu_stream null
    in

    let gx, gy, gz = grid in
    let bx, by, bz = block in

    check
      "cuLaunchKernel"
      (cuLaunchKernel
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
         (from_voidp (ptr void) null))
end

(** {1 Utility Functions} *)

let driver_version () =
  let v = allocate int 0 in
  check "cuDriverGetVersion" (cuDriverGetVersion v) ;
  let ver = !@v in
  (ver / 1000, ver mod 1000 / 10)

(** Driver-only availability: libcuda is loadable and reports at least one
    device. Sufficient for the PTX backend, which never calls NVRTC — this is
    what makes SPOC work on non-NVIDIA CUDA implementations such as ZLUDA, which
    ship the driver API without libnvrtc. *)
let is_driver_available () =
  if not (Cuda_bindings.is_available ()) then false
  else
    try
      Device.init () ;
      Device.count () > 0
    with _ -> false

(* NVRTC is probed before the driver so that a host without libnvrtc (e.g.
   ZLUDA) returns false without cuInit-ing the driver as a side effect. *)
let is_available () =
  Cuda_bindings.is_available ()
  && Cuda_nvrtc.is_available () && is_driver_available ()

let memory_info device =
  Device.set_current device ;
  let free = allocate size_t Unsigned.Size_t.zero in
  let total = allocate size_t Unsigned.Size_t.zero in
  check "cuMemGetInfo" (cuMemGetInfo free total) ;
  (Unsigned.Size_t.to_int64 !@free, Unsigned.Size_t.to_int64 !@total)
