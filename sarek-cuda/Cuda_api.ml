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

exception Cuda_error of cu_result * string

(** Check CUDA result and raise exception on error *)
let check (ctx : string) (result : cu_result) : unit =
  match result with CUDA_SUCCESS -> () | err -> raise (Cuda_error (err, ctx))

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

  let destroy dev = check "cuCtxDestroy" (cuCtxDestroy dev.context)
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
        raise (Cuda_error (err, "cuModuleLoadData"))) ;
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

  let compile_cached device ~name ~source =
    (* Cache key must include device ID - modules are device-specific *)
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
        let _ = cuModuleUnload k.module_ in
        ())
      cache ;
    Hashtbl.clear cache

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

let is_available () =
  (* First check if the library is available at all *)
  if not (Cuda_bindings.is_available ()) then false
  else if not (Cuda_nvrtc.is_available ()) then false
  else
    try
      Device.init () ;
      Device.count () > 0
    with _ -> false

let memory_info device =
  Device.set_current device ;
  let free = allocate size_t Unsigned.Size_t.zero in
  let total = allocate size_t Unsigned.Size_t.zero in
  check "cuMemGetInfo" (cuMemGetInfo free total) ;
  (Unsigned.Size_t.to_int64 !@free, Unsigned.Size_t.to_int64 !@total)
