(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * OpenCL API - High-Level Wrappers
 *
 * Provides a safe, OCaml-friendly interface to OpenCL functionality.
 * Handles error checking, resource management, and type conversions.
 ******************************************************************************)

open Ctypes
open Opencl_types
open Opencl_bindings

(** {1 Exceptions} *)

exception Opencl_error of cl_error * string

(** Check OpenCL result and raise exception on error *)
let check (ctx : string) (result : int32) : unit =
  let err = cl_error_of_int32 result in
  match err with CL_SUCCESS -> () | _ -> raise (Opencl_error (err, ctx))

(** {1 Platform Management} *)

module Platform = struct
  type t = cl_platform_id

  let get_all () =
    let num = allocate cl_uint Unsigned.UInt32.zero in
    check
      "clGetPlatformIDs(count)"
      (clGetPlatformIDs
         Unsigned.UInt32.zero
         (from_voidp cl_platform_id null)
         num) ;
    let count = Unsigned.UInt32.to_int !@num in
    if count = 0 then [||]
    else begin
      let platforms = CArray.make cl_platform_id count in
      check
        "clGetPlatformIDs"
        (clGetPlatformIDs
           (Unsigned.UInt32.of_int count)
           (CArray.start platforms)
           num) ;
      Array.init count (CArray.get platforms)
    end

  let get_info platform info_type =
    let size = allocate size_t Unsigned.Size_t.zero in
    check
      "clGetPlatformInfo(size)"
      (clGetPlatformInfo
         platform
         (Unsigned.UInt32.of_int (int_of_platform_info info_type))
         Unsigned.Size_t.zero
         (to_voidp null)
         size) ;
    let buf_size = Unsigned.Size_t.to_int !@size in
    let buf = allocate_n char ~count:buf_size in
    check
      "clGetPlatformInfo"
      (clGetPlatformInfo
         platform
         (Unsigned.UInt32.of_int (int_of_platform_info info_type))
         !@size
         (to_voidp buf)
         size) ;
    string_from_ptr buf ~length:(buf_size - 1)

  let name p = get_info p CL_PLATFORM_NAME

  let vendor p = get_info p CL_PLATFORM_VENDOR

  let version p = get_info p CL_PLATFORM_VERSION
end

(** {1 Device Management} *)

module Device = struct
  type t = {
    id : int;
    handle : cl_device_id;
    platform : Platform.t;
    name : string;
    vendor : string;
    max_compute_units : int;
    max_work_group_size : int;
    max_work_item_dims : int;
    max_work_item_sizes : int array;
    global_mem_size : int64;
    local_mem_size : int64;
    max_clock_freq : int;
    supports_fp64 : bool;
    is_cpu : bool; (* True for CPU OpenCL devices - enables zero-copy *)
  }

  let get_devices platform device_type =
    let num = allocate cl_uint Unsigned.UInt32.zero in
    let result =
      clGetDeviceIDs
        platform
        device_type
        Unsigned.UInt32.zero
        (from_voidp cl_device_id null)
        num
    in
    if result = -1l then [||] (* CL_DEVICE_NOT_FOUND *)
    else begin
      check "clGetDeviceIDs(count)" result ;
      let count = Unsigned.UInt32.to_int !@num in
      if count = 0 then [||]
      else begin
        let devices = CArray.make cl_device_id count in
        check
          "clGetDeviceIDs"
          (clGetDeviceIDs
             platform
             device_type
             (Unsigned.UInt32.of_int count)
             (CArray.start devices)
             num) ;
        Array.init count (CArray.get devices)
      end
    end

  let get_info_string device info_type =
    let size = allocate size_t Unsigned.Size_t.zero in
    check
      "clGetDeviceInfo(size)"
      (clGetDeviceInfo
         device
         (Unsigned.UInt32.of_int (int_of_device_info info_type))
         Unsigned.Size_t.zero
         (to_voidp null)
         size) ;
    let buf_size = Unsigned.Size_t.to_int !@size in
    let buf = allocate_n char ~count:buf_size in
    check
      "clGetDeviceInfo"
      (clGetDeviceInfo
         device
         (Unsigned.UInt32.of_int (int_of_device_info info_type))
         !@size
         (to_voidp buf)
         size) ;
    string_from_ptr buf ~length:(buf_size - 1)

  let get_info_int device info_type =
    let value = allocate cl_uint Unsigned.UInt32.zero in
    check
      "clGetDeviceInfo"
      (clGetDeviceInfo
         device
         (Unsigned.UInt32.of_int (int_of_device_info info_type))
         (Unsigned.Size_t.of_int (sizeof cl_uint))
         (to_voidp value)
         (from_voidp size_t null)) ;
    Unsigned.UInt32.to_int !@value

  let get_info_long device info_type =
    let value = allocate cl_ulong Unsigned.UInt64.zero in
    check
      "clGetDeviceInfo"
      (clGetDeviceInfo
         device
         (Unsigned.UInt32.of_int (int_of_device_info info_type))
         (Unsigned.Size_t.of_int (sizeof cl_ulong))
         (to_voidp value)
         (from_voidp size_t null)) ;
    Unsigned.UInt64.to_int64 !@value

  let get_info_size device info_type =
    let value = allocate size_t Unsigned.Size_t.zero in
    check
      "clGetDeviceInfo"
      (clGetDeviceInfo
         device
         (Unsigned.UInt32.of_int (int_of_device_info info_type))
         (Unsigned.Size_t.of_int (sizeof size_t))
         (to_voidp value)
         (from_voidp size_t null)) ;
    Unsigned.Size_t.to_int !@value

  let make_device platform idx handle =
    let name = get_info_string handle CL_DEVICE_NAME in
    let vendor = get_info_string handle CL_DEVICE_VENDOR in
    let max_compute_units = get_info_int handle CL_DEVICE_MAX_COMPUTE_UNITS in
    let max_work_group_size =
      get_info_size handle CL_DEVICE_MAX_WORK_GROUP_SIZE
    in
    let max_work_item_dims =
      get_info_int handle CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS
    in
    let global_mem_size = get_info_long handle CL_DEVICE_GLOBAL_MEM_SIZE in
    let local_mem_size = get_info_long handle CL_DEVICE_LOCAL_MEM_SIZE in
    let max_clock_freq = get_info_int handle CL_DEVICE_MAX_CLOCK_FREQUENCY in

    (* Get max work item sizes *)
    let max_work_item_sizes = Array.make max_work_item_dims 0 in
    let sizes = CArray.make size_t max_work_item_dims in
    let _ =
      clGetDeviceInfo
        handle
        (Unsigned.UInt32.of_int
           (int_of_device_info CL_DEVICE_MAX_WORK_ITEM_SIZES))
        (Unsigned.Size_t.of_int (max_work_item_dims * sizeof size_t))
        (to_voidp (CArray.start sizes))
        (from_voidp size_t null)
    in
    for i = 0 to max_work_item_dims - 1 do
      max_work_item_sizes.(i) <- Unsigned.Size_t.to_int (CArray.get sizes i)
    done ;

    (* Check FP64 support *)
    let extensions = get_info_string handle CL_DEVICE_EXTENSIONS in
    let contains sub str =
      try
        let _ = Str.search_forward (Str.regexp_string sub) str 0 in
        true
      with Not_found -> false
    in
    let supports_fp64 =
      contains "cl_khr_fp64" extensions || contains "cl_amd_fp64" extensions
    in

    (* Check if device is CPU type *)
    let device_type = get_info_long handle CL_DEVICE_TYPE in
    let is_cpu = Int64.logand device_type 2L <> 0L in
    (* CL_DEVICE_TYPE_CPU = 2 *)

    {
      id = idx;
      handle;
      platform;
      name;
      vendor;
      max_compute_units;
      max_work_group_size;
      max_work_item_dims;
      max_work_item_sizes;
      global_mem_size;
      local_mem_size;
      max_clock_freq;
      supports_fp64;
      is_cpu;
    }

  let init () = () (* OpenCL doesn't require explicit init *)

  let count () =
    let platforms = Platform.get_all () in
    Array.fold_left
      (fun acc p -> acc + Array.length (get_devices p cl_device_type_all))
      0
      platforms

  let get idx =
    let platforms = Platform.get_all () in
    let current = ref 0 in
    let result = ref None in
    Array.iter
      (fun p ->
        let devices = get_devices p cl_device_type_all in
        Array.iteri
          (fun _i d ->
            if !current = idx then result := Some (make_device p idx d) ;
            incr current)
          devices)
      platforms ;
    match !result with
    | Some d -> d
    | None ->
        let max_devices = count () in
        Opencl_error.raise_error (Opencl_error.device_not_found idx max_devices)
end

(** {1 Context Management} *)

module Context = struct
  type t = {handle : cl_context; device : Device.t}

  let create device =
    let err = allocate cl_int 0l in
    let devices = CArray.make cl_device_id 1 in
    CArray.set devices 0 device.Device.handle ;
    let ctx =
      clCreateContext
        (from_voidp cl_ulong null)
        Unsigned.UInt32.one
        (CArray.start devices)
        (from_voidp void null)
        (from_voidp void null)
        err
    in
    check "clCreateContext" !@err ;
    {handle = ctx; device}

  let release ctx = check "clReleaseContext" (clReleaseContext ctx.handle)
end

(** {1 Command Queue Management} *)

module CommandQueue = struct
  type t = {handle : cl_command_queue; context : Context.t}

  let create context ?(profiling = false) () =
    let err = allocate cl_int 0l in
    let queue =
      (* Try OpenCL 2.0+ API first *)
      let props =
        if profiling then cl_queue_profiling_enable else Unsigned.UInt64.zero
      in
      let props_arr = CArray.make cl_ulong 3 in
      CArray.set props_arr 0 (Unsigned.UInt64.of_int 0x1093) ;
      (* CL_QUEUE_PROPERTIES *)
      CArray.set props_arr 1 props ;
      CArray.set props_arr 2 Unsigned.UInt64.zero ;
      (* Terminator *)
      match
        clCreateCommandQueueWithProperties
          context.Context.handle
          context.device.Device.handle
          (CArray.start props_arr)
          err
      with
      | Some queue -> queue
      | None ->
          (* Fall back to OpenCL 1.x API (for macOS and older implementations) *)
          let props_bits =
            if profiling then Unsigned.UInt64.of_int 0x0002
            else Unsigned.UInt64.zero
          in
          (* CL_QUEUE_PROFILING_ENABLE = 0x0002 *)
          clCreateCommandQueue
            context.Context.handle
            context.device.Device.handle
            props_bits
            err
    in
    check "clCreateCommandQueue" !@err ;
    {handle = queue; context}

  let release queue =
    check "clReleaseCommandQueue" (clReleaseCommandQueue queue.handle)

  let finish queue = check "clFinish" (clFinish queue.handle)

  let flush queue = check "clFlush" (clFlush queue.handle)
end

(** {1 Memory Management} *)

module Memory = struct
  type 'a buffer = {
    handle : cl_mem;
    size : int;
    elem_size : int;
    context : Context.t;
    zero_copy : bool; (* True if using CL_MEM_USE_HOST_PTR - skip transfers *)
  }

  let alloc context size kind =
    let elem_size = Ctypes_static.sizeof (Ctypes.typ_of_bigarray_kind kind) in
    let bytes = Unsigned.Size_t.of_int (size * elem_size) in
    let err = allocate cl_int 0l in
    let mem =
      clCreateBuffer
        context.Context.handle
        cl_mem_read_write
        bytes
        (from_voidp void null)
        err
    in
    check "clCreateBuffer" !@err ;
    {handle = mem; size; elem_size; context; zero_copy = false}

  (** Allocate buffer with zero-copy using host pointer. For CPU OpenCL devices,
      this avoids memory copies entirely. *)
  let alloc_with_host_ptr context size kind host_ptr =
    let elem_size = Ctypes_static.sizeof (Ctypes.typ_of_bigarray_kind kind) in
    let bytes = Unsigned.Size_t.of_int (size * elem_size) in
    let err = allocate cl_int 0l in
    (* CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR *)
    let flags = Unsigned.UInt64.logor cl_mem_read_write cl_mem_use_host_ptr in
    let mem = clCreateBuffer context.Context.handle flags bytes host_ptr err in
    check "clCreateBuffer (use_host_ptr)" !@err ;
    {handle = mem; size; elem_size; context; zero_copy = true}

  (** Allocate buffer for custom types with explicit element size in bytes *)
  let alloc_custom context ~size ~elem_size =
    let bytes = Unsigned.Size_t.of_int (size * elem_size) in
    let err = allocate cl_int 0l in
    let mem =
      clCreateBuffer
        context.Context.handle
        cl_mem_read_write
        bytes
        (from_voidp void null)
        err
    in
    check "clCreateBuffer (custom)" !@err ;
    {handle = mem; size; elem_size; context; zero_copy = false}

  (** Check if buffer uses zero-copy (no transfers needed) *)
  let is_zero_copy buf = buf.zero_copy

  let free buf = check "clReleaseMemObject" (clReleaseMemObject buf.handle)

  let host_to_device queue ~src ~dst =
    (* Skip transfer for zero-copy buffers - data is already shared *)
    if dst.zero_copy then ()
    else begin
      let bytes = Unsigned.Size_t.of_int (Bigarray.Array1.size_in_bytes src) in
      let src_ptr = bigarray_start array1 src |> to_voidp in
      check
        "clEnqueueWriteBuffer"
        (clEnqueueWriteBuffer
           queue.CommandQueue.handle
           dst.handle
           cl_true
           Unsigned.Size_t.zero
           bytes
           src_ptr
           Unsigned.UInt32.zero
           (from_voidp cl_event null)
           (from_voidp cl_event null))
    end

  let device_to_host queue ~src ~dst =
    (* Skip transfer for zero-copy buffers - data is already shared *)
    if src.zero_copy then ()
    else begin
      let bytes = Unsigned.Size_t.of_int (Bigarray.Array1.size_in_bytes dst) in
      let dst_ptr = bigarray_start array1 dst |> to_voidp in
      check
        "clEnqueueReadBuffer"
        (clEnqueueReadBuffer
           queue.CommandQueue.handle
           src.handle
           cl_true
           Unsigned.Size_t.zero
           bytes
           dst_ptr
           Unsigned.UInt32.zero
           (from_voidp cl_event null)
           (from_voidp cl_event null))
    end

  (** Transfer from raw pointer to device buffer (for custom types) *)
  let host_ptr_to_device queue ~src_ptr ~byte_size ~dst =
    (* Skip transfer for zero-copy buffers *)
    if dst.zero_copy then ()
    else begin
      let bytes = Unsigned.Size_t.of_int byte_size in
      check
        "clEnqueueWriteBuffer (ptr)"
        (clEnqueueWriteBuffer
           queue.CommandQueue.handle
           dst.handle
           cl_true
           Unsigned.Size_t.zero
           bytes
           src_ptr
           Unsigned.UInt32.zero
           (from_voidp cl_event null)
           (from_voidp cl_event null))
    end

  (** Transfer from device buffer to raw pointer (for custom types) *)
  let device_to_host_ptr queue ~src ~dst_ptr ~byte_size =
    (* Skip transfer for zero-copy buffers *)
    if src.zero_copy then ()
    else begin
      let bytes = Unsigned.Size_t.of_int byte_size in
      check
        "clEnqueueReadBuffer (ptr)"
        (clEnqueueReadBuffer
           queue.CommandQueue.handle
           src.handle
           cl_true
           Unsigned.Size_t.zero
           bytes
           dst_ptr
           Unsigned.UInt32.zero
           (from_voidp cl_event null)
           (from_voidp cl_event null))
    end
end

(** {1 Program Management} *)

module Program = struct
  type t = {
    handle : cl_program;
    context : Context.t;
    _source :
      (char, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t;
  }

  let create_from_source context source =
    let err = allocate cl_int 0l in
    (* Use a Bigarray to hold a pinned copy of the source string.
       The OCaml GC cannot move Bigarray data, so the pointer passed to
       clCreateProgramWithSource remains valid through clBuildProgram.
       This fixes a GC-related corruption where ctypes 'string' type
       pointers become stale on OCaml 5. *)
    let len = String.length source in
    let ba = Bigarray.Array1.create Bigarray.char Bigarray.c_layout (len + 1) in
    for i = 0 to len - 1 do
      Bigarray.Array1.set ba i source.[i]
    done ;
    Bigarray.Array1.set ba len '\000' ;
    let c_ptr = bigarray_start array1 ba in
    let src_ptrs = allocate (ptr char) c_ptr in
    let lengths = allocate size_t (Unsigned.Size_t.of_int len) in
    let prog =
      clCreateProgramWithSource
        context.Context.handle
        Unsigned.UInt32.one
        (coerce (ptr (ptr char)) (ptr string) src_ptrs)
        lengths
        err
    in
    check "clCreateProgramWithSource" !@err ;
    (* Store bigarray in record to prevent GC until after build *)
    {handle = prog; context; _source = ba}

  let build program ?(options = "") () =
    let devices = CArray.make cl_device_id 1 in
    CArray.set devices 0 program.context.device.Device.handle ;
    let result =
      clBuildProgram
        program.handle
        Unsigned.UInt32.one
        (CArray.start devices)
        options
        (from_voidp void null)
        (from_voidp void null)
    in
    if result <> 0l then begin
      (* Get build log *)
      let size = allocate size_t Unsigned.Size_t.zero in
      let _ =
        clGetProgramBuildInfo
          program.handle
          program.context.device.handle
          (Unsigned.UInt32.of_int
             (int_of_program_build_info CL_PROGRAM_BUILD_LOG))
          Unsigned.Size_t.zero
          (to_voidp null)
          size
      in
      let log_size = Unsigned.Size_t.to_int !@size in
      let log_buf = allocate_n char ~count:log_size in
      let _ =
        clGetProgramBuildInfo
          program.handle
          program.context.device.handle
          (Unsigned.UInt32.of_int
             (int_of_program_build_info CL_PROGRAM_BUILD_LOG))
          !@size
          (to_voidp log_buf)
          size
      in
      let log = string_from_ptr log_buf ~length:(log_size - 1) in
      Opencl_error.raise_error
        (Opencl_error.compilation_failed "OpenCL kernel" log)
    end

  let release program =
    check "clReleaseProgram" (clReleaseProgram program.handle)
end

(** {1 Kernel Management} *)

module Kernel = struct
  type t = {handle : cl_kernel; program : Program.t; name : string}

  let create program name =
    let err = allocate cl_int 0l in
    let kernel = clCreateKernel program.Program.handle name err in
    check "clCreateKernel" !@err ;
    {handle = kernel; program; name}

  let release kernel = check "clReleaseKernel" (clReleaseKernel kernel.handle)

  let set_arg_buffer kernel idx buf =
    let mem_ptr = allocate cl_mem buf.Memory.handle in
    check
      "clSetKernelArg"
      (clSetKernelArg
         kernel.handle
         (Unsigned.UInt32.of_int idx)
         (Unsigned.Size_t.of_int (sizeof cl_mem))
         (to_voidp mem_ptr))

  let set_arg_int32 kernel idx value =
    let v = allocate int32_t value in
    check
      "clSetKernelArg"
      (clSetKernelArg
         kernel.handle
         (Unsigned.UInt32.of_int idx)
         (Unsigned.Size_t.of_int (sizeof int32_t))
         (to_voidp v))

  let set_arg_int64 kernel idx value =
    let v = allocate int64_t value in
    check
      "clSetKernelArg"
      (clSetKernelArg
         kernel.handle
         (Unsigned.UInt32.of_int idx)
         (Unsigned.Size_t.of_int (sizeof int64_t))
         (to_voidp v))

  let set_arg_float32 kernel idx value =
    let v = allocate float value in
    check
      "clSetKernelArg"
      (clSetKernelArg
         kernel.handle
         (Unsigned.UInt32.of_int idx)
         (Unsigned.Size_t.of_int (sizeof float))
         (to_voidp v))

  let set_arg_float64 kernel idx value =
    let v = allocate double value in
    check
      "clSetKernelArg"
      (clSetKernelArg
         kernel.handle
         (Unsigned.UInt32.of_int idx)
         (Unsigned.Size_t.of_int (sizeof double))
         (to_voidp v))

  let set_arg_local kernel idx bytes =
    check
      "clSetKernelArg"
      (clSetKernelArg
         kernel.handle
         (Unsigned.UInt32.of_int idx)
         (Unsigned.Size_t.of_int bytes)
         (from_voidp void null))

  let launch queue kernel ~global ~local =
    let work_dim = 3 in
    let global_arr = CArray.make size_t work_dim in
    let local_arr = CArray.make size_t work_dim in
    let gx, gy, gz = global in
    let lx, ly, lz = local in
    CArray.set global_arr 0 (Unsigned.Size_t.of_int gx) ;
    CArray.set global_arr 1 (Unsigned.Size_t.of_int gy) ;
    CArray.set global_arr 2 (Unsigned.Size_t.of_int gz) ;
    CArray.set local_arr 0 (Unsigned.Size_t.of_int lx) ;
    CArray.set local_arr 1 (Unsigned.Size_t.of_int ly) ;
    CArray.set local_arr 2 (Unsigned.Size_t.of_int lz) ;
    check
      "clEnqueueNDRangeKernel"
      (clEnqueueNDRangeKernel
         queue.CommandQueue.handle
         kernel.handle
         (Unsigned.UInt32.of_int work_dim)
         (from_voidp size_t null)
         (CArray.start global_arr)
         (CArray.start local_arr)
         Unsigned.UInt32.zero
         (from_voidp cl_event null)
         (from_voidp cl_event null))
end

(** {1 Utility Functions} *)

let is_available () =
  (* First check if the library is available at all *)
  if not (Opencl_bindings.is_available ()) then false
  else
    try
      let platforms = Platform.get_all () in
      Array.length platforms > 0
    with _ -> false

(* String helper *)
module String = struct
  include String

  let is_substring ~sub s =
    let len_sub = String.length sub in
    let len_s = String.length s in
    if len_sub > len_s then false
    else
      let rec check i =
        if i > len_s - len_sub then false
        else if String.sub s i len_sub = sub then true
        else check (i + 1)
      in
      check 0
end
