(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 mathias <mathias@ladon.local> *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Metal API - High-Level Wrappers
 *
 * Provides a safe, OCaml-friendly interface to Metal functionality.
 * Handles error checking, resource management, and type conversions.
 ******************************************************************************)

open Ctypes
open Foreign
open Metal_types
open Metal_bindings

(** memcpy from libc for memory transfers *)
let memcpy ~dst ~src ~size =
  let memcpy_c =
    foreign "memcpy" (ptr void @-> ptr void @-> size_t @-> returning (ptr void))
  in
  let _ = memcpy_c dst src (Unsigned.Size_t.of_int size) in
  ()

(** {1 Exceptions} *)

exception Metal_error of string

(** Check Metal result and raise exception on error *)
let check (ctx : string) (result : mtl_error) : unit =
  match result with
  | MTL_SUCCESS -> ()
  | NS_ERROR msg -> raise (Metal_error (ctx ^ ": " ^ msg))

(** {1 Device Management} *)

module Device = struct
  type t = {
    id : int;
    handle : mtl_device;
    name : string;
    max_threads_per_threadgroup : mtl_size structure;
    max_threadgroup_memory : int;
    supports_fp64 : bool; (* Metal always supports FP64 on macOS *)
    is_cpu : bool; (* Metal doesn't distinguish CPU/GPU - always false *)
  }

  let init () =
    (* Metal doesn't require explicit initialization *)
    ()

  let count () =
    let devices_array = mtl_copy_all_devices () in
    let n =
      if is_null devices_array then 0
      else
        (* Get count from NSArray *)
        let sel = sel_registerName "count" in
        let fn =
          foreign
            ~from:(get_objc_lib ())
            "objc_msgSend"
            (ptr void @-> ptr void @-> returning uint64_t)
        in
        Unsigned.UInt64.to_int (fn devices_array sel)
    in
    if n > 0 then n
    else
      (* Fallback: Check for system default device if enumeration failed *)
      let default_dev = mtl_create_system_default_device () in
      if is_null default_dev then 0 else 1

  let get_all_handles () =
    let devices_array = mtl_copy_all_devices () in
    if is_null devices_array then [||]
    else
      let sel_count = sel_registerName "count" in
      let fn_count =
        foreign
          ~from:(get_objc_lib ())
          "objc_msgSend"
          (ptr void @-> ptr void @-> returning uint64_t)
      in
      let count = Unsigned.UInt64.to_int (fn_count devices_array sel_count) in

      let sel_obj = sel_registerName "objectAtIndex:" in
      let fn_obj =
        foreign
          ~from:(get_objc_lib ())
          "objc_msgSend"
          (ptr void @-> ptr void @-> uint64_t @-> returning mtl_device)
      in

      Array.init count (fun i ->
          fn_obj devices_array sel_obj (Unsigned.UInt64.of_int i))

  let make_device idx handle =
    let name = mtl_device_name handle in

    let max_threads =
      try mtl_device_max_threads_per_threadgroup handle
      with _ -> make_mtl_size ~width:1024 ~height:1024 ~depth:64
    in

    let max_threadgroup_memory =
      try mtl_device_max_threadgroup_memory_length handle with _ -> 32768
    in

    {
      id = idx;
      handle;
      name;
      max_threads_per_threadgroup = max_threads;
      max_threadgroup_memory;
      supports_fp64 = false;
      (* Metal shading language does NOT support double precision *)
      is_cpu = false;
      (* Metal devices are GPU-like *)
    }

  let get idx =
    if idx = 0 then
      (* Get default device *)
      let dev = mtl_create_system_default_device () in
      if is_null dev then
        (* Fallback: Try to get from list *)
        let devices = get_all_handles () in
        if Array.length devices > 0 then make_device 0 devices.(0)
        else
          Metal_error.raise_error
            (Metal_error.backend_unavailable "No Metal device found")
      else make_device 0 dev
    else
      (* Get from device list *)
      let devices = get_all_handles () in
      if idx >= Array.length devices then
        Metal_error.raise_error
          (Metal_error.device_not_found idx (Array.length devices))
      else make_device idx devices.(idx)

  let id dev = dev.id

  let name dev = dev.name

  let handle dev = dev.handle
end

(** {1 Command Queue Management} *)

module CommandQueue = struct
  type t = {handle : mtl_command_queue; device : Device.t}

  let create device =
    let queue = mtl_device_new_command_queue device.Device.handle in
    if is_null queue then raise (Metal_error "Failed to create command queue")
    else {handle = queue; device}

  let release queue = release queue.handle

  let device queue = queue.device
end

(** {1 Memory Management} *)

module Memory = struct
  type 'a buffer = {
    handle : mtl_buffer;
    size : int;
    elem_size : int;
    device : Device.t;
    contents : unit ptr; (* Pointer to GPU memory *)
  }

  let alloc device size elem_size =
    let byte_size = size * elem_size in
    (* Use shared storage mode for automatic CPU-GPU sync *)
    let buf =
      mtl_device_new_buffer_with_length
        device.Device.handle
        byte_size
        mtl_resource_storage_mode_shared
    in
    if is_null buf then raise (Metal_error "Failed to allocate buffer")
    else
      let contents = mtl_buffer_contents buf in
      {handle = buf; size; elem_size; device; contents}

  let alloc_bigarray device ba elem_size =
    let size = Bigarray.Array1.dim ba in
    let buf = alloc device size elem_size in
    (* Copy data to GPU *)
    let ba_ptr = bigarray_start array1 ba in
    let byte_size = size * elem_size in
    memcpy ~dst:buf.contents ~src:(to_voidp ba_ptr) ~size:byte_size ;
    buf

  let to_bigarray (type a b) buf (kind : (a, b) Bigarray.kind) =
    let ba = Bigarray.Array1.create kind Bigarray.c_layout buf.size in
    let ba_ptr = bigarray_start array1 ba in
    let byte_size = buf.size * buf.elem_size in
    memcpy ~dst:(to_voidp ba_ptr) ~src:buf.contents ~size:byte_size ;
    ba

  let release buf = release buf.handle

  let size buf = buf.size

  let handle buf = buf.handle

  let contents buf = buf.contents
end

(** {1 Library and Function Management} *)

module Library = struct
  type t = {handle : mtl_library; device : Device.t}

  let create_from_source device source =
    match
      mtl_device_new_library_with_source device.Device.handle source None
    with
    | Ok lib -> {handle = lib; device}
    | Error msg -> raise (Metal_error ("Library compilation failed: " ^ msg))

  let release lib = release lib.handle

  let get_function lib name =
    let func = mtl_library_new_function_with_name lib.handle name in
    if is_null func then
      raise (Metal_error ("Function '" ^ name ^ "' not found in library"))
    else func
end

(** {1 Compute Pipeline Management} *)

module ComputePipeline = struct
  type t = {
    handle : mtl_compute_pipeline_state;
    device : Device.t;
    max_threads_per_threadgroup : int;
    thread_execution_width : int;
  }

  let create device func =
    match mtl_device_new_compute_pipeline_state device.Device.handle func with
    | Ok pso ->
        let max_threads =
          mtl_compute_pipeline_state_max_total_threads_per_threadgroup pso
        in
        let thread_width =
          mtl_compute_pipeline_state_threadgroup_memory_length pso
        in
        {
          handle = pso;
          device;
          max_threads_per_threadgroup = max_threads;
          thread_execution_width = thread_width;
        }
    | Error msg ->
        raise (Metal_error ("Pipeline state creation failed: " ^ msg))

  let release pipeline = release pipeline.handle

  let handle pipeline = pipeline.handle
end

(** {1 Kernel Execution} *)

module Kernel = struct
  type arg =
    | Buffer of unit ptr * int (* buffer, offset *)
    | Int32 of int32
    | Int64 of int64
    | Float32 of float
    | Float64 of float

  type args = arg list

  type t = {pipeline : ComputePipeline.t; function_name : string}

  let create pipeline function_name = {pipeline; function_name}

  let execute queue kernel ~grid_size ~block_size args =
    (* Create command buffer *)
    let cmd_buffer =
      mtl_command_queue_command_buffer queue.CommandQueue.handle
    in
    if is_null cmd_buffer then
      raise (Metal_error "Failed to create command buffer") ;

    (* Create compute command encoder *)
    let encoder = mtl_command_buffer_compute_command_encoder cmd_buffer in
    if is_null encoder then
      raise (Metal_error "Failed to create compute encoder") ;

    (* Set pipeline state *)
    mtl_compute_command_encoder_set_compute_pipeline_state
      encoder
      kernel.pipeline.handle ;

    (* Set arguments *)
    List.iteri
      (fun idx arg ->
        match arg with
        | Buffer (ptr, offset) ->
            (* For Metal buffers, we pass the buffer directly *)
            mtl_compute_command_encoder_set_buffer encoder ptr offset idx
        | Int32 v ->
            let ptr = allocate int32_t v in
            mtl_compute_command_encoder_set_bytes
              encoder
              (to_voidp ptr)
              (sizeof int32_t)
              idx
        | Int64 v ->
            let ptr = allocate int64_t v in
            mtl_compute_command_encoder_set_bytes
              encoder
              (to_voidp ptr)
              (sizeof int64_t)
              idx
        | Float32 v ->
            let ptr = allocate float v in
            mtl_compute_command_encoder_set_bytes
              encoder
              (to_voidp ptr)
              (sizeof float)
              idx
        | Float64 v ->
            let ptr = allocate double v in
            mtl_compute_command_encoder_set_bytes
              encoder
              (to_voidp ptr)
              (sizeof double)
              idx)
      args ;

    (* Dispatch threads *)
    let grid_x, grid_y, grid_z = grid_size in
    let block_x, block_y, block_z = block_size in
    let threads = make_mtl_size ~width:grid_x ~height:grid_y ~depth:grid_z in
    let threadgroup =
      make_mtl_size ~width:block_x ~height:block_y ~depth:block_z
    in
    mtl_compute_command_encoder_dispatch_threads encoder threads threadgroup ;

    (* End encoding *)
    mtl_compute_command_encoder_end_encoding encoder ;

    (* Commit and wait *)
    mtl_command_buffer_commit cmd_buffer ;
    mtl_command_buffer_wait_until_completed cmd_buffer
end

(** {1 Synchronization} *)

let synchronize _device =
  (* Metal synchronization is handled per command buffer *)
  ()

(** {1 Profiling} *)

module Profiling = struct
  type event = unit (* Metal uses different profiling mechanism *)

  let enabled = ref false

  let enable () = enabled := true

  let disable () = enabled := false

  let event_elapsed_time _event = 0.0 (* TODO: Implement with GPU timestamps *)
end
