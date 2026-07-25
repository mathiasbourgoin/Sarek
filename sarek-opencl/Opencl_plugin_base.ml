(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * OpenCL Plugin - Framework Implementation
 *
 * Implements the Framework_sig.S interface for OpenCL devices.
 * This plugin is auto-registered when loaded.
 ******************************************************************************)

[@@@warning "-69"]

open Spoc_framework

module Opencl : Framework_sig.PLUGIN_BASE = struct
  let name = "OpenCL"

  let version = (3, 0, 0)

  (* Per-device state: context and default queue *)
  type device_state = {
    device : Opencl_api.Device.t;
    context : Opencl_api.Context.t;
    queue : Opencl_api.CommandQueue.t;
  }

  let device_states : (int, device_state) Hashtbl.t = Hashtbl.create 8

  (** Current device for kernel compilation/execution *)
  let current_device : Opencl_api.Device.t option ref = ref None

  let get_state device_id =
    match Hashtbl.find_opt device_states device_id with
    | Some s -> s
    | None ->
        let device = Opencl_api.Device.get device_id in
        let context = Opencl_api.Context.create device in
        let queue = Opencl_api.CommandQueue.create context () in
        let state = {device; context; queue} in
        Hashtbl.add device_states device_id state ;
        state

  module Device = struct
    type t = Opencl_api.Device.t

    type id = int

    let init = Opencl_api.Device.init

    let count = Opencl_api.Device.count

    let get = Opencl_api.Device.get

    let id (d : t) = d.Opencl_api.Device.id

    let name (d : t) = d.Opencl_api.Device.name

    let capabilities (d : t) : Framework_sig.capabilities =
      let open Opencl_api.Device in
      let max_dims = d.max_work_item_dims in
      let max_sizes = d.max_work_item_sizes in
      {
        Framework_sig.max_threads_per_block = d.max_work_group_size;
        max_block_dims =
          ( (if max_dims >= 1 then max_sizes.(0) else 1),
            (if max_dims >= 2 then max_sizes.(1) else 1),
            if max_dims >= 3 then max_sizes.(2) else 1 );
        max_grid_dims = (max_int, max_int, max_int);
        (* OpenCL doesn't limit grid *)
        shared_mem_per_block = Int64.to_int d.local_mem_size;
        total_global_mem = d.global_mem_size;
        compute_capability = (0, 0);
        (* OpenCL doesn't have this concept *)
        supports_fp64 = d.supports_fp64;
        supports_atomics = true;
        (* Most OpenCL devices support atomics *)
        warp_size = 32;
        (* Typical, could query CL_DEVICE_PREFERRED_WORK_GROUP_SIZE_MULTIPLE *)
        max_registers_per_block = 0;
        (* Not exposed in OpenCL *)
        clock_rate_khz = d.max_clock_freq * 1000;
        multiprocessor_count = d.max_compute_units;
        is_cpu = d.is_cpu;
      }

    let set_current (d : t) =
      let _ = get_state d.id in
      current_device := Some d

    let get_current_device () = !current_device

    let synchronize (d : t) =
      let state = get_state d.id in
      Opencl_api.CommandQueue.finish state.queue
  end

  module Memory = struct
    type 'a buffer = {buf : 'a Opencl_api.Memory.buffer; device_id : int}

    let alloc device size kind =
      let state = get_state device.Opencl_api.Device.id in
      let buf = Opencl_api.Memory.alloc state.context size kind in
      {buf; device_id = device.id}

    let alloc_zero_copy device ba kind =
      if not device.Opencl_api.Device.is_cpu then None
      else begin
        let state = get_state device.Opencl_api.Device.id in
        let size = Bigarray.Array1.dim ba in
        (* Not Ctypes.bigarray_start: no Float16 arm (#57 slice 1 review MF2).
           CL_MEM_USE_HOST_PTR keeps this address for the buffer's whole
           lifetime, so the OWNING Vector.t (which holds [ba]) is what roots it
           — the managed pointer only covers this call. *)
        let host_ptr = Spoc_core.Memory.bigarray_void_ptr ba in
        let buf =
          Opencl_api.Memory.alloc_with_host_ptr state.context size kind host_ptr
        in
        Some {buf; device_id = device.id}
      end

    let is_zero_copy b = Opencl_api.Memory.is_zero_copy b.buf

    let alloc_custom device ~size ~elem_size =
      let state = get_state device.Opencl_api.Device.id in
      let buf = Opencl_api.Memory.alloc_custom state.context ~size ~elem_size in
      {buf; device_id = device.id}

    let free b = Opencl_api.Memory.free b.buf

    let host_to_device ~src ~dst =
      let state = get_state dst.device_id in
      Opencl_api.Memory.host_to_device state.queue ~src ~dst:dst.buf

    let device_to_host ~src ~dst =
      let state = get_state src.device_id in
      Opencl_api.Memory.device_to_host state.queue ~src:src.buf ~dst

    let host_ptr_to_device ~src_ptr ~byte_size ~dst =
      let state = get_state dst.device_id in
      Opencl_api.Memory.host_ptr_to_device
        state.queue
        ~src_ptr:(Ctypes.ptr_of_raw_address src_ptr)
        ~byte_size
        ~dst:dst.buf

    let device_to_host_ptr ~src ~dst_ptr ~byte_size =
      let state = get_state src.device_id in
      Opencl_api.Memory.device_to_host_ptr
        state.queue
        ~src:src.buf
        ~dst_ptr:(Ctypes.ptr_of_raw_address dst_ptr)
        ~byte_size

    let device_to_device ~src ~dst =
      (* OpenCL doesn't have direct D2D copy across contexts *)
      (* Would need to implement via host staging *)
      ignore (src, dst) ;
      Opencl_error.raise_error
        (Opencl_error.feature_not_supported "device-to-device copy")

    let size b = b.buf.Opencl_api.Memory.size

    let device_ptr _b =
      (* OpenCL doesn't expose raw device pointers *)
      Nativeint.zero
  end

  module Stream = struct
    type t = {queue : Opencl_api.CommandQueue.t; device_id : int}

    let create device =
      let state = get_state device.Opencl_api.Device.id in
      let queue = Opencl_api.CommandQueue.create state.context () in
      {queue; device_id = device.id}

    let destroy stream = Opencl_api.CommandQueue.release stream.queue

    let synchronize stream = Opencl_api.CommandQueue.finish stream.queue

    let default device =
      let state = get_state device.Opencl_api.Device.id in
      {queue = state.queue; device_id = device.id}
  end

  module Event = struct
    type t = {mutable start_time : float; mutable end_time : float}

    let create () = {start_time = 0.0; end_time = 0.0}

    let destroy _event = ()

    let record event _stream = event.end_time <- Unix.gettimeofday ()

    let synchronize _event = ()

    let elapsed ~start ~stop = (stop.end_time -. start.start_time) *. 1000.0
  end

  module Kernel = struct
    type compiled = {
      kernel : Opencl_api.Kernel.t;
      program : Opencl_api.Program.t;
      device_id : int;
    }

    type t = compiled

    type arg =
      | ArgBuffer of {buf : Opencl_types.cl_mem; idx : int}
      | ArgInt32 of {value : int32; idx : int}
      | ArgInt64 of {value : int64; idx : int}
      | ArgFloat32 of {value : float; idx : int}
      | ArgFloat64 of {value : float; idx : int}

    (* Indexed by idx (last-set-wins) instead of accumulated by call order:
       see Spoc_framework.Kernel_args. *)
    type args = arg Spoc_framework.Kernel_args.t

    (* Cache: key -> compiled kernel. Guarded against concurrent multi-domain
       access by [Spoc_framework.Guarded_cache]: lookup/insert and clearing are
       atomic critical sections, while clBuildProgram runs outside the lock. *)
    let cache : (string, t) Spoc_framework.Guarded_cache.t =
      Spoc_framework.Guarded_cache.create
        ~destroy:(fun k ->
          Opencl_api.Kernel.release k.kernel ;
          Opencl_api.Program.release k.program)
        ()

    let compile device ~name ~source =
      let state = get_state device.Opencl_api.Device.id in
      let program =
        Opencl_api.Program.create_from_source state.context source
      in
      Opencl_api.Program.build program () ;
      let kernel = Opencl_api.Kernel.create program name in
      {kernel; program; device_id = device.id}

    let compile_cached device ~name ~source =
      (* Compile_cache.make_key gives the same collision-resistant,
         digest-per-field encoding every other backend uses (the July 2026
         cache-key standardization missed this hand-rolled join - audit
         finding; a ':' in a kernel name could shift bytes between fields). *)
      let key =
        Spoc_framework.Compile_cache.make_key
          ~device:(string_of_int device.Opencl_api.Device.id)
          ~name
          ~source
          ()
      in
      Spoc_framework.Guarded_cache.find_or_build cache ~key (fun () ->
          compile device ~name ~source)

    let clear_cache () = Spoc_framework.Guarded_cache.clear cache

    let load_from_ptx ~name:_ ~ptx:_ =
      Opencl_error.raise_error
        (Opencl_error.feature_not_supported "PTX kernels")

    let create_args () = Spoc_framework.Kernel_args.create ()

    let set_arg_buffer args idx buf =
      Spoc_core.Log.debugf
        Spoc_core.Log.Kernel
        "OpenCL set_arg_buffer idx=%d (before=%d)"
        idx
        (Spoc_framework.Kernel_args.count args) ;
      Spoc_framework.Kernel_args.set
        args
        idx
        (ArgBuffer {buf = buf.Memory.buf.Opencl_api.Memory.handle; idx}) ;
      Spoc_core.Log.debugf
        Spoc_core.Log.Kernel
        "OpenCL set_arg_buffer done (after=%d)"
        (Spoc_framework.Kernel_args.count args)

    let set_arg_int32 args idx value =
      Spoc_core.Log.debugf
        Spoc_core.Log.Kernel
        "OpenCL set_arg_int32 idx=%d"
        idx ;
      Spoc_framework.Kernel_args.set args idx (ArgInt32 {value; idx})

    let set_arg_int64 args idx value =
      Spoc_framework.Kernel_args.set args idx (ArgInt64 {value; idx})

    let set_arg_float32 args idx value =
      Spoc_framework.Kernel_args.set args idx (ArgFloat32 {value; idx})

    let set_arg_float64 args idx value =
      Spoc_framework.Kernel_args.set args idx (ArgFloat64 {value; idx})

    let set_arg_ptr _args _idx _ptr =
      Opencl_error.raise_error
        (Opencl_error.feature_not_supported "raw pointer arguments")

    let launch kernel ~args ~grid ~block ~shared_mem:_ ~stream =
      let open Framework_sig in
      (* KNOWN GAP: OpenCL kernel handles carry no arity metadata in this
         plugin (CL_KERNEL_NUM_ARGS is queryable via clGetKernelInfo but not
         currently wired up), so -- as with Native/CUDA/Metal --
         expected_count falls back to the number of distinct indices
         actually set. This still rejects internal gaps/duplicates but
         cannot catch a caller that consistently omits a trailing
         argument. *)
      let expected_count = Spoc_framework.Kernel_args.count args in
      Spoc_core.Log.debugf
        Spoc_core.Log.Kernel
        "OpenCL launch: args count=%d"
        expected_count ;
      let ordered_args =
        match
          Spoc_framework.Kernel_args.validate_and_extract args ~expected_count
        with
        | Ok arr -> arr
        | Error reason ->
            Opencl_error.raise_error
              (Opencl_error.kernel_launch_failed
                 (Printf.sprintf "device %d" kernel.device_id)
                 reason)
      in
      let state = get_state kernel.device_id in
      let queue =
        match stream with Some s -> s.Stream.queue | None -> state.queue
      in

      (* Set arguments -- last-set-wins per idx is already resolved by
         Kernel_args, so each idx is set exactly once here. *)
      Array.iter
        (function
          | ArgBuffer {buf; idx} ->
              Opencl_api.Kernel.set_arg_mem kernel.kernel idx buf
          | ArgInt32 {value; idx} ->
              Opencl_api.Kernel.set_arg_int32 kernel.kernel idx value
          | ArgInt64 {value; idx} ->
              Opencl_api.Kernel.set_arg_int64 kernel.kernel idx value
          | ArgFloat32 {value; idx} ->
              Opencl_api.Kernel.set_arg_float32 kernel.kernel idx value
          | ArgFloat64 {value; idx} ->
              Opencl_api.Kernel.set_arg_float64 kernel.kernel idx value)
        ordered_args ;

      (* Calculate global work size = grid * block *)
      let global = (grid.x * block.x, grid.y * block.y, grid.z * block.z) in
      let local = (block.x, block.y, block.z) in

      Opencl_api.Kernel.launch queue kernel.kernel ~global ~local
  end

  let profiling_enabled = ref false

  let enable_profiling () = profiling_enabled := true

  let disable_profiling () = profiling_enabled := false

  let is_available = Opencl_api.is_available
end

(* Legacy init retained for compatibility; backend registration now handled by
   Opencl_plugin. *)
let init () = ()
