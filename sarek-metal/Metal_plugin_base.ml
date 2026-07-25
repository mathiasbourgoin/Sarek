(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Metal Plugin - Framework Implementation
 *
 * Implements the Framework_sig.S interface for Metal devices.
 * This plugin is auto-registered when loaded.
 ******************************************************************************)

[@@@warning "-69"]

open Spoc_framework

module Metal : Framework_sig.PLUGIN_BASE = struct
  let name = "Metal"

  let version = (1, 0, 0)

  (* Per-device state: command queue *)
  type device_state = {
    device : Metal_api.Device.t;
    queue : Metal_api.CommandQueue.t;
  }

  let device_states : (int, device_state) Hashtbl.t = Hashtbl.create 8

  (** Current device for kernel compilation/execution *)
  let current_device : Metal_api.Device.t option ref = ref None

  let get_state device_id =
    match Hashtbl.find_opt device_states device_id with
    | Some s -> s
    | None ->
        let device = Metal_api.Device.get device_id in
        let queue = Metal_api.CommandQueue.create device in
        let state = {device; queue} in
        Hashtbl.add device_states device_id state ;
        state

  module Device = struct
    type t = Metal_api.Device.t

    type id = int

    let init = Metal_api.Device.init

    let count = Metal_api.Device.count

    let get = Metal_api.Device.get

    let id (d : t) = d.Metal_api.Device.id

    let name (d : t) = d.Metal_api.Device.name ^ " (Metal)"

    let capabilities (d : t) : Framework_sig.capabilities =
      let open Metal_api.Device in
      let open Metal_types in
      let open Ctypes in
      try
        let max_threads = d.max_threads_per_threadgroup in
        let width = Unsigned.Size_t.to_int (getf max_threads mtl_size_width) in
        let height =
          Unsigned.Size_t.to_int (getf max_threads mtl_size_height)
        in
        let depth = Unsigned.Size_t.to_int (getf max_threads mtl_size_depth) in
        {
          Framework_sig.max_threads_per_block = width * height * depth;
          max_block_dims = (width, height, depth);
          max_grid_dims = (max_int, max_int, max_int);
          shared_mem_per_block = d.max_threadgroup_memory;
          total_global_mem = Int64.of_int (4 * 1024 * 1024 * 1024);
          (* Estimate, Metal doesn't expose this *)
          compute_capability = (0, 0);
          supports_fp64 = d.supports_fp64;
          supports_atomics = true;
          warp_size = 32;
          (* SIMD width on Apple GPUs *)
          max_registers_per_block = 0;
          clock_rate_khz = 0;
          multiprocessor_count = 0;
          (* Metal doesn't expose these *)
          is_cpu = d.is_cpu;
        }
      with e -> raise e

    let set_current (d : t) =
      let _ = get_state d.id in
      current_device := Some d

    let get_current_device () = !current_device

    let synchronize (d : t) =
      let state = get_state d.id in
      (* Metal synchronizes via command buffers, not devices *)
      ignore state ;
      ()
  end

  module Memory = struct
    type 'a buffer = {buf : 'a Metal_api.Memory.buffer; device_id : int}

    let alloc device size kind =
      (* NOT Ctypes_static.sizeof (Ctypes.typ_of_bigarray_kind kind): ctypes'
         kind GADT has no Float16 arm and raises Failure "Unsupported bigarray
         kind", so [Vector.create Vector.float16 n] died here with an opaque
         ctypes error (#57 slice 1 review, MF2). *)
      let elem_size = Spoc_core.Memory.bigarray_elem_size kind in
      let buf = Metal_api.Memory.alloc device size elem_size in
      {buf; device_id = device.id}

    let alloc_zero_copy _device _ba _kind =
      (* Metal uses shared memory mode by default, effectively zero-copy *)
      None

    let is_zero_copy _b = false (* Metal doesn't distinguish zero-copy *)

    let alloc_custom device ~size ~elem_size =
      let buf = Metal_api.Memory.alloc device size elem_size in
      {buf; device_id = device.id}

    let free b = Metal_api.Memory.release b.buf

    (* Host pointers come from [Spoc_core.Memory.bigarray_void_ptr], not
       [Ctypes.bigarray_start]: the latter raises Failure "Unsupported bigarray
       kind" for Float16 (MF2), and the former is MANAGED so the bigarray stays
       GC-rooted across the memcpy (MF3). *)
    let host_to_device ~src ~dst =
      (* Metal shared memory: just memcpy *)
      let ba_ptr = Spoc_core.Memory.bigarray_void_ptr src in
      let byte_size =
        Bigarray.Array1.dim src * dst.buf.Metal_api.Memory.elem_size
      in
      Metal_api.memcpy ~dst:dst.buf.contents ~src:ba_ptr ~size:byte_size ;
      ignore (Sys.opaque_identity src)

    let device_to_host ~src ~dst =
      (* Metal shared memory: just memcpy *)
      let ba_ptr = Spoc_core.Memory.bigarray_void_ptr dst in
      let byte_size =
        Bigarray.Array1.dim dst * src.buf.Metal_api.Memory.elem_size
      in
      Metal_api.memcpy ~dst:ba_ptr ~src:src.buf.contents ~size:byte_size ;
      ignore (Sys.opaque_identity dst)

    let host_ptr_to_device ~src_ptr ~byte_size ~dst =
      Metal_api.memcpy
        ~dst:dst.buf.contents
        ~src:(Ctypes.ptr_of_raw_address src_ptr)
        ~size:byte_size

    let device_to_host_ptr ~src ~dst_ptr ~byte_size =
      Metal_api.memcpy
        ~dst:(Ctypes.ptr_of_raw_address dst_ptr)
        ~src:src.buf.contents
        ~size:byte_size

    let device_to_device ~src ~dst =
      let byte_size =
        min (src.buf.size * src.buf.elem_size) (dst.buf.size * dst.buf.elem_size)
      in
      Metal_api.memcpy
        ~dst:dst.buf.contents
        ~src:src.buf.contents
        ~size:byte_size

    let size b = b.buf.Metal_api.Memory.size

    let device_ptr b = Ctypes.raw_address_of_ptr b.buf.Metal_api.Memory.contents
  end

  module Stream = struct
    type t = {queue : Metal_api.CommandQueue.t; device_id : int}

    let create device =
      let queue = Metal_api.CommandQueue.create device in
      {queue; device_id = device.id}

    let destroy stream = Metal_api.CommandQueue.release stream.queue

    let synchronize _stream = ()
    (* Metal commands are synchronous by default *)

    let default device =
      let state = get_state device.Metal_api.Device.id in
      {queue = state.queue; device_id = device.id}
  end

  module Event = struct
    type t = {mutable start_time : float; mutable end_time : float}

    let create () = {start_time = 0.0; end_time = 0.0}

    let destroy _event = ()

    let record event _stream =
      event.start_time <- event.end_time ;
      event.end_time <- Unix.gettimeofday ()

    let synchronize _event = ()

    let elapsed ~start ~stop = (stop.end_time -. start.start_time) *. 1000.0
  end

  module Kernel = struct
    type compiled = {
      library : Metal_api.Library.t;
      pipeline : Metal_api.ComputePipeline.t;
      function_name : string;
      device_id : int;
    }

    type t = compiled

    type arg =
      | ArgBuffer of {buf : Metal_types.mtl_buffer; idx : int}
      | ArgInt32 of {value : int32; idx : int}
      | ArgInt64 of {value : int64; idx : int}
      | ArgFloat32 of {value : float; idx : int}
      | ArgFloat64 of {value : float; idx : int}

    (* Indexed by idx (last-set-wins) instead of accumulated by call order:
       see Spoc_framework.Kernel_args. *)
    type args = arg Spoc_framework.Kernel_args.t

    (* Cache: key -> compiled kernel. Guarded against concurrent multi-domain
       access by [Spoc_framework.Guarded_cache]: lookup/insert and clearing are
       atomic critical sections, while the Metal library/pipeline compile runs
       outside the lock. *)
    let cache : (string, t) Spoc_framework.Guarded_cache.t =
      Spoc_framework.Guarded_cache.create
        ~destroy:(fun k ->
          Metal_api.Library.release k.library ;
          Metal_api.ComputePipeline.release k.pipeline)
        ()

    (* Per-device eviction, the model every backend cache follows (see
       Cache_hooks.mli). Metal exposes no device-destroy entry point, so nothing
       in this backend fires the notification today; registering anyway keeps
       the model uniform instead of "per-device on some backends, global on
       others". Match on the family name, never on the index alone: backend-local
       indices collide across backends. *)
    let () =
      Spoc_framework.Cache_hooks.on_device_destroy (fun ~backend index ->
          if String.equal backend "Metal" then
            Spoc_framework.Guarded_cache.evict_device cache index)

    let compile device ~name ~source =
      let library = Metal_api.Library.create_from_source device source in
      let func = Metal_api.Library.get_function library name in
      let pipeline = Metal_api.ComputePipeline.create device func in
      {library; pipeline; function_name = name; device_id = device.id}

    let compile_cached device ~name ~source =
      (* Cache key must include device ID and the kernel/entry name - a
         source file may define more than one kernel, and a resolved kernel
         handle for one name must never be returned for another (see
         Compile_cache.mli). Delegates to the shared, collision-resistant
         key builder used by CUDA/Vulkan instead of hand-rolling a
         delimiter-unsafe Printf.sprintf join. *)
      let key =
        Spoc_framework.Compile_cache.make_key
          ~device:(string_of_int device.Metal_api.Device.id)
          ~name
          ~source
          ()
      in
      (* [~device_id] is the same backend-local index the key already carries;
         without it the entry is not grouped by device and [evict_device] can
         never reach it. *)
      Spoc_framework.Guarded_cache.find_or_build
        cache
        ~key
        ~device_id:device.Metal_api.Device.id
        (fun () -> compile device ~name ~source)

    let clear_cache () =
      Spoc_framework.Cache_hooks.around_clear (fun () ->
          Spoc_framework.Guarded_cache.clear cache)

    let load_from_ptx ~name:_ ~ptx:_ =
      Metal_error.raise_error (Metal_error.feature_not_supported "PTX kernels")

    let create_args () = Spoc_framework.Kernel_args.create ()

    let set_arg_buffer args idx buf =
      Spoc_framework.Kernel_args.set
        args
        idx
        (ArgBuffer {buf = buf.Memory.buf.Metal_api.Memory.handle; idx})

    let set_arg_int32 args idx value =
      Spoc_framework.Kernel_args.set args idx (ArgInt32 {value; idx})

    let set_arg_int64 args idx value =
      Spoc_framework.Kernel_args.set args idx (ArgInt64 {value; idx})

    let set_arg_float32 args idx value =
      Spoc_framework.Kernel_args.set args idx (ArgFloat32 {value; idx})

    let set_arg_float64 args idx value =
      Spoc_framework.Kernel_args.set args idx (ArgFloat64 {value; idx})

    let set_arg_ptr _args _idx _ptr =
      Metal_error.raise_error
        (Metal_error.feature_not_supported "raw pointer kernel arguments")

    let launch kernel ~args ~grid ~block ~shared_mem:_ ~stream =
      let open Framework_sig in
      let state = get_state kernel.device_id in
      let queue =
        match stream with Some s -> s.Stream.queue | None -> state.queue
      in

      (* KNOWN GAP: Metal compiled-kernel handles carry no arity metadata, so
         -- as with Native/CUDA -- expected_count falls back to the number
         of distinct indices actually set. This still rejects internal
         gaps/duplicates but cannot catch a caller that consistently omits a
         trailing argument. *)
      let expected_count = Spoc_framework.Kernel_args.count args in
      let ordered_args =
        match
          Spoc_framework.Kernel_args.validate_and_extract args ~expected_count
        with
        | Ok arr -> arr
        | Error reason ->
            Metal_error.raise_error
              (Metal_error.kernel_launch_failed kernel.function_name reason)
      in
      (* Convert to Metal_api.Kernel.arg format. Metal_api.Kernel.execute
         binds each element of this list to its *position* in the list
         (atIndex:), so ordering here -- by idx, via validate_and_extract --
         is what makes the binding correct, not the arg's embedded idx
         field. *)
      let metal_args =
        Array.to_list ordered_args
        |> List.map (function
          | ArgBuffer {buf; idx = _} -> Metal_api.Kernel.Buffer (buf, 0)
          | ArgInt32 {value; idx = _} -> Metal_api.Kernel.Int32 value
          | ArgInt64 {value; idx = _} -> Metal_api.Kernel.Int64 value
          | ArgFloat32 {value; idx = _} -> Metal_api.Kernel.Float32 value
          | ArgFloat64 {value; idx = _} -> Metal_api.Kernel.Float64 value)
      in

      (* Calculate grid and block sizes *)
      let grid_size = (grid.x * block.x, grid.y * block.y, grid.z * block.z) in
      let block_size = (block.x, block.y, block.z) in

      (* Execute kernel *)
      let metal_kernel =
        Metal_api.Kernel.create kernel.pipeline kernel.function_name
      in
      Metal_api.Kernel.execute
        queue
        metal_kernel
        ~grid_size
        ~block_size
        metal_args
  end

  let profiling_enabled = ref false

  let enable_profiling () = profiling_enabled := true

  let disable_profiling () = profiling_enabled := false

  let is_available = Metal_bindings.is_available
end

let init () = ()
