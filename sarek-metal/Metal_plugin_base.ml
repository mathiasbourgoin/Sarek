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

  (* Several domains may launch or copy at once (a downstream consumer issues expert copies
     from an I/O domain): the lazily filled table is guarded. *)
  let device_states_lock = Mutex.create ()

  (** Current device for kernel compilation/execution *)
  let current_device : Metal_api.Device.t option ref = ref None

  let get_state device_id =
    Mutex.protect device_states_lock (fun () ->
        match Hashtbl.find_opt device_states device_id with
        | Some s -> s
        | None ->
            let device = Metal_api.Device.get device_id in
            let queue = Metal_api.CommandQueue.create device in
            let state = {device; queue} in
            Hashtbl.add device_states device_id state ;
            state)

  module Device = struct
    type t = Metal_api.Device.t

    type id = int

    let init = Metal_api.Device.init

    let count = Metal_api.Device.count

    let get = Metal_api.Device.get

    let id (d : t) = d.Metal_api.Device.id

    let name (d : t) = d.Metal_api.Device.name ^ " (Metal)"

    (* The cooperative-matrix probe compiles a kernel on the device, so it is
       memoised: [capabilities] is read repeatedly and a runtime compile is
       milliseconds. A compile failure is the answer "this device offers no
       configuration" (Metal_coopmat.support), not an error. *)
    let coopmat_cache : (int, Sarek_coopmat.device_support) Hashtbl.t =
      Hashtbl.create 4

    let coopmat_cache_lock = Mutex.create ()

    let probe_coopmat (d : t) : Sarek_coopmat.device_support * bool =
      let compile () =
        let lib =
          Metal_api.Library.create_from_source d Metal_coopmat.probe_source
        in
        Fun.protect
          ~finally:(fun () -> Metal_api.Library.release lib)
          (fun () ->
            let func =
              Metal_api.Library.get_function lib Metal_coopmat.probe_kernel_name
            in
            let pipeline = Metal_api.ComputePipeline.create d func in
            Fun.protect
              ~finally:(fun () -> Metal_api.ComputePipeline.release pipeline)
              (fun () ->
                pipeline.Metal_api.ComputePipeline.thread_execution_width))
      in
      (* A compiler rejection is deterministic: the answer "no configuration"
         is final and cached. Any other failure (API error, allocation) may be
         transient: it is reported the same way but not cached, so the next
         [capabilities] call probes again. Neither escapes [capabilities] /
         [Device.init]. Apple GPUs are 32 wide; the width is used only to
         describe a device offering nothing. *)
      let simd_width, probe, final =
        match compile () with
        | width -> (width, Ok (), true)
        | exception (Sarek_backend_error.Backend_error.Backend_error _ as e) ->
            (32, Error (Printexc.to_string e), true)
        | exception e -> (32, Error (Printexc.to_string e), false)
      in
      (Metal_coopmat.support ~simd_width ~probe, final)

    let coopmat_support (d : t) =
      Mutex.protect coopmat_cache_lock (fun () ->
          match Hashtbl.find_opt coopmat_cache d.Metal_api.Device.id with
          | Some s -> s
          | None ->
              let s, final = probe_coopmat d in
              if final then
                Hashtbl.replace coopmat_cache d.Metal_api.Device.id s ;
              s)

    let capabilities (d : t) : Framework_sig.capabilities =
      let open Metal_api.Device in
      let open Metal_types in
      let open Ctypes in
      let coopmat_support = coopmat_support d in
      let max_threads = d.max_threads_per_threadgroup in
      let width = Unsigned.Size_t.to_int (getf max_threads mtl_size_width) in
      let height = Unsigned.Size_t.to_int (getf max_threads mtl_size_height) in
      let depth = Unsigned.Size_t.to_int (getf max_threads mtl_size_depth) in
      {
        Framework_sig.max_threads_per_block = width * height * depth;
        max_block_dims = (width, height, depth);
        max_grid_dims = (max_int, max_int, max_int);
        shared_mem_per_block = d.max_threadgroup_memory;
        (* [recommendedMaxWorkingSetSize]: on Apple silicon there is no
             separate VRAM, and this is the share of unified memory the GPU
             can keep resident — what a cache sized from this field must stay
             under. *)
        total_global_mem =
          Int64.of_int (Metal_api.Device.recommended_working_set d);
        compute_capability = (0, 0);
        (* [d.supports_fp64] is hardcoded false in Metal_api: MSL has no
             [double] at all, which is Backend_structural and refused at
             codegen through Sarek_capability.float64_absent_metal. It is
             threaded through rather than dropped so the two cannot disagree.
             MSL does have [long], so int64 is provided. *)
        device_features =
          (if d.supports_fp64 then [Sarek_ir_analysis.Float64] else [])
          @ [Sarek_ir_analysis.Int64; Sarek_ir_analysis.Warp_shuffle]
          (* Warp_shuffle: simd_shuffle is part of MSL for every device Sarek
             supports (Apple GPU family 4+, macOS 10.14+). Not executed in CI;
             golden text only (docs/plans/warp-shuffle-intake-2026-10-17.md,
             decision D7). *);
        (* Probed by compiling a simdgroup_matrix kernel on the device
             (specs/coopmat-metal-simdgroup.md): [Some] with the lowered
             configuration, or [Some] with none when the compiler rejects it. *)
        coopmat = Some coopmat_support;
        supports_atomics = true;
        warp_size = coopmat_support.Sarek_coopmat.ds_subgroup_size;
        (* SIMD width, read from a compiled pipeline *)
        max_registers_per_block = 0;
        clock_rate_khz = 0;
        multiprocessor_count = 0;
        (* Metal doesn't expose these *)
        is_cpu = d.is_cpu;
      }

    let set_current (d : t) =
      let _ = get_state d.id in
      current_device := Some d

    let get_current_device () = !current_device

    (* Launches are asynchronous (Metal_api.Pending): wait for every pending
       command buffer. Pending work is not tracked per device; on a Mac with
       one GPU that is the same thing, and on several it only over-waits. *)
    let synchronize (_ : t) = Metal_api.Pending.wait_all ()
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

    let alloc_host_ptr _device _ba _kind =
      Error Framework_sig.Host_ptr_unsupported

    let is_zero_copy _b = false (* Metal doesn't distinguish zero-copy *)

    let alloc_custom device ~size ~elem_size =
      let buf = Metal_api.Memory.alloc device size elem_size in
      {buf; device_id = device.id}

    let free b = Metal_api.Memory.release b.buf

    (* Host pointers come from [Spoc_core.Memory.borrow_bigarray_ptr], not
       [Ctypes.bigarray_start]: the latter raises Failure "Unsupported bigarray
       kind" for Float16. The pointer does not root the bigarray; the
       [Sys.opaque_identity] after each memcpy keeps it reachable. *)
    (* Every host access to a buffer's [contents] first waits for the pending
       command buffers that bind it (Metal_api.Pending): a write must not land
       under a kernel still reading, a read must see what the kernel wrote. *)
    let wait b = Metal_api.Pending.wait_buffers [b.buf.Metal_api.Memory.handle]

    let host_to_device ~src ~dst =
      wait dst ;
      (* Metal shared memory: just memcpy *)
      let ba_ptr = Spoc_core.Memory.borrow_bigarray_ptr src in
      let byte_size =
        Bigarray.Array1.dim src * dst.buf.Metal_api.Memory.elem_size
      in
      Metal_api.memcpy ~dst:dst.buf.contents ~src:ba_ptr ~size:byte_size ;
      ignore (Sys.opaque_identity src)

    let device_to_host ~src ~dst =
      wait src ;
      (* Metal shared memory: just memcpy *)
      let ba_ptr = Spoc_core.Memory.borrow_bigarray_ptr dst in
      let byte_size =
        Bigarray.Array1.dim dst * src.buf.Metal_api.Memory.elem_size
      in
      Metal_api.memcpy ~dst:ba_ptr ~src:src.buf.contents ~size:byte_size ;
      ignore (Sys.opaque_identity dst)

    let host_ptr_to_device ~src_ptr ~byte_size ~dst =
      wait dst ;
      Metal_api.memcpy
        ~dst:dst.buf.contents
        ~src:(Ctypes.ptr_of_raw_address src_ptr)
        ~size:byte_size

    let device_to_host_ptr ~src ~dst_ptr ~byte_size =
      wait src ;
      Metal_api.memcpy
        ~dst:(Ctypes.ptr_of_raw_address dst_ptr)
        ~src:src.buf.contents
        ~size:byte_size

    let device_to_device ~src ~dst =
      let byte_size =
        min (src.buf.size * src.buf.elem_size) (dst.buf.size * dst.buf.elem_size)
      in
      Metal_api.Pending.wait_buffers
        [src.buf.Metal_api.Memory.handle; dst.buf.Metal_api.Memory.handle] ;
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

    (* Each Metal command queue is independent; no implicit ordering against
       the default queue exists. *)
    let create_non_blocking = create

    let synchronize stream =
      Metal_api.Pending.wait_queue stream.queue.Metal_api.CommandQueue.handle

    (* Released even when the wait reports a failed command buffer. *)
    let destroy stream =
      Fun.protect
        ~finally:(fun () ->
          Metal_api.Pending.forget_queue
            stream.queue.Metal_api.CommandQueue.handle ;
          Metal_api.CommandQueue.release stream.queue)
        (fun () -> synchronize stream)

    let default device =
      let state = get_state device.Metal_api.Device.id in
      {queue = state.queue; device_id = device.id}
  end

  (* An event is the newest pending command buffer of the stream it was
     recorded on (retained), or none when that stream had no pending work:
     command buffers on one queue complete in commit order, so it completes
     after everything enqueued on the stream before the record. *)
  module Event = struct
    type t = {
      mutable host_time : float;
          (** [CLOCK_UPTIME_RAW] seconds at the last record. *)
      mutable cmd : unit Ctypes.ptr option;  (** Retained. *)
      lock : Mutex.t;
    }

    let create () = {host_time = 0.0; cmd = None; lock = Mutex.create ()}

    let take event =
      Mutex.protect event.lock (fun () ->
          let c = event.cmd in
          event.cmd <- None ;
          c)

    let destroy event = Option.iter Metal_bindings.release (take event)

    let record event (stream : Stream.t) =
      let c =
        Metal_api.Pending.last_on_queue
          stream.Stream.queue.Metal_api.CommandQueue.handle
      in
      let old =
        Mutex.protect event.lock (fun () ->
            let old = event.cmd in
            event.cmd <- c ;
            event.host_time <- Metal_bindings.uptime_raw_seconds () ;
            old)
      in
      Option.iter Metal_bindings.release old

    (* The recorded command buffer, retained for the caller. *)
    let cmd_ref event =
      Mutex.protect event.lock (fun () ->
          Option.iter (fun c -> ignore (Metal_bindings.retain c)) event.cmd ;
          event.cmd)

    let with_cmd event f =
      match cmd_ref event with
      | None -> None
      | Some c ->
          Some
            (Fun.protect
               ~finally:(fun () -> Metal_bindings.release c)
               (fun () -> f c))

    (* Through Pending, so a failure is reported once, whoever observes it. *)
    let synchronize event =
      ignore
        (with_cmd event (fun c ->
             Metal_api.Pending.wait_cmd c ;
             (* No longer pending (or never was): another waiter may still be
                waiting on it and reports it; wait without reporting. *)
             Metal_bindings.mtl_command_buffer_wait_until_completed c))

    (* Completion, not success, is what is queried; a failed command buffer is
       reported here, where it is observed. *)
    let completed event =
      match
        with_cmd event (fun c -> Metal_bindings.mtl_command_buffer_status c)
      with
      | None -> true
      | Some s when s = Metal_bindings.mtl_command_buffer_status_error ->
          synchronize event ;
          true
      | Some s -> s >= Metal_api.Pending.mtl_status_completed

    (* When the work the event covers finished, on CLOCK_UPTIME_RAW: the later
       of the record time and the GPU end time of the command buffer it
       captured (one that finished before the record must not date the event
       back). *)
    let time_point event =
      synchronize event ;
      let recorded = Mutex.protect event.lock (fun () -> event.host_time) in
      match
        with_cmd event (fun c ->
            Metal_bindings.mtl_command_buffer_gpu_end_time c)
      with
      | Some t -> Float.max t recorded
      | None -> recorded

    let elapsed ~start ~stop = (time_point stop -. time_point start) *. 1000.0
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
        match stream with
        | Some s ->
            (* A pipeline compiled for one device cannot run on another
               device's queue. *)
            if s.Stream.device_id <> kernel.device_id then
              Metal_error.raise_error
                (Metal_error.kernel_launch_failed
                   kernel.function_name
                   (Printf.sprintf
                      "stream of device %d, kernel compiled for device %d"
                      s.Stream.device_id
                      kernel.device_id)) ;
            s.Stream.queue
        | None -> state.queue
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

  (* Ranged copies, pinned memory and stream ordering on unified memory.

     Storage: every Metal buffer, device-only ones included, is
     [MTLResourceStorageModeShared]. On Apple silicon the GPU reads the same
     physical memory the CPU writes, so [contents + offset] IS the device
     range and a ranged copy is one memcpy. Private storage would need a
     shared staging buffer and a blit encoder: the same bytes moved twice.

     Ordering: launches are asynchronous and batched per command queue
     (Metal_api.Pending); a launch on a stream goes to that stream's own
     queue ([stream_launch = true]). Copies are CPU memcpys that complete
     before returning ([async_copy = false]: overlap with compute comes from
     the asynchronous launches, or from issuing copies from another domain).
     Each copy first waits for the pending command buffers that bind its
     buffer, so it never lands under a kernel still reading it and always
     sees what a kernel wrote; a copy issued before a launch is visible to it
     (CPU writes to shared storage before [commit]). An event is the newest
     command buffer of its stream at record time. [stream_wait_event] adds
     nothing: kernels depend on one another only through buffers, and a launch
     already waits for work on other queues that binds its buffers. Core's
     keep-alive ledger still roots stream operands until the stream or an
     event is synchronised.

     Pinned memory: unified memory has no DMA engine to page-lock for, so
     [alloc_pinned] is page-aligned host memory ([posix_memalign]) and
     [register_host] has nothing to do. Copies from either run at the same
     memcpy rate as from pageable memory. *)
  module Async = struct
    let support =
      {Framework_sig.full_async_support with Framework_sig.async_copy = false}

    let total (b : _ Memory.buffer) =
      b.Memory.buf.Metal_api.Memory.size
      * b.Memory.buf.Metal_api.Memory.elem_size

    let range_addr (b : _ Memory.buffer) off =
      Nativeint.add (Memory.device_ptr b) (Nativeint.of_int off)

    let host_to_device ~src_ptr ~dst ~dst_offset ~byte_size ~stream:_ =
      Memory.wait dst ;
      Spoc_core.Host_async.check_range
        ~what:"Metal.Async.host_to_device"
        ~total:(total dst)
        ~off:dst_offset
        ~byte_size ;
      Spoc_core.Host_async.blit
        ~src:src_ptr
        ~dst:(range_addr dst dst_offset)
        ~byte_size ;
      ignore (Sys.opaque_identity dst)

    let device_to_host ~src ~src_offset ~dst_ptr ~byte_size ~stream:_ =
      Memory.wait src ;
      Spoc_core.Host_async.check_range
        ~what:"Metal.Async.device_to_host"
        ~total:(total src)
        ~off:src_offset
        ~byte_size ;
      Spoc_core.Host_async.blit
        ~src:(range_addr src src_offset)
        ~dst:dst_ptr
        ~byte_size ;
      ignore (Sys.opaque_identity src)

    (* Kernels depend on one another only through buffers, and a launch
       already waits for pending work on other queues that binds its buffers
       (Metal_api.Pending), while host copies wait for the buffer itself: the
       ordering a wait would add is already there. *)
    let stream_wait_event _stream _event = ()

    let event_query event = Event.completed event

    (* Live [alloc_pinned] addresses: [free_pinned] refuses anything else. *)
    let pinned : (nativeint, unit) Hashtbl.t = Hashtbl.create 8

    let pinned_lock = Mutex.create ()

    let alloc_pinned _dev ~byte_size =
      if byte_size <= 0 then
        invalid_arg "Metal.Async.alloc_pinned: byte_size must be positive" ;
      let addr = Metal_api.alloc_aligned ~byte_size in
      Mutex.protect pinned_lock (fun () -> Hashtbl.replace pinned addr ()) ;
      addr

    let free_pinned _dev addr =
      Mutex.protect pinned_lock (fun () ->
          if not (Hashtbl.mem pinned addr) then
            invalid_arg "Metal.Async.free_pinned: not a pinned allocation" ;
          Hashtbl.remove pinned addr) ;
      Metal_api.free_aligned addr

    let register_host _dev _addr ~byte_size =
      if byte_size <= 0 then
        invalid_arg "Metal.Async.register_host: byte_size must be positive"

    let unregister_host _dev _addr = ()
  end

  let enable_profiling () = profiling_enabled := true

  let disable_profiling () = profiling_enabled := false

  let is_available = Metal_bindings.is_available
end

let init () = ()
