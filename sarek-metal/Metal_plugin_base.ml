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
      let elem_size = Ctypes_static.sizeof (Ctypes.typ_of_bigarray_kind kind) in
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

    let host_to_device ~src ~dst =
      (* Metal shared memory: just memcpy *)
      let ba_ptr = Ctypes.(bigarray_start array1 src) in
      let byte_size =
        Bigarray.Array1.dim src * dst.buf.Metal_api.Memory.elem_size
      in
      Metal_api.memcpy
        ~dst:dst.buf.contents
        ~src:(Ctypes.to_voidp ba_ptr)
        ~size:byte_size

    let device_to_host ~src ~dst =
      (* Metal shared memory: just memcpy *)
      let ba_ptr = Ctypes.(bigarray_start array1 dst) in
      let byte_size =
        Bigarray.Array1.dim dst * src.buf.Metal_api.Memory.elem_size
      in
      Metal_api.memcpy
        ~dst:(Ctypes.to_voidp ba_ptr)
        ~src:src.buf.contents
        ~size:byte_size

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

    type args = arg list ref

    (* Cache: key -> compiled kernel *)
    let cache : (string, t) Hashtbl.t = Hashtbl.create 16

    let compile device ~name ~source =
      let library = Metal_api.Library.create_from_source device source in
      let func = Metal_api.Library.get_function library name in
      let pipeline = Metal_api.ComputePipeline.create device func in
      {library; pipeline; function_name = name; device_id = device.id}

    let compile_cached device ~name ~source =
      let key =
        Printf.sprintf
          "%d:%s:%s"
          device.Metal_api.Device.id
          name
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
          Metal_api.Library.release k.library ;
          Metal_api.ComputePipeline.release k.pipeline)
        cache ;
      Hashtbl.clear cache

    let load_from_ptx ~name:_ ~ptx:_ =
      Metal_error.raise_error (Metal_error.feature_not_supported "PTX kernels")

    let create_args () = ref []

    let set_arg_buffer args idx buf =
      args :=
        ArgBuffer {buf = buf.Memory.buf.Metal_api.Memory.handle; idx} :: !args

    let set_arg_int32 args idx value = args := ArgInt32 {value; idx} :: !args

    let set_arg_int64 args idx value = args := ArgInt64 {value; idx} :: !args

    let set_arg_float32 args idx value =
      args := ArgFloat32 {value; idx} :: !args

    let set_arg_float64 args idx value =
      args := ArgFloat64 {value; idx} :: !args

    let set_arg_ptr _args _idx _ptr =
      Metal_error.raise_error
        (Metal_error.feature_not_supported "raw pointer kernel arguments")

    let launch kernel ~args ~grid ~block ~shared_mem:_ ~stream =
      let open Framework_sig in
      let state = get_state kernel.device_id in
      let queue =
        match stream with Some s -> s.Stream.queue | None -> state.queue
      in

      (* Convert args to Metal_api.Kernel.arg format *)
      let metal_args =
        List.map
          (function
            | ArgBuffer {buf; idx = _} -> Metal_api.Kernel.Buffer (buf, 0)
            | ArgInt32 {value; idx = _} -> Metal_api.Kernel.Int32 value
            | ArgInt64 {value; idx = _} -> Metal_api.Kernel.Int64 value
            | ArgFloat32 {value; idx = _} -> Metal_api.Kernel.Float32 value
            | ArgFloat64 {value; idx = _} -> Metal_api.Kernel.Float64 value)
          !args
        |> List.rev
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
