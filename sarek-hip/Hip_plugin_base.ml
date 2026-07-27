(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * HIP Plugin Base - Framework_sig.PLUGIN_BASE implementation.
 *
 * Device/Memory/Stream/Event/Kernel core on top of Hip_api, mirroring
 * Cuda_plugin_base. The full BACKEND (codegen + run_source + intrinsics) is
 * assembled on top of this in Hip_plugin.ml.
 ******************************************************************************)

open Spoc_framework

module Hip : Framework_sig.PLUGIN_BASE = struct
  let name = "HIP"

  let version = (7, 2, 0)

  let current_device : Hip_api.Device.t option ref = ref None

  module Device = struct
    type t = Hip_api.Device.t

    type id = int

    let init = Hip_api.Device.init

    let count = Hip_api.Device.count

    let get = Hip_api.Device.get

    let id (d : t) = d.Hip_api.Device.id

    let name (d : t) = d.Hip_api.Device.name

    let set_current (d : t) =
      Hip_api.Device.set_current d ;
      current_device := Some d

    let get_current_device () = !current_device

    let synchronize = Hip_api.Device.synchronize

    let capabilities (d : t) : Framework_sig.capabilities =
      let open Hip_api.Device in
      {
        Framework_sig.max_threads_per_block = d.max_threads_per_block;
        max_block_dims = d.max_block_dims;
        max_grid_dims = d.max_grid_dims;
        shared_mem_per_block = d.shared_mem_per_block;
        total_global_mem = d.total_mem;
        compute_capability = d.compute_capability;
        (* As CUDA: fp64 and int64 are core on every HIP target. Float16 is
           omitted for the same reason — unprobed, so not claimed. *)
        device_features = [Sarek_ir_analysis.Float64; Sarek_ir_analysis.Int64];
        supports_atomics = true;
        warp_size = d.warp_size;
        max_registers_per_block = 65536;
        clock_rate_khz = 0;
        multiprocessor_count = d.multiprocessor_count;
        is_cpu = false;
      }
  end

  module Memory = struct
    type 'a buffer = 'a Hip_api.Memory.buffer

    let alloc = Hip_api.Memory.alloc

    let alloc_custom = Hip_api.Memory.alloc_custom

    let alloc_zero_copy _device _ba _kind = None

    let is_zero_copy _buf = false

    let free = Hip_api.Memory.free

    let host_to_device = Hip_api.Memory.host_to_device

    let device_to_host = Hip_api.Memory.device_to_host

    let host_ptr_to_device ~src_ptr ~byte_size ~dst =
      Hip_api.Memory.host_ptr_to_device
        ~src_ptr:(Ctypes.ptr_of_raw_address src_ptr)
        ~byte_size
        ~dst

    let device_to_host_ptr ~src ~dst_ptr ~byte_size =
      Hip_api.Memory.device_to_host_ptr
        ~src
        ~dst_ptr:(Ctypes.ptr_of_raw_address dst_ptr)
        ~byte_size

    let device_to_device = Hip_api.Memory.device_to_device

    let size (buf : 'a buffer) = buf.Hip_api.Memory.size

    let device_ptr (buf : 'a buffer) =
      Ctypes.raw_address_of_ptr (Ctypes.to_voidp buf.Hip_api.Memory.ptr)
  end

  module Stream = struct
    type t = Hip_api.Stream.t

    let create = Hip_api.Stream.create

    let destroy = Hip_api.Stream.destroy

    let synchronize = Hip_api.Stream.synchronize

    let default = Hip_api.Stream.default
  end

  module Event = struct
    type t = Hip_api.Event.t

    let create = Hip_api.Event.create

    let destroy = Hip_api.Event.destroy

    let record = Hip_api.Event.record

    let synchronize = Hip_api.Event.synchronize

    let elapsed = Hip_api.Event.elapsed
  end

  module Kernel = struct
    type t = Hip_api.Kernel.t

    type args = Hip_api.Kernel.arg Spoc_framework.Kernel_args.t

    let compile dev ~name ~source = Hip_api.Kernel.compile dev ~name ~source

    let compile_cached dev ~name ~source =
      Hip_api.Kernel.compile_cached dev ~name ~source

    let clear_cache = Hip_api.Kernel.clear_cache

    (* HIP has no PTX path; kernels are HIP C++ compiled via hiprtc. This is
       required by the interface but never invoked (the HIP backend advertises
       only CUDA_Source and never registers a PTX run path). *)
    let load_from_ptx ~name:_ ~ptx:_ =
      Hip_error.raise_error (Hip_error.unsupported_source_lang "PTX")

    let create_args () = Spoc_framework.Kernel_args.create ()

    let set_arg_buffer args idx buf =
      Spoc_framework.Kernel_args.set args idx (Hip_api.Kernel.ArgBuffer buf)

    let set_arg_int32 args idx v =
      Spoc_framework.Kernel_args.set args idx (Hip_api.Kernel.ArgInt32 v)

    let set_arg_int64 args idx v =
      Spoc_framework.Kernel_args.set args idx (Hip_api.Kernel.ArgInt64 v)

    let set_arg_float32 args idx v =
      Spoc_framework.Kernel_args.set args idx (Hip_api.Kernel.ArgFloat32 v)

    let set_arg_float64 args idx v =
      Spoc_framework.Kernel_args.set args idx (Hip_api.Kernel.ArgFloat64 v)

    let set_arg_ptr args idx ptr =
      Spoc_framework.Kernel_args.set args idx (Hip_api.Kernel.ArgPtr ptr)

    let launch kernel ~args ~grid ~block ~shared_mem ~stream =
      let open Framework_sig in
      let expected_count = Spoc_framework.Kernel_args.count args in
      let arg_list =
        match
          Spoc_framework.Kernel_args.validate_and_extract args ~expected_count
        with
        | Ok arr -> Array.to_list arr
        | Error reason ->
            Hip_error.(
              raise_error
                (kernel_launch_failed kernel.Hip_api.Kernel.name reason))
      in
      Hip_api.Kernel.launch
        kernel
        ~args:arg_list
        ~grid:(grid.x, grid.y, grid.z)
        ~block:(block.x, block.y, block.z)
        ~shared_mem
        ~stream
  end

  let enable_profiling () = ()

  let disable_profiling () = ()

  let is_available = Hip_api.is_available
end
