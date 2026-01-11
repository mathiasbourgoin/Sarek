(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Vulkan Plugin Base - Core Implementation
 *
 * Provides the BACKEND implementation wrapping Vulkan_api.
 * Bridges Vulkan API to the SPOC framework interface.
 ******************************************************************************)

open Spoc_framework

(** Vulkan Backend Implementation *)
module Vulkan = struct
  let name = "Vulkan"

  let version = (1, 0, 0)

  let is_available = Vulkan_api.is_available

  (** Current device tracking for run_source *)
  let current_device_ref : Vulkan_api.Device.t option ref = ref None

  module Device = struct
    type t = Vulkan_api.Device.t

    type id = int

    let init = Vulkan_api.Device.init

    let count = Vulkan_api.Device.count

    let get = Vulkan_api.Device.get

    let id dev = dev.Vulkan_api.Device.id

    let name dev = dev.Vulkan_api.Device.name

    let capabilities dev : Framework_sig.capabilities =
      (* Query Vulkan device properties and memory *)
      let major, minor, _ = dev.Vulkan_api.Device.api_version in
      let total_mem =
        Vulkan_api.Device.get_total_device_memory
          dev.Vulkan_api.Device.memory_properties
      in
      {
        max_threads_per_block = 1024;
        max_block_dims = (1024, 1024, 64);
        max_grid_dims = (65535, 65535, 65535);
        shared_mem_per_block = 49152;
        total_global_mem = total_mem;
        compute_capability = (major, minor);
        supports_fp64 = true;
        supports_atomics = true;
        warp_size = 32;
        (* subgroup size varies by vendor *)
        max_registers_per_block = 65536;
        clock_rate_khz = 1000000;
        multiprocessor_count = 1;
        is_cpu = false;
      }

    let set_current dev =
      current_device_ref := Some dev ;
      Vulkan_api.Device.set_current dev

    let synchronize = Vulkan_api.Device.synchronize

    let get_current_device () = !current_device_ref
  end

  module Stream = struct
    type t = Vulkan_api.Stream.t

    let create = Vulkan_api.Stream.create

    let destroy = Vulkan_api.Stream.destroy

    let synchronize = Vulkan_api.Stream.synchronize

    let default = Vulkan_api.Stream.default
  end

  module Memory = struct
    type 'a buffer = 'a Vulkan_api.Memory.buffer

    let alloc = Vulkan_api.Memory.alloc

    let alloc_custom = Vulkan_api.Memory.alloc_custom

    let alloc_zero_copy _dev _arr _kind = None
    (* Vulkan doesn't support zero-copy in this simple implementation *)

    let free = Vulkan_api.Memory.free

    let host_to_device = Vulkan_api.Memory.host_to_device

    let device_to_host = Vulkan_api.Memory.device_to_host

    let host_ptr_to_device = Vulkan_api.Memory.host_ptr_to_device

    let device_to_host_ptr = Vulkan_api.Memory.device_to_host_ptr

    let device_to_device = Vulkan_api.Memory.device_to_device

    let size buf = buf.Vulkan_api.Memory.size

    let device_ptr _buf = Nativeint.zero
    (* Vulkan buffers aren't directly addressable *)

    let is_zero_copy _buf = false
  end

  module Event = struct
    type t = Vulkan_api.Event.t

    let create () =
      match !current_device_ref with
      | Some dev -> Vulkan_api.Event.create_with_device dev
      | None ->
          Vulkan_error.raise_error
            (Vulkan_error.no_device_selected "Event.create")

    let destroy = Vulkan_api.Event.destroy

    let record = Vulkan_api.Event.record

    let synchronize = Vulkan_api.Event.synchronize

    let elapsed = Vulkan_api.Event.elapsed
  end

  module Kernel = struct
    type t = Vulkan_api.Kernel.t

    type args = Vulkan_api.Kernel.args

    let compile = Vulkan_api.Kernel.compile

    let compile_cached = Vulkan_api.Kernel.compile_cached

    let clear_cache = Vulkan_api.Kernel.clear_cache

    let create_args = Vulkan_api.Kernel.create_args

    let set_arg_buffer args idx buf =
      Vulkan_api.Kernel.set_arg_buffer args idx buf

    let set_arg_int32 = Vulkan_api.Kernel.set_arg_int32

    let set_arg_int64 = Vulkan_api.Kernel.set_arg_int64

    let set_arg_float32 = Vulkan_api.Kernel.set_arg_float32

    let set_arg_float64 = Vulkan_api.Kernel.set_arg_float64

    let set_arg_ptr _args _idx _ptr =
      Vulkan_error.raise_error
        (Vulkan_error.feature_not_supported "raw pointer kernel arguments")

    let launch kernel ~args ~grid ~block ~shared_mem ~stream =
      Vulkan_api.Kernel.launch kernel ~args ~grid ~block ~shared_mem ~stream
  end

  let enable_profiling () = ()

  let disable_profiling () = ()
end
