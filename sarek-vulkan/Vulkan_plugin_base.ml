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
        (* Both entries come from vkGetPhysicalDeviceFeatures and both are
           mirrored into the logical device's pEnabledFeatures, so this list
           reports what the device will actually ACCEPT, not merely what the
           physical device advertises (#142).

           Float16 joined the list in backlog-62 slice 2, and only because the
           condition its previous absence was justified by no longer holds:
           shaderFloat16 is now PROBED (through the VkPhysicalDeviceFeatures2
           chain) and REQUESTED (in VkDeviceCreateInfo.pNext), so
           [supports_fp16] is a statement about what this logical device
           enabled rather than a guess. This does not lift the GLSL f16
           refusal, which is a Policy/Toolchain_semantic decision at codegen
           and is slice 3's business: [device_features] says what the DEVICE
           provides, not what Sarek is willing to emit. *)
        device_features =
          List.concat
            [
              (if dev.Vulkan_api.Device.supports_fp64 then
                 [Sarek_ir_analysis.Float64]
               else []);
              (if dev.Vulkan_api.Device.supports_int64 then
                 [Sarek_ir_analysis.Int64]
               else []);
              (if dev.Vulkan_api.Device.supports_fp16 then
                 [Sarek_ir_analysis.Float16]
               else []);
            ];
        coopmat = dev.Vulkan_api.Device.coopmat;
        supports_atomics = true;
        (* VkPhysicalDeviceSubgroupProperties.subgroupSize, not a constant.
           This read 32 until backlog-62 slice 2, and 32 is WRONG on the one
           discrete GPU this project measures on: radv / Mesa 26.1.4-arch3.1
           reports 64 for the RX 7900 XTX (RADV NAVI31). It reports 64 for the
           Raphael iGPU too, so the old value was not right for either local
           device. A cooperative-matrix fragment is distributed across exactly
           subgroupSize invocations, so this is ABI, not a statistic. *)
        warp_size = dev.Vulkan_api.Device.subgroup_size;
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

    let host_ptr_to_device ~src_ptr ~byte_size ~dst =
      Vulkan_api.Memory.host_ptr_to_device
        ~src_ptr:(Ctypes.ptr_of_raw_address src_ptr)
        ~byte_size
        ~dst

    let device_to_host_ptr ~src ~dst_ptr ~byte_size =
      Vulkan_api.Memory.device_to_host_ptr
        ~src
        ~dst_ptr:(Ctypes.ptr_of_raw_address dst_ptr)
        ~byte_size

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

    let load_from_ptx ~name:_ ~ptx:_ =
      failwith "PTX kernels not supported by Vulkan backend"

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
