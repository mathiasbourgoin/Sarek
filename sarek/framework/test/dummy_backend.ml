(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Dummy Backend Plugin for Testing

    A minimal backend implementation that exercises the framework registry and
    intrinsic system without requiring actual GPU hardware.

    This plugin is used purely for testing framework functionality. *)

open Spoc_framework.Framework_sig
open Spoc_framework_registry

(** Dummy backend that implements all required interfaces with no-ops *)
module Dummy_backend : BACKEND = struct
  let name = "Dummy"

  let version = (1, 0, 0)

  let is_available () = true (* Always available for testing *)

  let execution_model = Custom

  let supported_source_langs = []

  let generate_source ?block:_ _ir = None

  let execute_direct ~native_fn:_ ~ir:_ ~block:_ ~grid:_ _args = ()

  let wrap_kargs () = failwith "Dummy backend: wrap_kargs not implemented"

  let unwrap_kargs _ = None

  let run_source ~source:_ ~lang:_ ~kernel_name:_ ~block:_ ~grid:_ ~shared_mem:_
      _ =
    ()

  (** Intrinsics registry - register a few test intrinsics *)
  module Intrinsics =
  Intrinsic_registry.Make ()

  let () =
    (* Register some test intrinsics *)
    Intrinsics.register
      "test_thread_id"
      (Intrinsic_registry.make_divergent_intrinsic
         ~name:"test_thread_id"
         ~codegen:"THREAD_ID") ;
    Intrinsics.register
      "test_barrier"
      (Intrinsic_registry.make_sync_intrinsic
         ~name:"test_barrier"
         ~codegen:"BARRIER") ;
    Intrinsics.register
      "test_add"
      (Intrinsic_registry.make_simple_intrinsic ~name:"test_add" ~codegen:"ADD")

  module Device = struct
    type t = unit

    type id = int

    let init () = ()

    let count () = 1 (* One dummy device *)

    let get _id = ()

    let id () = 0

    let set_current () = ()

    let synchronize () = ()

    let name () = "Dummy Test Device"

    let capabilities () =
      {
        max_threads_per_block = 256;
        max_block_dims = (256, 256, 64);
        max_grid_dims = (65535, 65535, 65535);
        shared_mem_per_block = 16384;
        total_global_mem = 1073741824L;
        (* 1GB *)
        compute_capability = (0, 0);
        supports_fp64 = false;
        supports_atomics = false;
        warp_size = 1;
        max_registers_per_block = 0;
        clock_rate_khz = 0;
        multiprocessor_count = 1;
        is_cpu = true;
      }
  end

  module Stream = struct
    type t = unit

    let create () = ()

    let default () = ()

    let synchronize () = ()

    let destroy () = ()
  end

  module Memory = struct
    type 'a buffer = unit

    let alloc _dev _size _kind = ()

    let alloc_custom _dev ~size:_ ~elem_size:_ = ()

    let alloc_zero_copy _dev _ba _kind = None

    let device_ptr _buf = Nativeint.zero

    let is_zero_copy _buf = false

    let host_to_device ~src:_ ~dst:_ = ()

    let device_to_host ~src:_ ~dst:_ = ()

    let device_to_device ~src:_ ~dst:_ = ()

    let host_ptr_to_device ~src_ptr:_ ~byte_size:_ ~dst:_ = ()

    let device_to_host_ptr ~src:_ ~dst_ptr:_ ~byte_size:_ = ()

    let size _buf = 0

    let free _buf = ()
  end

  module Kernel = struct
    type t = unit

    type args = unit

    let compile _dev ~name:_ ~source:_ = ()

    let compile_cached _dev ~name:_ ~source:_ = ()

    let clear_cache () = ()

    let load_from_ptx ~name:_ ~ptx:_ = ()

    let create_args () = ()

    let set_arg_buffer _args _idx _buf = ()

    let set_arg_int32 _args _idx _v = ()

    let set_arg_int64 _args _idx _v = ()

    let set_arg_float32 _args _idx _v = ()

    let set_arg_float64 _args _idx _v = ()

    let set_arg_ptr _args _idx _ptr = ()

    let launch _kernel ~args:_ ~grid:_ ~block:_ ~shared_mem:_ ~stream:_ = ()
  end

  module Event = struct
    type t = unit

    let create () = ()

    let record _event _stream = ()

    let synchronize _event = ()

    let elapsed ~start:_ ~stop:_ = 0.0

    let destroy _event = ()
  end

  let enable_profiling () = ()

  let disable_profiling () = ()
end

(** Auto-register the dummy backend for testing *)
let () = Framework_registry.register_backend ~priority:1 (module Dummy_backend)
