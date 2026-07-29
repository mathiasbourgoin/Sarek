(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** A BACKEND whose [is_available] probe is programmable, for
    test_registry_probe_policy.

    Everything except [is_available] is the same inert stub as {!Dummy_backend}
    \- the registry's probe policy is the only thing under test, so nothing else
    here is ever called. It does NOT self-register: the test decides when the
    backend enters the registry, because the registry is process-global and an
    auto-registration would leak into the "no backends registered" baseline. *)

open Spoc_framework.Framework_sig
open Spoc_framework_registry

(** What the next probe does. [Ordinary] stands for any run-of-the-mill probe
    failure (a missing driver, a failed dlopen); [Fatal e] for the
    asynchronous/fatal class the registry must not swallow. *)
type mode = Ok | Unavailable | Ordinary | Fatal of exn

let mode = ref Ok

let probe_calls = ref 0

let backend_name = "ProbeBackend"

module Probe_backend : BACKEND = struct
  let name = backend_name

  let version = (1, 0, 0)

  let is_available () =
    incr probe_calls ;
    match !mode with
    | Ok -> true
    | Unavailable -> false
    | Ordinary -> raise Not_found
    | Fatal e -> raise e

  let execution_model = Custom

  let supported_source_langs = []

  let generate_source ?block:_ ?soa_params:_ _ir = None

  let execute_direct ~native_fn:_ ~ir:_ ~block:_ ~grid:_ _args = ()

  let wrap_kargs () = failwith "Probe backend: wrap_kargs not implemented"

  let unwrap_kargs _ = None

  let run_source ~source:_ ~lang:_ ~kernel_name:_ ~block:_ ~grid:_ ~shared_mem:_
      _ =
    ()

  module Intrinsics = Intrinsic_registry.Make ()

  module Device = struct
    type t = unit

    type id = int

    let init () = ()

    let count () = 0

    let get _id = ()

    let id () = 0

    let set_current () = ()

    let synchronize () = ()

    let name () = "Probe Test Device"

    let capabilities () =
      {
        max_threads_per_block = 1;
        max_block_dims = (1, 1, 1);
        max_grid_dims = (1, 1, 1);
        shared_mem_per_block = 0;
        total_global_mem = 0L;
        compute_capability = (0, 0);
        device_features = [];
        coopmat = None;
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
