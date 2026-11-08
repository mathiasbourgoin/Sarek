(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * CUDA/C Plugin - CUDA C Backend (hidden by default)
 *
 * Emits CUDA C source from Sarek IR; NVRTC JIT-compiles to PTX at runtime.
 * Supports the full IR including records, variants, shared memory, and device
 * function calls — constructs the PTX backend cannot yet emit.
 *
 * NOT auto-registered. Opt in explicitly:
 *   Cuda_c_plugin.register ()
 *
 * Typical use: benchmarking against CUDA/PTX, or kernels that need
 * constructs outside the PTX emitter's current coverage.
 ******************************************************************************)

open Spoc_framework
open Spoc_framework_registry

module Backend : Framework_sig.BACKEND = struct
  include Cuda_plugin_base.Cuda

  let name = "CUDA/C"

  let execution_model = Framework_sig.JIT

  (* [SCoopmat] lowers to nvcuda::wmma ({!Cuda_coopmat}); CUDA/PTX has its own
     lowering and reports the f16 subset of this probe. *)
  module Device = struct
    include Device

    let capabilities (d : t) : Framework_sig.capabilities =
      let caps = capabilities d in
      {
        caps with
        Framework_sig.coopmat =
          Some
            (Cuda_coopmat.support
               ~device_name:(name d)
               ~compute_capability:caps.Framework_sig.compute_capability
               ~warp_size:caps.Framework_sig.warp_size);
      }
  end

  (* Codegen never legitimately declines a kernel by returning [None]: it either
     produces source or raises a located
     [Backend_error (Codegen {backend = "CUDA/C"; _})] naming the offending IR
     node. That error is allowed to PROPAGATE so callers surface the name rather
     than the opaque "generate_source returned None" — catching it here and
     returning [None] hid it identically to the Vulkan swallow (PR #259). *)
  let generate_source ?block ?(soa_params = []) (ir : Sarek_ir_types.kernel) :
      string option =
    (* Refused rather than bound away: this emitter has no SoA lowering, and
       [name] is passed so the message says CUDA/C — the sibling CUDA/PTX
       backend does honour it, and [Cuda_error]'s own backend string is "CUDA"
       for both, which would not distinguish them (backlog-214). *)
    Sarek_backend_error.Backend_error.reject_soa_params ~backend:name soa_params ;
    Some
      (Sarek_ir_cuda.generate_with_types
         ?block:
           (Option.map (fun (d : Framework_sig.dims) -> (d.x, d.y, d.z)) block)
         ~types:ir.kern_types
         ir)

  let execute_direct ~native_fn:_ ~ir:_ ~block:_ ~grid:_ _args =
    Cuda_error.raise_error
      (Cuda_error.unsupported_source_lang "direct execution")

  module Intrinsics = Cuda_shared.Cuda_intrinsics

  let supported_source_langs = [Framework_sig.CUDA_Source]

  let get_current_dev caller =
    match Device.get_current_device () with
    | Some d -> d
    | None -> Cuda_error.raise_error (Cuda_error.no_device_selected caller)

  (* [stream = None] launches on the device's default stream (the historical
     behaviour of [run_source]); [Some s] enqueues on [s]. *)
  let launch_source ~stream ~source ~lang ~kernel_name ~block ~grid ~shared_mem
      args =
    match lang with
    | Framework_sig.CUDA_Source ->
        let dev = get_current_dev "run_source:CUDA" in
        let compiled = Kernel.compile_cached dev ~name:kernel_name ~source in
        let kargs = Kernel.create_args () in
        Cuda_shared.bind_args (Cuda_shared.Cuda_kargs kargs) kargs args ;
        let stream =
          match stream with Some s -> s | None -> Stream.default dev
        in
        Kernel.launch
          compiled
          ~args:kargs
          ~grid
          ~block
          ~shared_mem
          ~stream:(Some stream)
    | _ ->
        Cuda_error.raise_error
          (Cuda_error.unsupported_source_lang
             "CUDA/C device only accepts CUDA_Source")

  let run_source ~source ~lang ~kernel_name ~block ~grid ~shared_mem args =
    launch_source
      ~stream:None
      ~source
      ~lang
      ~kernel_name
      ~block
      ~grid
      ~shared_mem
      args

  let run_source_on_stream ~stream ~source ~lang ~kernel_name ~block ~grid
      ~shared_mem args =
    launch_source
      ~stream:(Some stream)
      ~source
      ~lang
      ~kernel_name
      ~block
      ~grid
      ~shared_mem
      args

  include Framework_sig.Make_refs (struct
    type stream = Stream.t

    type event = Event.t
  end)

  let wrap_kargs args = Cuda_shared.Cuda_kargs args

  let unwrap_kargs = function
    | Cuda_shared.Cuda_kargs args -> Some args
    | _ -> None
end

let registered_backend =
  lazy
    (Spoc_core.Log.debug
       Spoc_core.Log.Device
       "Cuda_c_plugin: checking availability" ;
     if Backend.is_available () then begin
       Spoc_core.Log.debug
         Spoc_core.Log.Device
         "Cuda_c_plugin: CUDA available, registering CUDA/C backend" ;
       Cuda_shared.install_shared_memory_query ~framework:"CUDA/C" ;
       Framework_registry.register_backend
         ~priority:90
         (module Backend : Framework_sig.BACKEND) ;
       Cuda_keepalive.register ()
     end
     else
       Spoc_core.Log.debug
         Spoc_core.Log.Device
         "Cuda_c_plugin: CUDA not available")

let register () =
  if not (Cuda_shared.is_disabled ()) then Lazy.force registered_backend

let init = register
