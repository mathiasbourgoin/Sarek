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

  let generate_source ?block:_ (ir : Sarek_ir_types.kernel) : string option =
    try Some (Sarek_ir_cuda.generate_with_types ~types:ir.kern_types ir)
    with _ -> None

  let execute_direct ~native_fn:_ ~ir:_ ~block:_ ~grid:_ _args =
    Cuda_error.raise_error
      (Cuda_error.unsupported_source_lang "direct execution")

  module Intrinsics = Cuda_shared.Cuda_intrinsics

  let supported_source_langs = [Framework_sig.CUDA_Source]

  let get_current_dev caller =
    match Device.get_current_device () with
    | Some d -> d
    | None -> Cuda_error.raise_error (Cuda_error.no_device_selected caller)

  let run_source ~source ~lang ~kernel_name ~block ~grid ~shared_mem args =
    match lang with
    | Framework_sig.CUDA_Source ->
        let dev = get_current_dev "run_source:CUDA" in
        let compiled = Kernel.compile_cached dev ~name:kernel_name ~source in
        let kargs = Kernel.create_args () in
        Cuda_shared.bind_args (Cuda_shared.Cuda_kargs kargs) kargs args ;
        let stream = Stream.default dev in
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
       Framework_registry.register_backend
         ~priority:90
         (module Backend : Framework_sig.BACKEND)
     end
     else
       Spoc_core.Log.debug
         Spoc_core.Log.Device
         "Cuda_c_plugin: CUDA not available")

let register () =
  if not (Cuda_shared.is_disabled ()) then Lazy.force registered_backend

let init = register
