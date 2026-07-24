(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * CUDA/PTX Plugin - PTX Backend
 *
 * The default CUDA backend. Emits PTX directly from Sarek IR via
 * Sarek_ir_ptx; loads kernels with cuModuleLoadData (no NVRTC).
 * Auto-registered on module load unless SPOC_DISABLE_GPU/CUDA is set.
 *
 * Records, variants and tuples are supported (SROA registers + field-wise
 * global access; see spoc/ir/Sarek_ir_layout.ml). Remaining unsupported IR
 * nodes (e.g. bounded recursion, f64 transcendentals) raise a located
 * [Ptx_codegen_error] from generate_source (propagated, not swallowed).
 ******************************************************************************)

open Spoc_framework
open Spoc_framework_registry

module Backend : Framework_sig.BACKEND = struct
  include Cuda_plugin_base.Cuda

  let name = "CUDA/PTX"

  (* Unlike the C backend, PTX loading needs only the driver API — no NVRTC.
     Keeps this backend usable on ZLUDA and other libnvrtc-less stacks. *)
  let is_available = Cuda_api.is_driver_available

  let execution_model = Framework_sig.JIT

  (* A located [Ptx_codegen_error] (its string message carries the unsupported
     IR node) is allowed to PROPAGATE so callers surface it, rather than being
     converted to [None] and re-raised by Execute as the opaque
     "generate_source returned None" with the detail lost (PR #259). *)
  let generate_source ?block:_ (ir : Sarek_ir_types.kernel) : string option =
    Some (Sarek_ir_ptx.generate_with_types ~types:ir.kern_types ir)

  let execute_direct ~native_fn:_ ~ir:_ ~block:_ ~grid:_ _args =
    Cuda_error.raise_error
      (Cuda_error.unsupported_source_lang "direct execution")

  module Intrinsics = Cuda_shared.Cuda_intrinsics

  let supported_source_langs = [Framework_sig.PTX]

  let get_current_dev caller =
    match Device.get_current_device () with
    | Some d -> d
    | None -> Cuda_error.raise_error (Cuda_error.no_device_selected caller)

  let run_source ~source ~lang ~kernel_name ~block ~grid ~shared_mem args =
    match lang with
    | Framework_sig.PTX ->
        let dev = get_current_dev "run_source:PTX" in
        let compiled = Kernel.load_from_ptx ~name:kernel_name ~ptx:source in
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
             "CUDA/PTX device only accepts PTX source")

  let wrap_kargs args = Cuda_shared.Cuda_kargs args

  let unwrap_kargs = function
    | Cuda_shared.Cuda_kargs args -> Some args
    | _ -> None
end

let registered_backend =
  lazy
    (Spoc_core.Log.debug
       Spoc_core.Log.Device
       "Cuda_ptx_plugin: checking availability" ;
     if Backend.is_available () then begin
       Spoc_core.Log.debug
         Spoc_core.Log.Device
         "Cuda_ptx_plugin: CUDA available, registering CUDA/PTX backend" ;
       Framework_registry.register_backend
         ~priority:100
         (module Backend : Framework_sig.BACKEND)
     end
     else
       Spoc_core.Log.debug
         Spoc_core.Log.Device
         "Cuda_ptx_plugin: CUDA not available")

let () = if not (Cuda_shared.is_disabled ()) then Lazy.force registered_backend

let init () =
  if not (Cuda_shared.is_disabled ()) then Lazy.force registered_backend
