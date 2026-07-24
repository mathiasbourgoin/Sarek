(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * HIP Plugin - native ROCm/HIP backend (bypasses ZLUDA/PTX).
 *
 * Emits CUDA-C source from Sarek IR (REUSED VERBATIM from the CUDA backend -
 * HIP C++ is source-compatible with the CUDA-C subset Sarek emits:
 * extern "C" __global__, threadIdx/blockIdx, __shared__, __restrict__), then
 * JIT-compiles it with hiprtc to a gfx code object and launches via
 * hipModuleLaunchKernel. Auto-registered on load unless SPOC_DISABLE_GPU/HIP.
 *
 * Advertises CUDA_Source (Execute uses the head of supported_source_langs as
 * the lang tag for the generated source); no #include prelude is needed -
 * hiprtc implicitly provides the HIP runtime for extern "C" __global__ kernels.
 ******************************************************************************)

open Spoc_framework
open Spoc_framework_registry

module Backend : Framework_sig.BACKEND = struct
  include Hip_plugin_base.Hip

  let name = "HIP"

  let execution_model = Framework_sig.JIT

  (* Codegen is reused verbatim from the CUDA-C generator. A located
     Backend_error (Codegen ...) is allowed to PROPAGATE rather than being
     swallowed as [None] (matches the CUDA/C backend, PR #259). *)
  let generate_source ?block:_ ?soa_params:_ (ir : Sarek_ir_types.kernel) :
      string option =
    Some
      (Sarek_codegen.Sarek_ir_cuda.generate_with_types ~types:ir.kern_types ir)

  let execute_direct ~native_fn:_ ~ir:_ ~block:_ ~grid:_ _args =
    Hip_error.raise_error (Hip_error.unsupported_source_lang "direct execution")

  module Intrinsics = Hip_shared.Hip_intrinsics

  let supported_source_langs = [Framework_sig.CUDA_Source]

  let get_current_dev caller =
    match Device.get_current_device () with
    | Some d -> d
    | None -> Hip_error.raise_error (Hip_error.no_device_selected caller)

  let run_source ~source ~lang ~kernel_name ~block ~grid ~shared_mem args =
    match lang with
    | Framework_sig.CUDA_Source ->
        let dev = get_current_dev "run_source:HIP" in
        let compiled = Kernel.compile_cached dev ~name:kernel_name ~source in
        let kargs = Kernel.create_args () in
        Hip_shared.bind_args (Hip_shared.Hip_kargs kargs) kargs args ;
        let stream = Stream.default dev in
        Kernel.launch
          compiled
          ~args:kargs
          ~grid
          ~block
          ~shared_mem
          ~stream:(Some stream)
    | _ ->
        Hip_error.raise_error
          (Hip_error.unsupported_source_lang
             "HIP device only accepts CUDA_Source (HIP C++)")

  let wrap_kargs args = Hip_shared.Hip_kargs args

  let unwrap_kargs = function
    | Hip_shared.Hip_kargs args -> Some args
    | _ -> None
end

let registered_backend =
  lazy
    (Spoc_core.Log.debug
       Spoc_core.Log.Device
       "Hip_plugin: checking availability" ;
     if Backend.is_available () then begin
       Spoc_core.Log.debug
         Spoc_core.Log.Device
         "Hip_plugin: HIP available, registering HIP backend" ;
       Framework_registry.register_backend
         ~priority:80
         (module Backend : Framework_sig.BACKEND)
     end
     else
       Spoc_core.Log.debug Spoc_core.Log.Device "Hip_plugin: HIP not available")

let () = if not (Hip_shared.is_disabled ()) then Lazy.force registered_backend

let register () =
  if not (Hip_shared.is_disabled ()) then Lazy.force registered_backend

let init = register
