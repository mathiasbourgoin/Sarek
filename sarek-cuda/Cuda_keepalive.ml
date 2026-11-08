(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Provider of Spoc_core.Link_keepalive for CUDA: an empty kernel (NVRTC) and a
   non-blocking stream on the device's context, and the addresses of the
   driver's cuLaunchKernel and cuCtxSetCurrent for the native thread. Runs in a
   short-lived domain (see Link_keepalive), so selecting the context here does
   not touch the caller's. *)

open Ctypes

let native (dev : Spoc_core.Device.t) : Spoc_core.Link_keepalive.native =
  let device = Cuda_api.Device.get dev.backend_id in
  Cuda_api.Device.set_current device ;
  let kernel =
    Cuda_api.Kernel.compile_cached
      device
      ~name:Spoc_core.Link_keepalive.kernel_name
      ~source:Spoc_core.Link_keepalive.kernel_source
  in
  let stream = Cuda_api.Stream.create_non_blocking device in
  let lib = Cuda_bindings.get_cuda_lib () in
  let symbol name = Dl.dlsym ~handle:lib ~symbol:name in
  {
    Spoc_core.Link_keepalive.launch_fn = symbol "cuLaunchKernel";
    prep_fn = symbol "cuCtxSetCurrent";
    prep_arg = raw_address_of_ptr (to_voidp device.Cuda_api.Device.context);
    kernel = raw_address_of_ptr (to_voidp kernel.Cuda_api.Kernel.function_);
    stream = raw_address_of_ptr (to_voidp stream.Cuda_api.Stream.handle);
    release =
      (fun () ->
        Cuda_api.Stream.synchronize stream ;
        Cuda_api.Stream.destroy stream);
  }

let register () = Spoc_core.Link_keepalive.register_native ~family:"CUDA" native
