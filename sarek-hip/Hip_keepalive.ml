(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Provider of Spoc_core.Link_keepalive for HIP: an empty kernel and a
   non-blocking stream on the device, and the addresses of the runtime's launch
   and device-selection calls for the native thread. Runs in a short-lived
   domain (see Link_keepalive), so selecting the device here does not touch the
   caller's. *)

open Ctypes

let native (dev : Spoc_core.Device.t) : Spoc_core.Link_keepalive.native =
  let device = Hip_api.Device.get dev.backend_id in
  Hip_api.Device.set_current device ;
  let kernel =
    Hip_api.Kernel.compile_cached
      device
      ~name:Spoc_core.Link_keepalive.kernel_name
      ~source:Spoc_core.Link_keepalive.kernel_source
  in
  let stream = Hip_api.Stream.create_non_blocking device in
  let lib = Hip_bindings.get_hip_lib () in
  let symbol name = Dl.dlsym ~handle:lib ~symbol:name in
  {
    Spoc_core.Link_keepalive.launch_fn = symbol "hipModuleLaunchKernel";
    prep_fn = symbol "hipSetDevice";
    prep_arg = Nativeint.of_int device.Hip_api.Device.id;
    kernel = raw_address_of_ptr (to_voidp kernel.Hip_api.Kernel.function_);
    stream = raw_address_of_ptr (to_voidp stream.Hip_api.Stream.handle);
    release =
      (fun () ->
        Hip_api.Stream.synchronize stream ;
        Hip_api.Stream.destroy stream);
  }

let register () = Spoc_core.Link_keepalive.register_native ~family:"HIP" native
