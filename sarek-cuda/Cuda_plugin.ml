(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Cuda_plugin — thin backward-compatibility shim.
    The real backend now lives in Cuda_ptx_plugin (auto-registered) and
    Cuda_c_plugin. Shared helpers live in Cuda_shared. *)

(** Force Cuda_ptx_plugin to initialize (it auto-registers on load). *)
let init () = Cuda_ptx_plugin.init ()

let () = init ()

(** Register a custom CUDA intrinsic. *)
let register_intrinsic = Cuda_shared.Cuda_intrinsics.register

(** Look up a CUDA intrinsic. *)
let find_intrinsic = Cuda_shared.Cuda_intrinsics.find

(** Generate CUDA source with custom types. *)
let generate_with_types = Sarek_ir_cuda.generate_with_types

(** Generate CUDA source for a kernel. *)
let generate_source = Sarek_ir_cuda.generate
