(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * CUDA Error Types - Structured Error Handling for CUDA Backend
 *
 * Now uses shared Backend_error module from spoc.framework.
 * Provides same interface as before for backward compatibility.
 ******************************************************************************)

(** Instantiate shared backend error module for CUDA *)
include Sarek_backend_error.Backend_error.Make (struct
  let name = "CUDA"
end)

(** Backward compatibility: re-export exception type *)
exception Cuda_error = Sarek_backend_error.Backend_error.Backend_error

(** Backward compatibility: module_load_failed used 'ptx_size' parameter name *)
let module_load_failed ptx_size reason = module_load_failed ptx_size reason
