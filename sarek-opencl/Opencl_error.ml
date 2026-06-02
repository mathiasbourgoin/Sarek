(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * OpenCL Error Types - Structured Error Handling for OpenCL Backend
 *
 * Uses shared Backend_error module from spoc.framework.
 * Provides same interface pattern as CUDA backend.
 ******************************************************************************)

(** Instantiate shared backend error module for OpenCL *)
include Sarek_backend_error.Backend_error.Make (struct
  let name = "OpenCL"
end)

(** Exception type for OpenCL errors *)
exception Opencl_error = Sarek_backend_error.Backend_error.Backend_error
