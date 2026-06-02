(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Vulkan Error Types - Structured Error Handling for Vulkan Backend
 *
 * Uses shared Backend_error module from spoc.framework.
 * Provides same interface pattern as CUDA/OpenCL backends.
 ******************************************************************************)

(** Instantiate shared backend error module for Vulkan *)
include Sarek_backend_error.Backend_error.Make (struct
  let name = "Vulkan"
end)

(** Exception type for Vulkan errors *)
exception Vulkan_error = Sarek_backend_error.Backend_error.Backend_error
