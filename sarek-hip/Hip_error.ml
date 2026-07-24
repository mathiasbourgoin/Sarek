(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * HIP Error Types - structured error handling for the HIP backend.
 *
 * Uses the shared Backend_error module (same funnel as the CUDA/OpenCL/Vulkan
 * backends) so every handler catches one exception shape across backends.
 ******************************************************************************)

(** Instantiate shared backend error module for HIP *)
include Sarek_backend_error.Backend_error.Make (struct
  let name = "HIP"
end)

(** Backward-compat / convenience alias for the canonical exception. *)
exception Hip_error = Sarek_backend_error.Backend_error.Backend_error
