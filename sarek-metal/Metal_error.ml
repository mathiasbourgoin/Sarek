(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Metal_error - Error Handling for Metal Backend
 *
 * Uses the shared Backend_error module instantiated for Metal backend.
 ******************************************************************************)

include Sarek_backend_error.Backend_error.Make (struct
  let name = "Metal"
end)

(** Re-export Backend_error exception for pattern matching *)
exception Metal_error = Sarek_backend_error.Backend_error.Backend_error
