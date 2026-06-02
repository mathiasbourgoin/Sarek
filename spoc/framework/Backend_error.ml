(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Backend Error Types - re-export from sarek_backend_error
 *
 * Kept here for backward compatibility with Spoc_framework.Backend_error.
 * New code should depend on sarek_backend_error directly (ctypes-free).
 ******************************************************************************)

include Sarek_backend_error.Backend_error
