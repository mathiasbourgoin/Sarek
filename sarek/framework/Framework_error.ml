(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Framework Errors - Structured Error Types for Plugin System
 *
 * Provides structured error types for framework operations including:
 * - Plugin registration and lookup failures
 * - Intrinsic registration and resolution errors
 * - Cache operations failures
 ******************************************************************************)

(** Error types for framework operations *)
type framework_error =
  | Backend_not_found of {name : string}
      (** Backend plugin not found in registry *)
  | No_backends_available of {reason : string}
      (** No backends available (all failed is_available check) *)
  | Backend_unavailable of {name : string; reason : string}
      (** Backend exists but is_available returned false *)
  | Plugin_registration_failed of {name : string; reason : string}
      (** Failed to register plugin *)
  | Intrinsic_not_found of {name : string; backend : string option}
      (** Intrinsic function not found for backend *)
  | Intrinsic_registration_failed of {name : string; reason : string}
      (** Failed to register intrinsic *)
  | Cache_error of {operation : string; reason : string}
      (** Cache operation failed (directory creation, file I/O, etc.) *)

(** Exception wrapper for framework errors *)
exception Framework_error of framework_error

(** Raise a framework error *)
let raise_error err = raise (Framework_error err)

(** Convert error to human-readable string *)
let to_string = function
  | Backend_not_found {name} -> Printf.sprintf "Backend not found: %s" name
  | No_backends_available {reason} ->
      Printf.sprintf "No backends available: %s" reason
  | Backend_unavailable {name; reason} ->
      Printf.sprintf "Backend '%s' unavailable: %s" name reason
  | Plugin_registration_failed {name; reason} ->
      Printf.sprintf "Failed to register plugin '%s': %s" name reason
  | Intrinsic_not_found {name; backend = Some b} ->
      Printf.sprintf "Intrinsic '%s' not found for backend '%s'" name b
  | Intrinsic_not_found {name; backend = None} ->
      Printf.sprintf "Intrinsic '%s' not found in any backend" name
  | Intrinsic_registration_failed {name; reason} ->
      Printf.sprintf "Failed to register intrinsic '%s': %s" name reason
  | Cache_error {operation; reason} ->
      Printf.sprintf "Cache error during %s: %s" operation reason

(** Print error to stderr *)
let print_error err = Printf.eprintf "Framework error: %s\n%!" (to_string err)

(** Get error with optional fallback *)
let with_default ~default f = try f () with Framework_error _ -> default
