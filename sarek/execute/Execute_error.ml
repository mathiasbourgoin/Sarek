(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Execute_error - Structured errors for kernel execution
 *
 * Replaces string-based failwith calls with typed errors for better
 * error handling and debugging.
 ******************************************************************************)

(** Execution error type *)
type error =
  | Unbound_variable of string
  | Type_mismatch of {expected : string; actual : string; context : string}
  | Unsupported_argument of {arg_type : string; context : string}
  | Backend_error of {backend : string; message : string}
  | Compilation_failed of {kernel : string; reason : string}
  | Invalid_dimensions of {grid : string; block : string; reason : string}
  | Missing_ir of {kernel : string}
  | Missing_native_fn of {kernel : string}
  | Transfer_failed of {vector : string; reason : string}
  | Interp_error of string
  | Invalid_file of {path : string; reason : string}
  | Type_helper_not_found of {type_name : string; context : string}

(** Exception wrapper for execution errors *)
exception Execution_error of error

(** Convert error to human-readable string *)
let error_to_string = function
  | Unbound_variable name -> Printf.sprintf "Unbound variable: %s" name
  | Type_mismatch {expected; actual; context} ->
      Printf.sprintf
        "Type mismatch in %s: expected %s, got %s"
        context
        expected
        actual
  | Unsupported_argument {arg_type; context} ->
      Printf.sprintf "Unsupported argument type %s in %s" arg_type context
  | Backend_error {backend; message} ->
      Printf.sprintf "Backend error (%s): %s" backend message
  | Compilation_failed {kernel; reason} ->
      Printf.sprintf "Kernel compilation failed for %s: %s" kernel reason
  | Invalid_dimensions {grid; block; reason} ->
      Printf.sprintf
        "Invalid dimensions (grid=%s, block=%s): %s"
        grid
        block
        reason
  | Missing_ir {kernel} ->
      Printf.sprintf "Missing IR for kernel %s (JIT backend requires IR)" kernel
  | Missing_native_fn {kernel} ->
      Printf.sprintf
        "Missing native function for kernel %s (Direct backend requires \
         function)"
        kernel
  | Transfer_failed {vector; reason} ->
      Printf.sprintf "Vector transfer failed for %s: %s" vector reason
  | Interp_error msg -> Printf.sprintf "Interpreter error: %s" msg
  | Invalid_file {path; reason} ->
      Printf.sprintf "Invalid file %s: %s" path reason
  | Type_helper_not_found {type_name; context} ->
      Printf.sprintf
        "No type helper found for '%s' in %s. Did you forget [@@sarek.type]?"
        type_name
        context

(** Raise an execution error *)
let raise_error err = raise (Execution_error err)

(** Convert exception to string for logging *)
let () =
  Printexc.register_printer (function
    | Execution_error err -> Some (error_to_string err)
    | _ -> None)
