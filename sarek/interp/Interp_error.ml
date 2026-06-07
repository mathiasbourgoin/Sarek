(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Interpreter Error Types
 *
 * Structured error types for the Sarek IR interpreter, replacing string-based
 * failwith calls with typed errors for better error handling and debugging.
 ******************************************************************************)

(** Interpreter-specific errors *)
type error =
  | Unbound_variable of {name : string; context : string}
  | Type_conversion_error of {
      from_type : string;
      to_type : string;
      context : string;
    }
  | Array_bounds_error of {array_name : string; index : int; length : int}
  | Unknown_intrinsic of {name : string}
  | Unknown_function of {name : string}
  | Pattern_match_failure of {context : string}
  | Not_an_array of {expr : string}
  | Not_a_record of {expr : string}
  | Unsupported_operation of {operation : string; reason : string}
  | BSP_deadlock of {message : string}

exception Interpreter_error of error

(** Convert error to human-readable string *)
let error_to_string = function
  | Unbound_variable {name; context} ->
      Printf.sprintf "Unbound variable '%s' in %s" name context
  | Type_conversion_error {from_type; to_type; context} ->
      Printf.sprintf
        "Cannot convert from %s to %s in %s"
        from_type
        to_type
        context
  | Array_bounds_error {array_name; index; length} ->
      Printf.sprintf
        "Array '%s' index %d out of bounds (length %d)"
        array_name
        index
        length
  | Unknown_intrinsic {name} -> Printf.sprintf "Unknown intrinsic: %s" name
  | Unknown_function {name} -> Printf.sprintf "Unknown function: %s" name
  | Pattern_match_failure {context} ->
      Printf.sprintf "Pattern match failure in %s" context
  | Not_an_array {expr} -> Printf.sprintf "Expected array, got: %s" expr
  | Not_a_record {expr} -> Printf.sprintf "Expected record, got: %s" expr
  | Unsupported_operation {operation; reason} ->
      Printf.sprintf "Unsupported operation '%s': %s" operation reason
  | BSP_deadlock {message} -> Printf.sprintf "BSP deadlock: %s" message

(** Raise an interpreter error *)
let raise_error err = raise (Interpreter_error err)
