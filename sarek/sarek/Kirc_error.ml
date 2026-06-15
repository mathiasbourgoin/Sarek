(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Kirc_error: Structured error types for Kirc_kernel module
 *
 * Provides typed exceptions for kernel execution and argument conversion errors.
 ******************************************************************************)

(** {1 Error Types} *)

type t =
  | No_native_function of {kernel_name : string; context : string}
      (** Kernel has no native function but native execution was requested *)
  | No_ir of {kernel_name : string}
      (** Native-only kernel has no IR (should use lazy exception) *)
  | Unsupported_arg_type of {
      arg_type : string;
      reason : string;
      context : string;
    }  (** Argument type not supported in native execution *)
  | Type_conversion_failed of {
      from_type : string;
      to_type : string;
      index : int option;
      context : string;
    }  (** Failed to convert between types during argument marshalling *)
  | Backend_not_found of {backend : string}  (** Unknown backend name *)
  | No_source_generation of {backend : string}
      (** Backend does not support source code generation *)
  | Wrong_backend of {expected : string; got : string; operation : string}
      (** Operation only works with specific backend *)

(** {1 Exception} *)

exception Kirc_error of t

(** {1 Utilities} *)

let raise_error err = raise (Kirc_error err)

let error_to_string = function
  | No_native_function {kernel_name; context} ->
      Printf.sprintf
        "Kernel '%s' has no native function (context: %s)"
        kernel_name
        context
  | No_ir {kernel_name} ->
      Printf.sprintf "Native-only kernel '%s' has no IR" kernel_name
  | Unsupported_arg_type {arg_type; reason; context} ->
      Printf.sprintf
        "Unsupported argument type '%s' in %s: %s"
        arg_type
        context
        reason
  | Type_conversion_failed {from_type; to_type; index; context} ->
      let idx_str =
        match index with
        | Some i -> Printf.sprintf " at index %d" i
        | None -> ""
      in
      Printf.sprintf
        "Type conversion failed%s in %s: cannot convert %s to %s"
        idx_str
        context
        from_type
        to_type
  | Backend_not_found {backend} -> Printf.sprintf "Unknown backend: %s" backend
  | No_source_generation {backend} ->
      Printf.sprintf "%s backend does not generate source code" backend
  | Wrong_backend {expected; got; operation} ->
      Printf.sprintf
        "%s only works for %s backend; got %s"
        operation
        expected
        got

let () =
  Printexc.register_printer (function
    | Kirc_error err -> Some (error_to_string err)
    | _ -> None)
