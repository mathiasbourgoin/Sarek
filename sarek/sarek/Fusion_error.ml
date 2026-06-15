(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Fusion_error: Structured error types for Sarek_fusion module
 *
 * Provides typed exceptions for kernel fusion errors.
 ******************************************************************************)

(** {1 Error Types} *)

type t =
  | Empty_pipeline of {function_name : string}
      (** Fusion function called with empty kernel list *)
  | Fusion_incompatible of {
      producer : string;
      consumer : string;
      reason : string;
    }  (** Kernels cannot be fused *)
  | Invalid_fusion of {kernel : string; reason : string}
      (** Malformed fusion result *)

(** {1 Exception} *)

exception Fusion_error of t

(** {1 Utilities} *)

let raise_error err = raise (Fusion_error err)

let error_to_string = function
  | Empty_pipeline {function_name} ->
      Printf.sprintf "%s: cannot fuse empty kernel list" function_name
  | Fusion_incompatible {producer; consumer; reason} ->
      Printf.sprintf "Cannot fuse kernels %s -> %s: %s" producer consumer reason
  | Invalid_fusion {kernel; reason} ->
      Printf.sprintf "Invalid fusion result for %s: %s" kernel reason

let () =
  Printexc.register_printer (function
    | Fusion_error err -> Some (error_to_string err)
    | _ -> None)
