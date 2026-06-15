(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Sarek Test - Source code generation
 *
 * This module generates source code in both old (camlp4) and new (PPX)
 * syntax from neutral kernel definitions.
 ******************************************************************************)

open Test_kernels

(** Generate old camlp4 syntax *)
let to_old_syntax (k : test_kernel) : string =
  let params =
    k.params
    |> List.map (fun p -> Printf.sprintf "(%s : %s)" p.name p.typ)
    |> String.concat " "
  in
  Printf.sprintf "kern %s ->\n%s" params k.body

(** Generate new PPX syntax *)
let to_new_syntax (k : test_kernel) : string =
  let params =
    k.params
    |> List.map (fun p -> Printf.sprintf "(%s : %s)" p.name p.typ)
    |> String.concat " "
  in
  Printf.sprintf "[%%kernel fun %s ->\n%s]" params k.body

(** Generate a complete test file using old syntax *)
let generate_old_test_file (kernels : test_kernel list) : string =
  let kernel_defs =
    List.map
      (fun k -> Printf.sprintf "let %s = %s\n" k.id (to_old_syntax k))
      kernels
  in
  String.concat
    "\n"
    [
      "(* Auto-generated test file for old camlp4 syntax *)";
      "open Spoc";
      "open Kirc";
      "";
    ]
  ^ String.concat "\n" kernel_defs

(** Generate a complete test file using new PPX syntax *)
let generate_new_test_file (kernels : test_kernel list) : string =
  let kernel_defs =
    List.map
      (fun k -> Printf.sprintf "let %s = %s\n" k.id (to_new_syntax k))
      kernels
  in
  String.concat
    "\n"
    [
      "(* Auto-generated test file for new PPX syntax *)";
      "open Spoc";
      "open Kirc";
      "";
    ]
  ^ String.concat "\n" kernel_defs

(** Utility: wrap kernel body with standard opens for evaluation *)
let wrap_for_eval (body : string) : string =
  Printf.sprintf
    {|
    let open Kirc.Std in
    let open Kirc.Math in
    %s
  |}
    body
