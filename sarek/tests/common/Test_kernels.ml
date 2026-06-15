(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Sarek Test - Test kernel definitions
 *
 * This module defines test kernels in a neutral format that can be converted
 * to both old (camlp4) and new (PPX) syntax for comparison testing.
 ******************************************************************************)

(** A parameter definition *)
type param = {
  name : string;
  typ : string;  (** Type as string, e.g., "float32 vector" *)
}

(** A test kernel definition *)
type test_kernel = {
  id : string;
  description : string;
  params : param list;
  body : string;
  expected_return_type : string option;
  tags : string list;
      (** Categories: "basic", "control_flow", "custom_types", etc. *)
}

(** Basic kernels *)
let vec_add =
  {
    id = "vec_add";
    description = "Basic vector addition";
    params =
      [
        {name = "a"; typ = "float32 vector"};
        {name = "b"; typ = "float32 vector"};
        {name = "c"; typ = "float32 vector"};
      ];
    body =
      {|
    let open Std in
    let idx = global_thread_id in
    c.[idx] <- a.[idx] +. b.[idx]
  |};
    expected_return_type = Some "unit";
    tags = ["basic"; "vector"];
  }

let vec_scale =
  {
    id = "vec_scale";
    description = "Vector scalar multiplication";
    params =
      [
        {name = "a"; typ = "float32 vector"};
        {name = "k"; typ = "float32"};
        {name = "b"; typ = "float32 vector"};
      ];
    body =
      {|
    let open Std in
    let idx = global_thread_id in
    b.[idx] <- a.[idx] *. k
  |};
    expected_return_type = Some "unit";
    tags = ["basic"; "vector"; "scalar"];
  }

let int_add =
  {
    id = "int_add";
    description = "Integer vector addition";
    params =
      [
        {name = "a"; typ = "int32 vector"};
        {name = "b"; typ = "int32 vector"};
        {name = "c"; typ = "int32 vector"};
      ];
    body =
      {|
    let open Std in
    let idx = global_thread_id in
    c.[idx] <- a.[idx] + b.[idx]
  |};
    expected_return_type = Some "unit";
    tags = ["basic"; "vector"; "int"];
  }

(** Control flow kernels *)
let if_simple =
  {
    id = "if_simple";
    description = "Simple if-then-else";
    params =
      [{name = "a"; typ = "int32 vector"}; {name = "b"; typ = "int32 vector"}];
    body =
      {|
    let open Std in
    let idx = global_thread_id in
    if a.[idx] > 0l then
      b.[idx] <- 1l
    else
      b.[idx] <- 0l
  |};
    expected_return_type = Some "unit";
    tags = ["control_flow"; "if"];
  }

let for_simple =
  {
    id = "for_simple";
    description = "Simple for loop";
    params = [{name = "a"; typ = "float32 vector"}; {name = "n"; typ = "int32"}];
    body =
      {|
    let open Std in
    let idx = global_thread_id in
    for i = 0l to n do
      a.[idx] <- a.[idx] +. 1.0
    done
  |};
    expected_return_type = Some "unit";
    tags = ["control_flow"; "for"];
  }

let while_simple =
  {
    id = "while_simple";
    description = "Simple while loop";
    params =
      [{name = "a"; typ = "int32 vector"}; {name = "limit"; typ = "int32"}];
    body =
      {|
    let open Std in
    let idx = global_thread_id in
    let count = 0l in
    while count < limit do
      a.[idx] <- a.[idx] + 1l
    done
  |};
    expected_return_type = Some "unit";
    tags = ["control_flow"; "while"];
  }

(** Intrinsic kernels *)
let thread_indices =
  {
    id = "thread_indices";
    description = "Using thread indices";
    params = [{name = "out"; typ = "int32 vector"}];
    body =
      {|
    let open Std in
    let idx = thread_idx_x + (block_idx_x * block_dim_x) in
    out.[idx] <- idx
  |};
    expected_return_type = Some "unit";
    tags = ["intrinsics"; "basic"];
  }

let math_float32 =
  {
    id = "math_float32";
    description = "Float32 math functions";
    params =
      [
        {name = "a"; typ = "float32 vector"};
        {name = "b"; typ = "float32 vector"};
      ];
    body =
      {|
    let open Std in
    let open Math.Float32 in
    let idx = global_thread_id in
    b.[idx] <- sin a.[idx]
  |};
    expected_return_type = Some "unit";
    tags = ["intrinsics"; "math"];
  }

(** All test kernels *)
let all_kernels =
  [
    vec_add;
    vec_scale;
    int_add;
    if_simple;
    for_simple;
    while_simple;
    thread_indices;
    math_float32;
  ]

(** Filter kernels by tag *)
let kernels_by_tag tag = List.filter (fun k -> List.mem tag k.tags) all_kernels

(** Get kernel by ID *)
let find_kernel id = List.find_opt (fun k -> k.id = id) all_kernels
