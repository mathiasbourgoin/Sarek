(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Cross-module Sarek type definitions
 ******************************************************************************)

[@@@warning "-32"]

type float32 = float

type vec2 = {x : float32; y : float32} [@@sarek.type]

let[@sarek.module] add_vec (p : vec2) : float32 = p.x +. p.y

let[@sarek.module] scale_fun (x : float32) : float32 = x *. 2.0
