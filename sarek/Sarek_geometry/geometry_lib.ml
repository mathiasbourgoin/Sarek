(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

[@@@warning "-32"]

type float32 = float

type point = {x : float32; y : float32} [@@sarek.type]

let[@sarek.module] distance (p1 : point) (p2 : point) : float32 =
  let dx = p1.x -. p2.x in
  let dy = p1.y -. p2.y in
  sqrt ((dx *. dx) +. (dy *. dy))
