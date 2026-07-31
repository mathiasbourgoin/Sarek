(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Companion module for test_poly_aggregate_eq: the record type must live in a
   SEPARATE compilation unit. Declaring it beside the kernel makes the PPX
   emit a self-reference to the current unit ("The module M is an alias for
   M__M, which is the current compilation unit"), which would make that test
   red for a reason unrelated to the refusal it exists to pin. *)

type float32 = float

type pt = {x : float32; y : float32} [@@sarek.type]
