(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Stub for Metal generator when sarek_metal is unavailable *)

open Sarek_ir_types

let reset_state () = ()

let generate_with_types ~types:_ (k : kernel) =
  ignore k ;
  "(* Metal backend unavailable *)"
