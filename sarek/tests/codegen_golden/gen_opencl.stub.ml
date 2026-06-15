(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Stub for OpenCL generator when sarek_opencl is unavailable *)

open Sarek_ir_types

let reset_state () = ()

let generate_with_types ~types:_ (k : kernel) =
  ignore k ;
  "(* OpenCL backend unavailable *)"
