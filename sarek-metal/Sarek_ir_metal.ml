(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Re-export the Metal code generator from the pure [sarek_codegen] library.
    Consumers of [Sarek_metal.Sarek_ir_metal] and in-package [Sarek_ir_metal]
    are unchanged. *)
include Sarek_codegen.Sarek_ir_metal
