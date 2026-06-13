(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Re-export the PTX code generator from the pure [sarek_codegen] library.
    Consumers of [Sarek_cuda.Sarek_ir_ptx] and in-package [Sarek_ir_ptx] are
    unchanged. *)
include Sarek_codegen.Sarek_ir_ptx
