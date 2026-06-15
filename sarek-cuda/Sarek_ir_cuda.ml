(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Re-export the CUDA code generator from the pure [sarek_codegen] library.
    Consumers of [Sarek_cuda.Sarek_ir_cuda] and in-package [Sarek_ir_cuda] are
    unchanged. *)
include Sarek_codegen.Sarek_ir_cuda
