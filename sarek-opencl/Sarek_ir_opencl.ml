(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Re-export the OpenCL code generator from the pure [sarek_codegen] library.
    Consumers of [Sarek_opencl.Sarek_ir_opencl] and in-package [Sarek_ir_opencl]
    are unchanged. *)
include Sarek_codegen.Sarek_ir_opencl
