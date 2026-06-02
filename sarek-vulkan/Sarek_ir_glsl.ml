(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Re-export the GLSL/Vulkan code generator from the pure [sarek_codegen]
    library. Consumers of [Sarek_vulkan.Sarek_ir_glsl] and in-package
    [Sarek_ir_glsl] are unchanged. *)
include Sarek_codegen.Sarek_ir_glsl
