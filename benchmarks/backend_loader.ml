(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Backend loader - conditionally initializes available backends *)
(* This module is selected by dune based on which backends are available *)

let init () =
  (* Native and Interpreter are always available *)
  Sarek_native.Native_plugin.init () ;
  Sarek_interpreter.Interpreter_plugin.init () ;
  (* GPU backends are conditionally included via select *)
  Backend_cuda.init () ;
  Backend_opencl.init () ;
  Backend_vulkan.init () ;
  Backend_metal.init ()
