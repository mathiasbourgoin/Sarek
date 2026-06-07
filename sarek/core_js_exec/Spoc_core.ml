(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Spoc_core — jsoo Execute shim that mirrors the native Spoc_core interface.
 *
 * When the jsoo Execute target does [open Spoc_core], it opens THIS module
 * and gets the jsoo-side Vector/Device/Transfer/Log/Kernel_arg, without ctypes.
 ******************************************************************************)

module Vector = Vector_jsx
module Device = Device_jsx
module Transfer = Transfer_jsx
module Log = Log_jsx
module Kernel_arg = Kernel_arg_jsx
