(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Forwarding alias: Sarek.Sarek_float16 → Sarek_interp.Sarek_float16

    Re-exported under the [Sarek] namespace because PPX-generated NATIVE kernel
    code refers to it by this path (see the f16 conversion arms in
    [sarek/ppx/Sarek_native_intrinsics.ml]): generated code lives in user
    compilation units, which can only be assumed to see [sarek]. *)
include Sarek_interp.Sarek_float16
