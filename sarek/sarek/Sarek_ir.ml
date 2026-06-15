(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Module alias for backward compatibility.

    Sarek_ir module was removed (it contained dead conversion functions). This
    alias redirects to Sarek_ir_types which contains the actual IR types. *)

include Sarek_ir_types
