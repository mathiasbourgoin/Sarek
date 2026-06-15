(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Error handling for Interpreter backend plugin *)

include Spoc_framework.Backend_error.Make (struct
  let name = "Interpreter"
end)
