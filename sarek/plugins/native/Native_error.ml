(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Error handling for Native backend plugin *)

include Spoc_framework.Backend_error.Make (struct
  let name = "Native"
end)
