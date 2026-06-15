(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Device type - SDK layer
 *
 * Re-exports Framework_sig.device for backward compatibility.
 * New code should use Framework_sig.device directly.
 ******************************************************************************)

(** Device representation - alias to Framework_sig.device *)
type t = Framework_sig.device = {
  id : int;
  backend_id : int;
  name : string;
  framework : string;
  capabilities : Framework_sig.capabilities;
}
