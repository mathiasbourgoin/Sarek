(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Device_jsx — jsoo device abstraction for the Execute path.
 *
 * Reuses Spoc_framework.Device_type.t as the device type (matches
 * Jsoo_exec_ops.device_t). Device enumeration raises until PR 3 lands.
 ******************************************************************************)

(** Device type — same as the framework's device record *)
type t = Spoc_framework.Device_type.t = {
  id : int;
  backend_id : int;
  name : string;
  framework : string;
  capabilities : Spoc_framework.Framework_sig.capabilities;
}

let _ns op = failwith ("Device_jsx: " ^ op ^ " — no jsoo backend until PR 3")

(** Get device by index — raises until WebGPU backend is wired *)
let get _idx = _ns "get"

(** Get default device — raises until WebGPU backend is wired *)
let get_default () = _ns "get_default"

(** Set current device — raises until WebGPU backend is wired *)
let set_current _d = _ns "set_current"
