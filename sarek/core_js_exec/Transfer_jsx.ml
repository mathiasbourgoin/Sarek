(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Transfer_jsx — jsoo transfer stubs for the Execute path.
 *
 * Execute calls Transfer.to_device and Transfer.to_cpu. The device branch
 * raises (unreachable in the empty-registry path); to_cpu is a no-op for
 * CPU-resident vectors.
 ******************************************************************************)

let _ns op =
  failwith ("Transfer_jsx: " ^ op ^ " — no jsoo backend device path until PR 3")

(** Transfer vector to device — raises (no jsoo device backend yet) *)
let to_device (type a b) (_vec : (a, b) Vector_jsx.t) (_dev : Device_jsx.t) :
    unit =
  _ns "to_device"

(** Transfer vector from device to CPU — no-op for CPU vectors *)
let to_cpu (type a b) (vec : (a, b) Vector_jsx.t) : unit =
  match vec.Vector_jsx.location with
  | Vector_jsx.CPU | Vector_jsx.Stale_GPU _ -> ()
  | Vector_jsx.GPU _ | Vector_jsx.Stale_CPU _ | Vector_jsx.Both _ ->
      _ns "to_cpu (device path)"
