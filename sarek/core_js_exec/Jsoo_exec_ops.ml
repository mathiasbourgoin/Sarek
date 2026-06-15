(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Jsoo_exec_ops — CUSTOM_OPS for the jsoo Execute path.
 *
 * The key difference from Js_ops: device_t = Spoc_framework.Device_type.t
 * (not unit). This matches the type Execute uses in its location match.
 *
 * All operations that touch custom storage or device transfers raise, because
 * the pure Bigarray (scalar) path is what Execute exercises on jsoo.
 *
 * No ctypes, no unix.
 ******************************************************************************)

let _ns op = failwith ("spoc_core_js_exec: " ^ op ^ " not yet implemented")

(** Opaque handle for custom storage (unused on pure Bigarray path) *)
type handle = unit

(** The real Device_type.t — not unit, so Execute's location match compiles *)
type device_t = Spoc_framework.Device_type.t

(** Device buffer type: first-class module satisfying Memory_jsx.BUFFER *)
type device_buf = (module Memory_jsx.BUFFER)

let alloc ~elem_size:_ ~length:_ = _ns "alloc"

let free _ = _ns "free"

let of_raw _ = _ns "of_raw"

let to_raw _ = _ns "to_raw"

let add_offset _ _ = _ns "add_offset"

let copy_elems ~src:_ ~dst:_ ~elem_count:_ ~get:_ ~set:_ = _ns "copy_elems"

let bigarray_to_handle _ = _ns "bigarray_to_handle"

let device_id (d : device_t) = d.Spoc_framework.Device_type.id

let custom_to_bytes ~set:_ ~elem_size:_ _ = _ns "custom_to_bytes"

let custom_of_bytes ~get:_ ~elem_size:_ _ = _ns "custom_of_bytes"
