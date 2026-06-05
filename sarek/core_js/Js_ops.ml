(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Js_ops — browser-targeted CUSTOM_OPS implementation for spoc_core_js.
 *
 * The pure Bigarray host path (numeric scalars) never calls any of these ops.
 * Custom-type and device-transfer operations raise a clear message indicating
 * they will be available once the WebGPU runtime lands.
 *
 * Constraints:
 *   - NO ctypes, NO unix.
 *   - Types are kept minimal (unit aliases) matching Stub_ops conventions.
 ******************************************************************************)

(** Opaque handle for custom storage. Unused on the pure Bigarray path. *)
type handle = unit

(** Opaque device type. WebGPU device will replace this when the runtime lands.
*)
type device_t = unit

(** Opaque device-buffer type. WebGPU buffer will replace this. *)
type device_buf = unit

let _not_supported op =
  failwith
    ("spoc_core_js: " ^ op
   ^ " not supported in the browser yet (custom types / device transfers \
      arrive with the WebGPU runtime)")

let alloc ~elem_size:_ ~length:_ = _not_supported "alloc"

let free _ = _not_supported "free"

let of_raw _ = _not_supported "of_raw"

let to_raw _ = _not_supported "to_raw"

let add_offset _ _ = _not_supported "add_offset"

let copy_elems ~src:_ ~dst:_ ~elem_count:_ ~get:_ ~set:_ =
  _not_supported "copy_elems"

let bigarray_to_handle _ = _not_supported "bigarray_to_handle"

let device_id _ = _not_supported "device_id"
