(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Stub_ops — FFI-free implementation of Spoc_core_base.CUSTOM_OPS
 *
 * Used only by the ffi_free_gate regression target. All custom-type
 * operations fail with an informative message — the gate test only exercises
 * the pure Bigarray (numeric) path, which never calls any of these.
 ******************************************************************************)

type handle = unit

type device_t = unit

type device_buf = unit

let alloc ~elem_size:_ ~length:_ = failwith "Stub_ops.alloc: not available"

let free _ = failwith "Stub_ops.free: not available"

let of_raw _ = failwith "Stub_ops.of_raw: not available"

let to_raw _ = failwith "Stub_ops.to_raw: not available"

let add_offset _ _ = failwith "Stub_ops.add_offset: not available"

let copy_elems ~src:_ ~dst:_ ~elem_count:_ ~get:_ ~set:_ =
  failwith "Stub_ops.copy_elems: not available"

let bigarray_to_handle _ = failwith "Stub_ops.bigarray_to_handle: not available"

let device_id _ = failwith "Stub_ops.device_id: not available"
