(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Vector types — native instantiation of spoc_core_base functor
 *
 * Re-exports all types and helpers from Spoc_core_base.Make(Ctypes_ops)
 * so that the rest of spoc_core sees an identical API to the pre-refactor
 * Vector_types module. Custom_helpers (ctypes pointer arithmetic) is defined
 * here since it needs the locally-resolved custom_type.
 ******************************************************************************)

include Spoc_core_base.Make (Ctypes_ops)

(** Re-export DEVICE_BUFFER module type for backward compatibility *)
module type DEVICE_BUFFER = Memory.BUFFER

(** Device buffer type alias *)
type device_buffer = (module DEVICE_BUFFER)

(** Helper functions for custom type implementations. These wrap Ctypes
    operations to provide simpler APIs for PPX-generated code.

    All operations are non-allocating pointer arithmetic — no memory is
    allocated or released. *)
module Custom_helpers = struct
  let read_float32 (ptr : unit Ctypes.ptr) (byte_offset : int) : float =
    let byte_ptr = Ctypes.from_voidp Ctypes.uint8_t ptr in
    let target_ptr = Ctypes.(byte_ptr +@ byte_offset) in
    Ctypes.(!@(Ctypes.from_voidp Ctypes.float (Ctypes.to_voidp target_ptr)))

  let write_float32 (ptr : unit Ctypes.ptr) (byte_offset : int) (v : float) :
      unit =
    let byte_ptr = Ctypes.from_voidp Ctypes.uint8_t ptr in
    let target_ptr = Ctypes.(byte_ptr +@ byte_offset) in
    Ctypes.(Ctypes.from_voidp Ctypes.float (Ctypes.to_voidp target_ptr) <-@ v)

  let read_int32 (ptr : unit Ctypes.ptr) (byte_offset : int) : int32 =
    let byte_ptr = Ctypes.from_voidp Ctypes.uint8_t ptr in
    let target_ptr = Ctypes.(byte_ptr +@ byte_offset) in
    Ctypes.(!@(Ctypes.from_voidp Ctypes.int32_t (Ctypes.to_voidp target_ptr)))

  let write_int32 (ptr : unit Ctypes.ptr) (byte_offset : int) (v : int32) : unit
      =
    let byte_ptr = Ctypes.from_voidp Ctypes.uint8_t ptr in
    let target_ptr = Ctypes.(byte_ptr +@ byte_offset) in
    Ctypes.(Ctypes.from_voidp Ctypes.int32_t (Ctypes.to_voidp target_ptr) <-@ v)

  let read_int64 (ptr : unit Ctypes.ptr) (byte_offset : int) : int64 =
    let byte_ptr = Ctypes.from_voidp Ctypes.uint8_t ptr in
    let target_ptr = Ctypes.(byte_ptr +@ byte_offset) in
    Ctypes.(!@(Ctypes.from_voidp Ctypes.int64_t (Ctypes.to_voidp target_ptr)))

  let write_int64 (ptr : unit Ctypes.ptr) (byte_offset : int) (v : int64) : unit
      =
    let byte_ptr = Ctypes.from_voidp Ctypes.uint8_t ptr in
    let target_ptr = Ctypes.(byte_ptr +@ byte_offset) in
    Ctypes.(Ctypes.from_voidp Ctypes.int64_t (Ctypes.to_voidp target_ptr) <-@ v)

  let read_float64 (ptr : unit Ctypes.ptr) (byte_offset : int) : float =
    let byte_ptr = Ctypes.from_voidp Ctypes.uint8_t ptr in
    let target_ptr = Ctypes.(byte_ptr +@ byte_offset) in
    Ctypes.(!@(Ctypes.from_voidp Ctypes.double (Ctypes.to_voidp target_ptr)))

  let write_float64 (ptr : unit Ctypes.ptr) (byte_offset : int) (v : float) :
      unit =
    let byte_ptr = Ctypes.from_voidp Ctypes.uint8_t ptr in
    let target_ptr = Ctypes.(byte_ptr +@ byte_offset) in
    Ctypes.(Ctypes.from_voidp Ctypes.double (Ctypes.to_voidp target_ptr) <-@ v)

  let read_int (ptr : unit Ctypes.ptr) (byte_offset : int) : int =
    Int32.to_int (read_int32 ptr byte_offset)

  let write_int (ptr : unit Ctypes.ptr) (byte_offset : int) (v : int) : unit =
    write_int32 ptr byte_offset (Int32.of_int v)

  let read_custom (custom : 'a custom_type) (ptr : unit Ctypes.ptr)
      (byte_offset : int) : 'a =
    let byte_ptr = Ctypes.from_voidp Ctypes.uint8_t ptr in
    let target_ptr = Ctypes.to_voidp Ctypes.(byte_ptr +@ byte_offset) in
    custom.get target_ptr 0

  let write_custom (custom : 'a custom_type) (ptr : unit Ctypes.ptr)
      (byte_offset : int) (v : 'a) : unit =
    let byte_ptr = Ctypes.from_voidp Ctypes.uint8_t ptr in
    let target_ptr = Ctypes.to_voidp Ctypes.(byte_ptr +@ byte_offset) in
    custom.set target_ptr 0 v
end
