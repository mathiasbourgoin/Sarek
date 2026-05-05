(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Vector types and helpers (split from Vector.ml)
 *
 * This module holds type definitions and helper functions shared by Vector.ml
 * and other runtime modules. Vector.ml includes this module to re-export the
 * public API.
 ******************************************************************************)

(** {1 Element Types} *)

(** Standard numeric kinds backed by Bigarray *)
type (_, _) scalar_kind =
  | Float32 : (float, Bigarray.float32_elt) scalar_kind
  | Float64 : (float, Bigarray.float64_elt) scalar_kind
  | Int32 : (int32, Bigarray.int32_elt) scalar_kind
  | Int64 : (int64, Bigarray.int64_elt) scalar_kind
  | Char : (char, Bigarray.int8_unsigned_elt) scalar_kind
  | Complex32 : (Complex.t, Bigarray.complex32_elt) scalar_kind

(** {1 Location Tracking} *)

(** Where the authoritative copy of data resides *)
type location =
  | CPU  (** Data only on host *)
  | GPU of Device.t  (** Data only on specific device *)
  | Both of Device.t  (** Synced on host and device *)
  | Stale_CPU of Device.t  (** GPU is authoritative, CPU outdated *)
  | Stale_GPU of Device.t  (** CPU is authoritative, GPU outdated *)

(** {1 Device Buffer Abstraction} *)

(** Device buffer reuses Memory.BUFFER module type. We use the raw module type
    here (not the phantom-typed Memory.buffer) because the hashtable stores
    buffers for a single vector type, and the type safety comes from the
    Vector's ('a, 'b) t type parameter. *)
module type DEVICE_BUFFER = Memory.BUFFER

type device_buffer = (module DEVICE_BUFFER)

(** Device buffer storage - maps device ID to buffer *)
type device_buffers = (int, device_buffer) Hashtbl.t

(** Custom type descriptor for ctypes-based structures *)
type 'a custom_type = {
  elem_size : int;  (** Size of each element in bytes *)
  type_id : 'a Sarek_ir_types.Type_id.t;  (** Runtime element type identity *)
  vector_type_id : ('a, unit) t Sarek_ir_types.Type_id.t;
      (** Runtime identity for the vector type carrying this element type *)
  get : unit Ctypes.ptr -> int -> 'a;  (** Read element at index *)
  set : unit Ctypes.ptr -> int -> 'a -> unit;  (** Write element at index *)
  name : string;  (** Type name for debugging *)
}

(** Unified kind type supporting both scalar and custom types *)
and (_, _) kind =
  | Scalar : ('a, 'b) scalar_kind -> ('a, 'b) kind
  | Custom : 'a custom_type -> ('a, unit) kind

and (_, _) host_storage =
  | Bigarray_storage :
      ('a, 'b, Bigarray.c_layout) Bigarray.Array1.t
      -> ('a, 'b) host_storage
  | Custom_storage : {
      ptr : unit Ctypes.ptr;
      custom : 'a custom_type;
      length : int;
    }
      -> ('a, unit) host_storage

(** High-level vector with location tracking *)
and ('a, 'b) t = {
  host : ('a, 'b) host_storage;
  device_buffers : device_buffers;
  length : int;
  kind : ('a, 'b) kind;
  mutable location : location;
  mutable auto_sync : bool;  (** Enable automatic CPU sync on get *)
  id : int;  (** Unique vector ID for debugging *)
}

(** Helper functions for custom type implementations. These wrap Ctypes
    operations to provide simpler APIs for PPX-generated code. *)
module Custom_helpers = struct
  (** Read a float32 value at byte offset from a void pointer *)
  let read_float32 (ptr : unit Ctypes.ptr) (byte_offset : int) : float =
    let byte_ptr = Ctypes.from_voidp Ctypes.uint8_t ptr in
    let target_ptr = Ctypes.(byte_ptr +@ byte_offset) in
    let float_ptr =
      Ctypes.from_voidp Ctypes.float (Ctypes.to_voidp target_ptr)
    in
    Ctypes.(!@float_ptr)

  (** Write a float32 value at byte offset to a void pointer *)
  let write_float32 (ptr : unit Ctypes.ptr) (byte_offset : int) (v : float) :
      unit =
    let byte_ptr = Ctypes.from_voidp Ctypes.uint8_t ptr in
    let target_ptr = Ctypes.(byte_ptr +@ byte_offset) in
    let float_ptr =
      Ctypes.from_voidp Ctypes.float (Ctypes.to_voidp target_ptr)
    in
    Ctypes.(float_ptr <-@ v)

  (** Read an int32 value at byte offset from a void pointer *)
  let read_int32 (ptr : unit Ctypes.ptr) (byte_offset : int) : int32 =
    let byte_ptr = Ctypes.from_voidp Ctypes.uint8_t ptr in
    let target_ptr = Ctypes.(byte_ptr +@ byte_offset) in
    let int_ptr =
      Ctypes.from_voidp Ctypes.int32_t (Ctypes.to_voidp target_ptr)
    in
    Ctypes.(!@int_ptr)

  (** Write an int32 value at byte offset to a void pointer *)
  let write_int32 (ptr : unit Ctypes.ptr) (byte_offset : int) (v : int32) : unit
      =
    let byte_ptr = Ctypes.from_voidp Ctypes.uint8_t ptr in
    let target_ptr = Ctypes.(byte_ptr +@ byte_offset) in
    let int_ptr =
      Ctypes.from_voidp Ctypes.int32_t (Ctypes.to_voidp target_ptr)
    in
    Ctypes.(int_ptr <-@ v)

  (** Read an int64 value at byte offset from a void pointer *)
  let read_int64 (ptr : unit Ctypes.ptr) (byte_offset : int) : int64 =
    let byte_ptr = Ctypes.from_voidp Ctypes.uint8_t ptr in
    let target_ptr = Ctypes.(byte_ptr +@ byte_offset) in
    let int_ptr =
      Ctypes.from_voidp Ctypes.int64_t (Ctypes.to_voidp target_ptr)
    in
    Ctypes.(!@int_ptr)

  (** Write an int64 value at byte offset to a void pointer *)
  let write_int64 (ptr : unit Ctypes.ptr) (byte_offset : int) (v : int64) : unit
      =
    let byte_ptr = Ctypes.from_voidp Ctypes.uint8_t ptr in
    let target_ptr = Ctypes.(byte_ptr +@ byte_offset) in
    let int_ptr =
      Ctypes.from_voidp Ctypes.int64_t (Ctypes.to_voidp target_ptr)
    in
    Ctypes.(int_ptr <-@ v)

  (** Read a float64 (double) value at byte offset from a void pointer *)
  let read_float64 (ptr : unit Ctypes.ptr) (byte_offset : int) : float =
    let byte_ptr = Ctypes.from_voidp Ctypes.uint8_t ptr in
    let target_ptr = Ctypes.(byte_ptr +@ byte_offset) in
    let float_ptr =
      Ctypes.from_voidp Ctypes.double (Ctypes.to_voidp target_ptr)
    in
    Ctypes.(!@float_ptr)

  (** Write a float64 (double) value at byte offset to a void pointer *)
  let write_float64 (ptr : unit Ctypes.ptr) (byte_offset : int) (v : float) :
      unit =
    let byte_ptr = Ctypes.from_voidp Ctypes.uint8_t ptr in
    let target_ptr = Ctypes.(byte_ptr +@ byte_offset) in
    let float_ptr =
      Ctypes.from_voidp Ctypes.double (Ctypes.to_voidp target_ptr)
    in
    Ctypes.(float_ptr <-@ v)

  (** Read an int value (stored as 4-byte int32) at byte offset from a void
      pointer *)
  let read_int (ptr : unit Ctypes.ptr) (byte_offset : int) : int =
    Int32.to_int (read_int32 ptr byte_offset)

  (** Write an int value (stored as 4-byte int32) at byte offset to a void
      pointer *)
  let write_int (ptr : unit Ctypes.ptr) (byte_offset : int) (v : int) : unit =
    write_int32 ptr byte_offset (Int32.of_int v)

  (** Read a nested custom type at byte offset. This allows composable custom
      types where one record contains another. *)
  let read_custom (custom : 'a custom_type) (ptr : unit Ctypes.ptr)
      (byte_offset : int) : 'a =
    let byte_ptr = Ctypes.from_voidp Ctypes.uint8_t ptr in
    let target_ptr = Ctypes.to_voidp Ctypes.(byte_ptr +@ byte_offset) in
    custom.get target_ptr 0

  (** Write a nested custom type at byte offset. *)
  let write_custom (custom : 'a custom_type) (ptr : unit Ctypes.ptr)
      (byte_offset : int) (v : 'a) : unit =
    let byte_ptr = Ctypes.from_voidp Ctypes.uint8_t ptr in
    let target_ptr = Ctypes.to_voidp Ctypes.(byte_ptr +@ byte_offset) in
    custom.set target_ptr 0 v
end

(** {1 Kind Helpers} *)

(** Convert scalar kind to Bigarray.kind *)
let to_bigarray_kind : type a b. (a, b) scalar_kind -> (a, b) Bigarray.kind =
  function
  | Float32 -> Bigarray.Float32
  | Float64 -> Bigarray.Float64
  | Int32 -> Bigarray.Int32
  | Int64 -> Bigarray.Int64
  | Char -> Bigarray.Char
  | Complex32 -> Bigarray.Complex32

(** Element size in bytes *)
let scalar_elem_size : type a b. (a, b) scalar_kind -> int = function
  | Float32 -> 4
  | Float64 -> 8
  | Int32 -> 4
  | Int64 -> 8
  | Char -> 1
  | Complex32 -> 8

let elem_size : type a b. (a, b) kind -> int = function
  | Scalar k -> scalar_elem_size k
  | Custom c -> c.elem_size

(** Kind name for debugging *)
let scalar_kind_name : type a b. (a, b) scalar_kind -> string = function
  | Float32 -> "Float32"
  | Float64 -> "Float64"
  | Int32 -> "Int32"
  | Int64 -> "Int64"
  | Char -> "Char"
  | Complex32 -> "Complex32"

let kind_name : type a b. (a, b) kind -> string = function
  | Scalar k -> scalar_kind_name k
  | Custom c -> "Custom(" ^ c.name ^ ")"

let float32_type_id : float Sarek_ir_types.Type_id.t =
  Sarek_ir_types.Type_id.create ()

let float64_type_id : float Sarek_ir_types.Type_id.t =
  Sarek_ir_types.Type_id.create ()

let int32_type_id : int32 Sarek_ir_types.Type_id.t =
  Sarek_ir_types.Type_id.create ()

let int64_type_id : int64 Sarek_ir_types.Type_id.t =
  Sarek_ir_types.Type_id.create ()

let char_type_id : char Sarek_ir_types.Type_id.t =
  Sarek_ir_types.Type_id.create ()

let complex32_type_id : Complex.t Sarek_ir_types.Type_id.t =
  Sarek_ir_types.Type_id.create ()

let scalar_type_id : type a b. (a, b) scalar_kind -> a Sarek_ir_types.Type_id.t =
  function
  | Float32 -> float32_type_id
  | Float64 -> float64_type_id
  | Int32 -> int32_type_id
  | Int64 -> int64_type_id
  | Char -> char_type_id
  | Complex32 -> complex32_type_id

let type_id : type a b. (a, b) kind -> a Sarek_ir_types.Type_id.t = function
  | Scalar k -> scalar_type_id k
  | Custom c -> c.type_id

let float32_vector_type_id :
    (float, Bigarray.float32_elt) t Sarek_ir_types.Type_id.t =
  Sarek_ir_types.Type_id.create ()

let float64_vector_type_id :
    (float, Bigarray.float64_elt) t Sarek_ir_types.Type_id.t =
  Sarek_ir_types.Type_id.create ()

let int32_vector_type_id :
    (int32, Bigarray.int32_elt) t Sarek_ir_types.Type_id.t =
  Sarek_ir_types.Type_id.create ()

let int64_vector_type_id :
    (int64, Bigarray.int64_elt) t Sarek_ir_types.Type_id.t =
  Sarek_ir_types.Type_id.create ()

let char_vector_type_id :
    (char, Bigarray.int8_unsigned_elt) t Sarek_ir_types.Type_id.t =
  Sarek_ir_types.Type_id.create ()

let complex32_vector_type_id :
    (Complex.t, Bigarray.complex32_elt) t Sarek_ir_types.Type_id.t =
  Sarek_ir_types.Type_id.create ()

let vector_type_id : type a b.
    (a, b) kind -> (a, b) t Sarek_ir_types.Type_id.t = function
  | Scalar Float32 -> float32_vector_type_id
  | Scalar Float64 -> float64_vector_type_id
  | Scalar Int32 -> int32_vector_type_id
  | Scalar Int64 -> int64_vector_type_id
  | Scalar Char -> char_vector_type_id
  | Scalar Complex32 -> complex32_vector_type_id
  | Custom c -> c.vector_type_id
