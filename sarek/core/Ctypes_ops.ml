(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Ctypes_ops — native implementation of Spoc_core_base.CUSTOM_OPS
 *
 * Provides the ctypes-backed operations for custom-type vectors. This module
 * is used only on the native side; FFI-free builds use Stub_ops instead.
 *
 * Buffer ownership: allocations via [alloc] are C heap memory managed by the
 * ctypes GC; [free] is a no-op (included for CUSTOM_OPS contract symmetry).
 ******************************************************************************)

type handle = unit Ctypes.ptr

type device_t = Device.t

type device_buf = (module Memory.BUFFER)

let alloc ~(elem_size : int) ~(length : int) : handle =
  let byte_size = length * elem_size in
  let ptr = Ctypes.(allocate_n (array 1 char) ~count:byte_size) in
  Ctypes.coerce Ctypes.(ptr (array 1 char)) Ctypes.(ptr void) ptr

let free (_ptr : handle) : unit = ()

let of_raw (addr : nativeint) : handle = Ctypes.ptr_of_raw_address addr

let to_raw (ptr : handle) : nativeint = Ctypes.raw_address_of_ptr ptr

let add_offset (ptr : handle) (byte_offset : int) : handle =
  let raw = Ctypes.raw_address_of_ptr ptr in
  Ctypes.ptr_of_raw_address (Nativeint.add raw (Nativeint.of_int byte_offset))

let copy_elems ~(src : handle) ~(dst : handle) ~(elem_count : int)
    ~(get : handle -> int -> 'a) ~(set : handle -> int -> 'a -> unit) : unit =
  for i = 0 to elem_count - 1 do
    set dst i (get src i)
  done

let bigarray_to_handle :
    type a b. (a, b, Bigarray.c_layout) Bigarray.Array1.t -> handle =
 fun ba -> Ctypes.(bigarray_start array1 ba |> to_voidp)

let device_id (dev : device_t) : int = dev.id
