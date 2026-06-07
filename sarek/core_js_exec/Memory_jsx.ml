(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Memory_jsx — FFI-free device buffer module type for the jsoo Execute path.
 *
 * Provides only the members Execute reads: device_ptr, size, elem_size,
 * bind_to_kargs, and free. No ctypes, no unix.
 ******************************************************************************)

(** FFI-free device buffer — only the members Execute reads. *)
module type BUFFER = sig
  (** Raw device pointer (nativeint; no ctypes) *)
  val device_ptr : nativeint

  (** Number of elements in this buffer *)
  val size : int

  (** Size of each element in bytes *)
  val elem_size : int

  (** Bind this buffer to kernel args at the given index *)
  val bind_to_kargs : Spoc_framework.Framework_sig.kargs -> int -> unit

  (** Release the buffer *)
  val free : unit -> unit
end
