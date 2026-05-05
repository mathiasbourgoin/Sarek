(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Sarek Runtime - Unified Memory Abstraction
 *
 * Provides a unified interface for GPU memory allocation and data transfer.
 * Uses first-class modules to wrap backend-specific buffers.
 ******************************************************************************)

open Spoc_framework
open Spoc_framework_registry

(** {1 Buffer Module Type} *)

(** A buffer packages backend-specific buffer with its operations. All transfers
    use raw pointers with byte sizes to avoid type parameter escaping issues in
    first-class modules. *)
module type BUFFER = sig
  (** The device this buffer is allocated on *)
  val device : Device.t

  (** Number of elements *)
  val size : int

  (** Size of each element in bytes *)
  val elem_size : int

  (** Get raw device pointer (for kernel arg binding) *)
  val device_ptr : nativeint

  (** Transfer from host pointer to device *)
  val host_ptr_to_device : unit Ctypes.ptr -> byte_size:int -> unit

  (** Transfer from device to host pointer *)
  val device_to_host_ptr : unit Ctypes.ptr -> byte_size:int -> unit

  (** Bind this buffer to kernel args at given index *)
  val bind_to_kargs : Framework_sig.kargs -> int -> unit

  (** Free the buffer *)
  val free : unit -> unit
end

(** Buffer with phantom type parameter for element type safety. The 'a parameter
    is not used at runtime but ensures type-safe transfers. *)
type _ buffer = Buffer : (module BUFFER) -> 'a buffer

(** {1 Allocation} *)

(** Allocate a buffer on a device for standard Bigarray types *)
let alloc (device : Device.t) (size : int) (kind : ('a, 'b) Bigarray.kind) :
    'a buffer =
  match Framework_registry.find_backend device.framework with
  | None -> failwith ("Unknown framework: " ^ device.framework)
  | Some (module B : Framework_sig.BACKEND) ->
      let dev = B.Device.get device.backend_id in
      let buf = B.Memory.alloc dev size kind in
      let elem_size = Ctypes_static.sizeof (Ctypes.typ_of_bigarray_kind kind) in
      Buffer
        (module struct
          let device = device

          let size = size

          let elem_size = elem_size

          let device_ptr = B.Memory.device_ptr buf

          let host_ptr_to_device src_ptr ~byte_size =
            B.Memory.host_ptr_to_device ~src_ptr ~byte_size ~dst:buf

          let device_to_host_ptr dst_ptr ~byte_size =
            B.Memory.device_to_host_ptr ~src:buf ~dst_ptr ~byte_size

          let bind_to_kargs kargs idx =
            match B.unwrap_kargs kargs with
            | Some args -> B.Kernel.set_arg_buffer args idx buf
            | None -> failwith "bind_to_kargs: backend mismatch"

          let free () = B.Memory.free buf
        end : BUFFER)

(** Allocate a buffer for custom types with explicit element size in bytes *)
let alloc_custom (device : Device.t) ~(size : int) ~(elem_size : int) :
    'a buffer =
  match Framework_registry.find_backend device.framework with
  | None -> failwith ("Unknown framework: " ^ device.framework)
  | Some (module B : Framework_sig.BACKEND) ->
      let dev = B.Device.get device.backend_id in
      let buf = B.Memory.alloc_custom dev ~size ~elem_size in
      Buffer
        (module struct
          let device = device

          let size = size

          let elem_size = elem_size

          let device_ptr = B.Memory.device_ptr buf

          let host_ptr_to_device src_ptr ~byte_size =
            B.Memory.host_ptr_to_device ~src_ptr ~byte_size ~dst:buf

          let device_to_host_ptr dst_ptr ~byte_size =
            B.Memory.device_to_host_ptr ~src:buf ~dst_ptr ~byte_size

          let bind_to_kargs kargs idx =
            match B.unwrap_kargs kargs with
            | Some args -> B.Kernel.set_arg_buffer args idx buf
            | None -> failwith "bind_to_kargs: backend mismatch"

          let free () = B.Memory.free buf
        end : BUFFER)

(** {1 Buffer Operations} *)

(** Free a buffer *)
let free : type a. a buffer -> unit = fun (Buffer (module B)) -> B.free ()

(** Copy data from host bigarray to device. Converts bigarray to pointer
    internally. Type parameter ensures bigarray element type matches buffer
    type. *)
let host_to_device : type a b.
    src:(a, b, Bigarray.c_layout) Bigarray.Array1.t -> dst:a buffer -> unit =
 fun ~src ~dst ->
  let (Buffer (module B)) = dst in
  let src_ptr = Ctypes.(bigarray_start array1 src |> to_voidp) in
  let byte_size = Bigarray.Array1.dim src * B.elem_size in
  B.host_ptr_to_device src_ptr ~byte_size

(** Copy data from device to host bigarray. Converts bigarray to pointer
    internally. Type parameter ensures bigarray element type matches buffer
    type. *)
let device_to_host : type a b.
    src:a buffer -> dst:(a, b, Bigarray.c_layout) Bigarray.Array1.t -> unit =
 fun ~src ~dst ->
  let (Buffer (module B)) = src in
  let dst_ptr = Ctypes.(bigarray_start array1 dst |> to_voidp) in
  let byte_size = Bigarray.Array1.dim dst * B.elem_size in
  B.device_to_host_ptr dst_ptr ~byte_size

(** Copy data from raw pointer to device buffer (for custom types) *)
let host_ptr_to_device : type a. src_ptr:unit Ctypes.ptr -> dst:a buffer -> unit
    =
 fun ~src_ptr ~dst ->
  let (Buffer (module B)) = dst in
  let byte_size = B.size * B.elem_size in
  B.host_ptr_to_device src_ptr ~byte_size

(** Copy data from device buffer to raw pointer (for custom types) *)
let device_to_host_ptr : type a. src:a buffer -> dst_ptr:unit Ctypes.ptr -> unit
    =
 fun ~src ~dst_ptr ->
  let (Buffer (module B)) = src in
  let byte_size = B.size * B.elem_size in
  B.device_to_host_ptr dst_ptr ~byte_size

(** Copy data between device buffers (same device). Type parameter ensures both
    buffers have same element type. *)
let device_to_device : type a. src:a buffer -> dst:a buffer -> unit =
 fun ~src ~dst ->
  let (Buffer (module Src)) = src in
  let (Buffer (module Dst)) = dst in
  if Src.device.id <> Dst.device.id then
    failwith "device_to_device requires buffers on same device"
  else begin
    (* Transfer via host - backends may optimize this in the future *)
    let byte_size = Src.size * Src.elem_size in
    let tmp = Ctypes.(allocate_n uint8_t ~count:byte_size) in
    let tmp_ptr = Ctypes.to_voidp tmp in
    Src.device_to_host_ptr tmp_ptr ~byte_size ;
    Dst.host_ptr_to_device tmp_ptr ~byte_size
  end

(** {1 Accessors} *)

(** Get buffer size in elements *)
let size : type a. a buffer -> int = fun (Buffer (module B)) -> B.size

(** Get buffer element size in bytes *)
let elem_size : type a. a buffer -> int = fun (Buffer (module B)) -> B.elem_size

(** Get buffer device *)
let device : type a. a buffer -> Device.t = fun (Buffer (module B)) -> B.device

(** Get raw device pointer *)
let device_ptr : type a. a buffer -> nativeint =
 fun (Buffer (module B)) -> B.device_ptr

(** Bind buffer to kernel args *)
let bind_to_kargs : type a. a buffer -> Framework_sig.kargs -> int -> unit =
 fun (Buffer (module B)) kargs idx -> B.bind_to_kargs kargs idx
