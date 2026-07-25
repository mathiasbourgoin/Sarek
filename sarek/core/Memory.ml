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

(** {1 Pure element-size helper} *)

(** Byte size of each Bigarray element kind on a 64-bit target. Replaces
    [Ctypes_static.sizeof (Ctypes.typ_of_bigarray_kind kind)] so that the
    numeric-only path is free of ctypes. Sizes are identical to what ctypes
    sizeof returns on 64-bit: the underlying C representation widths. *)
let bigarray_elem_size : type a b. (a, b) Bigarray.kind -> int = function
  | Bigarray.Float16 -> 2
  | Bigarray.Float32 -> 4
  | Bigarray.Float64 -> 8
  | Bigarray.Int8_signed -> 1
  | Bigarray.Int8_unsigned -> 1
  | Bigarray.Int16_signed -> 2
  | Bigarray.Int16_unsigned -> 2
  | Bigarray.Int32 -> 4
  | Bigarray.Int64 -> 8
  (* OCaml int / nativeint are word-sized: 8 on 64-bit, 4 on 32-bit — matches
     what Ctypes_static.sizeof returned per platform. *)
  | Bigarray.Int -> Sys.word_size / 8
  | Bigarray.Nativeint -> Sys.word_size / 8
  | Bigarray.Complex32 -> 8
  | Bigarray.Complex64 -> 16
  | Bigarray.Char -> 1

(** {1 Host bigarray -> raw pointer} *)

(** Raw data pointer of a host bigarray, for the H2D/D2H byte transfers.

    [Ctypes.bigarray_start] cannot be used for f16: ctypes maps a
    [Bigarray.kind] to its own [Ctypes_bigarray.kind] GADT, and that mapping has
    NO [Float16] arm — it ends in [failwith "Unsupported bigarray kind"]
    (ctypes_bigarray_stubs.ml). An f16 vector therefore raised on its first
    transfer even though every byte size involved was already correct.

    This corrects a claim in the f16 design spec (§3.3, "transfer is
    element-type-agnostic beyond byte size — no change needed"): the transfer
    ARITHMETIC is indeed element-agnostic, but ACQUIRING the host pointer is
    not.

    [Ctypes_bigarray.unsafe_address] is the kind-independent primitive (it is
    just [Caml_ba_data_val]) and is used for f16 only, keeping every other
    element type on the exact pre-existing code path.

    GC ROOTS (#57 slice 1 review, MF3). The f16 arm must NOT be built with
    [Ctypes.ptr_of_raw_address]: that is [make_unmanaged], and it silently drops
    the GC root that [Ctypes.bigarray_start] establishes. ctypes' own
    [bigarray_start] returns a MANAGED fat pointer —
    [Fat.make ~managed:(Some (Obj.repr ba)) ~reftyp raw] (ctypes_memory.ml
    :302-305) — so for every non-f16 kind the pointer VALUE is itself a root for
    the bigarray, and ctypes' FFI keeps it alive across the call
    (ctypes_ffi.ml:112-119). Probe, three pointers built over the same shape and
    then subjected to a major GC with the pointer still live:

    {v
    f32 bigarray_start (managed)           : freed during ffi call = 0
    f16 ptr_of_raw_address (unmanaged)     : freed during ffi call = 1
    f16 Fat.make ~managed (this arm)       : freed during ffi call = 0
    v}

    So the f16 arm reconstructs the SAME fat-pointer shape as ctypes, over the
    kind-independent address. That restores pre-existing GC semantics for f16
    with zero per-caller discipline, rather than asking five call sites to
    remember a keepalive (three of which did not).

    This does deepen an existing dependency on ctypes internals — already
    present via [Ctypes_bigarray.unsafe_address] here and via
    [Ctypes_ptr.voidp = nativeint] at [Vector_transfer]/[Framework_sig] — now
    also on [Ctypes_static.CPointer] and [Ctypes_ptr.Fat.make]. Both are exposed
    in [ctypes_static.mli] (the [pointer] GADT) and [ctypes_ptr.ml]'s signature,
    and they are the only way to express "managed pointer to a Float16 bigarray"
    while ctypes' kind GADT has no [Float16] arm. If ctypes ever grows one, this
    whole function collapses back to [bigarray_start].

    Callers converting the result to a bare [nativeint] via
    [Ctypes.raw_address_of_ptr] strip the root again — that is a SEPARATE
    obligation, stated verbatim at [Framework_sig.ml:218-223], and it applies to
    every element type, not just f16. It is discharged with
    [Sys.opaque_identity] at the sites below and in [Transfer.ml]. *)
let bigarray_void_ptr : type a b.
    (a, b, Bigarray.c_layout) Bigarray.Array1.t -> unit Ctypes.ptr =
 fun ba ->
  match Bigarray.Array1.kind ba with
  | Bigarray.Float16 ->
      Ctypes_static.CPointer
        (Ctypes_ptr.Fat.make
           ~managed:(Some (Obj.repr ba))
           ~reftyp:Ctypes_static.Void
           (Ctypes_bigarray.unsafe_address ba))
  | _ -> Ctypes.(bigarray_start array1 ba |> to_voidp)

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
      let elem_size = bigarray_elem_size kind in
      Buffer
        (module struct
          let device = device

          let size = size

          let elem_size = elem_size

          let device_ptr = B.Memory.device_ptr buf

          let host_ptr_to_device src_ptr ~byte_size =
            B.Memory.host_ptr_to_device
              ~src_ptr:(Ctypes.raw_address_of_ptr src_ptr)
              ~byte_size
              ~dst:buf

          let device_to_host_ptr dst_ptr ~byte_size =
            B.Memory.device_to_host_ptr
              ~src:buf
              ~dst_ptr:(Ctypes.raw_address_of_ptr dst_ptr)
              ~byte_size

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
            B.Memory.host_ptr_to_device
              ~src_ptr:(Ctypes.raw_address_of_ptr src_ptr)
              ~byte_size
              ~dst:buf

          let device_to_host_ptr dst_ptr ~byte_size =
            B.Memory.device_to_host_ptr
              ~src:buf
              ~dst_ptr:(Ctypes.raw_address_of_ptr dst_ptr)
              ~byte_size

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
  let src_ptr = bigarray_void_ptr src in
  let byte_size = Bigarray.Array1.dim src * B.elem_size in
  B.host_ptr_to_device src_ptr ~byte_size ;
  (* Keep [src] reachable until the transfer has consumed the raw address.
     [B.host_ptr_to_device] converts the pointer to a bare [nativeint] via
     [Ctypes.raw_address_of_ptr] (see the closures in [alloc] above), and a
     nativeint is not a GC root — Framework_sig.ml:218-223 states this caller
     obligation verbatim. It applies to EVERY element type, not only f16.

     There are five callers of [B.host_ptr_to_device] / [B.device_to_host_ptr]:
     this one, [device_to_host] below, the two raw-pointer forwarders (which
     pass the obligation up to their own caller), [device_to_device] (whose
     [tmp] is a live ctypes allocation), and [Transfer.ml]'s vector path — which
     discharges it the same way. *)
  ignore (Sys.opaque_identity src)

(** Copy data from device to host bigarray. Converts bigarray to pointer
    internally. Type parameter ensures bigarray element type matches buffer
    type. *)
let device_to_host : type a b.
    src:a buffer -> dst:(a, b, Bigarray.c_layout) Bigarray.Array1.t -> unit =
 fun ~src ~dst ->
  let (Buffer (module B)) = src in
  let dst_ptr = bigarray_void_ptr dst in
  let byte_size = Bigarray.Array1.dim dst * B.elem_size in
  B.device_to_host_ptr dst_ptr ~byte_size ;
  ignore (Sys.opaque_identity dst)

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
