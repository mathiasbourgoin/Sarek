(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * spoc_core_base — FFI-free numeric vector core (public interface)
 *
 * This library's dune stanza lists no ctypes, unix, or spoc_framework.
 * The native [sarek.core] library instantiates [Make(Ctypes_ops)] and
 * re-exports the result, keeping the public [Spoc_core.*] API unchanged.
 ******************************************************************************)

(** {1 Custom-ops module type} *)

(** Abstraction over the raw pointer, device type, and device-buffer type. The
    pure Bigarray path never calls any of these operations. *)
module type CUSTOM_OPS = sig
  (** Opaque handle replacing [unit Ctypes.ptr] in custom storage. *)
  type handle

  (** Opaque device type replacing [Device.t]. Native: [Spoc_core.Device.t].
      Stub: [unit]. *)
  type device_t

  (** Opaque device-buffer type stored in the per-vector device_buffers table.
      Native: [(module Memory.BUFFER)]. Stub: [unit]. *)
  type device_buf

  (** Allocate storage for [length] elements of [elem_size] bytes each. *)
  val alloc : elem_size:int -> length:int -> handle

  (** Release storage previously returned by [alloc]. *)
  val free : handle -> unit

  (** Wrap a raw address as a handle (native-only; may [failwith]). *)
  val of_raw : nativeint -> handle

  (** Compute the raw address of a handle (native-only; may [failwith]). *)
  val to_raw : handle -> nativeint

  (** Advance a handle by [byte_offset] bytes. *)
  val add_offset : handle -> int -> handle

  (** Copy [elem_count] elements using the provided get/set pair. *)
  val copy_elems :
    src:handle ->
    dst:handle ->
    elem_count:int ->
    get:(handle -> int -> 'a) ->
    set:(handle -> int -> 'a -> unit) ->
    unit

  (** Convert a host Bigarray to a handle for the device-transfer layer.
      Implementations that do not support device transfers may raise. *)
  val bigarray_to_handle :
    ('a, 'b, Bigarray.c_layout) Bigarray.Array1.t -> handle

  (** Extract the integer device-ID from a device (used as hashtable key). *)
  val device_id : device_t -> int

  (** Serialize a custom-type value to a byte string using the provided [set].
      The native implementation allocates a temporary ctypes char buffer; the
      jsoo stub raises. *)
  val custom_to_bytes :
    set:(handle -> int -> 'a -> unit) -> elem_size:int -> 'a -> bytes

  (** Deserialize a byte string to a custom-type value using the provided [get].
      The native implementation allocates a temporary ctypes char buffer; the
      jsoo stub raises. *)
  val custom_of_bytes :
    get:(handle -> int -> 'a) -> elem_size:int -> bytes -> 'a
end

(** {1 Hidden functor} *)

module Make (Ops : CUSTOM_OPS) : sig
  (** {2 Element types} *)

  type ('a, 'b) scalar_kind = ('a, 'b) Spoc_core_base_scalar.scalar_kind =
    | Float32 : (float, Bigarray.float32_elt) scalar_kind
    | Float64 : (float, Bigarray.float64_elt) scalar_kind
    | Int32 : (int32, Bigarray.int32_elt) scalar_kind
    | Int64 : (int64, Bigarray.int64_elt) scalar_kind
    | Char : (char, Bigarray.int8_unsigned_elt) scalar_kind
    | Complex32 : (Complex.t, Bigarray.complex32_elt) scalar_kind

  type location =
    | CPU
    | GPU of Ops.device_t
    | Both of Ops.device_t
    | Stale_CPU of Ops.device_t
    | Stale_GPU of Ops.device_t

  type 'a custom_type = {
    elem_size : int;
    type_id : 'a Sarek_ir_types.Type_id.t;
    vector_type_id : ('a, unit) t Sarek_ir_types.Type_id.t;
    get : Ops.handle -> int -> 'a;
    set : Ops.handle -> int -> 'a -> unit;
    name : string;
  }

  and (_, _) kind =
    | Scalar : ('a, 'b) scalar_kind -> ('a, 'b) kind
    | Custom : 'a custom_type -> ('a, unit) kind

  and (_, _) host_storage =
    | Bigarray_storage :
        ('a, 'b, Bigarray.c_layout) Bigarray.Array1.t
        -> ('a, 'b) host_storage
    | Custom_storage : {
        ptr : Ops.handle;
        custom : 'a custom_type;
        length : int;
      }
        -> ('a, unit) host_storage

  and ('a, 'b) t = {
    host : ('a, 'b) host_storage;
    device_buffers : (int, Ops.device_buf) Hashtbl.t;
    length : int;
    kind : ('a, 'b) kind;
    mutable location : location;
    mutable auto_sync : bool;
    id : int;
  }

  (** {2 Kind helpers — pure} *)

  val to_bigarray_kind : ('a, 'b) scalar_kind -> ('a, 'b) Bigarray.kind

  val bigarray_elem_size : ('a, 'b) Bigarray.kind -> int

  val scalar_elem_size : ('a, 'b) scalar_kind -> int

  val elem_size : ('a, 'b) kind -> int

  val scalar_kind_name : ('a, 'b) scalar_kind -> string

  val kind_name : ('a, 'b) kind -> string

  (** {2 Type-id helpers} *)

  val float32_type_id : float Sarek_ir_types.Type_id.t

  val float64_type_id : float Sarek_ir_types.Type_id.t

  val int32_type_id : int32 Sarek_ir_types.Type_id.t

  val int64_type_id : int64 Sarek_ir_types.Type_id.t

  val char_type_id : char Sarek_ir_types.Type_id.t

  val complex32_type_id : Complex.t Sarek_ir_types.Type_id.t

  val scalar_type_id : ('a, 'b) scalar_kind -> 'a Sarek_ir_types.Type_id.t

  val type_id : ('a, 'b) kind -> 'a Sarek_ir_types.Type_id.t

  val float32_vector_type_id :
    (float, Bigarray.float32_elt) t Sarek_ir_types.Type_id.t

  val float64_vector_type_id :
    (float, Bigarray.float64_elt) t Sarek_ir_types.Type_id.t

  val int32_vector_type_id :
    (int32, Bigarray.int32_elt) t Sarek_ir_types.Type_id.t

  val int64_vector_type_id :
    (int64, Bigarray.int64_elt) t Sarek_ir_types.Type_id.t

  val char_vector_type_id :
    (char, Bigarray.int8_unsigned_elt) t Sarek_ir_types.Type_id.t

  val complex32_vector_type_id :
    (Complex.t, Bigarray.complex32_elt) t Sarek_ir_types.Type_id.t

  val vector_type_id : ('a, 'b) kind -> ('a, 'b) t Sarek_ir_types.Type_id.t

  (** {2 Creation} *)

  val create_scalar :
    ('a, 'b) scalar_kind -> ?dev:Ops.device_t -> int -> ('a, 'b) t

  val create : ('a, 'b) kind -> ?dev:Ops.device_t -> int -> ('a, 'b) t

  val create_custom : 'a custom_type -> ?dev:Ops.device_t -> int -> ('a, unit) t

  val of_bigarray :
    ('a, 'b) scalar_kind ->
    ('a, 'b, Bigarray.c_layout) Bigarray.Array1.t ->
    ('a, 'b) t

  val of_raw_handle : 'a custom_type -> nativeint -> int -> ('a, unit) t

  (** {2 Accessors} *)

  val to_bigarray : ('a, 'b) t -> ('a, 'b, Bigarray.c_layout) Bigarray.Array1.t

  val has_buffer : ('a, 'b) t -> Ops.device_t -> bool

  val get_buffer : ('a, 'b) t -> Ops.device_t -> Ops.device_buf option

  (** {2 Custom-type marshal helpers} *)

  (** Serialize a custom-type value to bytes. Delegates to
      [Ops.custom_to_bytes]. Raises on jsoo builds (custom-type device path not
      yet implemented). *)
  val custom_to_bytes : 'a custom_type -> 'a -> bytes

  (** Deserialize bytes to a custom-type value. Delegates to
      [Ops.custom_of_bytes]. Raises on jsoo builds (custom-type device path not
      yet implemented). *)
  val custom_of_bytes : 'a custom_type -> bytes -> 'a

  (** {2 Subvector metadata} *)

  type sub_meta = {
    parent_id : int;
    start : int;
    ok_range : int;
    ko_range : int;
    depth : int;
  }

  val is_sub : ('a, 'b) t -> bool

  val get_sub_meta : ('a, 'b) t -> sub_meta option

  val depth : ('a, 'b) t -> int

  val parent_id : ('a, 'b) t -> int option

  val sub_start : ('a, 'b) t -> int option

  val sub_ok_range : ('a, 'b) t -> int option

  val sub_ko_range : ('a, 'b) t -> int option

  (** {2 Copy & slicing} *)

  val copy_host_only : ('a, 'b) t -> ('a, 'b) t

  val sub_vector_host : ('a, 'b) t -> start:int -> len:int -> ('a, 'b) t

  val sub_vector :
    ('a, 'b) t ->
    start:int ->
    len:int ->
    ok_range:int ->
    ko_range:int ->
    ('a, 'b) t

  val partition_host : ('a, 'b) t -> Ops.device_t array -> ('a, 'b) t array

  (** {2 List / array creation} *)

  val of_list : ('a, 'b) kind -> 'a list -> ('a, 'b) t

  val of_array : ('a, 'b) kind -> 'a array -> ('a, 'b) t

  (** {2 Auto-sync callback} *)

  type sync_callback = {sync : 'a 'b. ('a, 'b) t -> bool}

  val register_sync_callback : sync_callback -> unit

  val ensure_cpu_sync : ('a, 'b) t -> unit

  (** {2 Handle access for the transfer layer} *)

  val host_handle : ('a, 'b) t -> Ops.handle

  val host_raw : ('a, 'b) t -> nativeint

  (** {2 Convenience scalar-kind values} *)

  val float32 : (float, Bigarray.float32_elt) kind

  val float64 : (float, Bigarray.float64_elt) kind

  val int32 : (int32, Bigarray.int32_elt) kind

  val int64 : (int64, Bigarray.int64_elt) kind

  val char : (char, Bigarray.int8_unsigned_elt) kind

  val complex32 : (Complex.t, Bigarray.complex32_elt) kind
end
