(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Typed_value - Extensible typed value system
 *
 * Provides type-safe value representation for:
 * - Kernel argument passing (execute_direct)
 * - Interpreter value storage
 * - Vector element access
 *
 * Extensible: new scalar types (Float128, SIMD) just implement SCALAR_TYPE.
 * PPX generates implementations automatically for [%sarek_intrinsic] and
 * [@@sarek.type].
 ******************************************************************************)

(** {1 Primitive Storage}

    Efficient runtime representations for common types. Exotic types (Float128,
    SIMD) use PBytes. *)

type primitive =
  | PInt32 of int32
  | PInt64 of int64
  | PFloat of float  (** Used by both float32 and float64 *)
  | PBool of bool
  | PBytes of bytes  (** For exotic scalars: Float128, SIMD, etc. *)

(** {1 Scalar Type Interface}

    Each intrinsic type (float32, float64, etc.) implements this interface. PPX
    generates these automatically for [%sarek_intrinsic]. *)

module type SCALAR_TYPE = sig
  type t

  (** Type name for registry lookup (e.g., "float32", "float64") *)
  val name : string

  (** Size in bytes *)
  val size : int

  (** Serialize to primitive storage *)
  val to_primitive : t -> primitive

  (** Deserialize from primitive storage *)
  val of_primitive : primitive -> t
end

(** {1 Composite Type Interface}

    For records and structured types. Uses bytes for field serialization to
    avoid circular dependency with typed_value. PPX generates these
    automatically for [@@sarek.type]. *)

(** Field descriptor *)
type field_desc = {
  fd_name : string;  (** Field name *)
  fd_type : string;  (** Type name for lookup *)
  fd_offset : int;  (** Byte offset in serialized form *)
  fd_size : int;  (** Field size in bytes *)
}

module type COMPOSITE_TYPE = sig
  type t

  (** Type name for registry lookup *)
  val name : string

  (** Total size in bytes *)
  val size : int

  (** Field descriptors *)
  val fields : field_desc list

  (** Serialize entire value to bytes *)
  val to_bytes : t -> bytes

  (** Deserialize from bytes *)
  val of_bytes : bytes -> t
end

(** {1 Typed Values}

    Existential wrappers for type-safe value storage. *)

(** Scalar value with its type module *)
type scalar_value =
  | SV : (module SCALAR_TYPE with type t = 'a) * 'a -> scalar_value

(** Composite value with its type module *)
type composite_value =
  | CV : (module COMPOSITE_TYPE with type t = 'a) * 'a -> composite_value

(** Unified typed value *)
type typed_value = TV_Scalar of scalar_value | TV_Composite of composite_value

(** {1 Vector Interface}

    Abstract interface for vectors, allowing execute_direct to access vector
    contents without depending on the full Vector module. *)

module type EXEC_VECTOR = sig
  type elt

  type underlying

  (** Number of elements *)
  val length : int

  (** Element type name *)
  val type_name : string

  (** Get element as typed_value *)
  val get : int -> typed_value

  (** Set element from typed_value *)
  val set : int -> typed_value -> unit

  (** Type-preserving element access for direct backends. *)
  val get_typed : int -> elt

  val set_typed : int -> elt -> unit

  (** Runtime identity for element and underlying vector/buffer types. *)
  val type_id : elt Sarek_ir_types.Type_id.t

  val underlying_type_id : underlying Sarek_ir_types.Type_id.t

  val underlying : underlying

  (** Get raw device pointer (for kernel binding) *)
  val device_ptr : unit -> nativeint

  (** Element size in bytes *)
  val elem_size : int
end

(** {1 Execution Arguments}

    Type-safe kernel arguments for execute_direct. *)

type exec_arg =
  | EA_Int32 of int32
  | EA_Int64 of int64
  | EA_Float32 of float
  | EA_Float64 of float
  | EA_Scalar : (module SCALAR_TYPE with type t = 'a) * 'a -> exec_arg
  | EA_Composite : (module COMPOSITE_TYPE with type t = 'a) * 'a -> exec_arg
  | EA_Vec of (module EXEC_VECTOR)

(** {1 Type Registry}

    Global registry for type lookup by name. *)

module Registry = struct
  let scalar_types : (string, (module SCALAR_TYPE)) Hashtbl.t =
    Hashtbl.create 32

  let composite_types : (string, (module COMPOSITE_TYPE)) Hashtbl.t =
    Hashtbl.create 32

  let register_scalar (module S : SCALAR_TYPE) =
    Hashtbl.replace scalar_types S.name (module S : SCALAR_TYPE)

  let register_composite (module C : COMPOSITE_TYPE) =
    Hashtbl.replace composite_types C.name (module C : COMPOSITE_TYPE)

  let find_scalar name = Hashtbl.find_opt scalar_types name

  let find_composite name = Hashtbl.find_opt composite_types name

  let list_scalars () =
    Hashtbl.fold (fun name _ acc -> name :: acc) scalar_types []
    |> List.sort String.compare

  let list_composites () =
    Hashtbl.fold (fun name _ acc -> name :: acc) composite_types []
    |> List.sort String.compare
end

(** {1 Built-in Scalar Types}

    Core scalar types always available. *)

module Int32_type : SCALAR_TYPE with type t = int32 = struct
  type t = int32

  let name = "int32"

  let size = 4

  let to_primitive v = PInt32 v

  let of_primitive = function
    | PInt32 v -> v
    | _ -> failwith "Int32_type.of_primitive: expected PInt32"
end

module Int64_type : SCALAR_TYPE with type t = int64 = struct
  type t = int64

  let name = "int64"

  let size = 8

  let to_primitive v = PInt64 v

  let of_primitive = function
    | PInt64 v -> v
    | _ -> failwith "Int64_type.of_primitive: expected PInt64"
end

module Float32_type : SCALAR_TYPE with type t = float = struct
  type t = float

  let name = "float32"

  let size = 4

  let to_primitive v = PFloat v

  let of_primitive = function
    | PFloat v -> v
    | _ -> failwith "Float32_type.of_primitive: expected PFloat"
end

module Float64_type : SCALAR_TYPE with type t = float = struct
  type t = float

  let name = "float64"

  let size = 8

  let to_primitive v = PFloat v

  let of_primitive = function
    | PFloat v -> v
    | _ -> failwith "Float64_type.of_primitive: expected PFloat"
end

module Bool_type : SCALAR_TYPE with type t = bool = struct
  type t = bool

  let name = "bool"

  let size = 1

  let to_primitive v = PBool v

  let of_primitive = function
    | PBool v -> v
    | _ -> failwith "Bool_type.of_primitive: expected PBool"
end

(* Register built-in types *)
let () =
  Registry.register_scalar (module Int32_type) ;
  Registry.register_scalar (module Int64_type) ;
  Registry.register_scalar (module Float32_type) ;
  Registry.register_scalar (module Float64_type) ;
  Registry.register_scalar (module Bool_type)

(** {1 Convenience Functions} *)

(** Create exec_arg from typed_value *)
let exec_arg_of_typed_value = function
  | TV_Scalar (SV ((module S), v)) -> EA_Scalar ((module S), v)
  | TV_Composite (CV ((module C), v)) -> EA_Composite ((module C), v)

(** Create typed_value from exec_arg (for scalars/composites only) *)
let typed_value_of_exec_arg = function
  | EA_Int32 n -> TV_Scalar (SV ((module Int32_type), n))
  | EA_Int64 n -> TV_Scalar (SV ((module Int64_type), n))
  | EA_Float32 f -> TV_Scalar (SV ((module Float32_type), f))
  | EA_Float64 f -> TV_Scalar (SV ((module Float64_type), f))
  | EA_Scalar ((module S), v) -> TV_Scalar (SV ((module S), v))
  | EA_Composite ((module C), v) -> TV_Composite (CV ((module C), v))
  | EA_Vec _ -> failwith "typed_value_of_exec_arg: cannot convert vector"

(** Get primitive value from scalar_value *)
let primitive_of_scalar (SV ((module S), v)) = S.to_primitive v

(** Get type name from primitive *)
let primitive_type_name = function
  | PInt32 _ -> "int32"
  | PInt64 _ -> "int64"
  | PFloat _ -> "float"
  | PBool _ -> "bool"
  | PBytes _ -> "bytes"

(** Get type name from typed_value *)
let typed_value_type_name = function
  | TV_Scalar (SV ((module S), _)) -> S.name
  | TV_Composite (CV ((module C), _)) -> C.name

(** Get type name from exec_arg *)
let type_name_of_exec_arg = function
  | EA_Int32 _ -> "int32"
  | EA_Int64 _ -> "int64"
  | EA_Float32 _ -> "float32"
  | EA_Float64 _ -> "float64"
  | EA_Scalar ((module S), _) -> S.name
  | EA_Composite ((module C), _) -> C.name
  | EA_Vec (module V) -> V.type_name ^ " vector"
