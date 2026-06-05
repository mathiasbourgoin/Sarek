(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * spoc_core_base — FFI-free numeric vector core (hidden functor)
 *
 * Parameterises custom-type storage and device references so that the
 * Bigarray (numeric) path compiles without ctypes or unix.
 * The native sarek.core library instantiates Make(Ctypes_ops) and
 * re-exports the result, keeping the public Spoc_core.* API byte-identical.
 *
 * Users never write Make(...) directly.
 ******************************************************************************)

(** {1 Custom-ops module type} *)

module type CUSTOM_OPS = sig
  (** Opaque handle replacing [unit Ctypes.ptr] in custom storage. *)
  type handle

  (** Opaque device type replacing [Device.t].
      Native: [Spoc_core.Device.t].  Stub: [unit]. *)
  type device_t

  (** Opaque device-buffer type stored in the per-vector device_buffers table.
      Native: [(module Memory.BUFFER)].  Stub: [unit]. *)
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
       src:handle
    -> dst:handle
    -> elem_count:int
    -> get:(handle -> int -> 'a)
    -> set:(handle -> int -> 'a -> unit)
    -> unit

  (** Wrap a Bigarray as a handle (native-only; may [failwith]). *)
  val bigarray_to_handle :
    ('a, 'b, Bigarray.c_layout) Bigarray.Array1.t -> handle

  (** Extract the integer device-ID from a device (used as hashtable key). *)
  val device_id : device_t -> int
end

(** {1 Hidden functor} *)

module Make (Ops : CUSTOM_OPS) = struct
  (** {2 Element types} *)

  type (_, _) scalar_kind =
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

  let to_bigarray_kind : type a b. (a, b) scalar_kind -> (a, b) Bigarray.kind =
    function
    | Float32 -> Bigarray.Float32
    | Float64 -> Bigarray.Float64
    | Int32 -> Bigarray.Int32
    | Int64 -> Bigarray.Int64
    | Char -> Bigarray.Char
    | Complex32 -> Bigarray.Complex32

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
    | Bigarray.Int -> 8
    | Bigarray.Nativeint -> 8
    | Bigarray.Complex32 -> 8
    | Bigarray.Complex64 -> 16
    | Bigarray.Char -> 1

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

  (** {2 Type-id helpers} *)

  let float32_type_id : float Sarek_ir_types.Type_id.t = Sarek_ir_types.Type_id.create ()

  let float64_type_id : float Sarek_ir_types.Type_id.t = Sarek_ir_types.Type_id.create ()

  let int32_type_id : int32 Sarek_ir_types.Type_id.t = Sarek_ir_types.Type_id.create ()

  let int64_type_id : int64 Sarek_ir_types.Type_id.t = Sarek_ir_types.Type_id.create ()

  let char_type_id : char Sarek_ir_types.Type_id.t = Sarek_ir_types.Type_id.create ()

  let complex32_type_id : Complex.t Sarek_ir_types.Type_id.t = Sarek_ir_types.Type_id.create ()

  let scalar_type_id : type a b. (a, b) scalar_kind -> a Sarek_ir_types.Type_id.t
      = function
    | Float32 -> float32_type_id
    | Float64 -> float64_type_id
    | Int32 -> int32_type_id
    | Int64 -> int64_type_id
    | Char -> char_type_id
    | Complex32 -> complex32_type_id

  let type_id : type a b. (a, b) kind -> a Sarek_ir_types.Type_id.t = function
    | Scalar k -> scalar_type_id k
    | Custom c -> c.type_id

  let float32_vector_type_id : (float, Bigarray.float32_elt) t Sarek_ir_types.Type_id.t = Sarek_ir_types.Type_id.create ()

  let float64_vector_type_id : (float, Bigarray.float64_elt) t Sarek_ir_types.Type_id.t = Sarek_ir_types.Type_id.create ()

  let int32_vector_type_id : (int32, Bigarray.int32_elt) t Sarek_ir_types.Type_id.t = Sarek_ir_types.Type_id.create ()

  let int64_vector_type_id : (int64, Bigarray.int64_elt) t Sarek_ir_types.Type_id.t = Sarek_ir_types.Type_id.create ()

  let char_vector_type_id : (char, Bigarray.int8_unsigned_elt) t Sarek_ir_types.Type_id.t = Sarek_ir_types.Type_id.create ()

  let complex32_vector_type_id : (Complex.t, Bigarray.complex32_elt) t Sarek_ir_types.Type_id.t = Sarek_ir_types.Type_id.create ()

  let vector_type_id :
      type a b. (a, b) kind -> (a, b) t Sarek_ir_types.Type_id.t = function
    | Scalar Float32 -> float32_vector_type_id
    | Scalar Float64 -> float64_vector_type_id
    | Scalar Int32 -> int32_vector_type_id
    | Scalar Int64 -> int64_vector_type_id
    | Scalar Char -> char_vector_type_id
    | Scalar Complex32 -> complex32_vector_type_id
    | Custom c -> c.vector_type_id

  (** {2 Creation} *)

  let next_id = ref 0

  let create_scalar (sk : ('a, 'b) scalar_kind) ?(dev : Ops.device_t option)
      (length : int) : ('a, 'b) t =
    incr next_id ;
    let ba_kind = to_bigarray_kind sk in
    let ba = Bigarray.Array1.create ba_kind Bigarray.c_layout length in
    let vec =
      {
        host = Bigarray_storage ba;
        device_buffers = Hashtbl.create 4;
        length;
        kind = Scalar sk;
        location = CPU;
        auto_sync = true;
        id = !next_id;
      }
    in
    (match dev with Some d -> vec.location <- Stale_GPU d | None -> ()) ;
    vec

  let create : type a b. (a, b) kind -> ?dev:Ops.device_t -> int -> (a, b) t =
   fun kind ?dev length ->
    match kind with
    | Scalar sk -> create_scalar sk ?dev length
    | Custom c ->
        incr next_id ;
        let handle = Ops.alloc ~elem_size:c.elem_size ~length in
        let vec =
          {
            host = Custom_storage {ptr = handle; custom = c; length};
            device_buffers = Hashtbl.create 4;
            length;
            kind = Custom c;
            location = CPU;
            auto_sync = true;
            id = !next_id;
          }
        in
        (match dev with Some d -> vec.location <- Stale_GPU d | None -> ()) ;
        vec

  let create_custom (c : 'a custom_type) ?(dev : Ops.device_t option)
      (length : int) : ('a, unit) t =
    create (Custom c) ?dev length

  let of_bigarray (sk : ('a, 'b) scalar_kind)
      (ba : ('a, 'b, Bigarray.c_layout) Bigarray.Array1.t) : ('a, 'b) t =
    incr next_id ;
    {
      host = Bigarray_storage ba;
      device_buffers = Hashtbl.create 4;
      length = Bigarray.Array1.dim ba;
      kind = Scalar sk;
      location = CPU;
      auto_sync = true;
      id = !next_id;
    }

  let of_raw_handle (c : 'a custom_type) (raw : nativeint) (length : int) :
      ('a, unit) t =
    incr next_id ;
    let handle = Ops.of_raw raw in
    {
      host = Custom_storage {ptr = handle; custom = c; length};
      device_buffers = Hashtbl.create 4;
      length;
      kind = Custom c;
      location = CPU;
      auto_sync = true;
      id = !next_id;
    }

  (** {2 Accessors} *)

  let to_bigarray : type a b.
      (a, b) t -> (a, b, Bigarray.c_layout) Bigarray.Array1.t =
   fun vec ->
    match vec.host with
    | Bigarray_storage ba -> ba
    | Custom_storage _ -> invalid_arg "to_bigarray: vector uses custom storage"

  let has_buffer (vec : ('a, 'b) t) (dev : Ops.device_t) : bool =
    Hashtbl.mem vec.device_buffers (Ops.device_id dev)

  let get_buffer (vec : ('a, 'b) t) (dev : Ops.device_t) :
      Ops.device_buf option =
    Hashtbl.find_opt vec.device_buffers (Ops.device_id dev)

  (** {2 Subvector metadata} *)

  type sub_meta = {
    parent_id : int;
    start : int;
    ok_range : int;
    ko_range : int;
    depth : int;
  }

  let subvector_meta : (int, sub_meta) Hashtbl.t = Hashtbl.create 16

  let is_sub (vec : ('a, 'b) t) : bool = Hashtbl.mem subvector_meta vec.id

  let get_sub_meta (vec : ('a, 'b) t) : sub_meta option =
    Hashtbl.find_opt subvector_meta vec.id

  let depth (vec : ('a, 'b) t) : int =
    match get_sub_meta vec with Some meta -> meta.depth | None -> 0

  let parent_id (vec : ('a, 'b) t) : int option =
    match get_sub_meta vec with Some meta -> Some meta.parent_id | None -> None

  let sub_start (vec : ('a, 'b) t) : int option =
    match get_sub_meta vec with Some meta -> Some meta.start | None -> None

  let sub_ok_range (vec : ('a, 'b) t) : int option =
    match get_sub_meta vec with Some meta -> Some meta.ok_range | None -> None

  let sub_ko_range (vec : ('a, 'b) t) : int option =
    match get_sub_meta vec with Some meta -> Some meta.ko_range | None -> None

  (** {2 Copy & slicing} *)

  let copy_host_only (type a b) (vec : (a, b) t) : (a, b) t =
    incr next_id ;
    let host =
      match vec.host with
      | Bigarray_storage ba ->
          let new_ba =
            Bigarray.Array1.create
              (Bigarray.Array1.kind ba)
              Bigarray.c_layout
              vec.length
          in
          Bigarray.Array1.blit ba new_ba ;
          Bigarray_storage new_ba
      | Custom_storage {ptr = handle; custom; length} ->
          let new_handle = Ops.alloc ~elem_size:custom.elem_size ~length in
          Ops.copy_elems ~src:handle ~dst:new_handle ~elem_count:length
            ~get:custom.get ~set:custom.set ;
          Custom_storage {ptr = new_handle; custom; length}
    in
    {
      host;
      device_buffers = Hashtbl.create 4;
      length = vec.length;
      kind = vec.kind;
      location = CPU;
      auto_sync = vec.auto_sync;
      id = !next_id;
    }

  let sub_vector_host (type a b) (vec : (a, b) t) ~(start : int) ~(len : int) :
      (a, b) t =
    if start < 0 || start + len > vec.length then
      invalid_arg
        (Printf.sprintf
           "sub_vector: range [%d, %d) out of bounds [0, %d)"
           start
           (start + len)
           vec.length) ;
    incr next_id ;
    let host =
      match vec.host with
      | Bigarray_storage ba ->
          Bigarray_storage (Bigarray.Array1.sub ba start len)
      | Custom_storage {ptr = handle; custom; _} ->
          let byte_offset = start * custom.elem_size in
          let offset_handle = Ops.add_offset handle byte_offset in
          Custom_storage {ptr = offset_handle; custom; length = len}
    in
    {
      host;
      device_buffers = Hashtbl.create 4;
      length = len;
      kind = vec.kind;
      location = CPU;
      auto_sync = vec.auto_sync;
      id = !next_id;
    }

  let sub_vector (type a b) (vec : (a, b) t) ~(start : int) ~(len : int)
      ~(ok_range : int) ~(ko_range : int) : (a, b) t =
    let sub = sub_vector_host vec ~start ~len in
    let parent_depth =
      match get_sub_meta vec with Some meta -> meta.depth | None -> 0
    in
    Hashtbl.replace
      subvector_meta
      sub.id
      {parent_id = vec.id; start; ok_range; ko_range; depth = parent_depth + 1} ;
    sub

  let partition_host (type a b) (vec : (a, b) t) (devices : Ops.device_t array)
      : (a, b) t array =
    let n = Array.length devices in
    if n = 0 then [||]
    else
      let base = vec.length / n in
      let rem = vec.length mod n in
      Array.init n (fun i ->
          let extra = if i < rem then 1 else 0 in
          let len = base + extra in
          let start = (i * base) + min i rem in
          sub_vector vec ~start ~len ~ok_range:len ~ko_range:0)

  (** {2 List / array creation} *)

  let of_list : type a b. (a, b) kind -> a list -> (a, b) t =
   fun kind lst ->
    let len = List.length lst in
    let vec = create kind len in
    List.iteri
      (fun i v ->
        match vec.host with
        | Bigarray_storage ba -> Bigarray.Array1.set ba i v
        | Custom_storage {ptr; custom; _} -> custom.set ptr i v)
      lst ;
    vec

  let of_array : type a b. (a, b) kind -> a array -> (a, b) t =
   fun kind arr ->
    let vec = create kind (Array.length arr) in
    Array.iteri
      (fun i v ->
        match vec.host with
        | Bigarray_storage ba -> Bigarray.Array1.set ba i v
        | Custom_storage {ptr; custom; _} -> custom.set ptr i v)
      arr ;
    vec

  (** {2 Auto-sync callback} *)

  type sync_callback = {sync : 'a 'b. ('a, 'b) t -> bool}

  let sync_to_cpu_callback : sync_callback option ref = ref None

  let register_sync_callback (cb : sync_callback) : unit =
    sync_to_cpu_callback := Some cb

  let ensure_cpu_sync (type a b) (vec : (a, b) t) : unit =
    if vec.auto_sync then
      match vec.location with
      | Stale_CPU _ -> (
          match !sync_to_cpu_callback with
          | Some cb -> ignore (cb.sync vec)
          | None -> ())
      | _ -> ()

  (** {2 Handle access for the transfer layer} *)

  let host_handle : type a b. (a, b) t -> Ops.handle =
   fun vec ->
    match vec.host with
    | Bigarray_storage ba -> Ops.bigarray_to_handle ba
    | Custom_storage {ptr; _} -> ptr

  let host_raw : type a b. (a, b) t -> nativeint =
   fun vec ->
    match vec.host with
    | Bigarray_storage ba -> Ops.to_raw (Ops.bigarray_to_handle ba)
    | Custom_storage {ptr; _} -> Ops.to_raw ptr

  (** {2 Convenience scalar-kind values} *)

  let float32 = Scalar Float32

  let float64 = Scalar Float64

  let int32 = Scalar Int32

  let int64 = Scalar Int64

  let char = Scalar Char

  let complex32 = Scalar Complex32
end
