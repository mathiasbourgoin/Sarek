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

module Make (Ops : CUSTOM_OPS) = struct
  (** {2 Element types} *)

  (** Re-export scalar_kind with constructors so [Make(Ops).Float32] etc.
      resolve. The underlying type is [Spoc_core_base_scalar.scalar_kind]. *)
  type ('a, 'b) scalar_kind = ('a, 'b) Spoc_core_base_scalar.scalar_kind =
    | Float16 : (float, Bigarray.float16_elt) scalar_kind
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
    ir_fields : (string * Sarek_ir_types.elttype) list option;
        (** Immediate fields of the element type, in declaration order, when the
            element is a flat scalar record whose byte layout is derivable by
            {!Sarek_ir_layout.record_layout}. [None] means "layout not derivable
            here" — variants, hand-written descriptors, and any record with a
            field type outside the six the host marshaller can read/write.
            Consumers must treat [None] as "no SoA", never as "no fields".

            Trust contract: [ir_fields] carries exactly the same trust as
            [elem_size] — it is metadata *about* ['a], not a witness *of* it,
            and the type system does not relate the two. What makes it sound is
            that the producer derives [ir_fields], [elem_size] and [get]/[set]
            from one source. Concretely, for a PPX-generated record all four
            come from [Sarek_ppx.aligned_record_offsets]; for {!Sarek_tuple_vec}
            all four come from one [Sarek_ir_layout.record_layout] call. A
            producer that derives them separately can desynchronize them
            silently, which is wrong data rather than a type error — see
            [test_ir_fields.ml], which pins the agreement by probing the bytes
            [set] actually writes. *)
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

  (** Opt-in Structure-of-Arrays binding for a custom (flat-record) vector
      (backlog-54). Present iff the vector came from
      [Soa_vector.create_transparent], which is the only producer of this
      record.

      There is no [Vector.create ~layout:SoA] and no [layout] parameter
      anywhere; that shape was PROPOSED by the Tier 1b handoff and rejected —
      see [Soa_vector.ml]'s [create_transparent] comment for why (it would
      invert the [Vector] -> [Soa_vector] dependency, and the transparency the
      item is about is at the LAUNCH site, not at the constructor).

      {b Why this is a field of closures and ints, and NOT a [host_storage]
         constructor carrying a [Soa.plan].} The Tier 1b handoff doc proposed
      the latter; it is not buildable. This library is deliberately FFI-free and
      is compiled to [.bc.js] as well as [.bc] — its dune stanza must not list
      [ctypes], and [sarek/core/ffi_free_gate] enforces that at build time.
      [Soa]/[Soa_vector] live in [spoc_core] and use [Ctypes] directly to copy
      raw words, and [Vector_types] is above this layer too, so naming any of
      them here would either break that gate or force the plan representation to
      be duplicated across the layer boundary — a drift hazard of exactly the
      kind the [ir_fields] trust note warns about.

      So the layer inversion is resolved with behaviour instead of types: the
      producer ({!Vector.create} in [spoc_core], which may use ctypes freely)
      supplies the transpose as closures plus the two plain numbers a launch
      needs. Nothing ctypes-shaped crosses down, and this stays a value a jsoo
      build can hold.

      Choosing a record field over a GADT constructor is also what makes this
      affordable: a new [host_storage] constructor forces all 51 match arms
      across 5 files; a field forces only the 6 sites that build this record,
      all of them in this file. It keeps the same property that made the GADT
      attractive — the compiler finds every site — without the fan-out, and
      without a hidden global side table keyed by vector id. *)
  and soa_binding = {
    soa_num_leaves : int;
        (** Number of scalar leaves = number of base pointers a launch binds. *)
    soa_aos_stride : int;  (** Packed AoS element size, in bytes. *)
    soa_scatter : unit -> unit;
        (** Transpose the AoS host buffer into the N per-leaf host buffers. Call
            before transferring the leaves to a device. *)
    soa_gather : unit -> unit;
        (** Transpose the N per-leaf host buffers back into the AoS host buffer.
            Call after reading leaves back from a device that wrote them. *)
    soa_to_device : Ops.device_t -> unit;
        (** Scatter, then transfer every leaf to [device]. One closure rather
            than "scatter" + "transfer" separately, so a caller cannot do the
            first and forget the second. *)
    soa_leaf_bufs : Ops.device_t -> Ops.device_buf list;
        (** The N leaf device buffers in {!Soa.plan} leaf (record declaration)
            order — exactly the order the emitted PTX param block expects. Typed
            in [Ops] terms, which is what lets [Execute.ml] consume it: it is
            compiled for the jsoo target too, where [Soa_vector] does not exist
            (sarek/execute/jsoo/dune copies Execute.ml but not Soa_launch.ml),
            so it can never name the SoA modules directly. *)
    soa_from_device : Ops.device_t -> unit;
        (** The inverse of {!soa_to_device}: read every leaf back from [device],
            then gather into the packed AoS host buffer.

            Without this a kernel's OUTPUT is silently lost. A launch that took
            the SoA ABI wrote into the N leaf buffers, and the packed device
            buffer it did not touch is what an ordinary [Transfer.to_cpu]
            downloads — so the host would see whatever the AoS buffer last held,
            with no error anywhere. Paired with {!soa_leaves_live} below, which
            is what says the leaves are the ones holding the results. *)
    soa_free_leaves : Ops.device_t option -> unit;
        (** Release the leaf device buffers: [Some dev] frees them on that
            device only, [None] on every device. Then RE-DERIVES
            {!soa_leaves_live} from the leaves that are still allocated, rather
            than clearing it: a freed leaf holds nothing a read-back could
            fetch, but the flag covers the whole vector, so a [Some dev] free
            must leave it set while leaves survive on another device. Clearing
            it unconditionally made [free_buffer] on device B disown live leaves
            on device A, and the drain-before-free then skipped A — freeing a
            device the results were not on discarded them.

            Its absence was a LEAK rather than a correctness bug, which is why
            it is easy to miss. Under this ABI the packed AoS buffer is never
            allocated, so [Transfer.free_all_buffers] iterated an EMPTY
            [device_buffers] table and returned having released zero bytes —
            measured 2026-07-30 with [Gpu_memory.usage()]: 3840 B before a
            launch, 4224 B after the free (delta +384 = 32 elements x 3 leaves x
            4 B, i.e. the leaves the call was asked to release). The data was
            still correct afterwards, and the leaves were still reclaimed
            EVENTUALLY — each carries a [Gpu_memory.register_finalizer] — so
            nothing looked wrong. What the caller lost was the one thing an
            explicit free is for: releasing the memory at a time it chooses,
            while the structure is still reachable. *)
    soa_leaves_live : bool ref;
        (** Have the leaf buffers been written by an SoA-ABI launch?

            The launch site and the read-back site must not answer "SoA or AoS?"
            independently — that is the same divergence hazard the single
            [soa_dispatch] predicate exists to prevent, one step later in the
            round trip. So this is not a second predicate: read-back only READS
            it. A launch that never took the SoA ABI (an external source through
            [Execute.run_source], or any non-PTX backend) leaves it [false], and
            read-back then correctly downloads the packed buffer.

            THREE writers, and the list is exhaustive — anything else that flips
            this is a bug: {!soa_to_device} sets it; {!soa_free_leaves}
            re-derives it from the leaves that survive the free; and
            [Execute.transfer_vectors_to_device] clears it when a launch takes
            the packed ABI. Between them it states the ABI of the MOST RECENT
            operation, not of some operation — which is the property the
            read-back paths rely on and the one that was missing when nothing
            ever cleared it.

            Note what "clears it" cost in {!soa_free_leaves}: this flag is
            whole-vector while that free is per-device, so the two only agree
            when the free covered every device. Deriving instead of assigning is
            what keeps a single flag honest across a per-device release.

            A [bool ref] rather than a mutable field so that the closure which
            performs the upload can set it itself. A mutable field would have to
            be set by the CALLER after invoking [soa_to_device] — exactly the
            "do the first and forget the second" split that closure is shaped to
            make impossible. *)
  }

  and ('a, 'b) t = {
    host : ('a, 'b) host_storage;
    device_buffers : (int, Ops.device_buf) Hashtbl.t;
    length : int;
    kind : ('a, 'b) kind;
    mutable location : location;
    mutable auto_sync : bool;
    id : int;
    mutable soa : soa_binding option;
        (** [None] for every vector except one from
            [Soa_vector.create_transparent] (there is no [~layout] parameter —
            see {!soa_binding}). Read by the launch path to decide whether to
            bind N leaf pointers instead of one packed buffer; ignored by every
            host-side operation, which is why [get]/[set] and the PPX accessors
            need no changes at all. *)
  }

  (** {2 Kind helpers — delegated to Spoc_core_base_scalar} *)

  let to_bigarray_kind = Spoc_core_base_scalar.to_bigarray_kind

  let bigarray_elem_size = Spoc_core_base_scalar.bigarray_elem_size

  let scalar_elem_size = Spoc_core_base_scalar.scalar_elem_size

  let elem_size : type a b. (a, b) kind -> int = function
    | Scalar k -> scalar_elem_size k
    | Custom c -> c.elem_size

  let scalar_kind_name = Spoc_core_base_scalar.scalar_kind_name

  let kind_name : type a b. (a, b) kind -> string = function
    | Scalar k -> scalar_kind_name k
    | Custom c -> "Custom(" ^ c.name ^ ")"

  (** {2 Type-id helpers — delegated to Spoc_core_base_scalar} *)

  let float16_type_id = Spoc_core_base_scalar.float16_type_id

  let float32_type_id = Spoc_core_base_scalar.float32_type_id

  let float64_type_id = Spoc_core_base_scalar.float64_type_id

  let int32_type_id = Spoc_core_base_scalar.int32_type_id

  let int64_type_id = Spoc_core_base_scalar.int64_type_id

  let char_type_id = Spoc_core_base_scalar.char_type_id

  let complex32_type_id = Spoc_core_base_scalar.complex32_type_id

  let scalar_type_id = Spoc_core_base_scalar.scalar_type_id

  let type_id : type a b. (a, b) kind -> a Sarek_ir_types.Type_id.t = function
    | Scalar k -> scalar_type_id k
    | Custom c -> c.type_id

  let float16_vector_type_id :
      (float, Bigarray.float16_elt) t Sarek_ir_types.Type_id.t =
    Sarek_ir_types.Type_id.create ()

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
    | Scalar Float16 -> float16_vector_type_id
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
        soa = None;
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
            soa = None;
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
      soa = None;
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
      soa = None;
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

  let get_buffer (vec : ('a, 'b) t) (dev : Ops.device_t) : Ops.device_buf option
      =
    Hashtbl.find_opt vec.device_buffers (Ops.device_id dev)

  (** {2 Custom-type marshal helpers — delegated to Ops} *)

  (** Serialize a custom-type value to bytes via [Ops.custom_to_bytes]. *)
  let custom_to_bytes (type a) (c : a custom_type) (v : a) : bytes =
    Ops.custom_to_bytes ~set:c.set ~elem_size:c.elem_size v

  (** Deserialize bytes to a custom-type value via [Ops.custom_of_bytes]. *)
  let custom_of_bytes (type a) (c : a custom_type) (b : bytes) : a =
    Ops.custom_of_bytes ~get:c.get ~elem_size:c.elem_size b

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
    match get_sub_meta vec with
    | Some meta -> Some meta.parent_id
    | None -> None

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
          Ops.copy_elems
            ~src:handle
            ~dst:new_handle
            ~elem_count:length
            ~get:custom.get
            ~set:custom.set ;
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
      (* The copy is AoS-only BY DESIGN, and this is the one [soa] decision in
         this module that a reader could mistake for an omission. A binding's
         closures capture the SOURCE vector's leaves — inheriting it would give
         the copy a fast path that scatters from, and gathers into, memory the
         copy does not own, so a launch on the copy would write the original's
         leaves. Copying the leaves as well is a different feature, not a smaller
         one: it needs a per-device duplicate of every leaf buffer. Dropping the
         binding costs nothing but speed — the copy is still a complete packed AoS
         vector, which is what every non-CUDA/PTX launch of the original uses
         anyway. *)
      soa = None;
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
      (* AoS-only for a stronger reason than {!copy_host_only} above: a
         subvector's length is [len], while the parent binding's leaves are
         [vec.length] long and its plan's scatter/gather run over that whole
         length. There is no offset anywhere in the binding to narrow, so the
         parent's binding does not describe this vector at all — it is not a
         speed choice here but the only correct answer. *)
      soa = None;
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
