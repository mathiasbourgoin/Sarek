(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Sarek Runtime - High-Level Vector Abstraction
 *
 * Provides a unified Vector type that tracks data location (CPU/GPU) and
 * supports automatic synchronization. Replaces the old SPOC Vector module.
 *
 * Supports two storage modes:
 * - Bigarray: Standard numeric types (Float32, Float64, Int32, etc.)
 * - Custom: User-defined ctypes structures with custom get/set
 *
 * Design principles:
 * - Type-safe: Uses GADTs and existentials
 * - Location-aware: Tracks where data resides (CPU, GPU, or both)
 * - Lazy allocation: GPU memory allocated on first device use
 * - Auto-sync: Element access triggers sync when needed
 ******************************************************************************)

include Vector_types

(** Wrap an existing ctypes pointer as a custom-storage vector (shares memory). *)
let of_ctypes_ptr c ptr len = of_raw_handle c (Ctypes.raw_address_of_ptr ptr) len

(** {1 Accessors} *)

let length (vec : ('a, 'b) t) : int = vec.length

let kind (vec : ('a, 'b) t) : ('a, 'b) kind = vec.kind

let location (vec : ('a, 'b) t) : location = vec.location

let id (vec : ('a, 'b) t) : int = vec.id

(** Get device if vector has GPU data *)
let device (vec : ('a, 'b) t) : Device.t option =
  match vec.location with
  | CPU -> None
  | GPU d | Both d | Stale_CPU d | Stale_GPU d -> Some d

(** {1 Raw Data Access} *)

(** Get underlying Bigarray (only for scalar vectors) *)
let to_bigarray : type a b.
    (a, b) t -> (a, b, Bigarray.c_layout) Bigarray.Array1.t =
 fun vec ->
  match vec.host with
  | Bigarray_storage ba -> ba
  | Custom_storage _ -> invalid_arg "to_bigarray: vector uses custom storage"

(** {1 Sync and host pointer helpers (delegated)} *)

type sync_callback = Vector_transfer.sync_callback = {
  sync : 'a 'b. ('a, 'b) t -> bool;
}

let to_ctypes_ptr = Vector_transfer.to_ctypes_ptr [@@warning "-32"]

let host_ptr = Vector_transfer.host_ptr [@@warning "-32"]

let register_sync_callback = Vector_transfer.register_sync_callback

let ensure_cpu_sync = Vector_transfer.ensure_cpu_sync

(** {1 Element Access} *)

(** Get element (works for both storage types). Auto-syncs from GPU if location
    is Stale_CPU and auto_sync is enabled. *)
let get : type a b. (a, b) t -> int -> a =
 fun vec idx ->
  if idx < 0 || idx >= vec.length then
    invalid_arg
      (Printf.sprintf
         "Vector.get: index %d out of bounds [0, %d)"
         idx
         vec.length) ;
  ensure_cpu_sync vec ;
  match vec.host with
  | Bigarray_storage ba -> Bigarray.Array1.get ba idx
  | Custom_storage {ptr; custom; _} -> custom.get ptr idx

(** Set element (works for both storage types) *)
let set : type a b. (a, b) t -> int -> a -> unit =
 fun vec idx value ->
  if idx < 0 || idx >= vec.length then
    invalid_arg
      (Printf.sprintf
         "Vector.set: index %d out of bounds [0, %d)"
         idx
         vec.length) ;
  (match vec.host with
  | Bigarray_storage ba -> Bigarray.Array1.set ba idx value
  | Custom_storage {ptr; custom; _} -> custom.set ptr idx value) ;
  (* Mark GPU as stale if we had synced data *)
  match vec.location with
  | Both d -> vec.location <- Stale_GPU d
  | GPU d -> vec.location <- Stale_GPU d
  | CPU | Stale_CPU _ | Stale_GPU _ -> ()

(** Indexing operators *)
let ( .%[] ) = get

let ( .%[]<- ) = set

(** {1 Unsafe Access (no bounds check)} *)

let unsafe_get : type a b. (a, b) t -> int -> a =
 fun vec idx ->
  match vec.host with
  | Bigarray_storage ba -> Bigarray.Array1.unsafe_get ba idx
  | Custom_storage {ptr; custom; _} -> custom.get ptr idx

let unsafe_set : type a b. (a, b) t -> int -> a -> unit =
 fun vec idx value ->
  (match vec.host with
  | Bigarray_storage ba -> Bigarray.Array1.unsafe_set ba idx value
  | Custom_storage {ptr; custom; _} -> custom.set ptr idx value) ;
  match vec.location with
  | Both d -> vec.location <- Stale_GPU d
  | GPU d -> vec.location <- Stale_GPU d
  | CPU | Stale_CPU _ | Stale_GPU _ -> ()

(** Kernel-safe set: no bounds check, no location update. Use this in parallel
    kernel execution where:
    - Bounds are guaranteed by kernel logic
    - Location tracking is not needed (data stays on same device)
    - Multiple threads may write to different indices concurrently *)
let kernel_set : type a b. (a, b) t -> int -> a -> unit =
 fun vec idx value ->
  match vec.host with
  | Bigarray_storage ba -> Bigarray.Array1.unsafe_set ba idx value
  | Custom_storage {ptr; custom; _} -> custom.set ptr idx value

(** {1 Auto-sync Control} *)

let set_auto_sync (vec : ('a, 'b) t) (enabled : bool) : unit =
  vec.auto_sync <- enabled

let auto_sync (vec : ('a, 'b) t) : bool = vec.auto_sync

(** {1 Location Queries} *)

let is_on_cpu (vec : ('a, 'b) t) : bool =
  match vec.location with
  | CPU | Both _ | Stale_GPU _ -> true
  | GPU _ | Stale_CPU _ -> false

let is_on_gpu (vec : ('a, 'b) t) : bool =
  match vec.location with
  | GPU _ | Both _ | Stale_CPU _ -> true
  | CPU | Stale_GPU _ -> false

let is_synced (vec : ('a, 'b) t) : bool =
  match vec.location with Both _ -> true | _ -> false

let needs_gpu_update (vec : ('a, 'b) t) : bool =
  match vec.location with Stale_GPU _ -> true | _ -> false

let needs_cpu_update (vec : ('a, 'b) t) : bool =
  match vec.location with Stale_CPU _ | GPU _ -> true | _ -> false

(** {1 Device Buffer Management} *)

(** [has_buffer] and [get_buffer] are provided by Vector_types via the functor. *)

(** {1 Pretty Printing} *)

let location_to_string : location -> string = function
  | CPU -> "CPU"
  | GPU d -> Printf.sprintf "GPU(%s)" d.name
  | Both d -> Printf.sprintf "Both(%s)" d.name
  | Stale_CPU d -> Printf.sprintf "Stale_CPU(%s)" d.name
  | Stale_GPU d -> Printf.sprintf "Stale_GPU(%s)" d.name

let to_string (vec : ('a, 'b) t) : string =
  Printf.sprintf
    "Vector#%d<%s>[%d] @ %s"
    vec.id
    (kind_name vec.kind)
    vec.length
    (location_to_string vec.location)

(** {1 Convenience Constructors for Scalar Types} *)

let float32 = Scalar Float32

let float64 = Scalar Float64

let int32 = Scalar Int32

let int64 = Scalar Int64

let char = Scalar Char

let complex32 = Scalar Complex32

let create_float32 ?dev n = create float32 ?dev n

let create_float64 ?dev n = create float64 ?dev n

let create_int32 ?dev n = create int32 ?dev n

let create_int64 ?dev n = create int64 ?dev n

(** {1 Initialization Helpers} *)

(** Fill vector with a value *)
let fill : type a b. (a, b) t -> a -> unit =
 fun vec value ->
  (match vec.host with
  | Bigarray_storage ba -> Bigarray.Array1.fill ba value
  | Custom_storage {ptr; custom; length; _} ->
      for i = 0 to length - 1 do
        custom.set ptr i value
      done) ;
  match vec.location with
  | Both d -> vec.location <- Stale_GPU d
  | GPU d -> vec.location <- Stale_GPU d
  | _ -> ()

(** Initialize with a function *)
let init : type a b. (a, b) kind -> int -> (int -> a) -> (a, b) t =
 fun kind length f ->
  let vec = create kind length in
  for i = 0 to length - 1 do
    unsafe_set vec i (f i)
  done ;
  vec

(** Copy vector (CPU data only, auto-syncs source if needed) *)
let copy : type a b. (a, b) t -> (a, b) t =
 fun vec ->
  ensure_cpu_sync vec ;
  copy_host_only vec

(** {1 Subvector Support} *)


(** Create a subvector that shares CPU memory with parent.
    @param vec Parent vector
    @param start Starting index in parent
    @param len Length of subvector
    @param ok_range Elements safe to read (default: len)
    @param ko_range Elements to avoid writing (default: 0) *)
let sub_vector (type a b) (vec : (a, b) t) ~(start : int) ~(len : int)
    ?(ok_range : int = len) ?(ko_range : int = 0) () : (a, b) t =
  sub_vector vec ~start ~len ~ok_range ~ko_range

(** {1 Multi-GPU Helpers} *)

(** Partition a vector across multiple devices. Creates subvectors, one per
    device, that together cover the full vector. *)
let partition (type a b) (vec : (a, b) t) (devices : Device.t array) :
    (a, b) t array =
  let subs = partition_host vec devices in
  Array.iteri
    (fun i sub ->
      if i < Array.length devices then sub.location <- Stale_GPU devices.(i))
    subs ;
  subs

(** Gather subvectors back to parent (sync all to CPU). Assumes subvectors were
    created by partition and don't overlap. *)
let gather (subs : (_, _) t array) : unit =
  Array.iter
    (fun sub ->
      match sub.location with
      | GPU _ | Stale_CPU _ -> (
          (* Need to sync from GPU *)
          match get_sub_meta sub with
          | Some _meta ->
              (* Transfer handled by Transfer module *)
              ()
          | None -> ())
      | _ -> ())
    subs

(** {1 Phase 6: Vector Utilities} *)

(** {2 Iteration} *)

(** Iterate over all elements (CPU-side, auto-syncs if needed) *)
let iter : type a b. (a -> unit) -> (a, b) t -> unit =
 fun f vec ->
  ensure_cpu_sync vec ;
  for i = 0 to vec.length - 1 do
    f (unsafe_get vec i)
  done

(** Iterate with index *)
let iteri : type a b. (int -> a -> unit) -> (a, b) t -> unit =
 fun f vec ->
  ensure_cpu_sync vec ;
  for i = 0 to vec.length - 1 do
    f i (unsafe_get vec i)
  done

(** {2 Mapping} *)

(** Map function over vector, creating new vector with given kind *)
let map : type a b c d. (a -> c) -> (c, d) kind -> (a, b) t -> (c, d) t =
 fun f target_kind vec ->
  ensure_cpu_sync vec ;
  let result = create target_kind vec.length in
  for i = 0 to vec.length - 1 do
    unsafe_set result i (f (unsafe_get vec i))
  done ;
  result

(** Map with index *)
let mapi : type a b c d. (int -> a -> c) -> (c, d) kind -> (a, b) t -> (c, d) t
    =
 fun f target_kind vec ->
  ensure_cpu_sync vec ;
  let result = create target_kind vec.length in
  for i = 0 to vec.length - 1 do
    unsafe_set result i (f i (unsafe_get vec i))
  done ;
  result

(** In-place map (same type) *)
let map_inplace : type a b. (a -> a) -> (a, b) t -> unit =
 fun f vec ->
  ensure_cpu_sync vec ;
  for i = 0 to vec.length - 1 do
    unsafe_set vec i (f (unsafe_get vec i))
  done

(** {2 Folding} *)

(** Fold left *)
let fold_left : type a b acc. (acc -> a -> acc) -> acc -> (a, b) t -> acc =
 fun f init vec ->
  ensure_cpu_sync vec ;
  let acc = ref init in
  for i = 0 to vec.length - 1 do
    acc := f !acc (unsafe_get vec i)
  done ;
  !acc

(** Fold right *)
let fold_right : type a b acc. (a -> acc -> acc) -> (a, b) t -> acc -> acc =
 fun f vec init ->
  ensure_cpu_sync vec ;
  let acc = ref init in
  for i = vec.length - 1 downto 0 do
    acc := f (unsafe_get vec i) !acc
  done ;
  !acc

(** {2 Predicates} *)

(** Check if all elements satisfy predicate *)
let for_all : type a b. (a -> bool) -> (a, b) t -> bool =
 fun p vec ->
  ensure_cpu_sync vec ;
  let result = ref true in
  let i = ref 0 in
  while !result && !i < vec.length do
    result := p (unsafe_get vec !i) ;
    incr i
  done ;
  !result

(** Check if any element satisfies predicate *)
let exists : type a b. (a -> bool) -> (a, b) t -> bool =
 fun p vec ->
  ensure_cpu_sync vec ;
  let result = ref false in
  let i = ref 0 in
  while (not !result) && !i < vec.length do
    result := p (unsafe_get vec !i) ;
    incr i
  done ;
  !result

(** Find first element satisfying predicate *)
let find : type a b. (a -> bool) -> (a, b) t -> a option =
 fun p vec ->
  ensure_cpu_sync vec ;
  let result = ref None in
  let i = ref 0 in
  while Option.is_none !result && !i < vec.length do
    let v = unsafe_get vec !i in
    if p v then result := Some v ;
    incr i
  done ;
  !result

(** Find index of first element satisfying predicate *)
let find_index : type a b. (a -> bool) -> (a, b) t -> int option =
 fun p vec ->
  ensure_cpu_sync vec ;
  let result = ref None in
  let i = ref 0 in
  while Option.is_none !result && !i < vec.length do
    if p (unsafe_get vec !i) then result := Some !i ;
    incr i
  done ;
  !result

(** {2 Aggregation} *)

(** Sum elements (requires + operation via fold) *)
let sum (type a b) ~(zero : a) ~(add : a -> a -> a) (vec : (a, b) t) : a =
  fold_left add zero vec (* fold_left already calls ensure_cpu_sync *)

(** Find minimum element *)
let min_elt (type a b) ~(compare : a -> a -> int) (vec : (a, b) t) : a option =
  if vec.length = 0 then None
  else begin
    ensure_cpu_sync vec ;
    let m = ref (unsafe_get vec 0) in
    for i = 1 to vec.length - 1 do
      let v = unsafe_get vec i in
      if compare v !m < 0 then m := v
    done ;
    Some !m
  end

(** Find maximum element *)
let max_elt (type a b) ~(compare : a -> a -> int) (vec : (a, b) t) : a option =
  if vec.length = 0 then None
  else begin
    ensure_cpu_sync vec ;
    let m = ref (unsafe_get vec 0) in
    for i = 1 to vec.length - 1 do
      let v = unsafe_get vec i in
      if compare v !m > 0 then m := v
    done ;
    Some !m
  end

(** {2 Conversion} *)

(** Convert to OCaml list *)
let to_list : type a b. (a, b) t -> a list =
 fun vec ->
  fold_right (fun x acc -> x :: acc) vec [] (* fold_right already syncs *)

(** Convert to OCaml array *)
let to_array : type a b. (a, b) t -> a array =
 fun vec ->
  if vec.length = 0 then [||]
  else begin
    ensure_cpu_sync vec ;
    let arr = Array.make vec.length (unsafe_get vec 0) in
    for i = 1 to vec.length - 1 do
      arr.(i) <- unsafe_get vec i
    done ;
    arr
  end

(** {2 Blitting} *)

(** Copy elements from one vector to another *)
let blit : type a b.
    src:(a, b) t ->
    src_off:int ->
    dst:(a, b) t ->
    dst_off:int ->
    len:int ->
    unit =
 fun ~src ~src_off ~dst ~dst_off ~len ->
  if src_off < 0 || src_off + len > src.length then
    invalid_arg "Vector.blit: source range out of bounds" ;
  if dst_off < 0 || dst_off + len > dst.length then
    invalid_arg "Vector.blit: destination range out of bounds" ;
  ensure_cpu_sync src ;
  for i = 0 to len - 1 do
    unsafe_set dst (dst_off + i) (unsafe_get src (src_off + i))
  done
