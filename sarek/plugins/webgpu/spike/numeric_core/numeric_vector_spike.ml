(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * SPIKE: FFI-free numeric vector core — de-risking proof
 *
 * PURPOSE: Prove that SPOC's Bigarray-backed numeric vector core can compile
 * FFI-free to both native/bytecode AND js_of_ocaml.  Not for production use.
 *
 * DEPS: stdlib only (Bigarray is in-stdlib; jsoo provides it via typed arrays).
 * NO ctypes, NO unix, NO spoc_core.
 *
 * Key replacements demonstrated:
 *   - elem_size_of_kind  : pure lookup replacing Memory.ml:61
 *                          Ctypes_static.sizeof (typ_of_bigarray_kind kind)
 *   - now                : injectable clock ref replacing Profiling's
 *                          Unix.gettimeofday
 ******************************************************************************)

(** {1 Kind GADT}
    Mirrors Vector_types.scalar_kind but references only Bigarray, no Ctypes. *)
type (_, _) kind =
  | Float32 : (float, Bigarray.float32_elt) kind
  | Float64 : (float, Bigarray.float64_elt) kind
  | Int32 : (int32, Bigarray.int32_elt) kind
  | Int64 : (int64, Bigarray.int64_elt) kind
  | Int8u : (int, Bigarray.int8_unsigned_elt) kind
  | Complex32 : (Complex.t, Bigarray.complex32_elt) kind

(** Convert kind to Bigarray.kind *)
let to_bigarray_kind : type a b. (a, b) kind -> (a, b) Bigarray.kind = function
  | Float32 -> Bigarray.Float32
  | Float64 -> Bigarray.Float64
  | Int32 -> Bigarray.Int32
  | Int64 -> Bigarray.Int64
  | Int8u -> Bigarray.Int8_unsigned
  | Complex32 -> Bigarray.Complex32

(** Pure byte-size lookup — replaces Memory.ml:61
    [Ctypes_static.sizeof (Ctypes.typ_of_bigarray_kind kind)]. No FFI required:
    sizes are fixed by the IEEE / ABI spec. *)
let elem_size_of_kind : type a b. (a, b) kind -> int = function
  | Float32 -> 4
  | Float64 -> 8
  | Int32 -> 4
  | Int64 -> 8
  | Int8u -> 1
  | Complex32 -> 8

(** {1 Injectable clock}
    Native build leaves this as [Sys.time]; a jsoo bootstrap can swap it for
    [Js.to_float (new%js Js.date_now)##valueOf] before the first use. *)
let now : (unit -> float) ref = ref Sys.time

(** {1 Vector type} *)
type ('a, 'b) t = {
  ba : ('a, 'b, Bigarray.c_layout) Bigarray.Array1.t;
  kind : ('a, 'b) kind;
  elem_size : int;
}

(** {1 Core operations} *)

(** [create kind n] allocates an uninitialised vector of [n] elements. *)
let create : type a b. (a, b) kind -> int -> (a, b) t =
 fun kind n ->
  let ba = Bigarray.Array1.create (to_bigarray_kind kind) Bigarray.c_layout n in
  {ba; kind; elem_size = elem_size_of_kind kind}

(** [length v] returns the number of elements. *)
let length (v : ('a, 'b) t) : int = Bigarray.Array1.dim v.ba

(** [get v i] reads element [i]. Raises [Invalid_argument] on out-of-bounds. *)
let get : type a b. (a, b) t -> int -> a =
 fun v i ->
  if i < 0 || i >= Bigarray.Array1.dim v.ba then
    invalid_arg
      (Printf.sprintf
         "Numeric_vector_spike.get: index %d out of bounds [0,%d)"
         i
         (Bigarray.Array1.dim v.ba)) ;
  Bigarray.Array1.get v.ba i

(** [set v i x] writes [x] to element [i]. Raises [Invalid_argument] on
    out-of-bounds. *)
let set : type a b. (a, b) t -> int -> a -> unit =
 fun v i x ->
  if i < 0 || i >= Bigarray.Array1.dim v.ba then
    invalid_arg
      (Printf.sprintf
         "Numeric_vector_spike.set: index %d out of bounds [0,%d)"
         i
         (Bigarray.Array1.dim v.ba)) ;
  Bigarray.Array1.set v.ba i x

(** [to_array v] copies the vector contents to a plain OCaml array. *)
let to_array : type a b. (a, b) t -> a array =
 fun v ->
  let n = Bigarray.Array1.dim v.ba in
  if n = 0 then [||]
  else
    let arr = Array.make n (Bigarray.Array1.get v.ba 0) in
    for i = 1 to n - 1 do
      arr.(i) <- Bigarray.Array1.get v.ba i
    done ;
    arr

(** [kind v] returns the element kind. *)
let kind (v : ('a, 'b) t) : ('a, 'b) kind = v.kind

(** [elem_size v] returns the byte size of each element. *)
let elem_size (v : ('a, 'b) t) : int = v.elem_size
