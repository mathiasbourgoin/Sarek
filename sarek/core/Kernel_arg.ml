(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Kernel_arg - Type-safe kernel arguments
 *
 * Provides a GADT for kernel arguments. Backends consume
 * these via pattern matching or the fold interface.
 ******************************************************************************)

(** Kernel argument GADT - each variant carries its typed value *)
type t =
  | Vec : ('a, 'b) Vector.t -> t
      (** V2 Vector - carries full vector for Direct backends *)
  | Int : int -> t  (** Integer scalar (converted to int32) *)
  | Int32 : int32 -> t  (** 32-bit integer scalar *)
  | Int64 : int64 -> t  (** 64-bit integer scalar *)
  | Float32 : float -> t  (** 32-bit float scalar *)
  | Float64 : float -> t  (** 64-bit float scalar *)

(** Existential wrapper for vectors with element type info *)
type any_vec = AnyVec : ('a, 'b) Vector.t * ('a, 'b) Vector.kind -> any_vec

(** Extract vector if argument is a Vec, with type info *)
let as_vector (arg : t) : any_vec option =
  match arg with Vec v -> Some (AnyVec (v, Vector.kind v)) | _ -> None

(** Get vector length if argument is a Vec *)
let vector_length (arg : t) : int option =
  match arg with Vec v -> Some (Vector.length v) | _ -> None

(** Get scalar as int32 if possible *)
let as_int32 (arg : t) : int32 option =
  match arg with
  | Int n -> Some (Int32.of_int n)
  | Int32 n -> Some n
  | _ -> None

(** Get scalar as int64 if possible *)
let as_int64 (arg : t) : int64 option =
  match arg with
  | Int n -> Some (Int64.of_int n)
  | Int32 n -> Some (Int64.of_int32 n)
  | Int64 n -> Some n
  | _ -> None

(** Get scalar as float if possible *)
let as_float (arg : t) : float option =
  match arg with Float32 f | Float64 f -> Some f | _ -> None

(** Fold over kernel arguments with typed handlers. This is the main way
    backends consume args in a type-safe manner.

    For JIT backends, the vector handler receives the vector and should bind
    (buffer_ptr, length) to the kernel args.

    For Direct backends, the vector handler can access the vector directly. *)
type 'acc folder = {
  on_vec : 'a 'b. ('a, 'b) Vector.t -> 'acc -> 'acc;
  on_int : int -> 'acc -> 'acc;
  on_int32 : int32 -> 'acc -> 'acc;
  on_int64 : int64 -> 'acc -> 'acc;
  on_float32 : float -> 'acc -> 'acc;
  on_float64 : float -> 'acc -> 'acc;
}

let fold (f : 'acc folder) (args : t list) (init : 'acc) : 'acc =
  List.fold_left
    (fun acc arg ->
      match arg with
      | Vec v -> f.on_vec v acc
      | Int n -> f.on_int n acc
      | Int32 n -> f.on_int32 n acc
      | Int64 n -> f.on_int64 n acc
      | Float32 x -> f.on_float32 x acc
      | Float64 x -> f.on_float64 x acc)
    init
    args

(** Iterate over args with side effects *)
type iterator = {
  iter_vec : 'a 'b. ('a, 'b) Vector.t -> unit;
  iter_int : int -> unit;
  iter_int32 : int32 -> unit;
  iter_int64 : int64 -> unit;
  iter_float32 : float -> unit;
  iter_float64 : float -> unit;
}

let iter (f : iterator) (args : t list) : unit =
  List.iter
    (fun arg ->
      match arg with
      | Vec v -> f.iter_vec v
      | Int n -> f.iter_int n
      | Int32 n -> f.iter_int32 n
      | Int64 n -> f.iter_int64 n
      | Float32 x -> f.iter_float32 x
      | Float64 x -> f.iter_float64 x)
    args

(** Map args to a list of values using typed handlers *)
type 'a mapper = {
  map_vec : 'b 'c. ('b, 'c) Vector.t -> 'a;
  map_int : int -> 'a;
  map_int32 : int32 -> 'a;
  map_int64 : int64 -> 'a;
  map_float32 : float -> 'a;
  map_float64 : float -> 'a;
}

let map (f : 'a mapper) (args : t list) : 'a list =
  List.map
    (fun arg ->
      match arg with
      | Vec v -> f.map_vec v
      | Int n -> f.map_int n
      | Int32 n -> f.map_int32 n
      | Int64 n -> f.map_int64 n
      | Float32 x -> f.map_float32 x
      | Float64 x -> f.map_float64 x)
    args
