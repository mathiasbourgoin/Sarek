(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Vector transfer helpers (split from Vector.ml)
 *
 * This module is intended to hold transfer-related helpers to reduce the size
 * of Vector.ml. Currently it only re-exports selected helpers used by other
 * modules; future work can move more logic here.
 ******************************************************************************)

(** Raw host pointers *)
let to_ctypes_ptr : type a b. (a, b) Vector_types.t -> unit Ctypes.ptr =
 fun vec ->
  match vec.host with
  | Vector_types.Bigarray_storage ba ->
      Ctypes.(bigarray_start array1 ba |> to_voidp)
  | Vector_types.Custom_storage {ptr; _} -> ptr

let host_ptr : type a b. (a, b) Vector_types.t -> nativeint =
 fun vec ->
  match vec.host with
  | Vector_types.Bigarray_storage ba ->
      Ctypes.(raw_address_of_ptr (bigarray_start array1 ba |> to_voidp))
  | Vector_types.Custom_storage {ptr; _} -> Ctypes.raw_address_of_ptr ptr

(** Convert bigarray to raw pointer and byte size *)
let bigarray_to_ptr (type a b)
    (ba : (a, b, Bigarray.c_layout) Bigarray.Array1.t) (elem_size : int) :
    unit Ctypes.ptr * int =
  let len = Bigarray.Array1.dim ba in
  let byte_size = len * elem_size in
  let ptr = Ctypes.bigarray_start Ctypes.array1 ba in
  (Ctypes.to_voidp ptr, byte_size)

(** Auto-sync callback registration *)
type sync_callback = {sync : 'a 'b. ('a, 'b) Vector_types.t -> bool}

let sync_to_cpu_callback : sync_callback option ref = ref None

let register_sync_callback (cb : sync_callback) : unit =
  sync_to_cpu_callback := Some cb

let ensure_cpu_sync (type a b) (vec : (a, b) Vector_types.t) : unit =
  if vec.Vector_types.auto_sync then
    match vec.Vector_types.location with
    | Vector_types.Stale_CPU _ -> (
        match !sync_to_cpu_callback with
        | Some cb -> ignore (cb.sync vec)
        | None -> ())
    | _ -> ()
