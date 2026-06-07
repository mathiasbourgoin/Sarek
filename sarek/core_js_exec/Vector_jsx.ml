(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Vector_jsx — jsoo Execute instantiation of the FFI-free numeric vector core.
 *
 * Instantiates Spoc_core_base.Make(Jsoo_exec_ops) and adds the accessors
 * that Execute calls beyond what Make exposes: location, get, set, length,
 * kind, kind_name, elem_size, type_id, vector_type_id, get_buffer, DEVICE_BUFFER,
 * device_buffer, custom_to_bytes, custom_of_bytes, kernel_set.
 *
 * No ctypes, no unix.
 ******************************************************************************)

include Spoc_core_base.Make (Jsoo_exec_ops)

(** DEVICE_BUFFER module type: the FFI-free buffer contract *)
module type DEVICE_BUFFER = Memory_jsx.BUFFER

(** Packed first-class module of a DEVICE_BUFFER *)
type device_buffer = (module DEVICE_BUFFER)

(** Location accessor — Execute matches on this *)
let location (v : ('a, 'b) t) : location = v.location

(** Length accessor *)
let length (v : ('a, 'b) t) : int = v.length

(** Kind accessor *)
let kind (v : ('a, 'b) t) : ('a, 'b) kind = v.kind

(** Get element — auto-syncs via ensure_cpu_sync *)
let get : type a b. (a, b) t -> int -> a =
 fun vec idx ->
  ensure_cpu_sync vec ;
  match vec.host with
  | Bigarray_storage ba -> Bigarray.Array1.get ba idx
  | Custom_storage {ptr; custom; _} -> custom.get ptr idx

(** Set element *)
let set : type a b. (a, b) t -> int -> a -> unit =
 fun vec idx value ->
  (match vec.host with
  | Bigarray_storage ba -> Bigarray.Array1.set ba idx value
  | Custom_storage {ptr; custom; _} -> custom.set ptr idx value) ;
  match vec.location with
  | Both d -> vec.location <- Stale_GPU d
  | GPU d -> vec.location <- Stale_GPU d
  | CPU | Stale_CPU _ | Stale_GPU _ -> ()

(** Kernel-safe set: no bounds check, no location update *)
let kernel_set : type a b. (a, b) t -> int -> a -> unit =
 fun vec idx value ->
  match vec.host with
  | Bigarray_storage ba -> Bigarray.Array1.unsafe_set ba idx value
  | Custom_storage {ptr; custom; _} -> custom.set ptr idx value

(** get_buffer — look up device buffer for a given device *)
let get_buffer (type a b) (vec : (a, b) t) (dev : Jsoo_exec_ops.device_t) :
    device_buffer option =
  Hashtbl.find_opt vec.device_buffers (Jsoo_exec_ops.device_id dev)
