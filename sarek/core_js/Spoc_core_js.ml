(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Spoc_core_js — browser instantiation of the FFI-free numeric vector core.
 *
 * Instantiates [Spoc_core_base.Make(Js_ops)], exposing the full browser-safe
 * numeric Vector API: create, to_bigarray, get, set, to_array, length, kind,
 * elem_size, kind_name, and scalar-kind convenience values.
 *
 * Profiling and native clock injection are skipped — they are native-only.
 ******************************************************************************)

include Spoc_core_base.Make (Js_ops)

(** Convenience: get the i-th element of a scalar vector via Bigarray. *)
let get (type a b) (vec : (a, b) t) (i : int) : a =
  Bigarray.Array1.get (to_bigarray vec) i

(** Convenience: set the i-th element of a scalar vector via Bigarray. *)
let set (type a b) (vec : (a, b) t) (i : int) (v : a) : unit =
  Bigarray.Array1.set (to_bigarray vec) i v

(** Convert a scalar vector to an OCaml array. *)
let to_array (type a b) (vec : (a, b) t) : a array =
  let ba = to_bigarray vec in
  Array.init vec.length (fun i -> Bigarray.Array1.get ba i)
