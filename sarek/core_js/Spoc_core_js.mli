(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Spoc_core_js — public interface for the browser numeric vector core.
 *
 * This module is [Spoc_core_base.Make(Js_ops)] with convenience accessors
 * added on top. It is safe to compile to JavaScript via js_of_ocaml.
 *
 * Custom-type and device-transfer operations are NOT available in this build;
 * they will be filled in by the WebGPU runtime layer.
 ******************************************************************************)

(** All types, kind helpers, type-id helpers, creation, accessors, copy,
    slicing, list/array creation, and sync-callback API from the base functor.
    Refer to {!Spoc_core_base.Make} for full documentation. *)
include module type of Spoc_core_base.Make (Js_ops)

(** [get vec i] returns the [i]-th element of a scalar vector. Raises
    [Invalid_argument] if [vec] uses custom storage (not reachable on the pure
    Bigarray path). *)
val get : ('a, 'b) t -> int -> 'a

(** [set vec i v] sets the [i]-th element of a scalar vector. Raises
    [Invalid_argument] if [vec] uses custom storage. *)
val set : ('a, 'b) t -> int -> 'a -> unit

(** [to_array vec] converts a scalar vector to an OCaml array. Raises
    [Invalid_argument] if [vec] uses custom storage. *)
val to_array : ('a, 'b) t -> 'a array
