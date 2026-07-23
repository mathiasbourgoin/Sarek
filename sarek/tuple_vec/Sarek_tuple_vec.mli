(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** L13 — host-side [custom_type] builders for tuple-typed vector elements.

    Build the [Spoc_core.Vector.custom_type] needed by [Vector.create_custom]
    for a [('a * 'b) vector] (or 3-tuple), so tuples can be stored/read from the
    host with OCaml tuple literals while the kernel manipulates them on device.
    The byte layout is taken from the shared layout authority so it matches the
    device element layout exactly. Only scalar-primitive components are
    supported in this tier. *)

module Vector = Spoc_core.Vector

(** A scalar tuple component. *)
type 'a component

val float32 : float component

val float64 : float component

val int32 : int32 component

val int64 : int64 component

(** [pair a b] builds the custom type for a [(a, b)] tuple vector element. *)
val pair : 'a component -> 'b component -> ('a * 'b) Vector.custom_type

(** [triple a b c] builds the custom type for an [(a, b, c)] tuple vector
    element. *)
val triple :
  'a component ->
  'b component ->
  'c component ->
  ('a * 'b * 'c) Vector.custom_type

(** [descriptor_by_name name] returns the canonical [custom_type] registered for
    a mangled tuple-shape name (e.g. ["_tup_float32_int32"]) by a previous
    [pair]/[triple] call. Generated Native kernel code uses this to obtain the
    same [Type_id] the host vector carries. Raises if the shape was never built
    on the host. Not normally called directly. *)
val descriptor_by_name : string -> 'a Vector.custom_type
