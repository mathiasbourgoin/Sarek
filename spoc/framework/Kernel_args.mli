(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Indexed kernel-argument container.

    Every GPU backend plugin ([Native], [Interpreter], CUDA, Metal, Vulkan,
    OpenCL, ...) exposes a family of [set_arg_*] functions that take an explicit
    argument index [idx] alongside the value being bound. This module is the
    single shared implementation of "store a value at [idx], last write wins,
    then validate the full set before launch" so backends no longer each
    hand-roll a slightly different (and slightly buggy) accumulate-by-call-order
    scheme.

    Values are stored in an indexed slot rather than accumulated in list order,
    so callers may invoke {!set} in any order and with duplicate indices; only
    the most recent call for a given index is kept. *)

(** A store of ['a] values keyed by non-negative argument index. *)
type 'a t

(** [create ()] returns a fresh, empty argument store. *)
val create : unit -> 'a t

(** [set t idx v] stores [v] at [idx], overwriting any value previously stored
    at that index. Last-set-wins on duplicate indices. *)
val set : 'a t -> int -> 'a -> unit

(** [count t] returns the number of distinct indices currently stored. *)
val count : 'a t -> int

(** [to_sorted_list t] returns all stored [(idx, value)] pairs ordered by
    ascending [idx], without any contiguity or count validation. Useful for
    backends (e.g. Vulkan descriptor bindings) that need a subset of arguments
    in index order but do not require the full [0..n-1] contiguity contract
    enforced by {!validate_and_extract} — for example when indices share a
    numbering space with arguments stored in a different container. *)
val to_sorted_list : 'a t -> (int * 'a) list

(** [validate_and_extract t ~expected_count] checks that [t] holds exactly the
    indices [0 .. expected_count - 1] (no gaps, no indices at or beyond
    [expected_count]) and returns the corresponding values as an array ordered
    by index.

    On failure, returns [Error msg] where [msg] names the missing and/or
    unexpected indices, e.g. ["missing indices: [2]; expected 3 args, got 2"] or
    ["unexpected index: [5]; expected contiguous 0..2"]. Callers are expected to
    wrap [msg] into their backend's own structured error (e.g.
    [Kernel_launch_failed]) rather than raise directly from here.

    A negative [expected_count] returns [Error msg] rather than raising
    [Invalid_argument] (which the underlying [Array.init] would otherwise
    raise). *)
val validate_and_extract :
  'a t -> expected_count:int -> ('a array, string) result
