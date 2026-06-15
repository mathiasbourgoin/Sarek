(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Sarek PPX - Core Primitives
 *
 * This module defines semantic properties for GPU primitives that are known
 * at compile time by the PPX. This separates semantic analysis (variance,
 * convergence, purity) from device implementations (resolved at JIT time).
 *
 * Core primitives are the foundation of the GPU execution model and cannot
 * be redefined by user libraries.
 ******************************************************************************)

(** Variance levels in the GPU execution model. Forms a lattice: Uniform ≤
    BlockVarying ≤ WarpVarying ≤ ThreadVarying *)
type variance =
  | Uniform  (** Same value for all threads in grid *)
  | BlockVarying  (** Uniform within block, varies between blocks *)
  | WarpVarying  (** Uniform within warp, varies between warps *)
  | ThreadVarying  (** Varies per thread *)

(** Convergence requirements for synchronization primitives *)
type convergence =
  | NoEffect  (** Does not affect convergence *)
  | ConvergencePoint  (** All workgroup threads must reach together *)
  | WarpConvergence  (** All warp threads must reach together *)

(** Purity classification for optimization *)
type purity =
  | Pure  (** No side effects, can CSE/DCE *)
  | Impure  (** Has side effects *)
  | Atomic  (** Atomic memory operation *)

(** A core primitive definition with compile-time semantics *)
type primitive = {
  name : string;
  typ : Sarek_types.typ;
  variance : variance;
  convergence : convergence;
  purity : purity;
  category : string;  (** For documentation/grouping *)
}

(** All registered core primitives *)
val primitives : primitive list

(** Lookup a primitive by name *)
val find : string -> primitive option

val find_exn : string -> primitive

(** Predicates *)
val is_core_primitive : string -> bool

val is_thread_varying : string -> bool

(** Check if a primitive has warp-level or finer variance (WarpVarying or
    ThreadVarying) *)
val is_warp_varying : string -> bool

val is_convergence_point : string -> bool

(** Check if a primitive requires warp-level convergence *)
val is_warp_convergence_point : string -> bool

(** Check if a primitive requires any convergence (block or warp level) *)
val requires_convergence : string -> bool

val is_pure : string -> bool

(** Check if a primitive is an atomic memory operation *)
val is_atomic : string -> bool

(** Get variance of a named primitive *)
val variance_of : string -> variance option

(** Variance lattice operations *)
val join_variance : variance -> variance -> variance

val variance_leq : variance -> variance -> bool

(** Get all primitives in a category *)
val primitives_in_category : string -> primitive list

(** Pretty printing *)
val pp_variance : Format.formatter -> variance -> unit

val pp_convergence : Format.formatter -> convergence -> unit

val pp_purity : Format.formatter -> purity -> unit

val pp_primitive : Format.formatter -> primitive -> unit
