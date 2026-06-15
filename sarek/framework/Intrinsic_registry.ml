(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Intrinsic Registry - Backend-specific Intrinsic Management
 *
 * Provides registration and lookup for intrinsics (built-in functions) that
 * have backend-specific implementations. Each backend (CUDA, OpenCL, Native)
 * registers its own intrinsics with appropriate codegen callbacks.
 *
 * Example intrinsics:
 * - thread_id_x: CUDA → threadIdx.x, OpenCL → get_local_id(0)
 * - block_barrier: CUDA → __syncthreads(), OpenCL → barrier(CLK_LOCAL_MEM_FENCE)
 * - atomic_add: CUDA → atomicAdd, OpenCL → atomic_add
 ******************************************************************************)

open Spoc_framework.Framework_sig

(** Generic intrinsic implementation record. This is the concrete type used by
    backends; the module type INTRINSIC_REGISTRY uses an abstract type to allow
    flexibility. *)
type intrinsic_impl = {
  intr_name : string;
  intr_codegen : string;  (** Code to generate (simple string for now) *)
  intr_convergence : convergence;
}

(** Create an intrinsic registry for a specific backend. Each backend should
    create its own registry instance. *)
module Make () : INTRINSIC_REGISTRY with type intrinsic_impl = intrinsic_impl =
struct
  type nonrec intrinsic_impl = intrinsic_impl

  let table : (string, intrinsic_impl) Hashtbl.t = Hashtbl.create 64

  let register name impl = Hashtbl.replace table name impl

  let find name = Hashtbl.find_opt table name

  let list_all () =
    Hashtbl.fold (fun name _ acc -> name :: acc) table [] |> List.sort compare
end

(** Global registry that combines all backend registries. Used for intrinsic
    lookup when the backend is not yet known. *)
module Global = struct
  (** Registry entry with backend association *)
  type entry = {
    backend : string;  (** Backend name: "CUDA", "OpenCL", "Native" *)
    impl : intrinsic_impl;
  }

  let table : (string, entry list) Hashtbl.t = Hashtbl.create 64

  (** Register an intrinsic for a specific backend *)
  let register ~backend name impl =
    let entry = {backend; impl} in
    let existing = Hashtbl.find_opt table name |> Option.value ~default:[] in
    Hashtbl.replace table name (entry :: existing)

  (** Find all implementations of an intrinsic across backends *)
  let find_all name : entry list =
    Hashtbl.find_opt table name |> Option.value ~default:[]

  (** Find implementation for a specific backend *)
  let find ~backend name : intrinsic_impl option =
    find_all name
    |> List.find_opt (fun e -> e.backend = backend)
    |> Option.map (fun e -> e.impl)

  (** List all registered intrinsic names *)
  let list_all () : string list =
    Hashtbl.fold (fun name _ acc -> name :: acc) table [] |> List.sort compare

  (** List all backends that implement a given intrinsic *)
  let backends_for name : string list =
    find_all name |> List.map (fun e -> e.backend) |> List.sort compare
end

(** {1 Standard Intrinsic Definitions} *)

(** Thread indexing intrinsics - must be implemented by all GPU backends *)
module Thread_intrinsics = struct
  let names =
    [
      "thread_id_x";
      "thread_id_y";
      "thread_id_z";
      "block_id_x";
      "block_id_y";
      "block_id_z";
      "block_dim_x";
      "block_dim_y";
      "block_dim_z";
      "grid_dim_x";
      "grid_dim_y";
      "grid_dim_z";
      "global_thread_id";
      "global_size";
    ]

  (** Check if an intrinsic is a thread indexing intrinsic *)
  let is_thread_intrinsic name = List.mem name names
end

(** Synchronization intrinsics *)
module Sync_intrinsics = struct
  let names = ["block_barrier"; "warp_barrier"; "memory_fence"; "thread_fence"]

  let is_sync_intrinsic name = List.mem name names
end

(** Atomic operation intrinsics *)
module Atomic_intrinsics = struct
  let names =
    [
      "atomic_add";
      "atomic_sub";
      "atomic_min";
      "atomic_max";
      "atomic_and";
      "atomic_or";
      "atomic_xor";
      "atomic_exch";
      "atomic_cas";
    ]

  let is_atomic_intrinsic name = List.mem name names
end

(** Math intrinsics - typically map to hardware instructions *)
module Math_intrinsics = struct
  let names =
    [
      (* Trigonometric *)
      "sin";
      "cos";
      "tan";
      "asin";
      "acos";
      "atan";
      "atan2";
      (* Hyperbolic *)
      "sinh";
      "cosh";
      "tanh";
      (* Exponential and logarithmic *)
      "exp";
      "exp2";
      "log";
      "log2";
      "log10";
      (* Power and root *)
      "pow";
      "sqrt";
      "rsqrt";
      "cbrt";
      (* Rounding *)
      "floor";
      "ceil";
      "round";
      "trunc";
      (* Other *)
      "abs";
      "fabs";
      "fma";
      "min";
      "max";
      "clamp";
    ]

  let is_math_intrinsic name = List.mem name names
end

(** {1 Helper Functions} *)

(** Create a simple intrinsic that maps to a direct code string *)
let make_simple_intrinsic ~name ~codegen =
  {intr_name = name; intr_codegen = codegen; intr_convergence = Uniform}

(** Create a synchronization intrinsic *)
let make_sync_intrinsic ~name ~codegen =
  {intr_name = name; intr_codegen = codegen; intr_convergence = Sync}

(** Create a divergent intrinsic (result varies per thread) *)
let make_divergent_intrinsic ~name ~codegen =
  {intr_name = name; intr_codegen = codegen; intr_convergence = Divergent}

(** {1 Intrinsic Validation} *)

(** Check if an intrinsic is safe to call in divergent control flow *)
let is_safe_in_divergent_flow impl =
  match impl.intr_convergence with
  | Uniform | Divergent -> true
  | Sync -> false (* Barriers in divergent flow cause deadlock *)

(** Check if an intrinsic requires all threads to participate *)
let requires_uniform_execution impl = impl.intr_convergence = Sync
