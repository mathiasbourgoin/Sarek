(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Unit tests for Sarek_core_primitives *)

open Alcotest
open Sarek_core_primitives

(* === Lookup tests === *)

let test_find_thread_idx () =
  let p = find "thread_idx_x" in
  check bool "thread_idx_x exists" true (Option.is_some p)

let test_find_nonexistent () =
  let p = find "nonexistent_primitive" in
  check bool "nonexistent returns None" true (Option.is_none p)

let test_is_core_primitive () =
  check bool "thread_idx_x is core" true (is_core_primitive "thread_idx_x") ;
  check bool "block_barrier is core" true (is_core_primitive "block_barrier") ;
  check bool "sin is core" true (is_core_primitive "sin") ;
  check bool "sin_f64 is core" true (is_core_primitive "sin_f64") ;
  check bool "abs_int32 is core" true (is_core_primitive "abs_int32") ;
  check
    bool
    "memory_fence_block is core"
    true
    (is_core_primitive "memory_fence_block") ;
  check bool "random is not core" false (is_core_primitive "random")

(* === Variance tests === *)

let test_thread_varying () =
  check
    bool
    "thread_idx_x is thread varying"
    true
    (is_thread_varying "thread_idx_x") ;
  check
    bool
    "thread_idx_y is thread varying"
    true
    (is_thread_varying "thread_idx_y") ;
  check
    bool
    "global_thread_id is thread varying"
    true
    (is_thread_varying "global_thread_id") ;
  check
    bool
    "block_dim_x is not thread varying"
    false
    (is_thread_varying "block_dim_x") ;
  check
    bool
    "block_idx_x is not thread varying"
    false
    (is_thread_varying "block_idx_x")

let test_warp_varying () =
  (* ThreadVarying implies WarpVarying *)
  check
    bool
    "thread_idx_x is warp varying"
    true
    (is_warp_varying "thread_idx_x") ;
  (* BlockVarying is NOT warp varying *)
  check
    bool
    "block_idx_x is not warp varying"
    false
    (is_warp_varying "block_idx_x") ;
  (* Uniform is NOT warp varying *)
  check
    bool
    "block_dim_x is not warp varying"
    false
    (is_warp_varying "block_dim_x")

let test_variance_of () =
  check
    (option (of_pp pp_variance))
    "thread_idx_x variance"
    (Some ThreadVarying)
    (variance_of "thread_idx_x") ;
  check
    (option (of_pp pp_variance))
    "block_idx_x variance"
    (Some BlockVarying)
    (variance_of "block_idx_x") ;
  check
    (option (of_pp pp_variance))
    "block_dim_x variance"
    (Some Uniform)
    (variance_of "block_dim_x") ;
  check
    (option (of_pp pp_variance))
    "nonexistent variance"
    None
    (variance_of "nonexistent")

let test_join_variance () =
  check
    (of_pp pp_variance)
    "Uniform + Uniform"
    Uniform
    (join_variance Uniform Uniform) ;
  check
    (of_pp pp_variance)
    "Uniform + ThreadVarying"
    ThreadVarying
    (join_variance Uniform ThreadVarying) ;
  check
    (of_pp pp_variance)
    "ThreadVarying + Uniform"
    ThreadVarying
    (join_variance ThreadVarying Uniform) ;
  check
    (of_pp pp_variance)
    "BlockVarying + BlockVarying"
    BlockVarying
    (join_variance BlockVarying BlockVarying) ;
  check
    (of_pp pp_variance)
    "BlockVarying + ThreadVarying"
    ThreadVarying
    (join_variance BlockVarying ThreadVarying) ;
  check
    (of_pp pp_variance)
    "ThreadVarying + BlockVarying"
    ThreadVarying
    (join_variance ThreadVarying BlockVarying) ;
  (* WarpVarying tests *)
  check
    (of_pp pp_variance)
    "WarpVarying + WarpVarying"
    WarpVarying
    (join_variance WarpVarying WarpVarying) ;
  check
    (of_pp pp_variance)
    "BlockVarying + WarpVarying"
    WarpVarying
    (join_variance BlockVarying WarpVarying) ;
  check
    (of_pp pp_variance)
    "WarpVarying + ThreadVarying"
    ThreadVarying
    (join_variance WarpVarying ThreadVarying) ;
  check
    (of_pp pp_variance)
    "Uniform + WarpVarying"
    WarpVarying
    (join_variance Uniform WarpVarying)

let test_variance_leq () =
  check bool "Uniform <= Uniform" true (variance_leq Uniform Uniform) ;
  check bool "Uniform <= BlockVarying" true (variance_leq Uniform BlockVarying) ;
  check bool "Uniform <= WarpVarying" true (variance_leq Uniform WarpVarying) ;
  check
    bool
    "Uniform <= ThreadVarying"
    true
    (variance_leq Uniform ThreadVarying) ;
  check
    bool
    "BlockVarying <= BlockVarying"
    true
    (variance_leq BlockVarying BlockVarying) ;
  check
    bool
    "BlockVarying <= WarpVarying"
    true
    (variance_leq BlockVarying WarpVarying) ;
  check
    bool
    "BlockVarying <= ThreadVarying"
    true
    (variance_leq BlockVarying ThreadVarying) ;
  check
    bool
    "WarpVarying <= WarpVarying"
    true
    (variance_leq WarpVarying WarpVarying) ;
  check
    bool
    "WarpVarying <= ThreadVarying"
    true
    (variance_leq WarpVarying ThreadVarying) ;
  check
    bool
    "ThreadVarying <= ThreadVarying"
    true
    (variance_leq ThreadVarying ThreadVarying) ;
  check
    bool
    "ThreadVarying <= Uniform"
    false
    (variance_leq ThreadVarying Uniform) ;
  check
    bool
    "WarpVarying <= BlockVarying"
    false
    (variance_leq WarpVarying BlockVarying) ;
  check bool "BlockVarying <= Uniform" false (variance_leq BlockVarying Uniform)

(* === Convergence tests === *)

let test_convergence_point () =
  check
    bool
    "block_barrier is convergence point"
    true
    (is_convergence_point "block_barrier") ;
  check bool "sin is not convergence point" false (is_convergence_point "sin") ;
  check
    bool
    "thread_idx_x is not convergence point"
    false
    (is_convergence_point "thread_idx_x") ;
  (* Warp primitives have WarpConvergence, not ConvergencePoint *)
  check
    bool
    "warp_shuffle is not block convergence point"
    false
    (is_convergence_point "warp_shuffle")

let test_warp_convergence () =
  check
    bool
    "warp_shuffle requires warp convergence"
    true
    (is_warp_convergence_point "warp_shuffle") ;
  check
    bool
    "warp_vote_all requires warp convergence"
    true
    (is_warp_convergence_point "warp_vote_all") ;
  check
    bool
    "warp_ballot requires warp convergence"
    true
    (is_warp_convergence_point "warp_ballot") ;
  check
    bool
    "warp_active_mask does not require convergence"
    false
    (is_warp_convergence_point "warp_active_mask") ;
  check
    bool
    "block_barrier does not require warp convergence"
    false
    (is_warp_convergence_point "block_barrier")

let test_requires_convergence () =
  check
    bool
    "block_barrier requires convergence"
    true
    (requires_convergence "block_barrier") ;
  check
    bool
    "warp_shuffle requires convergence"
    true
    (requires_convergence "warp_shuffle") ;
  check
    bool
    "sin does not require convergence"
    false
    (requires_convergence "sin") ;
  check
    bool
    "warp_active_mask does not require convergence"
    false
    (requires_convergence "warp_active_mask")

(* === Purity tests === *)

let test_pure () =
  check bool "sin is pure" true (is_pure "sin") ;
  check bool "cos is pure" true (is_pure "cos") ;
  check bool "sqrt is pure" true (is_pure "sqrt") ;
  check bool "sin_f64 is pure" true (is_pure "sin_f64") ;
  check bool "abs_int32 is pure" true (is_pure "abs_int32") ;
  check bool "thread_idx_x is pure" true (is_pure "thread_idx_x") ;
  check bool "block_barrier is not pure" false (is_pure "block_barrier") ;
  check
    bool
    "memory_fence_block is not pure"
    false
    (is_pure "memory_fence_block") ;
  check bool "atomic_add_int32 is not pure" false (is_pure "atomic_add_int32")

(* === Atomic tests === *)

let test_atomic () =
  check bool "atomic_add_int32 is atomic" true (is_atomic "atomic_add_int32") ;
  check bool "atomic_sub_int32 is atomic" true (is_atomic "atomic_sub_int32") ;
  check bool "atomic_exch_int32 is atomic" true (is_atomic "atomic_exch_int32") ;
  check bool "atomic_min_int32 is atomic" true (is_atomic "atomic_min_int32") ;
  check bool "atomic_max_int32 is atomic" true (is_atomic "atomic_max_int32") ;
  check bool "atomic_cas_int32 is atomic" true (is_atomic "atomic_cas_int32") ;
  check bool "atomic_and_int32 is atomic" true (is_atomic "atomic_and_int32") ;
  check bool "atomic_or_int32 is atomic" true (is_atomic "atomic_or_int32") ;
  check bool "atomic_xor_int32 is atomic" true (is_atomic "atomic_xor_int32") ;
  check
    bool
    "atomic_add_float32 is atomic"
    true
    (is_atomic "atomic_add_float32") ;
  check bool "atomic_inc_int32 is atomic" true (is_atomic "atomic_inc_int32") ;
  check bool "atomic_dec_int32 is atomic" true (is_atomic "atomic_dec_int32") ;
  check bool "atomic_add_int64 is atomic" true (is_atomic "atomic_add_int64") ;
  check
    bool
    "atomic_add_float64 is atomic"
    true
    (is_atomic "atomic_add_float64") ;
  check bool "sin is not atomic" false (is_atomic "sin") ;
  check bool "block_barrier is not atomic" false (is_atomic "block_barrier")

(* === Category tests === *)

let test_category () =
  let thread_ids = primitives_in_category "thread_id" in
  check
    bool
    "thread_id category has primitives"
    true
    (List.length thread_ids >= 4) ;
  let math = primitives_in_category "math_f32" in
  check bool "math_f32 category has primitives" true (List.length math >= 10) ;
  let math64 = primitives_in_category "math_f64" in
  check bool "math_f64 category has primitives" true (List.length math64 >= 10) ;
  let ints = primitives_in_category "int" in
  check bool "int category has primitives" true (List.length ints >= 5) ;
  let fences = primitives_in_category "fence" in
  check bool "fence category has primitives" true (List.length fences >= 2) ;
  let warp = primitives_in_category "warp" in
  check bool "warp category has primitives" true (List.length warp >= 8) ;
  let sync = primitives_in_category "sync" in
  check bool "sync category has barrier" true (List.length sync >= 1) ;
  let atomics = primitives_in_category "atomic" in
  check bool "atomic category has primitives" true (List.length atomics >= 14)

(* === Test suites === *)

let lookup_tests =
  [
    ("find thread_idx", `Quick, test_find_thread_idx);
    ("find nonexistent", `Quick, test_find_nonexistent);
    ("is_core_primitive", `Quick, test_is_core_primitive);
  ]

let variance_tests =
  [
    ("thread varying", `Quick, test_thread_varying);
    ("warp varying", `Quick, test_warp_varying);
    ("variance_of", `Quick, test_variance_of);
    ("join variance", `Quick, test_join_variance);
    ("variance leq", `Quick, test_variance_leq);
  ]

let convergence_tests =
  [
    ("convergence point", `Quick, test_convergence_point);
    ("warp convergence", `Quick, test_warp_convergence);
    ("requires convergence", `Quick, test_requires_convergence);
  ]

let purity_tests =
  [("pure", `Quick, test_pure); ("atomic", `Quick, test_atomic)]

let category_tests = [("category", `Quick, test_category)]

let () =
  run
    "Sarek_core_primitives"
    [
      ("lookup", lookup_tests);
      ("variance", variance_tests);
      ("convergence", convergence_tests);
      ("purity", purity_tests);
      ("category", category_tests);
    ]
