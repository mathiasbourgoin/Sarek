(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Test suite for Intrinsic_registry *)

open Spoc_framework.Framework_sig
open Spoc_framework_registry

let test_make_registry () =
  let module R = Intrinsic_registry.Make () in
  Alcotest.(check bool) "Registry created successfully" true true

let test_register_and_find () =
  let module R = Intrinsic_registry.Make () in
  let impl =
    Intrinsic_registry.make_simple_intrinsic
      ~name:"test_intr"
      ~codegen:"test_code"
  in
  R.register "test_intr" impl ;

  let found = R.find "test_intr" in
  Alcotest.(check bool)
    "Intrinsic found after registration"
    true
    (match found with Some _ -> true | None -> false)

let test_find_nonexistent () =
  let module R = Intrinsic_registry.Make () in
  let found = R.find "nonexistent" in
  Alcotest.(check bool)
    "Nonexistent intrinsic returns None"
    true
    (match found with None -> true | Some _ -> false)

let test_list_all () =
  let module R = Intrinsic_registry.Make () in
  let impl1 =
    Intrinsic_registry.make_simple_intrinsic ~name:"intr1" ~codegen:"code1"
  in
  let impl2 =
    Intrinsic_registry.make_simple_intrinsic ~name:"intr2" ~codegen:"code2"
  in

  R.register "intr1" impl1 ;
  R.register "intr2" impl2 ;

  let all = R.list_all () in
  Alcotest.(check bool)
    "list_all returns both intrinsics"
    true
    (List.length all = 2 && List.mem "intr1" all && List.mem "intr2" all)

let test_global_register () =
  let impl =
    Intrinsic_registry.make_simple_intrinsic ~name:"global_test" ~codegen:"test"
  in
  Intrinsic_registry.Global.register ~backend:"TestBackend" "global_test" impl ;

  let found =
    Intrinsic_registry.Global.find ~backend:"TestBackend" "global_test"
  in
  Alcotest.(check bool)
    "Global intrinsic found"
    true
    (match found with Some _ -> true | None -> false)

let test_global_find_all () =
  let impl1 =
    Intrinsic_registry.make_simple_intrinsic ~name:"multi_intr" ~codegen:"code1"
  in
  let impl2 =
    Intrinsic_registry.make_simple_intrinsic ~name:"multi_intr" ~codegen:"code2"
  in

  Intrinsic_registry.Global.register ~backend:"Backend1" "multi_intr" impl1 ;
  Intrinsic_registry.Global.register ~backend:"Backend2" "multi_intr" impl2 ;

  let all = Intrinsic_registry.Global.find_all "multi_intr" in
  Alcotest.(check bool)
    "find_all returns multiple backends"
    true
    (List.length all >= 2)

let test_backends_for () =
  let impl =
    Intrinsic_registry.make_simple_intrinsic ~name:"test" ~codegen:"code"
  in
  Intrinsic_registry.Global.register ~backend:"CUDA" "test" impl ;
  Intrinsic_registry.Global.register ~backend:"OpenCL" "test" impl ;

  let backends = Intrinsic_registry.Global.backends_for "test" in
  Alcotest.(check bool)
    "backends_for returns correct backends"
    true
    (List.mem "CUDA" backends && List.mem "OpenCL" backends)

let test_make_simple_intrinsic () =
  let impl =
    Intrinsic_registry.make_simple_intrinsic
      ~name:"simple"
      ~codegen:"simple_code"
  in
  Alcotest.(check string) "Name stored correctly" "simple" impl.intr_name ;
  Alcotest.(check string)
    "Codegen stored correctly"
    "simple_code"
    impl.intr_codegen ;
  Alcotest.(check bool)
    "Convergence is Uniform"
    true
    (impl.intr_convergence = Uniform)

let test_make_sync_intrinsic () =
  let impl =
    Intrinsic_registry.make_sync_intrinsic
      ~name:"barrier"
      ~codegen:"__syncthreads()"
  in
  Alcotest.(check bool) "Convergence is Sync" true (impl.intr_convergence = Sync)

let test_make_divergent_intrinsic () =
  let impl =
    Intrinsic_registry.make_divergent_intrinsic
      ~name:"thread_id"
      ~codegen:"threadIdx.x"
  in
  Alcotest.(check bool)
    "Convergence is Divergent"
    true
    (impl.intr_convergence = Divergent)

let test_is_safe_in_divergent_flow () =
  let uniform_impl =
    Intrinsic_registry.make_simple_intrinsic ~name:"uniform" ~codegen:"code"
  in
  let sync_impl =
    Intrinsic_registry.make_sync_intrinsic ~name:"sync" ~codegen:"barrier"
  in
  let divergent_impl =
    Intrinsic_registry.make_divergent_intrinsic ~name:"divergent" ~codegen:"tid"
  in

  Alcotest.(check bool)
    "Uniform is safe in divergent flow"
    true
    (Intrinsic_registry.is_safe_in_divergent_flow uniform_impl) ;
  Alcotest.(check bool)
    "Sync is NOT safe in divergent flow"
    false
    (Intrinsic_registry.is_safe_in_divergent_flow sync_impl) ;
  Alcotest.(check bool)
    "Divergent is safe in divergent flow"
    true
    (Intrinsic_registry.is_safe_in_divergent_flow divergent_impl)

let test_requires_uniform_execution () =
  let uniform_impl =
    Intrinsic_registry.make_simple_intrinsic ~name:"uniform" ~codegen:"code"
  in
  let sync_impl =
    Intrinsic_registry.make_sync_intrinsic ~name:"sync" ~codegen:"barrier"
  in

  Alcotest.(check bool)
    "Uniform doesn't require uniform execution"
    false
    (Intrinsic_registry.requires_uniform_execution uniform_impl) ;
  Alcotest.(check bool)
    "Sync requires uniform execution"
    true
    (Intrinsic_registry.requires_uniform_execution sync_impl)

let test_thread_intrinsics () =
  Alcotest.(check bool)
    "thread_id_x is thread intrinsic"
    true
    (Intrinsic_registry.Thread_intrinsics.is_thread_intrinsic "thread_id_x") ;
  Alcotest.(check bool)
    "block_dim_y is thread intrinsic"
    true
    (Intrinsic_registry.Thread_intrinsics.is_thread_intrinsic "block_dim_y") ;
  Alcotest.(check bool)
    "sqrt is NOT thread intrinsic"
    false
    (Intrinsic_registry.Thread_intrinsics.is_thread_intrinsic "sqrt")

let test_sync_intrinsics () =
  Alcotest.(check bool)
    "block_barrier is sync intrinsic"
    true
    (Intrinsic_registry.Sync_intrinsics.is_sync_intrinsic "block_barrier") ;
  Alcotest.(check bool)
    "memory_fence is sync intrinsic"
    true
    (Intrinsic_registry.Sync_intrinsics.is_sync_intrinsic "memory_fence") ;
  Alcotest.(check bool)
    "thread_id_x is NOT sync intrinsic"
    false
    (Intrinsic_registry.Sync_intrinsics.is_sync_intrinsic "thread_id_x")

let test_atomic_intrinsics () =
  Alcotest.(check bool)
    "atomic_add is atomic intrinsic"
    true
    (Intrinsic_registry.Atomic_intrinsics.is_atomic_intrinsic "atomic_add") ;
  Alcotest.(check bool)
    "atomic_cas is atomic intrinsic"
    true
    (Intrinsic_registry.Atomic_intrinsics.is_atomic_intrinsic "atomic_cas") ;
  Alcotest.(check bool)
    "sqrt is NOT atomic intrinsic"
    false
    (Intrinsic_registry.Atomic_intrinsics.is_atomic_intrinsic "sqrt")

let test_math_intrinsics () =
  Alcotest.(check bool)
    "sin is math intrinsic"
    true
    (Intrinsic_registry.Math_intrinsics.is_math_intrinsic "sin") ;
  Alcotest.(check bool)
    "sqrt is math intrinsic"
    true
    (Intrinsic_registry.Math_intrinsics.is_math_intrinsic "sqrt") ;
  Alcotest.(check bool)
    "fma is math intrinsic"
    true
    (Intrinsic_registry.Math_intrinsics.is_math_intrinsic "fma") ;
  Alcotest.(check bool)
    "thread_id_x is NOT math intrinsic"
    false
    (Intrinsic_registry.Math_intrinsics.is_math_intrinsic "thread_id_x")

let test_global_list_all () =
  (* Register a few intrinsics *)
  let impl =
    Intrinsic_registry.make_simple_intrinsic ~name:"test_list" ~codegen:"code"
  in
  Intrinsic_registry.Global.register ~backend:"Test" "test_list" impl ;

  let all = Intrinsic_registry.Global.list_all () in
  Alcotest.(check bool)
    "Global list_all returns intrinsics"
    true
    (List.length all >= 1)

let () =
  let open Alcotest in
  run
    "Intrinsic_registry"
    [
      ( "basic_operations",
        [
          test_case "Create registry" `Quick test_make_registry;
          test_case "Register and find" `Quick test_register_and_find;
          test_case "Find nonexistent" `Quick test_find_nonexistent;
          test_case "List all intrinsics" `Quick test_list_all;
        ] );
      ( "global_registry",
        [
          test_case "Global register" `Quick test_global_register;
          test_case "Global find all" `Quick test_global_find_all;
          test_case "Backends for intrinsic" `Quick test_backends_for;
          test_case "Global list all" `Quick test_global_list_all;
        ] );
      ( "intrinsic_types",
        [
          test_case "Simple intrinsic" `Quick test_make_simple_intrinsic;
          test_case "Sync intrinsic" `Quick test_make_sync_intrinsic;
          test_case "Divergent intrinsic" `Quick test_make_divergent_intrinsic;
        ] );
      ( "convergence_safety",
        [
          test_case
            "Safe in divergent flow"
            `Quick
            test_is_safe_in_divergent_flow;
          test_case
            "Requires uniform execution"
            `Quick
            test_requires_uniform_execution;
        ] );
      ( "standard_intrinsics",
        [
          test_case "Thread intrinsics" `Quick test_thread_intrinsics;
          test_case "Sync intrinsics" `Quick test_sync_intrinsics;
          test_case "Atomic intrinsics" `Quick test_atomic_intrinsics;
          test_case "Math intrinsics" `Quick test_math_intrinsics;
        ] );
    ]
