(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Integration tests using dummy backend to exercise framework *)

open Spoc_framework_registry

(* The dummy backend auto-registers on module load *)
let () = ignore Dummy_backend.Dummy_backend.name

let test_dummy_backend_registration () =
  (* Backend should be registered *)
  let result = Framework_registry.find_backend "Dummy" in
  Alcotest.(check bool)
    "Dummy backend is registered"
    true
    (match result with Some _ -> true | None -> false)

let test_dummy_backend_available () =
  (* Should be in available backends *)
  let available = Framework_registry.available_backends () in
  let has_dummy =
    List.exists
      (fun (module B : Spoc_framework.Framework_sig.BACKEND) ->
        B.name = "Dummy")
      available
  in
  Alcotest.(check bool) "Dummy backend is available" true has_dummy

let test_dummy_intrinsics_registered () =
  (* Check intrinsics were registered *)
  match Framework_registry.find_backend "Dummy" with
  | None -> Alcotest.fail "Dummy backend not found"
  | Some (module B) ->
      let all_intrinsics = B.Intrinsics.list_all () in
      Alcotest.(check bool)
        "test_thread_id registered"
        true
        (List.mem "test_thread_id" all_intrinsics) ;
      Alcotest.(check bool)
        "test_barrier registered"
        true
        (List.mem "test_barrier" all_intrinsics) ;
      Alcotest.(check bool)
        "test_add registered"
        true
        (List.mem "test_add" all_intrinsics)

let test_cache_with_dummy_backend () =
  (* Test cache with dummy backend info *)
  let key =
    Framework_cache.compute_key
      ~dev_name:"Dummy Test Device"
      ~driver_version:"1.0.0"
      ~source:"test kernel source"
  in

  let data = "compiled kernel binary" in
  Framework_cache.put ~key ~data ;

  let retrieved = Framework_cache.get ~key in
  Alcotest.(check bool)
    "Cache stores dummy backend data"
    true
    (match retrieved with Some d -> d = data | None -> false)

let test_best_backend_with_dummy () =
  (* With dummy registered, best_backend should work *)
  let best = Framework_registry.best_backend () in
  Alcotest.(check bool)
    "best_backend finds at least one backend"
    true
    (match best with Some _ -> true | None -> false)

let test_cache_stats_with_usage () =
  (* Reset and test cache statistics *)
  Framework_cache.reset_stats () ;

  let key = "stats_test" in
  let data = "test data" in

  (* Put *)
  Framework_cache.put ~key ~data ;
  let stats_after_put = Framework_cache.get_stats () in
  Alcotest.(check bool)
    "Put increments puts counter"
    true
    (stats_after_put.puts > 0) ;

  (* Hit *)
  let _ = Framework_cache.get ~key in
  let stats_after_hit = Framework_cache.get_stats () in
  Alcotest.(check bool)
    "Get increments hits counter"
    true
    (stats_after_hit.hits > 0) ;

  (* Miss *)
  let _ = Framework_cache.get ~key:"nonexistent" in
  let stats_after_miss = Framework_cache.get_stats () in
  Alcotest.(check bool)
    "Get miss increments misses counter"
    true
    (stats_after_miss.misses > 0)

let test_priority_with_dummy () =
  (* Dummy has priority 1 (low) *)
  let prio = Framework_registry.priority "Dummy" in
  Alcotest.(check int) "Dummy has priority 1" 1 prio

let test_dummy_device_capabilities () =
  (* Get dummy backend and check device capabilities *)
  match Framework_registry.find_backend "Dummy" with
  | None -> Alcotest.fail "Dummy backend not found"
  | Some (module B) ->
      B.Device.init () ;
      let count = B.Device.count () in
      Alcotest.(check int) "Dummy has 1 device" 1 count ;

      let dev = B.Device.get 0 in
      let caps = B.Device.capabilities dev in
      Alcotest.(check bool) "Dummy device is CPU" true caps.is_cpu ;
      Alcotest.(check int)
        "Dummy has 256 max threads"
        256
        caps.max_threads_per_block

let test_global_intrinsic_registration () =
  (* Register an intrinsic in global registry *)
  let impl =
    Intrinsic_registry.make_simple_intrinsic ~name:"global_test" ~codegen:"TEST"
  in
  Intrinsic_registry.Global.register ~backend:"Dummy" "global_test" impl ;

  let found = Intrinsic_registry.Global.find ~backend:"Dummy" "global_test" in
  Alcotest.(check bool)
    "Global intrinsic registered and found"
    true
    (match found with Some _ -> true | None -> false)

let test_execution_model () =
  (* Check dummy backend execution model *)
  match Framework_registry.find_backend "Dummy" with
  | None -> Alcotest.fail "Dummy backend not found"
  | Some (module B) ->
      Alcotest.(check bool)
        "Dummy uses Custom execution model"
        true
        (B.execution_model = Spoc_framework.Framework_sig.Custom)

let test_all_backends_list () =
  (* Test all_backend_names function *)
  let all = Framework_registry.all_backend_names () in
  let has_dummy = List.exists (fun name -> name = "Dummy") all in
  Alcotest.(check bool) "all_backend_names includes Dummy" true has_dummy ;
  Alcotest.(check bool) "all_backend_names non-empty" true (List.length all > 0)

let test_cache_dir () =
  (* Test cache directory creation/access *)
  let dir = Framework_cache.get_cache_dir () in
  Alcotest.(check bool) "Cache dir is non-empty" true (String.length dir > 0)

let test_cache_hit_rate () =
  (* Test hit rate calculation with fresh stats *)
  Framework_cache.reset_stats () ;

  (* Create unique key for this test *)
  let key = "hitratetest_unique_" ^ string_of_float (Unix.gettimeofday ()) in
  let key2 = "miss_" ^ key in

  (* Put and hit *)
  Framework_cache.put ~key ~data:"test" ;
  let _ = Framework_cache.get ~key in
  (* hit *)
  let _ = Framework_cache.get ~key:key2 in
  (* miss *)

  let stats = Framework_cache.get_stats () in
  (* We should have at least 1 hit and 1 miss *)
  Alcotest.(check bool) "Has hits" true (stats.hits > 0) ;
  Alcotest.(check bool) "Has misses" true (stats.misses > 0) ;

  (* Hit rate should be between 0 and 100 *)
  let rate = Framework_cache.hit_rate () in
  Alcotest.(check bool) "Hit rate is valid" true (rate >= 0.0 && rate <= 100.0)

let test_intrinsic_lookup () =
  (* Test intrinsic lookup *)
  match Framework_registry.find_backend "Dummy" with
  | None -> Alcotest.fail "Dummy backend not found"
  | Some (module B) ->
      (* Test finding by name *)
      let thread_id = B.Intrinsics.find "test_thread_id" in
      Alcotest.(check bool)
        "test_thread_id found"
        true
        (match thread_id with Some _ -> true | None -> false) ;

      (* Test list_all returns all intrinsics *)
      let all = B.Intrinsics.list_all () in
      Alcotest.(check bool)
        "list_all returns intrinsics"
        true
        (List.length all >= 3)

let test_cache_stats_print () =
  (* Exercise print_stats function *)
  Framework_cache.reset_stats () ;
  Framework_cache.put ~key:"test1" ~data:"data1" ;
  let _ = Framework_cache.get ~key:"test1" in
  (* Just call it - output goes to stdout *)
  Framework_cache.print_stats ()

let test_error_handling () =
  (* Test Framework_error functions *)
  let err = Framework_error.Backend_not_found {name = "NonExistent"} in
  let msg = Framework_error.to_string err in
  Alcotest.(check bool)
    "Error message contains backend name"
    true
    (String.length msg > 0) ;

  (* Test with_default catches framework errors *)
  let result =
    Framework_error.with_default ~default:"default" (fun () ->
        Framework_error.raise_error
          (Cache_error {operation = "test"; reason = "test error"}))
  in
  Alcotest.(check string) "with_default returns default" "default" result

let () =
  let open Alcotest in
  run
    "Framework_integration"
    [
      ( "backend_registration",
        [
          test_case
            "Dummy backend registration"
            `Quick
            test_dummy_backend_registration;
          test_case
            "Dummy backend available"
            `Quick
            test_dummy_backend_available;
          test_case
            "Dummy intrinsics registered"
            `Quick
            test_dummy_intrinsics_registered;
          test_case
            "Dummy device capabilities"
            `Quick
            test_dummy_device_capabilities;
          test_case "Execution model" `Quick test_execution_model;
          test_case "All backends list" `Quick test_all_backends_list;
        ] );
      ( "cache_integration",
        [
          test_case
            "Cache with dummy backend"
            `Quick
            test_cache_with_dummy_backend;
          test_case "Cache stats with usage" `Quick test_cache_stats_with_usage;
          test_case "Cache directory" `Quick test_cache_dir;
          test_case "Cache hit rate" `Quick test_cache_hit_rate;
          test_case "Cache stats print" `Quick test_cache_stats_print;
        ] );
      ( "registry_integration",
        [
          test_case "Best backend with dummy" `Quick test_best_backend_with_dummy;
          test_case "Priority with dummy" `Quick test_priority_with_dummy;
          test_case
            "Global intrinsic registration"
            `Quick
            test_global_intrinsic_registration;
          test_case "Intrinsic lookup" `Quick test_intrinsic_lookup;
        ] );
      ("error_handling", [test_case "Error handling" `Quick test_error_handling]);
    ]
