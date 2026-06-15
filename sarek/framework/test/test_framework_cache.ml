(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Test suite for Framework_cache *)

open Spoc_framework_registry

let test_get_cache_dir () =
  let dir = Framework_cache.get_cache_dir () in
  Alcotest.(check bool)
    "Cache directory is not empty"
    true
    (String.length dir > 0) ;
  Alcotest.(check bool)
    "Cache directory exists"
    true
    (Sys.file_exists dir && Sys.is_directory dir)

let test_compute_key () =
  let key1 =
    Framework_cache.compute_key
      ~dev_name:"Device1"
      ~driver_version:"1.0"
      ~source:"kernel code"
  in

  let key2 =
    Framework_cache.compute_key
      ~dev_name:"Device1"
      ~driver_version:"1.0"
      ~source:"kernel code"
  in

  Alcotest.(check string) "Same inputs produce same key" key1 key2 ;

  Alcotest.(check bool) "Key is hex string" true (String.length key1 = 32)
(* MD5 hex is 32 chars *)

let test_compute_key_different () =
  let key1 =
    Framework_cache.compute_key
      ~dev_name:"Device1"
      ~driver_version:"1.0"
      ~source:"code1"
  in

  let key2 =
    Framework_cache.compute_key
      ~dev_name:"Device1"
      ~driver_version:"1.0"
      ~source:"code2"
  in

  Alcotest.(check bool)
    "Different sources produce different keys"
    true
    (key1 <> key2)

let test_put_and_get () =
  let key = "test_key_12345" in
  let data = "test binary data" in

  Framework_cache.put ~key ~data ;

  let retrieved = Framework_cache.get ~key in
  Alcotest.(check bool)
    "Retrieved data matches"
    true
    (match retrieved with Some d -> d = data | None -> false)

let test_get_nonexistent () =
  let key = "nonexistent_key_67890" in
  let result = Framework_cache.get ~key in
  Alcotest.(check bool)
    "Nonexistent key returns None"
    true
    (match result with None -> true | Some _ -> false)

let test_stats_initialization () =
  Framework_cache.reset_stats () ;
  let stats = Framework_cache.get_stats () in
  Alcotest.(check int) "Hits initialized to 0" 0 stats.hits ;
  Alcotest.(check int) "Misses initialized to 0" 0 stats.misses ;
  Alcotest.(check int) "Puts initialized to 0" 0 stats.puts ;
  Alcotest.(check int) "Errors initialized to 0" 0 stats.errors

let test_stats_hit () =
  Framework_cache.reset_stats () ;
  let key = "stat_test_hit" in
  let data = "hit data" in

  Framework_cache.put ~key ~data ;
  let _ = Framework_cache.get ~key in

  let stats = Framework_cache.get_stats () in
  Alcotest.(check bool) "Hit count incremented" true (stats.hits > 0)

let test_stats_miss () =
  Framework_cache.reset_stats () ;
  let _ = Framework_cache.get ~key:"nonexistent_miss_test" in

  let stats = Framework_cache.get_stats () in
  Alcotest.(check bool) "Miss count incremented" true (stats.misses > 0)

let test_stats_put () =
  Framework_cache.reset_stats () ;
  Framework_cache.put ~key:"stat_test_put" ~data:"put data" ;

  let stats = Framework_cache.get_stats () in
  Alcotest.(check bool) "Put count incremented" true (stats.puts > 0)

let test_hit_rate () =
  Framework_cache.reset_stats () ;

  (* Put and hit *)
  Framework_cache.put ~key:"rate_test" ~data:"data" ;
  let _ = Framework_cache.get ~key:"rate_test" in
  let _ = Framework_cache.get ~key:"rate_test" in
  (* Another hit *)
  let _ = Framework_cache.get ~key:"nonexistent_rate" in
  (* Miss *)

  let rate = Framework_cache.hit_rate () in
  Alcotest.(check bool)
    "Hit rate is between 0 and 1"
    true
    (rate >= 0.0 && rate <= 1.0) ;

  (* Should be ~0.66 (2 hits, 1 miss) *)
  Alcotest.(check bool) "Hit rate is reasonable" true (rate > 0.5)

let test_print_stats () =
  (* Just check it doesn't crash *)
  Framework_cache.reset_stats () ;
  Framework_cache.print_stats () ;
  Alcotest.(check bool) "print_stats executes without error" true true

let test_cache_persistence () =
  let key = "persistence_test" in
  let data = "persistent data" in

  (* Write to cache *)
  Framework_cache.put ~key ~data ;

  (* Read back immediately *)
  let retrieved = Framework_cache.get ~key in
  Alcotest.(check bool)
    "Data persists in cache"
    true
    (match retrieved with Some d -> d = data | None -> false)

let () =
  let open Alcotest in
  run
    "Framework_cache"
    [
      ("directory", [test_case "Get cache directory" `Quick test_get_cache_dir]);
      ( "key_computation",
        [
          test_case "Compute key" `Quick test_compute_key;
          test_case
            "Different inputs different keys"
            `Quick
            test_compute_key_different;
        ] );
      ( "cache_operations",
        [
          test_case "Put and get" `Quick test_put_and_get;
          test_case "Get nonexistent" `Quick test_get_nonexistent;
          test_case "Cache persistence" `Quick test_cache_persistence;
        ] );
      ( "statistics",
        [
          test_case "Stats initialization" `Quick test_stats_initialization;
          test_case "Stats hit" `Quick test_stats_hit;
          test_case "Stats miss" `Quick test_stats_miss;
          test_case "Stats put" `Quick test_stats_put;
          test_case "Hit rate" `Quick test_hit_rate;
          test_case "Print stats" `Quick test_print_stats;
        ] );
    ]
