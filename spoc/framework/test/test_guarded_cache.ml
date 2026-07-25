(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Guarded_cache Tests - concurrent multi-domain (OCaml 5 Domain) safety.
 *
 * These are backend-agnostic: they exercise the shared guarded cache directly
 * with plain int values, so no GPU / driver is required and the same
 * regression guard covers every backend that uses Spoc_framework.Guarded_cache
 * (CUDA, HIP, OpenCL, Metal, Vulkan) plus the core Runtime cache.
 *
 * Why the primary test is a GREEN stress test rather than a deterministic
 * red-on-race: concurrent writes to a *raw* stdlib Hashtbl from several Domains
 * are undefined behaviour - the observed failure ranges from a caught
 * exception (bucket half-linked during a resize) to a hard segfault that would
 * take down the whole test binary non-deterministically. There is no reliable,
 * catchable, deterministic "red" to assert on. Instead we hammer the *guarded*
 * cache hard enough to hit the concurrent-insert / double-compile / evict paths
 * and assert its invariants hold. A runnable red demonstration against a raw
 * Hashtbl is provided but gated behind SAREK_GUARDED_CACHE_RED=1 so the default
 * suite stays safe (see [red_demo_unguarded]).
 ******************************************************************************)

module GC = Spoc_framework.Guarded_cache

let num_domains = 8

let iterations_per_domain = 4000

let shared_key_count = 16

(* A tiny variable delay inside [build] widens the miss window so concurrent
   compiles of the same key actually race (exercising the loser-destroy path),
   rather than the fast domain always winning uncontended. *)
let jitter () =
  for _ = 0 to Random.int 40 do
    Domain.cpu_relax ()
  done

(* Every built value is a globally unique int; every destroyed value is
   recorded. Invariant under any interleaving: builds - destroys = table size,
   i.e. every value ever built is either retained in the table or destroyed as
   a race loser - never leaked, never double-counted. *)
let test_stress_shared_and_distinct_keys () =
  let builds = Atomic.make 0 in
  let destroys = Atomic.make 0 in
  let cache =
    GC.create
      ~destroy:(fun (_ : int) -> ignore (Atomic.fetch_and_add destroys 1))
      ()
  in
  let build () =
    jitter () ;
    Atomic.fetch_and_add builds 1
  in
  (* Each domain repeatedly resolves shared keys (contended) and its own
     domain-private keys (uncontended inserts), checking that a shared key
     always hands back one stable value. *)
  let worker d () =
    let last_for_shared = Array.make shared_key_count (-1) in
    for i = 0 to iterations_per_domain - 1 do
      let use_shared = i land 1 = 0 in
      if use_shared then begin
        (* [i] is even in this branch, so [i mod shared_key_count] would only
           ever hit even indices; divide first so all shared keys are used. *)
        let k = i / 2 mod shared_key_count in
        let key = Printf.sprintf "shared_%d" k in
        let v = GC.find_or_build cache ~key ~device_id:0 build in
        if last_for_shared.(k) >= 0 && last_for_shared.(k) <> v then
          Alcotest.failf
            "shared key %s resolved to two different values (%d then %d)"
            key
            last_for_shared.(k)
            v ;
        last_for_shared.(k) <- v
      end
      else begin
        let key = Printf.sprintf "d%d_k%d" d i in
        let _ = GC.find_or_build cache ~key ~device_id:(1 + d) build in
        ()
      end
    done
  in
  let domains = List.init num_domains (fun d -> Domain.spawn (worker d)) in
  List.iter Domain.join domains ;
  let total_built = Atomic.get builds in
  let total_destroyed = Atomic.get destroys in
  let retained = GC.length cache in
  (* Distinct keys inserted = shared keys + one private key per (odd) iteration
     per domain. *)
  let private_per_domain =
    (* count of odd i in [0, iterations_per_domain) *)
    iterations_per_domain / 2
  in
  let expected_distinct =
    shared_key_count + (num_domains * private_per_domain)
  in
  Alcotest.(check int)
    "retained entries = number of distinct keys"
    expected_distinct
    retained ;
  Alcotest.(check int)
    "every built value is either retained or destroyed (no leak, no \
     double-count)"
    retained
    (total_built - total_destroyed) ;
  Alcotest.(check bool)
    "at least one build per distinct key happened"
    true
    (total_built >= expected_distinct)

(* Concurrent clear vs. find_or_build: no exception, and afterwards the cache is
   internally consistent (length matches a fresh recount). *)
let test_concurrent_clear () =
  let cache = GC.create ~destroy:(fun (_ : int) -> ()) () in
  let counter = Atomic.make 0 in
  let build () = Atomic.fetch_and_add counter 1 in
  let writer () =
    for i = 0 to iterations_per_domain - 1 do
      let key = Printf.sprintf "k%d" (i mod 32) in
      let _ = GC.find_or_build cache ~key build in
      ()
    done
  in
  let clearer () =
    for _ = 0 to 200 do
      GC.clear cache ;
      Domain.cpu_relax ()
    done
  in
  let ds =
    Domain.spawn clearer :: List.init num_domains (fun _ -> Domain.spawn writer)
  in
  List.iter Domain.join ds ;
  (* A final clear leaves it empty; the key assertion is that nothing above
     raised or corrupted the table. *)
  GC.clear cache ;
  Alcotest.(check int) "cache empty after final clear" 0 (GC.length cache)

(* Concurrent evict_device: each domain owns a device id and evicts it while
   others keep inserting. Must not raise; destroy count must never exceed the
   number of builds. *)
let test_concurrent_evict_device () =
  let builds = Atomic.make 0 in
  let destroys = Atomic.make 0 in
  let cache =
    GC.create
      ~destroy:(fun (_ : int) -> ignore (Atomic.fetch_and_add destroys 1))
      ()
  in
  let build () = Atomic.fetch_and_add builds 1 in
  let worker d () =
    for i = 0 to iterations_per_domain - 1 do
      let key = Printf.sprintf "d%d_k%d" d (i mod 64) in
      (* A build whose device is evicted mid-flight is reported rather than
         installed (the value is destroyed, so conservation below still holds).
         Expected here: this test evicts concurrently with building on purpose. *)
      (try ignore (GC.find_or_build cache ~key ~device_id:d build)
       with GC.Device_destroyed_during_build _ -> ()) ;
      if i mod 500 = 0 then GC.evict_device cache d
    done
  in
  let ds = List.init num_domains (fun d -> Domain.spawn (worker d)) in
  List.iter Domain.join ds ;
  Alcotest.(check bool)
    "destroys never exceed builds"
    true
    (Atomic.get destroys <= Atomic.get builds) ;
  Alcotest.(check bool)
    "retained + destroyed = built (conservation)"
    true
    (GC.length cache + Atomic.get destroys = Atomic.get builds)

(* Deterministic regression test for the insert-after-evict window.

   Domain A misses on key K for device 7 and releases the lock to run the (slow)
   build. While the build is in flight, device 7 is destroyed, which runs
   [evict_device 7]. A's entry is not in the table yet, so eviction finds
   nothing; before the eviction-epoch check, A then re-acquired the lock and
   INSERTED a value built against a now-dead device, recorded against a dead
   device id: never unloaded (leak), and served to every later lookup - possibly
   after the id was recreated, i.e. a launch on a module from a destroyed
   context. The conservation assertion in [test_concurrent_evict_device] is
   satisfied by that interleaving, which is why the suite was green.

   Deterministic because the two domains hand off through atomics rather than
   racing: the eviction is guaranteed to land inside the build window. *)
let test_evict_device_during_build () =
  let destroyed = Atomic.make 0 in
  let cache =
    GC.create
      ~destroy:(fun (_ : string) -> ignore (Atomic.fetch_and_add destroyed 1))
      ()
  in
  let build_started = Atomic.make false in
  let evict_done = Atomic.make false in
  let builder () =
    try
      `Returned
        (GC.find_or_build cache ~key:"K" ~device_id:7 (fun () ->
             Atomic.set build_started true ;
             while not (Atomic.get evict_done) do
               Domain.cpu_relax ()
             done ;
             "module-for-device-7"))
    with GC.Device_destroyed_during_build id -> `Rejected id
  in
  let d = Domain.spawn builder in
  while not (Atomic.get build_started) do
    Domain.cpu_relax ()
  done ;
  (* Device 7 is destroyed here: its context is gone, its modules must all be
     unloaded and none may be installed afterwards. *)
  GC.evict_device cache 7 ;
  Atomic.set evict_done true ;
  (match Domain.join d with
  | `Rejected id ->
      Alcotest.(check int) "rejection names the destroyed device" 7 id
  | `Returned v ->
      Alcotest.failf
        "find_or_build installed and returned %S for a device destroyed during \
         the build"
        v) ;
  Alcotest.(check int)
    "the value built for the destroyed device was destroyed, not leaked"
    1
    (Atomic.get destroyed) ;
  Alcotest.(check int)
    "nothing was installed for the destroyed device"
    0
    (GC.length cache) ;
  (* And the key is genuinely free again: a later lookup rebuilds rather than
     being served the value compiled against the destroyed device. *)
  let later =
    GC.find_or_build cache ~key:"K" ~device_id:7 (fun () -> "freshly-rebuilt")
  in
  Alcotest.(check string) "later lookup rebuilds" "freshly-rebuilt" later

(* On a BORROWING cache ([~invalidated_by_clear:true]), [clear] must reject an
   in-flight build.

   The interleaving, which the per-device epoch alone does not cover because no
   device is destroyed: (1) domain A misses on key K and releases the lock; its
   build finishes - for a real borrowing cache that means the inner cache handed
   back a value closing over handle H - but is not installed yet. (2) Domain B
   clears: the clear walks the table, does not see A's entry, and (in the real
   system) the layer underneath then releases H. (3) A re-acquires, finds the
   key absent, and installs. The cache now serves a value over a released handle
   PERMANENTLY, having survived the very teardown that was in progress.

   Modelled here with a handle whose release is observable, so the assertion is
   on the poisoning itself ("a later lookup is served a released handle") rather
   than on a crash. Deterministic: the two domains hand off through atomics. *)
let test_clear_during_build_rejected_when_borrowing () =
  (* Stand-in for a backend handle owned by the layer underneath. *)
  let released = Atomic.make false in
  let cache =
    GC.create ~invalidated_by_clear:true ~destroy:(fun (_ : string) -> ()) ()
  in
  let build_started = Atomic.make false in
  let clear_done = Atomic.make false in
  let builder () =
    match
      GC.find_or_build cache ~key:"K" ~device_id:0 (fun () ->
          Atomic.set build_started true ;
          while not (Atomic.get clear_done) do
            Domain.cpu_relax ()
          done ;
          "closure-over-handle-1")
    with
    | v -> `Returned v
    | exception GC.Cache_cleared_during_build -> `Rejected
  in
  let d = Domain.spawn builder in
  while not (Atomic.get build_started) do
    Domain.cpu_relax ()
  done ;
  (* The teardown: drop the memo, then the layer underneath releases the handle
     every cached closure was built over. *)
  GC.clear cache ;
  Atomic.set released true ;
  Atomic.set clear_done true ;
  (match Domain.join d with
  | `Rejected -> ()
  | `Returned v ->
      Alcotest.failf
        "a build that finished across a clear was installed and returned %S"
        v) ;
  Alcotest.(check int) "nothing survived the clear" 0 (GC.length cache) ;
  (* The decisive check: a LATER lookup must rebuild against live handles, not
     be served the closure over the handle the clear released. *)
  (* Seeded [false] on purpose: the failure under test is the later lookup being
     served the poisoned entry, in which case this closure never runs at all.
     Seeding it with the asserted value would make the check pass in exactly
     that case, i.e. assert nothing. *)
  let rebuilt_after_release = ref false in
  let later =
    GC.find_or_build cache ~key:"K" ~device_id:0 (fun () ->
        rebuilt_after_release := Atomic.get released ;
        "closure-over-handle-2")
  in
  Alcotest.(check string)
    "a later lookup rebuilds instead of being served the poisoned value"
    "closure-over-handle-2"
    later ;
  Alcotest.(check bool)
    "the build closure actually ran, after the release, i.e. against fresh \
     handles"
    true
    !rebuilt_after_release

(* The mirror image: on an OWNING cache (the default), [clear] must NOT reject
   an in-flight build. It releases the handles this cache owns; it does not
   touch the device, so the value being built is still valid and installing it
   is correct. Guards against applying the generation check globally - the
   distinction is per-cache, and getting it backwards turns benign concurrent
   clears into spurious compile failures for every backend. *)
let test_clear_during_build_still_installs () =
  let cache = GC.create ~destroy:(fun (_ : string) -> ()) () in
  let build_started = Atomic.make false in
  let clear_done = Atomic.make false in
  let d =
    Domain.spawn (fun () ->
        GC.find_or_build cache ~key:"K" ~device_id:3 (fun () ->
            Atomic.set build_started true ;
            while not (Atomic.get clear_done) do
              Domain.cpu_relax ()
            done ;
            "v"))
  in
  while not (Atomic.get build_started) do
    Domain.cpu_relax ()
  done ;
  GC.clear cache ;
  Atomic.set clear_done true ;
  Alcotest.(check string)
    "build result installed after a clear"
    "v"
    (Domain.join d) ;
  Alcotest.(check int) "and it is cached" 1 (GC.length cache)

(* A raising [destroy] must not strand the other victims: they are already
   unlinked from the table, so a bail-out on the first failure leaks them
   outright. Every victim is attempted; the failure is still reported. *)
let test_raising_destroy_does_not_strand_victims () =
  let destroyed = Atomic.make 0 in
  let cache =
    GC.create
      ~destroy:(fun v ->
        ignore (Atomic.fetch_and_add destroyed 1) ;
        if v = "boom" then failwith "release failed")
      ()
  in
  List.iter
    (fun (k, v) ->
      ignore (GC.find_or_build cache ~key:k ~device_id:1 (fun () -> v)))
    [("a", "ok1"); ("b", "boom"); ("c", "ok2"); ("d", "ok3")] ;
  let raised =
    try
      GC.clear cache ;
      false
    with Failure _ -> true
  in
  Alcotest.(check bool) "the failing release is reported" true raised ;
  Alcotest.(check int) "every victim was attempted" 4 (Atomic.get destroyed) ;
  Alcotest.(check int) "the table is empty" 0 (GC.length cache)

(* A raising [destroy] on the lost-race path must not destroy the winner: a
   valid value exists and the caller asked for a value, not for the release
   error of the copy we threw away. *)
let test_raising_destroy_on_lost_race_returns_winner () =
  let cache =
    GC.create ~destroy:(fun (_ : string) -> failwith "release failed") ()
  in
  let build_started = Atomic.make false in
  let winner_installed = Atomic.make false in
  let loser =
    Domain.spawn (fun () ->
        GC.find_or_build cache ~key:"K" (fun () ->
            Atomic.set build_started true ;
            while not (Atomic.get winner_installed) do
              Domain.cpu_relax ()
            done ;
            "loser"))
  in
  while not (Atomic.get build_started) do
    Domain.cpu_relax ()
  done ;
  ignore (GC.find_or_build cache ~key:"K" (fun () -> "winner")) ;
  Atomic.set winner_installed true ;
  Alcotest.(check string)
    "loser observes the winner, not the release error"
    "winner"
    (Domain.join loser)

(* Runnable red demonstration, OFF by default (would be UB/segfault-prone in
   the suite). With SAREK_GUARDED_CACHE_RED=1 it hammers a *raw* unguarded
   Hashtbl from several domains and reports the corruption/exception - the very
   failure Guarded_cache prevents. *)
let red_demo_unguarded () =
  match Sys.getenv_opt "SAREK_GUARDED_CACHE_RED" with
  | Some "1" ->
      (* Each domain inserts its OWN disjoint block of keys, so with correct
         synchronization the final length is exactly [num_domains * m]. A raw
         Hashtbl started small forces many resizes; concurrent resizes drop
         entries (silent corruption) - reliably observed as a large length
         shortfall - and can also raise or segfault (all are manifestations of
         the same data race). *)
      let m = 200_000 in
      let raw : (int, int) Hashtbl.t = Hashtbl.create 4 in
      let failed = Atomic.make false in
      let worker d () =
        try
          for i = 0 to m - 1 do
            let key = (d * m) + i in
            Hashtbl.replace raw key i ;
            ignore (Hashtbl.find_opt raw key)
          done
        with e ->
          Atomic.set failed true ;
          Printf.eprintf
            "[red-demo] raw Hashtbl raised under contention: %s\n%!"
            (Printexc.to_string e)
      in
      let ds = List.init num_domains (fun d -> Domain.spawn (worker d)) in
      List.iter Domain.join ds ;
      let expected = num_domains * m in
      let got = Hashtbl.length raw in
      Printf.eprintf
        "[red-demo] raw unguarded Hashtbl: expected %d distinct entries, \
         retained %d (lost %d), raised=%b -> CORRUPTION %s\n\
         %!"
        expected
        got
        (expected - got)
        (Atomic.get failed)
        (if got <> expected || Atomic.get failed then "OBSERVED (RED)"
         else "not observed this run (race is non-deterministic)")
  | _ ->
      Printf.printf
        "  red demo skipped (set SAREK_GUARDED_CACHE_RED=1 to run the \
         raw-Hashtbl race)\n\
         %!"

let () =
  Random.self_init () ;
  Alcotest.run
    "Guarded_cache"
    [
      ( "concurrency",
        [
          Alcotest.test_case
            "stress: shared + distinct keys, N domains"
            `Slow
            test_stress_shared_and_distinct_keys;
          Alcotest.test_case
            "concurrent clear vs find_or_build"
            `Slow
            test_concurrent_clear;
          Alcotest.test_case
            "concurrent evict_device"
            `Slow
            test_concurrent_evict_device;
          Alcotest.test_case
            "evict_device during an in-flight build (deterministic)"
            `Quick
            test_evict_device_during_build;
          Alcotest.test_case
            "clear during an in-flight build is rejected on a borrowing cache"
            `Quick
            test_clear_during_build_rejected_when_borrowing;
          Alcotest.test_case
            "clear during an in-flight build still installs on an owning cache"
            `Quick
            test_clear_during_build_still_installs;
          Alcotest.test_case
            "a raising destroy does not strand the other victims"
            `Quick
            test_raising_destroy_does_not_strand_victims;
          Alcotest.test_case
            "a raising destroy on a lost race still returns the winner"
            `Quick
            test_raising_destroy_on_lost_race_returns_winner;
          Alcotest.test_case
            "red demo (unguarded, opt-in)"
            `Quick
            red_demo_unguarded;
        ] );
    ]
