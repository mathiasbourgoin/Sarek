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
      let _ = GC.find_or_build cache ~key ~device_id:d build in
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
            "red demo (unguarded, opt-in)"
            `Quick
            red_demo_unguarded;
        ] );
    ]
