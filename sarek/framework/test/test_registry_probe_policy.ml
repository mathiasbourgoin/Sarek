(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Regression test for the Framework_registry error-swallowing fix.
 *
 * WHY THIS FILE EXISTS
 *   `Framework_registry` used to filter its availability lists with
 *   `try is_available () with _ -> false`, so an Out_of_memory or a
 *   Stack_overflow raised inside a backend's probe was reported to the user as
 *   "no backends found" — a real failure rendered as an empty result. Two
 *   commits fixed it: 22745124 introduced the re-raise, but only in
 *   `available`; 339c8e2c factored the policy into `probe_available` and
 *   applied it to ALL THREE accessors (`available`, `available_backends`,
 *   `best_backend`). The 2026-07-29 audit flagged the whole thing as untested,
 *   and specifically noted that its own earlier patch had covered one accessor
 *   of three.
 *
 *   So this file tests all three, symmetrically, rather than the one that
 *   happened to be fixed first. Reverting `Framework_registry.ml` to 22745124
 *   leaves `available` green and turns `available_backends` and `best_backend`
 *   red, which is the distinction the audit is pointing at.
 *
 * BOTH POLARITIES
 *   Good behaviour: an ORDINARY probe failure still counts as unavailable and
 *   escapes as no exception at all. A "fix" that simply removed the handler
 *   would satisfy the fatal cases and break these.
 *   Bad behaviour absent: each of Out_of_memory, Stack_overflow and Sys.Break
 *   must PROPAGATE out of each accessor rather than being turned into a silent
 *   absence.
 *
 * NON-VACUITY
 *   Every case asserts the probe was actually CALLED (`probe_calls`). Without
 *   that, an accessor which never reached the probe — because the backend was
 *   not registered, or because the table was empty — would satisfy "no
 *   exception escaped" and "the list does not contain ProbeBackend" perfectly.
 *   The `Ok` cases pin the other end: the backend IS found when its probe says
 *   so, so "absent" is a decision and not a default.
 ******************************************************************************)

open Spoc_framework.Framework_sig
open Spoc_framework_registry

let () =
  Framework_registry.register_backend
    ~priority:42
    (module Probe_backend.Probe_backend)

let plugin_names l = List.map (fun (module P : S) -> P.name) l

let backend_names l = List.map (fun (module B : BACKEND) -> B.name) l

(** Run [f] with the probe in [mode], and report how many times the probe was
    hit. *)
let with_mode mode f =
  Probe_backend.mode := mode ;
  Probe_backend.probe_calls := 0 ;
  let r = f () in
  (r, !Probe_backend.probe_calls)

let check_probed n =
  Alcotest.(check bool)
    "the probe was actually called (else this case is vacuous)"
    true
    (n > 0)

(* ---------------------------------------------------------------- *)
(* Non-vacuity controls: the probe decides, in both directions.      *)
(* ---------------------------------------------------------------- *)

let available_finds_the_backend () =
  let names, n =
    with_mode Probe_backend.Ok (fun () ->
        plugin_names (Framework_registry.available ()))
  in
  check_probed n ;
  Alcotest.(check bool)
    "an available probe puts the backend in available ()"
    true
    (List.mem Probe_backend.backend_name names)

let available_backends_finds_the_backend () =
  let names, n =
    with_mode Probe_backend.Ok (fun () ->
        backend_names (Framework_registry.available_backends ()))
  in
  check_probed n ;
  Alcotest.(check bool)
    "an available probe puts the backend in available_backends ()"
    true
    (List.mem Probe_backend.backend_name names)

let best_backend_finds_the_backend () =
  let r, n =
    with_mode Probe_backend.Ok (fun () -> Framework_registry.best_backend ())
  in
  check_probed n ;
  Alcotest.(check bool)
    "an available probe makes best_backend () return it"
    true
    (match r with
    | Some (module B : BACKEND) -> B.name = Probe_backend.backend_name
    | None -> false)

let unavailable_is_excluded () =
  let names, n =
    with_mode Probe_backend.Unavailable (fun () ->
        plugin_names (Framework_registry.available ()))
  in
  check_probed n ;
  Alcotest.(check bool)
    "a probe returning false excludes the backend"
    false
    (List.mem Probe_backend.backend_name names)

(* ---------------------------------------------------------------- *)
(* GOOD BEHAVIOUR: an ordinary probe failure = unavailable, no raise *)
(* ---------------------------------------------------------------- *)

let ordinary_failure_is_unavailable_in_available () =
  let names, n =
    with_mode Probe_backend.Ordinary (fun () ->
        plugin_names (Framework_registry.available ()))
  in
  check_probed n ;
  Alcotest.(check bool)
    "available (): ordinary probe failure excludes, does not raise"
    false
    (List.mem Probe_backend.backend_name names)

let ordinary_failure_is_unavailable_in_available_backends () =
  let names, n =
    with_mode Probe_backend.Ordinary (fun () ->
        backend_names (Framework_registry.available_backends ()))
  in
  check_probed n ;
  Alcotest.(check bool)
    "available_backends (): ordinary probe failure excludes, does not raise"
    false
    (List.mem Probe_backend.backend_name names)

let ordinary_failure_is_unavailable_in_best_backend () =
  let r, n =
    with_mode Probe_backend.Ordinary (fun () ->
        Framework_registry.best_backend ())
  in
  check_probed n ;
  Alcotest.(check bool)
    "best_backend (): ordinary probe failure excludes, does not raise"
    true
    (r = None)

(* ---------------------------------------------------------------- *)
(* BAD BEHAVIOUR ABSENT: fatal exceptions propagate, all 3 accessors *)
(* ---------------------------------------------------------------- *)

let fatal_exceptions = [Out_of_memory; Stack_overflow; Sys.Break]

let name_of_exn = Printexc.to_string

(** [accessor] is forced to a value, so a lazy Seq.filter cannot postpone the
    raise past the assertion. *)
let fatal_propagates ~what accessor exn () =
  Probe_backend.mode := Probe_backend.Fatal exn ;
  Probe_backend.probe_calls := 0 ;
  Alcotest.check_raises
    (Printf.sprintf
       "%s must propagate %s, not swallow it"
       what
       (name_of_exn exn))
    exn
    (fun () -> accessor ()) ;
  check_probed !Probe_backend.probe_calls

let fatal_cases what accessor =
  List.map
    (fun exn ->
      Alcotest.test_case
        (Printf.sprintf "%s propagates %s" what (name_of_exn exn))
        `Quick
        (fatal_propagates ~what accessor exn))
    fatal_exceptions

let () =
  Alcotest.run
    "Framework_registry probe policy"
    [
      ( "the probe decides (non-vacuity controls)",
        [
          Alcotest.test_case
            "available finds it"
            `Quick
            available_finds_the_backend;
          Alcotest.test_case
            "available_backends finds it"
            `Quick
            available_backends_finds_the_backend;
          Alcotest.test_case
            "best_backend finds it"
            `Quick
            best_backend_finds_the_backend;
          Alcotest.test_case "false excludes it" `Quick unavailable_is_excluded;
        ] );
      ( "ordinary probe failure counts as unavailable",
        [
          Alcotest.test_case
            "available"
            `Quick
            ordinary_failure_is_unavailable_in_available;
          Alcotest.test_case
            "available_backends"
            `Quick
            ordinary_failure_is_unavailable_in_available_backends;
          Alcotest.test_case
            "best_backend"
            `Quick
            ordinary_failure_is_unavailable_in_best_backend;
        ] );
      ( "fatal probe failure propagates - available",
        fatal_cases "available" (fun () ->
            ignore (Framework_registry.available ())) );
      ( "fatal probe failure propagates - available_backends",
        fatal_cases "available_backends" (fun () ->
            ignore (Framework_registry.available_backends ())) );
      ( "fatal probe failure propagates - best_backend",
        fatal_cases "best_backend" (fun () ->
            ignore (Framework_registry.best_backend ())) );
    ]
