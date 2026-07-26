(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * MTLCompileOptions fail-soft contract (backlog #125).
 *
 * SCOPE. The Metal FP behaviour itself is now MEASURED on hardware — Apple M4,
 * macOS 15.6.1 (24G90), Apple clang 17.0.0 — by
 * tools/probes/metal_math_mode_probe.m and
 * tools/probes/metal_contraction_barrier_probe.m, written up in
 * docs/fp-contraction-policy.md §10. Those probes, not this file, are what
 * establish that Metal's defaults are fast on both knobs, that the options are
 * honoured, and that contraction needs a source pragma instead.
 *
 * This file verifies the one thing those probes cannot: the FAIL-SOFT
 * mechanism, on a host where the Objective-C runtime is ABSENT.
 * [mtl_compile_options_conformant] must return [None] — never raise — because
 * [None] makes the caller pass null options, which is exactly the behaviour
 * that shipped before backlog #125. A machine with no libobjc (this Linux box,
 * and CI) is the right instrument for that question, and it is the strongest
 * failure the mechanism has to survive: nothing resolves at all.
 *
 * On a Mac this test cannot reach that path and skips, rather than passing
 * while checking nothing.
 ******************************************************************************)

open Sarek_metal

(* The fail-soft path is only exercised where the Objective-C runtime is
   ABSENT. On a Mac this test cannot reach it, and rather than pretend
   otherwise it skips — a skip is not a pass, and this one is honest about
   which machine can answer the question. *)
let objc_runtime_absent () = not (Metal_bindings.is_available ())

let test_returns_none_without_objc () =
  if not (objc_runtime_absent ()) then begin
    Printf.printf
      "[SKIP] Metal is available on this host, so the no-Objective-C fail-soft \
       path cannot be reached here\n\
       %!" ;
    Alcotest.skip ()
  end
  else
    match Metal_bindings.mtl_compile_options_conformant () with
    | None -> ()
    | Some _ ->
        Alcotest.fail
          "mtl_compile_options_conformant returned Some on a host with no \
           Objective-C runtime; it cannot have built a real MTLCompileOptions, \
           so this is a bogus pointer heading for newLibraryWithSource:"
    | exception e ->
        Alcotest.failf
          "mtl_compile_options_conformant raised %s instead of returning None. \
           The safety argument for landing this unverified Objective-C is that \
           every failure degrades to the behaviour before backlog #125 (null \
           options); an escaping exception breaks Metal kernel compilation \
           outright on any host where the runtime, the class or the selector \
           is missing."
          (Printexc.to_string e)

(* Called twice: a first call that succeeded in caching something bad, or a
   lazy that raised once and is now poisoned, must not change the answer. *)
let test_is_repeatable () =
  if not (objc_runtime_absent ()) then Alcotest.skip ()
  else
    let attempt n =
      match Metal_bindings.mtl_compile_options_conformant () with
      | None -> ()
      | Some _ -> Alcotest.failf "call %d returned Some without a runtime" n
      | exception e ->
          Alcotest.failf "call %d raised %s" n (Printexc.to_string e)
    in
    attempt 1 ;
    attempt 2 ;
    attempt 3

let () =
  Alcotest.run
    "Metal_compile_options"
    [
      ( "fail-soft without an Objective-C runtime",
        [
          Alcotest.test_case
            "returns None rather than raising"
            `Quick
            test_returns_none_without_objc;
          Alcotest.test_case "is repeatable" `Quick test_is_repeatable;
        ] );
    ]
