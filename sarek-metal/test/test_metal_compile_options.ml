(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * MTLCompileOptions fail-soft contract (#125).
 *
 * READ THIS BEFORE TREATING A GREEN RUN AS EVIDENCE OF ANYTHING.
 *
 * #125 makes the Metal backend ask for fastMathEnabled=NO, because Metal's
 * default is fast math ON and the binding previously ignored its options
 * argument outright. NONE of that has been executed on Apple hardware — there
 * is no Apple device on the machine it was written on, and this test does NOT
 * verify it. Whether Metal actually honours the request, and whether Metal
 * float results then agree with the interpreter, are both OPEN. See
 * docs/fp-contraction-policy.md §11.
 *
 * What this file DOES verify, and it is the only claim it makes: the
 * FAIL-SOFT mechanism that is the entire safety argument for landing
 * unverified Objective-C. [mtl_compile_options_conformant] must return [None]
 * — never raise — when the Objective-C runtime is unavailable, because [None]
 * makes the caller pass null options, which is exactly the behaviour that
 * shipped before #125. A host with no libobjc (this Linux box, and CI) is a
 * perfectly good instrument for that one question, and it is the strongest
 * failure the mechanism has to survive: nothing resolves at all.
 *
 * On a Mac this test asserts nothing beyond "it returned without raising",
 * and says so.
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
           every failure degrades to the pre-#125 behaviour (null options); an \
           escaping exception breaks Metal kernel compilation outright on any \
           host where the runtime, the class or the selector is missing."
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
