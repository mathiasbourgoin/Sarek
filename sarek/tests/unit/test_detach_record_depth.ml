(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * The depth bound on [Sarek_ir_interp_value.detach_record].
 *
 * backlog-172 round 4. [detach_record] is the copy-at-bind that makes a record
 * bound to a LOCAL a value rather than a window onto vector storage. It recurses
 * through [VRecord] fields and [VVariant] payloads, and had no depth or cycle
 * guard: on a cyclic [value] it would spin forever.
 *
 * A cyclic value is not constructible through the DSL — [@@sarek.type] refuses a
 * self-referential field type for lack of a layout, so no declaration can close
 * the loop (the argument is recorded on [detach_record] itself). It IS
 * constructible at the OCaml level, because [VRecord] carries a MUTABLE
 * [value array], and that is what these cases build. So the guard is checkable
 * without any device, and without waiting for the unreachability argument to
 * stop holding.
 *
 * Three cases, because a depth guard is exactly the shape that passes by
 * refusing everything:
 *
 *   1. A cyclic record TERMINATES with an error rather than hanging. This is the
 *      case the guard exists for, and it is run under a timeout-free but
 *      bounded shape: if the guard is removed this test does not fail, it hangs,
 *      so the red proof for it is "the runtest alias never finishes".
 *   2. A cyclic VARIANT payload does too — the second recursion arm, which a
 *      guard placed only on the record arm would miss.
 *   3. A legal deep nesting (below the bound) still COPIES correctly, at every
 *      level. Without this the guard could be [fun _ -> raise] and cases 1 and 2
 *      would still pass.
 ******************************************************************************)

module V = Sarek_interp.Sarek_ir_interp_value
module Interp_error = Sarek_interp.Interp_error

(* A record whose single field is the record itself. Built through the mutable
   [value array] [VRecord] carries — the only way in, since no [@@sarek.type]
   declaration can produce one. *)
let cyclic_record () : V.value =
  let fields = Array.make 1 V.VUnit in
  let r = V.VRecord ("cyclic", fields) in
  fields.(0) <- r ;
  r

(* The same back-edge through the VARIANT arm. [VVariant] carries a [value list],
   which is immutable, so the cycle is closed through a [VRecord] hop: the
   variant's payload is a record whose field is the variant. Both arms of
   [detach_record]'s recursion are therefore on the path. *)
let cyclic_variant () : V.value =
  let fields = Array.make 1 V.VUnit in
  let holder = V.VRecord ("holder", fields) in
  let var = V.VVariant ("cyclic_v", 0, [holder]) in
  fields.(0) <- var ;
  var

(* A legal chain of [depth] nested records with a scalar at the bottom. *)
let rec nest depth =
  if depth = 0 then V.VFloat32 42.0
  else V.VRecord ("n", [|V.VFloat32 (float_of_int depth); nest (depth - 1)|])

let rec leaf_of = function
  | V.VRecord (_, [|_; inner|]) -> leaf_of inner
  | v -> v

(* Does the copy share the [value array] with the original at any level? A
   shallow copy is the failure this pins: [detach_record]'s whole point is that
   [e.sub.p <- 42.0] must not reach an inner record still shared with storage. *)
let rec shares_array (a : V.value) (b : V.value) : bool =
  match (a, b) with
  | V.VRecord (_, fa), V.VRecord (_, fb) ->
      fa == fb
      ||
      let n = min (Array.length fa) (Array.length fb) in
      let rec go i = i < n && (shares_array fa.(i) fb.(i) || go (i + 1)) in
      go 0
  | _ -> false

let test_cyclic_record_terminates () =
  (* Terminates with an error. Without the guard this call does not return, so
     the red proof is a hang of the runtest alias rather than a failed
     assertion — stated here so a future reader does not mistake a timeout for
     an unrelated flake. *)
  match V.detach_record (cyclic_record ()) with
  | _ ->
      Alcotest.fail
        "detach_record returned a value for a CYCLIC record: it must raise, \
         not answer"
  | exception
      Interp_error.Interpreter_error
        (Interp_error.Unsupported_operation {operation; reason}) ->
      Alcotest.(check string)
        "the refusal names copy-at-bind"
        "record copy-at-bind"
        operation ;
      Alcotest.(check bool)
        "the refusal explains the depth bound"
        true
        (String.length reason > 0)

let test_cyclic_variant_terminates () =
  match V.detach_record (cyclic_variant ()) with
  | _ ->
      Alcotest.fail
        "detach_record returned a value for a CYCLIC variant payload: the \
         variant arm needs the same guard as the record arm"
  | exception
      Interp_error.Interpreter_error
        (Interp_error.Unsupported_operation {operation; _}) ->
      Alcotest.(check string)
        "the refusal names copy-at-bind"
        "record copy-at-bind"
        operation

let test_legal_nesting_still_copies () =
  (* Comfortably below the bound, and deeper than anything a kernel type can
     express. A guard that refused everything would fail here. *)
  let depth = 16 in
  let orig = nest depth in
  let copy = V.detach_record orig in
  Alcotest.(check bool)
    "the copy is not the original, at any level"
    false
    (shares_array orig copy) ;
  match (leaf_of orig, leaf_of copy) with
  | V.VFloat32 a, V.VFloat32 b ->
      Alcotest.(check (float 1e-6)) "the deepest leaf survived the copy" a b
  | _ -> Alcotest.fail "the nesting did not survive detach_record"

let () =
  Alcotest.run
    "detach_record depth bound"
    [
      ( "cycles terminate",
        [
          Alcotest.test_case "cyclic record" `Quick test_cyclic_record_terminates;
          Alcotest.test_case
            "cyclic variant payload"
            `Quick
            test_cyclic_variant_terminates;
        ] );
      ( "legal nesting is unaffected",
        [
          Alcotest.test_case
            "deep but legal nesting still deep-copies"
            `Quick
            test_legal_nesting_still_copies;
        ] );
    ]
