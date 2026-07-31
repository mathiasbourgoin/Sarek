(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** backlog-206, Interpreter half, WITHOUT a device.

    The end-to-end evidence for this fix lives in
    [sarek/tests/e2e/test_shared_record_slots.ml], which enumerates devices — so
    on CI, which has no GPU, it exercises the Interpreter and Native only, and
    on a host where the CPU plugins somehow did not register it would assert
    nothing at all. This file checks the interpreter's array allocator directly:
    it needs no device, no PPX and no kernel, so it holds wherever the test
    suite runs.

    Three properties, and they are not the same property.

    1. A record element type produces a [VRecord] with zeroed fields. Before the
    fix it produced [VUnit], which is why [s.(i).a <- e] raised "Expected
    record, got: assignment target of .a (got unit)" on the Interpreter while
    Native accepted the same store.

    2. The slots are DISTINCT allocations. [VRecord] carries a MUTABLE
    [value array], so [Array.make size (default_value ...)] would put one array
    in every slot and a field store through one index would be visible through
    all of them — exactly the Native defect, reproduced in the interpreter. Only
    [Array.init] avoids it, and only a mutation-then-read check can tell the two
    apart: both spellings give a structurally equal array.

    3. A VARIANT element type produces a value the interpreter's own matcher can
    select an arm for, carrying the same constructor Native puts there. The tag
    is [Hashtbl.hash ctor mod 256], not a positional index, and the first
    version of the fix stored a literal [0] — structurally a fine [VVariant],
    and unmatchable. Pinned against [variant_tag_of_ctor] rather than a
    hard-coded number. *)

open Alcotest
open Sarek_ir_types
open Sarek_interp.Sarek_ir_interp_value

let tri : elttype =
  TRecord ("tri", [("a", TFloat32); ("b", TFloat32); ("c", TInt32)])

let field_f32 v i =
  match v with
  | VRecord (_, fields) -> (
      match fields.(i) with
      | VFloat32 f -> f
      | _ -> failf "field %d is not a VFloat32" i)
  | _ -> failf "not a VRecord"

let test_default_record_is_zeroed () =
  match default_value_of_elttype tri with
  | VRecord (name, fields) -> (
      check string "record name" "tri" name ;
      check int "field count" 3 (Array.length fields) ;
      check (float 0.0) "a" 0.0 (field_f32 (VRecord (name, fields)) 0) ;
      check (float 0.0) "b" 0.0 (field_f32 (VRecord (name, fields)) 1) ;
      match fields.(2) with
      | VInt32 n -> check int32 "c" 0l n
      | _ -> fail "field c is not a VInt32")
  | v ->
      failf
        "a record element type must produce a VRecord, got %s"
        (match v with
        | VUnit -> "VUnit (the pre-fix behaviour)"
        | _ -> "some other value")

(* The tag a default variant slot carries must be the tag the interpreter's own
   matcher looks for, NOT a positional index. [EMatch]/[SMatch] select an arm
   with [variant_tag_of_ctor name = tag]; the first version of this default
   stored a literal [0], which matches the chosen constructor only if its name
   happens to hash to zero, so reading a default slot raised "Pattern match
   failure in SMatch" (measured: Interpreter x2 raised where Native answered the
   nullary constructor). Asserted as the MATCHER'S PREDICATE rather than as a
   hard-coded number, so the two cannot drift apart. *)
let test_default_variant_is_matchable () =
  match
    default_value_of_elttype (TVariant ("choice", [("Zero", []); ("One", [])]))
  with
  | VVariant (name, tag, args) ->
      check string "variant name" "choice" name ;
      check
        bool
        "the matcher selects the chosen constructor's arm"
        true
        (variant_tag_of_ctor "Zero" = tag) ;
      check
        bool
        "and does NOT select the other constructor's arm"
        false
        (variant_tag_of_ctor "One" = tag) ;
      check int "no payload" 0 (List.length args)
  | _ -> fail "a variant element type must produce a VVariant"

(* Which constructor: the first NULLARY one, matching what the Native backend's
   [Sarek_native_helpers.default_value_for_type] puts in the same slot. A
   CPU-backend disagreement about a freshly declared shared array is the shape
   of divergence backlog-206 was filed as, so the agreement is pinned. *)
let test_default_variant_prefers_nullary () =
  match
    default_value_of_elttype (TVariant ("c2", [("A", [TFloat32]); ("B", [])]))
  with
  | VVariant (_, tag, args) ->
      check
        bool
        "B, the nullary constructor, not A which comes first"
        true
        (variant_tag_of_ctor "B" = tag) ;
      check int "no payload" 0 (List.length args)
  | _ -> fail "a variant element type must produce a VVariant"

(* No nullary constructor anywhere: fall back to the first one, payload zeroed.
   Native does the same. *)
let test_default_variant_without_nullary () =
  match
    default_value_of_elttype
      (TVariant ("pair", [("L", [TInt32]); ("R", [TFloat32])]))
  with
  | VVariant (_, tag, args) -> (
      check bool "L, the first constructor" true (variant_tag_of_ctor "L" = tag) ;
      match args with
      | [VInt32 n] -> check int32 "payload zeroed" 0l n
      | _ -> fail "payload is not a single zeroed VInt32")
  | _ -> fail "a variant element type must produce a VVariant"

(* THE property. Mutate slot 0 in place and require the other slots to be
   unchanged AND physically distinct. The value check is what a user sees; the
   [!=] check is the direct statement, and each would pass under a different
   plausible wrong fix, so both are asserted. *)
let test_slots_are_independent () =
  let arr = alloc_kernel_array tri 4 in
  check int "length" 4 (Array.length arr) ;
  (match arr.(0) with
  | VRecord (_, fields) -> fields.(0) <- VFloat32 7.0
  | _ -> fail "slot 0 is not a VRecord") ;
  check (float 0.0) "slot 0 was written" 7.0 (field_f32 arr.(0) 0) ;
  for i = 1 to 3 do
    check
      (float 0.0)
      (Printf.sprintf "slot %d untouched" i)
      0.0
      (field_f32 arr.(i) 0) ;
    check
      bool
      (Printf.sprintf "slot 0 and slot %d are distinct allocations" i)
      true
      (arr.(0) != arr.(i))
  done

let test_scalar_slots_still_zeroed () =
  let arr = alloc_kernel_array TInt32 3 in
  check int "length" 3 (Array.length arr) ;
  Array.iteri
    (fun i v ->
      match v with
      | VInt32 n -> check int32 (Printf.sprintf "slot %d" i) 0l n
      | _ -> failf "slot %d is not a VInt32" i)
    arr

let () =
  run
    "interp shared array slots"
    [
      ( "backlog-206",
        [
          test_case
            "record default is zeroed"
            `Quick
            test_default_record_is_zeroed;
          test_case
            "variant default is matchable by the interpreter"
            `Quick
            test_default_variant_is_matchable;
          test_case
            "variant default prefers the nullary constructor"
            `Quick
            test_default_variant_prefers_nullary;
          test_case
            "variant default without a nullary constructor"
            `Quick
            test_default_variant_without_nullary;
          test_case
            "slots are independent allocations"
            `Quick
            test_slots_are_independent;
          test_case
            "scalar slots still zeroed"
            `Quick
            test_scalar_slots_still_zeroed;
        ] );
    ]
