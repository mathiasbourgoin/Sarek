(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Kernel_args Tests - Verify Indexed Argument Container
 ******************************************************************************)

module KA = Spoc_framework.Kernel_args

(** Setting indices out of call order must not affect the extracted order:
    values always come back sorted by index. *)
let test_out_of_order_set () =
  let t = KA.create () in
  KA.set t 2 "c" ;
  KA.set t 0 "a" ;
  KA.set t 1 "b" ;
  match KA.validate_and_extract t ~expected_count:3 with
  | Ok arr ->
      Alcotest.(check (array string)) "sorted by index" [|"a"; "b"; "c"|] arr
  | Error msg -> Alcotest.fail msg

(** Duplicate idx: the later call wins. *)
let test_duplicate_idx_last_wins () =
  let t = KA.create () in
  KA.set t 1 "first" ;
  KA.set t 1 "second" ;
  KA.set t 0 "zero" ;
  match KA.validate_and_extract t ~expected_count:2 with
  | Ok arr ->
      Alcotest.(check (array string)) "last write wins" [|"zero"; "second"|] arr
  | Error msg -> Alcotest.fail msg

(** A gap in the indices is rejected and named in the error. *)
let test_gap_detection () =
  let t = KA.create () in
  KA.set t 0 "a" ;
  KA.set t 2 "c" ;
  match KA.validate_and_extract t ~expected_count:3 with
  | Ok _ -> Alcotest.fail "expected gap at index 1 to be rejected"
  | Error msg ->
      Alcotest.(check bool)
        "error names missing index 1"
        true
        (Str.string_match (Str.regexp ".*missing indices: \\[1\\].*") msg 0)

(** Setting more indices than expected_count is rejected as unexpected/extra. *)
let test_count_mismatch_extra () =
  let t = KA.create () in
  KA.set t 0 "a" ;
  KA.set t 1 "b" ;
  KA.set t 2 "c" ;
  match KA.validate_and_extract t ~expected_count:2 with
  | Ok _ -> Alcotest.fail "expected extra index 2 to be rejected"
  | Error msg ->
      Alcotest.(check bool)
        "error names unexpected index 2"
        true
        (Str.string_match (Str.regexp ".*2.*") msg 0)

(** An index beyond expected_count (with no gap below it) is also reported as
    unexpected. *)
let test_unexpected_index_beyond_range () =
  let t = KA.create () in
  KA.set t 0 "a" ;
  KA.set t 1 "b" ;
  KA.set t 2 "c" ;
  KA.set t 5 "f" ;
  match KA.validate_and_extract t ~expected_count:3 with
  | Ok _ -> Alcotest.fail "expected unexpected index 5 to be rejected"
  | Error msg ->
      Alcotest.(check bool)
        "error names unexpected index 5"
        true
        (Str.string_match (Str.regexp ".*5.*") msg 0)

let test_count () =
  let t = KA.create () in
  Alcotest.(check int) "empty" 0 (KA.count t) ;
  KA.set t 0 "a" ;
  KA.set t 3 "d" ;
  Alcotest.(check int) "two slots" 2 (KA.count t)

let test_to_sorted_list () =
  let t = KA.create () in
  KA.set t 3 "d" ;
  KA.set t 0 "a" ;
  KA.set t 1 "b" ;
  Alcotest.(check (list (pair int string)))
    "sorted by index, no contiguity required"
    [(0, "a"); (1, "b"); (3, "d")]
    (KA.to_sorted_list t)

(** Test suite *)
let () =
  Alcotest.run
    "Kernel_args"
    [
      ( "ordering",
        [
          Alcotest.test_case "out of order set" `Quick test_out_of_order_set;
          Alcotest.test_case "to_sorted_list" `Quick test_to_sorted_list;
          Alcotest.test_case "count" `Quick test_count;
        ] );
      ( "last-set-wins",
        [
          Alcotest.test_case
            "duplicate idx last wins"
            `Quick
            test_duplicate_idx_last_wins;
        ] );
      ( "validation",
        [
          Alcotest.test_case "gap detection" `Quick test_gap_detection;
          Alcotest.test_case
            "count mismatch extra"
            `Quick
            test_count_mismatch_extra;
          Alcotest.test_case
            "unexpected index beyond range"
            `Quick
            test_unexpected_index_beyond_range;
        ] );
    ]
