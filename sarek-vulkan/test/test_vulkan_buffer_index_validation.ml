(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Key-level (non-hardware) regression test for
 * Vulkan_api_kernel.validate_buffer_indices.
 *
 * Pre-fix behavior (not reproducible here since the old code path is gone):
 * resolve_bindings silently compressed whatever set of buffer indices was
 * present into dense, ascending descriptor bindings via List.mapi, with no
 * check that the index set made any sense - negative indices, or simply the
 * wrong number of buffers (e.g. one dropped, another duplicated), passed
 * through unnoticed and produced a plausible-looking but wrong binding
 * layout. This test exercises validate_buffer_indices directly, without any
 * Vulkan device, proving bad index sets are now rejected before
 * resolve_bindings ever sees them.
 ******************************************************************************)

open Sarek_vulkan

(* Buffers are wrapped in Vulkan_api_kernel.any_buffer via the existential
   AnyBuf constructor, which requires a real Vulkan_api_memory.buffer value.
   validate_buffer_indices only inspects indices, not the stored values, so a
   [unit Kernel_args.t] is sufficient here - no device or real buffer needed. *)
let store_of_indices idxs =
  let store = Spoc_framework.Kernel_args.create () in
  List.iter (fun idx -> Spoc_framework.Kernel_args.set store idx ()) idxs ;
  store

let expect_ok label result =
  match result with
  | Ok () -> ()
  | Error msg -> Alcotest.failf "%s: expected Ok, got Error %s" label msg

let expect_error_containing label substring result =
  match result with
  | Ok () -> Alcotest.failf "%s: expected Error, got Ok" label
  | Error msg ->
      Alcotest.(check bool)
        (Printf.sprintf "%s: error mentions %S (got %S)" label substring msg)
        true
        (Str.string_match
           (Str.regexp (".*" ^ Str.quote substring ^ ".*"))
           msg
           0)

let test_valid_sparse_indices_ok () =
  (* idx shares its numbering space with scalar args, so buffer indices are
     legitimately sparse/non-contiguous - e.g. (vec a, float scale, vec b)
     gives buffer indices {0; 3}, not {0; 1}. *)
  let store = store_of_indices [0; 3] in
  expect_ok
    "sparse-but-correct-count indices"
    (Vulkan_api_kernel.validate_buffer_indices ~expected_count:2 store)

let test_negative_index_rejected () =
  let store = store_of_indices [0; -1] in
  expect_error_containing
    "negative index"
    "negative buffer index"
    (Vulkan_api_kernel.validate_buffer_indices ~expected_count:2 store)

let test_wrong_count_too_few_rejected () =
  (* Only one buffer bound but the kernel expects two: a caller-side idx typo
     dropped a buffer instead of registering it under its own idx. *)
  let store = store_of_indices [0] in
  expect_error_containing
    "too few buffers"
    "expected 2 buffer"
    (Vulkan_api_kernel.validate_buffer_indices ~expected_count:2 store)

let test_wrong_count_too_many_rejected () =
  let store = store_of_indices [0; 1; 2] in
  expect_error_containing
    "too many buffers"
    "expected 2 buffer"
    (Vulkan_api_kernel.validate_buffer_indices ~expected_count:2 store)

let test_empty_store_matches_zero_expected () =
  let store = store_of_indices [] in
  expect_ok
    "no buffers expected, none bound"
    (Vulkan_api_kernel.validate_buffer_indices ~expected_count:0 store)

let () =
  Alcotest.run
    "Vulkan_buffer_index_validation"
    [
      ( "validate_buffer_indices",
        [
          Alcotest.test_case
            "valid sparse indices accepted"
            `Quick
            test_valid_sparse_indices_ok;
          Alcotest.test_case
            "negative index rejected"
            `Quick
            test_negative_index_rejected;
          Alcotest.test_case
            "too few buffers rejected"
            `Quick
            test_wrong_count_too_few_rejected;
          Alcotest.test_case
            "too many buffers rejected"
            `Quick
            test_wrong_count_too_many_rejected;
          Alcotest.test_case
            "empty store matches zero expected"
            `Quick
            test_empty_store_matches_zero_expected;
        ] );
    ]
