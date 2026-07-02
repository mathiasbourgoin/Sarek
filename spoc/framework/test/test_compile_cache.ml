(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Compile_cache Tests - Verify Shared Cache-Key Builder
 *
 * These are key-level (non-hardware) tests: they exercise the pure key
 * builder, not an actual backend compile pipeline. Executable, hardware-
 * verified end-to-end coverage for the Vulkan/OpenCL in-memory pipeline
 * caches lives in sarek-vulkan/test and sarek-opencl/test respectively.
 ******************************************************************************)

module CC = Spoc_framework.Compile_cache

let test_same_inputs_same_key () =
  let key1 = CC.make_key ~device:"0" ~name:"kernel_a" ~source:"code" () in
  let key2 = CC.make_key ~device:"0" ~name:"kernel_a" ~source:"code" () in
  Alcotest.(check string) "identical inputs produce identical keys" key1 key2

(** Non-vacuous regression test: this is precisely the CUDA/Vulkan bug fixed in
    this change. Two kernels compiled from the same source string, on the same
    device, must never share a cache key - before the fix the key was built from
    device + source digest only, so [kernel_b] would silently resolve to
    [kernel_a]'s cached handle. *)
let test_same_source_different_name_different_key () =
  let shared_source = "source defining both kernel_a and kernel_b" in
  let key_a =
    CC.make_key ~device:"0" ~name:"kernel_a" ~source:shared_source ()
  in
  let key_b =
    CC.make_key ~device:"0" ~name:"kernel_b" ~source:shared_source ()
  in
  Alcotest.(check bool)
    "different kernel names in the same source must not collide"
    true
    (key_a <> key_b)

let test_different_device_different_key () =
  let key0 = CC.make_key ~device:"0" ~name:"k" ~source:"code" () in
  let key1 = CC.make_key ~device:"1" ~name:"k" ~source:"code" () in
  Alcotest.(check bool) "different devices must not collide" true (key0 <> key1)

let test_different_source_different_key () =
  let key1 = CC.make_key ~device:"0" ~name:"k" ~source:"code1" () in
  let key2 = CC.make_key ~device:"0" ~name:"k" ~source:"code2" () in
  Alcotest.(check bool) "different sources must not collide" true (key1 <> key2)

let test_options_are_canonicalized () =
  let key1 =
    CC.make_key
      ~device:"0"
      ~name:"k"
      ~source:"code"
      ~options:[("b", "2"); ("a", "1")]
      ()
  in
  let key2 =
    CC.make_key
      ~device:"0"
      ~name:"k"
      ~source:"code"
      ~options:[("a", "1"); ("b", "2")]
      ()
  in
  Alcotest.(check string) "option order does not affect the key" key1 key2

let test_different_options_different_key () =
  let key1 =
    CC.make_key ~device:"0" ~name:"k" ~source:"code" ~options:[("opt", "1")] ()
  in
  let key2 =
    CC.make_key ~device:"0" ~name:"k" ~source:"code" ~options:[("opt", "2")] ()
  in
  Alcotest.(check bool)
    "different option values must not collide"
    true
    (key1 <> key2)

let () =
  Alcotest.run
    "Compile_cache"
    [
      ( "key_identity",
        [
          Alcotest.test_case
            "same inputs same key"
            `Quick
            test_same_inputs_same_key;
          Alcotest.test_case
            "same source, different kernel name -> different key"
            `Quick
            test_same_source_different_name_different_key;
          Alcotest.test_case
            "different device -> different key"
            `Quick
            test_different_device_different_key;
          Alcotest.test_case
            "different source -> different key"
            `Quick
            test_different_source_different_key;
        ] );
      ( "options",
        [
          Alcotest.test_case
            "option order canonicalized"
            `Quick
            test_options_are_canonicalized;
          Alcotest.test_case
            "different option values -> different key"
            `Quick
            test_different_options_different_key;
        ] );
    ]
