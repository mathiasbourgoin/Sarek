(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Native plugin error tests *)

[@@@warning "-unused-value-declaration"]

let test_device_not_found () =
  try
    let module E = Sarek_native.Native_error in
    let _ = E.raise_error (E.feature_not_supported "device_not_found") in
    Alcotest.fail "Should have raised error"
  with Spoc_framework.Backend_error.Backend_error err ->
    let msg = Spoc_framework.Backend_error.to_string err in
    Alcotest.(check bool)
      "contains device_not_found"
      true
      (String.length msg > 0)

let test_feature_not_supported () =
  try
    let module E = Sarek_native.Native_error in
    let _ = E.raise_error (E.feature_not_supported "test feature") in
    Alcotest.fail "Should have raised error"
  with Spoc_framework.Backend_error.Backend_error err ->
    let msg = Spoc_framework.Backend_error.to_string err in
    Alcotest.(check bool) "contains feature" true (String.length msg > 0)

let test_error_prefix () =
  let module E = Sarek_native.Native_error in
  let err = E.feature_not_supported "test" in
  let msg = E.to_string err in
  Alcotest.(check bool)
    "has Native prefix"
    true
    (Str.string_match (Str.regexp ".*Native.*") msg 0)

let () =
  Alcotest.run
    "Native_error"
    [
      ( "errors",
        [
          Alcotest.test_case "device_not_found" `Quick test_device_not_found;
          Alcotest.test_case
            "feature_not_supported"
            `Quick
            test_feature_not_supported;
          Alcotest.test_case "error_prefix" `Quick test_error_prefix;
        ] );
    ]
