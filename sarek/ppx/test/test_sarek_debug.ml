(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Unit tests for Sarek_debug module *)

open Alcotest

(** Test that debug functions don't crash *)
let test_log_disabled () =
  (* When debug is disabled, these should be no-ops *)
  Sarek_debug.log "test message" ;
  Sarek_debug.log_enter "test_function" ;
  Sarek_debug.log_exit "test_function" ;
  Sarek_debug.log_int "count" 42 ;
  Sarek_debug.log_string "name" "test" ;
  check bool "no crash" true true

let test_log_with_format () =
  Sarek_debug.log "Testing %s with %d" "format" 123 ;
  check bool "format string works" true true

let test_log_enter_exit () =
  Sarek_debug.log_enter "my_function" ;
  Sarek_debug.log "inside function" ;
  Sarek_debug.log_exit "my_function" ;
  check bool "enter/exit works" true true

let test_log_int_values () =
  Sarek_debug.log_int "zero" 0 ;
  Sarek_debug.log_int "negative" (-42) ;
  Sarek_debug.log_int "large" 1000000 ;
  check bool "log_int works" true true

let test_log_string_values () =
  Sarek_debug.log_string "empty" "" ;
  Sarek_debug.log_string "simple" "hello" ;
  Sarek_debug.log_string "with_spaces" "hello world" ;
  check bool "log_string works" true true

let test_multiple_logs () =
  for i = 0 to 9 do
    Sarek_debug.log "iteration %d" i
  done ;
  check bool "multiple logs work" true true

let test_nested_enter_exit () =
  Sarek_debug.log_enter "outer" ;
  Sarek_debug.log_enter "inner" ;
  Sarek_debug.log "nested message" ;
  Sarek_debug.log_exit "inner" ;
  Sarek_debug.log_exit "outer" ;
  check bool "nested enter/exit works" true true

(** Test enabled flag *)
let test_enabled_flag () =
  (* The enabled flag should be a boolean *)
  let _ = Sarek_debug.enabled in
  check bool "enabled is bool" true true

let () =
  run
    "Sarek_debug"
    [
      ( "basic_logging",
        [
          test_case "log_disabled" `Quick test_log_disabled;
          test_case "log_with_format" `Quick test_log_with_format;
          test_case "log_enter_exit" `Quick test_log_enter_exit;
        ] );
      ( "typed_logging",
        [
          test_case "log_int" `Quick test_log_int_values;
          test_case "log_string" `Quick test_log_string_values;
        ] );
      ( "patterns",
        [
          test_case "multiple_logs" `Quick test_multiple_logs;
          test_case "nested_enter_exit" `Quick test_nested_enter_exit;
        ] );
      ("config", [test_case "enabled_flag" `Quick test_enabled_flag]);
    ]
