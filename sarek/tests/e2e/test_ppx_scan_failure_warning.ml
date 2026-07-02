(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Regression test: Sarek_ppx.scan_file_for_sarek_types's error handling
 * (Sarek_ppx.ml, formerly `with _ -> ()` at the end of the function).
 *
 * %sarek_include's scan target (fixture_scan_failure_include.ml.fixture) is
 * deliberately unparseable OCaml. Pre-fix, scan_file_for_sarek_types would
 * silently swallow the parse failure and this file would compile with no
 * trace that anything went wrong. Post-fix, the same parse failure is
 * printed to stderr (naming the file and the exception) during this file's
 * compilation - but compilation still succeeds, which this executable
 * existing and running proves. Run `dune build
 * sarek/tests/e2e/test_ppx_scan_failure_warning.exe --force` to see the
 * diagnostic on stderr.
 ******************************************************************************)

let%sarek_include _ = "fixture_scan_failure_include.ml.fixture"

let () = print_endline "test_ppx_scan_failure_warning: PASSED (build succeeded)"
