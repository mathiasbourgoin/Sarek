(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Unit tests for Sarek_ppx.scan_file_for_sarek_types's error handling.
 *
 * Pre-fix, this function wrapped its whole body in `with _ -> ()`, silently
 * discarding the scanned file's name and the exception on ANY failure
 * (unreadable file, parse error, or a registration error triggered by
 * malformed [@sarek.*] content) while still letting compilation of the
 * *caller* succeed. Post-fix it prints a diagnostic naming the file and the
 * exception to stderr (via Printf.eprintf) and also returns
 * `string option`: `None` on success, `Some diagnostic` on failure - so the
 * failure is visible (in the build log) without breaking the "compilation
 * still succeeds" property (an `[@ocaml.ppwarning ...]` attribute was
 * considered and rejected: this project's dune "dev" profile promotes
 * warning 22 to a hard error, which would defeat that same property - see
 * Sarek_ppx.ml's scan_file_for_sarek_types doc comment for the full
 * rationale).
 *
 * This test exercises the return-value contract directly (the mechanism
 * that is actually assertable): a malformed fixture must yield
 * `Some diagnostic` naming the file and mentioning a parse failure; a
 * well-formed fixture must yield `None`. The "compilation still succeeds"
 * half of the contract is proven separately by
 * sarek/tests/e2e/test_ppx_scan_failure_warning.ml, which %sarek_includes
 * a malformed sibling file and must still build.
 *
 * Fixture content is written to real temp files at runtime (rather than
 * checked-in fixture files copied in by dune `(deps ...)`) so the test does
 * not depend on dune's deps-vs-runtest-action copying semantics working the
 * same way under both `dune build`/`dune runtest` and `dune exec`.
 ******************************************************************************)

(* Small substring check (avoids pulling in an extra string-utility library
   just for this one test). *)
let contains ~needle haystack =
  let nlen = String.length needle in
  let hlen = String.length haystack in
  let rec go i =
    if i + nlen > hlen then false
    else if String.sub haystack i nlen = needle then true
    else go (i + 1)
  in
  nlen = 0 || go 0

let with_temp_file ~suffix contents f =
  let path = Filename.temp_file "sarek_scan_test" suffix in
  let oc = open_out path in
  output_string oc contents ;
  close_out oc ;
  Fun.protect
    ~finally:(fun () -> try Sys.remove path with Sys_error _ -> ())
    (fun () -> f path)

let unparseable_contents = "let broken = fun x -> x +\n"

let wellformed_contents = "let ok = 1 + 1\n"

let test_scan_reports_unparseable_file () =
  with_temp_file ~suffix:".ml" unparseable_contents (fun path ->
      (* Pre-fix this call returned unit unconditionally and there was
         nothing to assert; the function's whole point pre-fix was to be
         silent. Post-fix, scanning a file that fails to parse must surface
         a diagnostic naming both the file and the failure - not swallow
         it. *)
      match Sarek_ppx.scan_file_for_sarek_types path with
      | None ->
          Alcotest.fail
            "expected Some diagnostic for an unparseable file, got None \
             (pre-fix behavior: failure silently swallowed)"
      | Some msg ->
          Alcotest.(check bool)
            "diagnostic names the scanned file"
            true
            (contains ~needle:path msg))

let test_scan_ok_on_wellformed_file () =
  with_temp_file ~suffix:".ml" wellformed_contents (fun path ->
      match Sarek_ppx.scan_file_for_sarek_types path with
      | None -> ()
      | Some msg ->
          Alcotest.failf
            "expected None for a well-formed file with no [@sarek.*] \
             declarations, got Some %S"
            msg)

let () =
  Alcotest.run
    "ppx_scan_diagnostics"
    [
      ( "scan_file_for_sarek_types",
        [
          ( "unparseable file yields Some diagnostic naming the file",
            `Quick,
            test_scan_reports_unparseable_file );
          ( "well-formed file yields None",
            `Quick,
            test_scan_ok_on_wellformed_file );
        ] );
    ]
