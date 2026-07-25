(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Locations carried by Sarek's own AST must survive the round trip back to
 * ppxlib, because that location is what the OCaml driver uses to SLICE THE
 * SOURCE FILE when it renders a Sarek diagnostic.
 *
 * [Lexing.position] is not (line, column): [pos_cnum] is the ABSOLUTE byte
 * offset from the start of the file and [pos_bol] is the absolute offset of
 * the start of the current line, so the column is [pos_cnum - pos_bol]. A
 * conversion that keeps only line+column and then rebuilds a position with
 * [pos_bol = 0; pos_cnum = column] produces a position whose column is right
 * but whose absolute offset points at byte `column` of the FILE.
 *
 * The user-visible consequence is the #97 report: a real, correct Sarek type
 * error is rendered against unrelated bytes taken from the top of the file.
 * Observed on this tree before the fix, for a polymorphic [@sarek.module]
 * helper instantiated at float64:
 *
 *   File "p64.ml", line 10, characters 15-31:
 *   10 | ...............float
 *      |
 *   10 | let[@sare..................................................
 *   Error: Cannot unify types: float32 and float64
 *
 * — the caret region and the echoed line are nonsense, so the diagnostic
 * cannot be acted on even though the message itself is accurate. The kernel is
 * on line 10 but the bytes shown come from line 1.
 *
 * These tests pin the round trip itself, which is where the loss happens.
 ******************************************************************************)

let pos ~fname ~lnum ~bol ~cnum =
  {Lexing.pos_fname = fname; pos_lnum = lnum; pos_bol = bol; pos_cnum = cnum}

(* A location on a line that does NOT start at byte 0 — i.e. every line of
   every real source file except the first. Line 10 starts at byte 300; the
   span covers bytes 315..331, which is columns 15..31 of that line. *)
let sample : Ppxlib.Location.t =
  {
    loc_start = pos ~fname:"p64.ml" ~lnum:10 ~bol:300 ~cnum:315;
    loc_end = pos ~fname:"p64.ml" ~lnum:10 ~bol:300 ~cnum:331;
    loc_ghost = false;
  }

let check_position label (expected : Lexing.position) (got : Lexing.position) =
  Alcotest.(check string) (label ^ ": file") expected.pos_fname got.pos_fname ;
  Alcotest.(check int) (label ^ ": line") expected.pos_lnum got.pos_lnum ;
  Alcotest.(check int)
    (label
   ^ ": absolute byte offset (pos_cnum) — this is what the compiler uses to \
      slice the source line")
    expected.pos_cnum
    got.pos_cnum ;
  Alcotest.(check int)
    (label ^ ": line-start byte offset (pos_bol)")
    expected.pos_bol
    got.pos_bol ;
  Alcotest.(check int)
    (label ^ ": column (pos_cnum - pos_bol)")
    (expected.pos_cnum - expected.pos_bol)
    (got.pos_cnum - got.pos_bol)

(** [loc_to_ppxlib (loc_of_ppxlib l)] must be [l]. Anything less means every
    Sarek diagnostic that passes through the Sarek AST is rendered against the
    wrong bytes. *)
let test_ast_loc_round_trip () =
  let got = Sarek_ast.loc_to_ppxlib (Sarek_ast.loc_of_ppxlib sample) in
  check_position "start" sample.loc_start got.loc_start ;
  check_position "end" sample.loc_end got.loc_end

(** The native-generation copy of the same conversion must agree. It is used for
    the locations attached to generated OCaml, so a bogus offset there sends the
    OCaml type-checker's own errors to the wrong place. *)
let test_native_helpers_loc_round_trip () =
  let got =
    Sarek_native_helpers.ppxlib_loc_of_sarek (Sarek_ast.loc_of_ppxlib sample)
  in
  check_position "start" sample.loc_start got.loc_start ;
  check_position "end" sample.loc_end got.loc_end

(** The column must still be derivable, so nothing downstream that reads
    [loc_col] changes meaning. *)
let test_column_is_preserved () =
  let l = Sarek_ast.loc_of_ppxlib sample in
  Alcotest.(check int) "loc_line" 10 l.Sarek_ast.loc_line ;
  Alcotest.(check int) "loc_col" 15 l.Sarek_ast.loc_col ;
  Alcotest.(check int) "loc_end_col" 31 l.Sarek_ast.loc_end_col

(** A location built by hand from line/column only (the dummy and the
    parser-synthesised cases) must still round-trip to something the compiler
    renders sanely: line 1 columns are absolute offsets, so this degenerate case
    is consistent by construction. *)
let test_dummy_loc_is_consistent () =
  let got = Sarek_ast.loc_to_ppxlib Sarek_ast.dummy_loc in
  Alcotest.(check int) "dummy line" 1 got.loc_start.pos_lnum ;
  Alcotest.(check int)
    "dummy column"
    0
    (got.loc_start.pos_cnum - got.loc_start.pos_bol)

let () =
  Alcotest.run
    "diagnostic_locations"
    [
      ( "source locations survive the Sarek AST round trip",
        [
          Alcotest.test_case
            "Sarek_ast round trip is the identity"
            `Quick
            test_ast_loc_round_trip;
          Alcotest.test_case
            "Sarek_native_helpers round trip is the identity"
            `Quick
            test_native_helpers_loc_round_trip;
          Alcotest.test_case
            "line and column are preserved"
            `Quick
            test_column_is_preserved;
          Alcotest.test_case
            "dummy location stays consistent"
            `Quick
            test_dummy_loc_is_consistent;
        ] );
    ]
