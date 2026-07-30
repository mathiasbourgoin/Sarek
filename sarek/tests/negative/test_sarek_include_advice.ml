(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* A sarek_include with a non-string payload must be refused, and the ADVICE it
   prints must name a spelling that works.

   backlog-193, second sweep. The payload refusal in expand_sarek_include is a
   Location.raise_errorf, so it is Format-based, and its literal
   "[%%sarek_include \"file.ml\"]" reached the terminal as
   "[%sarek_include \"file.ml\"]". sarek_include is declared in
   Extension.Context.structure_item, so that single-% spelling is NOT this
   extension: it parses as a structure-level expression and comes back
   "Uninterpreted extension 'sarek_include'". Measured, both directions:

     [%sarek_include  "probe.ml"]  -> Uninterpreted extension 'sarek_include'
     [%%sarek_include "probe.ml"]  -> compiles

   The static gate (scripts/check-ppx-construct-names.sh) reasons about the
   rendering; this asserts on the compiler's real output, which is the only place
   the collapse is observable. The Makefile greps for BOTH percent signs.

   Self-contained on purpose (backlog-208): no `open`, no library value. If the
   payload guard stops firing, this file compiles and the Makefile assertion is
   what fails, rather than an unrelated missing-module error reading as a pass. *)

let%sarek_include _ = 42

let () =
  print_endline "This should not print - test should have failed to compile"
