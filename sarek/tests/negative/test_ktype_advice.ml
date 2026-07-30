(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Test type declaration that should FAIL to compile, with a SPECIFIC message.

   backlog-193: the size/alignment refusal for an unregistered field type told
   the user to "register it with [%%ktype]". No `ktype` extension has ever
   existed in this tree, so the one line of the diagnostic that was supposed to
   help sent the reader to a construct they cannot write.

   test_unregistered_field.ml already covers the REFUSAL. It asserts only on
   "unknown (size|alignment) for field type", so the false advice sat inside a
   string the suite was already reading and no assertion touched it. This case
   exists to assert on the ADVICE:

     - it must name [@@sarek.type], the attribute that actually populates
       Sarek_ppx's size/alignment registries; and
     - it must SAY "[@@sarek.type]", not "[@sarek.type]". Location.raise_errorf
       is Format-based, so a literal "@@" in the format string prints as a
       single "@" -- and a single-@ attribute cannot sit on a type declaration,
       so the corrected name would still have been unusable advice. `make
       test_negative` greps the compiler's real output, which is the only place
       that distinction is observable.

   Deliberately SELF-CONTAINED (backlog-208): no `open`, no library value is
   referenced. Eight sibling negative tests carry `open Spoc`/`open Kirc` that
   the dune stanza does not provide, and pass only because the PPX refuses the
   file before typechecking reaches the `open` -- so a guard that stopped firing
   would still fail the build there, for the wrong reason, and read as a pass.
   Here, if the PPX guard stops firing, this file compiles cleanly and the
   assertion in the Makefile is what fails. *)

type unregistered_alias = float

type bad_record = {distance : unregistered_alias; count : int32} [@@sarek.type]

let () =
  ignore (fun (r : bad_record) -> r.count) ;
  print_endline "This should not print - test should have failed to compile"
