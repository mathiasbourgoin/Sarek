(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Negative test (backlog-194): `<>` on two VARIANT values.

   This case carries two jobs the four sibling cases do not.

   1. VARIANT is one of the four members of the refused set
      (Sarek_types.is_uncomparable_operand_typ = tuple, record, variant,
      function). Without this file, `TVariant` could be deleted from that
      predicate and `make test_negative` would stay green — the other cases
      exercise tuples, records and functions. It is also the member the old Eq/Ne comment named
      explicitly ("equality is legal on bool, records and variants"), so it is
      the claim that most needs a pin.

   2. It is the only case using `<>` rather than `=`. Both gates render the
      operator through Sarek_error.binop_display_name, and the Makefile row
      asserts the rendered `'<>'` — so the Ne branch, and the fact that the
      message names the operator the user actually wrote, are both covered.

   Nullary constructors are used deliberately: `c <> Red` on an enum-like
   variant is idiomatic OCaml and is the shape most likely to be written by a
   user. It compiled before the fix and worked on the native backend
   (Sarek_native_gen.ml emits OCaml `<>`, which is structural), while the
   C-family backends emitted `a != b` on a struct. That combination — right
   answer on one backend, uncompilable on five — is what the refusal removes.

   Expected error:
     "'<>' cannot compare two values of type" *)

type float32 = float

type color = Red | Green [@@sarek.type]

let () =
  let bad_kernel =
    [%kernel
      fun (src : float32 vector) (dst : float32 vector) (n : int32) ->
        let tid = thread_idx_x + (block_dim_x * block_idx_x) in
        let c = if src.(tid) >. 0.0 then Red else Green in
        if c <> Red then dst.(tid) <- src.(tid)]
  in
  ignore bad_kernel ;
  print_endline "This should not print - test should have failed to compile"
