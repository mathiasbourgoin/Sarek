(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Negative test: f16 arithmetic reached through LET-POLYMORPHIC GENERALIZATION.

   The storage-only guard is keyed by type-variable ID. [Sarek_scheme]'s
   [copy_for_scheme] and [instantiate] mint FRESH ids, so before the fix the
   constraint recorded on [x] while typing [twice]'s body was simply absent from
   the call-site instance, and unifying it with a float16 element succeeded. The
   typer emitted no diagnostic at all; the build only failed later, in generated
   native OCaml, with an unintelligible object-type error naming
   Spoc_core_base.Make(Spoc_core.Ctypes_ops).t.

   That the guard was what should have fired is shown by the control: the
   IDENTICAL shape at float32 (see sarek/tests/e2e, and w_c in the review notes)
   compiles cleanly, so the two differ only in the constraint.

   Expected error:
     "float16 is a storage-only type and has no arithmetic: a polymorphic
      helper instantiated at float16" *)

let[@sarek.module] twice (x : 'a) : 'a = x +. x

let k =
  [%kernel
    fun (out : float16 vector) (a : float16 vector) ->
      let open Sarek_stdlib.Std in
      let tid = global_thread_id in
      out.(tid) <- twice a.(tid)]
