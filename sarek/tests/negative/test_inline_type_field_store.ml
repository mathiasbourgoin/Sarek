(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Negative test (backlog-172): a field store into a record type declared INSIDE
   the kernel, held in a vector, must be REFUSED at PPX time.

   Such a type is reached through a generated first-class-module GETTER
   (`Sarek_native_gen_expr.field_access_of` answers `Field_fcm`) and there is no
   corresponding setter. The vector-element store the rest of backlog-172 adds
   needs a record label to put in a record-update expression, and for this shape
   there is none — so the choice is refuse or emit something that does not mean
   what it says. Emitting a `setfield` here is what the whole of backlog-172 was
   about, so it refuses.

   WHY THIS FILE EXISTS AT ALL, given the refusal is one `raise_errorf`: the
   message is the only thing the user gets, and its two previous spellings were
   both wrong in ways no build could see.

     1. It said "an inline [%%sarek.type] record". `Location.raise_errorf` takes a
        FORMAT string, in which `%%` renders as one `%`, so the user was told to
        look for `[%sarek.type]` — which is not a construct. And the diagnosis was
        wrong regardless of escaping: what lands in `ctx.inline_types` is a type
        declared in the kernel payload, bearing no attribute at all.

     2. The correction dropped every mention of the attribute, including from the
        ADVICE — where it belongs, because "move the declaration out of the
        kernel" is not sufficient on its own. An out-of-kernel type is
        Sarek-visible only once registered, and `[@@sarek.type]` is what registers
        it. The narrowing overshot.

   Both defects live entirely in the RENDERED text, and `[@@sarek.type]` needs
   FOUR `@`s in the literal to render as two. Nothing but reading the compiler's
   actual output distinguishes a correct literal from either mistake, which is
   what the Makefile case for this file greps for.

   Expected error:
     "cannot assign to field \"b\" of a record type declared inside the kernel"
   and, in the same message:
     "register it with [@@sarek.type]"

   Self-contained (backlog-208): no `open Spoc`, no `open Kirc`, so the red is
   caused by the refusal and nothing else. *)

type float32 = float

let k =
  [%kernel
    let module Types = struct
      type pair = {a : float32; b : float32}
    end in
    fun (v : pair vector) (n : int32) ->
      let tid = thread_idx_x + (block_dim_x * block_idx_x) in
      if tid < n then v.(tid).b <- 1.0]
