(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Negative test: a polymorphic [@sarek.module] helper instantiated at a
   non-default element type (#97).

   [norm]'s body calls [sqrt], which the Sarek stdlib types at float32, so the
   helper's type variable is pinned to float32 and it cannot be monomorphised
   at float64. That IS a legitimate compile error — the defect was how it was
   reported. Before the fix, on this tree:

     File "p64.ml", line 10, characters 15-31:
     10 | ...............float
        |
     10 | let[@sare.......................................................
     Error: Cannot unify types: float32 and float64

   The caret region and the echoed source line are unrelated bytes taken from
   the TOP of the file, because [Sarek_ast.loc] dropped [pos_bol] and rebuilt
   the position with [pos_bol = 0; pos_cnum = column] — so the driver, which
   seeks to [pos_bol] to echo the line, read from byte `column` of the file.
   The message also named neither the helper nor why float32 was required.

   Expected error:
     "'norm' cannot be used at this call site" *)

type float64 = float

let[@sarek.module] norm (x : 'a) (y : 'a) : 'a = sqrt ((x *. x) +. (y *. y))

let k =
  [%kernel
    fun (o : float64 vector) (a : float64 vector) ->
      let open Sarek_stdlib.Std in
      let tid = global_thread_id in
      o.(tid) <- norm a.(tid) a.(tid)]
