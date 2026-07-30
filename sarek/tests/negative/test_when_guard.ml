(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Negative test (backlog-191): a `when` guard on a match case must be REFUSED
   at PPX time.

   Before the fix, Sarek_parse's Pexp_match arm read pc_lhs and pc_rhs and never
   looked at pc_guard, so the guard was dropped in the parser — upstream of the
   typer, of every lowering pass, and of backend selection. The arm became
   unconditional in the single AST all backends are generated from.

   The guard below is LOAD-BEARING, which is what makes the old behaviour a
   wrong ANSWER rather than a harmless control-flow difference: every element
   takes a Circle, and the guard is the only thing choosing between the two
   Circle arms. With src.(i) = i+1, elements 0..9 must take the second arm
   (r +. 100.0) and the rest the first (r *. 2.0). With the guard dropped the
   first arm is unconditional and EVERY element gets r *. 2.0.

   Measured on this tree before the fix, with exactly this kernel: it compiled
   and returned the dropped-guard answer on all 9 devices present here —
   Interpreter (sequential + parallel), Native, CUDA/PTX x2 (ZLUDA on AMD),
   OpenCL x2 (radeonsi), Vulkan x2 (RADV). 10/64 elements wrong, 0/64 wrong
   against the dropped-guard oracle, on every one of them.

   It compiled, but NOT with no diagnostic — and the two facts are linked. What
   makes the guard load-bearing above is that both Circle arms exist; drop the
   guard and they become syntactically identical, which is warning 11
   [redundant-case]. Under the e2e suites' flags ((:standard -w -32-33-34-69),
   which leave 11 on) that is a hard error. So the very shape needed to show a
   wrong ANSWER is the shape the compiler complains about, just for the wrong
   reason: a redundant case, not a discarded guard. The silent shapes are the
   ones where no two arms share a constructor — there the guard disappears with
   no warning at all, and the arm is simply unconditional. This case
   deliberately picks the wrong-answer shape over the silent one, because a
   demonstrated wrong result is the stronger statement of the defect; the price
   is that "compiled with no diagnostic", which earlier revisions of this header
   and of kb/sarek/ppx/parser.md asserted, is false of this kernel.

   Expected error:
     "`when` guards on match cases are not supported in kernels"

   Deliberately SELF-CONTAINED (backlog-208): no `open Spoc`, no `open Kirc`.
   Eight sibling cases carry such an open while the dune stanza declares only
   `sarek sarek.stdlib sarek.ppx.lib`, so neither module resolves; they pass
   only because the PPX refusal fires before type-checking reaches the open,
   with `-w -33` hiding the hint. This case's red must be caused by the guard
   refusal and nothing else. *)

type float32 = float

type shape = Circle of float32 | Square of float32 [@@sarek.type]

let k =
  [%kernel
    fun (src : float32 vector) (dst : float32 vector) (n : int32) ->
      let tid = thread_idx_x + (block_dim_x * block_idx_x) in
      if tid < n then begin
        let s = Circle src.(tid) in
        let got =
          match s with
          | Circle r when r >. 10.0 -> r *. 2.0
          | Circle r -> r +. 100.0
          | Square q -> q
        in
        dst.(tid) <- got
      end]
