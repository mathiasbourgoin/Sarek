(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Negative test (backlog-194): a non-primitive tuple in plain EXPRESSION
   position — here the kernel body's tail expression.

   This case pins the [TETuple] lowering arm itself, which is the arm the
   backlog item named. It is deliberately NOT an equality: the operator refusal
   in Sarek_typer cannot fire here, so if this file's expected message ever
   stops appearing, the missing thing is the lowering arm and nothing else.

   Measured on 97a062a2 the PPX ACCEPTED this kernel, built [Ir.ETuple], and
   the failure came later and from somewhere else entirely — the generated
   native OCaml, with

     Error: The value __native_kern has type ... -> 'a * ('a * 'a)
            but an expression was expected of type ... -> unit

   which names neither tuples nor the unsupported construct. So the old
   behaviour was not "refused by OCaml"; it was "accepted by Sarek, then
   rejected by the host compiler for an unrelated-looking reason". A kernel
   shape that happened not to trip that host-side check (equality — see
   test_tuple_eq_nonprim) went all the way to malformed device C.

   Expected error:
     "Tuple values support only scalar components" *)

type float32 = float

let () =
  let bad_kernel =
    [%kernel
      fun (src : float32 vector) (dst : float32 vector) (n : int32) ->
        let tid = thread_idx_x + (block_dim_x * block_idx_x) in
        (src.(tid), (src.(tid), src.(tid)))]
  in
  ignore bad_kernel ;
  print_endline "This should not print - test should have failed to compile"
