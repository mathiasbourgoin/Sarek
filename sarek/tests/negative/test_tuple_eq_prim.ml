(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Negative test (backlog-194): comparing two PRIMITIVE tuples with `=`.

   The sibling case (test_tuple_eq_nonprim) is the one the backlog item filed;
   this is the polarity that shows the refusal is not keyed to the tuple's
   COMPONENTS. A primitive tuple lowers to the synthesized [_tup_*] record, so
   it never went near [Ir.ETuple] — and it was just as broken. Measured on
   97a062a2 this kernel compiled and OpenCL emitted

     if (((_tup_float32_float32){._0 = src[tid], ._1 = src[tid]}
          == (_tup_float32_float32){._0 = src[tid], ._1 = src[tid]}))

   which clang -x cl rejects with "invalid operands to binary expression
   ('_tup_float32_float32' and '_tup_float32_float32')".

   Fixing only the [Ir.ETuple] arm — which is what the item as filed asks for —
   would have left this case emitting uncompilable OpenCL. That is why the
   refusal is at the operator, on the operand TYPE, and not at the tuple
   literal.

   Expected error:
     "'=' cannot compare two values of type" *)

type float32 = float

let () =
  let bad_kernel =
    [%kernel
      fun (src : float32 vector) (dst : float32 vector) (n : int32) ->
        let tid = thread_idx_x + (block_dim_x * block_idx_x) in
        if (src.(tid), src.(tid)) = (src.(tid), src.(tid)) then
          dst.(tid) <- src.(tid)]
  in
  ignore bad_kernel ;
  print_endline "This should not print - test should have failed to compile"
