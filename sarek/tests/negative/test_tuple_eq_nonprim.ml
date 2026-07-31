(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Negative test (backlog-194): comparing two NON-PRIMITIVE tuples with `=`.

   This is the shape that made the backlog item real. Every other consumer of a
   non-primitive tuple already refused it — vector elements, kernel-local
   bindings, `let`-pattern destructures, `match` scrutinees, `let%shared`
   arrays, tuple-typed kernel parameters — so `Ir.ETuple` looked unreachable
   from source. Equality was the hole: `infer_binop`'s Eq/Ne arm ran no operand
   check beyond f16, and an `if` condition is not a data slot, so nothing on
   the way down typed the tuple.

   Measured on 97a062a2, this exact kernel COMPILED (dune build exit 0) and the
   OpenCL emitter produced

     if (({src[tid], (_tup_float32_float32){._0 = src[tid], ._1 = src[tid]}}
          == {src[tid], (_tup_float32_float32){._0 = src[tid], ._1 = src[tid]}}))

   — a bare, type-less brace list, which is the malformed C the item described.
   clang -x cl -cl-std=CL1.2 rejected it with "expected ';' after expression",
   "expected ')'" and "statement requires expression of scalar type".

   Expected error:
     "'=' cannot compare two values of type" *)

type float32 = float

let () =
  let bad_kernel =
    [%kernel
      fun (src : float32 vector) (dst : float32 vector) (n : int32) ->
        let tid = thread_idx_x + (block_dim_x * block_idx_x) in
        if
          (src.(tid), (src.(tid), src.(tid)))
          = (src.(tid), (src.(tid), src.(tid)))
        then dst.(tid) <- src.(tid)]
  in
  ignore bad_kernel ;
  print_endline "This should not print - test should have failed to compile"
