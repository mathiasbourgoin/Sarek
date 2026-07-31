(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Negative test (backlog-194): an aggregate reaching `=` through a POLYMORPHIC
   [@sarek.module] helper.

   This case pins the lowering backstop, not the typer refusal, and the two are
   distinguishable: [Sarek_typer.reject_aggregate_equality] runs inside
   [infer_binop], where both operands of `a = b` in the helper's body are still
   the same unresolved tvar. Monomorphisation instantiates the body afterwards
   and does not re-run operator inference, so the typer gate provably cannot
   see this shape — exactly the generalization hole that let f16 arithmetic
   through in #57 (see test_f16_poly_helper).

   Measured with the typer gate in place but before the lowering arm existed:
   this kernel compiled and OpenCL emitted

     int same__Ranon_record_Ranon_record(anon_record a, anon_record b) {
       return (a == b);
     }

   rejected by clang -x cl with "invalid operands to binary expression".

   Expected error:
     "'=' cannot compare two values of type" *)

type float32 = float

let[@sarek.module] same (a : 'a) (b : 'a) : bool = a = b

let () =
  let bad_kernel =
    [%kernel
      fun (src : float32 vector) (dst : float32 vector) (n : int32) ->
        let open Test_poly_aggregate_eq_ty in
        let tid = thread_idx_x + (block_dim_x * block_idx_x) in
        let p = {x = src.(tid); y = src.(tid)} in
        let q = {x = src.(tid); y = src.(tid)} in
        if same p q then dst.(tid) <- src.(tid)]
  in
  ignore bad_kernel ;
  print_endline "This should not print - test should have failed to compile"
