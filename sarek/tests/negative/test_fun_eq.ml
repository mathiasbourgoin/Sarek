(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Negative test (backlog-194): comparing two FUNCTION values with `=`.

   Found by the cross-runtime review pass, and worth recording as such: an
   earlier revision of this branch excluded `TFun` from the refused set with
   the justification that "refusing it would be a removal with no evidence
   behind it". That was a claim about evidence not gathered. Gathering it took
   one kernel:

     let f (x : float32) = x +. 1.0 in
     let g (x : float32) = x +. 2.0 in
     if f = g then ...

   compiled clean on this branch, and the OpenCL emitter produced

     if ((f == g)) {

   naming two identifiers that appear NOWHERE in the emitted source — a
   kernel-local helper is inlined at its call sites, not declared as a device
   function. `clang -x cl -cl-std=CL1.2 -fsyntax-only` exit 1:
   "use of undeclared identifier 'f'", "use of undeclared identifier 'g'".

   This member differs from the other three in kind, which is why its
   diagnostic is a different sentence: a tuple, record or variant fails for
   want of a field-wise lowering, and each has a "compare the parts" remedy.
   A function value has no device object at all, so there is nothing to compare
   and nothing to decompose. Nothing is lost by refusing it — there is no
   backend on which it ever meant anything.

   Expected error:
     "'=' cannot compare two function values in a kernel" *)

type float32 = float

let () =
  let bad_kernel =
    [%kernel
      fun (src : float32 vector) (dst : float32 vector) (n : int32) ->
        let tid = thread_idx_x + (block_dim_x * block_idx_x) in
        let f (x : float32) = x +. 1.0 in
        let g (x : float32) = x +. 2.0 in
        if f = g then dst.(tid) <- src.(tid)]
  in
  ignore bad_kernel ;
  print_endline "This should not print - test should have failed to compile"
