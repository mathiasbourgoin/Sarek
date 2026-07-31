(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Negative test (backlog-194): comparing two RECORD values with `=`.

   The comment that used to sit on [infer_binop]'s Eq/Ne arm read "equality is
   legal on bool, records and variants". Records and variants were in it
   because nothing refused them, not because anything emitted a working
   comparison. Measured on 97a062a2 this kernel compiled and OpenCL emitted

     if ((a == b)) {

   on two `struct`s, which clang -x cl rejects with "invalid operands to binary
   expression". The claim was wider than the code in the most literal way: it
   asserted support for two shapes no C-family backend can express.

   The native backend is the reason this was never a loud failure everywhere:
   Sarek_native_gen.ml:213 emits OCaml `=`, which is structural and answers
   correctly. So the construct compiled and ran on Native while failing to
   BUILD on OpenCL/CUDA/Metal/GLSL/WGSL — silently non-portable rather than
   uniformly broken.

   Expected error:
     "'=' cannot compare two values of type" *)

type float32 = float

type pt = {x : float32; y : float32} [@@sarek.type]

let () =
  let bad_kernel =
    [%kernel
      fun (src : float32 vector) (dst : float32 vector) (n : int32) ->
        let tid = thread_idx_x + (block_dim_x * block_idx_x) in
        let a = {x = src.(tid); y = src.(tid)} in
        let b = {x = src.(tid); y = src.(tid)} in
        if a = b then dst.(tid) <- src.(tid)]
  in
  ignore bad_kernel ;
  print_endline "This should not print - test should have failed to compile"
