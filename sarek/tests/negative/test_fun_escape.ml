(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Test kernel that should FAIL to compile (L12 escape rule, Sarek_defunc):
   a function value bound to `let f` is returned from the kernel body instead
   of being applied to arguments. Tier-0 defunctionalization has no runtime
   representation for an escaping function value, so this is rejected with a
   `Function_value_escapes` diagnostic ("Function value escapes: ...") rather
   than an opaque codegen failure downstream. *)

module Std = Sarek_stdlib.Std

let () =
  let bad_kernel =
    [%kernel
      let open Std in
      let addf (a : int32) (b : int32) : int32 = a + b in
      let mulf (a : int32) (b : int32) : int32 = a * b in
      fun (op : int32) ->
        let f = if op = 0l then addf else mulf in
        f]
  in
  ignore bad_kernel ;
  print_endline "This should not print - test should have failed to compile"
