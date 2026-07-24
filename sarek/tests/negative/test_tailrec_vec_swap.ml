(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Test kernel that should FAIL to compile:
   - A tail-recursive helper takes two vector parameters [a] and [b] and, in the
     recursive call, SWAPS them ([sum_two b a ...]). A vector/buffer cannot be
     reassigned on GPU backends (GLSL/WGSL inline the helper and substitute the
     buffer name), so passing a DIFFERENT vector than the one received is
     rejected by the tail-recursion elimination pass with a located error.

   Expected error: "must pass its own vector parameter 'a' unchanged". *)

module Std = Sarek_stdlib.Std
open Sarek

let () =
  let bad_kernel =
    [%kernel
      let open Std in
      let rec sum_two (a : int32 vector) (b : int32 vector) (i : int32)
          (n : int32) (acc : int32) : int32 =
        if i >= n then acc else sum_two b a (i + 1l) n (acc + a.(i))
      in
      fun (out : int32 vector)
          (x : int32 vector)
          (y : int32 vector)
          (n : int32)
        ->
        let idx = global_idx_x in
        out.(idx) <- sum_two x y 0l n 0l]
  in
  ignore bad_kernel ;
  print_endline "This should not print - test should have failed to compile"
