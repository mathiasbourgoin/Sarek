(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Negative test: Compile-time error when inline depth causes node exhaustion
 *
 * This file is expected to FAIL at compile time with:
 *   "Inlining produced N nodes (limit: 10000). Reduce inline depth..."
 *
 * Fibonacci with inline depth 8 produces ~13821 nodes, exceeding the limit.
 ******************************************************************************)

open Spoc
open Sarek

(* This should fail to compile - inline depth 8 on fib produces too many nodes *)
let fib_exhaustion_kernel =
  [%kernel
    let open Std in
    (* Fibonacci grows exponentially: 2^8 = 256 copies of the body *)
    let rec fib (n : int32) : int32 =
      pragma
        ["sarek.inline 8"]
        (if n <= 0l then 0l
         else if n = 1l then 1l
         else fib (n - 1l) + fib (n - 2l))
    in
    fun (output : int32 vector) (n : int32) ->
      let idx = global_idx_x in
      if idx = 0l then output.(idx) <- fib n]

let () =
  (* This code should never run - compilation should fail above *)
  print_endline "ERROR: This should not compile!"
