(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Test kernel that should FAIL convergence analysis:
   - Calls barrier inside thread-varying conditional *)

let () =
  (* This should fail because barrier is in diverged control flow *)
  let _bad_kernel =
    [%kernel
      fun (v : float32 vector) -> if thread_idx_x > 16 then block_barrier ()]
  in
  print_endline "This should not print - test should have failed to compile"

(* Additional bad patterns that should also fail:

   (* While loop with thread-varying condition *)
   let _bad_while = [%kernel fun (v : float32 vector) (n : int32) ->
     while thread_idx_x < n do
       block_barrier ()
     done
   ]

   (* Array access with thread-varying index in condition *)
   let _bad_array = [%kernel fun (v : float32 vector) ->
     if v.(thread_idx_x) > 0.0 then
       block_barrier ()
   ]
*)
