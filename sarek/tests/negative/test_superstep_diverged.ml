(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Test kernel that should FAIL convergence analysis:
   - Thread-varying condition inside non-divergent superstep

   This should be rejected because the implicit barrier at the end
   of the superstep would be in diverged control flow. *)

let () =
  (* This should fail: thread-varying condition without ~divergent flag *)
  let _bad_kernel =
    [%kernel
      fun (input : float32 vector) (output : float32 vector) ->
        let%shared (tile : float32) = () in
        let%superstep bad_step =
          (* Thread-varying condition - diverges control flow *)
          if thread_idx_x > 16l then
            tile.(thread_idx_x) <- input.(global_thread_id)
        in
        output.(global_thread_id) <- tile.(thread_idx_x)]
  in
  print_endline "This should not print - test should have failed to compile"

(* To fix, use [@divergent] attribute:
   let%superstep [@divergent] step = ... in
*)
