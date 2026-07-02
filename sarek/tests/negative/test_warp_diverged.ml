(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Test kernel that should FAIL convergence analysis:
   - Calls warp_shuffle (a warp-collective) inside thread-varying
     conditional. All threads in the warp must participate. *)

let () =
  (* This should fail because warp_shuffle is a warp collective called in
     diverged control flow (only threads > 16 participate). *)
  let _bad_kernel =
    [%kernel
      fun (v : int32 vector) ->
        if thread_idx_x > 16 then
          let x = warp_shuffle v.(thread_idx_x) 1l in
          v.(thread_idx_x) <- x]
  in
  print_endline "This should not print - test should have failed to compile"
