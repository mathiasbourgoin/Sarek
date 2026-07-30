(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Negative test: An uninterpreted attribute on a superstep binding must be REFUSED
   (backlog-192).

   `parse_superstep` tested `attr_name.txt = "divergent"` and ignored every other
   attribute in silence. So this misspelling read as "not divergent" and the
   convergence checker was applied to a step the user had declared divergent --
   the diagnostic the user then got was about a barrier in diverged control flow,
   from a step they thought they had marked. *)

let k =
  [%kernel
    fun (input : float32 vector) (output : float32 vector) ->
      let%shared (tile : float32) = () in
      let%superstep[@divergnt] step =
        if thread_idx_x > 16l then tile.(thread_idx_x) <- input.(thread_idx_x)
      in
      output.(thread_idx_x) <- tile.(thread_idx_x)]
