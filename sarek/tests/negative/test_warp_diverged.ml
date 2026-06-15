(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Negative test: warp collective in diverged control flow should fail *)

[%%kernel
let bad_warp_shuffle (input : int32 Arr.t Global.t)
    (output : int32 Arr.t Global.t) =
  (* Divergent branch - not all threads take the same path *)
  if thread_idx_x > 16l then
    (* ERROR: warp_shuffle requires all warp threads to participate *)
    let v = warp_shuffle input.(thread_idx_x) 1l in
    output.(thread_idx_x) <- v]
