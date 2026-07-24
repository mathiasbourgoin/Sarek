(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Test kernel that should FAIL to compile:
   - A kernel-LOCAL tuple binding whose component is itself a tuple (a
     non-primitive component). Local primitive tuples are lowered to the
     synthesized [_tup_*] record, but a non-primitive component must produce
     the located tuple-component error (the same one vector-of-tuple elements
     raise), NOT silently collapse to the placeholder int. *)

let () =
  let bad_kernel =
    [%kernel
      fun (src : float32 vector) (dst : float32 vector) (n : int32) ->
        let tid = thread_idx_x + (block_dim_x * block_idx_x) in
        if tid < n then begin
          let p = (src.(tid), (tid, tid)) in
          match p with a, _ -> dst.(tid) <- a
        end]
  in
  ignore bad_kernel ;
  print_endline "This should not print - test should have failed to compile"
