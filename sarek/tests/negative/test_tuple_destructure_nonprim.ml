(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Test kernel that should FAIL to compile:
   - A NON-PRIMITIVE tuple DESTRUCTURE with a non-variable scrutinee. The
     scrutinee never passes through the kernel-local slot-typing path
     (no make_var), so the rejection must come from the match-scrutinee guard
     in Sarek_lower_ir.ml — NOT surface as a confusing backend C error
     (`switch((...).tag)`). The let-pattern is desugared by the parser to the
     single-arm tuple match this guard covers. *)

let () =
  let bad_kernel =
    [%kernel
      fun (src : float32 vector) (dst : float32 vector) (n : int32) ->
        let tid = thread_idx_x + (block_dim_x * block_idx_x) in
        if tid < n then begin
          let (a, b), c = ((src.(tid), src.(tid)), src.(tid)) in
          dst.(tid) <- a +. b +. c
        end]
  in
  ignore bad_kernel ;
  print_endline "This should not print - test should have failed to compile"
