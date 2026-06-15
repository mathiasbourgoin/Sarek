(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Test kernel that should FAIL type checking:
   - Uses undefined function 'undefined_func' *)

let () =
  (* This should fail because undefined_func does not exist *)
  let bad_kernel =
    [%kernel
      fun (v : float32 vector) (n : int32) ->
        let tid = thread_idx_x + (block_dim_x * block_idx_x) in
        if tid < n then v.(tid) <- undefined_func v.(tid)]
  in

  let _, kirc = bad_kernel in
  Sarek.Kirc.print_ast kirc.Sarek.Kirc.body ;
  print_endline "This should not print - test should have failed to compile"
