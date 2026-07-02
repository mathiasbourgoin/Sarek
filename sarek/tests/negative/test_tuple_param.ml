(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Test kernel that should FAIL to compile:
   - Uses a tuple-typed kernel parameter, which the V2 lowering path
     (Sarek_lower_ir.ml, elttype_of_typ) cannot represent. *)

let () =
  let bad_kernel =
    [%kernel
      fun (v : float32 vector) (pair : int32 * int32) ->
        let tid = thread_idx_x + (block_dim_x * block_idx_x) in
        v.(tid) <- Float32.of_int (Int32.to_int tid)]
  in
  ignore bad_kernel ;
  print_endline "This should not print - test should have failed to compile"
