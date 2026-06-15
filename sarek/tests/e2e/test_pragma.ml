(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Test pragma support in Sarek PPX *)

let kernel_with_pragma =
  [%kernel
    fun (a : int32 vector) (n : int32) ->
      let tid = thread_idx_x in
      let sum = mut 0l in
      pragma
        ["unroll"]
        (for i = 0 to 3 do
           sum := sum + a.(tid + i)
         done) ;
      a.(tid) <- sum]

let () = print_endline "Pragma test compiled successfully"
