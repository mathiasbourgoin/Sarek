(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Test kernel that should FAIL to compile:
   - A record type declared for the kernel has a field named [sarek_len], which
     begins with the [sarek_] prefix reserved by the code generator. The
     reserved-prefix policy rejects it at kernel elaboration with the located
     reservation error. *)

open Spoc
open Sarek

let () =
  let bad_kernel =
    [%kernel
      let module Types = struct
        type box = {sarek_len : int32; value : float32}
      end in
      let make_box (n : int32) (v : float32) : box =
        {sarek_len = n; value = v}
      in
      fun (a : float32 vector) (r : float32 vector) ->
        let open Std in
        let idx = global_idx_x in
        let b = make_box idx a.(idx) in
        r.(idx) <- b.value]
  in
  let _, kirc = bad_kernel in
  Kirc.print_ast kirc.Kirc.body ;
  print_endline "This should not print - test should have failed to compile"
