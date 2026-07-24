(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Test kernel that should FAIL to compile:
   - A [for] loop induction variable is named [sarek_i], which begins with the
     [sarek_] prefix reserved by the code generator. The loop variable is
     emitted verbatim into device code (Sarek_ir_pp / Sarek_ir_glsl) and is NOT
     covered by the #255 collision-safe name set (params + helpers only), so the
     reserved-prefix policy must reject it at kernel elaboration. *)

open Spoc
open Sarek

let () =
  let bad_kernel =
    [%kernel
      fun (r : int32 vector) ->
        let open Std in
        let idx = global_idx_x in
        for sarek_i = 0 to 3 do
          r.(idx) <- r.(idx) + 1l
        done]
  in
  let _, kirc = bad_kernel in
  Kirc.print_ast kirc.Kirc.body ;
  print_endline "This should not print - test should have failed to compile"
