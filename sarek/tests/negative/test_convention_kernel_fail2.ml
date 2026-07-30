(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* backlog-208: this case used to carry `open Spoc` while its dune stanza
   declared only `sarek sarek.stdlib sarek.ppx.lib`. There is no `Spoc` module in
   this tree at all (nor a `Kirc` one -- both are SPOC-v1 vestige), so the file
   only ever compiled as far as the PPX: the refusal below fired before
   type-checking reached the `open`, and `-w -33` hid the unused-open hint. The
   case was therefore red for TWO reasons, and would have looked exactly the same
   if the refusal had stopped working -- an unbound-module error is still a
   non-zero exit with the file's name on it. The `open` and the dead
   `Kirc.print_ast` tail are gone so the red is the refusal and nothing else. *)

(* Test kernel that should FAIL type checking:
   - Computes float32 distance but tries to write to int32 vector *)
open Sarek_geometry

let () =
  (* This should fail because we're writing float32 to int32 vector *)
  let bad_kernel =
    [%kernel
      fun (points : Geometry_lib.point vector)
          (distances : int32 vector) (* int32 instead of float32! *)
          (n : int32) ->
        let tid = thread_idx_x + (block_dim_x * block_idx_x) in
        if tid < n then
          let p = points.(tid) in
          let x = p.x in
          let y = p.y in
          distances.(tid) <- sqrt ((x *. x) +. (y *. y))
      (* float32 result to int32 vector *)]
  in

  let _, kirc = bad_kernel in
  ignore kirc ;
  print_endline "This should not print - test should have failed to compile"
