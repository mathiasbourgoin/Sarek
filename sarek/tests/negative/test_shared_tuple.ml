(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Test kernel that should FAIL to compile:
   - Declares a [let%shared] array whose ELEMENT type is a primitive-component
     tuple. Routing this through the data mapper to the synthesized [_tup_*]
     record was attempted (the TELetShared sibling of the helper-return
     wrong-width fix) but PROVEN to miscompile on shared-capable backends: the
     PTX backend raises "unsupported construct: btype of custom type" and the
     Native backend "Cannot create default value for this type" (OpenCL/Vulkan
     happen to accept it). A compound in shared memory is therefore rejected at
     the [let%shared] boundary, mirroring lower_param's tuple-parameter
     rejection — a clean located compile error, not unproven codegen.
     See briefs/helper-return-wrong-width-impl.md (round 2). *)

let () =
  let bad_kernel =
    snd
      [%kernel
        fun (src : float32 vector) (dst : float32 vector) (len : int32) ->
          let open Sarek_stdlib.Std in
          let%shared (tile : float32 * float32) = 64l in
          let tid = global_thread_id in
          let lid = thread_idx_x in
          if tid < len then begin
            tile.(lid) <- (src.(tid), src.(tid) +. 1.0) ;
            block_barrier () ;
            let p = tile.(lid) in
            match p with a, b -> dst.(tid) <- a +. b
          end]
  in
  ignore bad_kernel ;
  print_endline "This should not print - test should have failed to compile"
