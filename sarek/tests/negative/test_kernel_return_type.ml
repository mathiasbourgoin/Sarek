(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Negative test: A return-type annotation on the kernel function must be REFUSED
   (backlog-192).

   `Sarek_ast.kernel` has `kern_params` and `kern_body` and no return type, so
   this annotation was read by nobody at all -- neither checked nor reported. *)

let k =
  [%kernel
    fun (src : int32 vector) (dst : int32 vector) : unit ->
      let tid = thread_idx_x in
      dst.(tid) <- src.(tid)]
