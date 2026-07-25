(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Negative test: `char` as a Sarek element type (wrong-width family, #96).

   `Spoc_core.Vector.char` is a Bigarray of OCaml chars — ONE byte per element.
   Source `char` lowered to [Ir.TInt32] ("char represented as int32"), so the
   emitted device signature was `int*`:

     __global__ void sarek_kern(int* __restrict__ o, ..., int* __restrict__ a, ...)

   i.e. the buffer was strode at four times the host's element size, with no
   diagnostic at any stage. Nothing could ever run: [Execute.check_launch_args]
   already refuses a [Vector.Char] argument against a [TInt32] parameter on
   physical width, on both the device and interpreter entry points. All that
   was missing was telling the user at compile time, and why.

   Expected error:
     "`char` is not a supported Sarek element type" *)

let k =
  [%kernel
    fun (o : char vector) (a : char vector) ->
      let open Sarek_stdlib.Std in
      let tid = global_thread_id in
      o.(tid) <- a.(tid)]
