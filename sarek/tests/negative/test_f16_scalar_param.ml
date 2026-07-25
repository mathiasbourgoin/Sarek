(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Negative test (#57 slice 1 review): a SCALAR float16 kernel parameter must be
   a compile error. The delivered surface is `float16 vector`.

   Why it matters: this type-checked, and CUDA/HIP mapped it to a by-value
   `__half` formal — but Execute.vector_arg has no float16 constructor, so the
   only way to supply it is [Float32 f], which pushes a 4-byte C float whose
   address the device reads as a 2-byte __half. Executed on gfx1100 with
   `Float32 3.14159`: HIP produced 0.000476837158 while the interpreter produced
   the correct 3.140625, with no error raised anywhere — the two oracles silently
   disagreed, which is exactly the property test_hip_f16 exists to guarantee.

   Expected error:
     "has type float16: f16 is a storage-only element type and cannot be a
      scalar kernel parameter" *)

let k =
  [%kernel
    fun (out : float16 vector) (s : float16) ->
      let open Sarek_stdlib.Std in
      let tid = global_thread_id in
      out.(tid) <- s]
