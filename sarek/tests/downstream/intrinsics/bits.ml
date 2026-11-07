(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Reinterpret an int32 bit pattern as a float32: the downstream intrinsic that
   exposed the gap, with one spelling per C-family framework that consults the
   intrinsic registry. *)

let device = function
  | "OpenCL" -> "as_float(%s)"
  | "Metal" -> "as_type<float>(%s)"
  | _ -> "__int_as_float(%s)"

let%sarek_intrinsic (float_of_bits : int32 -> float32) =
  {device; ocaml = Int32.float_of_bits}
