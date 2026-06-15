(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Sarek Math Standard Library
 *
 * Provides integer math operations for Sarek kernels.
 * These are commonly used in GPU algorithms like bitonic sort.
 ******************************************************************************)

let dev cuda opencl d = Sarek_registry.cuda_or_opencl d cuda opencl

(******************************************************************************
 * Bitwise operations on int32
 ******************************************************************************)

let%sarek_intrinsic (xor : int32 -> int32 -> int32) =
  {device = dev "(%s ^ %s)" "(%s ^ %s)"; ocaml = Stdlib.Int32.logxor}

let%sarek_intrinsic (logical_and : int32 -> int32 -> int32) =
  {device = dev "(%s & %s)" "(%s & %s)"; ocaml = Stdlib.Int32.logand}

(******************************************************************************
 * Power function
 ******************************************************************************)

let%sarek_intrinsic (pow : int32 -> int32 -> int32) =
  {
    device =
      dev "(int)powf((float)%s, (float)%s)" "(int)pow((float)%s, (float)%s)";
    ocaml =
      (fun a b ->
        Stdlib.Int32.of_float
          (Float.pow (Stdlib.Int32.to_float a) (Stdlib.Int32.to_float b)));
  }
