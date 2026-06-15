(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Sarek Standard Library
 *
 * Provides GPU intrinsics and type aliases for Sarek kernels.
 *
 * Exports:
 * - Std/Gpu: GPU intrinsics module (thread_idx_x, global_thread_id, etc.)
 * - Vector: V2 Vector module (Spoc_core.Vector)
 * - vector: Type alias for kernel parameter annotations
 * - Float32, Int32, Int64, Math: GPU numeric intrinsics (use inside kernels)
 *
 * NOTE: Float64 is NOT part of stdlib - it's a separate library (sarek_float64)
 * because not all devices support double precision.
 ******************************************************************************)

(* GPU intrinsic modules - for use inside kernels with `let open Std in` *)
(* NOTE: Tests should NOT do `open Sarek_stdlib` as it will shadow OCaml's
   Int32/Int64 modules. Instead, import specific modules or use the Vector
   type alias. *)
module Float32 = Float32
module Int32 = Int32
module Int64 = Int64
module Gpu = Gpu
module Math = Math

(* Backward compatibility alias *)
module Std = Gpu

(* V2 Vector module - replaces SPOC's Vector *)
module Vector = Spoc_core.Vector

(* Type alias for kernel parameter annotations.
   When code writes `(x : float32 vector)`, this resolves to V2 Vector.t *)
type ('a, 'b) vector = ('a, 'b) Spoc_core.Vector.t

(******************************************************************************
 * Force initialization
 *
 * OCaml's lazy module initialization means modules only initialize when
 * something from them is actually used. We force all stdlib modules to
 * initialize at module load time, which registers their intrinsics.
 ******************************************************************************)

let () =
  ignore Int64.of_float ;
  ignore Float32.sin ;
  ignore Int32.of_float ;
  ignore Gpu.thread_idx_x ;
  ignore Math.xor

let force_init () = ()
