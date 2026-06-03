(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Sarek Standard Library Metadata (FFI-free)
 *
 * Pure metadata library: registers stdlib intrinsic signatures
 * (name/sarek-type/device-template/size) into Sarek_ppx_registry
 * WITHOUT any Ctypes or Spoc_core dependency.
 *
 * This enables FFI-free transpile (PR-5b): a transpiler can resolve
 * Float32.sin, Int32.add, Gpu.thread_idx_x etc. by linking only
 * sarek_frontend + sarek_stdlib_meta, with no GPU FFI required.
 *
 * Coverage: Float32, Int32, Int64, Math, Gpu (all modules in Sarek_stdlib).
 *
 * The FFI execution path (ctype marshalling + Spoc_core.Vector host impls)
 * remains in sarek_stdlib, which depends on sarek_stdlib_meta.
 ******************************************************************************)

module Float32 = Float32
module Int32 = Int32
module Int64 = Int64
module Gpu = Gpu
module Math = Math
module Std = Gpu

(******************************************************************************
 * Force initialization
 *
 * Ensures all metadata modules register their intrinsics at load time.
 ******************************************************************************)

let () =
  ignore Int64.of_float ;
  ignore Float32.sin ;
  ignore Int32.of_float ;
  ignore Gpu.thread_idx_x ;
  ignore Math.xor

let force_init () = ()
