(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * ffi_free_gate — regression gate for the FFI-free numeric vector core
 *
 * This executable must build as .bc and .bc.js (js_of_ocaml). It depends
 * ONLY on spoc_core_base instantiated with Stub_ops. If ctypes or unix
 * re-enter the numeric core, this target fails to build.
 *
 * Exercises: create, set, get, to_array on Bigarray (Float32) vectors.
 ******************************************************************************)

module V = Spoc_core_base.Make (Stub_ops)

let () =
  let n = 8 in
  let vec = V.create V.float32 n in
  (* Float32 vectors always use Bigarray_storage; the GADT guarantees this. *)
  let ba = V.to_bigarray vec in
  for i = 0 to n - 1 do
    Bigarray.Array1.set ba i (float_of_int i)
  done ;
  let arr = Array.init n (fun i -> Bigarray.Array1.get ba i) in
  assert (Array.length arr = n) ;
  Array.iteri (fun i v -> assert (v = float_of_int i)) arr ;
  assert (V.elem_size V.float32 = 4) ;
  assert (V.bigarray_elem_size Bigarray.Float32 = 4) ;
  assert (V.bigarray_elem_size Bigarray.Float64 = 8) ;
  assert (V.bigarray_elem_size Bigarray.Int32 = 4) ;
  assert (V.bigarray_elem_size Bigarray.Int8_signed = 1) ;
  (* Test create/copy/sub *)
  let vec2 = V.copy_host_only vec in
  let ba2 = V.to_bigarray vec2 in
  Array.iteri (fun i v -> assert (Bigarray.Array1.get ba2 i = v)) arr ;
  (* Test kind name *)
  assert (V.kind_name V.float32 = "Float32") ;
  Printf.printf "ffi_free_gate: all assertions passed\n%!"
