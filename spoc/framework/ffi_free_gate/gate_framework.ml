(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * ffi_free_gate — regression gate for the FFI-free spoc_framework library.
 *
 * This executable must build as .bc and .bc.js (js_of_ocaml). It depends
 * ONLY on spoc_framework. If ctypes or unix re-enter spoc_framework,
 * this target fails to build.
 *
 * Exercises: dims helpers, typed_value primitives, typed_value registry.
 ******************************************************************************)

open Spoc_framework

(* Compile-time pin: the FFI-free boundary methods must stay [nativeint]. If
   they revert to [Ctypes.ptr] this fails to type-check (and would require
   ctypes, which this gate does not link) — so the gate catches both a ctypes
   re-entry AND a silent signature regression. Never called; only type-checked. *)
let _pin_boundary (module B : Framework_sig.BACKEND) =
  let _h2d : src_ptr:nativeint -> byte_size:int -> dst:_ B.Memory.buffer -> unit
      =
    B.Memory.host_ptr_to_device
  in
  let _d2h : src:_ B.Memory.buffer -> dst_ptr:nativeint -> byte_size:int -> unit
      =
    B.Memory.device_to_host_ptr
  in
  ignore _h2d ;
  ignore _d2h

let () =
  (* dims constructors — pure functions, no FFI *)
  let d1 = Framework_sig.dims_1d 32 in
  assert (d1.Framework_sig.x = 32) ;
  assert (d1.Framework_sig.y = 1) ;
  assert (d1.Framework_sig.z = 1) ;
  let d2 = Framework_sig.dims_2d 8 16 in
  assert (d2.Framework_sig.x = 8) ;
  assert (d2.Framework_sig.y = 16) ;
  assert (d2.Framework_sig.z = 1) ;
  (* Typed_value: primitive storage and registry, no ctypes *)
  let open Typed_value in
  let p = PInt32 42l in
  assert (primitive_type_name p = "int32") ;
  let scalars = Registry.list_scalars () in
  assert (List.mem "float32" scalars) ;
  assert (List.mem "float64" scalars) ;
  Printf.printf "spoc_framework ffi_free_gate: all assertions passed\n%!"
