(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Unit tests for the host-side SoA layout/transpose (Spoc_core.Soa).
 * Pure ctypes buffers, no device — verifies plan derivation and that
 * AoS -> SoA -> AoS round-trips bit-identically.
 ******************************************************************************)

module Soa = Spoc_core.Soa
module Helpers = Spoc_core.Vector_types.Custom_helpers
open Sarek_ir_types

let alloc_bytes n = Ctypes.(to_voidp (allocate_n uint8_t ~count:n))

let point3d =
  TRecord ("point3d", [("x", TFloat32); ("y", TFloat32); ("z", TFloat32)])

(* {a:int32; b:float64} — mixed width, exercises the 8-byte leaf and aligned
   offsets (b lands at offset 8 under the aligned ABI, stride 16). *)
let ab = TRecord ("ab", [("a", TInt32); ("b", TFloat64)])

let test_plan_point3d () =
  let p = Soa.plan_of_elttype point3d in
  Alcotest.(check int) "num_leaves" 3 (Soa.num_leaves p) ;
  Alcotest.(check int) "aos_stride" 12 p.Soa.aos_stride ;
  let offs = List.map (fun (l : Soa.leaf) -> l.aos_offset) p.Soa.leaves in
  Alcotest.(check (list int)) "offsets" [0; 4; 8] offs ;
  let sizes = List.map (fun (l : Soa.leaf) -> l.size) p.Soa.leaves in
  Alcotest.(check (list int)) "sizes" [4; 4; 4] sizes

let test_plan_mixed () =
  let p = Soa.plan_of_elttype ab in
  Alcotest.(check int) "num_leaves" 2 (Soa.num_leaves p) ;
  (* aligned ABI: int32 at 0, float64 aligned to 8, struct padded to 16 *)
  Alcotest.(check int) "aos_stride" 16 p.Soa.aos_stride ;
  let offs = List.map (fun (l : Soa.leaf) -> l.aos_offset) p.Soa.leaves in
  Alcotest.(check (list int)) "offsets" [0; 8] offs

let test_rejects () =
  let nested = TRecord ("outer", [("p", point3d); ("w", TFloat32)]) in
  Alcotest.check_raises
    "nested record rejected"
    (Soa.Unsupported
       "nested-record field \"p\" in \"outer\": v1 SoA supports flat records \
        only")
    (fun () -> ignore (Soa.plan_of_elttype nested)) ;
  Alcotest.check_raises
    "non-record rejected"
    (Soa.Unsupported "SoA plan requires a record (TRecord) element type")
    (fun () -> ignore (Soa.plan_of_elttype TFloat32))

(* Fill AoS[i] = {x=i, y=100+i, z=200+i}; scatter; check contiguous leaves;
   gather back; check bit-identical. *)
let test_roundtrip_point3d () =
  let p = Soa.plan_of_elttype point3d in
  let length = 257 in
  let aos = alloc_bytes (Soa.aos_bytes p ~length) in
  for i = 0 to length - 1 do
    Helpers.write_float32 aos ((i * 12) + 0) (float_of_int i) ;
    Helpers.write_float32 aos ((i * 12) + 4) (float_of_int (100 + i)) ;
    Helpers.write_float32 aos ((i * 12) + 8) (float_of_int (200 + i))
  done ;
  let leaves = Array.init 3 (fun _ -> alloc_bytes (length * 4)) in
  Soa.scatter p ~aos ~length ~leaves ;
  (* Each leaf buffer must hold its field's values contiguously. *)
  for i = 0 to length - 1 do
    Alcotest.(check (float 0.0))
      "leaf x"
      (float_of_int i)
      (Helpers.read_float32 leaves.(0) (i * 4)) ;
    Alcotest.(check (float 0.0))
      "leaf y"
      (float_of_int (100 + i))
      (Helpers.read_float32 leaves.(1) (i * 4)) ;
    Alcotest.(check (float 0.0))
      "leaf z"
      (float_of_int (200 + i))
      (Helpers.read_float32 leaves.(2) (i * 4))
  done ;
  (* Gather back into a fresh AoS buffer and compare bit-identical. *)
  let aos2 = alloc_bytes (Soa.aos_bytes p ~length) in
  Soa.gather p ~leaves ~length ~aos:aos2 ;
  for i = 0 to (length * 12) - 1 do
    let b1 = Ctypes.(!@(from_voidp uint8_t aos +@ i)) in
    let b2 = Ctypes.(!@(from_voidp uint8_t aos2 +@ i)) in
    Alcotest.(check int)
      (Printf.sprintf "byte %d" i)
      (Unsigned.UInt8.to_int b1)
      (Unsigned.UInt8.to_int b2)
  done

let test_roundtrip_mixed () =
  let p = Soa.plan_of_elttype ab in
  let length = 64 in
  let stride = p.Soa.aos_stride in
  let aos = alloc_bytes (Soa.aos_bytes p ~length) in
  for i = 0 to length - 1 do
    Helpers.write_int32 aos ((i * stride) + 0) (Int32.of_int (i * 3)) ;
    Helpers.write_float64 aos ((i * stride) + 8) (float_of_int i +. 0.5)
  done ;
  let leaves = [|alloc_bytes (length * 4); alloc_bytes (length * 8)|] in
  Soa.scatter p ~aos ~length ~leaves ;
  for i = 0 to length - 1 do
    Alcotest.(check int32)
      "leaf a"
      (Int32.of_int (i * 3))
      (Helpers.read_int32 leaves.(0) (i * 4)) ;
    Alcotest.(check (float 0.0))
      "leaf b"
      (float_of_int i +. 0.5)
      (Helpers.read_float64 leaves.(1) (i * 8))
  done ;
  let aos2 = alloc_bytes (Soa.aos_bytes p ~length) in
  Soa.gather p ~leaves ~length ~aos:aos2 ;
  for i = 0 to length - 1 do
    Alcotest.(check int32)
      "rt a"
      (Helpers.read_int32 aos (i * stride))
      (Helpers.read_int32 aos2 (i * stride)) ;
    Alcotest.(check (float 0.0))
      "rt b"
      (Helpers.read_float64 aos ((i * stride) + 8))
      (Helpers.read_float64 aos2 ((i * stride) + 8))
  done

let () =
  Alcotest.run
    "soa"
    [
      ( "plan",
        [
          Alcotest.test_case "point3d" `Quick test_plan_point3d;
          Alcotest.test_case "mixed" `Quick test_plan_mixed;
          Alcotest.test_case "rejects" `Quick test_rejects;
        ] );
      ( "roundtrip",
        [
          Alcotest.test_case "point3d" `Quick test_roundtrip_point3d;
          Alcotest.test_case "mixed" `Quick test_roundtrip_mixed;
        ] );
    ]
