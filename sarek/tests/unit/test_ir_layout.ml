(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Unit tests for Sarek_ir_layout: pins the aligned (C-ABI) byte layout
    (offsets and sizes) of the aggregate types used by the e2e tests, the newly
    supported mixed-alignment aggregates ([{i32;f64}] records, f64/i64-payload
    variants), and the still-typed rejections of nested-variant / array-field
    aggregates. These offsets are the host ABI (Sarek_ppx.ml calc_offsets and
    the variant [tag@0][payload@max(4,align)] encoding) — a change here is a
    host/device ABI break. The all-4-byte e2e types (point/point3d/color/
    particle) are UNCHANGED by the L8 packed->aligned migration (zero-breakage).
*)

open Sarek_ir_types
open Sarek_ir_layout

let get_ok ~what = function
  | Ok v -> v
  | Error e ->
      Alcotest.fail
        (Printf.sprintf
           "%s: unexpected rejection: %s"
           what
           (layout_error_message e))

let get_error ~what = function
  | Ok _ -> Alcotest.fail (Printf.sprintf "%s: expected a rejection" what)
  | Error e -> e

let check_leaves what expected (leaves : leaf list) =
  Alcotest.(check (list (pair string int)))
    what
    expected
    (List.map (fun l -> (l.leaf_path, l.leaf_offset)) leaves)

(** point {x; y} : f32 -> offsets 0, 4; size 8. *)
let test_point () =
  let rl =
    get_ok
      ~what:"point"
      (record_layout ~type_name:"point" [("x", TFloat32); ("y", TFloat32)])
  in
  check_leaves "point leaves" [("x", 0); ("y", 4)] rl.rl_leaves ;
  Alcotest.(check (list (pair string int)))
    "point fields"
    [("x", 0); ("y", 4)]
    rl.rl_fields ;
  Alcotest.(check int) "point size" 8 rl.rl_size

(** point3d {x; y; z} : f32 -> offsets 0, 4, 8; size 12. *)
let test_point3d () =
  let rl =
    get_ok
      ~what:"point3d"
      (record_layout
         ~type_name:"point3d"
         [("x", TFloat32); ("y", TFloat32); ("z", TFloat32)])
  in
  check_leaves "point3d leaves" [("x", 0); ("y", 4); ("z", 8)] rl.rl_leaves ;
  Alcotest.(check int) "point3d size" 12 rl.rl_size

(** color = Red | Value of f32 -> tag@0, payload@4, size 8. *)
let test_color () =
  let vl =
    get_ok
      ~what:"color"
      (variant_layout ~type_name:"color" [("Red", []); ("Value", [TFloat32])])
  in
  Alcotest.(check int) "tag offset" 0 vl.vl_tag_offset ;
  Alcotest.(check int) "payload offset" 4 vl.vl_payload_offset ;
  Alcotest.(check int) "color size" 8 vl.vl_size ;
  let tags = List.map (fun c -> (c.ctor_name, c.ctor_tag)) vl.vl_ctors in
  Alcotest.(check (list (pair string int)))
    "ctor tags = declaration order"
    [("Red", 0); ("Value", 1)]
    tags ;
  let value = List.nth vl.vl_ctors 1 in
  check_leaves "Value payload" [("Value._0", 4)] value.ctor_leaves ;
  Alcotest.(check int)
    "Red payload size"
    0
    (List.hd vl.vl_ctors).ctor_payload_size

(** particle {pos_x; pos_y; vel_x; vel_y; mass} : f32 -> 0..16; size 20. *)
let test_particle () =
  let rl =
    get_ok
      ~what:"particle"
      (record_layout
         ~type_name:"particle"
         [
           ("pos_x", TFloat32);
           ("pos_y", TFloat32);
           ("vel_x", TFloat32);
           ("vel_y", TFloat32);
           ("mass", TFloat32);
         ])
  in
  check_leaves
    "particle leaves"
    [("pos_x", 0); ("pos_y", 4); ("vel_x", 8); ("vel_y", 12); ("mass", 16)]
    rl.rl_leaves ;
  Alcotest.(check int) "particle size" 20 rl.rl_size

(** Nested record {a:f32; inner:{b:f32; c:f32}} -> leaves at 0, 4, 8; size 12. *)
let test_nested_record () =
  let inner = TRecord ("inner_t", [("b", TFloat32); ("c", TFloat32)]) in
  let rl =
    get_ok
      ~what:"outer"
      (record_layout ~type_name:"outer" [("a", TFloat32); ("inner", inner)])
  in
  check_leaves
    "nested leaves"
    [("a", 0); ("inner.b", 4); ("inner.c", 8)]
    rl.rl_leaves ;
  Alcotest.(check (list (pair string int)))
    "nested immediate fields"
    [("a", 0); ("inner", 4)]
    rl.rl_fields ;
  Alcotest.(check int) "nested size" 12 rl.rl_size

let contains hay needle =
  let hl = String.length hay and nl = String.length needle in
  let found = ref false in
  for i = 0 to hl - nl do
    if String.sub hay i nl = needle then found := true
  done ;
  !found

let assert_msg_contains what msg fragments =
  List.iter
    (fun frag ->
      if not (contains msg frag) then
        Alcotest.fail
          (Printf.sprintf "%s: message %S should contain %S" what msg frag))
    fragments

(** L8: [{a:i32; b:f64}] is now ACCEPTED with aligned offsets: a@0, b@8 (4 bytes
    of padding after a), total size 16 (rounded to the 8-byte max alignment).
    This was a rejection before the packed->aligned migration. *)
let test_mixed_align_record () =
  let rl =
    get_ok
      ~what:"mixed_align"
      (record_layout ~type_name:"mixed_align" [("a", TInt32); ("b", TFloat64)])
  in
  check_leaves "mixed_align leaves" [("a", 0); ("b", 8)] rl.rl_leaves ;
  Alcotest.(check (list (pair string int)))
    "mixed_align fields"
    [("a", 0); ("b", 8)]
    rl.rl_fields ;
  Alcotest.(check int) "mixed_align size" 16 rl.rl_size

(** L8: [{b:f64; a:i32}] reordered largest-first packs tighter: b@0, a@8, size
    16 (a fits in the trailing 8 bytes; still 16 due to 8-byte struct
    alignment). *)
let test_mixed_align_record_reordered () =
  let rl =
    get_ok
      ~what:"reordered"
      (record_layout ~type_name:"reordered" [("b", TFloat64); ("a", TInt32)])
  in
  check_leaves "reordered leaves" [("b", 0); ("a", 8)] rl.rl_leaves ;
  Alcotest.(check int) "reordered size" 16 rl.rl_size

(** L8: [{flag:bool; d:f64}]: bool is 4 bytes on the host, d needs 8 -> d@8,
    size 16. *)
let test_bool_f64_record () =
  let rl =
    get_ok
      ~what:"bool_f64"
      (record_layout ~type_name:"bool_f64" [("flag", TBool); ("d", TFloat64)])
  in
  check_leaves "bool_f64 leaves" [("flag", 0); ("d", 8)] rl.rl_leaves ;
  Alcotest.(check int) "bool_f64 size" 16 rl.rl_size

(** L8: variant with an f64 payload is now ACCEPTED. Payload region moves to
    offset 8 (max(4, 8-byte align)); Some_._0@8, size 16. *)
let test_f64_variant_payload () =
  let vl =
    get_ok
      ~what:"f64_variant"
      (variant_layout
         ~type_name:"f64_variant"
         [("None_", []); ("Some_", [TFloat64])])
  in
  Alcotest.(check int) "tag offset" 0 vl.vl_tag_offset ;
  Alcotest.(check int) "payload offset" 8 vl.vl_payload_offset ;
  Alcotest.(check int) "f64_variant size" 16 vl.vl_size ;
  let some = List.nth vl.vl_ctors 1 in
  check_leaves "Some_ payload" [("Some_._0", 8)] some.ctor_leaves ;
  Alcotest.(check int) "Some_ payload size" 8 some.ctor_payload_size

(** L8: variant with an i64 payload behaves like f64 (8-byte payload): payload
    region @8, size 16. *)
let test_i64_variant_payload () =
  let vl =
    get_ok
      ~what:"i64_variant"
      (variant_layout
         ~type_name:"i64_variant"
         [("None_", []); ("Some_", [TInt64])])
  in
  Alcotest.(check int) "payload offset" 8 vl.vl_payload_offset ;
  Alcotest.(check int) "i64_variant size" 16 vl.vl_size ;
  let some = List.nth vl.vl_ctors 1 in
  check_leaves "Some_ payload" [("Some_._0", 8)] some.ctor_leaves

(** L8: a variant carrying a mixed-alignment tuple payload [{i32;f64}]: payload@8,
    _0 (i32) @8, _1 (f64) @16, payload size 16, total size 24. *)
let test_mixed_tuple_variant_payload () =
  let vl =
    get_ok
      ~what:"mixed_tuple"
      (variant_layout
         ~type_name:"mixed_tuple"
         [("Nil", []); ("Pair", [TInt32; TFloat64])])
  in
  Alcotest.(check int) "payload offset" 8 vl.vl_payload_offset ;
  Alcotest.(check int) "mixed_tuple size" 24 vl.vl_size ;
  let pair = List.nth vl.vl_ctors 1 in
  check_leaves "Pair payload" [("Pair._0", 8); ("Pair._1", 16)] pair.ctor_leaves ;
  Alcotest.(check int) "Pair payload size" 16 pair.ctor_payload_size

(** Record containing a variant field -> rejected (FR-005). *)
let test_reject_variant_in_record () =
  let color = TVariant ("color", [("Red", []); ("Value", [TFloat32])]) in
  let err =
    get_error
      ~what:"has_variant"
      (record_layout ~type_name:"has_variant" [("a", TFloat32); ("c", color)])
  in
  (match err with
  | Nested_variant {type_name; field} ->
      Alcotest.(check string) "type name" "has_variant" type_name ;
      Alcotest.(check string) "field" "c" field
  | e ->
      Alcotest.fail ("expected Nested_variant, got: " ^ layout_error_message e)) ;
  assert_msg_contains
    "nested variant message"
    (layout_error_message err)
    ["has_variant"; "'c'"; "variant nested below top level"]

(** Record containing a TVec field -> rejected. *)
let test_reject_vec_in_record () =
  let err =
    get_error
      ~what:"has_vec"
      (record_layout
         ~type_name:"has_vec"
         [("a", TFloat32); ("v", TVec TFloat32)])
  in
  (match err with
  | Unsupported_field {type_name; field; what} ->
      Alcotest.(check string) "type name" "has_vec" type_name ;
      Alcotest.(check string) "field" "v" field ;
      Alcotest.(check string) "what" "TVec" what
  | e ->
      Alcotest.fail
        ("expected Unsupported_field, got: " ^ layout_error_message e)) ;
  assert_msg_contains
    "vec field message"
    (layout_error_message err)
    ["has_vec"; "'v'"; "TVec"]

(** elttype_layout dispatches scalars/records/variants consistently. *)
let test_elttype_layout_dispatch () =
  (match elttype_layout TFloat32 with
  | Ok (LScalar {size = 4; align = 4}) -> ()
  | _ -> Alcotest.fail "TFloat32 should be LScalar {size=4; align=4}") ;
  (match elttype_layout TInt64 with
  | Ok (LScalar {size = 8; align = 8}) -> ()
  | _ -> Alcotest.fail "TInt64 should be LScalar {size=8; align=8}") ;
  (match
     elttype_layout (TRecord ("point", [("x", TFloat32); ("y", TFloat32)]))
   with
  | Ok (LRecord rl) -> Alcotest.(check int) "record size" 8 rl.rl_size
  | _ -> Alcotest.fail "point should be LRecord") ;
  (match
     elttype_layout (TVariant ("color", [("Red", []); ("Value", [TFloat32])]))
   with
  | Ok (LVariant vl) -> Alcotest.(check int) "variant size" 8 vl.vl_size
  | _ -> Alcotest.fail "color should be LVariant") ;
  match elttype_layout (TVec TFloat32) with
  | Error (Unsupported_field _) -> ()
  | _ -> Alcotest.fail "TVec element should be rejected"

(** Host size mapping pinned (field_byte_size): bool is 4 bytes on the host
    (catch-all in get_type_size_from_core_type), i64/f64 are 8. *)
let test_scalar_sizes_match_host () =
  Alcotest.(check int) "i32" 4 (scalar_size TInt32) ;
  Alcotest.(check int) "f32" 4 (scalar_size TFloat32) ;
  Alcotest.(check int) "bool = host catch-all" 4 (scalar_size TBool) ;
  Alcotest.(check int) "unit" 4 (scalar_size TUnit) ;
  Alcotest.(check int) "i64" 8 (scalar_size TInt64) ;
  Alcotest.(check int) "f64" 8 (scalar_size TFloat64) ;
  Alcotest.(check int) "align i32" 4 (scalar_align TInt32) ;
  Alcotest.(check int) "align i64" 8 (scalar_align TInt64) ;
  Alcotest.(check int) "align f64" 8 (scalar_align TFloat64)

let () =
  Alcotest.run
    "ir_layout"
    [
      ( "layout",
        [
          Alcotest.test_case "point offsets 0,4 size 8" `Quick test_point;
          Alcotest.test_case "point3d offsets 0,4,8 size 12" `Quick test_point3d;
          Alcotest.test_case "color tag@0 payload@4 size 8" `Quick test_color;
          Alcotest.test_case
            "particle offsets 0..16 size 20"
            `Quick
            test_particle;
          Alcotest.test_case
            "nested record leaves 0,4,8 size 12"
            `Quick
            test_nested_record;
          Alcotest.test_case
            "scalar sizes match host field_byte_size"
            `Quick
            test_scalar_sizes_match_host;
          Alcotest.test_case
            "elttype_layout dispatch"
            `Quick
            test_elttype_layout_dispatch;
        ] );
      ( "mixed-alignment (L8 unlocked)",
        [
          Alcotest.test_case
            "{a:i32; b:f64} accepted: a@0 b@8 size 16"
            `Quick
            test_mixed_align_record;
          Alcotest.test_case
            "{b:f64; a:i32} reordered: b@0 a@8 size 16"
            `Quick
            test_mixed_align_record_reordered;
          Alcotest.test_case
            "{flag:bool; d:f64} accepted: d@8 size 16"
            `Quick
            test_bool_f64_record;
          Alcotest.test_case
            "variant f64 payload accepted: payload@8 size 16"
            `Quick
            test_f64_variant_payload;
          Alcotest.test_case
            "variant i64 payload accepted: payload@8 size 16"
            `Quick
            test_i64_variant_payload;
          Alcotest.test_case
            "variant {i32;f64} tuple payload: _0@8 _1@16 size 24"
            `Quick
            test_mixed_tuple_variant_payload;
        ] );
      ( "rejections",
        [
          Alcotest.test_case
            "variant field in record rejected"
            `Quick
            test_reject_variant_in_record;
          Alcotest.test_case
            "TVec field in record rejected"
            `Quick
            test_reject_vec_in_record;
        ] );
    ]
