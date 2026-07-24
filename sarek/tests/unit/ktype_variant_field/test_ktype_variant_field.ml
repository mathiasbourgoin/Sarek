(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Device-free unit guard for the [@@sarek.type] interpreter value-model
    helpers with a VARIANT-typed record field (deriver-variant-fields).

    Exercises the generated helpers through the public
    {!Sarek.Sarek_type_helpers} registry: a variant type round-trips
    OCaml<->VVariant, and a record with a variant field round-trips
    OCaml<->VRecord (whose field is a VVariant). Regression guard for the L14-S2
    PR #251 `Field 'kind' expected record` boundary bug, independent of any
    backend. *)

module V = Sarek.Sarek_value
module H = Sarek.Sarek_type_helpers

type float32 = float

type color = Red | Green | Shade of float32 [@@sarek.type]

type cell = {kind : color; scale : float32} [@@sarek.type]

let color_helpers () =
  match H.lookup_typed color_custom.Spoc_core.Vector.type_id with
  | Some (module C : H.HELPERS with type t = color) ->
      (module C : H.HELPERS with type t = color)
  | None -> Alcotest.fail "no helper registered for color"

let cell_helpers () =
  match H.lookup_typed cell_custom.Spoc_core.Vector.type_id with
  | Some (module C : H.HELPERS with type t = cell) ->
      (module C : H.HELPERS with type t = cell)
  | None -> Alcotest.fail "no helper registered for cell"

let color_eq a b =
  match (a, b) with
  | Red, Red | Green, Green -> true
  | Shade x, Shade y -> abs_float (x -. y) < 1e-6
  | _ -> false

(* The variant helper's [to_value] must tag with the interpreter convention
   [Hashtbl.hash ctor mod 256] so an in-kernel match recognises a host-set
   element. *)
let test_color_to_value () =
  let (module C) = color_helpers () in
  (match C.to_value Red with
  | V.VVariant (_, tag, []) ->
      Alcotest.(check int) "Red tag" (Hashtbl.hash "Red" mod 256) tag
  | _ -> Alcotest.fail "Red should be a nullary VVariant") ;
  match C.to_value (Shade 2.5) with
  | V.VVariant (_, tag, [V.VFloat32 f]) ->
      Alcotest.(check int) "Shade tag" (Hashtbl.hash "Shade" mod 256) tag ;
      Alcotest.(check (float 1e-6)) "Shade payload" 2.5 f
  | _ -> Alcotest.fail "Shade should be a unary VVariant carrying its float"

let test_color_roundtrip () =
  let (module C) = color_helpers () in
  List.iter
    (fun c ->
      let back = C.from_value (C.to_value c) in
      Alcotest.(check bool)
        ("color round-trip "
        ^ match c with Red -> "Red" | Green -> "Green" | Shade _ -> "Shade")
        true
        (color_eq c back))
    [Red; Green; Shade 3.5; Shade (-1.0)]

let test_cell_roundtrip () =
  let (module C) = cell_helpers () in
  List.iter
    (fun cell ->
      (* to_value produces a VRecord whose [kind] field is a VVariant. *)
      (match C.to_value cell with
      | V.VRecord (_, [|V.VVariant _; V.VFloat32 _|]) -> ()
      | _ ->
          Alcotest.fail
            "cell to_value should be a VRecord with a VVariant kind field") ;
      let back = C.from_value (C.to_value cell) in
      Alcotest.(check bool)
        "cell.kind round-trip"
        true
        (color_eq cell.kind back.kind) ;
      Alcotest.(check (float 1e-6))
        "cell.scale round-trip"
        cell.scale
        back.scale)
    [
      {kind = Red; scale = 1.0};
      {kind = Green; scale = 2.0};
      {kind = Shade 4.5; scale = 3.0};
    ]

(* A record field holding a VVariant must be accepted by the record helper's
   from_value (the exact site of the former "expected record" failure). *)
let test_from_value_accepts_variant_field () =
  let (module C) = cell_helpers () in
  let vrec =
    V.VRecord
      ( "Test_ktype_variant_field.cell",
        [|
          V.VVariant ("color", Hashtbl.hash "Shade" mod 256, [V.VFloat32 7.0]);
          V.VFloat32 9.0;
        |] )
  in
  let cell = C.from_value vrec in
  Alcotest.(check bool) "kind" true (color_eq cell.kind (Shade 7.0)) ;
  Alcotest.(check (float 1e-6)) "scale" 9.0 cell.scale

let () =
  Alcotest.run
    "ktype_variant_field"
    [
      ( "variant-field helpers",
        [
          Alcotest.test_case "color to_value tags" `Quick test_color_to_value;
          Alcotest.test_case "color round-trip" `Quick test_color_roundtrip;
          Alcotest.test_case "cell round-trip" `Quick test_cell_roundtrip;
          Alcotest.test_case
            "record from_value accepts variant field"
            `Quick
            test_from_value_accepts_variant_field;
        ] );
    ]
