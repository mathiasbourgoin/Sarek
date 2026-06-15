(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Unit tests for Sarek_value module *)

open Alcotest
open Sarek_value

(** Test value type name reporting *)
let test_value_type_name_int32 () =
  let v = VInt32 42l in
  check string "int32 type name" "int32" (value_type_name v)

let test_value_type_name_int64 () =
  let v = VInt64 42L in
  check string "int64 type name" "int64" (value_type_name v)

let test_value_type_name_float32 () =
  let v = VFloat32 3.14 in
  check string "float32 type name" "float32" (value_type_name v)

let test_value_type_name_float64 () =
  let v = VFloat64 2.718 in
  check string "float64 type name" "float64" (value_type_name v)

let test_value_type_name_bool () =
  let v = VBool true in
  check string "bool type name" "bool" (value_type_name v)

let test_value_type_name_unit () =
  let v = VUnit in
  check string "unit type name" "unit" (value_type_name v)

let test_value_type_name_array () =
  let v = VArray [| VInt32 1l; VInt32 2l |] in
  check string "array type name" "array" (value_type_name v)

let test_value_type_name_record () =
  let v = VRecord ("point", [| VFloat32 1.0; VFloat32 2.0 |]) in
  check string "record type name" "point" (value_type_name v)

let test_value_type_name_variant () =
  let v = VVariant ("option", 0, []) in
  check string "variant type name" "option" (value_type_name v)

(** Test value construction *)
let test_construct_int32 () =
  let v = VInt32 123l in
  match v with VInt32 n -> check int32 "int32 value" 123l n | _ -> fail "wrong type"

let test_construct_array () =
  let arr = [| VInt32 1l; VInt32 2l; VInt32 3l |] in
  let v = VArray arr in
  match v with
  | VArray a -> check int "array length" 3 (Array.length a)
  | _ -> fail "wrong type"

let test_construct_record () =
  let fields = [| VFloat32 1.0; VFloat32 2.0 |] in
  let v = VRecord ("point", fields) in
  match v with
  | VRecord (name, f) ->
      check string "record name" "point" name ;
      check int "field count" 2 (Array.length f)
  | _ -> fail "wrong type"

let test_construct_variant_none () =
  let v = VVariant ("option", 0, []) in
  match v with
  | VVariant (name, tag, args) ->
      check string "variant name" "option" name ;
      check int "variant tag" 0 tag ;
      check int "arg count" 0 (List.length args)
  | _ -> fail "wrong type"

let test_construct_variant_some () =
  let v = VVariant ("option", 1, [ VInt32 42l ]) in
  match v with
  | VVariant (name, tag, args) ->
      check string "variant name" "option" name ;
      check int "variant tag" 1 tag ;
      check int "arg count" 1 (List.length args)
  | _ -> fail "wrong type"

(** Test empty collections *)
let test_empty_array () =
  let v = VArray [||] in
  match v with
  | VArray a -> check int "empty array length" 0 (Array.length a)
  | _ -> fail "wrong type"

let test_nested_array () =
  let inner = [| VInt32 1l; VInt32 2l |] in
  let outer = [| VArray inner; VArray inner |] in
  let v = VArray outer in
  match v with
  | VArray a ->
      check int "outer array length" 2 (Array.length a) ;
      (match a.(0) with
      | VArray inner_a -> check int "inner array length" 2 (Array.length inner_a)
      | _ -> fail "inner not array")
  | _ -> fail "wrong type"

(** Test edge cases *)
let test_max_int32 () =
  let v = VInt32 Int32.max_int in
  match v with
  | VInt32 n -> check bool "max int32" true (n = Int32.max_int)
  | _ -> fail "wrong type"

let test_min_int32 () =
  let v = VInt32 Int32.min_int in
  match v with
  | VInt32 n -> check bool "min int32" true (n = Int32.min_int)
  | _ -> fail "wrong type"

let test_nan_float () =
  let v = VFloat32 nan in
  match v with
  | VFloat32 f -> check bool "is nan" true (Float.is_nan f)
  | _ -> fail "wrong type"

let test_infinity_float () =
  let v = VFloat64 infinity in
  match v with
  | VFloat64 f -> check bool "is infinity" true (f = infinity)
  | _ -> fail "wrong type"

let () =
  run
    "Sarek_value"
    [
      ( "value_type_name",
        [
          test_case "int32" `Quick test_value_type_name_int32;
          test_case "int64" `Quick test_value_type_name_int64;
          test_case "float32" `Quick test_value_type_name_float32;
          test_case "float64" `Quick test_value_type_name_float64;
          test_case "bool" `Quick test_value_type_name_bool;
          test_case "unit" `Quick test_value_type_name_unit;
          test_case "array" `Quick test_value_type_name_array;
          test_case "record" `Quick test_value_type_name_record;
          test_case "variant" `Quick test_value_type_name_variant;
        ] );
      ( "construction",
        [
          test_case "int32" `Quick test_construct_int32;
          test_case "array" `Quick test_construct_array;
          test_case "record" `Quick test_construct_record;
          test_case "variant_none" `Quick test_construct_variant_none;
          test_case "variant_some" `Quick test_construct_variant_some;
        ] );
      ( "collections",
        [
          test_case "empty_array" `Quick test_empty_array;
          test_case "nested_array" `Quick test_nested_array;
        ] );
      ( "edge_cases",
        [
          test_case "max_int32" `Quick test_max_int32;
          test_case "min_int32" `Quick test_min_int32;
          test_case "nan_float" `Quick test_nan_float;
          test_case "infinity_float" `Quick test_infinity_float;
        ] );
    ]
