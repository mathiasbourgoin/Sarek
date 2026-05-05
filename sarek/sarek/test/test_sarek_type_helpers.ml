(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Unit tests for Sarek_type_helpers module *)

open Alcotest
open Sarek_value
open Sarek_type_helpers

(** Mock helper module for testing *)
module Mock_point_helpers : HELPERS with type t = float * float = struct
  type t = float * float

  let type_id = Sarek_ir_types.Type_id.create ()

  let from_values arr =
    match arr with
    | [| VFloat64 x; VFloat64 y |] -> (x, y)
    | _ -> failwith "invalid point values"

  let to_values (x, y) = [| VFloat64 x; VFloat64 y |]

  let get_field (x, y) name =
    match name with
    | "x" -> VFloat64 x
    | "y" -> VFloat64 y
    | _ -> failwith ("unknown field: " ^ name)

  let to_value (x, y) = VRecord ("point", [| VFloat64 x; VFloat64 y |])

  let from_value = function
    | VRecord ("point", [| VFloat64 x; VFloat64 y |]) -> (x, y)
    | _ -> failwith "invalid point record"
end

(** Test registering a type helper *)
let test_register_helper () =
  register "test_point" (AnyHelpers (module Mock_point_helpers)) ;
  check bool "Helper registered" true (has_helpers "test_point")

(** Test checking for non-existent helpers *)
let test_has_helpers_missing () =
  check bool "Non-existent helper" false (has_helpers "nonexistent_type_xyz")

(** Test lookup of registered helper *)
let test_lookup_helper () =
  register "lookup_test_point" (AnyHelpers (module Mock_point_helpers)) ;
  let helpers_opt = lookup "lookup_test_point" in
  check bool "Lookup succeeds" true (Option.is_some helpers_opt)

(** Test lookup of missing helper *)
let test_lookup_missing () =
  let helpers_opt = lookup "missing_type_xyz" in
  check bool "Lookup missing returns None" true (Option.is_none helpers_opt)

(** Test from_values conversion *)
let test_from_values () =
  register "from_values_point" (AnyHelpers (module Mock_point_helpers)) ;
  match lookup "from_values_point" with
  | Some helpers ->
      let arr = [| VFloat64 3.0; VFloat64 4.0 |] in
      let v = helpers.from_values arr in
      (match v with
      | VRecord ("point", fields) ->
          check int "2 fields" 2 (Array.length fields) ;
          (match fields.(0) with
          | VFloat64 x -> check (float 0.001) "x = 3.0" 3.0 x
          | _ -> fail "wrong x type") ;
          (match fields.(1) with
          | VFloat64 y -> check (float 0.001) "y = 4.0" 4.0 y
          | _ -> fail "wrong y type")
      | _ -> fail "not a record")
  | None -> fail "helper not found"

(** Test to_values conversion *)
let test_to_values () =
  register "to_values_point" (AnyHelpers (module Mock_point_helpers)) ;
  match lookup "to_values_point" with
  | Some helpers ->
      let v = VRecord ("point", [| VFloat64 1.0; VFloat64 2.0 |]) in
      let arr = helpers.to_values v in
      check int "2 values" 2 (Array.length arr) ;
      (match arr.(0) with
      | VFloat64 x -> check (float 0.001) "x = 1.0" 1.0 x
      | _ -> fail "wrong x type") ;
      (match arr.(1) with
      | VFloat64 y -> check (float 0.001) "y = 2.0" 2.0 y
      | _ -> fail "wrong y type")
  | None -> fail "helper not found"

(** Test get_field *)
let test_get_field () =
  register "get_field_point" (AnyHelpers (module Mock_point_helpers)) ;
  match lookup "get_field_point" with
  | Some helpers ->
      let v = VRecord ("point", [| VFloat64 5.0; VFloat64 6.0 |]) in
      let x = helpers.get_field v "x" in
      (match x with
      | VFloat64 xval -> check (float 0.001) "field x = 5.0" 5.0 xval
      | _ -> fail "wrong x type") ;
      let y = helpers.get_field v "y" in
      (match y with
      | VFloat64 yval -> check (float 0.001) "field y = 6.0" 6.0 yval
      | _ -> fail "wrong y type")
  | None -> fail "helper not found"

(** Test multiple type registrations *)
let test_multiple_registrations () =
  register "type_a" (AnyHelpers (module Mock_point_helpers)) ;
  register "type_b" (AnyHelpers (module Mock_point_helpers)) ;
  check bool "Type A registered" true (has_helpers "type_a") ;
  check bool "Type B registered" true (has_helpers "type_b")

(** Test re-registration overwrites *)
let test_re_registration () =
  register "overwrite_test" (AnyHelpers (module Mock_point_helpers)) ;
  check bool "Initially registered" true (has_helpers "overwrite_test") ;
  (* Re-register - should overwrite *)
  register "overwrite_test" (AnyHelpers (module Mock_point_helpers)) ;
  check bool "Still registered after overwrite" true
    (has_helpers "overwrite_test")

(** Test AnyHelpers type wrapper *)
let test_any_helpers_wrapper () =
  let wrapped = AnyHelpers (module Mock_point_helpers) in
  register "wrapped_test" wrapped ;
  check bool "Wrapped helper works" true (has_helpers "wrapped_test")

let () =
  run
    "Sarek_type_helpers"
    [
      ( "registration",
        [
          test_case "register_helper" `Quick test_register_helper;
          test_case "has_helpers_missing" `Quick test_has_helpers_missing;
          test_case "multiple_registrations" `Quick test_multiple_registrations;
          test_case "re_registration" `Quick test_re_registration;
        ] );
      ( "lookup",
        [
          test_case "lookup_helper" `Quick test_lookup_helper;
          test_case "lookup_missing" `Quick test_lookup_missing;
        ] );
      ( "conversion",
        [
          test_case "from_values" `Quick test_from_values;
          test_case "to_values" `Quick test_to_values;
          test_case "get_field" `Quick test_get_field;
        ] );
      ( "wrapper",
        [
          test_case "any_helpers_wrapper" `Quick test_any_helpers_wrapper;
        ] );
    ]
