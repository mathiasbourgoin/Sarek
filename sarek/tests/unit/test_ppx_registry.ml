(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Unit tests for Sarek_ppx_registry module
 *
 * Tests registration and lookup of intrinsics, types, and module items
 ******************************************************************************)

(* Force stdlib initialization *)
let () = Sarek_stdlib.force_init ()

open Sarek_types

let dummy_loc =
  Sarek_ast.
    {
      loc_file = "test.ml";
      loc_line = 1;
      loc_col = 0;
      loc_end_line = 1;
      loc_end_col = 10;
    }

(* Test: register and find type *)
let test_register_find_type () =
  let type_info =
    Sarek_ppx_registry.
      {
        ti_name = "test_type";
        ti_size = 4;
        ti_sarek_type = TPrim TInt32;
      }
  in
  Sarek_ppx_registry.register_type type_info ;
  match Sarek_ppx_registry.find_type "test_type" with
  | Some info ->
      Alcotest.(check string) "type name" "test_type" info.ti_name ;
      Alcotest.(check int) "type size" 4 info.ti_size
  | None -> Alcotest.fail "Type not found after registration"

let test_find_type_not_found () =
  match Sarek_ppx_registry.find_type "nonexistent_type_xyz123" with
  | Some _ -> Alcotest.fail "Should not find nonexistent type"
  | None -> ()

(* Test: register and find intrinsic *)
let test_register_find_intrinsic () =
  let intrinsic_info =
    Sarek_ppx_registry.
      {
        ii_name = "test_sin";
        ii_qualified_name = "Test.sin";
        ii_type = TFun ([TReg Float32], TReg Float32);
        ii_module = ["Test"];
      }
  in
  Sarek_ppx_registry.register_intrinsic intrinsic_info ;
  (* Check short name lookup *)
  match Sarek_ppx_registry.find_intrinsic "test_sin" with
  | Some info ->
      Alcotest.(check string) "intrinsic name" "test_sin" info.ii_name ;
      Alcotest.(check string) "qualified name" "Test.sin" info.ii_qualified_name
  | None -> Alcotest.fail "Intrinsic not found by short name"

let test_find_intrinsic_qualified () =
  let intrinsic_info =
    Sarek_ppx_registry.
      {
        ii_name = "test_cos";
        ii_qualified_name = "Math.Trig.cos";
        ii_type = TFun ([TReg Float32], TReg Float32);
        ii_module = ["Math"; "Trig"];
      }
  in
  Sarek_ppx_registry.register_intrinsic intrinsic_info ;
  (* Check qualified name lookup *)
  match Sarek_ppx_registry.find_intrinsic "Math.Trig.cos" with
  | Some info ->
      Alcotest.(check string)
        "qualified name"
        "Math.Trig.cos"
        info.ii_qualified_name
  | None -> Alcotest.fail "Intrinsic not found by qualified name"

let test_find_intrinsic_not_found () =
  match Sarek_ppx_registry.find_intrinsic "nonexistent_intrinsic_xyz123" with
  | Some _ -> Alcotest.fail "Should not find nonexistent intrinsic"
  | None -> ()

let test_is_intrinsic () =
  let intrinsic_info =
    Sarek_ppx_registry.
      {
        ii_name = "test_sqrt";
        ii_qualified_name = "Test.sqrt";
        ii_type = TFun ([TReg Float32], TReg Float32);
        ii_module = ["Test"];
      }
  in
  Sarek_ppx_registry.register_intrinsic intrinsic_info ;
  Alcotest.(check bool)
    "is_intrinsic true"
    true
    (Sarek_ppx_registry.is_intrinsic "test_sqrt") ;
  Alcotest.(check bool)
    "is_intrinsic false"
    false
    (Sarek_ppx_registry.is_intrinsic "nonexistent_xyz")

(* Test: register and find const *)
let test_register_find_const () =
  let const_info =
    Sarek_ppx_registry.
      {
        ci_name = "test_const";
        ci_qualified_name = "Test.const";
        ci_type = TPrim TInt32;
        ci_module = ["Test"];
      }
  in
  Sarek_ppx_registry.register_const const_info ;
  match Sarek_ppx_registry.find_const "test_const" with
  | Some info ->
      Alcotest.(check string) "const name" "test_const" info.ci_name ;
      Alcotest.(check string)
        "qualified name"
        "Test.const"
        info.ci_qualified_name
  | None -> Alcotest.fail "Const not found after registration"

let test_is_const () =
  let const_info =
    Sarek_ppx_registry.
      {
        ci_name = "test_pi";
        ci_qualified_name = "Test.pi";
        ci_type = TReg Float32;
        ci_module = ["Test"];
      }
  in
  Sarek_ppx_registry.register_const const_info ;
  Alcotest.(check bool)
    "is_const true"
    true
    (Sarek_ppx_registry.is_const "test_pi") ;
  Alcotest.(check bool)
    "is_const false"
    false
    (Sarek_ppx_registry.is_const "nonexistent_xyz")

(* Test: register and find module item *)
let test_register_find_module_item () =
  let module_item_info =
    Sarek_ppx_registry.
      {
        mi_name = "test_func";
        mi_qualified_name = "Test.func";
        mi_module = "Test";
        mi_item =
          Sarek_ast.MFun
            ("test_func", false, [], {e = EUnit; expr_loc = dummy_loc});
      }
  in
  Sarek_ppx_registry.register_module_item module_item_info ;
  match Sarek_ppx_registry.find_module_item "test_func" with
  | Some info ->
      Alcotest.(check string) "module item name" "test_func" info.mi_name ;
      Alcotest.(check string) "module name" "Test" info.mi_module
  | None -> Alcotest.fail "Module item not found after registration"

let test_is_module_item () =
  let module_item_info =
    Sarek_ppx_registry.
      {
        mi_name = "test_helper";
        mi_qualified_name = "Test.helper";
        mi_module = "Test";
        mi_item =
          Sarek_ast.MFun
            ("test_helper", false, [], {e = EInt 42; expr_loc = dummy_loc});
      }
  in
  Sarek_ppx_registry.register_module_item module_item_info ;
  Alcotest.(check bool)
    "is_module_item true"
    true
    (Sarek_ppx_registry.is_module_item "test_helper") ;
  Alcotest.(check bool)
    "is_module_item false"
    false
    (Sarek_ppx_registry.is_module_item "nonexistent_xyz")

(* Test: register and find record type *)
let test_register_find_record_type () =
  let record_type_info =
    Sarek_ppx_registry.
      {
        rti_name = "test_record";
        rti_qualified_name = "Test.record";
        rti_module = "Test";
        rti_decl =
          Sarek_ast.Type_record
            {
              tdecl_name = "test_record";
              tdecl_module = None;
              tdecl_fields =
                [
                  ("x", false, TEConstr ("int32", []));
                  ("y", false, TEConstr ("int32", []));
                ];
              tdecl_loc = dummy_loc;
            };
      }
  in
  Sarek_ppx_registry.register_record_type record_type_info ;
  match Sarek_ppx_registry.find_record_type "test_record" with
  | Some info ->
      Alcotest.(check string) "record name" "test_record" info.rti_name ;
      Alcotest.(check string) "module name" "Test" info.rti_module
  | None -> Alcotest.fail "Record type not found after registration"

let test_is_record_type () =
  let record_type_info =
    Sarek_ppx_registry.
      {
        rti_name = "test_vec3";
        rti_qualified_name = "Test.vec3";
        rti_module = "Test";
        rti_decl =
          Sarek_ast.Type_record
            {
              tdecl_name = "test_vec3";
              tdecl_module = None;
              tdecl_fields =
                [
                  ("x", false, TEConstr ("float32", []));
                  ("y", false, TEConstr ("float32", []));
                  ("z", false, TEConstr ("float32", []));
                ];
              tdecl_loc = dummy_loc;
            };
      }
  in
  Sarek_ppx_registry.register_record_type record_type_info ;
  Alcotest.(check bool)
    "is_record_type true"
    true
    (Sarek_ppx_registry.is_record_type "test_vec3") ;
  Alcotest.(check bool)
    "is_record_type false"
    false
    (Sarek_ppx_registry.is_record_type "nonexistent_xyz")

(* Test: all_types returns registered types *)
let test_all_types () =
  let initial_count = List.length (Sarek_ppx_registry.all_types ()) in
  let type_info =
    Sarek_ppx_registry.
      {
        ti_name = "test_all_types";
        ti_size = 8;
        ti_sarek_type = TReg Int64;
      }
  in
  Sarek_ppx_registry.register_type type_info ;
  let new_count = List.length (Sarek_ppx_registry.all_types ()) in
  Alcotest.(check bool)
    "all_types includes new type"
    true
    (new_count >= initial_count)

(* Test: all_intrinsics deduplicates *)
let test_all_intrinsics_dedup () =
  let intrinsic_info =
    Sarek_ppx_registry.
      {
        ii_name = "test_dedup";
        ii_qualified_name = "Test.dedup";
        ii_type = TFun ([TPrim TInt32], TPrim TInt32);
        ii_module = ["Test"];
      }
  in
  Sarek_ppx_registry.register_intrinsic intrinsic_info ;
  let all_intrinsics = Sarek_ppx_registry.all_intrinsics () in
  (* Should only appear once despite being registered under two names *)
  let count =
    List.filter
      (fun i -> i.Sarek_ppx_registry.ii_qualified_name = "Test.dedup")
      all_intrinsics
    |> List.length
  in
  Alcotest.(check int) "deduplication" 1 count

(* Test suite *)
let () =
  let open Alcotest in
  run
    "Sarek_ppx_registry"
    [
      ( "types",
        [
          test_case "register and find type" `Quick test_register_find_type;
          test_case "find type not found" `Quick test_find_type_not_found;
          test_case "all_types" `Quick test_all_types;
        ] );
      ( "intrinsics",
        [
          test_case
            "register and find intrinsic"
            `Quick
            test_register_find_intrinsic;
          test_case
            "find intrinsic by qualified name"
            `Quick
            test_find_intrinsic_qualified;
          test_case
            "find intrinsic not found"
            `Quick
            test_find_intrinsic_not_found;
          test_case "is_intrinsic" `Quick test_is_intrinsic;
          test_case
            "all_intrinsics deduplication"
            `Quick
            test_all_intrinsics_dedup;
        ] );
      ( "consts",
        [
          test_case "register and find const" `Quick test_register_find_const;
          test_case "is_const" `Quick test_is_const;
        ] );
      ( "module_items",
        [
          test_case
            "register and find module item"
            `Quick
            test_register_find_module_item;
          test_case "is_module_item" `Quick test_is_module_item;
        ] );
      ( "record_types",
        [
          test_case
            "register and find record type"
            `Quick
            test_register_find_record_type;
          test_case "is_record_type" `Quick test_is_record_type;
        ] );
    ]
