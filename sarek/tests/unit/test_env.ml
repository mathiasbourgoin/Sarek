(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Unit tests for Sarek_env
 *
 * Tests typing environment operations.
 ******************************************************************************)

(* Force stdlib initialization to register intrinsics *)
let () = Sarek_stdlib.force_init ()

open Sarek_types
open Sarek_env

(* Test adding and finding variables *)
let test_add_find_var () =
  let env = empty in
  let info =
    {
      vi_type = t_int32;
      vi_mutable = false;
      vi_is_param = true;
      vi_index = 0;
      vi_is_vec = false;
    }
  in
  let env = add_var "x" info env in

  Alcotest.(check bool)
    "find_var returns Some for added var"
    true
    (match find_var "x" env with Some _ -> true | None -> false) ;

  Alcotest.(check bool)
    "find_var returns None for unknown var"
    true
    (match find_var "y" env with Some _ -> false | None -> true) ;

  match find_var "x" env with
  | Some found_info ->
      Alcotest.(check bool)
        "found var has correct type"
        true
        (match found_info.vi_type with TPrim TInt32 -> true | _ -> false) ;
      Alcotest.(check bool)
        "found var has correct is_param"
        true
        found_info.vi_is_param
  | None -> Alcotest.fail "var x should be found"

(* Test variable shadowing *)
let test_var_shadowing () =
  let env = empty in
  let info1 =
    {
      vi_type = t_int32;
      vi_mutable = false;
      vi_is_param = false;
      vi_index = 0;
      vi_is_vec = false;
    }
  in
  let info2 =
    {
      vi_type = t_float32;
      vi_mutable = true;
      vi_is_param = false;
      vi_index = 1;
      vi_is_vec = false;
    }
  in
  let env = add_var "x" info1 env in
  let env = add_var "x" info2 env in

  match find_var "x" env with
  | Some found ->
      Alcotest.(check bool)
        "shadowed var has new type"
        true
        (match found.vi_type with TReg Float32 -> true | _ -> false) ;
      Alcotest.(check bool)
        "shadowed var has new mutability"
        true
        found.vi_mutable
  | None -> Alcotest.fail "var x should be found"

(* Test intrinsic constants *)
let test_intrinsic_consts () =
  let env = with_stdlib empty in

  Alcotest.(check bool)
    "thread_idx_x is found"
    true
    (match find_intrinsic_const "thread_idx_x" env with
    | Some _ -> true
    | None -> false) ;

  Alcotest.(check bool)
    "block_idx_x is found"
    true
    (match find_intrinsic_const "block_idx_x" env with
    | Some _ -> true
    | None -> false) ;

  Alcotest.(check bool)
    "global_thread_id is found"
    true
    (match find_intrinsic_const "global_thread_id" env with
    | Some _ -> true
    | None -> false) ;

  match find_intrinsic_const "thread_idx_x" env with
  | Some info ->
      Alcotest.(check bool)
        "thread_idx_x has int32 type"
        true
        (match info.const_type with TPrim TInt32 -> true | _ -> false) ;
      (* Now uses IntrinsicRef for JIT resolution - path from registry *)
      let ref_name = intrinsic_ref_name info.const_ref in
      Alcotest.(check bool)
        "thread_idx_x has Gpu in path"
        true
        (String.length ref_name > 0
        && String.sub
             ref_name
             (String.length ref_name - String.length "thread_idx_x")
             (String.length "thread_idx_x")
           = "thread_idx_x")
  | None -> Alcotest.fail "thread_idx_x should be found"

(* Test intrinsic functions *)
let test_intrinsic_funs () =
  let env = with_stdlib empty in

  Alcotest.(check bool)
    "block_barrier is found"
    true
    (match find_intrinsic_fun "block_barrier" env with
    | Some _ -> true
    | None -> false) ;

  Alcotest.(check bool)
    "sin is found"
    true
    (match find_intrinsic_fun "sin" env with Some _ -> true | None -> false) ;

  Alcotest.(check bool)
    "cos is found"
    true
    (match find_intrinsic_fun "cos" env with Some _ -> true | None -> false) ;

  match find_intrinsic_fun "sin" env with
  | Some info ->
      (match info.intr_type with
      | TFun ([TReg Float32], TReg Float32) ->
          Alcotest.(check pass) "sin has correct type" () ()
      | _ -> Alcotest.fail "sin should have type float32 -> float32") ;
      (* Now uses IntrinsicRef for JIT resolution - path from registry *)
      let ref_name = intrinsic_ref_name info.intr_ref in
      Alcotest.(check bool)
        "sin has Float32 in path"
        true
        (String.length ref_name > 0
        && String.sub
             ref_name
             (String.length ref_name - String.length "sin")
             (String.length "sin")
           = "sin")
  | None -> Alcotest.fail "sin should be found"

(* Test lookup function *)
let test_lookup () =
  let env = with_stdlib empty in
  let var_info =
    {
      vi_type = t_int32;
      vi_mutable = false;
      vi_is_param = true;
      vi_index = 0;
      vi_is_vec = false;
    }
  in
  let env = add_var "my_var" var_info env in

  (* Test lookup of variable *)
  (match lookup "my_var" env with
  | LVar info ->
      Alcotest.(check bool)
        "lookup finds var with correct type"
        true
        (match info.vi_type with TPrim TInt32 -> true | _ -> false)
  | _ -> Alcotest.fail "my_var should be found as LVar") ;

  (* Test lookup of intrinsic const *)
  (match lookup "thread_idx_x" env with
  | LIntrinsicConst _ ->
      Alcotest.(check pass) "lookup finds intrinsic const" () ()
  | _ -> Alcotest.fail "thread_idx_x should be found as LIntrinsicConst") ;

  (* Test lookup of intrinsic fun *)
  (match lookup "sin" env with
  | LIntrinsicFun _ -> Alcotest.(check pass) "lookup finds intrinsic fun" () ()
  | _ -> Alcotest.fail "sin should be found as LIntrinsicFun") ;

  (* Test lookup of unknown *)
  match lookup "unknown_var" env with
  | LNotFound ->
      Alcotest.(check pass) "lookup returns LNotFound for unknown" () ()
  | _ -> Alcotest.fail "unknown_var should not be found"

(* Test custom types *)
let test_custom_types () =
  let env = empty in
  let record_info =
    TIRecord
      {
        ti_name = "point";
        ti_fields = [("x", t_float32, false); ("y", t_float32, false)];
      }
  in
  let env = add_type "point" record_info env in

  Alcotest.(check bool)
    "find_type returns Some for added type"
    true
    (match find_type "point" env with Some _ -> true | None -> false) ;

  (* Fields should be added automatically *)
  Alcotest.(check bool)
    "field x is found"
    true
    (match find_field "x" env with Some _ -> true | None -> false) ;

  Alcotest.(check bool)
    "field y is found"
    true
    (match find_field "y" env with Some _ -> true | None -> false) ;

  match find_field "x" env with
  | Some (type_name, idx, ty, mutable_) ->
      Alcotest.(check string) "field x has correct type name" "point" type_name ;
      Alcotest.(check int) "field x has correct index" 0 idx ;
      Alcotest.(check bool)
        "field x has correct type"
        true
        (match ty with TReg Float32 -> true | _ -> false) ;
      Alcotest.(check bool) "field x is immutable" false mutable_
  | None -> Alcotest.fail "field x should be found"

(* Test variant types *)
let test_variant_types () =
  let env = empty in
  let variant_info =
    TIVariant
      {
        ti_name = "option_int";
        ti_constrs = [("None", None); ("Some", Some t_int32)];
      }
  in
  let env = add_type "option_int" variant_info env in

  Alcotest.(check bool)
    "constructor None is found"
    true
    (match find_constructor "None" env with Some _ -> true | None -> false) ;

  Alcotest.(check bool)
    "constructor Some is found"
    true
    (match find_constructor "Some" env with Some _ -> true | None -> false) ;

  match find_constructor "Some" env with
  | Some (type_name, arg_type) ->
      Alcotest.(check string)
        "Some has correct type name"
        "option_int"
        type_name ;
      Alcotest.(check bool)
        "Some has argument type"
        true
        (match arg_type with Some (TPrim TInt32) -> true | _ -> false)
  | None -> Alcotest.fail "constructor Some should be found"

(* Test level management *)
let test_levels () =
  let env = empty in
  Alcotest.(check int) "initial level is 0" 0 env.current_level ;

  let env = enter_level env in
  Alcotest.(check int) "after enter_level is 1" 1 env.current_level ;

  let env = enter_level env in
  Alcotest.(check int) "after second enter_level is 2" 2 env.current_level ;

  let env = exit_level env in
  Alcotest.(check int) "after exit_level is 1" 1 env.current_level ;

  let env = exit_level env in
  Alcotest.(check int) "after second exit_level is 0" 0 env.current_level ;

  let env = exit_level env in
  Alcotest.(check int) "exit_level doesn't go negative" 0 env.current_level

(* Test suite *)
let () =
  Alcotest.run
    "Sarek_env"
    [
      ( "variables",
        [
          Alcotest.test_case "add and find var" `Quick test_add_find_var;
          Alcotest.test_case "variable shadowing" `Quick test_var_shadowing;
        ] );
      ( "intrinsics",
        [
          Alcotest.test_case "intrinsic constants" `Quick test_intrinsic_consts;
          Alcotest.test_case "intrinsic functions" `Quick test_intrinsic_funs;
        ] );
      ("lookup", [Alcotest.test_case "lookup function" `Quick test_lookup]);
      ( "custom_types",
        [
          Alcotest.test_case "record types" `Quick test_custom_types;
          Alcotest.test_case "variant types" `Quick test_variant_types;
        ] );
      ("levels", [Alcotest.test_case "level management" `Quick test_levels]);
    ]
