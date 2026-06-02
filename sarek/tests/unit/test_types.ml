(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Unit tests for Sarek_types
 *
 * Tests type representation and unification.
 ******************************************************************************)

open Sarek_types

(* Test helpers *)
let check_ok msg = function Ok () -> () | Error _ -> Alcotest.fail msg

let check_error msg = function Ok () -> Alcotest.fail msg | Error _ -> ()

(* Type equality for testing *)
let typ_eq t1 t2 =
  let rec eq t1 t2 =
    match (repr t1, repr t2) with
    | TPrim p1, TPrim p2 -> p1 = p2
    | TVec t1, TVec t2 -> eq t1 t2
    | TArr (t1, m1), TArr (t2, m2) -> eq t1 t2 && m1 = m2
    | TFun (args1, ret1), TFun (args2, ret2) ->
        List.length args1 = List.length args2
        && List.for_all2 eq args1 args2
        && eq ret1 ret2
    | TRecord (n1, _), TRecord (n2, _) -> n1 = n2
    | TVariant (n1, _), TVariant (n2, _) -> n1 = n2
    | TTuple ts1, TTuple ts2 ->
        List.length ts1 = List.length ts2 && List.for_all2 eq ts1 ts2
    | TVar {contents = Unbound (id1, _)}, TVar {contents = Unbound (id2, _)} ->
        id1 = id2
    | _, _ -> false
  in
  eq t1 t2

let typ_testable = Alcotest.testable pp_typ typ_eq

(* Tests for primitive types *)
let test_prim_types () =
  Alcotest.(check bool)
    "t_unit is TPrim TUnit"
    true
    (match t_unit with TPrim TUnit -> true | _ -> false) ;
  Alcotest.(check bool)
    "t_bool is TPrim TBool"
    true
    (match t_bool with TPrim TBool -> true | _ -> false) ;
  Alcotest.(check bool)
    "t_int32 is TPrim TInt32"
    true
    (match t_int32 with TPrim TInt32 -> true | _ -> false) ;
  Alcotest.(check bool)
    "t_float32 is TReg float32"
    true
    (match t_float32 with TReg Float32 -> true | _ -> false)

(* Tests for type constructors *)
let test_type_constructors () =
  let vec_int = t_vec t_int32 in
  Alcotest.(check bool)
    "t_vec creates TVec"
    true
    (match vec_int with TVec (TPrim TInt32) -> true | _ -> false) ;

  let arr_float = t_arr t_float32 Local in
  Alcotest.(check bool)
    "t_arr creates TArr"
    true
    (match arr_float with TArr (TReg Float32, Local) -> true | _ -> false) ;

  let fn = t_fun [t_int32; t_float32] t_bool in
  Alcotest.(check bool)
    "t_fun creates TFun"
    true
    (match fn with
    | TFun ([TPrim TInt32; TReg Float32], TPrim TBool) -> true
    | _ -> false)

(* Tests for fresh type variables *)
let test_fresh_tvar () =
  reset_tvar_counter () ;
  let t1 = fresh_tvar () in
  let t2 = fresh_tvar () in
  Alcotest.(check bool)
    "fresh tvars are different"
    true
    (match (t1, t2) with
    | TVar {contents = Unbound (id1, _)}, TVar {contents = Unbound (id2, _)} ->
        id1 <> id2
    | _ -> false)

(* Tests for repr (following links) *)
let test_repr () =
  reset_tvar_counter () ;
  let tv = fresh_tvar () in
  (* Link the tvar to int32 *)
  (match tv with TVar r -> r := Link t_int32 | _ -> ()) ;
  let resolved = repr tv in
  Alcotest.(check typ_testable) "repr follows links" t_int32 resolved

(* Tests for unification - same types *)
let test_unify_same_types () =
  check_ok "int32 unifies with int32" (unify t_int32 t_int32) ;
  check_ok "float32 unifies with float32" (unify t_float32 t_float32) ;
  check_ok "bool unifies with bool" (unify t_bool t_bool) ;
  check_ok "unit unifies with unit" (unify t_unit t_unit)

(* Tests for unification - different primitive types *)
let test_unify_different_prims () =
  reset_tvar_counter () ;
  check_error "int32 doesn't unify with float32" (unify t_int32 t_float32) ;
  check_error "bool doesn't unify with int32" (unify t_bool t_int32)

(* Tests for unification with type variables *)
let test_unify_with_tvar () =
  reset_tvar_counter () ;
  let tv = fresh_tvar () in
  check_ok "tvar unifies with int32" (unify tv t_int32) ;
  (* After unification, tvar should be linked to int32 *)
  Alcotest.(check typ_testable) "tvar is now int32" t_int32 (repr tv)

let test_unify_tvar_with_tvar () =
  reset_tvar_counter () ;
  let tv1 = fresh_tvar () in
  let tv2 = fresh_tvar () in
  check_ok "tvar1 unifies with tvar2" (unify tv1 tv2) ;
  (* After unification, both should resolve to the same type *)
  let r1 = repr tv1 in
  let r2 = repr tv2 in
  Alcotest.(check typ_testable) "both tvars resolve to same" r1 r2

(* Tests for unification with compound types *)
let test_unify_vectors () =
  reset_tvar_counter () ;
  let v1 = t_vec t_int32 in
  let v2 = t_vec t_int32 in
  check_ok "vec int32 unifies with vec int32" (unify v1 v2) ;

  let v3 = t_vec t_float32 in
  check_error "vec int32 doesn't unify with vec float32" (unify v1 v3) ;

  let tv = fresh_tvar () in
  let v4 = t_vec tv in
  check_ok "vec tvar unifies with vec int32" (unify v4 v1) ;
  Alcotest.(check typ_testable) "tvar is now int32" t_int32 (repr tv)

let test_unify_functions () =
  reset_tvar_counter () ;
  let f1 = t_fun [t_int32] t_bool in
  let f2 = t_fun [t_int32] t_bool in
  check_ok "same function types unify" (unify f1 f2) ;

  let f3 = t_fun [t_float32] t_bool in
  check_error "different arg types don't unify" (unify f1 f3) ;

  let f4 = t_fun [t_int32; t_int32] t_bool in
  check_error "different arities don't unify" (unify f1 f4)

let test_unify_tuples () =
  reset_tvar_counter () ;
  let tup1 = TTuple [t_int32; t_float32] in
  let tup2 = TTuple [t_int32; t_float32] in
  check_ok "same tuples unify" (unify tup1 tup2) ;

  let tup3 = TTuple [t_int32; t_int32] in
  check_error "different element types don't unify" (unify tup1 tup3) ;

  let tup4 = TTuple [t_int32] in
  check_error "different lengths don't unify" (unify tup1 tup4)

(* Tests for occurs check *)
let test_occurs_check () =
  reset_tvar_counter () ;
  let tv = fresh_tvar () in
  let vec_tv = t_vec tv in
  (* Try to unify tv with vec tv - should fail due to occurs check *)
  check_error "occurs check prevents infinite type" (unify tv vec_tv)

(* Tests for is_numeric, is_integer, is_float *)
let test_type_predicates () =
  Alcotest.(check bool) "int32 is numeric" true (is_numeric t_int32) ;
  Alcotest.(check bool) "int64 is numeric" true (is_numeric t_int64) ;
  Alcotest.(check bool) "float32 is numeric" true (is_numeric t_float32) ;
  Alcotest.(check bool) "float64 is numeric" true (is_numeric t_float64) ;
  Alcotest.(check bool) "bool is not numeric" false (is_numeric t_bool) ;
  Alcotest.(check bool) "unit is not numeric" false (is_numeric t_unit) ;

  Alcotest.(check bool) "int32 is integer" true (is_integer t_int32) ;
  Alcotest.(check bool) "int64 is integer" true (is_integer t_int64) ;
  Alcotest.(check bool) "float32 is not integer" false (is_integer t_float32) ;

  Alcotest.(check bool) "float32 is float" true (is_float t_float32) ;
  Alcotest.(check bool) "float64 is float" true (is_float t_float64) ;
  Alcotest.(check bool) "int32 is not float" false (is_float t_int32)

(* Test suite *)
let () =
  Alcotest.run
    "Sarek_types"
    [
      ( "primitives",
        [
          Alcotest.test_case "primitive types" `Quick test_prim_types;
          Alcotest.test_case "type constructors" `Quick test_type_constructors;
          Alcotest.test_case "fresh tvar" `Quick test_fresh_tvar;
          Alcotest.test_case "repr follows links" `Quick test_repr;
        ] );
      ( "unification",
        [
          Alcotest.test_case "same types" `Quick test_unify_same_types;
          Alcotest.test_case "different prims" `Quick test_unify_different_prims;
          Alcotest.test_case "with tvar" `Quick test_unify_with_tvar;
          Alcotest.test_case "tvar with tvar" `Quick test_unify_tvar_with_tvar;
          Alcotest.test_case "vectors" `Quick test_unify_vectors;
          Alcotest.test_case "functions" `Quick test_unify_functions;
          Alcotest.test_case "tuples" `Quick test_unify_tuples;
          Alcotest.test_case "occurs check" `Quick test_occurs_check;
        ] );
      ( "predicates",
        [Alcotest.test_case "type predicates" `Quick test_type_predicates] );
    ]
