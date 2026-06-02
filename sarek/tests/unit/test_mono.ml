(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Sarek PPX - Unit Tests for Monomorphization
 *
 * Tests the Sarek_mono module for specializing polymorphic functions:
 * - Type variable detection
 * - Name mangling
 * - Type substitution
 ******************************************************************************)

open Alcotest
open Sarek_types
open Sarek_mono

(** Test has_type_vars on primitives *)
let test_has_type_vars_prim () =
  check bool "int32 has no vars" false (has_type_vars t_int32) ;
  check bool "bool has no vars" false (has_type_vars t_bool) ;
  check bool "unit has no vars" false (has_type_vars t_unit)

(** Test has_type_vars on type variables *)
let test_has_type_vars_tvar () =
  let tv = TVar (ref (Unbound (1, 0))) in
  check bool "tvar has vars" true (has_type_vars tv)

(** Test has_type_vars on linked type variables *)
let test_has_type_vars_linked () =
  let tv = TVar (ref (Link t_int32)) in
  check bool "linked var has no vars" false (has_type_vars tv)

(** Test has_type_vars on function types *)
let test_has_type_vars_fun () =
  let fn1 = TFun ([t_int32], t_bool) in
  check bool "concrete fun has no vars" false (has_type_vars fn1) ;
  let tv = TVar (ref (Unbound (1, 0))) in
  let fn2 = TFun ([tv], t_int32) in
  check bool "fun with tvar in args has vars" true (has_type_vars fn2) ;
  let fn3 = TFun ([t_int32], tv) in
  check bool "fun with tvar in ret has vars" true (has_type_vars fn3)

(** Test has_type_vars on composite types *)
let test_has_type_vars_composite () =
  let tv = TVar (ref (Unbound (1, 0))) in
  check bool "vec with tvar has vars" true (has_type_vars (TVec tv)) ;
  check bool "vec int32 has no vars" false (has_type_vars (TVec t_int32)) ;
  check bool "arr with tvar has vars" true (has_type_vars (TArr (tv, Local))) ;
  check
    bool
    "tuple with tvar has vars"
    true
    (has_type_vars (TTuple [t_int32; tv]))

(** Test normalize_type follows links *)
let test_normalize_type () =
  let linked = TVar (ref (Link t_int32)) in
  let normalized = normalize_type linked in
  match normalized with TPrim TInt32 -> () | _ -> fail "expected int32"

(** Test normalize_type on nested links *)
let test_normalize_nested () =
  let inner = TVar (ref (Link t_bool)) in
  let outer = TVar (ref (Link inner)) in
  let normalized = normalize_type outer in
  match normalized with TPrim TBool -> () | _ -> fail "expected bool"

(** Test types_equal on primitives *)
let test_types_equal_prim () =
  check bool "int32 = int32" true (types_equal t_int32 t_int32) ;
  check bool "int32 != bool" false (types_equal t_int32 t_bool)

(** Test types_equal on linked types *)
let test_types_equal_linked () =
  let linked = TVar (ref (Link t_int32)) in
  check bool "linked = concrete" true (types_equal linked t_int32)

(** Test types_equal on functions *)
let test_types_equal_fun () =
  let fn1 = TFun ([t_int32], t_bool) in
  let fn2 = TFun ([t_int32], t_bool) in
  let fn3 = TFun ([t_bool], t_bool) in
  check bool "same fn = same fn" true (types_equal fn1 fn2) ;
  check bool "diff fn != diff fn" false (types_equal fn1 fn3)

(** Test mangle_type on primitives *)
let test_mangle_type_prim () =
  check string "unit mangles to u" "u" (mangle_type t_unit) ;
  check string "bool mangles to b" "b" (mangle_type t_bool) ;
  check string "int32 mangles to i32" "i32" (mangle_type t_int32)

(** Test mangle_type on registered types *)
let test_mangle_type_reg () =
  check string "float32 mangles to f32" "f32" (mangle_type t_float32) ;
  check string "float64 mangles to f64" "f64" (mangle_type t_float64)

(** Test mangle_type on vectors *)
let test_mangle_type_vec () =
  check string "vec int32" "vi32" (mangle_type (TVec t_int32)) ;
  check string "vec float32" "vf32" (mangle_type (TVec t_float32))

(** Test mangle_type on arrays *)
let test_mangle_type_arr () =
  check string "arr int32" "ai32" (mangle_type (TArr (t_int32, Local)))

(** Test mangle_type on tuples *)
let test_mangle_type_tuple () =
  let t = TTuple [t_int32; t_bool] in
  check string "tuple 2" "T2i32b" (mangle_type t)

(** Test mangle_name *)
let test_mangle_name () =
  check string "no types" "foo" (mangle_name "foo" []) ;
  check string "one type" "foo__i32" (mangle_name "foo" [t_int32]) ;
  check string "two types" "foo__i32_b" (mangle_name "foo" [t_int32; t_bool])

(** Test create_mono_env *)
let test_create_mono_env () =
  let env = create_mono_env () in
  check int "counter starts at 0" 0 !(env.counter) ;
  check int "no specialized funcs" 0 (List.length !(env.specialized))

(** Test get_or_create_instance *)
let test_get_or_create_instance () =
  let env = create_mono_env () in
  let name1 = get_or_create_instance env "foo" [t_int32] in
  let name2 = get_or_create_instance env "foo" [t_int32] in
  let name3 = get_or_create_instance env "foo" [t_bool] in
  check string "same call = same name" name1 name2 ;
  check bool "diff types = diff names" true (name1 <> name3)

(** Test apply_subst on primitives *)
let test_apply_subst_prim () =
  let subst = [(1, t_int32)] in
  let result = apply_subst subst t_bool in
  match result with
  | TPrim TBool -> ()
  | _ -> fail "primitives unchanged by subst"

(** Test apply_subst on type variables *)
let test_apply_subst_tvar () =
  let tv = TVar (ref (Unbound (1, 0))) in
  let subst = [(1, t_int32)] in
  let result = apply_subst subst tv in
  match result with
  | TPrim TInt32 -> ()
  | _ -> fail "tvar should be substituted"

(** Test apply_subst on functions *)
let test_apply_subst_fun () =
  let tv = TVar (ref (Unbound (1, 0))) in
  let fn = TFun ([tv], tv) in
  let subst = [(1, t_int32)] in
  let result = apply_subst subst fn in
  match result with
  | TFun ([TPrim TInt32], TPrim TInt32) -> ()
  | _ -> fail "function args and ret should be substituted"

(** All monomorphization tests *)
let tests =
  [
    ("has_type_vars on primitives", `Quick, test_has_type_vars_prim);
    ("has_type_vars on tvar", `Quick, test_has_type_vars_tvar);
    ("has_type_vars on linked", `Quick, test_has_type_vars_linked);
    ("has_type_vars on functions", `Quick, test_has_type_vars_fun);
    ("has_type_vars on composites", `Quick, test_has_type_vars_composite);
    ("normalize_type follows links", `Quick, test_normalize_type);
    ("normalize_type nested links", `Quick, test_normalize_nested);
    ("types_equal on primitives", `Quick, test_types_equal_prim);
    ("types_equal on linked", `Quick, test_types_equal_linked);
    ("types_equal on functions", `Quick, test_types_equal_fun);
    ("mangle_type primitives", `Quick, test_mangle_type_prim);
    ("mangle_type registered", `Quick, test_mangle_type_reg);
    ("mangle_type vectors", `Quick, test_mangle_type_vec);
    ("mangle_type arrays", `Quick, test_mangle_type_arr);
    ("mangle_type tuples", `Quick, test_mangle_type_tuple);
    ("mangle_name", `Quick, test_mangle_name);
    ("create_mono_env", `Quick, test_create_mono_env);
    ("get_or_create_instance", `Quick, test_get_or_create_instance);
    ("apply_subst on primitives", `Quick, test_apply_subst_prim);
    ("apply_subst on tvar", `Quick, test_apply_subst_tvar);
    ("apply_subst on functions", `Quick, test_apply_subst_fun);
  ]

let () = Alcotest.run "Sarek_mono" [("monomorphization", tests)]
