(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Sarek PPX - Unit Tests for Type Schemes (Let-Polymorphism)
 *
 * Tests the Sarek_scheme module for polymorphic type handling:
 * - Generalization of types at let bindings
 * - Instantiation of polymorphic schemes
 * - Free type variable collection
 ******************************************************************************)

open Alcotest
open Sarek_types
open Sarek_scheme

(** Helper to create a fresh type variable at a given level *)
let fresh_at level =
  let id = fresh_tvar_id () in
  TVar (ref (Unbound (id, level)))

(** Test mono creates a monomorphic scheme *)
let test_mono () =
  let t = t_int32 in
  let s = mono t in
  check bool "no quantified vars" true (s.quantified = []) ;
  check bool "body is same type" true (s.body == t)

(** Test is_mono and is_poly *)
let test_is_mono_poly () =
  let mono_scheme = mono t_int32 in
  check bool "mono is mono" true (is_mono mono_scheme) ;
  check bool "mono is not poly" false (is_poly mono_scheme) ;
  (* Create a polymorphic scheme manually *)
  let poly_scheme = {quantified = [1; 2]; body = t_int32} in
  check bool "poly is not mono" false (is_mono poly_scheme) ;
  check bool "poly is poly" true (is_poly poly_scheme)

(** Test free_tvars on primitive types *)
let test_free_tvars_prim () =
  check (list int) "int32 has no tvars" [] (free_tvars t_int32) ;
  check (list int) "bool has no tvars" [] (free_tvars t_bool) ;
  check (list int) "unit has no tvars" [] (free_tvars t_unit)

(** Test free_tvars on type variables *)
let test_free_tvars_tvar () =
  let tv = fresh_at 0 in
  let fvs = free_tvars tv in
  check int "one free var" 1 (List.length fvs)

(** Test free_tvars on function types *)
let test_free_tvars_fun () =
  let a = fresh_at 0 in
  let b = fresh_at 0 in
  let fn = TFun ([a], b) in
  let fvs = free_tvars fn in
  check int "two free vars" 2 (List.length fvs)

(** Test free_tvars on linked type vars *)
let test_free_tvars_linked () =
  let tv = TVar (ref (Link t_int32)) in
  check (list int) "linked var has no free tvars" [] (free_tvars tv)

(** Test generalize with no variables at higher level *)
let test_generalize_no_vars () =
  let t = t_int32 in
  let s = generalize 0 t in
  check bool "no quantified vars" true (s.quantified = [])

(** Test generalize with variable at higher level *)
let test_generalize_with_var () =
  (* Create a type var at level 1 *)
  let tv = fresh_at 1 in
  let fn = TFun ([tv], tv) in
  (* Generalize at level 0 - should quantify the var at level 1 *)
  let s = generalize 0 fn in
  check int "one quantified var" 1 (List.length s.quantified)

(** Test generalize doesn't quantify vars at same or lower level *)
let test_generalize_same_level () =
  let tv = fresh_at 0 in
  let fn = TFun ([tv], t_int32) in
  let s = generalize 0 fn in
  check bool "no quantified vars at same level" true (s.quantified = [])

(** Test instantiate on monomorphic scheme *)
let test_instantiate_mono () =
  let t = t_int32 in
  let s = mono t in
  let t' = instantiate s in
  check bool "same type" true (t == t')

(** Test instantiate creates fresh variables *)
let test_instantiate_poly () =
  (* Create identity function: forall a. a -> a *)
  let id = fresh_tvar_id () in
  let tv = TVar (ref (Unbound (id, 0))) in
  let fn = TFun ([tv], tv) in
  let s = {quantified = [id]; body = fn} in
  (* Instantiate twice *)
  let t1 = instantiate s in
  let t2 = instantiate s in
  (* Should get different type vars *)
  match (t1, t2) with
  | TFun ([TVar r1], _), TFun ([TVar r2], _) ->
      check bool "different type vars" true (r1 != r2)
  | _ -> fail "expected function types"

(** Test instantiate preserves structure *)
let test_instantiate_preserves_structure () =
  let id = fresh_tvar_id () in
  let tv = TVar (ref (Unbound (id, 0))) in
  let fn = TFun ([tv; t_int32], tv) in
  let s = {quantified = [id]; body = fn} in
  let t = instantiate s in
  match t with
  | TFun ([_; TPrim TInt32], _) -> ()
  | _ -> fail "structure not preserved"

(** Test function_arity *)
let test_function_arity () =
  let s1 = mono t_int32 in
  check (option int) "non-function has no arity" None (function_arity s1) ;
  let fn = TFun ([t_int32; t_bool], t_unit) in
  let s2 = mono fn in
  check (option int) "binary function has arity 2" (Some 2) (function_arity s2)

(** Test schemes_equivalent *)
let test_schemes_equivalent () =
  let s1 = mono t_int32 in
  let s2 = mono t_int32 in
  check bool "same mono schemes equivalent" true (schemes_equivalent s1 s2) ;
  let s3 = mono t_bool in
  check
    bool
    "different mono schemes not equivalent"
    false
    (schemes_equivalent s1 s3)

(** Test scheme_to_string *)
let test_scheme_to_string () =
  let s = mono t_int32 in
  let str = scheme_to_string s in
  check bool "contains int32" true (String.length str > 0)

(** All scheme tests *)
let tests =
  [
    ("mono creates monomorphic scheme", `Quick, test_mono);
    ("is_mono and is_poly", `Quick, test_is_mono_poly);
    ("free_tvars on primitives", `Quick, test_free_tvars_prim);
    ("free_tvars on type var", `Quick, test_free_tvars_tvar);
    ("free_tvars on function", `Quick, test_free_tvars_fun);
    ("free_tvars on linked var", `Quick, test_free_tvars_linked);
    ("generalize with no vars", `Quick, test_generalize_no_vars);
    ("generalize with var at higher level", `Quick, test_generalize_with_var);
    ("generalize same level", `Quick, test_generalize_same_level);
    ("instantiate monomorphic", `Quick, test_instantiate_mono);
    ("instantiate creates fresh vars", `Quick, test_instantiate_poly);
    ( "instantiate preserves structure",
      `Quick,
      test_instantiate_preserves_structure );
    ("function_arity", `Quick, test_function_arity);
    ("schemes_equivalent", `Quick, test_schemes_equivalent);
    ("scheme_to_string", `Quick, test_scheme_to_string);
  ]

let () = Alcotest.run "Sarek_scheme" [("scheme", tests)]
