(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Unit tests for Sarek_pure_registry
 *
 * These tests snapshot the exact name set registered under each of the 8
 * module paths, so that a refactor of the registration tables (deduplicating
 * the shared static lists) cannot silently change which intrinsics resolve
 * under a given path. See briefs/backend-dry-correctness-step0.md (d)/(e)
 * for the full audit this snapshot is derived from.
 ******************************************************************************)

open Sarek_pure_registry

let float32_names =
  [
    "sin";
    "cos";
    "tan";
    "asin";
    "acos";
    "atan";
    "sinh";
    "cosh";
    "tanh";
    "exp";
    "exp2";
    "log";
    "log2";
    "log10";
    "sqrt";
    "rsqrt";
    "cbrt";
    "floor";
    "ceil";
    "round";
    "trunc";
    "fabs";
    "abs_float";
    "pow";
    "atan2";
    "fma";
    "min";
    "max";
    "expm1";
    "log1p";
    "hypot";
    "copysign";
  ]

let float64_names =
  [
    "sin";
    "cos";
    "tan";
    "asin";
    "acos";
    "atan";
    "sinh";
    "cosh";
    "tanh";
    "exp";
    "exp2";
    "log";
    "log2";
    "log10";
    "sqrt";
    "rsqrt";
    "cbrt";
    "floor";
    "ceil";
    "round";
    "trunc";
    "fabs";
    "pow";
    "atan2";
    "fma";
    "min";
    "max";
  ]

let math_float64_names =
  [
    "sin";
    "cos";
    "tan";
    "asin";
    "acos";
    "atan";
    "sinh";
    "cosh";
    "tanh";
    "exp";
    "log";
    "sqrt";
    "floor";
    "ceil";
    "pow";
    "atan2";
  ]

(* Sarek_pure_registry has no .mli, so [fun_registry] (the backing Hashtbl) is
   directly accessible here - fold over it to get the exact set of names
   registered under a given module_path, rather than only checking presence
   of the expected names one at a time. This proves absence of accidental
   extra registrations too, closing the gap the old presence-only check left
   open (an accidental extra entry under Float32/Float64 would previously
   have passed silently). *)
let names_for_path module_path =
  Hashtbl.fold
    (fun (path, name) _ acc -> if path = module_path then name :: acc else acc)
    fun_registry
    []
  |> List.sort_uniq String.compare

let assert_path_has_all module_path expected label =
  let expected_sorted = List.sort_uniq String.compare expected in
  let actual_sorted = names_for_path module_path in
  if actual_sorted <> expected_sorted then
    failwith
      (Printf.sprintf
         "%s: expected exactly [%s], got [%s]"
         label
         (String.concat ", " expected_sorted)
         (String.concat ", " actual_sorted))

let test_float32_paths () =
  assert_path_has_all ["Float32"] float32_names "Float32" ;
  assert_path_has_all ["Math"; "Float32"] float32_names "Math.Float32" ;
  assert_path_has_all
    ["Sarek_stdlib_meta"; "Float32"]
    float32_names
    "Sarek_stdlib_meta.Float32" ;
  assert_path_has_all
    ["Sarek_stdlib_meta"; "Math"; "Float32"]
    float32_names
    "Sarek_stdlib_meta.Math.Float32" ;
  print_endline "  float32 paths expose exactly the 32-entry set: OK"

let test_float64_paths () =
  assert_path_has_all ["Float64"] float64_names "Float64" ;
  assert_path_has_all
    ["Sarek_stdlib_meta"; "Float64"]
    float64_names
    "Sarek_stdlib_meta.Float64" ;
  print_endline "  Float64 paths expose exactly the 27-entry set: OK"

let test_math_float64_paths () =
  assert_path_has_all ["Math"; "Float64"] math_float64_names "Math.Float64" ;
  assert_path_has_all
    ["Sarek_stdlib_meta"; "Math"; "Float64"]
    math_float64_names
    "Sarek_stdlib_meta.Math.Float64" ;
  print_endline "  Math.Float64 paths expose exactly the 16-entry set: OK"

(** The 11 intrinsics present in the full Float64 tables but intentionally
    absent from the Math.Float64 tables (see step-0 report (e): interpreter
    and/or stdlib support is missing for all 11 — registering them here would
    convert a lookup failure into a miscompile). This test guards against
    accidentally "completing" the Math.Float64 tables without also doing the
    interpreter/stdlib work. *)
let missing_math_float64_names =
  [
    "exp2";
    "log2";
    "log10";
    "rsqrt";
    "cbrt";
    "round";
    "trunc";
    "fabs";
    "fma";
    "min";
    "max";
  ]

let test_math_float64_intentionally_missing () =
  List.iter
    (fun name ->
      (match fun_device_template ~module_path:["Math"; "Float64"] name with
      | None -> ()
      | Some _ ->
          failwith
            (Printf.sprintf
               "Math.Float64: %s is now registered - if this is intentional, \
                update the step-0 tracked-follow-up comment in \
                Sarek_pure_registry.ml and this test"
               name)) ;
      match
        fun_device_template
          ~module_path:["Sarek_stdlib_meta"; "Math"; "Float64"]
          name
      with
      | None -> ()
      | Some _ ->
          failwith
            (Printf.sprintf
               "Sarek_stdlib_meta.Math.Float64: %s is now registered - if this \
                is intentional, update the step-0 tracked-follow-up comment in \
                Sarek_pure_registry.ml and this test"
               name))
    missing_math_float64_names ;
  print_endline
    "  Math.Float64 paths still omit the 11 unsupported intrinsics: OK"

(** Step-0 bonus bug: the plain ["Float64"] table used a template that ignored
    [~framework], so [Float64.rsqrt] emitted the bare name "rsqrt" on GLSL
    (invalid - GLSL has no [rsqrt] builtin) instead of "inversesqrt". The
    [Sarek_stdlib_meta.Float64] twin already used the framework-aware
    [generic_math_template] and emitted the correct GLSL name. This test pins
    both paths to the same, correct GLSL name. *)
let test_float64_rsqrt_glsl_matches_meta_twin () =
  let plain =
    match fun_device_template ~module_path:["Float64"] "rsqrt" with
    | Some device -> device ~framework:"GLSL"
    | None -> failwith "Float64.rsqrt not registered"
  in
  let meta =
    match
      fun_device_template ~module_path:["Sarek_stdlib_meta"; "Float64"] "rsqrt"
    with
    | Some device -> device ~framework:"GLSL"
    | None -> failwith "Sarek_stdlib_meta.Float64.rsqrt not registered"
  in
  assert (plain = meta) ;
  assert (plain = "inversesqrt") ;
  print_endline
    "  Float64.rsqrt and Sarek_stdlib_meta.Float64.rsqrt agree on GLSL name \
     (inversesqrt): OK"

let () =
  test_float32_paths () ;
  test_float64_paths () ;
  test_math_float64_paths () ;
  test_math_float64_intentionally_missing () ;
  test_float64_rsqrt_glsl_matches_meta_twin () ;
  print_endline "All Sarek_pure_registry tests passed!"
