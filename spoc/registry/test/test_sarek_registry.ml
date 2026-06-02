(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Unit tests for Sarek_registry
 *
 * Tests type, record, variant, and function registration and lookup.
 ******************************************************************************)

open Sarek_registry

(** {1 Type Registry Tests} *)

let test_builtin_types () =
  (* bool and unit are registered at startup *)
  assert (is_type "bool") ;
  assert (is_type "unit") ;
  print_endline "  builtin types: OK"

let test_register_type () =
  register_type "test_int" ~device:(fun _ -> "int") ~size:4 ;
  assert (is_type "test_int") ;
  let info = find_type "test_int" in
  assert (Option.is_some info) ;
  (match info with
  | Some ti ->
      assert (ti.ti_name = "test_int") ;
      assert (ti.ti_size = 4)
  | None -> assert false) ;
  print_endline "  register_type: OK"

let test_find_type () =
  register_type "test_float" ~device:(fun _ -> "float") ~size:4 ;
  let info = find_type "test_float" in
  assert (Option.is_some info) ;
  let none = find_type "nonexistent_type" in
  assert (Option.is_none none) ;
  print_endline "  find_type: OK"

let test_type_device_code () =
  register_type "test_double" ~device:(fun _ -> "double") ~size:8 ;
  let code = type_device_code "test_double" "CUDA" in
  assert (code = "double") ;
  print_endline "  type_device_code: OK"

(** {1 Record Registry Tests} *)

let test_register_record () =
  let fields =
    [
      {field_name = "x"; field_type = "float32"; field_mutable = false};
      {field_name = "y"; field_type = "float32"; field_mutable = false};
    ]
  in
  register_record "test_point" ~fields ~size:8 ;
  assert (is_record "test_point") ;
  print_endline "  register_record: OK"

let test_find_record () =
  let fields =
    [
      {field_name = "r"; field_type = "float32"; field_mutable = false};
      {field_name = "g"; field_type = "float32"; field_mutable = false};
      {field_name = "b"; field_type = "float32"; field_mutable = false};
    ]
  in
  register_record "test_color" ~fields ~size:12 ;
  let info = find_record "test_color" in
  assert (Option.is_some info) ;
  (match info with
  | Some ri ->
      assert (ri.ri_name = "test_color") ;
      assert (ri.ri_size = 12) ;
      assert (List.length ri.ri_fields = 3)
  | None -> assert false) ;
  let none = find_record "nonexistent_record" in
  assert (Option.is_none none) ;
  print_endline "  find_record: OK"

let test_record_fields () =
  let fields =
    [
      {field_name = "a"; field_type = "int32"; field_mutable = true};
      {field_name = "b"; field_type = "int64"; field_mutable = false};
    ]
  in
  register_record "test_pair" ~fields ~size:12 ;
  let retrieved = record_fields "test_pair" in
  assert (List.length retrieved = 2) ;
  let f1 = List.hd retrieved in
  assert (f1.field_name = "a") ;
  assert (f1.field_type = "int32") ;
  assert (f1.field_mutable = true) ;
  print_endline "  record_fields: OK"

let test_find_record_by_short_name () =
  let fields =
    [{field_name = "value"; field_type = "int32"; field_mutable = false}]
  in
  register_record "Module.SubModule.test_wrapper" ~fields ~size:4 ;
  (* Full name lookup should work *)
  assert (is_record "Module.SubModule.test_wrapper") ;
  (* Short name lookup should also work *)
  let info = find_record_by_short_name "test_wrapper" in
  assert (Option.is_some info) ;
  (match info with
  | Some ri -> assert (ri.ri_name = "Module.SubModule.test_wrapper")
  | None -> assert false) ;
  print_endline "  find_record_by_short_name: OK"

(** {1 Variant Registry Tests} *)

let test_register_variant () =
  let constructors =
    [
      {ctor_name = "Red"; ctor_arg_type = None};
      {ctor_name = "Green"; ctor_arg_type = None};
      {ctor_name = "Blue"; ctor_arg_type = None};
    ]
  in
  register_variant "test_rgb" ~constructors ;
  assert (is_variant "test_rgb") ;
  print_endline "  register_variant: OK"

let test_find_variant () =
  let constructors =
    [
      {ctor_name = "None"; ctor_arg_type = None};
      {ctor_name = "Some"; ctor_arg_type = Some "int32"};
    ]
  in
  register_variant "test_option_int" ~constructors ;
  let info = find_variant "test_option_int" in
  assert (Option.is_some info) ;
  (match info with
  | Some vi ->
      assert (vi.vi_name = "test_option_int") ;
      assert (List.length vi.vi_constructors = 2)
  | None -> assert false) ;
  let none = find_variant "nonexistent_variant" in
  assert (Option.is_none none) ;
  print_endline "  find_variant: OK"

let test_variant_constructors () =
  let constructors =
    [
      {ctor_name = "Int"; ctor_arg_type = Some "int32"};
      {ctor_name = "Float"; ctor_arg_type = Some "float32"};
      {ctor_name = "Bool"; ctor_arg_type = Some "bool"};
    ]
  in
  register_variant "test_value" ~constructors ;
  let retrieved = variant_constructors "test_value" in
  assert (List.length retrieved = 3) ;
  let c1 = List.hd retrieved in
  assert (c1.ctor_name = "Int") ;
  assert (c1.ctor_arg_type = Some "int32") ;
  print_endline "  variant_constructors: OK"

(** {1 Function Registry Tests} *)

let test_register_fun () =
  register_fun
    "test_abs"
    ~arity:1
    ~device:(fun _ -> "abs")
    ~arg_types:["int32"]
    ~ret_type:"int32" ;
  assert (is_fun "test_abs") ;
  print_endline "  register_fun: OK"

let test_register_fun_with_module () =
  register_fun
    ~module_path:["Float32"]
    "test_sin"
    ~arity:1
    ~device:(fun _ -> "sinf")
    ~arg_types:["float32"]
    ~ret_type:"float32" ;
  assert (is_fun ~module_path:["Float32"] "test_sin") ;
  (* Without module path should not find it *)
  assert (not (is_fun "test_sin")) ;
  print_endline "  register_fun with module: OK"

let test_find_fun () =
  register_fun
    "test_max"
    ~arity:2
    ~device:(fun _ -> "max")
    ~arg_types:["int32"; "int32"]
    ~ret_type:"int32" ;
  let info = find_fun "test_max" in
  assert (Option.is_some info) ;
  (match info with
  | Some fi ->
      assert (fi.fi_name = "test_max") ;
      assert (fi.fi_arity = 2) ;
      assert (fi.fi_ret_type = "int32")
  | None -> assert false) ;
  let none = find_fun "nonexistent_fun" in
  assert (Option.is_none none) ;
  print_endline "  find_fun: OK"

let test_fun_device_code () =
  register_fun
    "test_min"
    ~arity:2
    ~device:(fun _ -> "min")
    ~arg_types:["int32"; "int32"]
    ~ret_type:"int32" ;
  let code = fun_device_code "test_min" "CUDA" in
  assert (code = "min") ;
  print_endline "  fun_device_code: OK"

let test_fun_device_template () =
  register_fun
    "test_clamp"
    ~arity:3
    ~device:(fun _ -> "clamp")
    ~arg_types:["float32"; "float32"; "float32"]
    ~ret_type:"float32" ;
  let template = fun_device_template "test_clamp" in
  assert (Option.is_some template) ;
  (match template with Some t -> assert (t = "clamp") | None -> assert false) ;
  let none = fun_device_template "nonexistent_fun" in
  assert (Option.is_none none) ;
  print_endline "  fun_device_template: OK"

(** {1 cuda_or_opencl Helper Tests} *)

let test_cuda_or_opencl () =
  assert (cuda_or_opencl "CUDA" "cuda_code" "opencl_code" = "cuda_code") ;
  assert (cuda_or_opencl "OpenCL" "cuda_code" "opencl_code" = "opencl_code") ;
  assert (cuda_or_opencl "Native" "cuda_code" "opencl_code" = "cuda_code") ;
  print_endline "  cuda_or_opencl: OK"

(** {1 Main} *)

let () =
  print_endline "Sarek_registry tests:" ;
  test_builtin_types () ;
  test_register_type () ;
  test_find_type () ;
  test_type_device_code () ;
  test_register_record () ;
  test_find_record () ;
  test_record_fields () ;
  test_find_record_by_short_name () ;
  test_register_variant () ;
  test_find_variant () ;
  test_variant_constructors () ;
  test_register_fun () ;
  test_register_fun_with_module () ;
  test_find_fun () ;
  test_fun_device_code () ;
  test_fun_device_template () ;
  test_cuda_or_opencl () ;
  print_endline "All Sarek_registry tests passed!"
