(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Unit tests for Typed_value
 *
 * Tests scalar types, registry, and conversion functions.
 ******************************************************************************)

open Spoc_framework.Typed_value

(** {1 Primitive Tests} *)

let test_primitive_int32 () =
  let p = PInt32 42l in
  (match p with PInt32 n -> assert (n = 42l) | _ -> assert false) ;
  print_endline "  PInt32: OK"

let test_primitive_int64 () =
  let p = PInt64 1234567890123456789L in
  (match p with
  | PInt64 n -> assert (n = 1234567890123456789L)
  | _ -> assert false) ;
  print_endline "  PInt64: OK"

let test_primitive_float () =
  let p = PFloat 3.14159265359 in
  (match p with
  | PFloat f -> assert (abs_float (f -. 3.14159265359) < 1e-10)
  | _ -> assert false) ;
  print_endline "  PFloat: OK"

let test_primitive_bool () =
  let pt = PBool true in
  let pf = PBool false in
  (match pt with PBool b -> assert b | _ -> assert false) ;
  (match pf with PBool b -> assert (not b) | _ -> assert false) ;
  print_endline "  PBool: OK"

let test_primitive_bytes () =
  let data = Bytes.of_string "test data" in
  let p = PBytes data in
  (match p with PBytes b -> assert (Bytes.equal b data) | _ -> assert false) ;
  print_endline "  PBytes: OK"

(** {1 Built-in Scalar Type Tests} *)

let test_int32_type () =
  assert (Int32_type.name = "int32") ;
  assert (Int32_type.size = 4) ;
  let v = 123l in
  let p = Int32_type.to_primitive v in
  let v' = Int32_type.of_primitive p in
  assert (v = v') ;
  print_endline "  Int32_type: OK"

let test_int64_type () =
  assert (Int64_type.name = "int64") ;
  assert (Int64_type.size = 8) ;
  let v = 9876543210L in
  let p = Int64_type.to_primitive v in
  let v' = Int64_type.of_primitive p in
  assert (v = v') ;
  print_endline "  Int64_type: OK"

let test_float32_type () =
  assert (Float32_type.name = "float32") ;
  assert (Float32_type.size = 4) ;
  let v = 2.71828 in
  let p = Float32_type.to_primitive v in
  let v' = Float32_type.of_primitive p in
  assert (abs_float (v -. v') < 1e-5) ;
  print_endline "  Float32_type: OK"

let test_float64_type () =
  assert (Float64_type.name = "float64") ;
  assert (Float64_type.size = 8) ;
  let v = 3.141592653589793 in
  let p = Float64_type.to_primitive v in
  let v' = Float64_type.of_primitive p in
  assert (v = v') ;
  print_endline "  Float64_type: OK"

let test_bool_type () =
  assert (Bool_type.name = "bool") ;
  assert (Bool_type.size = 1) ;
  let vt = true in
  let pt = Bool_type.to_primitive vt in
  let vt' = Bool_type.of_primitive pt in
  assert (vt = vt') ;
  let vf = false in
  let pf = Bool_type.to_primitive vf in
  let vf' = Bool_type.of_primitive pf in
  assert (vf = vf') ;
  print_endline "  Bool_type: OK"

(** {1 Registry Tests} *)

let test_registry_find_scalar () =
  (* Built-in types should be registered *)
  assert (Option.is_some (Registry.find_scalar "int32")) ;
  assert (Option.is_some (Registry.find_scalar "int64")) ;
  assert (Option.is_some (Registry.find_scalar "float32")) ;
  assert (Option.is_some (Registry.find_scalar "float64")) ;
  assert (Option.is_some (Registry.find_scalar "bool")) ;
  (* Non-existent type *)
  assert (Option.is_none (Registry.find_scalar "complex128")) ;
  print_endline "  Registry.find_scalar: OK"

let test_registry_list_scalars () =
  let scalars = Registry.list_scalars () in
  assert (List.mem "int32" scalars) ;
  assert (List.mem "int64" scalars) ;
  assert (List.mem "float32" scalars) ;
  assert (List.mem "float64" scalars) ;
  assert (List.mem "bool" scalars) ;
  (* List should be sorted *)
  let sorted = List.sort String.compare scalars in
  assert (scalars = sorted) ;
  print_endline "  Registry.list_scalars: OK"

let test_registry_custom_scalar () =
  (* Create a custom scalar type *)
  let module Custom : SCALAR_TYPE with type t = int = struct
    type t = int

    let name = "test_custom_int"

    let size = 4

    let to_primitive v = PInt32 (Int32.of_int v)

    let of_primitive = function
      | PInt32 n -> Int32.to_int n
      | _ -> failwith "expected PInt32"
  end in
  Registry.register_scalar (module Custom) ;
  assert (Option.is_some (Registry.find_scalar "test_custom_int")) ;
  print_endline "  Registry.register_scalar (custom): OK"

(** {1 Scalar Value Tests} *)

let test_scalar_value () =
  let sv = SV ((module Int32_type), 42l) in
  let p = primitive_of_scalar sv in
  (match p with PInt32 n -> assert (n = 42l) | _ -> assert false) ;
  print_endline "  scalar_value: OK"

(** {1 Typed Value Tests} *)

let test_typed_value_scalar () =
  let tv = TV_Scalar (SV ((module Float64_type), 1.23456789)) in
  (match tv with
  | TV_Scalar (SV ((module S), v)) -> (
      assert (S.name = "float64") ;
      let p = S.to_primitive v in
      match p with
      | PFloat f -> assert (abs_float (f -. 1.23456789) < 1e-8)
      | _ -> assert false)
  | _ -> assert false) ;
  print_endline "  typed_value (scalar): OK"

(** {1 exec_arg Conversion Tests} *)

let test_exec_arg_of_typed_value () =
  let tv = TV_Scalar (SV ((module Int32_type), 100l)) in
  let ea = exec_arg_of_typed_value tv in
  (match ea with
  | EA_Scalar ((module S), v) -> (
      assert (S.name = "int32") ;
      let p = S.to_primitive v in
      match p with PInt32 n -> assert (n = 100l) | _ -> assert false)
  | _ -> assert false) ;
  print_endline "  exec_arg_of_typed_value: OK"

let test_typed_value_of_exec_arg () =
  let ea = EA_Int32 999l in
  let tv = typed_value_of_exec_arg ea in
  (match tv with
  | TV_Scalar (SV ((module S), v)) -> (
      assert (S.name = "int32") ;
      let p = S.to_primitive v in
      match p with PInt32 n -> assert (n = 999l) | _ -> assert false)
  | _ -> assert false) ;
  print_endline "  typed_value_of_exec_arg (Int32): OK"

let test_typed_value_of_exec_arg_float () =
  let ea = EA_Float64 2.718281828 in
  let tv = typed_value_of_exec_arg ea in
  (match tv with
  | TV_Scalar (SV ((module S), v)) -> (
      assert (S.name = "float64") ;
      let p = S.to_primitive v in
      match p with
      | PFloat f -> assert (abs_float (f -. 2.718281828) < 1e-9)
      | _ -> assert false)
  | _ -> assert false) ;
  print_endline "  typed_value_of_exec_arg (Float64): OK"

(** {1 type_name_of_exec_arg Tests} *)

let test_type_name_of_exec_arg () =
  assert (type_name_of_exec_arg (EA_Int32 0l) = "int32") ;
  assert (type_name_of_exec_arg (EA_Int64 0L) = "int64") ;
  assert (type_name_of_exec_arg (EA_Float32 0.0) = "float32") ;
  assert (type_name_of_exec_arg (EA_Float64 0.0) = "float64") ;
  assert (type_name_of_exec_arg (EA_Scalar ((module Bool_type), true)) = "bool") ;
  print_endline "  type_name_of_exec_arg: OK"

(** {1 field_desc Tests} *)

let test_field_desc () =
  let fd : field_desc =
    {fd_name = "x"; fd_type = "float32"; fd_offset = 0; fd_size = 4}
  in
  assert (fd.fd_name = "x") ;
  assert (fd.fd_type = "float32") ;
  assert (fd.fd_offset = 0) ;
  assert (fd.fd_size = 4) ;
  print_endline "  field_desc: OK"

(** {1 Main} *)

let () =
  print_endline "Typed_value tests:" ;
  test_primitive_int32 () ;
  test_primitive_int64 () ;
  test_primitive_float () ;
  test_primitive_bool () ;
  test_primitive_bytes () ;
  test_int32_type () ;
  test_int64_type () ;
  test_float32_type () ;
  test_float64_type () ;
  test_bool_type () ;
  test_registry_find_scalar () ;
  test_registry_list_scalars () ;
  test_registry_custom_scalar () ;
  test_scalar_value () ;
  test_typed_value_scalar () ;
  test_exec_arg_of_typed_value () ;
  test_typed_value_of_exec_arg () ;
  test_typed_value_of_exec_arg_float () ;
  test_type_name_of_exec_arg () ;
  test_field_desc () ;
  print_endline "All Typed_value tests passed!"
