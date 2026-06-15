(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Unit tests for Kirc_kernel - kernel creation and management *)

open Sarek.Kirc_kernel
open Spoc_framework
open Alcotest

(** {1 Tests for native kernel construction} *)

let test_make_native () =
  (* Create a native-only kernel *)
  let native = fun ~block:_ ~grid:_ _args -> () in
  let k =
    make_native ~name:"native_kernel" ~native_fn:native ~param_types:[] ()
  in
  check string "kernel name" "native_kernel" (name k) ;
  check bool "has native fn" true (has_native k)

let test_native_with_params () =
  let native = fun ~block:_ ~grid:_ _args -> () in
  let types = [Sarek.Sarek_ir.TInt32; Sarek.Sarek_ir.TFloat32] in
  let k = make_native ~name:"typed" ~native_fn:native ~param_types:types () in
  check int "param count" 2 (List.length (param_types k))

(** {1 Tests for extension checking} *)

let test_no_extensions_native () =
  let native = fun ~block:_ ~grid:_ _args -> () in
  let k = make_native ~name:"basic" ~native_fn:native ~param_types:[] () in
  check bool "no fp64" false (requires_fp64 k) ;
  check bool "no fp32" false (requires_fp32 k)

(** {1 Tests for native function invocation} *)

let test_native_function_call () =
  let called = ref false in
  let native = fun ~block:_ ~grid:_ _args -> called := true in
  let k = make_native ~name:"test" ~native_fn:native ~param_types:[] () in
  let fn = native_fn k in
  fn ~block:(Framework_sig.dims_1d 1) ~grid:(Framework_sig.dims_1d 1) [||] ;
  check bool "native fn called" true !called

let test_native_function_args () =
  let arg_count = ref 0 in
  let native = fun ~block:_ ~grid:_ args -> arg_count := Array.length args in
  let k = make_native ~name:"test" ~native_fn:native ~param_types:[] () in
  let fn = native_fn k in
  let dummy_args = [|Framework_sig.EA_Int32 42l; Framework_sig.EA_Int32 99l|] in
  fn ~block:(Framework_sig.dims_1d 1) ~grid:(Framework_sig.dims_1d 1) dummy_args ;
  check int "args passed" 2 !arg_count

(** {1 Tests for exec_arg conversion} *)

let int64_exec_vector value =
  let stored = ref value in
  let module V :
    Typed_value.EXEC_VECTOR with type elt = int64 and type underlying = unit =
  struct
    type elt = int64

    type underlying = unit

    let length = 1

    let type_name = "int64"

    let elem_size = 8

    let get _ =
      Typed_value.TV_Scalar
        (Typed_value.SV ((module Typed_value.Int64_type), !stored))

    let set _ = function
      | Typed_value.TV_Scalar (Typed_value.SV ((module S), x)) -> (
          match S.to_primitive x with
          | Typed_value.PInt64 n -> stored := n
          | _ -> failwith "expected int64")
      | _ -> failwith "expected scalar"

    let get_typed _ = !stored

    let set_typed _ x = stored := x

    let type_id = Sarek_ir_types.Type_id.create ()

    let underlying_type_id = Sarek_ir_types.Type_id.create ()

    let underlying = ()

    let device_ptr () = Nativeint.zero
  end in
  Framework_sig.EA_Vec (module V)

let float32_exec_vector value =
  let stored = ref value in
  let module V :
    Typed_value.EXEC_VECTOR with type elt = float and type underlying = unit =
  struct
    type elt = float

    type underlying = unit

    let length = 1

    let type_name = "float32"

    let elem_size = 4

    let get _ =
      Typed_value.TV_Scalar
        (Typed_value.SV ((module Typed_value.Float32_type), !stored))

    let set _ = function
      | Typed_value.TV_Scalar (Typed_value.SV ((module S), x)) -> (
          match S.to_primitive x with
          | Typed_value.PFloat f -> stored := f
          | _ -> failwith "expected float")
      | _ -> failwith "expected scalar"

    let get_typed _ = !stored

    let set_typed _ x = stored := x

    let type_id = Sarek_ir_types.Type_id.create ()

    let underlying_type_id = Sarek_ir_types.Type_id.create ()

    let underlying = ()

    let device_ptr () = Nativeint.zero
  end in
  Framework_sig.EA_Vec (module V)

let test_exec_vector_numeric_conversions () =
  (match exec_arg_to_native_arg (int64_exec_vector 42L) with
  | Sarek_ir_types.NA_Vec (Sarek_ir_types.NV v) ->
      check (float 0.001) "int64 to f32" 42.0 (v.get_f32 0) ;
      check (float 0.001) "int64 to f64" 42.0 (v.get_f64 0) ;
      check int32 "int64 to i32" 42l (v.get_i32 0) ;
      check int64 "int64 to i64" 42L (v.get_i64 0)
  | _ -> fail "expected native vector") ;
  match exec_arg_to_native_arg (float32_exec_vector 42.75) with
  | Sarek_ir_types.NA_Vec (Sarek_ir_types.NV v) ->
      check int32 "float to i32" 42l (v.get_i32 0) ;
      check int64 "float to i64" 42L (v.get_i64 0) ;
      check (float 0.001) "float to f64" 42.75 (v.get_f64 0)
  | _ -> fail "expected native vector"

(** {1 Test suite} *)

let () =
  run
    "Kirc_kernel"
    [
      ( "native_construction",
        [
          test_case "make_native" `Quick test_make_native;
          test_case "native_with_params" `Quick test_native_with_params;
        ] );
      ( "extension_checking",
        [test_case "no_extensions" `Quick test_no_extensions_native] );
      ( "native_invocation",
        [
          test_case "call_native" `Quick test_native_function_call;
          test_case "pass_args" `Quick test_native_function_args;
        ] );
      ( "exec_arg_conversion",
        [
          test_case
            "numeric vector conversions"
            `Quick
            test_exec_vector_numeric_conversions;
        ] );
    ]
