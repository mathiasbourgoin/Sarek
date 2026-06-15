(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Unit tests for Sarek_ir_interp - IR-based interpreter *)

open Sarek.Sarek_ir_interp
open Sarek_ir_types
open Alcotest

(** Helper to create a test environment *)
let make_env () =
  {
    vars = Hashtbl.create 16;
    vars_by_name = Hashtbl.create 16;
    arrays = Hashtbl.create 8;
    shared = Hashtbl.create 8;
    funcs = Hashtbl.create 8;
  }

(** Helper to create a test thread state *)
let make_state ?(thread_idx = (0, 0, 0)) ?(block_idx = (0, 0, 0))
    ?(block_dim = (1, 1, 1)) ?(grid_dim = (1, 1, 1)) () =
  {thread_idx; block_idx; block_dim; grid_dim}

(** {1 Tests for eval_array_expr} *)

let test_array_read () =
  let env = make_env () in
  let state = make_state () in
  (* Create array: [10, 20, 30] *)
  let arr = Array.map (fun n -> VInt32 (Int32.of_int n)) [|10; 20; 30|] in
  Hashtbl.add env.arrays "data" arr ;
  (* Test: data[1] should return 20 *)
  let expr = EArrayRead ("data", EConst (CInt32 1l)) in
  let result = eval_expr state env expr in
  check
    (module struct
      type t = value

      let pp fmt = function
        | VInt32 n -> Format.fprintf fmt "VInt32 %ld" n
        | _ -> Format.fprintf fmt "<other>"

      let equal a b =
        match (a, b) with VInt32 x, VInt32 y -> x = y | _ -> false
    end)
    "array read"
    (VInt32 20l)
    result

let test_array_bounds () =
  let env = make_env () in
  let state = make_state () in
  let arr = Array.make 5 (VInt32 0l) in
  Hashtbl.add env.arrays "data" arr ;
  (* Test: data[10] should raise an error *)
  let expr = EArrayRead ("data", EConst (CInt32 10l)) in
  let raised_error = ref false in
  (try ignore (eval_expr state env expr) with _ -> raised_error := true) ;
  check bool "array bounds raises error" true !raised_error

let test_array_create () =
  let env = make_env () in
  let state = make_state () in
  (* Create array of size 5 *)
  let expr = EArrayCreate (TInt32, EConst (CInt32 5l), Global) in
  match eval_expr state env expr with
  | VArray arr ->
      check int "array length" 5 (Array.length arr) ;
      check
        (module struct
          type t = value

          let pp fmt = function
            | VInt32 n -> Format.fprintf fmt "VInt32 %ld" n
            | _ -> Format.fprintf fmt "<other>"

          let equal a b =
            match (a, b) with VInt32 x, VInt32 y -> x = y | _ -> false
        end)
        "array init"
        (VInt32 0l)
        arr.(0)
  | _ -> fail "expected VArray"

let test_array_len () =
  let env = make_env () in
  let state = make_state () in
  let arr = Array.make 7 (VInt32 0l) in
  Hashtbl.add env.arrays "data" arr ;
  (* Test: length(data) should return 7 *)
  let expr = EArrayLen "data" in
  match eval_expr state env expr with
  | VInt32 n -> check int32 "array length" 7l n
  | _ -> fail "expected VInt32"

(** {1 Tests for eval_composite_expr} *)

let test_record_creation () =
  let env = make_env () in
  let state = make_state () in
  (* Create record: {x = 10; y = 20} *)
  let expr =
    ERecord ("point", [("x", EConst (CInt32 10l)); ("y", EConst (CInt32 20l))])
  in
  match eval_expr state env expr with
  | VRecord (name, fields) ->
      check string "record type" "point" name ;
      check int "record fields" 2 (Array.length fields) ;
      check
        (module struct
          type t = value

          let pp fmt = function
            | VInt32 n -> Format.fprintf fmt "VInt32 %ld" n
            | _ -> Format.fprintf fmt "<other>"

          let equal a b =
            match (a, b) with VInt32 x, VInt32 y -> x = y | _ -> false
        end)
        "first field"
        (VInt32 10l)
        fields.(0)
  | _ -> fail "expected VRecord"

let test_variant_creation () =
  let env = make_env () in
  let state = make_state () in
  (* Create variant: Some(42) *)
  let expr = EVariant ("option", "Some", [EConst (CInt32 42l)]) in
  match eval_expr state env expr with
  | VVariant (ty, _tag, args) ->
      check string "variant type" "option" ty ;
      check int "variant args" 1 (List.length args) ;
      check
        (module struct
          type t = value

          let pp fmt = function
            | VInt32 n -> Format.fprintf fmt "VInt32 %ld" n
            | _ -> Format.fprintf fmt "<other>"

          let equal a b =
            match (a, b) with VInt32 x, VInt32 y -> x = y | _ -> false
        end)
        "variant arg"
        (VInt32 42l)
        (List.hd args)
  | _ -> fail "expected VVariant"

(** {1 Tests for eval_control_flow} *)

let test_if_then_else () =
  let env = make_env () in
  let state = make_state () in
  (* if true then 10 else 20 *)
  let expr =
    EIf (EConst (CBool true), EConst (CInt32 10l), EConst (CInt32 20l))
  in
  match eval_expr state env expr with
  | VInt32 n -> check int32 "if true" 10l n
  | _ -> fail "expected VInt32"

let test_if_false () =
  let env = make_env () in
  let state = make_state () in
  (* if false then 10 else 20 *)
  let expr =
    EIf (EConst (CBool false), EConst (CInt32 10l), EConst (CInt32 20l))
  in
  match eval_expr state env expr with
  | VInt32 n -> check int32 "if false" 20l n
  | _ -> fail "expected VInt32"

(** {1 Tests for eval_special_expr} *)

let test_cast_int32_to_float () =
  let env = make_env () in
  let state = make_state () in
  (* cast 42 to float32 *)
  let expr = ECast (TFloat32, EConst (CInt32 42l)) in
  match eval_expr state env expr with
  | VFloat32 f -> check (float 0.001) "cast to float" 42.0 f
  | _ -> fail "expected VFloat32"

let test_cast_float_to_int () =
  let env = make_env () in
  let state = make_state () in
  (* cast 42.7 to int32 *)
  let expr = ECast (TInt32, EConst (CFloat32 42.7)) in
  match eval_expr state env expr with
  | VInt32 n -> check int32 "cast to int" 42l n
  | _ -> fail "expected VInt32"

(** {1 Tests for binary operations} *)

let test_binop_add () =
  let env = make_env () in
  let state = make_state () in
  (* 10 + 20 = 30 *)
  let expr = EBinop (Add, EConst (CInt32 10l), EConst (CInt32 20l)) in
  match eval_expr state env expr with
  | VInt32 n -> check int32 "add" 30l n
  | _ -> fail "expected VInt32"

let test_binop_mul () =
  let env = make_env () in
  let state = make_state () in
  (* 5 * 7 = 35 *)
  let expr = EBinop (Mul, EConst (CInt32 5l), EConst (CInt32 7l)) in
  match eval_expr state env expr with
  | VInt32 n -> check int32 "multiply" 35l n
  | _ -> fail "expected VInt32"

let test_binop_lt () =
  let env = make_env () in
  let state = make_state () in
  (* 10 < 20 = true *)
  let expr = EBinop (Lt, EConst (CInt32 10l), EConst (CInt32 20l)) in
  match eval_expr state env expr with
  | VBool b -> check bool "less than true" true b
  | _ -> fail "expected VBool"

let test_binop_eq () =
  let env = make_env () in
  let state = make_state () in
  (* 42 = 42 = true *)
  let expr = EBinop (Eq, EConst (CInt32 42l), EConst (CInt32 42l)) in
  match eval_expr state env expr with
  | VBool b -> check bool "equals" true b
  | _ -> fail "expected VBool"

(** {1 Tests for unary operations} *)

let test_unop_neg () =
  let env = make_env () in
  let state = make_state () in
  (* -42 *)
  let expr = EUnop (Neg, EConst (CInt32 42l)) in
  match eval_expr state env expr with
  | VInt32 n -> check int32 "negate" (-42l) n
  | _ -> fail "expected VInt32"

let test_unop_not () =
  let env = make_env () in
  let state = make_state () in
  (* not true = false *)
  let expr = EUnop (Not, EConst (CBool true)) in
  match eval_expr state env expr with
  | VBool b -> check bool "not" false b
  | _ -> fail "expected VBool"

(** {1 Tests for statement execution} *)

let test_stmt_assign () =
  let env = make_env () in
  let state = make_state () in
  let arr = Array.make 5 (VInt32 0l) in
  Hashtbl.add env.arrays "output" arr ;
  (* output[2] <- 99 *)
  let stmt =
    SAssign (LArrayElem ("output", EConst (CInt32 2l)), EConst (CInt32 99l))
  in
  exec_stmt state env stmt ;
  match arr.(2) with
  | VInt32 n -> check int32 "assign" 99l n
  | _ -> fail "expected VInt32"

let test_stmt_let () =
  let env = make_env () in
  let state = make_state () in
  (* let x = 42 in () *)
  let var =
    {var_name = "x"; var_id = 1; var_type = TInt32; var_mutable = false}
  in
  let stmt = SLet (var, EConst (CInt32 42l), SEmpty) in
  exec_stmt state env stmt ;
  match lookup_var env var with
  | VInt32 n -> check int32 "let binding" 42l n
  | _ -> fail "expected VInt32"

let test_stmt_if () =
  let env = make_env () in
  let state = make_state () in
  Hashtbl.add env.vars_by_name "counter" (VInt32 0l) ;
  (* if true then () else () *)
  let stmt = SIf (EConst (CBool true), SEmpty, Some SEmpty) in
  exec_stmt state env stmt ;
  check bool "if executed" true true

let test_stmt_while () =
  let env = make_env () in
  let state = make_state () in
  let counter_var =
    {var_name = "i"; var_id = 1; var_type = TInt32; var_mutable = true}
  in
  bind_var env counter_var (VInt32 0l) ;
  (* while i < 3 do i <- i + 1 *)
  let cond = EBinop (Lt, EVar counter_var, EConst (CInt32 3l)) in
  let body =
    SAssign
      (LVar counter_var, EBinop (Add, EVar counter_var, EConst (CInt32 1l)))
  in
  let stmt = SWhile (cond, body) in
  exec_stmt state env stmt ;
  match lookup_var env counter_var with
  | VInt32 n -> check int32 "while loop" 3l n
  | _ -> fail "expected VInt32"

let test_stmt_for () =
  let env = make_env () in
  let state = make_state () in
  let i_var =
    {var_name = "i"; var_id = 1; var_type = TInt32; var_mutable = true}
  in
  let sum_var =
    {var_name = "sum"; var_id = 2; var_type = TInt32; var_mutable = true}
  in
  bind_var env sum_var (VInt32 0l) ;
  (* for i = 0 to 4 do sum <- sum + i *)
  let body = SAssign (LVar sum_var, EBinop (Add, EVar sum_var, EVar i_var)) in
  let stmt = SFor (i_var, EConst (CInt32 0l), EConst (CInt32 4l), Upto, body) in
  exec_stmt state env stmt ;
  match lookup_var env sum_var with
  | VInt32 n -> check int32 "for loop sum" 10l n (* 0+1+2+3+4 = 10 *)
  | _ -> fail "expected VInt32"

(** {1 Tests for value conversions} *)

let test_to_int32 () =
  check int32 "int32" 42l (to_int32 (VInt32 42l)) ;
  check int32 "int64" 100l (to_int32 (VInt64 100L)) ;
  check int32 "float32" 42l (to_int32 (VFloat32 42.9)) ;
  check int32 "bool true" 1l (to_int32 (VBool true)) ;
  check int32 "bool false" 0l (to_int32 (VBool false))

let test_to_bool () =
  check bool "true" true (to_bool (VBool true)) ;
  check bool "false" false (to_bool (VBool false)) ;
  check bool "int32 nonzero" true (to_bool (VInt32 42l)) ;
  check bool "int32 zero" false (to_bool (VInt32 0l))

(** {1 Test suite} *)

let () =
  run
    "Sarek_ir_interp"
    [
      ( "array_operations",
        [
          test_case "array_read" `Quick test_array_read;
          test_case "array_bounds" `Quick test_array_bounds;
          test_case "array_create" `Quick test_array_create;
          test_case "array_len" `Quick test_array_len;
        ] );
      ( "composite_types",
        [
          test_case "record_creation" `Quick test_record_creation;
          test_case "variant_creation" `Quick test_variant_creation;
        ] );
      ( "control_flow",
        [
          test_case "if_then_else" `Quick test_if_then_else;
          test_case "if_false" `Quick test_if_false;
        ] );
      ( "special_operations",
        [
          test_case "cast_int_to_float" `Quick test_cast_int32_to_float;
          test_case "cast_float_to_int" `Quick test_cast_float_to_int;
        ] );
      ( "binary_operations",
        [
          test_case "add" `Quick test_binop_add;
          test_case "multiply" `Quick test_binop_mul;
          test_case "less_than" `Quick test_binop_lt;
          test_case "equals" `Quick test_binop_eq;
        ] );
      ( "unary_operations",
        [
          test_case "negate" `Quick test_unop_neg;
          test_case "not" `Quick test_unop_not;
        ] );
      ( "statements",
        [
          test_case "assign" `Quick test_stmt_assign;
          test_case "let_binding" `Quick test_stmt_let;
          test_case "if_stmt" `Quick test_stmt_if;
          test_case "while_loop" `Quick test_stmt_while;
          test_case "for_loop" `Quick test_stmt_for;
        ] );
      ( "value_conversions",
        [
          test_case "to_int32" `Quick test_to_int32;
          test_case "to_bool" `Quick test_to_bool;
        ] );
    ]
