(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * The Int32/Int64 stdlib arithmetic templates (Sarek_ir_c_arith), split from
 * test_int32_bitcasts_codegen.
 *
 * 1. add, sub, mul, neg and shift_left: both registrations (Sarek_stdlib and
 *    Sarek_stdlib_meta) agree, compute through the unsigned type and come back
 *    through the dialect's conversion.
 * 2. abs on OpenCL C: [as_int(abs((int)(x)))] / [as_long(abs((long)(x)))],
 *    and clang -x cl accepts it on an 8-bit element read, the argument
 *    [as_int] alone would reject ("sizes ... must match").
 ******************************************************************************)

open Sarek_ir_types
open Sarek_codegen

let () = Sarek_stdlib.force_init ()

let () = Sarek_stdlib_meta.force_init ()

let contains s sub =
  let n = String.length s and m = String.length sub in
  let rec go i = i + m <= n && (String.sub s i m = sub || go (i + 1)) in
  go 0

let wrapping_intrinsics =
  [
    ("Int32", ["add_int32"; "sub_int32"; "mul_int32"; "neg"; "shift_left"]);
    ("Int64", ["add_int64"; "sub_int64"; "mul_int64"; "neg"; "shift_left"]);
  ]

(* OpenCL C's [abs] returns an unsigned type: the template reinterprets it, so
   [Int32.abs x - 1] stays signed arithmetic. *)
let check_abs_templates () =
  List.iter
    (fun (m, expected) ->
      List.iter
        (fun lib ->
          Alcotest.(check (option string))
            (Printf.sprintf "%s.%s.abs on OpenCL" lib m)
            (Some expected)
            (Sarek_registry.fun_device_template
               ~module_path:[lib; m]
               ~framework:"OpenCL"
               "abs"))
        ["Sarek_stdlib"; "Sarek_stdlib_meta"])
    [("Int32", "as_int(abs((int)(%s)))"); ("Int64", "as_long(abs((long)(%s)))")]

let check_wrapping_parity () =
  List.iter
    (fun (m, names) ->
      List.iter
        (fun name ->
          List.iter
            (fun framework ->
              let template lib =
                Sarek_registry.fun_device_template
                  ~module_path:[lib; m]
                  ~framework
                  name
              in
              let std = template "Sarek_stdlib" in
              let label = Printf.sprintf "%s.%s on %s" m name framework in
              Alcotest.(check (option string))
                (label ^ ": Sarek_stdlib_meta = Sarek_stdlib")
                std
                (template "Sarek_stdlib_meta") ;
              let unsigned =
                match (m, framework) with
                | "Int32", "CUDA" -> "(unsigned int)"
                | "Int32", _ -> "(uint)"
                | _, "CUDA" -> "(unsigned long long)"
                | _ -> "(ulong)"
              in
              (* the unsigned cast is on the operands of the operator itself
                 (for a shift, its left operand) and the result comes back
                 through the dialect's conversion *)
              let u = unsigned ^ "(%s)" in
              let operation =
                match name with
                | "neg" -> "-" ^ u
                | "shift_left" -> u ^ " << (%s)"
                | _ ->
                    let op = String.sub name 0 3 in
                    let sym =
                      match op with "add" -> "+" | "sub" -> "-" | _ -> "*"
                    in
                    u ^ " " ^ sym ^ " " ^ u
              in
              let back =
                match (m, framework) with
                | "Int32", "CUDA" -> "((int)("
                | "Int32", "OpenCL" -> "as_int(("
                | "Int32", _ -> "as_type<int>(("
                | _, "CUDA" -> "((long long)("
                | _, "OpenCL" -> "as_long(("
                | _ -> "as_type<long>(("
              in
              let t = Option.value std ~default:"" in
              Alcotest.(check bool)
                (label ^ " computes " ^ operation)
                true
                (contains t operation) ;
              Alcotest.(check bool)
                (label ^ " starts with " ^ back)
                true
                (String.starts_with ~prefix:back t))
            ["CUDA"; "OpenCL"; "Metal"])
        names)
    wrapping_intrinsics

let var name ty =
  {var_name = name; var_id = 0; var_type = ty; var_mutable = false}

let vec name elt =
  DParam (var name (TVec elt), Some {arr_elttype = elt; arr_memspace = Global})

(* [out.(0) <- Int32.abs src.(0)] with [src] uint8, and
   [out64.(0) <- Int64.abs (int64 of) vals.(0)] with [vals] int32. *)
let abs_kernel lib : kernel =
  let i0 = EConst (CInt32 0l) in
  {
    default_kernel with
    kern_name = "abs_widths";
    kern_params =
      [
        vec "src" TUint8; vec "vals" TInt32; vec "out" TInt32; vec "out64" TInt64;
      ];
    kern_body =
      SSeq
        [
          SAssign
            ( LArrayElem ("out", i0),
              EIntrinsic ([lib; "Int32"], "abs", [EArrayRead ("src", i0)]) );
          SAssign
            ( LArrayElem ("out64", i0),
              EIntrinsic ([lib; "Int64"], "abs", [EArrayRead ("vals", i0)]) );
        ];
  }

let check_abs_compiles () =
  if not (Opencl_gate.Opencl_clang.available ()) then Alcotest.skip () ;
  List.iter
    (fun lib ->
      let src = Sarek_ir_opencl.generate (abs_kernel lib) in
      Alcotest.(check bool)
        (lib ^ ": widened int32 abs")
        true
        (contains src "as_int(abs((int)(src[0])))") ;
      match Opencl_gate.Opencl_clang.run_clang src with
      | Ok () -> ()
      | Error e -> Alcotest.failf "%s: clang -x cl rejected:\n%s\n%s" lib e src)
    ["Sarek_stdlib"; "Sarek_stdlib_meta"]

let () =
  Alcotest.run
    "int-stdlib-templates"
    [
      ( "registry",
        [
          Alcotest.test_case "wrapping parity" `Quick check_wrapping_parity;
          Alcotest.test_case "abs on OpenCL" `Quick check_abs_templates;
        ] );
      ("clang", [Alcotest.test_case "abs widths" `Quick check_abs_compiles]);
    ]
