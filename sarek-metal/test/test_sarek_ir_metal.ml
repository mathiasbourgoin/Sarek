(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Sarek_ir_metal Tests - Verify Metal Code Generation
 ******************************************************************************)

open Sarek_metal
open Sarek_ir_types

let make_var name ty =
  {var_id = 0; var_name = name; var_type = ty; var_mutable = false}

let test_basic_literals () =
  let buf = Buffer.create 64 in
  Sarek_ir_metal.gen_expr buf (EConst (CInt32 42l)) ;
  Alcotest.(check string) "int32 literal" "42" (Buffer.contents buf) ;

  let buf = Buffer.create 64 in
  Sarek_ir_metal.gen_expr buf (EConst (CInt64 42L)) ;
  Alcotest.(check string) "int64 literal" "42L" (Buffer.contents buf) ;

  let buf = Buffer.create 64 in
  Sarek_ir_metal.gen_expr buf (EConst (CFloat32 3.14)) ;
  let result = Buffer.contents buf in
  Alcotest.(check bool)
    "float32 literal is numeric"
    true
    (String.length result > 0 && result.[0] >= '0' && result.[0] <= '9') ;

  let buf = Buffer.create 64 in
  Sarek_ir_metal.gen_expr buf (EConst (CBool true)) ;
  Alcotest.(check string) "bool true literal" "1" (Buffer.contents buf) ;

  let buf = Buffer.create 64 in
  Sarek_ir_metal.gen_expr buf (EConst (CBool false)) ;
  Alcotest.(check string) "bool false literal" "0" (Buffer.contents buf)

let test_operations () =
  let buf = Buffer.create 64 in
  let x = make_var "x" TInt32 in
  let y = make_var "y" TInt32 in
  Sarek_ir_metal.gen_expr buf (EBinop (Add, EVar x, EVar y)) ;
  Alcotest.(check string) "addition" "(x + y)" (Buffer.contents buf) ;

  let buf = Buffer.create 64 in
  Sarek_ir_metal.gen_expr buf (EBinop (Sub, EVar x, EVar y)) ;
  Alcotest.(check string) "subtraction" "(x - y)" (Buffer.contents buf) ;

  let buf = Buffer.create 64 in
  Sarek_ir_metal.gen_expr buf (EBinop (Mul, EVar x, EVar y)) ;
  Alcotest.(check string) "multiplication" "(x * y)" (Buffer.contents buf)

let test_basics () =
  let buf = Buffer.create 64 in
  Sarek_ir_metal.gen_stmt buf "" SEmpty ;
  Alcotest.(check string) "empty statement" "" (Buffer.contents buf)

let test_assignment () =
  let buf = Buffer.create 64 in
  let x = make_var "x" TInt32 in
  Sarek_ir_metal.gen_stmt buf "" (SAssign (LVar x, EConst (CInt32 42l))) ;
  Alcotest.(check string) "assignment" "x = 42;\n" (Buffer.contents buf)

let test_if_statement () =
  let buf = Buffer.create 64 in
  let x = make_var "x" TInt32 in
  Sarek_ir_metal.gen_stmt
    buf
    ""
    (SIf
       ( EBinop (Gt, EVar x, EConst (CInt32 0l)),
         SAssign (LVar x, EConst (CInt32 1l)),
         None )) ;
  let result = Buffer.contents buf in
  Alcotest.(check bool)
    "if statement contains 'if'"
    true
    (Str.string_match (Str.regexp ".*if.*") result 0)

let test_while_loop () =
  let buf = Buffer.create 64 in
  let i = make_var "i" TInt32 in
  Sarek_ir_metal.gen_stmt
    buf
    ""
    (SWhile
       ( EBinop (Lt, EVar i, EConst (CInt32 10l)),
         SAssign (LVar i, EBinop (Add, EVar i, EConst (CInt32 1l))) )) ;
  let result = Buffer.contents buf in
  Alcotest.(check bool)
    "while loop contains 'while'"
    true
    (Str.string_match (Str.regexp ".*while.*") result 0)

let test_for_loop () =
  let buf = Buffer.create 64 in
  let i = make_var "i" TInt32 in
  Sarek_ir_metal.gen_stmt
    buf
    ""
    (SFor (i, EConst (CInt32 0l), EConst (CInt32 10l), Upto, SEmpty)) ;
  let result = Buffer.contents buf in
  Alcotest.(check bool)
    "for loop contains 'for'"
    true
    (Str.string_match (Str.regexp ".*for.*") result 0) ;
  Alcotest.(check bool)
    "for loop uses <= for upto"
    true
    (Str.string_match (Str.regexp ".*<=.*") result 0)

let test_barriers () =
  let buf = Buffer.create 64 in
  Sarek_ir_metal.gen_stmt buf "  " SBarrier ;
  let result = Buffer.contents buf in
  Alcotest.(check bool)
    "barrier generates threadgroup_barrier"
    true
    (Str.string_match (Str.regexp ".*threadgroup_barrier.*") result 0) ;

  let buf = Buffer.create 64 in
  Sarek_ir_metal.gen_stmt buf "  " SMemFence ;
  let result = Buffer.contents buf in
  Alcotest.(check bool)
    "memfence generates threadgroup_barrier"
    true
    (Str.string_match (Str.regexp ".*threadgroup_barrier.*") result 0)

let test_thread_intrinsics () =
  let result = Sarek_ir_metal.metal_thread_intrinsic "thread_idx_x" in
  Alcotest.(check bool)
    "thread_idx_x uses __metal_tid.x"
    true
    (Str.string_match (Str.regexp ".*__metal_tid.*x.*") result 0) ;

  let result = Sarek_ir_metal.metal_thread_intrinsic "global_idx_x" in
  Alcotest.(check bool)
    "global_idx_x uses __metal_gid.x"
    true
    (Str.string_match (Str.regexp ".*__metal_gid.*x.*") result 0)

let test_atomics () =
  let buf = Buffer.create 128 in
  let addr = make_var "counter" TInt32 in
  let value = EConst (CInt32 1l) in
  Sarek_ir_metal.Dispatch.gen_intrinsic
    Sarek_ir_metal.metal_backend
    buf
    []
    "atomic_add"
    [EVar addr; value] ;
  let result = Buffer.contents buf in
  Alcotest.(check bool)
    "atomic_add generates atomic_fetch_add_explicit"
    true
    (Str.string_match (Str.regexp ".*atomic_fetch_add_explicit.*") result 0)

let test_type_mapping () =
  Alcotest.(check string)
    "int32 maps to int"
    "int"
    (Sarek_ir_metal.metal_type_of_elttype TInt32) ;
  Alcotest.(check string)
    "int64 maps to long"
    "long"
    (Sarek_ir_metal.metal_type_of_elttype TInt64) ;
  Alcotest.(check string)
    "float32 maps to float"
    "float"
    (Sarek_ir_metal.metal_type_of_elttype TFloat32) ;
  (* This assertion used to read `"float64 maps to float (no double)"` and
     expect `"float"`. It encoded the defect and certified it as the contract.
     Metal has no double, which is a reason to REFUSE, not a licence to hand
     back half the width. #64 slice 1 makes it a refusal; the assertion is
     inverted to match.

     The severity is higher than "silently halved the precision", which is how
     this was originally described: the IR element type also fixes the buffer
     stride, and the host lays a float64 out in 8 bytes, so `device float*`
     strode the buffer at 4 and every element after the first was a bit-half of
     its neighbour (#141). Wrong answer, not degraded answer. *)
  (match Sarek_ir_metal.metal_type_of_elttype TFloat64 with
  | (_ : string) ->
      Alcotest.fail "float64 must be refused by Metal, not mapped to a type"
  | exception Sarek_backend_error.Backend_error.Backend_error _ -> ()) ;
  (* MSL `bool` is 1 byte; the host gives a Sarek bool a 4-byte slot
     (Sarek_ir_layout.scalar_size TBool = 4, mirroring Sarek_ppx), so a bool
     record field desynced silently — host {bool;bool;int} at 0/4/8 size 12
     against a device struct at 0/1/4 size 8. CUDA and OpenCL already emit `int`
     here. Not a capability refusal, by the §5.1 test: a correct lowering
     exists in the target language, so it is a codegen bug and gets emitted
     correctly rather than refused. Width invariant swept for every backend by
     sarek/tests/codegen_golden/test_backend_type_width_totality.ml. *)
  Alcotest.(check string)
    "bool maps to int (host bool slot is 4 bytes, MSL bool is 1)"
    "int"
    (Sarek_ir_metal.metal_type_of_elttype TBool)

let test_var_decl () =
  let buf = Buffer.create 64 in
  let x = make_var "x" TInt32 in
  Sarek_ir_metal.gen_var_decl buf "" x.var_name x.var_type (EConst (CInt32 42l)) ;
  Alcotest.(check string)
    "gen_var_decl produces type var = expr;"
    "int x = 42;\n"
    (Buffer.contents buf)

let test_array_decl () =
  let buf = Buffer.create 64 in
  Sarek_ir_metal.gen_array_decl buf "" "arr" TFloat32 (EConst (CInt32 256l)) "" ;
  Alcotest.(check string)
    "gen_array_decl produces type arr[size];"
    "float arr[256];\n"
    (Buffer.contents buf)

let test_indent_nested () =
  let nested = Sarek_ir_metal.indent_nested "  " in
  Alcotest.(check string) "indent_nested adds two spaces" "    " nested

(* Local address space on a buffer parameter is REFUSED (#139 / PR #316 review).

   [metal_memspace Local] is "", so without this guard the emitter produced
   " float* v [[buffer(0)]]" — a pointer with no address space, the other half
   of MSL 3.2 §4.2 and precisely the shape Metal_gate.Metal_addrspace rejects.
   The emitter would have been generating source its own gate refuses.

   Both routes are exercised: the explicit [array_info] and the bare [TArray]
   the DParam-without-info arm derives its space from. The second is the newly
   reachable one. *)
let expect_local_refused label decl =
  let buf = Buffer.create 64 in
  match Sarek_ir_metal.gen_param_metal buf [] 0 decl with
  | (_ : int) ->
      Alcotest.failf
        "%s: Local was accepted and emitted %S — a pointer with no address \
         space, which Metal rejects"
        label
        (Buffer.contents buf)
  | exception _ -> ()

let test_local_buffer_param_refused () =
  let v =
    {var_name = "v"; var_id = 0; var_type = TFloat32; var_mutable = false}
  in
  expect_local_refused
    "explicit array_info"
    (DParam (v, Some {arr_elttype = TFloat32; arr_memspace = Local})) ;
  let a =
    {
      var_name = "a";
      var_id = 0;
      var_type = TArray (TFloat32, Local);
      var_mutable = false;
    }
  in
  expect_local_refused "TArray (_, Local) with no array_info" (DParam (a, None)) ;
  (* Control: Global on the same shapes still emits, so the guard rejects Local
     specifically rather than everything. *)
  let g =
    {
      var_name = "g";
      var_id = 0;
      var_type = TArray (TFloat32, Global);
      var_mutable = false;
    }
  in
  let buf = Buffer.create 64 in
  let _ = Sarek_ir_metal.gen_param_metal buf [] 0 (DParam (g, None)) in
  Alcotest.(check bool)
    "Global still emits a device pointer"
    true
    (String.length (Buffer.contents buf) > 0
    && String.sub (Buffer.contents buf) 0 6 = "device")

(* #64 slice 1: the whole-kernel f64 gate at Metal's [generate] entry.
   [test_type_mapping] covers the per-element-type arm; this covers the gate,
   which exists so the refusal cannot be routed around by a path that formats a
   type some other way. Both halves are needed — the f16 gate has the same
   shape for the same reason. *)
let mk_vec_kernel elt : kernel =
  let v = make_var "x" (TVec elt) in
  {
    default_kernel with
    kern_name = "capgate";
    kern_params = [DParam (v, Some {arr_elttype = elt; arr_memspace = Global})];
  }

let substring_present haystack needle =
  let nl = String.length needle and hl = String.length haystack in
  let rec go i =
    i + nl <= hl && (String.sub haystack i nl = needle || go (i + 1))
  in
  nl = 0 || go 0

let test_float64_kernel_gate () =
  (* Red: an f64 kernel is refused, and the message names the capability and
     the target rather than failing anonymously. *)
  (match
     try Ok (Sarek_ir_metal.generate (mk_vec_kernel TFloat64))
     with Sarek_backend_error.Backend_error.Backend_error _ as e ->
       Error (Printexc.to_string e)
   with
  | Ok (_ : string) ->
      Alcotest.fail "Metal must refuse an f64 kernel, not generate source"
  | Error msg ->
      Alcotest.(check bool)
        "refusal names float64"
        true
        (substring_present msg "float64") ;
      Alcotest.(check bool)
        "refusal names Metal"
        true
        (substring_present msg "Metal")) ;
  (* Positive control: the gate must discriminate. A gate that raised on every
     kernel would satisfy the assertion above and be useless. *)
  match Sarek_ir_metal.generate (mk_vec_kernel TFloat32) with
  | (_ : string) -> ()
  | exception Sarek_backend_error.Backend_error.Backend_error _ ->
      Alcotest.fail
        "Metal must still generate an f32 kernel (gate fires unconditionally)"

(* The case that makes the whole-kernel gate load-bearing rather than
   defence-in-depth. An f64 LITERAL never reaches [metal_type_of_elttype]:
   [gen_expr] emits [EConst (CFloat64 f)] as a bare `%.17g` with no type ever
   consulted and no `f` suffix (Sarek_ir_metal.ml, the CFloat64 arm). With only
   the per-element-type arm, this kernel — an f32 buffer written from a binary64
   constant — generated clean and lost the precision silently, exactly as the
   TFloat64 arm did. Removing [reject_float64_kernel] turns this test red while
   leaving the TFloat64 one green, which is what says the two checks are not
   redundant. *)
let test_float64_literal_gate () =
  let v = make_var "x" (TVec TFloat32) in
  let k : kernel =
    {
      default_kernel with
      kern_name = "caplit";
      kern_params =
        [DParam (v, Some {arr_elttype = TFloat32; arr_memspace = Global})];
      kern_body =
        SAssign (LArrayElem ("x", EConst (CInt32 0l)), EConst (CFloat64 3.14));
    }
  in
  match
    try Ok (Sarek_ir_metal.generate k)
    with Sarek_backend_error.Backend_error.Backend_error _ as e ->
      Error (Printexc.to_string e)
  with
  | Ok src ->
      Alcotest.fail
        ("Metal must refuse an f64 literal, not emit it silently; got: " ^ src)
  | Error msg ->
      Alcotest.(check bool)
        "literal refusal names float64"
        true
        (substring_present msg "float64")

let () =
  Alcotest.run
    "Sarek_ir_metal"
    [
      ( "literals",
        [Alcotest.test_case "basic literals" `Quick test_basic_literals] );
      ("operations", [Alcotest.test_case "operations" `Quick test_operations]);
      ("basics", [Alcotest.test_case "basic statements" `Quick test_basics]);
      ("assignment", [Alcotest.test_case "assignment" `Quick test_assignment]);
      ("if", [Alcotest.test_case "if statement" `Quick test_if_statement]);
      ("while", [Alcotest.test_case "while loop" `Quick test_while_loop]);
      ("for", [Alcotest.test_case "for loop" `Quick test_for_loop]);
      ( "barriers",
        [Alcotest.test_case "barrier intrinsics" `Quick test_barriers] );
      ( "thread",
        [Alcotest.test_case "thread intrinsics" `Quick test_thread_intrinsics]
      );
      ("atomics", [Alcotest.test_case "atomic operations" `Quick test_atomics]);
      ("types", [Alcotest.test_case "type mapping" `Quick test_type_mapping]);
      ( "capability",
        [
          Alcotest.test_case "f64 kernel gate" `Quick test_float64_kernel_gate;
          Alcotest.test_case "f64 literal gate" `Quick test_float64_literal_gate;
        ] );
      ("var_decl", [Alcotest.test_case "var declaration" `Quick test_var_decl]);
      ( "param_address_space",
        [
          Alcotest.test_case
            "Local buffer parameter is refused"
            `Quick
            test_local_buffer_param_refused;
        ] );
      ( "array_decl",
        [Alcotest.test_case "array declaration" `Quick test_array_decl] );
      ("indent", [Alcotest.test_case "indent helper" `Quick test_indent_nested]);
    ]
