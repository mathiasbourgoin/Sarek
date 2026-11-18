(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Wrapping signed integer arithmetic in the C-family emitters
 * (Sarek_ir_c_arith): spelling and typing, without a compiler. Compile and
 * UBSan execution checks: test_wrapping_signed_arith_exec.
 *
 * 1. Spelling: CUDA C, OpenCL C and MSL print int32/int64 [+], [-], [*],
 *    unary [-] and [lsl] through the unsigned type; float arithmetic, [/],
 *    [mod], [asr] and operands of unknown type keep the plain C operator;
 *    PTX, GLSL and WGSL keep their own (natively wrapping) forms.
 * 2. Typing: operands typed by array element, record field, helper return
 *    type, intrinsic signature and C's usual conversions (int32 + int64);
 *    index arithmetic (subscript roots, lets used only there) keeps C's
 *    operators; a deep chain is typed in linear time.
 ******************************************************************************)

open Sarek_ir_types
open Sarek_codegen
open Wrapping_support

let test_int32_spelling d () =
  let u = d.u32 in
  expect
    d
    "add"
    (EBinop (Add, ev x, ev y))
    (Printf.sprintf "((int)((%s)x + (%s)y))" u u) ;
  expect
    d
    "sub"
    (EBinop (Sub, ev x, ev y))
    (Printf.sprintf "((int)((%s)x - (%s)y))" u u) ;
  expect
    d
    "mul"
    (EBinop (Mul, ev x, ev y))
    (Printf.sprintf "((int)((%s)x * (%s)y))" u u) ;
  expect d "neg" (EUnop (Neg, ev x)) (Printf.sprintf "((int)(-(%s)x))" u) ;
  expect
    d
    "lsl: the count keeps its type"
    (EBinop (Shl, ev x, ev y))
    (Printf.sprintf "((int)((%s)x << y))" u) ;
  expect
    d
    "chain converts back once, literals take the u suffix"
    (EBinop (Add, EBinop (Mul, ev x, i32 6), i32 1))
    (Printf.sprintf "((int)(((%s)x * 6u) + 1u))" u) ;
  expect
    d
    "negative literal operand"
    (EBinop (Add, ev x, i32 (-5)))
    (Printf.sprintf "((int)((%s)x + (%s)(-5)))" u u) ;
  expect
    d
    "negated literal still folds (min_int)"
    (EUnop (Neg, EConst (CInt32 Int32.min_int)))
    "(-2147483647 - 1)"

let test_int64_spelling d () =
  let u = d.u64 and s = d.s64 in
  expect
    d
    "add"
    (EBinop (Add, ev p, ev q))
    (Printf.sprintf "((%s)((%s)p + (%s)q))" s u u) ;
  expect d "neg" (EUnop (Neg, ev p)) (Printf.sprintf "((%s)(-(%s)p))" s u) ;
  expect
    d
    "int32 literal in an int64 chain"
    (EBinop (Mul, ev p, i32 3))
    (Printf.sprintf "((%s)((%s)p * 3%s))" s u d.u64_suffix) ;
  expect
    d
    "int32 + int64 is int64 (usual conversions)"
    (EBinop (Add, ev x, ev p))
    (Printf.sprintf "((%s)((%s)x + (%s)p))" s u u)

let test_unchanged d () =
  expect d "float add" (EBinop (Add, ev f, ev f)) "(f + f)" ;
  expect d "float neg" (EUnop (Neg, ev f)) "(-f)" ;
  expect d "div" (EBinop (Div, ev x, ev y)) "(x / y)" ;
  expect d "mod" (EBinop (Mod, ev x, ev y)) "(x % y)" ;
  expect d "asr" (EBinop (Shr, ev x, ev y)) "(x >> y)" ;
  expect d "and" (EBinop (BitAnd, ev x, ev y)) "(x & y)" ;
  (* [arr] is not declared in the empty environment: its type is unknown, so
     the operator stays plain rather than risk wrapping a float. *)
  expect
    d
    "unknown operand type"
    (EBinop (Add, EArrayRead ("arr", ev x), ev x))
    "(arr[x] + x)"

(* ---- 2. typing through the kernel environment ---- *)

let test_typing () =
  let k = typed_kernel () in
  let env = C_arith.env_of_kernel k in
  let width e = C_arith.int_width env e in
  let check label expected e =
    Alcotest.(check bool) label true (width e = expected)
  in
  check "int32 array element" (Some C_arith.W32) (EArrayRead ("out", i32 0)) ;
  check "float array element" None (EArrayRead ("fo", i32 0)) ;
  check
    "record field"
    (Some C_arith.W32)
    (ERecordField (EVar (var "r" (TRecord ("pt", []))), "px")) ;
  check
    "float record field"
    None
    (ERecordField (EVar (var "r" (TRecord ("pt", []))), "pw")) ;
  check
    "helper result"
    (Some C_arith.W32)
    (EApp (EVar (var "twice" TInt32), [])) ;
  check
    "thread intrinsic"
    (Some C_arith.W32)
    (EIntrinsic ([], "thread_idx_x", [])) ;
  check "array length" (Some C_arith.W32) (EArrayLen "out") ;
  check "int64 cast" (Some C_arith.W64) (ECast (TInt64, ev f)) ;
  check "float cast" None (ECast (TFloat32, ev x)) ;
  check "int32 lsr" (Some C_arith.W32) (EBinop (Lshr32, ev x, ev y)) ;
  check "comparison is bool (int)" (Some C_arith.W32) (EBinop (Lt, ev f, ev f)) ;
  let src = Sarek_ir_opencl.generate k in
  check_has
    "OpenCL"
    src
    "out[i] = ((int)((uint)out[i] + ((uint)r.px * (uint)twice(n))));" ;
  check_has "OpenCL helper" src "return ((int)((uint)n + (uint)n));" ;
  check_has "OpenCL float stays plain" src "fo[i] = (fo[i] + " ;
  let cuda_src = Sarek_ir_cuda.generate k in
  check_has
    "CUDA"
    cuda_src
    "out[i] = ((int)((unsigned int)out[i] + ((unsigned int)r.px * (unsigned \
     int)twice(n))));" ;
  check_has "CUDA float stays plain" cuda_src "fo[i] = (fo[i] + "

(* ---- 2b. subscripts keep C's operators at their root ---- *)

let test_subscripts () =
  let out = var "out" (TVec TInt32) in
  let a = var "a" (TVec TInt32) in
  let k =
    {
      default_kernel with
      kern_name = "subs";
      kern_params =
        [
          DParam (out, Some {arr_elttype = TInt32; arr_memspace = Global});
          DParam (a, Some {arr_elttype = TInt32; arr_memspace = Global});
          DParam (x, None);
          DParam (y, None);
        ];
      kern_body =
        SSeq
          [
            (* root chain of the subscript: plain; the stored value: wraps *)
            SAssign
              ( LArrayElem ("out", EBinop (Add, EBinop (Mul, ev y, ev x), i32 1)),
                EArrayRead ("a", EBinop (Sub, ev x, i32 1)) );
            (* below [land] the hash wraps *)
            SAssign
              ( LArrayElem
                  ( "out",
                    EBinop
                      ( BitAnd,
                        EBinop (Add, EBinop (Mul, ev x, i32 31), ev y),
                        i32 7 ) ),
                EBinop (Add, ev x, ev y) );
          ];
    }
  in
  List.iter
    (fun (name, gen, u) ->
      let src = gen k in
      check_has name src "out[((y * x) + 1)] = a[(x - 1)];" ;
      check_has
        name
        src
        (Printf.sprintf
           "out[(((int)(((%s)x * 31u) + (%s)y)) & 7)] = ((int)((%s)x + (%s)y));"
           u
           u
           u
           u))
    [
      ("CUDA", Sarek_ir_cuda.generate, "unsigned int");
      ("OpenCL", Sarek_ir_opencl.generate, "uint");
      ("Metal", Sarek_ir_metal.generate, "uint");
    ]

(* ---- 1b. the natively wrapping backends are untouched ---- *)

let test_other_backends () =
  let k = arith_kernel () in
  let glsl = Sarek_ir_glsl.generate k in
  check_has "GLSL" glsl "(a + b)" ;
  check_has "GLSL" glsl "(a * b)" ;
  check_lacks "GLSL" glsl "uint(a)" ;
  let wgsl = Sarek_ir_wgsl.generate k in
  check_has "WGSL" wgsl "(params.a + params.b)" ;
  check_lacks "WGSL" wgsl "u32(params.a)" ;
  let ptx = Sarek_ir_ptx.generate k in
  check_has "PTX" ptx "add.u32" ;
  check_has "PTX" ptx "mul.lo.u32" ;
  check_has "PTX" ptx "neg.s32"

(* ---- 2c. lets used only in index arithmetic ---- *)

let int_let name = {(var name TInt32) with var_id = 1}

(* [let off = row * k + i * 4 in a.(off)] inside a loop, [let base = y * x]
   used only by [off]'s definition, and [let v = x + y] used both as an index
   and as a stored value. *)
let index_let_kernel () =
  let out = var "out" (TVec TInt32) in
  let a = var "a" (TVec TInt32) in
  let i = var "i" TInt32 in
  let base = int_let "base" and off = int_let "off" and v = int_let "v" in
  {
    default_kernel with
    kern_name = "lets";
    kern_params =
      [
        DParam (out, Some {arr_elttype = TInt32; arr_memspace = Global});
        DParam (a, Some {arr_elttype = TInt32; arr_memspace = Global});
        DParam (x, None);
        DParam (y, None);
      ];
    kern_body =
      SLet
        ( base,
          EBinop (Mul, ev y, ev x),
          SSeq
            [
              SFor
                ( i,
                  i32 0,
                  i32 3,
                  Upto,
                  SLet
                    ( off,
                      EBinop (Add, ev base, EBinop (Mul, ev i, i32 4)),
                      SAssign
                        (LArrayElem ("out", ev i), EArrayRead ("a", ev off)) )
                );
              SLet
                ( v,
                  EBinop (Add, ev x, ev y),
                  SAssign (LArrayElem ("out", ev v), ev v) );
            ] );
  }

let test_index_lets () =
  let k = index_let_kernel () in
  let env = C_arith.env_of_kernel k in
  let is name = C_arith.is_index_let env (int_let name) in
  Alcotest.(check bool) "off: only a subscript" true (is "off") ;
  Alcotest.(check bool) "base: only in off's definition" true (is "base") ;
  Alcotest.(check bool) "v: also a stored value" false (is "v") ;
  List.iter
    (fun (name, gen, u) ->
      let src = gen k in
      check_has name src "int base = (y * x);" ;
      check_has name src "int off = (base + (i * 4));" ;
      check_has name src (Printf.sprintf "int v = ((int)((%s)x + (%s)y));" u u))
    [
      ("CUDA", Sarek_ir_cuda.generate, "unsigned int");
      ("OpenCL", Sarek_ir_opencl.generate, "uint");
      ("Metal", Sarek_ir_metal.generate, "uint");
    ]

(* A chain of nested negations and shifts typed once per node: before the
   types were memoised, each level typed its operand twice (2^depth). *)
let test_deep_chain () =
  let rec chain n e =
    if n = 0 then e
    else
      chain (n - 1) (EUnop (Neg, EBinop (Shl, EBinop (Add, e, i32 1), i32 1)))
  in
  let e = chain 2000 (ev x) in
  let t0 = Unix.gettimeofday () in
  let src = cuda.expr (C_arith.empty ()) e in
  let dt = Unix.gettimeofday () -. t0 in
  Alcotest.(check bool) "printed" true (String.length src > 2000) ;
  if dt > 2.0 then Alcotest.failf "depth 2000 took %.2f s" dt

let () =
  let per_dialect name test =
    List.map
      (fun d -> Alcotest.test_case (name ^ " " ^ d.name) `Quick (test d))
      dialects
  in
  Alcotest.run
    "wrapping_signed_arith"
    [
      ( "spelling",
        per_dialect "int32" test_int32_spelling
        @ per_dialect "int64" test_int64_spelling
        @ per_dialect "unchanged" test_unchanged
        @ [Alcotest.test_case "PTX/GLSL/WGSL" `Quick test_other_backends] );
      ( "typing",
        [
          Alcotest.test_case "kernel environment" `Quick test_typing;
          Alcotest.test_case "subscripts" `Quick test_subscripts;
          Alcotest.test_case "index lets" `Quick test_index_lets;
          Alcotest.test_case "deep chain" `Quick test_deep_chain;
        ] );
    ]
