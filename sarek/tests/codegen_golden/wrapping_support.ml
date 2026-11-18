(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Shared by test_wrapping_signed_arith (spelling, typing) and
   test_wrapping_signed_arith_exec (clang compile, UBSan execution). *)

open Sarek_ir_types
open Sarek_codegen
module C_arith = Sarek_ir_c_arith

let var name ty =
  {var_name = name; var_id = 0; var_type = ty; var_mutable = false}

let x = var "x" TInt32

let y = var "y" TInt32

let p = var "p" TInt64

let q = var "q" TInt64

let f = var "f" TFloat32

let i32 n = EConst (CInt32 (Int32.of_int n))

let ev v = EVar v

let contains hay needle =
  let hl = String.length hay and nl = String.length needle in
  let rec go i = i + nl <= hl && (String.sub hay i nl = needle || go (i + 1)) in
  nl = 0 || go 0

let check_has label src needle =
  if not (contains src needle) then
    Alcotest.failf "%s: expected [%s] in:\n%s" label needle src

let check_lacks label src needle =
  if contains src needle then
    Alcotest.failf "%s: did not expect [%s] in:\n%s" label needle src

type dialect = {
  name : string;
  expr : C_arith.env -> expr -> string;
  u32 : string;
  s64 : string;
  u64 : string;
  u64_suffix : string;
  w32 : string -> string;  (** an unsigned int32 result back to [int] *)
  w64 : string -> string;
}

let render gen env e =
  let buf = Buffer.create 64 in
  gen env buf e ;
  Buffer.contents buf

let cuda =
  {
    name = "CUDA";
    expr =
      render (fun env ->
          Sarek_ir_cuda.gen_expr
            {(Sarek_ir_cuda.state default_kernel) with arith = env});
    u32 = "unsigned int";
    s64 = "long long";
    u64 = "unsigned long long";
    u64_suffix = "ull";
    w32 = (fun b -> "((int)" ^ b ^ ")");
    w64 = (fun b -> "((long long)" ^ b ^ ")");
  }

let opencl =
  {
    name = "OpenCL";
    expr =
      render (fun env ->
          Sarek_ir_opencl.gen_expr
            {(Sarek_ir_opencl.state default_kernel) with arith = env});
    u32 = "uint";
    s64 = "long";
    u64 = "ulong";
    u64_suffix = "ul";
    w32 = (fun b -> "as_int(" ^ b ^ ")");
    w64 = (fun b -> "as_long(" ^ b ^ ")");
  }

let metal =
  {
    name = "Metal";
    expr =
      render (fun env ->
          Sarek_ir_metal.gen_expr
            {(Sarek_ir_metal.state default_kernel) with arith = env});
    u32 = "uint";
    s64 = "long";
    u64 = "ulong";
    u64_suffix = "ul";
    w32 = (fun b -> "as_type<int>(" ^ b ^ ")");
    w64 = (fun b -> "as_type<long>(" ^ b ^ ")");
  }

let dialects = [cuda; opencl; metal]

let generate d k =
  match d.name with
  | "CUDA" -> Sarek_ir_cuda.generate k
  | "OpenCL" -> Sarek_ir_opencl.generate k
  | _ -> Sarek_ir_metal.generate k

let expect d label e expected =
  Alcotest.(check string)
    (d.name ^ ": " ^ label)
    expected
    (d.expr (C_arith.empty ()) e)

let typed_kernel () =
  let out = var "out" (TVec TInt32) in
  let fo = var "fo" (TVec TFloat32) in
  let n = var "n" TInt32 in
  let i = var "i" TInt32 in
  let r = var "r" (TRecord ("pt", [])) in
  let helper =
    {
      hf_name = "twice";
      hf_params = [n];
      hf_ret_type = TInt32;
      hf_body = SReturn (EBinop (Add, ev n, ev n));
    }
  in
  let body =
    SLet
      ( i,
        EIntrinsic ([], "global_thread_id", []),
        SLet
          ( r,
            ERecord ("pt", [("px", ev i); ("pw", EConst (CFloat32 1.0))]),
            SSeq
              [
                SAssign
                  ( LArrayElem ("out", ev i),
                    EBinop
                      ( Add,
                        EArrayRead ("out", ev i),
                        EBinop
                          ( Mul,
                            ERecordField (ev r, "px"),
                            EApp (EVar (var "twice" TInt32), [ev n]) ) ) );
                SAssign
                  ( LArrayElem ("fo", ev i),
                    EBinop
                      (Add, EArrayRead ("fo", ev i), ERecordField (ev r, "pw"))
                  );
              ] ) )
  in
  {
    default_kernel with
    kern_name = "typed";
    kern_params =
      [
        DParam (out, Some {arr_elttype = TInt32; arr_memspace = Global});
        DParam (fo, Some {arr_elttype = TFloat32; arr_memspace = Global});
        DParam (n, None);
      ];
    kern_types = [("pt", [("px", TInt32); ("pw", TFloat32)])];
    kern_funcs = [helper];
    kern_body = body;
  }

let arith_kernel () =
  let out = var "out" (TVec TInt32) in
  let a = var "a" TInt32 in
  let b = var "b" TInt32 in
  {
    default_kernel with
    kern_name = "arith";
    kern_params =
      [
        DParam (out, Some {arr_elttype = TInt32; arr_memspace = Global});
        DParam (a, None);
        DParam (b, None);
      ];
    kern_body =
      SSeq
        [
          SAssign (LArrayElem ("out", i32 0), EBinop (Add, ev a, ev b));
          SAssign (LArrayElem ("out", i32 1), EBinop (Mul, ev a, ev b));
          SAssign (LArrayElem ("out", i32 2), EUnop (Neg, ev a));
        ];
  }
