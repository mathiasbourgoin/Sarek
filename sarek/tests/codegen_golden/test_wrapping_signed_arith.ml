(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Wrapping signed integer arithmetic in the C-family emitters
 * (Sarek_ir_c_arith).
 *
 * 1. Spelling: CUDA C, OpenCL C and MSL print int32/int64 [+], [-], [*],
 *    unary [-] and [lsl] through the unsigned type; float arithmetic, [/],
 *    [mod], [asr] and operands of unknown type keep the plain C operator;
 *    PTX, GLSL and WGSL keep their own (natively wrapping) forms.
 * 2. Typing: operands typed by array element, record field, helper return
 *    type, intrinsic signature and C's usual conversions (int32 + int64);
 *    the root chain of an array subscript keeps C's operators.
 * 3. Compile: clang accepts the generated OpenCL C (CL1.2 and CL2.0) and
 *    CUDA C (device-only, as CUDA and as HIP) kernels.
 * 4. Execution: the printed expressions of every dialect, compiled as C++ with
 *    UBSan's signed-overflow and shift checks fatal, run on overflowing
 *    inputs without a report and return OCaml's Int32/Int64 results. A plain
 *    [a + b] on the same inputs is the positive control: it must trap.
 * Steps 3 and 4 skip when clang/clang++ (with UBSan) is not available.
 ******************************************************************************)

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

(* ---- 1. spelling of single expressions ---- *)

type dialect = {
  name : string;
  expr : C_arith.env -> expr -> string;
  u32 : string;
  s64 : string;
  u64 : string;
  u64_suffix : string;
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
  }

let dialects = [cuda; opencl; metal]

let expect d label e expected =
  Alcotest.(check string)
    (d.name ^ ": " ^ label)
    expected
    (d.expr C_arith.empty e)

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

(* ---- 3. clang accepts the generated kernels ---- *)

let read_file path = In_channel.with_open_bin path In_channel.input_all

let run_tool cmd =
  match Unix.system (cmd ^ " >/dev/null 2>&1") with
  | Unix.WEXITED 0 -> true
  | _ -> false

let has_clang = lazy (run_tool "command -v clang")

let has_clangxx = lazy (run_tool "command -v clang++")

let compile ~ext ~flags source =
  let dir = Filename.temp_dir "sarek_wrap_" "" in
  let src = Filename.concat dir ("k" ^ ext) in
  let log = Filename.concat dir "log" in
  Out_channel.with_open_bin src (fun oc -> output_string oc source) ;
  let rc =
    Unix.system
      (Printf.sprintf
         "clang %s -fsyntax-only -Werror=integer-overflow %s >%s 2>&1"
         flags
         (Filename.quote src)
         (Filename.quote log))
  in
  let out = read_file log in
  ignore (Unix.system ("rm -r " ^ Filename.quote dir)) ;
  match rc with Unix.WEXITED 0 -> Ok () | _ -> Error out

let check_compiles label ~ext ~flags source =
  match compile ~ext ~flags source with
  | Ok () -> ()
  | Error log -> Alcotest.failf "%s: clang rejected:\n%s\n%s" label log source

(* Declarations the CUDA headers would bring, for a device-only syntax check
   without a CUDA installation (-nocudainc). *)
let cuda_prelude =
  "#define __global__ __attribute__((global))\n\
   #define __device__ __attribute__((device))\n\
   #define __shared__ __attribute__((shared))\n\
   struct sarek_dim3 { unsigned int x, y, z; };\n\
   extern __device__ const sarek_dim3 threadIdx, blockIdx, blockDim, gridDim;\n"

let test_clang_compile () =
  if not (Lazy.force has_clang) then Alcotest.skip () ;
  let kernels = [arith_kernel (); typed_kernel ()] in
  List.iter
    (fun k ->
      let cl = Sarek_ir_opencl.generate k in
      List.iter
        (fun std ->
          check_compiles
            ("OpenCL " ^ std)
            ~ext:".cl"
            ~flags:("-x cl -cl-std=" ^ std ^ " -Xclang -finclude-default-header")
            cl)
        ["CL1.2"; "CL2.0"] ;
      let cu = cuda_prelude ^ Sarek_ir_cuda.generate k in
      check_compiles
        "CUDA"
        ~ext:".cu"
        ~flags:
          "-x cuda --cuda-device-only -nocudainc -nocudalib \
           --cuda-gpu-arch=sm_70"
        cu ;
      check_compiles
        "HIP"
        ~ext:".hip"
        ~flags:
          "-x hip --cuda-device-only -nogpuinc -nogpulib --offload-arch=gfx1100"
        cu)
    kernels

(* ---- 4. execution under UBSan ---- *)

let int32_cases =
  [
    (Int32.max_int, 1l);
    (Int32.min_int, -1l);
    (Int32.min_int, Int32.min_int);
    (65536l, 65536l);
    (-7l, 3l);
  ]

let int64_cases =
  [
    (Int64.max_int, 1L);
    (Int64.min_int, -1L);
    (4294967296L, 4294967296L);
    (-7L, 3L);
  ]

(* Expression, its OCaml value. Shift counts stay in range: outside it OCaml
   leaves the result unspecified (see Sarek_ir_c_arith). *)
let int32_exprs =
  [
    (EBinop (Add, ev x, ev y), fun a b -> Int32.add a b);
    (EBinop (Sub, ev x, ev y), fun a b -> Int32.sub a b);
    (EBinop (Mul, ev x, ev y), fun a b -> Int32.mul a b);
    (EUnop (Neg, ev x), fun a _ -> Int32.neg a);
    (EBinop (Shl, ev x, i32 31), fun a _ -> Int32.shift_left a 31);
    ( EBinop (Sub, EBinop (Mul, ev x, i32 3), EUnop (Neg, ev y)),
      fun a b -> Int32.sub (Int32.mul a 3l) (Int32.neg b) );
  ]

let int64_exprs =
  [
    (EBinop (Add, ev p, ev q), fun a b -> Int64.add a b);
    (EBinop (Sub, ev p, ev q), fun a b -> Int64.sub a b);
    (EBinop (Mul, ev p, ev q), fun a b -> Int64.mul a b);
    (EUnop (Neg, ev p), fun a _ -> Int64.neg a);
    (EBinop (Shl, ev p, EConst (CInt64 63L)), fun a _ -> Int64.shift_left a 63);
  ]

(* A C++ translation unit: one function per expression, a [main] that prints
   every result for every case. The dialect's type names are typedefs. *)
let program d =
  let buf = Buffer.create 4096 in
  Buffer.add_string
    buf
    "#include <cstdio>\n\
     typedef unsigned int uint;\n\
     typedef unsigned long ulong;\n" ;
  List.iteri
    (fun i (e, _) ->
      Printf.bprintf
        buf
        "int f32_%d(int x, int y) { (void)y; return %s; }\n"
        i
        (d.expr C_arith.empty e))
    int32_exprs ;
  List.iteri
    (fun i (e, _) ->
      Printf.bprintf
        buf
        "%s f64_%d(%s p, %s q) { (void)q; return %s; }\n"
        d.s64
        i
        d.s64
        d.s64
        (d.expr C_arith.empty e))
    int64_exprs ;
  Buffer.add_string buf "int main() {\n" ;
  List.iter
    (fun (a, b) ->
      Printf.bprintf buf "  { volatile int a = %LdLL, b = %LdLL;\n" a b ;
      List.iteri
        (fun i _ ->
          Printf.bprintf buf "    printf(\"%%d\\n\", f32_%d(a, b));\n" i)
        int32_exprs ;
      Buffer.add_string buf "  }\n")
    (List.map
       (fun (a, b) ->
         (* [-2147483648] is not an int literal in C++; spell it as a long
            long, which [int] then holds exactly. *)
         (Int64.of_int32 a, Int64.of_int32 b))
       int32_cases) ;
  List.iter
    (fun (a, b) ->
      let lit v =
        if v = Int64.min_int then "(-9223372036854775807LL - 1LL)"
        else Int64.to_string v ^ "LL"
      in
      Printf.bprintf
        buf
        "  { volatile %s a = %s, b = %s;\n"
        d.s64
        (lit a)
        (lit b) ;
      List.iteri
        (fun i _ ->
          Printf.bprintf
            buf
            "    printf(\"%%lld\\n\", (long long)f64_%d(a, b));\n"
            i)
        int64_exprs ;
      Buffer.add_string buf "  }\n")
    int64_cases ;
  Buffer.add_string buf "  return 0;\n}\n" ;
  Buffer.contents buf

let expected_output () =
  let lines =
    List.concat_map
      (fun (a, b) ->
        List.map (fun (_, f) -> Int32.to_string (f a b)) int32_exprs)
      int32_cases
    @ List.concat_map
        (fun (a, b) ->
          List.map (fun (_, f) -> Int64.to_string (f a b)) int64_exprs)
        int64_cases
  in
  String.concat "\n" lines ^ "\n"

let ubsan = "-fsanitize=signed-integer-overflow,shift -fno-sanitize-recover=all"

(* Compile [source] with UBSan and run it; [Ok stdout] when it exits 0. *)
let run_ubsan source =
  let dir = Filename.temp_dir "sarek_wrap_ubsan_" "" in
  let src = Filename.concat dir "t.cpp" in
  let exe = Filename.concat dir "t" in
  let out = Filename.concat dir "out" in
  Out_channel.with_open_bin src (fun oc -> output_string oc source) ;
  let result =
    match
      Unix.system
        (Printf.sprintf
           "clang++ -O2 %s %s -o %s >%s 2>&1"
           ubsan
           (Filename.quote src)
           (Filename.quote exe)
           (Filename.quote out))
    with
    | Unix.WEXITED 0 -> (
        match
          Unix.system
            (Printf.sprintf
               "%s >%s 2>&1"
               (Filename.quote exe)
               (Filename.quote out))
        with
        | Unix.WEXITED 0 -> Ok (read_file out)
        | _ -> Error ("run: " ^ read_file out))
    | _ -> Error ("compile: " ^ read_file out)
  in
  ignore (Unix.system ("rm -r " ^ Filename.quote dir)) ;
  result

(* The sanitizer must be able to fail: UBSan's runtime is a separate package
   on some systems, and a missing one would make every run below vacuous. *)
let ubsan_works =
  lazy
    (Lazy.force has_clangxx
    && run_ubsan "int main() { return 0; }\n" = Ok ""
    &&
    match
      run_ubsan
        "volatile int a = 2147483647, b = 1;\nint main() { return a + b; }\n"
    with
    | Error _ -> true
    | Ok _ -> false)

let test_ubsan d () =
  if not (Lazy.force ubsan_works) then Alcotest.skip () ;
  match run_ubsan (program d) with
  | Ok out ->
      Alcotest.(check string)
        (d.name ^ ": results equal OCaml's, no UB report")
        (expected_output ())
        out
  | Error e -> Alcotest.failf "%s: %s\n%s" d.name e (program d)

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
        ] );
      ("clang", [Alcotest.test_case "compile" `Quick test_clang_compile]);
      ("ubsan", per_dialect "execute" test_ubsan);
    ]
