(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Wrapping signed integer arithmetic (Sarek_ir_c_arith), compiled and run.
 *
 * 1. Compile: clang accepts the generated OpenCL C (CL1.2 and CL2.0) and
 *    CUDA C (device-only, as CUDA and as HIP) kernels.
 * 2. Execution: the printed expressions of every dialect, compiled as C++ with
 *    UBSan's signed-overflow and shift checks fatal, run on overflowing
 *    inputs without a report and return OCaml's Int32/Int64 results. A plain
 *    [a + b] on the same inputs is the positive control: it must trap.
 * Both skip when clang/clang++ (with UBSan) is not available.
 ******************************************************************************)

open Sarek_ir_types
open Sarek_codegen
open Wrapping_support

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

(* The targets, each with a probe kernel. A target is checked only when its
   probe compiles (a POSITIVE control, as in Opencl_clang): a clang built
   without the AMDGPU or NVPTX backend, or without OpenCL headers, is on PATH
   and useless for that target, and says so instead of failing the test. *)
let targets =
  let cl std =
    ( "OpenCL " ^ std,
      ".cl",
      "-x cl -cl-std=" ^ std ^ " -Xclang -finclude-default-header",
      "__kernel void probe(__global int *o) { o[get_global_id(0)] = 1; }\n",
      Sarek_ir_opencl.generate )
  in
  let cu name flags =
    ( name,
      (if name = "HIP" then ".hip" else ".cu"),
      flags,
      cuda_prelude ^ "__global__ void probe(int *o) { o[threadIdx.x] = 1; }\n",
      fun k -> cuda_prelude ^ Sarek_ir_cuda.generate k )
  in
  [
    cl "CL1.2";
    cl "CL2.0";
    cu
      "CUDA"
      "-x cuda --cuda-device-only -nocudainc -nocudalib --cuda-gpu-arch=sm_70";
    cu
      "HIP"
      "-x hip --cuda-device-only -nogpuinc -nogpulib --offload-arch=gfx1100";
  ]

let test_clang_compile () =
  if not (Lazy.force has_clang) then Alcotest.skip () ;
  let kernels = [arith_kernel (); typed_kernel ()] in
  let checked =
    List.filter
      (fun (label, ext, flags, probe, gen) ->
        match compile ~ext ~flags probe with
        | Error log ->
            Printf.printf
              "  %s: SKIP (this clang cannot compile the probe)\n%s"
              label
              log ;
            false
        | Ok () ->
            List.iter
              (fun k -> check_compiles label ~ext ~flags (gen k))
              kernels ;
            true)
      targets
  in
  if checked = [] then Alcotest.skip ()

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
let functions d buf =
  let env = C_arith.empty () in
  List.iteri
    (fun i (e, _) ->
      Printf.bprintf
        buf
        "int f32_%d(int x, int y) { (void)y; return %s; }\n"
        i
        (d.expr env e))
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
        (d.expr env e))
    int64_exprs

(* [-2147483648] is not an int literal in C++: every operand is spelled as a
   long long literal, which [int] then holds exactly when it fits. *)
let lit64 v =
  if v = Int64.min_int then "(-9223372036854775807LL - 1LL)"
  else Int64.to_string v ^ "LL"

let calls buf ~ty ~prefix ~format ~count (a, b) =
  Printf.bprintf buf "  { volatile %s a = %s, b = %s;\n" ty (lit64 a) (lit64 b) ;
  for i = 0 to count - 1 do
    Printf.bprintf buf "    printf(\"%s\\n\", %s_%d(a, b));\n" format prefix i
  done ;
  Buffer.add_string buf "  }\n"

let program d =
  let buf = Buffer.create 4096 in
  Buffer.add_string
    buf
    "#include <cstdio>\n\
     typedef unsigned int uint;\n\
     typedef unsigned long ulong;\n" ;
  functions d buf ;
  Buffer.add_string buf "int main() {\n" ;
  List.iter
    (fun (a, b) ->
      calls
        buf
        ~ty:"int"
        ~prefix:"f32"
        ~format:"%d"
        ~count:(List.length int32_exprs)
        (Int64.of_int32 a, Int64.of_int32 b))
    int32_cases ;
  List.iter
    (calls
       buf
       ~ty:d.s64
       ~prefix:"(long long)f64"
       ~format:"%lld"
       ~count:(List.length int64_exprs))
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
  Alcotest.run
    "wrapping_signed_arith_exec"
    [
      ("clang", [Alcotest.test_case "compile" `Quick test_clang_compile]);
      ( "ubsan",
        List.map
          (fun d ->
            Alcotest.test_case ("execute " ^ d.name) `Quick (test_ubsan d))
          dialects );
    ]
