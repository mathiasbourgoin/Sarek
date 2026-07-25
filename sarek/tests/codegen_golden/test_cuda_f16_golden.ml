(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * f16 (#57 slice 1) codegen assertions.
 *
 * Deliberately a SEPARATE test file rather than new entries in the shared
 * test_codegen_golden.ml corpus: that corpus runs every kernel through all
 * backends, and f16 is only implemented for CUDA/HIP in slice 1. Keeping the f16
 * kernels out of the shared corpus is also what guarantees every committed
 * golden for a non-f16 kernel stays byte-identical.
 *
 * Two properties are asserted:
 *   1. An f16 kernel emits `__half` and the fp16 feature include.
 *   2. A NON-f16 kernel emits neither. That negative is the detector
 *      (Sarek_ir_analysis.kernel_uses_float16) doing its job, and it is the
 *      reason existing goldens are untouched.
 *
 * Plus: the deferred backends must REJECT f16 with a clear diagnostic rather
 * than silently emit a type their preamble never enabled.
 ******************************************************************************)

open Sarek_ir_types
open Sarek_codegen

let make_var name ty =
  {var_name = name; var_id = 0; var_type = ty; var_mutable = false}

let mk_kernel name params body =
  {
    kern_name = name;
    kern_params = params;
    kern_locals = [];
    kern_body = body;
    kern_types = [];
    kern_variants = [];
    kern_funcs = [];
    kern_native_fn = None;
  }

let reset () =
  Sarek_ir_cuda.current_framework := None ;
  Sarek_ir_cuda.current_variants := []

let gen k =
  reset () ;
  Sarek_ir_cuda.generate_with_types ~types:k.kern_types k

(** Substring search (no Str dependency in this test's library set). *)
let contains ~needle haystack =
  let nl = String.length needle and hl = String.length haystack in
  let rec go i =
    i + nl <= hl && (String.sub haystack i nl = needle || go (i + 1))
  in
  nl = 0 || go 0

let check_contains what src needle =
  if not (contains ~needle src) then
    Alcotest.failf "%s: expected to find %S in:\n%s" what needle src

let check_absent what src needle =
  if contains ~needle src then
    Alcotest.failf "%s: expected NOT to find %S in:\n%s" what needle src

(* ------------------------------------------------------------------ *)
(* Kernels                                                            *)
(* ------------------------------------------------------------------ *)

(** [f16_scale out inp]: reads an f16 element, widens to f32, doubles it in f32,
    narrows back to f16 on store. This is the "storage type, compute in f32"
    discipline expressed in IR. *)
let f16_scale_kernel () =
  let out = make_var "out" (TVec TFloat16) in
  let inp = make_var "inp" (TVec TFloat16) in
  let idx = make_var "idx" TInt32 in
  let body =
    SLet
      ( idx,
        EIntrinsic ([], "global_thread_id", []),
        SAssign
          ( LArrayElem ("out", EVar idx),
            ECast
              ( TFloat16,
                EBinop
                  ( Mul,
                    ECast (TFloat32, EArrayRead ("inp", EVar idx)),
                    EConst (CFloat32 2.0) ) ) ) )
  in
  mk_kernel
    "f16_scale"
    [
      DParam (out, Some {arr_elttype = TFloat16; arr_memspace = Global});
      DParam (inp, Some {arr_elttype = TFloat16; arr_memspace = Global});
    ]
    body

(** The same kernel shape with f32 throughout — the negative control. *)
let f32_scale_kernel () =
  let out = make_var "out" (TVec TFloat32) in
  let inp = make_var "inp" (TVec TFloat32) in
  let idx = make_var "idx" TInt32 in
  let body =
    SLet
      ( idx,
        EIntrinsic ([], "global_thread_id", []),
        SAssign
          ( LArrayElem ("out", EVar idx),
            EBinop (Mul, EArrayRead ("inp", EVar idx), EConst (CFloat32 2.0)) )
      )
  in
  mk_kernel
    "f32_scale"
    [
      DParam (out, Some {arr_elttype = TFloat32; arr_memspace = Global});
      DParam (inp, Some {arr_elttype = TFloat32; arr_memspace = Global});
    ]
    body

(* ------------------------------------------------------------------ *)
(* Tests                                                             *)
(* ------------------------------------------------------------------ *)

let test_f16_type_string () =
  let src = gen (f16_scale_kernel ()) in
  (* __half, not "half": `half` is only a typedef added by cuda_fp16.hpp / the
     HIP headers, whereas __half is the name both toolchains always define. *)
  check_contains "f16 param type" src "__half*" ;
  ()

let test_f16_include_emitted () =
  let src = gen (f16_scale_kernel ()) in
  check_contains "fp16 include" src "#include <cuda_fp16.h>" ;
  (* The include MUST be negatively guarded. This is not stylistic: HIP compiles
     through hiprtc, which pre-provides __half/__float2half but can resolve
     NEITHER <cuda_fp16.h> NOR <hip/hip_fp16.h> (verified against hiprtc on
     gfx1100 — both include forms fail "file not found", bare __half compiles).
     Since sarek-hip reuses this generator verbatim, an unguarded include would
     break every HIP f16 kernel. *)
  check_contains
    "hip guard"
    src
    "#if !defined(__HIP__) && !defined(__HIP_PLATFORM_AMD__)" ;
  check_contains "guard close" src "#endif" ;
  (* The include has to precede the extern "C" block: a C++ header cannot be
     pulled in with C linkage. *)
  let idx_of needle =
    let nl = String.length needle and hl = String.length src in
    let rec go i =
      if i + nl > hl then -1
      else if String.sub src i nl = needle then i
      else go (i + 1)
    in
    go 0
  in
  let i_inc = idx_of "#include <cuda_fp16.h>" in
  let i_extern = idx_of "extern \"C\"" in
  if not (i_inc >= 0 && i_extern >= 0 && i_inc < i_extern) then
    Alcotest.failf
      "fp16 include must appear before the extern \"C\" block (include@%d, \
       extern@%d):\n\
       %s"
      i_inc
      i_extern
      src ;
  ()

let test_f16_conversions () =
  let src = gen (f16_scale_kernel ()) in
  (* Narrowing goes through the documented intrinsic so the rounding mode is
     explicit and identical on CUDA and HIP. *)
  check_contains "narrowing" src "__float2half(" ;
  (* Widening rides __half's implicit conversion operator via a plain C cast. *)
  check_contains "widening" src "(float)" ;
  ()

let test_non_f16_kernel_unchanged () =
  let src = gen (f32_scale_kernel ()) in
  (* THE regression guard: a kernel with no f16 must be exactly as before, so
     every committed golden stays byte-identical. *)
  check_absent "no fp16 include in f32 kernel" src "cuda_fp16" ;
  check_absent "no __half in f32 kernel" src "__half" ;
  check_absent "no hip guard in f32 kernel" src "__HIP__" ;
  check_absent "no narrowing in f32 kernel" src "__float2half" ;
  (* And it still starts with the unmodified header. *)
  check_contains "extern C header" src "extern \"C\"" ;
  ()

let test_deferred_backends_reject_f16 () =
  let k = f16_scale_kernel () in
  let expect_raises name f =
    match f () with
    | (_ : string) ->
        Alcotest.failf
          "%s: expected f16 to be rejected (slice 2), but generation succeeded"
          name
    | exception _ -> ()
  in
  expect_raises "opencl" (fun () ->
      Sarek_ir_opencl.generate_with_types ~types:k.kern_types k) ;
  expect_raises "glsl" (fun () ->
      Sarek_ir_glsl.generate_with_types ~types:k.kern_types k) ;
  expect_raises "metal" (fun () ->
      Sarek_ir_metal.generate_with_types ~types:k.kern_types k) ;
  expect_raises "wgsl" (fun () ->
      Sarek_ir_wgsl.generate_with_types ~types:k.kern_types k) ;
  ()

let test_deferred_backends_still_accept_f32 () =
  (* The rejecting arms must be f16-specific — they must not have broken the
     backends they were added to. *)
  let k = f32_scale_kernel () in
  ignore (Sarek_ir_opencl.generate_with_types ~types:k.kern_types k) ;
  ignore (Sarek_ir_glsl.generate_with_types ~types:k.kern_types k) ;
  ignore (Sarek_ir_metal.generate_with_types ~types:k.kern_types k) ;
  ignore (Sarek_ir_wgsl.generate_with_types ~types:k.kern_types k) ;
  ()

let test_determinism () =
  let k = f16_scale_kernel () in
  let a = gen k in
  let b = gen k in
  Alcotest.(check string) "two generations agree" a b

let () =
  Alcotest.run
    "cuda_f16_golden"
    [
      ( "cuda_hip_f16",
        [
          Alcotest.test_case "type string is __half" `Quick test_f16_type_string;
          Alcotest.test_case
            "fp16 include emitted and HIP-guarded"
            `Quick
            test_f16_include_emitted;
          Alcotest.test_case
            "conversions use __float2half / C cast"
            `Quick
            test_f16_conversions;
          Alcotest.test_case
            "non-f16 kernel emits no f16 machinery"
            `Quick
            test_non_f16_kernel_unchanged;
          Alcotest.test_case
            "generation is deterministic"
            `Quick
            test_determinism;
        ] );
      ( "deferred_backends",
        [
          Alcotest.test_case
            "opencl/glsl/metal/wgsl reject f16"
            `Quick
            test_deferred_backends_reject_f16;
          Alcotest.test_case
            "opencl/glsl/metal/wgsl still accept f32"
            `Quick
            test_deferred_backends_still_accept_f32;
        ] );
    ]
