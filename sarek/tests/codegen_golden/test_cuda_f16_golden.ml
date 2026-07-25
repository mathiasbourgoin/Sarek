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

(** An f16 vector PARAMETER that the body never reads. This is the reviewer's
    exact case: PTX validates aggregate vector element types through a
    [| _ -> ()] fall-through, and with no f16 expression in the body nothing
    else ever asks for an f16 type string — so PTX emitted a complete, valid,
    silently-wrong module with no diagnostic at all. *)
let f16_untouched_param_kernel () =
  let out = make_var "out" (TVec TFloat32) in
  let unused = make_var "hin" (TVec TFloat16) in
  let idx = make_var "idx" TInt32 in
  let body =
    SLet
      ( idx,
        EIntrinsic ([], "global_thread_id", []),
        SAssign (LArrayElem ("out", EVar idx), EConst (CFloat32 1.0)) )
  in
  mk_kernel
    "f16_untouched_param"
    [
      DParam (out, Some {arr_elttype = TFloat32; arr_memspace = Global});
      DParam (unused, Some {arr_elttype = TFloat16; arr_memspace = Global});
    ]
    body

(** An f16 kernel whose ONLY f16 is a local binder. GLSL and WGSL never iterate
    [kern_locals], so no per-type arm can fire here — only the whole-kernel gate
    can. *)
let f16_local_only_kernel () =
  let out = make_var "out" (TVec TFloat32) in
  let idx = make_var "idx" TInt32 in
  let h = make_var "h" TFloat16 in
  let body =
    SLet
      ( idx,
        EIntrinsic ([], "global_thread_id", []),
        SLet
          ( h,
            ECast (TFloat16, EConst (CFloat32 1.5)),
            SAssign (LArrayElem ("out", EVar idx), ECast (TFloat32, EVar h)) )
      )
  in
  let k =
    mk_kernel
      "f16_local_only"
      [DParam (out, Some {arr_elttype = TFloat32; arr_memspace = Global})]
      body
  in
  {k with kern_locals = [DLocal (h, None)]}

(** An f16 kernel whose only f16 is a HELPER RETURN TYPE. PTX never inspected
    [hf_ret_type], so it emitted a complete valid module with no diagnostic. *)
let f16_helper_ret_kernel () =
  let out = make_var "out" (TVec TFloat32) in
  let idx = make_var "idx" TInt32 in
  let helper =
    {
      hf_name = "narrow";
      hf_params = [make_var "x" TFloat32];
      hf_ret_type = TFloat16;
      hf_body = SReturn (ECast (TFloat16, EVar (make_var "x" TFloat32)));
    }
  in
  let body =
    SLet
      ( idx,
        EIntrinsic ([], "global_thread_id", []),
        SAssign (LArrayElem ("out", EVar idx), EConst (CFloat32 0.0)) )
  in
  let k =
    mk_kernel
      "f16_helper_ret"
      [DParam (out, Some {arr_elttype = TFloat32; arr_memspace = Global})]
      body
  in
  {k with kern_funcs = [helper]}

(** Every backend that defers f16 must REJECT it with its own located
    diagnostic, at every entry point, for every position f16 can occupy.

    The exception is matched precisely: the previous [| exception _ -> ()]
    accepted ANY failure, so a Not_found or a Stack_overflow from an unrelated
    bug would have read as "correctly rejected". *)
let expect_f16_rejected ?tag ~backend ~expected_reason name f =
  let tag_expected = Option.value tag ~default:backend in
  match f () with
  | (_ : string) ->
      Alcotest.failf
        "%s: expected f16 to be rejected (slice 2), but generation succeeded"
        name
  | exception
      Sarek_backend_error.Backend_error.Backend_error
        (Sarek_backend_error.Backend_error.Codegen
           {
             backend = actual_tag;
             error =
               Sarek_backend_error.Backend_error.Unsupported_construct
                 {construct; reason};
           }) ->
      (* The diagnostic must name the backend, the construct ("f16") and the
         slice-2 deferral. Matching the exception SHAPE (not [_]) is the point:
         the previous [| exception _ -> ()] would have passed on any unrelated
         failure. *)
      Alcotest.(check string) (name ^ ": backend tag") tag_expected actual_tag ;
      Alcotest.(check string) (name ^ ": construct") "f16" construct ;
      (* The reason is compared EXACTLY, not by substring. All four of these are
         now composed by the single shared Sarek_ir_codegen.reject_feature, so
         pinning the whole string is what stops a future edit to that one helper
         from silently rewording every backend's diagnostic at once. *)
      Alcotest.(check string) (name ^ ": reason") expected_reason reason
  | exception e ->
      Alcotest.failf
        "%s: rejected with the WRONG exception (expected Codegen_error): %s"
        name
        (Printexc.to_string e)

(* Composed by the single shared Sarek_ir_codegen.reject_feature:
   "<backend>: <width> not yet supported (#57 slice 2[ — <hint>])".
   Metal omits the hint (its arm is a one-liner once it can be tested). *)
let reason_opencl =
  "OpenCL: float16 not yet supported (#57 slice 2 — needs cl_khr_fp16 \
   enablement)"

let reason_glsl =
  "GLSL: float16 not yet supported (#57 slice 2 — needs \
   GL_EXT_shader_explicit_arithmetic_types_float16)"

let reason_metal = "Metal: float16 not yet supported (#57 slice 2)"

let reason_wgsl =
  "WGSL: float16 not yet supported (#57 slice 2 — needs a module-top `enable \
   f16;` directive)"

let test_deferred_backends_reject_f16 () =
  let each label k =
    expect_f16_rejected
      ~backend:"OpenCL"
      ~expected_reason:reason_opencl
      (label ^ "/opencl generate")
      (fun () -> Sarek_ir_opencl.generate k) ;
    expect_f16_rejected
      ~backend:"OpenCL"
      ~expected_reason:reason_opencl
      (label ^ "/opencl generate_with_types")
      (fun () -> Sarek_ir_opencl.generate_with_types ~types:k.kern_types k) ;
    (* GLSL's Backend_error tag is "Vulkan" (that is the framework name). *)
    expect_f16_rejected
      ~tag:"Vulkan"
      ~backend:"GLSL"
      ~expected_reason:reason_glsl
      (label ^ "/glsl generate")
      (fun () -> Sarek_ir_glsl.generate k) ;
    expect_f16_rejected
      ~tag:"Vulkan"
      ~backend:"GLSL"
      ~expected_reason:reason_glsl
      (label ^ "/glsl generate_with_types")
      (fun () -> Sarek_ir_glsl.generate_with_types ~types:k.kern_types k) ;
    expect_f16_rejected
      ~backend:"Metal"
      ~expected_reason:reason_metal
      (label ^ "/metal generate")
      (fun () -> Sarek_ir_metal.generate k) ;
    expect_f16_rejected
      ~backend:"Metal"
      ~expected_reason:reason_metal
      (label ^ "/metal generate_with_types")
      (fun () -> Sarek_ir_metal.generate_with_types ~types:k.kern_types k) ;
    (* Likewise WGSL's tag is the framework name "WebGPU". *)
    expect_f16_rejected
      ~tag:"WebGPU"
      ~backend:"WGSL"
      ~expected_reason:reason_wgsl
      (label ^ "/wgsl generate")
      (fun () -> Sarek_ir_wgsl.generate k) ;
    expect_f16_rejected
      ~tag:"WebGPU"
      ~backend:"WGSL"
      ~expected_reason:reason_wgsl
      (label ^ "/wgsl generate_with_types")
      (fun () -> Sarek_ir_wgsl.generate_with_types ~types:k.kern_types k)
  in
  each "vector-param" (f16_scale_kernel ()) ;
  each "untouched-param" (f16_untouched_param_kernel ()) ;
  each "local-only" (f16_local_only_kernel ()) ;
  each "helper-return" (f16_helper_ret_kernel ())

(** PTX raises its own [Ptx_codegen_error], not [Codegen_error]. Before the
    whole-kernel gate, BOTH of these emitted complete valid PTX with no
    diagnostic at all: the f16 vector param fell through a [| _ -> ()] and the
    helper return type was never inspected. *)
let test_ptx_rejects_f16 () =
  let each label k =
    match Sarek_ir_ptx.generate k with
    | (ptx : string) ->
        Alcotest.failf
          "PTX %s: expected f16 to be rejected (slice 2), but it emitted:\n%s"
          label
          ptx
    | exception Sarek_ir_ptx_types.Ptx_codegen_error msg ->
        if not (contains ~needle:"float16 not supported by the PTX backend" msg)
        then Alcotest.failf "PTX %s: unexpected diagnostic: %s" label msg
    | exception e ->
        Alcotest.failf
          "PTX %s: rejected with the WRONG exception: %s"
          label
          (Printexc.to_string e)
  in
  each "vector-param" (f16_scale_kernel ()) ;
  each "untouched-param" (f16_untouched_param_kernel ()) ;
  each "local-only" (f16_local_only_kernel ()) ;
  each "helper-return" (f16_helper_ret_kernel ())

let test_deferred_backends_still_accept_f32 () =
  (* The rejecting arms must be f16-specific — they must not have broken the
     backends they were added to. Both entry points, plus PTX. *)
  let k = f32_scale_kernel () in
  ignore (Sarek_ir_opencl.generate k) ;
  ignore (Sarek_ir_opencl.generate_with_types ~types:k.kern_types k) ;
  ignore (Sarek_ir_glsl.generate k) ;
  ignore (Sarek_ir_glsl.generate_with_types ~types:k.kern_types k) ;
  ignore (Sarek_ir_metal.generate k) ;
  ignore (Sarek_ir_metal.generate_with_types ~types:k.kern_types k) ;
  ignore (Sarek_ir_wgsl.generate k) ;
  ignore (Sarek_ir_wgsl.generate_with_types ~types:k.kern_types k) ;
  ignore (Sarek_ir_ptx.generate k) ;
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
            "opencl/glsl/metal/wgsl reject f16 (param, local, helper return)"
            `Quick
            test_deferred_backends_reject_f16;
          Alcotest.test_case
            "ptx rejects f16 (param, local, helper return)"
            `Quick
            test_ptx_rejects_f16;
          Alcotest.test_case
            "opencl/glsl/metal/wgsl still accept f32"
            `Quick
            test_deferred_backends_still_accept_f32;
        ] );
    ]
