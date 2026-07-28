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
  {default_kernel with kern_name = name; kern_params = params; kern_body = body}

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

(** First index of [needle] at or after [from], or [-1]. *)
let index_from ~needle ~from haystack =
  let nl = String.length needle and hl = String.length haystack in
  let rec go i =
    if i + nl > hl then -1
    else if String.sub haystack i nl = needle then i
    else go (i + 1)
  in
  go (max 0 from)

let index_of ~needle haystack = index_from ~needle ~from:0 haystack

(** Drop C block- and line-comments, so a structural assertion about emitted
    CODE cannot be satisfied — or broken — by prose. Without this, an assertion
    that a preprocessor arm contains no ["volatile"] goes red the day someone
    writes the word in the comment explaining why there is none. *)
let strip_c_comments src =
  let n = String.length src in
  let buf = Buffer.create n in
  let rec go i =
    if i >= n then ()
    else if i + 1 < n && src.[i] = '/' && src.[i + 1] = '*' then
      let rec close j =
        if j + 1 >= n then n
        else if src.[j] = '*' && src.[j + 1] = '/' then j + 2
        else close (j + 1)
      in
      go (close (i + 2))
    else if i + 1 < n && src.[i] = '/' && src.[i + 1] = '/' then
      let rec eol j = if j >= n || src.[j] = '\n' then j else eol (j + 1) in
      go (eol (i + 2))
    else (
      Buffer.add_char buf src.[i] ;
      go (i + 1))
  in
  go 0 ;
  Buffer.contents buf

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
  (* CONFORMANCE, not cosmetics: every narrowing must go through the f32
     barrier. Without it the AMDGPU backend fuses the producing f32 op into the
     narrowing (v_fma_mixlo_f16 / v_add_f16) and skips a mandated rounding — 620
     of 63488 binary16 inputs came back wrong on gfx1100. A plain
     [check_contains "__float2half("] still matches the barriered form, so it
     would NOT catch a regression here; these two assertions are what do. *)
  check_contains
    "narrowing goes through the f32 barrier"
    src
    "__float2half(sarek_f32_barrier(" ;
  check_contains
    "barrier is declared"
    src
    "__device__ __forceinline__ float sarek_f32_barrier(float x)" ;
  (* AMDGPU is where the barrier is load-bearing: the "v" constraint pins the
     value in a VGPR ahead of the ISel combine that fuses. *)
  check_contains "AMDGPU constraint" src "asm volatile(\"\" : \"+v\"(x))" ;
  (* NVIDIA is where it is NOT, and that asymmetry is asserted rather than left
     to a reader's assumption (#110), and it is scoped to the NARROWING: the
     non-HIP branch used to carry a PTX "+f" variant, which contributes zero PTX
     instructions there, so ptxas sees an identical instruction stream and the
     cubins were byte-identical with and without it — measured on CUDA 13.3
     (nvcc/ptxas/nvdisasm V13.3.73, host-side, no NVIDIA device) for sm_75
     through sm_121. Keeping a no-op that reads as protection is what this
     assertion forbids. What actually keeps the multiply out of the narrowing
     on NVIDIA is ptxas, machine-checked by
     sarek-cuda/test/test_cuda_f16_sass.ml. NOTE the same barrier is NOT inert
     at a mul->add site (PTX mul.f32+add.f32 instead of fma.rn.f32) — but ptxas
     re-contracts that under the default -fmad=true, so it is still not a usable
     contraction barrier on NVIDIA. See docs/fp-contraction-policy.md. *)
  check_absent "no inert PTX barrier" src "\"+f\"" ;
  check_contains
    "NVIDIA branch documents why it is empty"
    src
    "NVIDIA: intentionally an identity" ;
  ()

(** The opacity barrier must be SCOPED to the AMD toolchain, not merely present
    (backlog #144).

    [test_f16_conversions] asserts the AMDGPU [asm volatile] appears and that no
    PTX ["+f"] variant does. Neither assertion looks at WHERE the asm sits, so
    both survive the mutation that matters here: widening the barrier so it is
    also emitted on the non-AMD arm.

    Why that mutation is a defect and not a harmless over-approximation.
    Measured on Intel Arc Graphics (Meteor Lake-P, Intel Compute Runtime / IGC),
    2026-07-27, exhaustive over all 63488 finite binary16 inputs
    (docs/fp-contraction-policy.md §11.3 / §11.4): the naive narrowing is
    correct — 0/63488 against the host binary16 reference, with the [fusedctl]
    positive control reproducing ACO's 620/63488 on the same device and run —
    and every volatile-based barrier makes it WRONG on 4774/63488. IGC folds the
    [f32(f16(x))] pair across the volatile boundary, a fold valid only when the
    value is exactly representable in binary16. So on that toolchain the barrier
    is not redundant, it is the defect.

    IGC cannot receive this particular source today: Sarek's OpenCL, GLSL, Metal
    and WGSL backends all refuse f16 outright, so the only compilers that ever
    see [sarek_f32_barrier] are hiprtc and nvrtc, and the preprocessor tells
    those two apart with certainty. That is a STRUCTURAL argument, and
    structural arguments are what this repository keeps discovering it had
    stopped re-checking. This case is the re-check: it pins that the opacity
    body lives inside the AMD arm and that the other arm is a bare identity, so
    a future maintainer who "simplifies" the [#if] away, or who adds a barrier
    on the NVIDIA arm on the strength of an AMD measurement, gets a red rather
    than a silently portable-looking one.

    Extended for #146 with the third thing the guard has to get right, after
    "the asm is inside the AMD arm" and "the other arm is bare": WHICH MACRO the
    arm is keyed on. [__HIP__] is a language predicate and constrains no target;
    [__HIP_PLATFORM_AMD__] names the target that makes ["+v"] legal. The guard
    carried both as a disjunction, and the redundant one is what made two
    successive readers conclude the barrier ships to NVPTX.

    Evidence tier for the IGC figures: executed (Intel Arc, IGC). For "the
    preprocessor is the right discriminator": by-construction. For ["+v"] being
    invalid on NVPTX: executed (clang 22.1.6, [--target=nvptx64-nvidia-cuda]
    rejects it, accepts ["+f"]). *)
let test_f16_barrier_is_amd_scoped () =
  let src = gen (f16_scale_kernel ()) in
  let guard = "#if defined(__HIP_PLATFORM_AMD__)" in
  let i_if = index_of ~needle:guard src in
  if i_if < 0 then
    Alcotest.failf
      "the f32 barrier must be emitted under the AMD toolchain guard %S; it \
       was not found at all, so nothing scopes the opacity body:\n\
       %s"
      guard
      src ;
  (* The guard names the PLATFORM, never the LANGUAGE (#146). [__HIP__] says
     "this translation unit is HIP", which constrains no target ISA;
     [__HIP_PLATFORM_AMD__] says "the target is AMD", which is the only thing
     that makes the "+v" VGPR constraint legal. Measured: clang rejects "+v" for
     --target=nvptx64-nvidia-cuda ("invalid output constraint '+v' in asm") and
     accepts "+f". The two are NOT interchangeable keys, even though HIP's own
     hip_common.h currently makes __HIP__ imply __HIP_PLATFORM_AMD__ under
     clang — that implication is a header's behaviour, not a language rule, and
     it is not what this barrier should be resting on. Asserted on the guard
     LINE rather than on the whole source because the f16 include above
     legitimately tests __HIP__ (there the question really is "is this HIP",
     and both arms fail loudly). *)
  let guard_line =
    let e = try String.index_from src i_if '\n' with Not_found -> i_if in
    String.sub src i_if (e - i_if)
  in
  if contains ~needle:"defined(__HIP__)" guard_line then
    Alcotest.failf
      "the barrier guard must not key on __HIP__: it is a LANGUAGE predicate \
       and admits any HIP target, while the asm body is AMDGPU-specific (clang \
       rejects \"+v\" for nvptx64 with \"invalid output constraint\"). Use \
       __HIP_PLATFORM_AMD__, which names the target. Guard line was:\n\
       %s"
      guard_line ;
  let i_else = index_from ~needle:"#else" ~from:i_if src in
  let i_endif = index_from ~needle:"#endif" ~from:i_if src in
  if not (i_else > i_if && i_endif > i_else) then
    Alcotest.failf
      "the AMD guard must be a two-armed #if/#else/#endif (if@%d, else@%d, \
       endif@%d):\n\
       %s"
      i_if
      i_else
      i_endif
      src ;
  (* The opacity body is inside the AMD arm. *)
  let i_asm = index_from ~needle:"asm volatile" ~from:i_if src in
  if not (i_asm > i_if && i_asm < i_else) then
    Alcotest.failf
      "the AMDGPU opacity barrier must sit INSIDE the AMD arm of the guard \
       (if@%d, asm@%d, else@%d). Emitting it outside ships to every toolchain \
       the barrier was never measured on — and on Intel IGC that same barrier \
       turns a correct narrowing into 4774/63488 wrong answers \
       (docs/fp-contraction-policy.md §11.4):\n\
       %s"
      i_if
      i_asm
      i_else
      src ;
  (* And the non-AMD arm carries no barrier of any kind. Comments are stripped
     first: the arm's whole purpose is a comment explaining why there is no
     barrier, and that prose must not be able to satisfy or break a check about
     code. *)
  let non_amd_arm =
    strip_c_comments (String.sub src i_else (i_endif - i_else))
  in
  List.iter
    (fun needle ->
      if contains ~needle non_amd_arm then
        Alcotest.failf
          "the non-AMD arm of the barrier guard must be a bare identity, but \
           it contains %S. A barrier here is not measured-neutral: it is \
           measured-HARMFUL on Intel IGC (4774/63488) and \
           measured-zero-instruction on NVIDIA (byte-identical cubins, \
           sm_75..sm_121). Arm was:\n\
           %s"
          needle
          non_amd_arm)
    ["asm"; "volatile"] ;
  (* Non-vacuity: the arm we just proved empty must be the one that really
     carries the NVIDIA identity, otherwise the substring above could be empty
     for an unrelated reason. *)
  check_contains
    "the non-AMD arm is the NVIDIA identity"
    non_amd_arm
    "return x;" ;
  ()

let test_non_f16_kernel_unchanged () =
  let src = gen (f32_scale_kernel ()) in
  (* THE regression guard: a kernel with no f16 must be exactly as before, so
     every committed golden stays byte-identical. *)
  check_absent "no fp16 include in f32 kernel" src "cuda_fp16" ;
  check_absent "no __half in f32 kernel" src "__half" ;
  check_absent "no hip guard in f32 kernel" src "__HIP__" ;
  check_absent "no narrowing in f32 kernel" src "__float2half" ;
  (* The barrier costs 4 extra VALU ops per f16 round-trip, so it must be gated
     on the kernel actually using f16. Its absence here is what makes that
     zero-cost claim for non-f16 kernels checkable rather than asserted. *)
  check_absent "no f32 barrier in f32 kernel" src "sarek_f32_barrier" ;
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

(* GLSL, Metal and WGSL are composed by the single shared
   Sarek_ir_codegen.reject_feature:
   "<backend>: <width> not yet supported (#57 slice 2[ — <hint>])".
   Metal omits the hint (its arm is a one-liner once it can be tested).

   OpenCL deliberately does NOT use that composer any more (#57 slice 2a), so
   its expectation is spelled out separately here on purpose. Pinning it
   verbatim is the point: the shared wording says "not YET supported" and
   blamed cl_khr_fp16, and both were measured false — cl_khr_fp16 is advertised
   and usable on both local devices, and the real blocker is that
   rusticl/radeonsi fuses the f32 multiply into the f32->f16 narrowing
   (620/63488 binary16 inputs disagree with the interpreter) with no affordable
   barrier available. If someone later re-folds OpenCL back into the shared
   composer, this assertion fails and forces them to re-read
   docs/fp-contraction-policy.md first. *)
let reason_opencl =
  "OpenCL: float16 is refused by measurement, not pending implementation — \
   rusticl/radeonsi fuses the f32 multiply into the f32->f16 narrowing, so \
   620/63488 binary16 inputs disagree with the interpreter, and no affordable \
   barrier exists on this path. See docs/fp-contraction-policy.md (#57 slice \
   2a)."

(* GLSL left the shared composer for the same reason OpenCL did (#57 slice 2b),
   and is pinned verbatim here for the same reason: the shared wording said "not
   YET supported" and blamed
   GL_EXT_shader_explicit_arithmetic_types_float16. Both were measured false —
   the extension compiles and runs on both local RADV devices, and the real
   blocker is that ACO absorbs the f32->f16 narrowing into the arithmetic
   feeding it. Re-folding GLSL back into the composer breaks this assertion,
   which is the intent. *)
let reason_glsl =
  "GLSL: float16 is refused by measurement, not pending implementation — \
   RADV's ACO backend absorbs the f32->f16 narrowing into the arithmetic that \
   feeds it, so 2912/63488 binary16 inputs disagree with the interpreter on a \
   single narrowing, and `precise` does not prevent it. See \
   docs/fp-contraction-policy.md (#57 slice 2b)."

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
    (* OpenCL has a THIRD public entry point. It delegates to [generate], so it
       is covered transitively today — but "covered transitively" is exactly how
       this repo's previous guard holes were argued, and THIS refusal is what
       keeps Sarek's f16 narrowing away from IGC, where the ACO barrier is
       measured to BREAK a correct narrowing (4774/63488, Intel Arc /
       Meteor Lake-P, docs/fp-contraction-policy.md §11.4). Asserted directly so
       a future non-delegating implementation cannot reopen it. *)
    expect_f16_rejected
      ~backend:"OpenCL"
      ~expected_reason:reason_opencl
      (label ^ "/opencl generate_with_fp64")
      (fun () -> Sarek_ir_opencl.generate_with_fp64 k) ;
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

(* ---------------------------------------------------------------------------
   #138: one measured claim, one copy of it.

   OpenCL and GLSL each refuse f16 in TWO places — the per-element-type arm of
   [<backend>_type_of_elttype] and the whole-kernel [reject_float16_kernel] —
   and each had its own copy of the sentence. Those sentences are not prose:
   they carry the MEASUREMENT that justifies the refusal (620/63488 on OpenCL
   via rusticl, 2912-5075/63488 on GLSL via RADV, both ACO), and a measurement
   stated twice is a measurement that will eventually be stated two ways.

   It already had been. Before this change the OpenCL pair read:

     elttype arm: "OpenCL: float16 is not supported — not because the codegen
                   is missing, but because rusticl/radeonsi fuses ..."
     kernel arm:  "OpenCL: float16 is refused by measurement, not pending
                   implementation — rusticl/radeonsi fuses ..."

   Same defect, same numbers, two different openings — and only one of them is
   the wording the docs and this file's [reason_opencl] pin. That is the drift
   the shared constant removes, and this case is what would have caught it.

   It also guards the direction that matters next: when one backend's fusion is
   fixed upstream and its refusal is lifted, a single constant makes every
   remaining reference obviously stale, whereas duplicates just rot. *)

let reason_of_exn name f =
  match f () with
  | (_ : string) ->
      Alcotest.failf "%s: expected an f16 refusal, but it succeeded" name
  | exception
      Sarek_backend_error.Backend_error.Backend_error
        (Sarek_backend_error.Backend_error.Codegen
           {
             backend = _;
             error =
               Sarek_backend_error.Backend_error.Unsupported_construct
                 {construct = _; reason};
           }) ->
      reason
  | exception e ->
      Alcotest.failf
        "%s: wrong exception (expected Codegen): %s"
        name
        (Printexc.to_string e)

(* Both refusal sites of one backend must produce the SAME string. Compared
   exactly, and compared against the constant the docs cite, so neither site can
   drift and neither can drift *together* away from [reason_opencl] /
   [reason_glsl] above. *)
let check_one_refusal_text ~backend ~expected ~from_elttype ~from_kernel =
  let a = reason_of_exn (backend ^ "/elttype arm") from_elttype in
  let b = reason_of_exn (backend ^ "/kernel arm") from_kernel in
  Alcotest.(check string)
    (backend
   ^ ": the per-element-type arm and the whole-kernel arm state the SAME \
      measured reason")
    b
    a ;
  Alcotest.(check string)
    (backend ^ ": that reason is the one the docs and goldens pin")
    expected
    a

let test_f16_refusal_text_is_single_sourced () =
  let k = f16_scale_kernel () in
  check_one_refusal_text
    ~backend:"OpenCL"
    ~expected:reason_opencl
    ~from_elttype:(fun () ->
      Sarek_codegen.Sarek_ir_opencl.opencl_type_of_elttype TFloat16)
    ~from_kernel:(fun () -> Sarek_codegen.Sarek_ir_opencl.generate k) ;
  check_one_refusal_text
    ~backend:"GLSL"
    ~expected:reason_glsl
    ~from_elttype:(fun () ->
      Sarek_codegen.Sarek_ir_glsl.glsl_type_of_elttype TFloat16)
    ~from_kernel:(fun () -> Sarek_codegen.Sarek_ir_glsl.generate k)

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
            "f32 barrier is scoped to the AMD toolchain arm (#144)"
            `Quick
            test_f16_barrier_is_amd_scoped;
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
          Alcotest.test_case
            "each backend's f16 refusal text has exactly one source (#138)"
            `Quick
            test_f16_refusal_text_is_single_sourced;
        ] );
    ]
