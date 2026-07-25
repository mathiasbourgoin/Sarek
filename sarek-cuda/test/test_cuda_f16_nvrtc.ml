(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * f16 NVRTC compile gate (#57 slice 1 review MF1).
 *
 * WHY THIS EXISTS: test_cuda_f16_golden asserts SUBSTRINGS on the generated
 * CUDA source ("__half*", "#include <cuda_fp16.h>", "__float2half("). Those
 * substrings were all present and correct, and the kernel still could not be
 * JIT-compiled: nvrtc has no default include path, so the emitted
 * `#include <cuda_fp16.h>` did not resolve and every f16 CUDA kernel failed
 * with NVRTC_ERROR_COMPILATION. A substring assertion cannot see that. This
 * test COMPILES the generated source through libnvrtc and inspects the PTX.
 *
 * No NVIDIA device is required: nvrtc runs entirely host-side.
 *
 * Skips cleanly (mirroring the ptxas gate in sarek/tests/unit/test_ptx_snapshot)
 * when libnvrtc cannot be dlopen'd or when no CUDA include directory holding
 * cuda_fp16.h can be found.
 ******************************************************************************)

open Sarek_ir_types

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

(** The same "storage type, compute in f32" kernel the golden test asserts on:
    read an f16 element, widen to f32, double it, narrow back on store. *)
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

(** An f32 kernel: the negative control that proves the gate is not vacuous and
    that the include path did not break ordinary compilation. *)
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

let gen k =
  Sarek_codegen.Sarek_ir_cuda.current_framework := None ;
  Sarek_codegen.Sarek_ir_cuda.current_variants := [] ;
  Sarek_codegen.Sarek_ir_cuda.generate_with_types ~types:k.kern_types k

let contains ~needle haystack =
  let nl = String.length needle and hl = String.length haystack in
  let rec go i =
    i + nl <= hl && (String.sub haystack i nl = needle || go (i + 1))
  in
  nl = 0 || go 0

(* ------------------------------------------------------------------ *)
(* Skip predicate                                                     *)
(* ------------------------------------------------------------------ *)

let nvrtc_ready () =
  Sarek_cuda.Cuda_nvrtc.is_available ()
  && Lazy.force Sarek_cuda.Cuda_nvrtc.cuda_include_paths <> []

let skip_reason () =
  if not (Sarek_cuda.Cuda_nvrtc.is_available ()) then "libnvrtc not loadable"
  else "no CUDA include directory containing cuda_fp16.h"

(* ------------------------------------------------------------------ *)
(* Tests                                                              *)
(* ------------------------------------------------------------------ *)

(** Compile [src] and return the PTX, or the nvrtc failure as an Error. *)
let compile src =
  match
    Sarek_cuda.Cuda_nvrtc.compile_to_ptx ~name:"f16_gate" ~arch:"compute_75" src
  with
  | ptx -> Ok ptx
  | exception e -> Error (Printexc.to_string e)

let test_f16_source_compiles () =
  if not (nvrtc_ready ()) then begin
    Printf.printf "  [SKIP] f16 nvrtc compile: %s\n" (skip_reason ()) ;
    Alcotest.skip ()
  end
  else
    let src = gen (f16_scale_kernel ()) in
    (* Precondition: this really is the f16 source shape under test. *)
    if not (contains ~needle:"#include <cuda_fp16.h>" src) then
      Alcotest.failf
        "generated f16 source no longer emits the fp16 include:\n%s"
        src ;
    match compile src with
    | Error err ->
        Alcotest.failf
          "NVRTC rejected the generated f16 kernel: %s\n--- source ---\n%s"
          err
          src
    | Ok ptx ->
        (* The f16 narrowing must actually be in the PTX. This is what a
           substring test on the CUDA source can never establish: it proves the
           header resolved, __half was a real type, and __float2half lowered to
           a hardware conversion. *)
        if not (contains ~needle:"cvt.rn.f16.f32" ptx) then
          Alcotest.failf
            "f16 kernel compiled but PTX has no f16 conversion (cvt.rn.f16.f32):\n\
             %s"
            ptx

let test_f32_source_still_compiles () =
  if not (Sarek_cuda.Cuda_nvrtc.is_available ()) then begin
    Printf.printf "  [SKIP] f32 nvrtc compile: libnvrtc not loadable\n" ;
    Alcotest.skip ()
  end
  else
    let src = gen (f32_scale_kernel ()) in
    match compile src with
    | Error err ->
        Alcotest.failf "NVRTC rejected a plain f32 kernel: %s\n%s" err src
    | Ok ptx ->
        (* Discrimination: no f16 machinery leaked into the f32 kernel. *)
        if contains ~needle:"cvt.rn.f16.f32" ptx then
          Alcotest.fail "f32 kernel PTX unexpectedly contains an f16 conversion"

let test_include_paths_exist () =
  (* The discovery must never hand nvrtc a nonexistent directory. *)
  List.iter
    (fun d ->
      if not (Sys.file_exists d) then
        Alcotest.failf "discovered CUDA include path does not exist: %s" d)
    (Lazy.force Sarek_cuda.Cuda_nvrtc.cuda_include_paths)

let () =
  Alcotest.run
    "cuda_f16_nvrtc"
    [
      ( "nvrtc_compile_gate",
        [
          Alcotest.test_case
            "generated f16 CUDA source compiles and PTX has cvt.rn.f16.f32"
            `Quick
            test_f16_source_compiles;
          Alcotest.test_case
            "plain f32 kernel still compiles"
            `Quick
            test_f32_source_still_compiles;
          Alcotest.test_case
            "discovered include paths exist"
            `Quick
            test_include_paths_exist;
        ] );
    ]
