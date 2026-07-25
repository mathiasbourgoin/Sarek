(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * FP-conformance guard for the CUDA/nvrtc path (#111).
 *
 * WHAT IS BEING GUARDED. Sarek's DSL semantics are IEEE-754 binary32 with every
 * operation rounded as written, and the INTERPRETER is the cross-backend oracle
 * that defines them (docs/fp-contraction-policy.md). [-use_fast_math] and
 * [-ftz=true] flush binary32 subnormals; [--prec-div=false] / [--prec-sqrt=false]
 * downgrade division and square root. None of those can be undone by a later
 * flag, so [Cuda_nvrtc] refuses them instead of warning.
 *
 * NON-VACUITY. Three separate things keep this from being a test of a constant:
 *
 *   1. NEGATIVE SIDE. Options that must stay accepted are asserted accepted --
 *      including [-ftz=false] and [--prec-div=true], which differ from the
 *      rejected forms only in the VALUE. A guard that matched on the option
 *      name alone would pass every rejection case and fail here.
 *   2. END-TO-END. [compile_to_ptx ~options:["-use_fast_math"]] must raise. The
 *      guard runs before libnvrtc is touched, so this fires on a host with no
 *      CUDA installed -- the check is on the real entry point, not only on the
 *      predicate.
 *   3. THE HAZARD IS REAL, MEASURED HERE. When nvcc and nvdisasm are on PATH,
 *      the generated f16 kernel is compiled twice and the SASS compared: with
 *      [-ftz=true] the binary32 arithmetic must acquire [.FTZ], without it must
 *      not. That is the divergence the guard exists to prevent, observed rather
 *      than quoted. Skips cleanly with no CUDA toolchain; host tools only, no
 *      NVIDIA device required.
 ******************************************************************************)

open Sarek_ir_types
module Nvrtc = Sarek_cuda.Cuda_nvrtc

(* ------------------------------------------------------------------ *)
(* The kernel under measurement (same shape as test_cuda_f16_sass)     *)
(* ------------------------------------------------------------------ *)

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

let f16_midround_kernel () =
  let out = make_var "out" (TVec TFloat16) in
  let inp = make_var "inp" (TVec TFloat16) in
  let idx = make_var "idx" TInt32 in
  let narrowed_product =
    ECast
      ( TFloat16,
        EBinop
          ( Mul,
            ECast (TFloat32, EArrayRead ("inp", EVar idx)),
            EConst (CFloat32 1.1) ) )
  in
  let body =
    SLet
      ( idx,
        EIntrinsic ([], "global_thread_id", []),
        SAssign
          ( LArrayElem ("out", EVar idx),
            ECast
              ( TFloat16,
                EBinop
                  ( Add,
                    ECast (TFloat32, narrowed_product),
                    EConst (CFloat32 1000.0) ) ) ) )
  in
  mk_kernel
    "f16_midround"
    [
      DParam (out, Some {arr_elttype = TFloat16; arr_memspace = Global});
      DParam (inp, Some {arr_elttype = TFloat16; arr_memspace = Global});
    ]
    body

let generated_source () =
  let k = f16_midround_kernel () in
  Sarek_codegen.Sarek_ir_cuda.current_framework := None ;
  Sarek_codegen.Sarek_ir_cuda.current_variants := [] ;
  Sarek_codegen.Sarek_ir_cuda.generate_with_types ~types:k.kern_types k

(* ------------------------------------------------------------------ *)
(* 1. Rejection                                                        *)
(* ------------------------------------------------------------------ *)

(* Every spelling nvcc/nvrtc accepts for the same hazard. Both dash forms,
   because nvrtc takes either. *)
let must_reject =
  [
    ("-use_fast_math", "subnormal");
    ("--use_fast_math", "subnormal");
    ("-ftz=true", "subnormal");
    ("--ftz=true", "subnormal");
    ("--ftz=1", "subnormal");
    ("--prec-div=false", "division");
    ("-prec-div=0", "division");
    ("--prec-sqrt=false", "square root");
  ]

let test_rejected () =
  List.iter
    (fun (opt, needle) ->
      match Nvrtc.fp_rejection_reason opt with
      | None ->
          Alcotest.failf
            "%S must be refused on the CUDA path; the guard accepted it"
            opt
      | Some msg ->
          (* Not merely "some error": the message must name the option and
             explain the hazard, or a caller cannot act on it. *)
          let contains hay needle =
            let n = String.length needle and h = String.length hay in
            let rec go i =
              i + n <= h && (String.sub hay i n = needle || go (i + 1))
            in
            n = 0 || go 0
          in
          if not (contains msg opt) then
            Alcotest.failf
              "rejection message for %S does not quote it: %s"
              opt
              msg ;
          if not (contains msg needle) then
            Alcotest.failf
              "rejection message for %S does not explain the hazard (expected \
               %S): %s"
              opt
              needle
              msg ;
          if not (contains msg "docs/fp-contraction-policy.md") then
            Alcotest.failf
              "rejection message for %S does not point at the policy: %s"
              opt
              msg)
    must_reject ;
  Printf.printf
    "  guard rejects %d FP-relaxing option spellings, each with a reason\n"
    (List.length must_reject)

(* ------------------------------------------------------------------ *)
(* 2. Acceptance — the half that makes rejection non-trivial           *)
(* ------------------------------------------------------------------ *)

let must_accept =
  [
    "--gpu-architecture=compute_75";
    "--include-path=/opt/cuda/include";
    "-I/opt/cuda/include";
    "-DFOO=1";
    "-ftz=false";
    "--ftz=false";
    "--prec-div=true";
    "--prec-sqrt=true";
    "-O3";
    "--extra-device-vectorization";
    "--std=c++17";
  ]

let test_accepted () =
  List.iter
    (fun opt ->
      match Nvrtc.fp_rejection_reason opt with
      | Some msg ->
          Alcotest.failf
            "%S is legitimate and must be accepted; the guard refused it: %s"
            opt
            msg
      | None -> ())
    must_accept ;
  (* --fmad=true is a real relaxation but is nvrtc's default, so it warns
     rather than rejects. Both halves matter: it must NOT reject, and it must
     NOT be silent. *)
  (match Nvrtc.fp_rejection_reason "--fmad=true" with
  | Some msg -> Alcotest.failf "--fmad=true must warn, not reject: %s" msg
  | None -> ()) ;
  (match Nvrtc.fp_warning_reason "--fmad=true" with
  | None -> Alcotest.fail "--fmad=true must produce a warning; it was silent"
  | Some _ -> ()) ;
  (match Nvrtc.fp_warning_reason "--fmad=false" with
  | Some msg ->
      Alcotest.failf
        "--fmad=false is the safe setting and must be silent: %s"
        msg
  | None -> ()) ;
  Printf.printf
    "  guard accepts %d legitimate options; --fmad=true warns, --fmad=false is \
     silent\n"
    (List.length must_accept)

(* ------------------------------------------------------------------ *)
(* 3. End-to-end: the real entry point raises, with no CUDA needed     *)
(* ------------------------------------------------------------------ *)

let test_compile_to_ptx_raises () =
  let src = generated_source () in
  match
    Nvrtc.compile_to_ptx
      ~name:"guarded"
      ~arch:"compute_75"
      ~options:["-use_fast_math"]
      src
  with
  | _ptx ->
      Alcotest.fail
        "compile_to_ptx accepted -use_fast_math and compiled; the guard did \
         not fire on the real entry point"
  | exception Nvrtc.Fp_conformance_violation msg ->
      Printf.printf "  compile_to_ptx refused -use_fast_math: %s\n" msg
  | exception e ->
      Alcotest.failf
        "compile_to_ptx must raise Fp_conformance_violation; it raised %s \
         (which means the guard did not run before libnvrtc was reached)"
        (Printexc.to_string e)

(* ------------------------------------------------------------------ *)
(* 4. The hazard, measured (host tools only)                           *)
(* ------------------------------------------------------------------ *)

let on_path tool =
  match Unix.system (Printf.sprintf "command -v %s >/dev/null 2>&1" tool) with
  | Unix.WEXITED 0 -> true
  | _ -> false

let read_file p =
  let ic = open_in_bin p in
  let n = in_channel_length ic in
  let s = really_input_string ic n in
  close_in ic ;
  s

let write_file p s =
  let oc = open_out_bin p in
  output_string oc s ;
  close_out oc

(* [.FTZ] on a binary32 arithmetic mnemonic. Deliberately anchored to FMUL and
   FADD -- the two instructions the generated kernel actually contains -- so a
   stray FTZ elsewhere cannot make the control pass. *)
let ftz_arith_re = Str.regexp "F\\(MUL\\|ADD\\)\\.FTZ"

let has_ftz_arith sass =
  try
    ignore (Str.search_forward ftz_arith_re sass 0) ;
    true
  with Not_found -> false

let sass_of ~dir ~tag ~extra_flags src =
  let cu = Filename.concat dir (tag ^ ".cu") in
  let cubin = Filename.concat dir (tag ^ ".cubin") in
  let sass = Filename.concat dir (tag ^ ".sass") in
  write_file cu src ;
  let rc =
    Sys.command
      (Printf.sprintf
         "nvcc -arch=sm_90 -cubin %s -o %s %s >/dev/null 2>&1"
         extra_flags
         (Filename.quote cubin)
         (Filename.quote cu))
  in
  if rc <> 0 then None
  else
    let rc =
      Sys.command
        (Printf.sprintf
           "nvdisasm -c %s > %s 2>/dev/null"
           (Filename.quote cubin)
           (Filename.quote sass))
    in
    if rc <> 0 then None else Some (read_file sass)

let test_ftz_hazard_is_real () =
  if not (on_path "nvcc" && on_path "nvdisasm") then
    Printf.printf
      "  [SKIP] FTZ hazard control: nvcc/nvdisasm not on PATH (host tools, no \
       device needed)\n"
  else
    let dir = Filename.temp_file "sarek_fpc" "" in
    Sys.remove dir ;
    Unix.mkdir dir 0o700 ;
    let src = generated_source () in
    match
      ( sass_of ~dir ~tag:"plain" ~extra_flags:"" src,
        sass_of ~dir ~tag:"ftz" ~extra_flags:"-ftz=true" src )
    with
    | None, _ | _, None ->
        Printf.printf
          "  [SKIP] FTZ hazard control: the local nvcc could not build sm_90\n"
    | Some plain, Some ftz ->
        if has_ftz_arith plain then
          Alcotest.fail
            "control is broken: the DEFAULT build already contains FMUL.FTZ / \
             FADD.FTZ, so this measurement cannot distinguish the flag" ;
        if not (has_ftz_arith ftz) then
          Alcotest.fail
            "control is broken: -ftz=true produced no FMUL.FTZ / FADD.FTZ, so \
             the option this guard rejects has not been shown to change \
             anything on this toolchain" ;
        Printf.printf
          "  FTZ hazard confirmed on this host: -ftz=true turns the generated \
           kernel's binary32 FMUL/FADD into FMUL.FTZ/FADD.FTZ at sm_90 \
           (default build has neither)\n"

let () =
  Alcotest.run
    "cuda_fp_conformance"
    [
      ( "fp_guard",
        [
          Alcotest.test_case
            "FP-relaxing nvrtc options are refused with a reason"
            `Quick
            test_rejected;
          Alcotest.test_case
            "legitimate options stay accepted"
            `Quick
            test_accepted;
          Alcotest.test_case
            "compile_to_ptx itself refuses -use_fast_math"
            `Quick
            test_compile_to_ptx_raises;
          Alcotest.test_case
            "the flushed-subnormal hazard is real (SASS)"
            `Quick
            test_ftz_hazard_is_real;
        ] );
    ]
