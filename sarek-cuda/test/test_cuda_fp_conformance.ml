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
 *   0. THE SEPARATED SPELLING. nvrtc takes an option and its value as two array
 *      elements, and the first version of this guard matched only the inline
 *      form -- ["--ftz"; "true"] compiled a subnormal-flushing kernel with no
 *      exception and no warning, confirmed against libnvrtc 13.3 through these
 *      bindings. Both spellings of every value-taking option are now asserted,
 *      as is the fail-closed behaviour of a bare ["--ftz"] with no value.
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
  {default_kernel with kern_name = name; kern_params = params; kern_body = body}

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

(* Every spelling nvrtc accepts for the same hazard: both dash forms, both the
   inline [--ftz=true] and the SEPARATED [--ftz; true] value, and the bare name
   with no value at all (fail-closed). Each entry is a whole option ARRAY,
   because the separated form is invisible to any per-element check. *)
let must_reject =
  [
    (["-use_fast_math"], "subnormal");
    (["--use_fast_math"], "subnormal");
    (["-ftz=true"], "subnormal");
    (["--ftz=true"], "subnormal");
    (["--ftz=1"], "subnormal");
    (* the confirmed bypass *)
    (["--ftz"; "true"], "subnormal");
    (["-ftz"; "true"], "subnormal");
    (["--ftz"; "1"], "subnormal");
    (* fail-closed: a value we cannot resolve is not assumed safe *)
    (["--ftz"], "subnormal");
    (["--ftz"; "--std=c++17"], "subnormal");
    (["--prec-div=false"], "division");
    (["-prec-div=0"], "division");
    (["--prec-div"; "false"], "division");
    (["--prec-sqrt=false"], "square root");
    (["--prec-sqrt"; "false"], "square root");
    (* the real thing it protects: a full array with the hazard buried in it *)
    ( ["--gpu-architecture=compute_75"; "--ftz"; "true"; "-I/opt/cuda/include"],
      "subnormal" );
  ]

let contains hay needle =
  let n = String.length needle and h = String.length hay in
  let rec go i = i + n <= h && (String.sub hay i n = needle || go (i + 1)) in
  n = 0 || go 0

let test_rejected () =
  List.iter
    (fun (opts, needle) ->
      let shown = String.concat " " opts in
      match Nvrtc.fp_rejection_reason_list opts with
      | None ->
          Alcotest.failf
            "%S must be refused on the CUDA path; the guard accepted it"
            shown
      | Some msg ->
          (* Not merely "some error": the message must name the option and
             explain the hazard, or a caller cannot act on it. *)
          (* The message must quote the OFFENDING option, wherever it sits in
             the array -- a message naming only the first element would be
             useless when the hazard is buried among include paths. *)
          if not (List.exists (fun o -> contains msg o) opts) then
            Alcotest.failf
              "rejection message for %S quotes none of its options: %s"
              shown
              msg ;
          if not (contains msg needle) then
            Alcotest.failf
              "rejection message for %S does not explain the hazard (expected \
               %S): %s"
              shown
              needle
              msg ;
          if not (contains msg "docs/fp-contraction-policy.md") then
            Alcotest.failf
              "rejection message for %S does not point at the policy: %s"
              shown
              msg)
    must_reject ;
  Printf.printf
    "  guard rejects %d FP-relaxing option arrays (inline, separated and \
     valueless spellings), each with a reason\n"
    (List.length must_reject)

(* ------------------------------------------------------------------ *)
(* 2. Acceptance — the half that makes rejection non-trivial           *)
(* ------------------------------------------------------------------ *)

let must_accept =
  [
    ["--gpu-architecture=compute_75"];
    ["--include-path=/opt/cuda/include"];
    ["-I/opt/cuda/include"];
    ["-DFOO=1"];
    ["-ftz=false"];
    ["--ftz=false"];
    (* the separated SAFE spellings must survive the separated-form handling *)
    ["--ftz"; "false"];
    ["-ftz"; "0"];
    ["--prec-div=true"];
    ["--prec-div"; "true"];
    ["--prec-sqrt=true"];
    ["--prec-sqrt"; "1"];
    ["-O3"];
    ["--extra-device-vectorization"];
    ["--std=c++17"];
    (* a realistic full array *)
    ["--gpu-architecture=compute_90"; "--include-path=/opt/cuda/include"];
  ]

let test_accepted () =
  List.iter
    (fun opts ->
      match Nvrtc.fp_rejection_reason_list opts with
      | Some msg ->
          Alcotest.failf
            "%S is legitimate and must be accepted; the guard refused it: %s"
            (String.concat " " opts)
            msg
      | None -> ())
    must_accept ;
  (* --fmad=true is a real relaxation but is nvrtc's default, so it warns
     rather than rejects. Both halves matter: it must NOT reject, and it must
     NOT be silent -- in BOTH spellings. *)
  List.iter
    (fun opts ->
      let shown = String.concat " " opts in
      (match Nvrtc.fp_rejection_reason_list opts with
      | Some msg -> Alcotest.failf "%S must warn, not reject: %s" shown msg
      | None -> ()) ;
      match Nvrtc.fp_warning_reason_list opts with
      | None -> Alcotest.failf "%S must produce a warning; it was silent" shown
      | Some _ -> ())
    [["--fmad=true"]; ["--fmad"; "true"]; ["-fmad"; "1"]] ;
  List.iter
    (fun opts ->
      match Nvrtc.fp_warning_reason_list opts with
      | Some msg ->
          Alcotest.failf
            "%S is the safe setting and must be silent: %s"
            (String.concat " " opts)
            msg
      | None -> ())
    [["--fmad=false"]; ["--fmad"; "false"]] ;
  Printf.printf
    "  guard accepts %d legitimate option arrays (incl. separated safe \
     values); --fmad=true warns in both spellings, --fmad=false is silent\n"
    (List.length must_accept)

(* ------------------------------------------------------------------ *)
(* 3. End-to-end: the real entry point raises, with no CUDA needed     *)
(* ------------------------------------------------------------------ *)

let expect_refused ~what options =
  let src = generated_source () in
  match
    Nvrtc.compile_to_ptx ~name:"guarded" ~arch:"compute_75" ~options src
  with
  | _ptx ->
      Alcotest.failf
        "compile_to_ptx accepted %s and compiled; the guard did not fire on \
         the real entry point"
        what
  | exception Nvrtc.Fp_conformance_violation msg ->
      Printf.printf "  compile_to_ptx refused %s: %s\n" what msg
  | exception e ->
      Alcotest.failf
        "compile_to_ptx must raise Fp_conformance_violation for %s; it raised \
         %s (which means the guard did not run before libnvrtc was reached)"
        what
        (Printexc.to_string e)

let test_compile_to_ptx_raises () =
  expect_refused ~what:"-use_fast_math" ["-use_fast_math"] ;
  (* The separated spelling, through the same entry point. This is the case
     that compiled successfully before the guard was made option-shaped. *)
  expect_refused ~what:"the separated form --ftz true" ["--ftz"; "true"]

(* The CHOKEPOINT copy of the guard screens the array this module ASSEMBLES,
   not just the caller's half. Exercised through [nvrtc_option_array], which is
   the exact composition [compile_to_ptx] hands to nvrtcCompileProgram, so a
   hazard introduced by this module's own flags is caught the same way. *)
let test_assembled_array_is_screened () =
  let assembled =
    Nvrtc.nvrtc_option_array
      ~arch:"compute_75"
      ~options:["--ftz"; "true"]
      ~include_opts:["--include-path=/opt/cuda/include"]
      ()
  in
  (* Non-vacuity: the hazard must have SURVIVED assembly, not been dropped. *)
  if not (List.mem "--ftz" assembled) then
    Alcotest.failf
      "the assembled array no longer contains the option under test (%s), so \
       this check proves nothing"
      (String.concat " " assembled) ;
  match Nvrtc.fp_rejection_reason_list assembled with
  | None ->
      Alcotest.failf
        "the array this module assembles (%s) is not screened; a hazard in the \
         module's OWN flags would reach nvrtcCompileProgram"
        (String.concat " " assembled)
  | Some msg -> Printf.printf "  assembled option array is screened: %s\n" msg

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
  if not (on_path "nvcc" && on_path "nvdisasm") then (
    Printf.printf
      "  [SKIP] FTZ hazard control: nvcc/nvdisasm not on PATH (host tools, no \
       device needed)\n" ;
    (* Alcotest.skip, not a bare return: a skipped gate that prints [OK] is
       indistinguishable from a gate that ran, which is exactly the regression
       commit cf58801b fixed. *)
    Alcotest.skip ())
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
          "  [SKIP] FTZ hazard control: the local nvcc could not build sm_90\n" ;
        Alcotest.skip ()
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
            "compile_to_ptx itself refuses relaxing options, both spellings"
            `Quick
            test_compile_to_ptx_raises;
          Alcotest.test_case
            "the array this module assembles is screened too"
            `Quick
            test_assembled_array_is_screened;
          Alcotest.test_case
            "the flushed-subnormal hazard is real (SASS)"
            `Quick
            test_ftz_hazard_is_real;
        ] );
    ]
