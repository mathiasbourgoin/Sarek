(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * OpenCL floating-point conformance: the build-option string this backend
 * hands to [clBuildProgram], and the screen applied to anything a caller adds.
 *
 * WHY THIS EXISTS (#136)
 *
 * Until this module, [Opencl_api.Program.build] called [clBuildProgram] with
 * an EMPTY option string. An empty option string is not "no policy": it is a
 * policy, chosen by omission, and it is whatever each vendor decided its
 * default should be. OpenCL's default permits a [sqrt] of up to 3 ulp and a
 * divide of up to 2.5 ulp (OpenCL C spec, "Relative error as ULPs"), and
 * Sarek's rule (docs/fp-contraction-policy.md §1) is that division and square
 * root are CORRECTLY ROUNDED.
 *
 * That gap was not hypothetical. On a GTX 1070 Max-Q (sm_61, CUDA 12.9,
 * driver 580.119.02), [test_real64]'s df64 fallback measured an OpenCL [sqrt]
 * worst-case relative error of 1.81e-14 against a 1.42e-14 tolerance — a
 * FAILURE — and building the same kernel with
 * [-cl-fp32-correctly-rounded-divide-sqrt] moved it to 8.87e-15, a PASS,
 * coinciding with the interpreter's figure for the same input set. Measured
 * during #298; quoted here rather than re-measured, because there is no NVIDIA
 * device on the machine this module was written on.
 *
 * This is the same shape as the PTX [sqrt.approx.f32] defect (#298) and the
 * OpenCL sibling of [Cuda_nvrtc.check_fp_conformance] and
 * [Hip_rtc.hiprtc_options]: a faster, less accurate default selected by
 * passing nothing.
 *
 * WHY THE CORRECTLY-ROUNDED FLAG IS CAPABILITY-GATED AND NOT UNCONDITIONAL
 *
 * The OpenCL spec makes [-cl-fp32-correctly-rounded-divide-sqrt] conditional:
 * it may only be specified if [CL_DEVICE_SINGLE_FP_CONFIG] contains
 * [CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT]; otherwise [clBuildProgram] "will
 * return CL_INVALID_BUILD_OPTIONS". Passing it unconditionally would therefore
 * break EVERY kernel build on every device that does not advertise the
 * capability — which is a strictly worse failure than the accuracy defect it
 * fixes, because it is total rather than numerical.
 *
 * This machine would not have caught that. MEASURED here, 2026-07-26, with
 * tools/probes/opencl_build_options_probe.c on rusticl/radeonsi 26.1.4-arch3.1
 * (RX 7900 XTX navi31, and the Raphael iGPU): both devices report
 * [CL_DEVICE_SINGLE_FP_CONFIG = 0x6] — [INF_NAN | ROUND_TO_NEAREST], with
 * neither [CL_FP_DENORM] nor [CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT] — and BOTH
 * accept [-cl-fp32-correctly-rounded-divide-sqrt] with [CL_SUCCESS] anyway.
 * That is rusticl departing from the spec in the permissive direction. An
 * unconditional flag would have looked perfectly healthy on this box and
 * failed on a conformant implementation elsewhere. The gate exists because the
 * local stack cannot be trusted to reveal its absence.
 *
 * WHAT THIS BUYS ON THE LOCAL DEVICES: NOTHING, MEASURABLY
 *
 * Also measured 2026-07-26 (tools/probes/opencl_build_options_probe.c, the
 * [effect] mode), 2^20 inputs per device: rusticl's [sqrt] is already within
 * 1 ulp and its divide within 2 ulp of the host's correctly-rounded result,
 * and the results are BIT-IDENTICAL (0 of 1048576 differ) with and without the
 * flag. So on this box the flag changes nothing and costs nothing.
 *
 * That "costs nothing" must be read with its control attached. The probe
 * carries two:
 *   - a PLUMBING control ([-DSAREK_PROBE_SCALE=1.0000001f]) which DOES change
 *     the result on 1048576/1048576 inputs, proving the option string really
 *     reaches rusticl's compiler and the comparison can go non-zero;
 *   - an FP LIVENESS control ([-cl-fast-relaxed-math]) which does NOT — it is
 *     bit-identical to the baseline too.
 * The second control FAILING is the finding: rusticl delivers the option
 * string but ignores the FP-relaxing options in it. So on this stack, "the
 * flag changed nothing" cannot be distinguished from "the flag was discarded",
 * and no accuracy or cost claim for this change may be founded on these
 * devices. The measurement that matters is the sm_61 one quoted above.
 *
 * WHAT IS *NOT* FIXED HERE
 *
 * [FP_CONTRACT] is ON by default in OpenCL C and NO build option turns it off.
 * The source-level [#pragma OPENCL FP_CONTRACT OFF] was measured on this stack
 * and does not work (see [Sarek_ir_opencl], the [TFloat16] rejection comment:
 * 620/63488 f16 disagreements survive it). Contraction on OpenCL is therefore
 * still defeated BY CONSTRUCTION, via [Sarek_df64]'s [mul_rn], exactly as on
 * CUDA — not by any flag this module sets.
 ******************************************************************************)

(** Raised when a build-option string would relax float semantics below what
    docs/fp-contraction-policy.md §1 promises. *)
exception Fp_conformance_violation of string

(** {1 CL_DEVICE_SINGLE_FP_CONFIG bits}

    The vendored [dependencies/CL/cl.h] is OpenCL 1.0-era and defines neither of
    the two bits this module needs, so they are spelled out. Values from the
    OpenCL 1.2+ headers. *)

let cl_fp_denorm = 1L

let cl_fp_correctly_rounded_divide_sqrt = 128L

let has_bit (config : int64) (bit : int64) = Int64.logand config bit <> 0L

(** {1 The options this backend sets explicitly} *)

(** [conformance_options ~single_fp_config] is the option list Sarek chooses on
    its own behalf, given a device's [CL_DEVICE_SINGLE_FP_CONFIG].

    The full audit of what an empty option string was silently accepting, and
    what is now decided explicitly. Every entry is a DEFAULT that was being
    inherited rather than chosen:

    - [-cl-fp32-correctly-rounded-divide-sqrt] — ADDED, gated on
      [CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT]. Default off means [sqrt] up to 3
      ulp and divide up to 2.5 ulp; §1 requires both correctly rounded, and
      [Sarek_df64]'s Newton/Karp step squares its seed's error and has no margin
      for a 3-ulp seed. Gated because the spec makes it an error to pass it
      otherwise (see the header).
    - [-cl-denorms-are-zero] — NEVER PASSED. It is a HINT that the device may
      flush binary32 subnormals; §1 says subnormals are not flushed. Its default
      (absent) is the conformant one, so the fix is to keep not passing it — but
      note this is a request, not a guarantee in either direction: whether
      subnormals actually survive is [CL_FP_DENORM] in the device's config, and
      on BOTH local devices that bit is CLEAR, so f32 subnormals are flushed
      here regardless of any build option. That is a device property Sarek
      cannot correct, and it is recorded rather than fixed.
    - [-cl-fast-relaxed-math] — NEVER PASSED, and REFUSED from callers. It
      implies [-cl-unsafe-math-optimizations] and [-cl-finite-math-only], and
      additionally lets the implementation substitute [native_*] builtins.
    - [-cl-unsafe-math-optimizations] — NEVER PASSED, REFUSED. Permits
      reassociation, which destroys an error-free transformation outright (§1
      corollary 1); implies [-cl-no-signed-zeros] and [-cl-mad-enable].
    - [-cl-finite-math-only] — NEVER PASSED, REFUSED. Assumes no NaN/Inf
      operand; the interpreter oracle assumes no such thing.
    - [-cl-no-signed-zeros] — NEVER PASSED, REFUSED. Discards the sign of zero,
      which [Sarek_df64]'s renormalisation steps rely on.
    - [-cl-mad-enable] — NEVER PASSED, REFUSED. This is the contraction hazard
      by name: it permits [a*b+c] to become a [mad] of reduced accuracy.
    - [-cl-single-precision-constant] — NEVER PASSED, REFUSED. Silently demotes
      double literals to single, changing the meaning of a written constant.
    - [-cl-opt-disable] — not passed, and ALLOWED from a caller: it is
      conservative, it cannot relax FP semantics, and it is useful for debugging
      codegen.

    Deliberately NOT here: anything that would need to be undone later. Unlike
    HIP — where appending [-ffp-contract=off] LAST neutralises whatever the
    caller passed, because clang resolves conflicting FP options by last
    occurrence — OpenCL has no build option that undoes [-cl-fast-relaxed-math].
    So the relaxing options are REFUSED rather than countered, for the same
    reason [Cuda_nvrtc] refuses [-use_fast_math] (docs/fp-contraction-policy.md
    §5). *)
let conformance_options ~(single_fp_config : int64) : string list =
  if has_bit single_fp_config cl_fp_correctly_rounded_divide_sqrt then
    ["-cl-fp32-correctly-rounded-divide-sqrt"]
  else []

(** {1 Screening what a caller adds} *)

type fp_verdict = Fp_reject of string | Fp_warn of string

(** [(token, verdict, why)].

    Every OpenCL FP option is a BARE SWITCH — none takes a value, inline or
    separated. That is why this screen can match on whole tokens and does not
    need [Cuda_nvrtc]'s value-resolution machinery, which exists because nvrtc
    accepts [--ftz true] as two array elements and a spelling-shaped matcher let
    the separated form straight through (§5). The hazard has no OpenCL analogue;
    if a future OpenCL option ever takes a value, this comment is the warning
    that this matcher is then the wrong shape. *)
let fp_option_classes : (string * [`Reject | `Warn] * string) list =
  [
    ( "-cl-fast-relaxed-math",
      `Reject,
      "implies -cl-unsafe-math-optimizations and -cl-finite-math-only, and \
       lets the implementation substitute native_* builtins; no later OpenCL \
       build option undoes it" );
    ( "-cl-unsafe-math-optimizations",
      `Reject,
      "permits reassociation, which destroys an error-free transformation \
       (Dekker/Knuth TwoSum, TwoProd) outright rather than by one ulp; implies \
       -cl-no-signed-zeros and -cl-mad-enable" );
    ( "-cl-finite-math-only",
      `Reject,
      "assumes no argument or result is NaN or Inf; the interpreter oracle \
       assumes no such thing, so device and oracle diverge on the inputs the \
       flag excluded" );
    ( "-cl-no-signed-zeros",
      `Reject,
      "discards the sign of zero, which Sarek_df64's renormalisation relies on"
    );
    ( "-cl-mad-enable",
      `Reject,
      "permits a*b+c to become a mad of reduced accuracy — contraction by \
       name; Sarek_df64 defeats contraction by construction (mul_rn) and a \
       reduced-accuracy mad is below what that defence assumes" );
    ( "-cl-denorms-are-zero",
      `Reject,
      "asks the device to flush binary32 subnormals; \
       docs/fp-contraction-policy.md §1 says subnormals are not flushed, and \
       unlike contraction there is no later option that restores them" );
    ( "-cl-single-precision-constant",
      `Reject,
      "silently treats double-precision literals as single, changing the value \
       of a written constant" );
    ( "-cl-fp32-correctly-rounded-divide-sqrt",
      `Warn,
      "is set by this backend already when the device advertises \
       CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT; passing it by hand on a device \
       that does NOT advertise it makes clBuildProgram return \
       CL_INVALID_BUILD_OPTIONS on a conformant implementation" );
  ]

(** Split an OpenCL build-option string on whitespace. OpenCL takes ONE string,
    not an array, so this is the only tokenisation there is. *)
let tokenise (opts : string) : string list =
  String.split_on_char ' ' opts
  |> List.concat_map (String.split_on_char '\t')
  |> List.concat_map (String.split_on_char '\n')
  |> List.filter (fun s -> s <> "")

let fp_scan (opts : string) : fp_verdict list =
  let toks = tokenise opts in
  List.filter_map
    (fun (name, verdict, why) ->
      if List.exists (String.equal name) toks then
        let msg =
          Printf.sprintf
            "OpenCL build option %s violates Sarek's floating-point contract: \
             %s. See docs/fp-contraction-policy.md."
            name
            why
        in
        Some
          (match verdict with `Reject -> Fp_reject msg | `Warn -> Fp_warn msg)
      else None)
    fp_option_classes

(** Raise {!Fp_conformance_violation} on a caller option string that would relax
    float semantics; warn on the merely redundant.

    Pure: no device, no OpenCL ICD and no [libOpenCL] are needed to run it, so
    it is reachable in a test suite on a host with no OpenCL at all — the same
    property that makes [Cuda_nvrtc.check_fp_conformance] testable without CUDA.
*)
let check_fp_conformance (opts : string) : unit =
  List.iter
    (function
      | Fp_reject msg -> raise (Fp_conformance_violation msg)
      | Fp_warn msg -> Spoc_core.Log.warnf Spoc_core.Log.Kernel "%s" msg)
    (fp_scan opts)

(** {1 The exact string handed to clBuildProgram} *)

(** [build_options ~single_fp_config ~caller] is the complete option string
    [clBuildProgram] receives.

    [caller] is screened first — so an offending option raises BEFORE any OpenCL
    entry point is touched — and Sarek's own options are appended after it.
    Ordering carries no meaning here (no OpenCL FP option overrides another,
    unlike clang's last-occurrence rule that [Hip_rtc.base_options] depends on);
    the options are appended last purely so the resulting string reads as "what
    the caller asked for, then what this backend requires". *)
let build_options ~(single_fp_config : int64) ~(caller : string) : string =
  check_fp_conformance caller ;
  let ours = conformance_options ~single_fp_config in
  String.concat " " (List.filter (fun s -> s <> "") (tokenise caller @ ours))
