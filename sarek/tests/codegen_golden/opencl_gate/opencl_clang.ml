(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** OpenCL C compile gate — the OpenCL counterpart of the [ptxas], [nvrtc],
    [glslangValidator] and [naga] gates (#84/#51, extended by #128).

    Uses [clang -x cl] rather than a vendor ICD on purpose. A gate must fail
    where WE can read the failure, and a real ICD is exactly what this project
    cannot rely on: on the reference machine (RX 7900 XTX, rusticl/radeonsi,
    Mesa) illegal generated OpenCL took the host process down with SIGSEGV
    instead of returning a build log (#53/#127). [clang] is hermetic, needs no
    device, runs in CI, and reports a diagnostic. Its blind spots are documented
    in {!Opencl_recursion}, which covers them. *)

let read_file f =
  try
    let ic = open_in f in
    let n = in_channel_length ic in
    let s = really_input_string ic n in
    close_in ic ;
    s
  with _ -> ""

let write_file f s =
  let oc = open_out f in
  output_string oc s ;
  close_out oc

(** fp64 SUPPRESSION SWITCH (#140). With [SAREK_OPENCL_GATE_NO_FP64=1] every
    invocation below gains [-cl-ext=-cl_khr_fp64], which makes this clang refuse
    [double] with

    {v error: use of type 'double' requires cl_khr_fp64 support v}

    — byte-for-byte the diagnostic Apple clang 17 produces on an M4, where the
    extension is simply absent from the target. It is a faithful emulation of
    the toolchain that exposed the split verdict, not a mock of the check: the
    probe and the corpus go through the same compiler with the same flag, so "no
    fp64" is established by compiling, exactly as on the real machine.

    That is what makes the skip path below provable on a machine that HAS fp64.
*)
let no_fp64_flag =
  match Sys.getenv_opt "SAREK_OPENCL_GATE_NO_FP64" with
  | Some ("1" | "true" | "yes") -> " -Xclang -cl-ext=-cl_khr_fp64"
  | _ -> ""

(** The invocation. [-cl-std=CL1.2] is the language level Sarek's OpenCL backend
    targets; [-finclude-default-header] supplies the OpenCL builtin declarations
    ([get_global_id], [sqrt], ...) that a real ICD injects, without which every
    kernel would fail on builtins rather than on its own defects. *)
let cmd_for src err =
  Printf.sprintf
    "clang -x cl -cl-std=CL1.2 -Xclang -finclude-default-header%s \
     -fsyntax-only %s >%s 2>&1"
    no_fp64_flag
    (Filename.quote src)
    (Filename.quote err)

let run_clang (source : string) : (unit, string) result =
  let base = Filename.temp_file "sarek_gate_ocl_" "" in
  let src = base ^ ".cl" in
  let err = base ^ ".err" in
  write_file src source ;
  let rc = Unix.system (cmd_for src err) in
  let out = read_file err in
  List.iter (fun f -> try Sys.remove f with _ -> ()) [src; err; base] ;
  match rc with Unix.WEXITED 0 -> Ok () | _ -> Error out

(** Availability is a POSITIVE CONTROL, not [command -v]: a clang build without
    OpenCL support, or without the default header, is on PATH and useless. The
    probe below is the smallest kernel that exercises both the [cl] language
    mode and the builtin header, so "available" means "has just compiled
    something". Mirrors the probe discipline in [ci/assert-toolchain.sh]. *)
let probe =
  "__kernel void probe(__global int *o) { o[get_global_id(0)] = 1; }\n"

let unavailable_reason : string option Lazy.t =
  lazy
    (match Unix.system "command -v clang >/dev/null 2>&1" with
    | Unix.WEXITED 0 -> (
        match run_clang probe with
        | Ok () -> None
        | Error e ->
            Some ("clang is on PATH but rejected the OpenCL probe kernel: " ^ e)
        )
    | _ -> Some "clang not on PATH")

let available () = Lazy.force unavailable_reason = None

let why_unavailable () =
  match Lazy.force unavailable_reason with Some r -> r | None -> ""

(** {1 fp64 capability (#140)}

    THE DEFECT THIS EXISTS FOR. On an Apple M4 the sweep reported two verdicts
    for one missing capability: [float64_abs_float_path] and
    [float64_copysign_path] SKIPPED while five other float64 cases FAILED. The
    real answer was worse than "one path forgot to check": NEITHER path checked.
    The two that skipped did so for an unrelated hardcoded exclusion (an
    unmapped [Float64.abs_float] intrinsic that dies at codegen) and would have
    skipped on a machine with full fp64 too; the five that failed simply ran and
    hit the toolchain wall. There was no fp64 predicate anywhere in the gate.

    WHAT THE CAPABILITY ACTUALLY IS, here. It is NOT the device's [cl_khr_fp64]
    — this gate deliberately never touches a device (see the header). The
    subject is host clang, and Apple clang targeting [arm64-apple-darwin] does
    not list [cl_khr_fp64] among the extensions that target supports, so
    [double] is an error in [-x cl] mode regardless of the
    [#pragma OPENCL EXTENSION cl_khr_fp64 : enable] the production path emits.
    On the Linux reference machine the same clang invocation accepts it. So the
    honest predicate is "can THIS clang compile a double kernel", and it is
    established the same way availability is: by compiling one.

    Established by a POSITIVE CONTROL, like [available] above: the smallest
    kernel that uses [double]. [SAREK_OPENCL_GATE_NO_FP64=1] (see
    [no_fp64_flag]) takes the extension away from the compiler itself, so this
    probe fails for the real reason and the skip path can be driven — and
    observed — on a machine that does have fp64. A skip nobody has watched
    happen is not a skip anybody should trust. *)
let fp64_probe =
  "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n\
   __kernel void probe64(__global double *o) { o[get_global_id(0)] = 1.0; }\n"

(** The fp64 probe's RAW compiler output, kept separate from any wording of
    ours.

    THE COMPOSED REASON BELOW MUST NOT NAME [cl_khr_fp64], and this split is
    why. An earlier version wrapped the diagnostic in "(the target does not
    support cl_khr_fp64)" — so the composed string contained that token whenever
    the probe failed, {e for any reason at all}. The negative control then
    asserted "the reason names cl_khr_fp64" against that composed string and was
    checking our own printf, not clang's behaviour: a probe broken by an
    unrelated syntax error would have satisfied it. That is a check that cannot
    fail, which is precisely what this gate exists to rule out.

    So the token is asserted against THIS value, which only clang can put there.
    [None] when fp64 works, or when clang is unusable for a reason that has
    nothing to do with fp64. *)
let no_fp64_diagnostic : string option Lazy.t =
  lazy
    (match Lazy.force unavailable_reason with
    | Some _ -> None
    | None -> (
        match run_clang fp64_probe with Ok () -> None | Error e -> Some e))

let no_fp64_reason : string option Lazy.t =
  lazy
    (match Lazy.force unavailable_reason with
    | Some r -> Some ("clang itself is unusable: " ^ r)
    | None -> (
        match Lazy.force no_fp64_diagnostic with
        | None -> None
        | Some e ->
            (* States only what was observed — that this clang rejected a
               `double` kernel — and then hands over the compiler's own words.
               No claim about WHY is manufactured here. *)
            Some
              ("this clang cannot compile an OpenCL kernel using `double`: " ^ e)
        ))

(** [true] iff the clang this gate drives can compile a [double] OpenCL kernel.
    The single authority for every float64 case in the sweep. *)
let fp64_available () = Lazy.force no_fp64_reason = None

let why_no_fp64 () =
  match Lazy.force no_fp64_reason with Some r -> r | None -> ""

(** The unedited compiler diagnostic from the fp64 probe, or [""]. Assert
    against this, never against {!why_no_fp64}. *)
let fp64_diagnostic () =
  match Lazy.force no_fp64_diagnostic with Some e -> e | None -> ""
