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

(** The invocation. [-cl-std=CL1.2] is the language level Sarek's OpenCL backend
    targets; [-finclude-default-header] supplies the OpenCL builtin declarations
    ([get_global_id], [sqrt], ...) that a real ICD injects, without which every
    kernel would fail on builtins rather than on its own defects. *)
let cmd_for src err =
  Printf.sprintf
    "clang -x cl -cl-std=CL1.2 -Xclang -finclude-default-header -fsyntax-only \
     %s >%s 2>&1"
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
