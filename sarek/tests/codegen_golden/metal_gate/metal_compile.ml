(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Metal Shading Language compile gate — layer 2 of the Metal gate (#139), and
    the Metal counterpart of {!Opencl_gate.Opencl_clang}, [glslangValidator] and
    [naga].

    Unlike those three, this one cannot run on the machines this project is
    developed on: [metal] ships inside Xcode and exists on macOS only. That is
    the whole reason Metal was the last backend with committed goldens and no
    validator, and the reason #139's two non-compiling goldens survived. So this
    layer is deliberately paired with {!Metal_addrspace}, which needs no
    toolchain and catches the specific class that bit us; a skip here is
    honestly reported and never silent (see [why_unavailable]), and
    ci/assert-toolchain.sh states which platform is expected to provide it. *)

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

(** [-fpreserve-invariance] is NOT passed: the backend's contraction defence is
    the [#pragma METAL fp contract(off)] inside the source (see the
    metal_contraction_pragma group in test_codegen_golden.ml), and adding a
    command-line flag here would validate a compile the project never performs.
    Everything else mirrors a plain AIR compile of one translation unit. *)
let cmd_for src obj err =
  Printf.sprintf
    "xcrun -sdk macosx metal -x metal -std=metal3.0 -c %s -o %s >%s 2>&1"
    (Filename.quote src)
    (Filename.quote obj)
    (Filename.quote err)

let run_metal (source : string) : (unit, string) result =
  let base = Filename.temp_file "sarek_gate_metal_" "" in
  let src = base ^ ".metal" in
  let obj = base ^ ".air" in
  let err = base ^ ".err" in
  write_file src source ;
  let rc = Unix.system (cmd_for src obj err) in
  let out = read_file err in
  List.iter (fun f -> try Sys.remove f with _ -> ()) [src; obj; err; base] ;
  match rc with Unix.WEXITED 0 -> Ok () | _ -> Error out

(** Availability is a POSITIVE CONTROL, not [command -v xcrun]: [xcrun] exists
    on any macOS with the Command Line Tools, and answers "tool 'metal' not
    found" when the full Xcode/Metal toolchain is absent. "Available" here means
    "has just compiled something". Mirrors {!Opencl_gate.Opencl_clang} and
    ci/assert-toolchain.sh. *)
let probe =
  "#include <metal_stdlib>\n\
   using namespace metal;\n\
   kernel void probe(device int* o [[buffer(0)]],\n\
  \                  uint3 gid [[thread_position_in_grid]]) {\n\
  \  o[gid.x] = 1;\n\
   }\n"

let unavailable_reason : string option Lazy.t =
  lazy
    (match Unix.system "command -v xcrun >/dev/null 2>&1" with
    | Unix.WEXITED 0 -> (
        match run_metal probe with
        | Ok () -> None
        | Error e ->
            Some
              ("xcrun is on PATH but the Metal compiler rejected the probe \
                kernel (no Xcode Metal toolchain?): " ^ e))
    | _ -> Some "xcrun not on PATH (Metal compiles on macOS only)")

let available () = Lazy.force unavailable_reason = None

let why_unavailable () =
  match Lazy.force unavailable_reason with Some r -> r | None -> ""
