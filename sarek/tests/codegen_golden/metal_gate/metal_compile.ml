(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Metal Shading Language compile gate — layer 2 of the Metal gate (#139), and
    the Metal counterpart of {!Opencl_gate.Opencl_clang}, [glslangValidator] and
    [naga].

    {1 Why not [xcrun metal]}

    Because it would never have run. The offline [metal] compiler ships inside
    Xcode, and the reference Apple M4 this project measures on (macOS 15.6.1,
    Apple clang 17) has only the Command Line Tools:

    {v xcrun: error: unable to find utility "metal", not a developer tool v}

    A gate keyed on [xcrun metal] would therefore have reported an honest,
    well-worded skip on the exact machine that found #139 — a skip quietly
    becoming the normal outcome everywhere, which is the failure mode this whole
    gate exists to stop.

    {1 What runs instead}

    [newLibraryWithSource:options:error:], through a small Objective-C driver
    built with [clang -framework Metal]. That call compiles through the Metal
    driver at runtime, needs no Xcode, and is documented as the reason the
    project's other probes work on this machine (see the build note atop
    tools/probes/metal_math_mode_probe.m). It is also the call [Sarek_metal]'s
    runtime actually makes, so this validates the production compile path rather
    than a neighbouring one.

    Confirmed on the M4 against both shapes of #139: [constant Point2* &pts]
    gives "invalid address space qualification for buffer pointee type 'const
    constant Point2 *'" and [device Point2* pts] compiles.

    {1 What it still cannot cover}

    The driver only exists on macOS, so on Linux this layer skips with a stated
    reason and {!Metal_addrspace} is the only cover — which is precisely why
    that layer is pure text and needs no toolchain. Nothing here executes a
    kernel either: a library that compiles is not a library that computes the
    right answer. *)

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

(* The driver, embedded rather than kept as a file under tools/probes and looked
   up at run time. A relative-path lookup would depend on dune's sandbox cwd and
   would degrade into "source not found -> skip" the first time that moved,
   reintroducing the silent skip by the back door. Embedded, the layer either
   builds or reports why. *)
let driver_source =
  {objc|
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>

int main(int argc, const char **argv) {
  @autoreleasepool {
    if (argc != 2) { fprintf(stderr, "usage: %s <file.metal>\n", argv[0]); return 2; }
    NSError *err = nil;
    NSString *path = [NSString stringWithUTF8String:argv[1]];
    NSString *src = [NSString stringWithContentsOfFile:path
                                              encoding:NSUTF8StringEncoding
                                                 error:&err];
    if (src == nil) {
      fprintf(stderr, "cannot read %s: %s\n", argv[1],
              err ? [[err localizedDescription] UTF8String] : "?");
      return 2;
    }
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (dev == nil) { fprintf(stderr, "no Metal device\n"); return 2; }
    /* Defaults on purpose: the contraction defence is the #pragma inside the
       source (metal_contraction_pragma group), not a compile option. Setting
       anything here would validate a compile Sarek never performs. */
    MTLCompileOptions *opts = [MTLCompileOptions new];
    err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:src options:opts error:&err];
    if (lib == nil) {
      fprintf(stderr, "%s\n",
              err ? [[err localizedDescription] UTF8String]
                  : "newLibraryWithSource failed with no error object");
      return 1;
    }
    return 0;
  }
}
|objc}

(* Built once per process. [Error] carries the reason the layer is unusable. *)
let driver : (string, string) result Lazy.t =
  lazy
    (if Sys.command "uname -s | grep -q Darwin" <> 0 then
       Error
         "not macOS — the Metal driver (newLibraryWithSource:) exists only \
          there, so no Metal source can be compiled on this machine"
     else begin
       let dir = Filename.temp_file "sarek_metal_drv_" "" in
       (try Sys.remove dir with _ -> ()) ;
       (try Unix.mkdir dir 0o700 with _ -> ()) ;
       let src = Filename.concat dir "probe.m" in
       let exe = Filename.concat dir "probe" in
       let log = Filename.concat dir "build.log" in
       write_file src driver_source ;
       let rc =
         Unix.system
           (Printf.sprintf
              "clang -fobjc-arc -O1 -framework Foundation -framework Metal %s \
               -o %s >%s 2>&1"
              (Filename.quote src)
              (Filename.quote exe)
              (Filename.quote log))
       in
       match rc with
       | Unix.WEXITED 0 -> Ok exe
       | _ ->
           Error
             ("could not build the Metal compile driver (clang -framework \
               Metal): " ^ read_file log)
     end)

let run_metal (source : string) : (unit, string) result =
  match Lazy.force driver with
  | Error e -> Error e
  | Ok exe -> (
      let base = Filename.temp_file "sarek_gate_metal_" "" in
      let src = base ^ ".metal" in
      let err = base ^ ".err" in
      write_file src source ;
      let rc =
        Unix.system
          (Printf.sprintf
             "%s %s >%s 2>&1"
             (Filename.quote exe)
             (Filename.quote src)
             (Filename.quote err))
      in
      let out = read_file err in
      List.iter (fun f -> try Sys.remove f with _ -> ()) [src; err; base] ;
      match rc with Unix.WEXITED 0 -> Ok () | _ -> Error out)

(** Availability is a POSITIVE CONTROL, not a [uname] test: a macOS without the
    Command Line Tools has no [clang], and a headless or virtualised host can
    have clang and no Metal device. "Available" here means "has just compiled a
    kernel". Mirrors {!Opencl_gate.Opencl_clang} and ci/assert-toolchain.sh. *)
let probe =
  "#include <metal_stdlib>\n\
   using namespace metal;\n\
   kernel void probe(device int* o [[buffer(0)]],\n\
  \                  uint3 gid [[thread_position_in_grid]]) {\n\
  \  o[gid.x] = 1;\n\
   }\n"

let unavailable_reason : string option Lazy.t =
  lazy
    (match run_metal probe with
    | Ok () -> None
    | Error e -> Some ("Metal source compilation is unavailable here: " ^ e))

let available () = Lazy.force unavailable_reason = None

let why_unavailable () =
  match Lazy.force unavailable_reason with Some r -> r | None -> ""
