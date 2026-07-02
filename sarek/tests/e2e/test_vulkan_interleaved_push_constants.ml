(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * E2E regression test: Vulkan push-constant layout for interleaved
 * (vector, scalar, vector) kernel arguments.
 *
 * Bug: Vulkan_api_kernel's scalar set_arg_int32/int64/float32/float64 used to
 * ignore idx and append raw bytes to the push-constant block in CALL ORDER.
 * The GLSL codegen's push-constant block layout (see
 * sarek/codegen/Sarek_ir_glsl.ml:889-919, gen_push_constants) is instead:
 * ALL vector lengths first (in vector-declaration order), THEN all user
 * scalars (in declaration order) - a fixed grouping, not call order.
 *
 * A kernel signature (a: vec, scale: scalar, b: vec, dst: vec, n: scalar)
 * exposes the bug: Execute.expand_to_run_source_args
 * (sarek/execute/Execute.ml:236-254) expands this to the flat call sequence
 *   Buf a; Int32 a_len; Float32 scale; Buf b; Int32 b_len; Buf dst;
 *   Int32 dst_len; Int32 n
 * i.e. the scalar [scale] is sandwiched between vector-length writes. The old
 * code appended push-constant bytes in exactly that call order, producing
 * [a_len; scale; b_len; dst_len; n] on the wire - shifted relative to the
 * GLSL block's [a_len; b_len; dst_len; scale; n], so every field after
 * [scale] reads garbage from its neighbor's bytes.
 *
 * This test is device-filtered to Vulkan only and skips cleanly (does not
 * fail) if no Vulkan device is available - see [main] below.
 ******************************************************************************)

module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer

let () = Sarek_vulkan.Vulkan_plugin.init ()

(** GLSL compute shader: dst[i] = a[i] * scale + b[i].

    Push-constant block matches exactly what Sarek_ir_glsl.gen_push_constants
    would emit for kernel params (a: vec, scale: float32, b: vec, dst: vec, n:
    int32): vector lengths [a_len; b_len; dst_len] first (in the order the
    vectors were declared / bound), then user scalars [scale; n] (in declaration
    order). *)
let glsl_axpy_interleaved =
  {|#version 450

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(std430, set=0, binding = 0) readonly buffer BufferA {
    float a[];
};

layout(std430, set=0, binding = 1) readonly buffer BufferB {
    float b[];
};

layout(std430, set=0, binding = 2) writeonly buffer BufferDst {
    float dst[];
};

layout(push_constant) uniform PushConstants {
    int a_len;
    int b_len;
    int dst_len;
    float scale;
    int n;
} pc;

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i < uint(pc.n)) {
        dst[i] = a[i] * pc.scale + b[i];
    }
}
|}

let scale = 3.5

let compute_expected size =
  Array.init size (fun i -> (float_of_int i *. scale) +. float_of_int (i * 2))

let verify_results result expected =
  let size = Array.length expected in
  let errors = ref 0 in
  for i = 0 to size - 1 do
    let diff = abs_float (result.(i) -. expected.(i)) in
    if diff > 1e-3 then begin
      if !errors < 5 then
        Printf.printf
          "  Mismatch at %d: expected %.4f, got %.4f\n"
          i
          expected.(i)
          result.(i) ;
      incr errors
    end
  done ;
  !errors = 0

let run_test (dev : Device.t) size block_size =
  let a = Vector.create Vector.float32 size in
  let b = Vector.create Vector.float32 size in
  let dst = Vector.create Vector.float32 size in

  for i = 0 to size - 1 do
    Vector.set a i (float_of_int i) ;
    Vector.set b i (float_of_int (i * 2)) ;
    Vector.set dst i (-999.0)
  done ;

  let threads = block_size in
  let grid_x = (size + threads - 1) / threads in

  (* Deliberately interleaved: (vec a, scalar scale, vec b, vec dst, scalar n)
     - the exact shape that exposes the push-constant ordering bug. *)
  Sarek.Execute.run_source
    ~device:dev
    ~source:glsl_axpy_interleaved
    ~lang:Sarek.Execute.GLSL_Source
    ~kernel_name:"axpy_interleaved"
    ~block:(Sarek.Execute.dims1d threads)
    ~grid:(Sarek.Execute.dims1d grid_x)
    [
      Sarek.Execute.Vec a;
      Sarek.Execute.Float32 scale;
      Sarek.Execute.Vec b;
      Sarek.Execute.Vec dst;
      Sarek.Execute.Int32 (Int32.of_int size);
    ] ;
  Transfer.flush dev ;
  Vector.to_array dst

let find_vulkan_device () =
  let vulkan_devices = Device.by_framework "Vulkan" in
  if Array.length vulkan_devices > 0 then Some vulkan_devices.(0) else None

let () =
  match find_vulkan_device () with
  | None ->
      Printf.printf
        "[SKIP] No Vulkan device available - skipping interleaved \
         push-constant test\n\
         %!"
  | Some dev ->
      let size = 4096 in
      let block_size = 256 in
      let expected = compute_expected size in
      let result = run_test dev size block_size in
      if verify_results result expected then
        Printf.printf
          "[PASS] Vulkan interleaved (vec, scalar, vec) push constants: %d \
           elements match\n\
           %!"
          size
      else begin
        Printf.printf
          "[FAIL] Vulkan interleaved (vec, scalar, vec) push constants \
           produced wrong results\n\
           %!" ;
        exit 1
      end
