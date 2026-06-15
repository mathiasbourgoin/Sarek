(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * E2E test: External kernel execution via run_source
 *
 * Tests running pre-written CUDA/OpenCL/Vulkan kernels through the GPU runtime
 * without using the Sarek DSL.
 ******************************************************************************)

module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer
module Benchmarks = Test_helpers.Benchmarks

(* Force backend registration *)
let () =
  Sarek_cuda.Cuda_plugin.init () ;
  Sarek_opencl.Opencl_plugin.init () ;
  Sarek_vulkan.Vulkan_plugin.init () ;
  Sarek_native.Native_plugin.init () ;
  Sarek_interpreter.Interpreter_plugin.init ()

(** OpenCL vector add kernel source *)
let opencl_vector_add =
  {|
__kernel void vector_add(
    __global const float* a,
    int a_len,
    __global const float* b,
    int b_len,
    __global float* c,
    int c_len,
    int n)
{
    int i = get_global_id(0);
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}
|}

(** CUDA vector add kernel source *)
let cuda_vector_add =
  {|
extern "C" __global__ void vector_add(
    const float* a,
    int a_len,
    const float* b,
    int b_len,
    float* c,
    int c_len,
    int n)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}
|}

(** Vulkan GLSL vector add compute shader *)
let glsl_vector_add =
  {|#version 450

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(std430, set=0, binding = 0) readonly buffer BufferA {
    float a[];
};

layout(std430, set=0, binding = 1) readonly buffer BufferB {
    float b[];
};

layout(std430, set=0, binding = 2) writeonly buffer BufferC {
    float c[];
};

layout(push_constant) uniform PushConstants {
    int a_len;
    int b_len;
    int c_len;
    int n;
} pc;

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i < pc.n) {
        c[i] = a[i] + b[i];
    }
}
|}

(* ========== Pure OCaml baseline ========== *)

let ocaml_vector_add size =
  let result = Array.make size 0.0 in
  for i = 0 to size - 1 do
    let a = float_of_int i in
    let b = float_of_int (i * 2) in
    result.(i) <- a +. b
  done ;
  result

(* ========== External kernel runner ========== *)

let run_external_kernel (dev : Device.t) size block_size =
  let n = size in
  let a = Vector.create Vector.float32 n in
  let b = Vector.create Vector.float32 n in
  let c = Vector.create Vector.float32 n in

  (* Initialize vectors *)
  for i = 0 to n - 1 do
    Vector.set a i (float_of_int i) ;
    Vector.set b i (float_of_int (i * 2)) ;
    Vector.set c i 0.0
  done ;

  let threads = block_size in
  let grid_x = (n + threads - 1) / threads in

  (* Select source based on device framework *)
  let source, lang =
    match dev.Device.framework with
    | "CUDA" -> (cuda_vector_add, Sarek.Execute.CUDA_Source)
    | "OpenCL" -> (opencl_vector_add, Sarek.Execute.OpenCL_Source)
    | "Vulkan" -> (glsl_vector_add, Sarek.Execute.GLSL_Source)
    | fw -> failwith ("External kernels not supported on " ^ fw)
  in

  let t0 = Unix.gettimeofday () in

  (* Run external kernel *)
  Sarek.Execute.run_source
    ~device:dev
    ~source
    ~lang
    ~kernel_name:"vector_add"
    ~block:(Sarek.Execute.dims1d threads)
    ~grid:(Sarek.Execute.dims1d grid_x)
    [
      Sarek.Execute.Vec a;
      Sarek.Execute.Vec b;
      Sarek.Execute.Vec c;
      Sarek.Execute.Int32 (Int32.of_int n);
    ] ;

  Transfer.flush dev ;

  let t1 = Unix.gettimeofday () in
  let time_ms = (t1 -. t0) *. 1000.0 in

  (time_ms, Vector.to_array c)

(* ========== Verification ========== *)

let verify result expected =
  let n = Array.length expected in
  let errors = ref 0 in
  for i = 0 to n - 1 do
    if abs_float (result.(i) -. expected.(i)) > 1e-3 then incr errors
  done ;
  !errors = 0

(* ========== Device filter ========== *)

(* Only run on devices that support external kernels (CUDA, OpenCL, Vulkan) *)
let supports_external_kernels (dev : Device.t) =
  match dev.framework with "CUDA" | "OpenCL" | "Vulkan" -> true | _ -> false

(* ========== Main ========== *)

let () =
  Benchmarks.init () ;
  Benchmarks.run
    ~baseline:ocaml_vector_add
    ~verify
    ~filter:supports_external_kernels
    "External Kernel (Vector Add)"
    run_external_kernel ;
  Benchmarks.exit ()
