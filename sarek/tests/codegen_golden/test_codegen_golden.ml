(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Phase 0A golden-snapshot harness
 *
 * Builds a small set of Sarek_ir_types.kernel values and captures the output
 * of each backend's generate_with_types as committed golden strings.
 *
 * Kernels covered:
 *   1. scalar_vec_add  - simple vector addition (no custom types)
 *   2. record_kernel   - uses a Point2 record type
 *   3. variant_kernel  - uses a simple option-like variant
 *   4. sin_kernel      - Float32.sin intrinsic call
 *
 * Properties tested per backend:
 *   - Byte-exact match against committed golden string
 *   - Two consecutive calls produce identical output (determinism)
 ******************************************************************************)

open Sarek_ir_types

(** {1 Kernel Builders} *)

let make_var name ty =
  {var_name = name; var_id = 0; var_type = ty; var_mutable = false}

let empty_kernel name params locals body =
  {
    kern_name = name;
    kern_params = params;
    kern_locals = locals;
    kern_body = body;
    kern_types = [];
    kern_variants = [];
    kern_funcs = [];
    kern_native_fn = None;
  }

(** Kernel 1: scalar vector-add. Equivalent to: fun (a : float32 vec) (b :
    float32 vec) (c : float32 vec) -> let idx = global_thread_id in c.[idx] <-
    a.[idx] +. b.[idx] *)
let scalar_vec_add_kernel () =
  let a = make_var "a" (TVec TFloat32) in
  let b = make_var "b" (TVec TFloat32) in
  let c = make_var "c" (TVec TFloat32) in
  let idx = make_var "idx" TInt32 in
  let body =
    SLet
      ( idx,
        EIntrinsic ([], "global_thread_id", []),
        SAssign
          ( LArrayElem ("c", EVar idx),
            EBinop (Add, EArrayRead ("a", EVar idx), EArrayRead ("b", EVar idx))
          ) )
  in
  empty_kernel
    "scalar_vec_add"
    [
      DParam (a, Some {arr_elttype = TFloat32; arr_memspace = Global});
      DParam (b, Some {arr_elttype = TFloat32; arr_memspace = Global});
      DParam (c, Some {arr_elttype = TFloat32; arr_memspace = Global});
    ]
    []
    body

(** Kernel 2: record kernel. Uses a Point2 record type [x: float32, y: float32].
    Reads a point, scales it, writes back. *)
let record_kernel () =
  let point_type = TRecord ("Point2", [("x", TFloat32); ("y", TFloat32)]) in
  let pts = make_var "pts" (TVec point_type) in
  let idx = make_var "idx" TInt32 in
  let p = make_var "p" point_type in
  let body =
    SLet
      ( idx,
        EIntrinsic ([], "global_thread_id", []),
        SLet
          ( p,
            EArrayRead ("pts", EVar idx),
            SAssign
              ( LArrayElem ("pts", EVar idx),
                ERecord
                  ( "Point2",
                    [
                      ( "x",
                        EBinop
                          ( Mul,
                            ERecordField (EVar p, "x"),
                            EConst (CFloat32 2.0) ) );
                      ( "y",
                        EBinop
                          ( Mul,
                            ERecordField (EVar p, "y"),
                            EConst (CFloat32 2.0) ) );
                    ] ) ) ) )
  in
  let k = empty_kernel "record_kernel" [DParam (pts, None)] [] body in
  {k with kern_types = [("Point2", [("x", TFloat32); ("y", TFloat32)])]}

(** Kernel 3: variant kernel. Uses a Opt variant: None | Some of float32. Reads
    an int32 flag, writes Some or None. *)
let variant_kernel () =
  let opt_constrs = [("OptNone", []); ("OptSome", [TFloat32])] in
  let opt_type = TVariant ("Opt", opt_constrs) in
  let flags = make_var "flags" (TVec TInt32) in
  let out = make_var "out" (TVec opt_type) in
  let idx = make_var "idx" TInt32 in
  let flag = make_var "flag" TInt32 in
  let body =
    SLet
      ( idx,
        EIntrinsic ([], "global_thread_id", []),
        SLet
          ( flag,
            EArrayRead ("flags", EVar idx),
            SIf
              ( EBinop (Ne, EVar flag, EConst (CInt32 0l)),
                SAssign
                  ( LArrayElem ("out", EVar idx),
                    EVariant ("Opt", "OptSome", [EConst (CFloat32 1.0)]) ),
                Some
                  (SAssign
                     ( LArrayElem ("out", EVar idx),
                       EVariant ("Opt", "OptNone", []) )) ) ) )
  in
  let k =
    empty_kernel
      "variant_kernel"
      [
        DParam (flags, Some {arr_elttype = TInt32; arr_memspace = Global});
        DParam (out, None);
      ]
      []
      body
  in
  {k with kern_variants = [("Opt", opt_constrs)]}

(** Kernel 4: Float32.sin intrinsic call (unqualified path=[]). fun (a : float32
    vec) (b : float32 vec) -> ... b.[idx] <- sin a.[idx] *)
let sin_kernel () =
  let a = make_var "a" (TVec TFloat32) in
  let b = make_var "b" (TVec TFloat32) in
  let idx = make_var "idx" TInt32 in
  let body =
    SLet
      ( idx,
        EIntrinsic ([], "global_thread_id", []),
        SAssign
          ( LArrayElem ("b", EVar idx),
            EIntrinsic ([], "sin", [EArrayRead ("a", EVar idx)]) ) )
  in
  empty_kernel
    "sin_kernel"
    [
      DParam (a, Some {arr_elttype = TFloat32; arr_memspace = Global});
      DParam (b, Some {arr_elttype = TFloat32; arr_memspace = Global});
    ]
    []
    body

(** Kernel 5: Float32.sin path-qualified intrinsic (path=["Float32"]). CUDA must
    emit sinf(); OpenCL/Metal/GLSL emit sin(). This is the PR-2 sinf-fix test
    kernel. *)
let float32_sin_path_kernel () =
  let a = make_var "a" (TVec TFloat32) in
  let b = make_var "b" (TVec TFloat32) in
  let idx = make_var "idx" TInt32 in
  let body =
    SLet
      ( idx,
        EIntrinsic ([], "global_thread_id", []),
        SAssign
          ( LArrayElem ("b", EVar idx),
            EIntrinsic (["Float32"], "sin", [EArrayRead ("a", EVar idx)]) ) )
  in
  empty_kernel
    "float32_sin_path"
    [
      DParam (a, Some {arr_elttype = TFloat32; arr_memspace = Global});
      DParam (b, Some {arr_elttype = TFloat32; arr_memspace = Global});
    ]
    []
    body

(** {1 Backend Adapter Type} *)

type backend = {
  name : string;
  reset : unit -> unit;
  generate : types:(string * (string * elttype) list) list -> kernel -> string;
}

let cuda_backend =
  {
    name = "cuda";
    reset = Gen_cuda.reset_state;
    generate = Gen_cuda.generate_with_types;
  }

let opencl_backend =
  {
    name = "opencl";
    reset = Gen_opencl.reset_state;
    generate = Gen_opencl.generate_with_types;
  }

let metal_backend =
  {
    name = "metal";
    reset = Gen_metal.reset_state;
    generate = Gen_metal.generate_with_types;
  }

let glsl_backend =
  {
    name = "glsl";
    reset = Gen_glsl.reset_state;
    generate = Gen_glsl.generate_with_types;
  }

let wgsl_backend =
  {
    name = "wgsl";
    reset = Gen_wgsl.reset_state;
    generate = Gen_wgsl.generate_with_types;
  }

let all_backends =
  [cuda_backend; opencl_backend; metal_backend; glsl_backend; wgsl_backend]

(** {1 Golden Registry} *)

(** Goldens are committed strings keyed by (backend_name, kernel_name). On first
    use run with GOLDEN_CAPTURE=1 to print actuals then commit. *)
let golden_table : (string * string, string) Hashtbl.t = Hashtbl.create 32

let register_golden backend_name kernel_name s =
  Hashtbl.replace golden_table (backend_name, kernel_name) s

let lookup_golden backend_name kernel_name =
  Hashtbl.find_opt golden_table (backend_name, kernel_name)

(** {1 Committed goldens}

    Captured from main on 2026-06-02. *)

let () =
  (* ---- CUDA goldens ---- *)
  register_golden
    "cuda"
    "scalar_vec_add"
    "\n\
     extern \"C\" {\n\
     __global__ void scalar_vec_add(float* __restrict__ a, int sarek_a_length, \
     float* __restrict__ b, int sarek_b_length, float* __restrict__ c, int \
     sarek_c_length) {\n\
    \  int idx = (threadIdx.x + blockIdx.x * blockDim.x);\n\
    \  c[idx] = (a[idx] + b[idx]);\n\
     }\n\
     }\n" ;

  register_golden
    "cuda"
    "record_kernel"
    "\n\
     extern \"C\" {\n\
     typedef struct {\n\
    \  float x;\n\
    \  float y;\n\
     } Point2;\n\n\
     __global__ void record_kernel(Point2* __restrict__ pts, int \
     sarek_pts_length) {\n\
    \  int idx = (threadIdx.x + blockIdx.x * blockDim.x);\n\
    \  Point2 p = pts[idx];\n\
    \  pts[idx].x = (p.x * 2.0f);\n\
    \  pts[idx].y = (p.y * 2.0f);\n\
     }\n\
     }\n" ;

  register_golden
    "cuda"
    "variant_kernel"
    "\n\
     extern \"C\" {\n\
     enum { OptNone = 0, OptSome = 1 };\n\
     typedef struct {\n\
    \  int tag;\n\
    \  union {\n\
    \    float OptSome_v;\n\
    \  } data;\n\
     } Opt;\n\n\
     __device__ __host__ inline Opt make_Opt_OptNone() {\n\
    \  Opt r;\n\
    \  r.tag = OptNone;\n\
    \  return r;\n\
     }\n\n\
     __device__ __host__ inline Opt make_Opt_OptSome(float v) {\n\
    \  Opt r;\n\
    \  r.tag = OptSome;\n\
    \  r.data.OptSome_v = v;\n\
    \  return r;\n\
     }\n\n\
     __global__ void variant_kernel(int* __restrict__ flags, int \
     sarek_flags_length, Opt* __restrict__ out, int sarek_out_length) {\n\
    \  int idx = (threadIdx.x + blockIdx.x * blockDim.x);\n\
    \  int flag = flags[idx];\n\
    \  if ((flag != 0)) {\n\
    \    out[idx] = make_Opt_OptSome(1.0f);\n\
    \  } else {\n\
    \    out[idx] = OptNone;\n\
    \  }\n\
     }\n\
     }\n" ;

  register_golden
    "cuda"
    "sin_kernel"
    "\n\
     extern \"C\" {\n\
     __global__ void sin_kernel(float* __restrict__ a, int sarek_a_length, \
     float* __restrict__ b, int sarek_b_length) {\n\
    \  int idx = (threadIdx.x + blockIdx.x * blockDim.x);\n\
    \  b[idx] = sin(a[idx]);\n\
     }\n\
     }\n" ;

  (* ---- OpenCL goldens ---- *)
  register_golden
    "opencl"
    "scalar_vec_add"
    "__kernel void scalar_vec_add(__global float* restrict a, int \
     sarek_a_length, __global float* restrict b, int sarek_b_length, __global \
     float* restrict c, int sarek_c_length) {\n\
    \  int idx = get_global_id(0);\n\
    \  c[idx] = (a[idx] + b[idx]);\n\
     }\n" ;

  register_golden
    "opencl"
    "record_kernel"
    "typedef struct {\n\
    \  float x;\n\
    \  float y;\n\
     } Point2;\n\n\
     __kernel void record_kernel(__global Point2* restrict pts, int \
     sarek_pts_length) {\n\
    \  int idx = get_global_id(0);\n\
    \  Point2 p = pts[idx];\n\
    \  pts[idx] = (Point2){.x = (p.x * 2.0f), .y = (p.y * 2.0f)};\n\
     }\n" ;

  register_golden
    "opencl"
    "variant_kernel"
    "enum { OptNone = 0, OptSome = 1 };\n\
     typedef struct {\n\
    \  int tag;\n\
    \  union {\n\
    \    float OptSome_v;\n\
    \  } data;\n\
     } Opt;\n\n\
     static inline Opt make_Opt_OptNone() {\n\
    \  Opt r;\n\
    \  r.tag = OptNone;\n\
    \  return r;\n\
     }\n\n\
     static inline Opt make_Opt_OptSome(float v) {\n\
    \  Opt r;\n\
    \  r.tag = OptSome;\n\
    \  r.data.OptSome_v = v;\n\
    \  return r;\n\
     }\n\n\
     __kernel void variant_kernel(__global int* restrict flags, int \
     sarek_flags_length, __global Opt* restrict out, int sarek_out_length) {\n\
    \  int idx = get_global_id(0);\n\
    \  int flag = flags[idx];\n\
    \  if ((flag != 0)) {\n\
    \    out[idx] = make_Opt_OptSome(1.0f);\n\
    \  } else {\n\
    \    out[idx] = make_Opt_OptNone();\n\
    \  }\n\
     }\n" ;

  register_golden
    "opencl"
    "sin_kernel"
    "__kernel void sin_kernel(__global float* restrict a, int sarek_a_length, \
     __global float* restrict b, int sarek_b_length) {\n\
    \  int idx = get_global_id(0);\n\
    \  b[idx] = sin(a[idx]);\n\
     }\n" ;

  (* ---- Metal goldens ---- *)
  register_golden
    "metal"
    "scalar_vec_add"
    "#include <metal_stdlib>\n\
     using namespace metal;\n\n\
     kernel void scalar_vec_add(device float* a [[buffer(0)]], constant int \
     &sarek_a_length [[buffer(1)]], device float* b [[buffer(2)]], constant \
     int &sarek_b_length [[buffer(3)]], device float* c [[buffer(4)]], \
     constant int &sarek_c_length [[buffer(5)]],\n\
     uint3 __metal_gid [[thread_position_in_grid]],\n\
     uint3 __metal_tid [[thread_position_in_threadgroup]],\n\
     uint3 __metal_bid [[threadgroup_position_in_grid]],\n\
     uint3 __metal_tpg [[threads_per_threadgroup]],\n\
     uint3 __metal_num_groups [[threadgroups_per_grid]]) {\n\
    \  int idx = __metal_gid.x;\n\
    \  c[idx] = (a[idx] + b[idx]);\n\
     }\n\n" ;

  register_golden
    "metal"
    "record_kernel"
    "#include <metal_stdlib>\n\
     using namespace metal;\n\n\
     typedef struct {\n\
    \  float x;\n\
    \  float y;\n\
     } Point2;\n\n\
     kernel void record_kernel(constant Point2* &pts [[buffer(0)]], constant \
     int &sarek_pts_length [[buffer(1)]],\n\
     uint3 __metal_gid [[thread_position_in_grid]],\n\
     uint3 __metal_tid [[thread_position_in_threadgroup]],\n\
     uint3 __metal_bid [[threadgroup_position_in_grid]],\n\
     uint3 __metal_tpg [[threads_per_threadgroup]],\n\
     uint3 __metal_num_groups [[threadgroups_per_grid]]) {\n\
    \  int idx = __metal_gid.x;\n\
    \  Point2 p = pts[idx];\n\
    \  pts[idx] = (Point2){.x = (p.x * 2.0f), .y = (p.y * 2.0f)};\n\
     }\n\n" ;

  register_golden
    "metal"
    "variant_kernel"
    "#include <metal_stdlib>\n\
     using namespace metal;\n\n\
     enum { OptNone = 0, OptSome = 1 };\n\
     typedef struct {\n\
    \  int tag;\n\
    \  union {\n\
    \    float OptSome_v;\n\
    \  } data;\n\
     } Opt;\n\n\
     static inline Opt make_Opt_OptNone() {\n\
    \  Opt r;\n\
    \  r.tag = OptNone;\n\
    \  return r;\n\
     }\n\n\
     static inline Opt make_Opt_OptSome(float v) {\n\
    \  Opt r;\n\
    \  r.tag = OptSome;\n\
    \  r.data.OptSome_v = v;\n\
    \  return r;\n\
     }\n\n\
     kernel void variant_kernel(device int* flags [[buffer(0)]], constant int \
     &sarek_flags_length [[buffer(1)]], constant Opt* &out [[buffer(2)]], \
     constant int &sarek_out_length [[buffer(3)]],\n\
     uint3 __metal_gid [[thread_position_in_grid]],\n\
     uint3 __metal_tid [[thread_position_in_threadgroup]],\n\
     uint3 __metal_bid [[threadgroup_position_in_grid]],\n\
     uint3 __metal_tpg [[threads_per_threadgroup]],\n\
     uint3 __metal_num_groups [[threadgroups_per_grid]]) {\n\
    \  int idx = __metal_gid.x;\n\
    \  int flag = flags[idx];\n\
    \  if ((flag != 0)) {\n\
    \    out[idx] = make_Opt_OptSome(1.0f);\n\
    \  } else {\n\
    \    out[idx] = make_Opt_OptNone();\n\
    \  }\n\
     }\n\n" ;

  register_golden
    "metal"
    "sin_kernel"
    "#include <metal_stdlib>\n\
     using namespace metal;\n\n\
     kernel void sin_kernel(device float* a [[buffer(0)]], constant int \
     &sarek_a_length [[buffer(1)]], device float* b [[buffer(2)]], constant \
     int &sarek_b_length [[buffer(3)]],\n\
     uint3 __metal_gid [[thread_position_in_grid]],\n\
     uint3 __metal_tid [[thread_position_in_threadgroup]],\n\
     uint3 __metal_bid [[threadgroup_position_in_grid]],\n\
     uint3 __metal_tpg [[threads_per_threadgroup]],\n\
     uint3 __metal_num_groups [[threadgroups_per_grid]]) {\n\
    \  int idx = __metal_gid.x;\n\
    \  b[idx] = sin(a[idx]);\n\
     }\n\n" ;

  (* ---- GLSL goldens ---- *)
  register_golden
    "glsl"
    "scalar_vec_add"
    "#version 450\n\n\
     // Sarek-generated compute shader: scalar_vec_add\n\
     layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;\n\n\
     layout(std430, set=0, binding = 0) buffer Buffer_a {\n\
    \  float a[];\n\
     };\n\
     layout(std430, set=0, binding = 1) buffer Buffer_b {\n\
    \  float b[];\n\
     };\n\
     layout(std430, set=0, binding = 2) buffer Buffer_c {\n\
    \  float c[];\n\
     };\n\
     layout(push_constant) uniform PushConstants {\n\
    \  int a_len;\n\
    \  int b_len;\n\
    \  int c_len;\n\
     } pc;\n\n\
     #define a_len pc.a_len\n\
     #define b_len pc.b_len\n\
     #define c_len pc.c_len\n\n\
     void main() {\n\
    \  int idx = int(gl_GlobalInvocationID.x);\n\
    \  c[idx] = (a[idx] + b[idx]);\n\
     }\n" ;

  register_golden
    "glsl"
    "record_kernel"
    "#version 450\n\n\
     // Sarek-generated compute shader: record_kernel\n\
     layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;\n\n\
     struct Point2 {\n\
    \  float x;\n\
    \  float y;\n\
     };\n\n\
     layout(std430, set=0, binding = 0) buffer Buffer_pts {\n\
    \  Point2 pts[];\n\
     };\n\
     layout(push_constant) uniform PushConstants {\n\
    \  int pts_len;\n\
     } pc;\n\n\
     #define pts_len pc.pts_len\n\n\
     void main() {\n\
    \  int idx = int(gl_GlobalInvocationID.x);\n\
    \  Point2 p = pts[idx];\n\
    \  pts[idx] = Point2((p.x * 2.0), (p.y * 2.0));\n\
     }\n" ;

  register_golden
    "glsl"
    "variant_kernel"
    "#version 450\n\n\
     // Sarek-generated compute shader: variant_kernel\n\
     layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;\n\n\
     const int OptNone = 0;\n\
     const int OptSome = 1;\n\n\
     struct Opt {\n\
    \  int tag;\n\
    \  float OptSome_v;\n\
     };\n\n\
     Opt make_Opt_OptNone() {\n\
    \  Opt r;\n\
    \  r.tag = OptNone;\n\
    \  return r;\n\
     }\n\n\
     Opt make_Opt_OptSome(float v) {\n\
    \  Opt r;\n\
    \  r.tag = OptSome;\n\
    \  r.OptSome_v = v;\n\
    \  return r;\n\
     }\n\n\
     layout(std430, set=0, binding = 0) buffer Buffer_flags {\n\
    \  int flags[];\n\
     };\n\
     layout(std430, set=0, binding = 1) buffer Buffer_outv {\n\
    \  Opt outv[];\n\
     };\n\
     layout(push_constant) uniform PushConstants {\n\
    \  int flags_len;\n\
    \  int outv_len;\n\
     } pc;\n\n\
     #define flags_len pc.flags_len\n\
     #define outv_len pc.outv_len\n\n\
     void main() {\n\
    \  int idx = int(gl_GlobalInvocationID.x);\n\
    \  int flag = flags[idx];\n\
    \  if ((flag != 0)) {\n\
    \    outv[idx] = make_Opt_OptSome(1.0);\n\
    \  } else {\n\
    \    outv[idx] = make_Opt_OptNone();\n\
    \  }\n\
     }\n" ;

  register_golden
    "glsl"
    "sin_kernel"
    "#version 450\n\n\
     // Sarek-generated compute shader: sin_kernel\n\
     layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;\n\n\
     layout(std430, set=0, binding = 0) buffer Buffer_a {\n\
    \  float a[];\n\
     };\n\
     layout(std430, set=0, binding = 1) buffer Buffer_b {\n\
    \  float b[];\n\
     };\n\
     layout(push_constant) uniform PushConstants {\n\
    \  int a_len;\n\
    \  int b_len;\n\
     } pc;\n\n\
     #define a_len pc.a_len\n\
     #define b_len pc.b_len\n\n\
     void main() {\n\
    \  int idx = int(gl_GlobalInvocationID.x);\n\
    \  b[idx] = sin(a[idx]);\n\
     }\n" ;

  (* ---- float32_sin_path goldens (PR-2 sinf-fix kernel) ---- *)
  (* CUDA: sinf (f-suffix for Float32 path-qualified math) *)
  register_golden
    "cuda"
    "float32_sin_path"
    "\n\
     extern \"C\" {\n\
     __global__ void float32_sin_path(float* __restrict__ a, int \
     sarek_a_length, float* __restrict__ b, int sarek_b_length) {\n\
    \  int idx = (threadIdx.x + blockIdx.x * blockDim.x);\n\
    \  b[idx] = sinf(a[idx]);\n\
     }\n\
     }\n" ;

  (* OpenCL: sin (un-suffixed for Float32) *)
  register_golden
    "opencl"
    "float32_sin_path"
    "__kernel void float32_sin_path(__global float* restrict a, int \
     sarek_a_length, __global float* restrict b, int sarek_b_length) {\n\
    \  int idx = get_global_id(0);\n\
    \  b[idx] = sin(a[idx]);\n\
     }\n" ;

  (* Metal: sin (un-suffixed for Float32) *)
  register_golden
    "metal"
    "float32_sin_path"
    "#include <metal_stdlib>\n\
     using namespace metal;\n\n\
     kernel void float32_sin_path(device float* a [[buffer(0)]], constant int \
     &sarek_a_length [[buffer(1)]], device float* b [[buffer(2)]], constant \
     int &sarek_b_length [[buffer(3)]],\n\
     uint3 __metal_gid [[thread_position_in_grid]],\n\
     uint3 __metal_tid [[thread_position_in_threadgroup]],\n\
     uint3 __metal_bid [[threadgroup_position_in_grid]],\n\
     uint3 __metal_tpg [[threads_per_threadgroup]],\n\
     uint3 __metal_num_groups [[threadgroups_per_grid]]) {\n\
    \  int idx = __metal_gid.x;\n\
    \  b[idx] = sin(a[idx]);\n\
     }\n\n" ;

  (* GLSL: sin (un-suffixed for Float32) *)
  register_golden
    "glsl"
    "float32_sin_path"
    "#version 450\n\n\
     // Sarek-generated compute shader: float32_sin_path\n\
     layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;\n\n\
     layout(std430, set=0, binding = 0) buffer Buffer_a {\n\
    \  float a[];\n\
     };\n\
     layout(std430, set=0, binding = 1) buffer Buffer_b {\n\
    \  float b[];\n\
     };\n\
     layout(push_constant) uniform PushConstants {\n\
    \  int a_len;\n\
    \  int b_len;\n\
     } pc;\n\n\
     #define a_len pc.a_len\n\
     #define b_len pc.b_len\n\n\
     void main() {\n\
    \  int idx = int(gl_GlobalInvocationID.x);\n\
    \  b[idx] = sin(a[idx]);\n\
     }\n" ;

  (* ---- WGSL goldens ---- *)
  register_golden
    "wgsl"
    "scalar_vec_add"
    "@group(0) @binding(0) var<storage, read_write> a : array<f32>;\n\
     @group(0) @binding(1) var<storage, read_write> b : array<f32>;\n\
     @group(0) @binding(2) var<storage, read_write> c : array<f32>;\n\
     struct Params {\n\
    \  sarek_a_length : i32,\n\
    \  sarek_b_length : i32,\n\
    \  sarek_c_length : i32,\n\
     }\n\
     @group(0) @binding(3) var<uniform> params : Params;\n\n\
     // Sarek-generated compute shader: scalar_vec_add\n\
     @compute @workgroup_size(256, 1, 1)\n\
     fn main(@builtin(global_invocation_id) gid : vec3<u32>) {\n\
    \  let idx : i32 = i32(gid.x);\n\
    \  c[idx] = (a[idx] + b[idx]);\n\
     }\n" ;

  register_golden
    "wgsl"
    "record_kernel"
    "struct Point2 {\n\
    \  x : f32,\n\
    \  y : f32,\n\
     }\n\n\
     @group(0) @binding(0) var<storage, read_write> pts : array<Point2>;\n\
     struct Params {\n\
    \  sarek_pts_length : i32,\n\
     }\n\
     @group(0) @binding(1) var<uniform> params : Params;\n\n\
     // Sarek-generated compute shader: record_kernel\n\
     @compute @workgroup_size(256, 1, 1)\n\
     fn main(@builtin(global_invocation_id) gid : vec3<u32>) {\n\
    \  let idx : i32 = i32(gid.x);\n\
    \  let p : Point2 = pts[idx];\n\
    \  pts[idx] = Point2((p.x * 2.0f), (p.y * 2.0f));\n\
     }\n" ;

  register_golden
    "wgsl"
    "variant_kernel"
    "const OptNone : i32 = 0i;\n\
     const OptSome : i32 = 1i;\n\n\
     struct Opt {\n\
    \  tag : i32,\n\
    \  OptSome_v : f32,\n\
     }\n\n\
     fn make_Opt_OptNone() -> Opt {\n\
    \  var r : Opt;\n\
    \  r.tag = OptNone;\n\
    \  return r;\n\
     }\n\n\
     fn make_Opt_OptSome(v : f32) -> Opt {\n\
    \  var r : Opt;\n\
    \  r.tag = OptSome;\n\
    \  r.OptSome_v = v;\n\
    \  return r;\n\
     }\n\n\
     @group(0) @binding(0) var<storage, read_write> flags : array<i32>;\n\
     @group(0) @binding(1) var<storage, read_write> out : array<Opt>;\n\
     struct Params {\n\
    \  sarek_flags_length : i32,\n\
    \  sarek_out_length : i32,\n\
     }\n\
     @group(0) @binding(2) var<uniform> params : Params;\n\n\
     // Sarek-generated compute shader: variant_kernel\n\
     @compute @workgroup_size(256, 1, 1)\n\
     fn main(@builtin(global_invocation_id) gid : vec3<u32>) {\n\
    \  let idx : i32 = i32(gid.x);\n\
    \  let flag : i32 = flags[idx];\n\
    \  if ((flag != 0i)) {\n\
    \    out[idx] = make_Opt_OptSome(1.0f);\n\
    \  } else {\n\
    \    out[idx] = make_Opt_OptNone();\n\
    \  }\n\
     }\n" ;

  register_golden
    "wgsl"
    "sin_kernel"
    "@group(0) @binding(0) var<storage, read_write> a : array<f32>;\n\
     @group(0) @binding(1) var<storage, read_write> b : array<f32>;\n\
     struct Params {\n\
    \  sarek_a_length : i32,\n\
    \  sarek_b_length : i32,\n\
     }\n\
     @group(0) @binding(2) var<uniform> params : Params;\n\n\
     // Sarek-generated compute shader: sin_kernel\n\
     @compute @workgroup_size(256, 1, 1)\n\
     fn main(@builtin(global_invocation_id) gid : vec3<u32>) {\n\
    \  let idx : i32 = i32(gid.x);\n\
    \  b[idx] = sin(a[idx]);\n\
     }\n" ;

  (* WGSL: sin (un-suffixed for Float32, matching GLSL/OpenCL/Metal) *)
  register_golden
    "wgsl"
    "float32_sin_path"
    "@group(0) @binding(0) var<storage, read_write> a : array<f32>;\n\
     @group(0) @binding(1) var<storage, read_write> b : array<f32>;\n\
     struct Params {\n\
    \  sarek_a_length : i32,\n\
    \  sarek_b_length : i32,\n\
     }\n\
     @group(0) @binding(2) var<uniform> params : Params;\n\n\
     // Sarek-generated compute shader: float32_sin_path\n\
     @compute @workgroup_size(256, 1, 1)\n\
     fn main(@builtin(global_invocation_id) gid : vec3<u32>) {\n\
    \  let idx : i32 = i32(gid.x);\n\
    \  b[idx] = sin(a[idx]);\n\
     }\n"

(** {1 Kernel list for test iteration} *)

let test_kernels () =
  [
    ("scalar_vec_add", scalar_vec_add_kernel ());
    ("record_kernel", record_kernel ());
    ("variant_kernel", variant_kernel ());
    ("sin_kernel", sin_kernel ());
    ("float32_sin_path", float32_sin_path_kernel ());
  ]

(** {1 Test helpers} *)

(** Run backend on kernel twice and assert identical output (determinism check)
*)
let check_determinism backend kernel_name k =
  backend.reset () ;
  let first = backend.generate ~types:k.kern_types k in
  backend.reset () ;
  let second = backend.generate ~types:k.kern_types k in
  if first <> second then
    Alcotest.failf "Non-deterministic output for %s/%s" backend.name kernel_name ;
  first

(** Assert byte-exact match against golden, or print actual if GOLDEN_CAPTURE=1
*)
let check_golden backend_name kernel_name actual =
  match Sys.getenv_opt "GOLDEN_CAPTURE" with
  | Some "1" ->
      Printf.printf
        "\n=== GOLDEN %s/%s ===\n%s\n=== END ===\n%!"
        backend_name
        kernel_name
        actual
  | _ -> (
      match lookup_golden backend_name kernel_name with
      | None ->
          Alcotest.failf
            "No golden registered for %s/%s - run with GOLDEN_CAPTURE=1 to \
             capture"
            backend_name
            kernel_name
      | Some expected ->
          if actual <> expected then begin
            Printf.eprintf "=== DIFF for %s/%s ===\n" backend_name kernel_name ;
            Printf.eprintf
              "--- expected ---\n%s\n--- actual ---\n%s\n"
              expected
              actual
          end ;
          Alcotest.(check string)
            (Printf.sprintf "%s/%s byte-exact" backend_name kernel_name)
            expected
            actual)

(** {1 Test cases} *)

let make_backend_tests backend =
  let tests =
    List.map
      (fun (kernel_name, k) ->
        Alcotest.test_case
          (Printf.sprintf "%s/%s" backend.name kernel_name)
          `Quick
          (fun () ->
            let actual = check_determinism backend kernel_name k in
            check_golden backend.name kernel_name actual))
      (test_kernels ())
  in
  (backend.name, tests)

let () =
  Alcotest.run "codegen_golden" (List.map make_backend_tests all_backends)
