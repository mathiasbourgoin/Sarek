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

(** Kernel 5b/5c/5d: path-qualified Float32 intrinsics whose GLSL builtin name
    differs from the OpenCL/Metal/CUDA generic name. GLSL has no [fabs] or
    [rsqrt] builtin (only [abs] and [inversesqrt]), and no [atan2] (the
    two-argument arctangent is the [atan] overload). These exist to prove the
    pure-registry framework dispatch renames correctly per backend, not just for
    [sin] (which happens to be spelled the same everywhere). *)
let float32_rsqrt_path_kernel () =
  let a = make_var "a" (TVec TFloat32) in
  let b = make_var "b" (TVec TFloat32) in
  let idx = make_var "idx" TInt32 in
  let body =
    SLet
      ( idx,
        EIntrinsic ([], "global_thread_id", []),
        SAssign
          ( LArrayElem ("b", EVar idx),
            EIntrinsic (["Float32"], "rsqrt", [EArrayRead ("a", EVar idx)]) ) )
  in
  empty_kernel
    "float32_rsqrt_path"
    [
      DParam (a, Some {arr_elttype = TFloat32; arr_memspace = Global});
      DParam (b, Some {arr_elttype = TFloat32; arr_memspace = Global});
    ]
    []
    body

let float32_abs_float_path_kernel () =
  let a = make_var "a" (TVec TFloat32) in
  let b = make_var "b" (TVec TFloat32) in
  let idx = make_var "idx" TInt32 in
  let body =
    SLet
      ( idx,
        EIntrinsic ([], "global_thread_id", []),
        SAssign
          ( LArrayElem ("b", EVar idx),
            EIntrinsic (["Float32"], "abs_float", [EArrayRead ("a", EVar idx)])
          ) )
  in
  empty_kernel
    "float32_abs_float_path"
    [
      DParam (a, Some {arr_elttype = TFloat32; arr_memspace = Global});
      DParam (b, Some {arr_elttype = TFloat32; arr_memspace = Global});
    ]
    []
    body

(* Float64.abs_float reaches the GLSL generator's hardcoded match arm (it is
   absent from Sarek_pure_registry.float64_list), and must lower to abs() — not
   the raw Float64.abs_float(...) that glslang rejects. See Sarek_ir_glsl.ml. *)
let float64_abs_float_path_kernel () =
  let a = make_var "a" (TVec TFloat64) in
  let b = make_var "b" (TVec TFloat64) in
  let idx = make_var "idx" TInt32 in
  let body =
    SLet
      ( idx,
        EIntrinsic ([], "global_thread_id", []),
        SAssign
          ( LArrayElem ("b", EVar idx),
            EIntrinsic (["Float64"], "abs_float", [EArrayRead ("a", EVar idx)])
          ) )
  in
  empty_kernel
    "float64_abs_float_path"
    [
      DParam (a, Some {arr_elttype = TFloat64; arr_memspace = Global});
      DParam (b, Some {arr_elttype = TFloat64; arr_memspace = Global});
    ]
    []
    body

(* Float64.copysign has no GLSL builtin (and is absent from
   Sarek_pure_registry.float64_list), so pre-fix it fell through to the raw-name
   fallback and emitted [Float64.copysign(...)], which glslang parses as a
   swizzle on a [Float64] variable and rejects. It must lower to a call to the
   bit-level [sarek_copysign] helper (emitted in the preamble). See
   Sarek_ir_glsl.ml. *)
let float64_copysign_path_kernel () =
  let a = make_var "a" (TVec TFloat64) in
  let b = make_var "b" (TVec TFloat64) in
  let c = make_var "c" (TVec TFloat64) in
  let idx = make_var "idx" TInt32 in
  let body =
    SLet
      ( idx,
        EIntrinsic ([], "global_thread_id", []),
        SAssign
          ( LArrayElem ("c", EVar idx),
            EIntrinsic
              ( ["Float64"],
                "copysign",
                [EArrayRead ("a", EVar idx); EArrayRead ("b", EVar idx)] ) ) )
  in
  empty_kernel
    "float64_copysign_path"
    [
      DParam (a, Some {arr_elttype = TFloat64; arr_memspace = Global});
      DParam (b, Some {arr_elttype = TFloat64; arr_memspace = Global});
      DParam (c, Some {arr_elttype = TFloat64; arr_memspace = Global});
    ]
    []
    body

(* Float32.copysign resolves through the pure registry to the raw un-suffixed
   [copysign(...)] (no GLSL builtin), so it too must lower to the
   [sarek_copysign] helper — here the float overload only, as the kernel is not
   float64. *)
let float32_copysign_path_kernel () =
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
            EIntrinsic
              ( ["Float32"],
                "copysign",
                [EArrayRead ("a", EVar idx); EArrayRead ("b", EVar idx)] ) ) )
  in
  empty_kernel
    "float32_copysign_path"
    [
      DParam (a, Some {arr_elttype = TFloat32; arr_memspace = Global});
      DParam (b, Some {arr_elttype = TFloat32; arr_memspace = Global});
      DParam (c, Some {arr_elttype = TFloat32; arr_memspace = Global});
    ]
    []
    body

let float32_atan2_path_kernel () =
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
            EIntrinsic
              ( ["Float32"],
                "atan2",
                [EArrayRead ("a", EVar idx); EArrayRead ("b", EVar idx)] ) ) )
  in
  empty_kernel
    "float32_atan2_path"
    [
      DParam (a, Some {arr_elttype = TFloat32; arr_memspace = Global});
      DParam (b, Some {arr_elttype = TFloat32; arr_memspace = Global});
      DParam (c, Some {arr_elttype = TFloat32; arr_memspace = Global});
    ]
    []
    body

(** Kernel 5e/5f/5g/5h: path-qualified Float32 intrinsics with NO GLSL/Metal
    builtin under any name (cbrt, hypot, expm1, log1p) — unlike
    fabs/rsqrt/atan2, these require a multi-token expression polyfill, not a
    rename. CUDA/OpenCL do have direct builtins for these, so only
    glsl_only/metal_only goldens are needed. *)
let float32_cbrt_path_kernel () =
  let a = make_var "a" (TVec TFloat32) in
  let b = make_var "b" (TVec TFloat32) in
  let idx = make_var "idx" TInt32 in
  let body =
    SLet
      ( idx,
        EIntrinsic ([], "global_thread_id", []),
        SAssign
          ( LArrayElem ("b", EVar idx),
            EIntrinsic (["Float32"], "cbrt", [EArrayRead ("a", EVar idx)]) ) )
  in
  empty_kernel
    "float32_cbrt_path"
    [
      DParam (a, Some {arr_elttype = TFloat32; arr_memspace = Global});
      DParam (b, Some {arr_elttype = TFloat32; arr_memspace = Global});
    ]
    []
    body

let float32_hypot_path_kernel () =
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
            EIntrinsic
              ( ["Float32"],
                "hypot",
                [EArrayRead ("a", EVar idx); EArrayRead ("b", EVar idx)] ) ) )
  in
  empty_kernel
    "float32_hypot_path"
    [
      DParam (a, Some {arr_elttype = TFloat32; arr_memspace = Global});
      DParam (b, Some {arr_elttype = TFloat32; arr_memspace = Global});
      DParam (c, Some {arr_elttype = TFloat32; arr_memspace = Global});
    ]
    []
    body

let float32_expm1_path_kernel () =
  let a = make_var "a" (TVec TFloat32) in
  let b = make_var "b" (TVec TFloat32) in
  let idx = make_var "idx" TInt32 in
  let body =
    SLet
      ( idx,
        EIntrinsic ([], "global_thread_id", []),
        SAssign
          ( LArrayElem ("b", EVar idx),
            EIntrinsic (["Float32"], "expm1", [EArrayRead ("a", EVar idx)]) ) )
  in
  empty_kernel
    "float32_expm1_path"
    [
      DParam (a, Some {arr_elttype = TFloat32; arr_memspace = Global});
      DParam (b, Some {arr_elttype = TFloat32; arr_memspace = Global});
    ]
    []
    body

let float32_log1p_path_kernel () =
  let a = make_var "a" (TVec TFloat32) in
  let b = make_var "b" (TVec TFloat32) in
  let idx = make_var "idx" TInt32 in
  let body =
    SLet
      ( idx,
        EIntrinsic ([], "global_thread_id", []),
        SAssign
          ( LArrayElem ("b", EVar idx),
            EIntrinsic (["Float32"], "log1p", [EArrayRead ("a", EVar idx)]) ) )
  in
  empty_kernel
    "float32_log1p_path"
    [
      DParam (a, Some {arr_elttype = TFloat32; arr_memspace = Global});
      DParam (b, Some {arr_elttype = TFloat32; arr_memspace = Global});
    ]
    []
    body

(** Kernel 6: bounds-check with if-expression. WGSL-specific: exercises EIf
    which must emit [select(else, then, cond)] (no ternary in WGSL). fun (a :
    float32 vec) (b : float32 vec) (n : int32) -> let i = global_thread_id in
    b.(i) <- if i < n then a.(i) else 0.0 *)
let bounds_check_kernel () =
  let a = make_var "a" (TVec TFloat32) in
  let b = make_var "b" (TVec TFloat32) in
  let n = make_var "n" TInt32 in
  let idx = make_var "idx" TInt32 in
  let body =
    SLet
      ( idx,
        EIntrinsic ([], "global_thread_id", []),
        SAssign
          ( LArrayElem ("b", EVar idx),
            EIf
              ( EBinop (Lt, EVar idx, EVar n),
                EArrayRead ("a", EVar idx),
                EConst (CFloat32 0.0) ) ) )
  in
  empty_kernel
    "bounds_check"
    [
      DParam (a, Some {arr_elttype = TFloat32; arr_memspace = Global});
      DParam (b, Some {arr_elttype = TFloat32; arr_memspace = Global});
      DParam (n, None);
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
     fn main(\n\
    \  @builtin(global_invocation_id) sarek_gid : vec3<u32>,\n\
    \  @builtin(local_invocation_id) sarek_lid : vec3<u32>,\n\
    \  @builtin(workgroup_id) sarek_wid : vec3<u32>,\n\
    \  @builtin(num_workgroups) sarek_nwg : vec3<u32>\n\
     ) {\n\
    \  let idx : i32 = i32(sarek_gid.x);\n\
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
     fn main(\n\
    \  @builtin(global_invocation_id) sarek_gid : vec3<u32>,\n\
    \  @builtin(local_invocation_id) sarek_lid : vec3<u32>,\n\
    \  @builtin(workgroup_id) sarek_wid : vec3<u32>,\n\
    \  @builtin(num_workgroups) sarek_nwg : vec3<u32>\n\
     ) {\n\
    \  let idx : i32 = i32(sarek_gid.x);\n\
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
     fn main(\n\
    \  @builtin(global_invocation_id) sarek_gid : vec3<u32>,\n\
    \  @builtin(local_invocation_id) sarek_lid : vec3<u32>,\n\
    \  @builtin(workgroup_id) sarek_wid : vec3<u32>,\n\
    \  @builtin(num_workgroups) sarek_nwg : vec3<u32>\n\
     ) {\n\
    \  let idx : i32 = i32(sarek_gid.x);\n\
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
     fn main(\n\
    \  @builtin(global_invocation_id) sarek_gid : vec3<u32>,\n\
    \  @builtin(local_invocation_id) sarek_lid : vec3<u32>,\n\
    \  @builtin(workgroup_id) sarek_wid : vec3<u32>,\n\
    \  @builtin(num_workgroups) sarek_nwg : vec3<u32>\n\
     ) {\n\
    \  let idx : i32 = i32(sarek_gid.x);\n\
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
     fn main(\n\
    \  @builtin(global_invocation_id) sarek_gid : vec3<u32>,\n\
    \  @builtin(local_invocation_id) sarek_lid : vec3<u32>,\n\
    \  @builtin(workgroup_id) sarek_wid : vec3<u32>,\n\
    \  @builtin(num_workgroups) sarek_nwg : vec3<u32>\n\
     ) {\n\
    \  let idx : i32 = i32(sarek_gid.x);\n\
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

(** {1 WGSL-only golden tests}

    These kernels exercise WGSL-specific correctness fixes (select, thread ids,
    etc.) that are not applicable or observable on other backends. They are
    tested only against the wgsl backend to keep the cross-backend loop clean.
*)

let () =
  (* WGSL bounds_check: EIf must emit select(else, then, cond) — no ternary in
     WGSL. This golden is WGSL-only because other backends do support ternary. *)
  register_golden
    "wgsl"
    "bounds_check"
    "@group(0) @binding(0) var<storage, read_write> a : array<f32>;\n\
     @group(0) @binding(1) var<storage, read_write> b : array<f32>;\n\
     struct Params {\n\
    \  sarek_a_length : i32,\n\
    \  sarek_b_length : i32,\n\
    \  n : i32,\n\
     }\n\
     @group(0) @binding(2) var<uniform> params : Params;\n\n\
     // Sarek-generated compute shader: bounds_check\n\
     @compute @workgroup_size(256, 1, 1)\n\
     fn main(\n\
    \  @builtin(global_invocation_id) sarek_gid : vec3<u32>,\n\
    \  @builtin(local_invocation_id) sarek_lid : vec3<u32>,\n\
    \  @builtin(workgroup_id) sarek_wid : vec3<u32>,\n\
    \  @builtin(num_workgroups) sarek_nwg : vec3<u32>\n\
     ) {\n\
    \  let idx : i32 = i32(sarek_gid.x);\n\
    \  b[idx] = select(0.0f, a[idx], (idx < params.n));\n\
     }\n"

let wgsl_only_kernels () = [("bounds_check", bounds_check_kernel ())]

let wgsl_only_tests () =
  List.map
    (fun (kernel_name, k) ->
      Alcotest.test_case
        (Printf.sprintf "wgsl/%s" kernel_name)
        `Quick
        (fun () ->
          Gen_wgsl.reset_state () ;
          let actual = Gen_wgsl.generate_with_types ~types:[] k in
          check_golden "wgsl" kernel_name actual))
    (wgsl_only_kernels ())

(** {1 GLSL-only golden tests}

    GLSL's math builtin names diverge from the CUDA/OpenCL/Metal generic names
    for a few functions: [fabs] -> [abs], [rsqrt] -> [inversesqrt], [atan2] ->
    the two-arg [atan] overload. These are path-qualified Float32 intrinsics, so
    they resolve through [Sarek_pure_registry], not through the per-backend
    unqualified-name match arms. Only GLSL needs its own goldens here because
    CUDA/OpenCL/Metal/WGSL all keep the generic spelling. *)

let () =
  register_golden
    "glsl"
    "float32_rsqrt_path"
    "#version 450\n\n\
     // Sarek-generated compute shader: float32_rsqrt_path\n\
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
    \  b[idx] = inversesqrt(a[idx]);\n\
     }\n" ;

  register_golden
    "glsl"
    "float32_abs_float_path"
    "#version 450\n\n\
     // Sarek-generated compute shader: float32_abs_float_path\n\
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
    \  b[idx] = abs(a[idx]);\n\
     }\n" ;

  register_golden
    "glsl"
    "float64_abs_float_path"
    "#version 450\n\
     #extension GL_ARB_gpu_shader_fp64 : require\n\n\
     // Sarek-generated compute shader: float64_abs_float_path\n\
     layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;\n\n\
     layout(std430, set=0, binding = 0) buffer Buffer_a {\n\
    \  double a[];\n\
     };\n\
     layout(std430, set=0, binding = 1) buffer Buffer_b {\n\
    \  double b[];\n\
     };\n\
     layout(push_constant) uniform PushConstants {\n\
    \  int a_len;\n\
    \  int b_len;\n\
     } pc;\n\n\
     #define a_len pc.a_len\n\
     #define b_len pc.b_len\n\n\
     void main() {\n\
    \  int idx = int(gl_GlobalInvocationID.x);\n\
    \  b[idx] = abs(a[idx]);\n\
     }\n" ;

  register_golden
    "glsl"
    "float64_copysign_path"
    "#version 450\n\
     #extension GL_ARB_gpu_shader_fp64 : require\n\n\
     // Sarek-generated compute shader: float64_copysign_path\n\
     layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;\n\n\
     layout(std430, set=0, binding = 0) buffer Buffer_a {\n\
    \  double a[];\n\
     };\n\
     layout(std430, set=0, binding = 1) buffer Buffer_b {\n\
    \  double b[];\n\
     };\n\
     layout(std430, set=0, binding = 2) buffer Buffer_c {\n\
    \  double c[];\n\
     };\n\
     layout(push_constant) uniform PushConstants {\n\
    \  int a_len;\n\
    \  int b_len;\n\
    \  int c_len;\n\
     } pc;\n\n\
     #define a_len pc.a_len\n\
     #define b_len pc.b_len\n\
     #define c_len pc.c_len\n\n\
     float sarek_copysign(float x, float y) { return \
     uintBitsToFloat((floatBitsToUint(x) & 0x7FFFFFFFu) | (floatBitsToUint(y) \
     & 0x80000000u)); }\n\n\
     double sarek_copysign(double x, double y) { uvec2 ux = \
     unpackDouble2x32(x); uvec2 uy = unpackDouble2x32(y); ux.y = (ux.y & \
     0x7FFFFFFFu) | (uy.y & 0x80000000u); return packDouble2x32(ux); }\n\n\
     void main() {\n\
    \  int idx = int(gl_GlobalInvocationID.x);\n\
    \  c[idx] = sarek_copysign(a[idx], b[idx]);\n\
     }\n" ;

  register_golden
    "glsl"
    "float32_copysign_path"
    "#version 450\n\n\
     // Sarek-generated compute shader: float32_copysign_path\n\
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
     float sarek_copysign(float x, float y) { return \
     uintBitsToFloat((floatBitsToUint(x) & 0x7FFFFFFFu) | (floatBitsToUint(y) \
     & 0x80000000u)); }\n\n\
     void main() {\n\
    \  int idx = int(gl_GlobalInvocationID.x);\n\
    \  c[idx] = sarek_copysign(a[idx], b[idx]);\n\
     }\n" ;

  register_golden
    "glsl"
    "float32_atan2_path"
    "#version 450\n\n\
     // Sarek-generated compute shader: float32_atan2_path\n\
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
    \  c[idx] = atan(a[idx], b[idx]);\n\
     }\n" ;

  register_golden
    "glsl"
    "float32_cbrt_path"
    "#version 450\n\n\
     // Sarek-generated compute shader: float32_cbrt_path\n\
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
    \  b[idx] = (sign(a[idx]) * pow(abs(a[idx]), 1.0 / 3.0));\n\
     }\n" ;

  register_golden
    "glsl"
    "float32_hypot_path"
    "#version 450\n\n\
     // Sarek-generated compute shader: float32_hypot_path\n\
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
    \  c[idx] = sqrt((a[idx]) * (a[idx]) + (b[idx]) * (b[idx]));\n\
     }\n" ;

  register_golden
    "glsl"
    "float32_expm1_path"
    "#version 450\n\n\
     // Sarek-generated compute shader: float32_expm1_path\n\
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
    \  b[idx] = (exp(a[idx]) - 1.0);\n\
     }\n" ;

  register_golden
    "glsl"
    "float32_log1p_path"
    "#version 450\n\n\
     // Sarek-generated compute shader: float32_log1p_path\n\
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
    \  b[idx] = log(1.0 + (a[idx]));\n\
     }\n"

let glsl_only_kernels () =
  [
    ("float32_rsqrt_path", float32_rsqrt_path_kernel ());
    ("float32_abs_float_path", float32_abs_float_path_kernel ());
    ("float64_abs_float_path", float64_abs_float_path_kernel ());
    ("float64_copysign_path", float64_copysign_path_kernel ());
    ("float32_copysign_path", float32_copysign_path_kernel ());
    ("float32_atan2_path", float32_atan2_path_kernel ());
    ("float32_cbrt_path", float32_cbrt_path_kernel ());
    ("float32_hypot_path", float32_hypot_path_kernel ());
    ("float32_expm1_path", float32_expm1_path_kernel ());
    ("float32_log1p_path", float32_log1p_path_kernel ());
  ]

let glsl_only_tests () =
  List.map
    (fun (kernel_name, k) ->
      Alcotest.test_case
        (Printf.sprintf "glsl/%s" kernel_name)
        `Quick
        (fun () ->
          Gen_glsl.reset_state () ;
          let actual = Gen_glsl.generate_with_types ~types:k.kern_types k in
          check_golden "glsl" kernel_name actual))
    (glsl_only_kernels ())

(** {1 Metal-only golden tests}

    Same rationale as the GLSL-only block above: cbrt/hypot/expm1/log1p have no
    MSL builtin under any name, so they resolve to a multi-token expression
    polyfill in [Sarek_ir_metal.gen_metal_polyfill] rather than a renamed
    function call. *)

let () =
  register_golden
    "metal"
    "float32_cbrt_path"
    "#include <metal_stdlib>\n\
     using namespace metal;\n\n\
     kernel void float32_cbrt_path(device float* a [[buffer(0)]], constant int \
     &sarek_a_length [[buffer(1)]], device float* b [[buffer(2)]], constant \
     int &sarek_b_length [[buffer(3)]],\n\
     uint3 __metal_gid [[thread_position_in_grid]],\n\
     uint3 __metal_tid [[thread_position_in_threadgroup]],\n\
     uint3 __metal_bid [[threadgroup_position_in_grid]],\n\
     uint3 __metal_tpg [[threads_per_threadgroup]],\n\
     uint3 __metal_num_groups [[threadgroups_per_grid]]) {\n\
    \  int idx = __metal_gid.x;\n\
    \  b[idx] = (sign(a[idx]) * pow(abs(a[idx]), 1.0 / 3.0));\n\
     }\n\n" ;

  register_golden
    "metal"
    "float32_hypot_path"
    "#include <metal_stdlib>\n\
     using namespace metal;\n\n\
     kernel void float32_hypot_path(device float* a [[buffer(0)]], constant \
     int &sarek_a_length [[buffer(1)]], device float* b [[buffer(2)]], \
     constant int &sarek_b_length [[buffer(3)]], device float* c \
     [[buffer(4)]], constant int &sarek_c_length [[buffer(5)]],\n\
     uint3 __metal_gid [[thread_position_in_grid]],\n\
     uint3 __metal_tid [[thread_position_in_threadgroup]],\n\
     uint3 __metal_bid [[threadgroup_position_in_grid]],\n\
     uint3 __metal_tpg [[threads_per_threadgroup]],\n\
     uint3 __metal_num_groups [[threadgroups_per_grid]]) {\n\
    \  int idx = __metal_gid.x;\n\
    \  c[idx] = sqrt((a[idx]) * (a[idx]) + (b[idx]) * (b[idx]));\n\
     }\n\n" ;

  register_golden
    "metal"
    "float32_expm1_path"
    "#include <metal_stdlib>\n\
     using namespace metal;\n\n\
     kernel void float32_expm1_path(device float* a [[buffer(0)]], constant \
     int &sarek_a_length [[buffer(1)]], device float* b [[buffer(2)]], \
     constant int &sarek_b_length [[buffer(3)]],\n\
     uint3 __metal_gid [[thread_position_in_grid]],\n\
     uint3 __metal_tid [[thread_position_in_threadgroup]],\n\
     uint3 __metal_bid [[threadgroup_position_in_grid]],\n\
     uint3 __metal_tpg [[threads_per_threadgroup]],\n\
     uint3 __metal_num_groups [[threadgroups_per_grid]]) {\n\
    \  int idx = __metal_gid.x;\n\
    \  b[idx] = (exp(a[idx]) - 1.0);\n\
     }\n\n" ;

  register_golden
    "metal"
    "float32_log1p_path"
    "#include <metal_stdlib>\n\
     using namespace metal;\n\n\
     kernel void float32_log1p_path(device float* a [[buffer(0)]], constant \
     int &sarek_a_length [[buffer(1)]], device float* b [[buffer(2)]], \
     constant int &sarek_b_length [[buffer(3)]],\n\
     uint3 __metal_gid [[thread_position_in_grid]],\n\
     uint3 __metal_tid [[thread_position_in_threadgroup]],\n\
     uint3 __metal_bid [[threadgroup_position_in_grid]],\n\
     uint3 __metal_tpg [[threads_per_threadgroup]],\n\
     uint3 __metal_num_groups [[threadgroups_per_grid]]) {\n\
    \  int idx = __metal_gid.x;\n\
    \  b[idx] = log(1.0 + (a[idx]));\n\
     }\n\n"

let metal_only_kernels () =
  [
    ("float32_cbrt_path", float32_cbrt_path_kernel ());
    ("float32_hypot_path", float32_hypot_path_kernel ());
    ("float32_expm1_path", float32_expm1_path_kernel ());
    ("float32_log1p_path", float32_log1p_path_kernel ());
  ]

let metal_only_tests () =
  List.map
    (fun (kernel_name, k) ->
      Alcotest.test_case
        (Printf.sprintf "metal/%s" kernel_name)
        `Quick
        (fun () ->
          Gen_metal.reset_state () ;
          let actual = Gen_metal.generate_with_types ~types:k.kern_types k in
          check_golden "metal" kernel_name actual))
    (metal_only_kernels ())

let () =
  Alcotest.run
    "codegen_golden"
    (List.map make_backend_tests all_backends
    @ [
        ("wgsl_only", wgsl_only_tests ());
        ("glsl_only", glsl_only_tests ());
        ("metal_only", metal_only_tests ());
      ])
