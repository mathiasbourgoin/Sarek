/* opencl_f16_contraction_probe.c — #57 slice 2a
 *
 * Standalone reproducer for the finding recorded in
 * docs/fp-contraction-policy.md, row "OpenCL / rusticl (f16 narrowing)":
 * the AMDGPU/ACO backend fuses the f32 multiply into the f32->f16 narrowing
 * that consumes it, rounding ONCE where Sarek's f16 discipline mandates twice.
 *
 * Deliberately standalone C rather than an OCaml test: it must be able to say
 * something about the OpenCL stack WITHOUT Sarek's codegen in the loop, since
 * the conclusion it supports is that Sarek's OpenCL backend should keep
 * refusing f16. It is therefore a documented reproducer, not a CI gate.
 *
 * Build:
 *   gcc -O2 -Wno-deprecated-declarations -I<repo>/dependencies \
 *       opencl_f16_contraction_probe.c -o probe -lOpenCL -lm
 *
 * Run (argv[1] = variant, argv[2] = device index):
 *   ./probe plain    0     # naive codegen        -> 620 / 63488
 *   ./probe vglobal  0     # volatile __global    ->   0 / 63488  (liveness control)
 *   ./probe vlocal   0     # volatile __local LDS ->   0 / 63488
 *   ./probe fpcontract|volatile|vpriv|bitcast|bitcast2|convert
 *                          # every affordable barrier -> still 620 / 63488
 *   ./probe barrier  0     # HIP's "+v" asm      -> does not compile here
 *   ./probe fusedctl 0     # POSITIVE CONTROL, deliberately fuses -> 620 / 63488
 *                          #   on any conforming device; run it whenever a
 *                          #   variant reports 0, or the 0 is not a result
 *
 * Device selection: the loop below scans the first platform that has devices,
 * so on a multi-platform host use the ICD loader's own filter to choose, e.g.
 *   OCL_ICD_FILENAMES=/usr/lib/intel-opencl/libigdrcl.so ./probe plain 0
 *
 * Measured 2026-07-26 on RX 7900 XTX (navi31) and the integrated Raphael iGPU
 * (gfx1036), rusticl/radeonsi, DRM 3.64, kernel 7.1.2-3-cachyos. Both devices
 * report the same 620/63488, first divergence at x=5.68359375.
 *
 * The reference is computed the way the Sarek interpreter computes it: round to
 * binary16 at EVERY narrowing. A "mismatch" is therefore a device that skipped
 * or fused a mandated rounding, not a device that is less accurate.
 */
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Variant A: plain cast narrowing (what a naive codegen emits). */
static const char *SRC_PLAIN =
"#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n"
"__kernel void midround(__global half *out, __global const half *in, int n) {\n"
"  int i = get_global_id(0);\n"
"  if (i < n) {\n"
"    float x = (float)in[i];\n"
"    half  m = (half)(x * 1.1f);\n"
"    out[i] = (half)((float)m + 1000.0f);\n"
"  }\n"
"}\n";

/* Variant B: same, with an opacity barrier on each narrowing's argument,
   mirroring Sarek_ir_cuda.sarek_f32_barrier_decl's HIP branch. */
static const char *SRC_BARRIER =
"#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n"
"static inline float sarek_f32_barrier(float x) {\n"
"  __asm__ volatile(\"\" : \"+v\"(x));\n"
"  return x;\n"
"}\n"
"__kernel void midround(__global half *out, __global const half *in, int n) {\n"
"  int i = get_global_id(0);\n"
"  if (i < n) {\n"
"    float x = (float)in[i];\n"
"    half  m = (half)sarek_f32_barrier(x * 1.1f);\n"
"    out[i] = (half)sarek_f32_barrier((float)m + 1000.0f);\n"
"  }\n"
"}\n";


/* Variant C: OpenCL's own standard contraction control. */
static const char *SRC_FPCONTRACT =
"#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n"
"#pragma OPENCL FP_CONTRACT OFF\n"
"__kernel void midround(__global half *out, __global const half *in, int n) {\n"
"  int i = get_global_id(0);\n"
"  if (i < n) {\n"
"    float x = (float)in[i];\n"
"    half  m = (half)(x * 1.1f);\n"
"    out[i] = (half)((float)m + 1000.0f);\n"
"  }\n"
"}\n";

/* Variant D: volatile local as the opacity barrier. */
static const char *SRC_VOLATILE =
"#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n"
"__kernel void midround(__global half *out, __global const half *in, int n) {\n"
"  int i = get_global_id(0);\n"
"  if (i < n) {\n"
"    float x = (float)in[i];\n"
"    volatile float t1 = x * 1.1f;\n"
"    half  m = (half)t1;\n"
"    volatile float t2 = (float)m + 1000.0f;\n"
"    out[i] = (half)t2;\n"
"  }\n"
"}\n";

/* Variant E: explicit convert_half_rte builtin instead of an implicit cast. */
static const char *SRC_CONVERT =
"#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n"
"__kernel void midround(__global half *out, __global const half *in, int n) {\n"
"  int i = get_global_id(0);\n"
"  if (i < n) {\n"
"    float x = (float)in[i];\n"
"    half  m = convert_half_rte(x * 1.1f);\n"
"    out[i] = convert_half_rte((float)m + 1000.0f);\n"
"  }\n"
"}\n";


/* Variant F: bitcast the narrowed value through ushort, so the f16 value
   passes through the integer domain and no float-level fold can see a
   f2f32(f2f16(x)) pair to collapse. */
static const char *SRC_BITCAST =
"#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n"
"__kernel void midround(__global half *out, __global const half *in, int n) {\n"
"  int i = get_global_id(0);\n"
"  if (i < n) {\n"
"    float x = (float)in[i];\n"
"    half  m = as_half(as_ushort((half)(x * 1.1f)));\n"
"    out[i] = as_half(as_ushort((half)((float)m + 1000.0f)));\n"
"  }\n"
"}\n";

/* Variant G: bitcast AND force the widening to read back through the integer,
   i.e. reconstruct the f32 from the f16 bits rather than casting the half. */
static const char *SRC_BITCAST2 =
"#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n"
"__kernel void midround(__global half *out, __global const half *in, int n) {\n"
"  int i = get_global_id(0);\n"
"  if (i < n) {\n"
"    float x = (float)in[i];\n"
"    ushort mb = as_ushort((half)(x * 1.1f));\n"
"    float m = (float)as_half(mb);\n"
"    out[i] = (half)(m + 1000.0f);\n"
"  }\n"
"}\n";


/* Variant H: force the intermediate through GLOBAL memory. Nothing can elide a
   round-trip through a __global buffer. If this still disagrees, the host
   reference is wrong, not the device. */
static const char *SRC_MEM =
"#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n"
"__kernel void midround(__global half *out, __global const half *in, int n) {\n"
"  int i = get_global_id(0);\n"
"  if (i < n) {\n"
"    float x = (float)in[i];\n"
"    out[i] = (half)(x * 1.1f);\n"
"  }\n"
"}\n";

/* Variant I: second pass of the memory version -- reads back what H stored. */
static const char *SRC_MEM2 =
"#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n"
"__kernel void midround(__global half *out, __global const half *in, int n) {\n"
"  int i = get_global_id(0);\n"
"  if (i < n) {\n"
"    out[i] = (half)((float)in[i] + 1000.0f);\n"
"  }\n"
"}\n";


/* Variant J: force the f32 product through GLOBAL memory before narrowing.
   Uses `out` as f32 scratch via a bitcast alias. If ACO still fuses across a
   global store/load, no source-level barrier exists on this path at all. */
static const char *SRC_SCRATCH =
"#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n"
"__kernel void midround(__global half *out, __global const half *in, int n,\n"
"                       __global float *scratch) {\n"
"  int i = get_global_id(0);\n"
"  if (i < n) {\n"
"    float x = (float)in[i];\n"
"    scratch[i] = x * 1.1f;\n"
"  }\n"
"}\n";
static const char *SRC_SCRATCH2 =
"#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n"
"__kernel void midround(__global half *out, __global const half *in, int n,\n"
"                       __global float *scratch) {\n"
"  int i = get_global_id(0);\n"
"  if (i < n) out[i] = (half)scratch[i];\n"
"}\n";


/* Variant K: volatile __global round-trip of the f32 product before narrowing.
   A volatile global store/load cannot be forwarded or fused by any legal
   optimiser. This is the decisive test for "is ANY barrier possible here". */
static const char *SRC_VGLOBAL =
"#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n"
"__kernel void midround(__global half *out, __global const half *in, int n,\n"
"                       __global volatile float *scratch) {\n"
"  int i = get_global_id(0);\n"
"  if (i < n) {\n"
"    float x = (float)in[i];\n"
"    scratch[i] = x * 1.1f;\n"
"    float p = scratch[i];\n"
"    half m = (half)p;\n"
"    scratch[i] = (float)m + 1000.0f;\n"
"    float q = scratch[i];\n"
"    out[i] = (half)q;\n"
"  }\n"
"}\n";


/* Variant L: volatile __local (LDS) round-trip -- same opacity, no global
   memory traffic. Cheaper than variant K if it works. */
static const char *SRC_VLOCAL =
"#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n"
"__kernel void midround(__global half *out, __global const half *in, int n) {\n"
"  __local volatile float s[256];\n"
"  int i = get_global_id(0); int l = get_local_id(0);\n"
"  if (i < n) {\n"
"    float x = (float)in[i];\n"
"    s[l] = x * 1.1f;\n"
"    half m = (half)s[l];\n"
"    s[l] = (float)m + 1000.0f;\n"
"    out[i] = (half)s[l];\n"
"  }\n"
"}\n";

/* Variant M: volatile __private pointer (register-level opacity, no memory). */
static const char *SRC_VPRIV =
"#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n"
"__kernel void midround(__global half *out, __global const half *in, int n) {\n"
"  int i = get_global_id(0);\n"
"  if (i < n) {\n"
"    float x = (float)in[i];\n"
"    float t = x * 1.1f;\n"
"    volatile float *pt = &t;\n"
"    half m = (half)(*pt);\n"
"    float u = (float)m + 1000.0f;\n"
"    volatile float *pu = &u;\n"
"    out[i] = (half)(*pu);\n"
"  }\n"
"}\n";

/* Variant N: POSITIVE CONTROL. Not a barrier and not a codegen candidate — a
   kernel that performs the fusion *deliberately*, so the harness can be shown
   able to report nonzero on a device that does not fuse on its own.

   This is required to read a 0 from any of the variants above as a result
   rather than as a broken harness. On rusticl/ACO the contrast is supplied for
   free (plain 620, vglobal 0), but on a non-fusing stack every variant returns
   0 and a silently-broken sweep is indistinguishable from a clean one.

   The semantics are chosen to match the ACO combine exactly, not merely to be
   wrong. `v_fma_mixlo_f16(x, 1.1f, 0)` rounds the EXACT f32 product straight to
   binary16 — one rounding where the DSL mandates two. Both `x` and the f32
   constant `1.1f` are exactly representable in binary64, so `(double)x *
   (double)1.1f` IS that exact product, and narrowing it in one step reproduces
   the fused value. The host reference is untouched. The expected count is
   therefore the SAME 620/63488 that ACO produces, on any conforming device —
   which makes this a calibration against a known figure, not just a liveness
   smoke test.

   Requires cl_khr_fp64. On a device without it the build fails loudly, which is
   the correct outcome: it means the calibration was not obtained. */
static const char *SRC_FUSED_CTL =
"#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n"
"#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
"__kernel void midround(__global half *out, __global const half *in, int n) {\n"
"  int i = get_global_id(0);\n"
"  if (i < n) {\n"
"    float x = (float)in[i];\n"
"    half  m = (half)((double)x * (double)1.1f);\n"
"    out[i] = (half)((float)m + 1000.0f);\n"
"  }\n"
"}\n";

/* ---- binary16 helpers (host reference) ---------------------------------- */
static float f16_value_of_bits(int b, int *is_finite) {
  int sign = (b & 0x8000) ? -1 : 1;
  int exp = (b >> 10) & 0x1f, man = b & 0x3ff;
  if (exp == 31) { *is_finite = 0; return 0.0f; }
  *is_finite = 1;
  if (exp == 0) return sign * (float)man * ldexpf(1.0f, -24);
  return sign * (float)(1024 + man) * ldexpf(1.0f, exp - 25);
}

/* Round an f32 to binary16 precision, returning the f32 value of the result.
   Uses the hardware _Float16 type so the rounding is IEEE RTE. */
static float round16(float x) { return (float)(_Float16)x; }

int main(int argc, char **argv) {
  const char *src = SRC_PLAIN; const char *label = "PLAIN (no barrier)";
  if (argc > 1 && strcmp(argv[1], "barrier") == 0) { src = SRC_BARRIER; label = "BARRIER"; }
  if (argc > 1 && strcmp(argv[1], "fpcontract") == 0) { src = SRC_FPCONTRACT; label = "FP_CONTRACT OFF"; }
  if (argc > 1 && strcmp(argv[1], "volatile") == 0) { src = SRC_VOLATILE; label = "VOLATILE LOCAL"; }
  if (argc > 1 && strcmp(argv[1], "bitcast") == 0) { src = SRC_BITCAST; label = "as_half(as_ushort(..))"; }
  if (argc > 1 && strcmp(argv[1], "bitcast2") == 0) { src = SRC_BITCAST2; label = "ushort intermediate"; }
  if (argc > 1 && strcmp(argv[1], "mem") == 0) { src = SRC_MEM; label = "MEM pass1 (mul only)"; }
  if (argc > 1 && strcmp(argv[1], "mem2") == 0) { src = SRC_MEM2; label = "MEM pass2 (add only)"; }
  if (argc > 1 && strcmp(argv[1], "scratch") == 0) { src = SRC_SCRATCH; label = "SCRATCH pass1"; }
  if (argc > 1 && strcmp(argv[1], "scratch2") == 0) { src = SRC_SCRATCH2; label = "SCRATCH pass2"; }
  if (argc > 1 && strcmp(argv[1], "vglobal") == 0) { src = SRC_VGLOBAL; label = "VOLATILE GLOBAL"; }
  if (argc > 1 && strcmp(argv[1], "vlocal") == 0) { src = SRC_VLOCAL; label = "VOLATILE LOCAL(LDS)"; }
  if (argc > 1 && strcmp(argv[1], "vpriv") == 0) { src = SRC_VPRIV; label = "VOLATILE PRIVATE PTR"; }
  if (argc > 1 && strcmp(argv[1], "convert") == 0) { src = SRC_CONVERT; label = "convert_half_rte"; }
  if (argc > 1 && strcmp(argv[1], "fusedctl") == 0) { src = SRC_FUSED_CTL; label = "FUSED (positive control)"; }

  cl_platform_id plats[8]; cl_uint nplat = 0;
  clGetPlatformIDs(8, plats, &nplat);
  cl_device_id devs[8]; cl_uint ndev = 0;
  for (cl_uint p = 0; p < nplat && ndev == 0; p++)
    clGetDeviceIDs(plats[p], CL_DEVICE_TYPE_ALL, 8, devs, &ndev);
  if (!ndev) { fprintf(stderr, "no OpenCL device\n"); return 2; }

  int which = 0;
  if (argc > 2) which = atoi(argv[2]);
  if (which >= (int)ndev) { fprintf(stderr, "no device %d\n", which); return 2; }
  cl_device_id dev = devs[which];
  char dname[256]; clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof dname, dname, NULL);
  printf("device: %s\nvariant: %s\n", dname, label);

  cl_int err;
  cl_context ctx = clCreateContext(NULL, 1, &dev, NULL, NULL, &err);
  cl_command_queue q = clCreateCommandQueue(ctx, dev, 0, &err);
  cl_program prog = clCreateProgramWithSource(ctx, 1, &src, NULL, &err);
  err = clBuildProgram(prog, 1, &dev, "", NULL, NULL);
  if (err != CL_SUCCESS) {
    size_t ls = 0; clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, NULL, &ls);
    char *log = malloc(ls + 1); clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, ls, log, NULL);
    log[ls] = 0; fprintf(stderr, "BUILD FAILED (%d):\n%s\n", err, log); return 3;
  }
  cl_kernel k = clCreateKernel(prog, "midround", &err);
  if (err != CL_SUCCESS) { fprintf(stderr, "clCreateKernel %d\n", err); return 3; }

  /* enumerate all finite binary16 values */
  int n = 0; static unsigned short bits[65536];
  for (int b = 0; b < 65536; b++) { int fin; f16_value_of_bits(b, &fin); if (fin) bits[n++] = (unsigned short)b; }
  printf("finite binary16 inputs: %d\n", n);

  _Float16 *hin = malloc(n * 2), *hout = malloc(n * 2);
  float *ref = malloc(n * sizeof(float));
  for (int i = 0; i < n; i++) {
    int fin; float x = f16_value_of_bits(bits[i], &fin);
    hin[i] = (_Float16)x;
    float mid = round16((float)(_Float16)x * 1.1f);
    ref[i] = round16(mid + 1000.0f);
    if (strcmp(label, "MEM pass1 (mul only)") == 0) ref[i] = mid;
    if (strncmp(label, "SCRATCH", 7) == 0) ref[i] = mid;
    if (strcmp(label, "MEM pass2 (add only)") == 0) ref[i] = round16((float)(_Float16)x + 1000.0f);
  }

  cl_mem din = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, n * 2, hin, &err);
  cl_mem dout = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, n * 2, NULL, &err);
  clSetKernelArg(k, 0, sizeof(cl_mem), &dout);
  clSetKernelArg(k, 1, sizeof(cl_mem), &din);
  clSetKernelArg(k, 2, sizeof(int), &n);
  cl_mem dscratch = clCreateBuffer(ctx, CL_MEM_READ_WRITE, n * sizeof(float), NULL, &err);
  if (strncmp(label, "SCRATCH", 7) == 0 || strcmp(label, "VOLATILE GLOBAL") == 0) clSetKernelArg(k, 3, sizeof(cl_mem), &dscratch);
  size_t local = 256, global = ((n + local - 1) / local) * local;
  err = clEnqueueNDRangeKernel(q, k, 1, NULL, &global, &local, 0, NULL, NULL);
  if (err != CL_SUCCESS) { fprintf(stderr, "enqueue %d\n", err); return 3; }
  clFinish(q);
  clEnqueueReadBuffer(q, dout, CL_TRUE, 0, n * 2, hout, 0, NULL, NULL);

  int bad = 0; int first = -1; int bad_sub = 0; int bad_norm = 0;
  const float MINNORM16 = 6.103515625e-05f;
  for (int i = 0; i < n; i++) {
    float got = (float)hout[i];
    /* bit-exact comparison at binary16 */
    if (memcmp(&hout[i], &(_Float16){(_Float16)ref[i]}, 2) != 0) {
      bad++;
      if (fabsf(ref[i]) < MINNORM16) bad_sub++; else bad_norm++;
      if (first < 0) {
        first = i;
        int fin; float x = f16_value_of_bits(bits[i], &fin);
        printf("first mismatch: x=%.9g got=%.9g expected=%.9g\n", x, got, ref[i]);
      }
    }
  }
  printf("RESULT: %d / %d mismatches  (expected-subnormal: %d, expected-normal: %d)\n", bad, n, bad_sub, bad_norm);
  return 0;
}
