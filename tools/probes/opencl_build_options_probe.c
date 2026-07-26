/* opencl_build_options_probe.c — #136
 *
 * What is an EMPTY clBuildProgram option string actually accepting, and does
 * setting the options explicitly change anything on the devices at hand?
 *
 * Until #136 the OpenCL backend called clBuildProgram with no options at all,
 * so it inherited each vendor's default — including a sqrt of up to 3 ulp,
 * which is what made test_real64's df64 fallback fail at 1.81e-14 on NVIDIA.
 * This probe is the instrument that decided the SHAPE of the fix.
 *
 * Standalone C rather than an OCaml test, for the same reason
 * opencl_f16_contraction_probe.c is: it must be able to say something about
 * the OpenCL stack with none of Sarek's codegen in the loop.
 *
 * Build:
 *   gcc -O2 -Wno-deprecated-declarations -I<repo>/dependencies \
 *       opencl_build_options_probe.c -o probe -lOpenCL -lm
 *
 * Run:
 *   ./probe accept    # which build options does each device accept?
 *   ./probe effect    # accuracy + cost of the FP options, with controls
 *   ./probe           # both
 *
 * ---------------------------------------------------------------------------
 * MEASURED 2026-07-26, rusticl/radeonsi driver 26.1.4-arch3.1, on
 * "AMD Radeon RX 7900 XTX (radeonsi, navi31, ACO, DRM 3.64, 7.1.2-3-cachyos)"
 * and "AMD Ryzen 9 7950X 16-Core Processor (radeonsi, raphael_mendocino, ...)".
 * There is NO NVIDIA and NO Apple device on this machine.
 *
 * accept mode:
 *   Both devices report CL_DEVICE_SINGLE_FP_CONFIG = 0x6, i.e.
 *   CL_FP_INF_NAN | CL_FP_ROUND_TO_NEAREST — NEITHER CL_FP_DENORM NOR
 *   CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT.
 *   Both nevertheless return CL_SUCCESS for
 *   -cl-fp32-correctly-rounded-divide-sqrt, which the OpenCL spec says must be
 *   CL_INVALID_BUILD_OPTIONS when the capability is absent. rusticl is
 *   permissive here, so THIS MACHINE CANNOT DETECT AN UNGATED FLAG. That is
 *   the argument for gating the option on CL_DEVICE_SINGLE_FP_CONFIG in
 *   Opencl_fp.conformance_options rather than passing it unconditionally: on a
 *   conformant implementation an ungated flag fails EVERY kernel build, and
 *   nothing on this box would have shown it.
 *   (Control that the accept/refuse distinction is real: the deliberately
 *   invalid -cl-this-option-does-not-exist is REFUSED on both devices, so
 *   "ACCEPTED" is not this probe's answer to everything.)
 *
 * effect mode, 2^20 inputs per device:
 *   baseline (empty)                        sqrt <=1 ulp, div <=2 ulp
 *   -cl-fp32-correctly-rounded-divide-sqrt  sqrt <=1 ulp, div <=2 ulp,
 *                                           BIT-IDENTICAL to baseline (0/1048576)
 *   PLUMBING control (-D<macro>)            differs on 1048576/1048576  <- passes
 *   FP LIVENESS control (-cl-fast-relaxed-math)
 *                                           BIT-IDENTICAL to baseline  <- FAILS
 *
 *   Read those two controls together before believing anything else here. The
 *   plumbing control proves the option string really does reach rusticl's
 *   compiler and that this comparison can go non-zero. The FP liveness control
 *   then FAILS: even -cl-fast-relaxed-math changes nothing. So rusticl accepts
 *   the FP-relaxing options and ignores them, and on this stack "the flag
 *   changed nothing" is INDISTINGUISHABLE from "the flag was discarded".
 *
 *   Consequently NO accuracy claim and NO cost claim for
 *   -cl-fp32-correctly-rounded-divide-sqrt may be founded on these devices.
 *   The timings this probe prints for it (-3.0% / +0.2% on the two devices)
 *   are run-to-run noise around an unchanged kernel, NOT a measured cost. The
 *   only real measurement of this option's effect is the sm_61 one quoted in
 *   Sarek_df64's PRECISION CONTRACT: 1.81e-14 FAIL -> 8.87e-15 PASS on a
 *   GTX 1070 Max-Q, and that device is not present here.
 * ---------------------------------------------------------------------------
 */
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* The vendored dependencies/CL/cl.h is OpenCL 1.0-era and predates both of
   these bits. Values from the OpenCL 1.2+ headers. Opencl_fp.ml spells them
   out for the same reason. */
#ifndef CL_FP_SOFT_FLOAT
#define CL_FP_SOFT_FLOAT (1 << 6)
#endif
#ifndef CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT
#define CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT (1 << 7)
#endif

#define N (1 << 20)
#define REPS 200

static const char *SRC =
"#ifndef SAREK_PROBE_SCALE\n#define SAREK_PROBE_SCALE 1.0f\n#endif\n"
"__kernel void k(__global float *os, __global float *od,\n"
"                __global const float *a, __global const float *b) {\n"
"  int i = get_global_id(0);\n"
"  os[i] = sqrt(a[i]) * SAREK_PROBE_SCALE;\n"
"  od[i] = a[i] / b[i];\n"
"}\n"
"__kernel void hot(__global float *o, __global const float *a,\n"
"                  __global const float *b, int reps) {\n"
"  int i = get_global_id(0);\n"
"  float x = a[i], y = b[i], acc = 0.0f;\n"
"  for (int r = 0; r < reps; r++) { acc += sqrt(x + (float)r) + x / (y + (float)r); }\n"
"  o[i] = acc;\n"
"}\n";

/* Every option Sarek could plausibly inherit or be handed, plus a deliberately
   invalid one as the control that "ACCEPTED" is a real verdict. */
static const char *ACCEPT_OPTS[] = {
  "",
  "-cl-fp32-correctly-rounded-divide-sqrt",
  "-cl-denorms-are-zero",
  "-cl-no-signed-zeros",
  "-cl-unsafe-math-optimizations",
  "-cl-finite-math-only",
  "-cl-fast-relaxed-math",
  "-cl-mad-enable",
  "-cl-opt-disable",
  "-cl-std=CL1.2",
  "-cl-single-precision-constant",
  "-cl-this-option-does-not-exist",   /* CONTROL: must be refused */
};
static const int N_ACCEPT = sizeof(ACCEPT_OPTS)/sizeof(ACCEPT_OPTS[0]);

static const char *EFFECT_OPTS[] = {
  "",
  "-cl-fp32-correctly-rounded-divide-sqrt",
  "-cl-fast-relaxed-math",
  "-DSAREK_PROBE_SCALE=1.0000001f",
};
static const char *EFFECT_LABELS[] = {
  "baseline (empty, what Sarek shipped before #136)",
  "-cl-fp32-correctly-rounded-divide-sqrt  (the fix)",
  "-cl-fast-relaxed-math                   (FP LIVENESS CONTROL)",
  "-D<macro>                               (PLUMBING CONTROL)",
};
static const int N_EFFECT = sizeof(EFFECT_OPTS)/sizeof(EFFECT_OPTS[0]);

static int ulp_diff(float a, float b) {
  int ia, ib; memcpy(&ia, &a, 4); memcpy(&ib, &b, 4);
  if (ia < 0) ia = 0x80000000 - ia;
  if (ib < 0) ib = 0x80000000 - ib;
  int d = ia - ib; return d < 0 ? -d : d;
}
static double now(void) {
  struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + t.tv_nsec * 1e-9;
}
static void print_fp_config(cl_device_fp_config fp) {
  printf("    CL_DEVICE_SINGLE_FP_CONFIG = 0x%llx  [%s%s%s%s%s%s]\n",
         (unsigned long long)fp,
         (fp & CL_FP_DENORM) ? "DENORM " : "",
         (fp & CL_FP_INF_NAN) ? "INF_NAN " : "",
         (fp & CL_FP_ROUND_TO_NEAREST) ? "RTN " : "",
         (fp & CL_FP_FMA) ? "FMA " : "",
         (fp & CL_FP_SOFT_FLOAT) ? "SOFT " : "",
         (fp & CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT) ? "CR_DIV_SQRT " : "");
}

struct res { float *s, *d; double ms; int max_ulp_s, max_ulp_d; int ok; };

static void run_effect(cl_context ctx, cl_device_id dev, cl_command_queue q,
                       const char *opt, float *A, float *B, struct res *out) {
  cl_int err;
  out->ok = 0;
  cl_program prog = clCreateProgramWithSource(ctx, 1, &SRC, NULL, &err);
  if (err != CL_SUCCESS) { printf("    (clCreateProgramWithSource failed: %d)\n", err); return; }
  cl_int b = clBuildProgram(prog, 1, &dev, opt, NULL, NULL);
  if (b != CL_SUCCESS) { printf("    BUILD FAILED (%d) for option '%s'\n", b, opt); return; }
  cl_kernel kk = clCreateKernel(prog, "k", &err);
  cl_kernel kh = clCreateKernel(prog, "hot", &err);
  size_t bytes = (size_t)N * 4;
  cl_mem ma = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, A, &err);
  cl_mem mb = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, B, &err);
  cl_mem ms = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, bytes, NULL, &err);
  cl_mem md = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, bytes, NULL, &err);
  cl_mem mo = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, bytes, NULL, &err);
  clSetKernelArg(kk,0,sizeof(cl_mem),&ms); clSetKernelArg(kk,1,sizeof(cl_mem),&md);
  clSetKernelArg(kk,2,sizeof(cl_mem),&ma); clSetKernelArg(kk,3,sizeof(cl_mem),&mb);
  size_t gs = N;
  clEnqueueNDRangeKernel(q, kk, 1, NULL, &gs, NULL, 0, NULL, NULL);
  clFinish(q);
  out->s = malloc(bytes); out->d = malloc(bytes);
  clEnqueueReadBuffer(q, ms, CL_TRUE, 0, bytes, out->s, 0, NULL, NULL);
  clEnqueueReadBuffer(q, md, CL_TRUE, 0, bytes, out->d, 0, NULL, NULL);

  out->max_ulp_s = out->max_ulp_d = 0;
  for (int i = 0; i < N; i++) {
    int u = ulp_diff(out->s[i], sqrtf(A[i]));
    if (u > out->max_ulp_s) out->max_ulp_s = u;
    u = ulp_diff(out->d[i], A[i] / B[i]);
    if (u > out->max_ulp_d) out->max_ulp_d = u;
  }

  int reps = REPS;
  clSetKernelArg(kh,0,sizeof(cl_mem),&mo); clSetKernelArg(kh,1,sizeof(cl_mem),&ma);
  clSetKernelArg(kh,2,sizeof(cl_mem),&mb); clSetKernelArg(kh,3,sizeof(int),&reps);
  for (int w = 0; w < 3; w++) clEnqueueNDRangeKernel(q,kh,1,NULL,&gs,NULL,0,NULL,NULL);
  clFinish(q);
  double best = 1e30;
  for (int t = 0; t < 7; t++) {
    double t0 = now();
    clEnqueueNDRangeKernel(q, kh, 1, NULL, &gs, NULL, 0, NULL, NULL);
    clFinish(q);
    double dt = (now() - t0) * 1000.0;
    if (dt < best) best = dt;
  }
  out->ms = best;
  out->ok = 1;

  clReleaseMemObject(ma); clReleaseMemObject(mb); clReleaseMemObject(ms);
  clReleaseMemObject(md); clReleaseMemObject(mo);
  clReleaseKernel(kk); clReleaseKernel(kh); clReleaseProgram(prog);
}

int main(int argc, char **argv) {
  int do_accept = 1, do_effect = 1;
  if (argc > 1) {
    do_accept = (strcmp(argv[1], "accept") == 0);
    do_effect = (strcmp(argv[1], "effect") == 0);
    if (!do_accept && !do_effect) {
      fprintf(stderr, "usage: %s [accept|effect]\n", argv[0]); return 2;
    }
  }

  float *A = malloc((size_t)N*4), *B = malloc((size_t)N*4);
  unsigned seed = 12345;
  for (int i = 0; i < N; i++) {
    seed = seed * 1103515245u + 12345u;
    A[i] = (float)((seed >> 8) & 0xFFFFFF) / 1024.0f + 1e-3f;
    seed = seed * 1103515245u + 12345u;
    B[i] = (float)((seed >> 8) & 0xFFFFFF) / 4096.0f + 1e-3f;
  }

  cl_platform_id plats[8]; cl_uint nplat = 0;
  if (clGetPlatformIDs(8, plats, &nplat) != CL_SUCCESS || nplat == 0) {
    printf("no OpenCL platform found\n"); return 1;
  }
  for (cl_uint p = 0; p < nplat; p++) {
    char pname[256] = {0};
    clGetPlatformInfo(plats[p], CL_PLATFORM_NAME, sizeof pname, pname, NULL);
    cl_device_id devs[8]; cl_uint ndev = 0;
    if (clGetDeviceIDs(plats[p], CL_DEVICE_TYPE_ALL, 8, devs, &ndev) != CL_SUCCESS) continue;
    for (cl_uint d = 0; d < ndev; d++) {
      char dname[256]={0}, dver[128]={0}, drv[128]={0};
      cl_device_fp_config fp = 0;
      clGetDeviceInfo(devs[d], CL_DEVICE_NAME, sizeof dname, dname, NULL);
      clGetDeviceInfo(devs[d], CL_DEVICE_VERSION, sizeof dver, dver, NULL);
      clGetDeviceInfo(devs[d], CL_DRIVER_VERSION, sizeof drv, drv, NULL);
      clGetDeviceInfo(devs[d], CL_DEVICE_SINGLE_FP_CONFIG, sizeof fp, &fp, NULL);
      printf("\n=== platform=%s\n    device=%s\n    version=%s driver=%s\n",
             pname, dname, dver, drv);
      print_fp_config(fp);

      cl_int err;
      cl_context ctx = clCreateContext(NULL, 1, &devs[d], NULL, NULL, &err);
      if (err != CL_SUCCESS) { printf("    (no context: %d)\n", err); continue; }

      if (do_accept) {
        printf("  -- accept mode --\n");
        for (int o = 0; o < N_ACCEPT; o++) {
          cl_program prog = clCreateProgramWithSource(ctx, 1, &SRC, NULL, &err);
          if (err != CL_SUCCESS) continue;
          cl_int b = clBuildProgram(prog, 1, &devs[d], ACCEPT_OPTS[o], NULL, NULL);
          const char *verdict =
            (b == CL_SUCCESS) ? "ACCEPTED"
            : (b == CL_INVALID_BUILD_OPTIONS) ? "REFUSED (CL_INVALID_BUILD_OPTIONS)"
            : "REFUSED (build failure)";
          printf("    %-42s -> %-36s (err %d)\n",
                 ACCEPT_OPTS[o][0] ? ACCEPT_OPTS[o] : "(empty option string)", verdict, b);
          clReleaseProgram(prog);
        }
      }

      if (do_effect) {
        printf("  -- effect mode --\n");
        cl_command_queue q = clCreateCommandQueue(ctx, devs[d], 0, &err);
        struct res r[4]; memset(r, 0, sizeof r);
        for (int v = 0; v < N_EFFECT; v++) {
          run_effect(ctx, devs[d], q, EFFECT_OPTS[v], A, B, &r[v]);
          if (!r[v].ok) continue;
          printf("    %-60s sqrt max %d ulp | div max %d ulp | hot %.3f ms\n",
                 EFFECT_LABELS[v], r[v].max_ulp_s, r[v].max_ulp_d, r[v].ms);
        }
        for (int v = 1; v < N_EFFECT; v++) {
          if (!r[v].ok || !r[0].ok) continue;
          int ds = 0, dd = 0;
          for (int i = 0; i < N; i++) {
            if (memcmp(&r[v].s[i], &r[0].s[i], 4)) ds++;
            if (memcmp(&r[v].d[i], &r[0].d[i], 4)) dd++;
          }
          printf("    vs baseline, %-38s : sqrt differs %d/%d, div differs %d/%d\n",
                 EFFECT_OPTS[v], ds, N, dd, N);
        }
        printf("    NOTE: read the two controls before the numbers above. If the\n"
               "          PLUMBING control differs and the FP LIVENESS control does\n"
               "          NOT, this stack ignores FP build options and no accuracy or\n"
               "          cost conclusion about them may be drawn from this device.\n");
        for (int v = 0; v < N_EFFECT; v++) { free(r[v].s); free(r[v].d); }
        clReleaseCommandQueue(q);
      }
      clReleaseContext(ctx);
    }
  }
  free(A); free(B);
  return 0;
}
