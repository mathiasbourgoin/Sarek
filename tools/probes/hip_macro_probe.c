/* hip_macro_probe.c — backlog #146
 *
 * Which macros does hiprtc actually predefine, and does the AMDGPU opacity
 * barrier of Sarek_ir_cuda.sarek_f32_barrier_decl still land on the AMD arm
 * under each candidate #if guard?
 *
 * The question it answers, from docs/fp-contraction-policy.md §11.4a: the
 * barrier guard used to read
 *
 *   #if defined(__HIP__) || defined(__HIP_PLATFORM_AMD__)
 *
 * and the first disjunct reads as though it also admits HIP compiled for an
 * NVIDIA target, where "+v" is not a valid constraint. Narrowing a guard is
 * only safe if the narrowed form still fires on the path that needs it, and
 * that is a measurement, not a review opinion.
 *
 * Deliberately standalone C rather than an OCaml test, for the same reason as
 * the OpenCL probes: it must be able to say something about the hiprtc
 * PREPROCESSOR without Sarek's codegen in the loop. It is a documented
 * reproducer, not a CI gate — sarek/tests/codegen_golden/test_cuda_f16_golden.ml
 * (test_f16_barrier_is_amd_scoped) is the gate.
 *
 * Non-vacuity: every guard case puts an #error in the NON-AMD arm, so
 * "COMPILES" proves the AMD arm was taken and that its "+v" asm is legal on
 * this target. A guard that matched nothing would report FAILS with
 * NON_AMD_ARM_TAKEN, and the last case exercises exactly that so the harness is
 * shown able to go both ways on the same run.
 *
 * Build:
 *   gcc -D__HIP_PLATFORM_AMD__ hip_macro_probe.c -I/opt/rocm/include \
 *       -L/opt/rocm/lib -lhiprtc -Wl,-rpath,/opt/rocm/lib -o hip_macro_probe
 *   (-D__HIP_PLATFORM_AMD__ is for the HOST compile of hiprtc.h, which errors
 *    without a platform macro; it does not reach the device source below.)
 *
 * Measured 2026-07-27, ROCm 7.2.53211 / libhiprtc.so.7, RX 7900 XTX host:
 *
 *   __HIP__                     FAILS  -> defined
 *   __HIP_PLATFORM_AMD__        FAILS  -> defined
 *   __HIP_PLATFORM_NVIDIA__     COMPILES -> NOT defined
 *   __HIP_PLATFORM_HCC__        COMPILES -> NOT defined (legacy, pre-4.x)
 *
 *   current: __HIP__ || __HIP_PLATFORM_AMD__   COMPILES  (AMD arm taken)
 *   narrow:  __HIP_PLATFORM_AMD__ only         COMPILES  (AMD arm taken)
 *   control: a macro nothing defines           FAILS     (non-AMD arm taken)
 */
#include <hip/hiprtc.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int try_compile(const char *name, const char *src) {
  hiprtcProgram p;
  if (hiprtcCreateProgram(&p, src, "probe.hip", 0, NULL, NULL) !=
      HIPRTC_SUCCESS) {
    printf("%-42s CREATE-FAILED\n", name);
    return -1;
  }
  hiprtcResult r = hiprtcCompileProgram(p, 0, NULL);
  size_t n = 0;
  hiprtcGetProgramLogSize(p, &n);
  char *log = calloc(n + 1, 1);
  if (n > 1) hiprtcGetProgramLog(p, log);
  printf("%-42s %s", name, r == HIPRTC_SUCCESS ? "COMPILES" : "FAILS");
  if (r != HIPRTC_SUCCESS) {
    char *m = strstr(log, "error:");
    if (m) {
      char *e = strchr(m, '\n');
      if (e) *e = 0;
      printf("  | %s", m);
    }
  }
  printf("\n");
  free(log);
  hiprtcDestroyProgram(&p);
  return r == HIPRTC_SUCCESS;
}

/* FAILS => the macro IS defined. Stated this way round because a preprocessor
   can only report a macro's presence by refusing to compile. */
#define MACRO_PROBE(m)                                                         \
  "#if defined(" m ")\n#error MACRO_" m "_IS_DEFINED\n#endif\n"                \
  "__global__ void k(float* o) { o[0] = 1.0f; }\n"

int main(void) {
  puts("-- macro presence: FAILS => the macro IS defined --");
  try_compile("__HIP__", MACRO_PROBE("__HIP__"));
  try_compile("__HIP_PLATFORM_AMD__", MACRO_PROBE("__HIP_PLATFORM_AMD__"));
  try_compile("__HIP_PLATFORM_NVIDIA__", MACRO_PROBE("__HIP_PLATFORM_NVIDIA__"));
  try_compile("__HIP_PLATFORM_HCC__ (legacy)",
              MACRO_PROBE("__HIP_PLATFORM_HCC__"));

  puts("\n-- does each guard select the AMD asm arm, and does it build? --");
  /* Body verbatim from Sarek_ir_cuda.sarek_f32_barrier_decl, except that the
     non-AMD arm carries an #error so the taken arm is observable. */
  const char *body =
      "__device__ __forceinline__ float sarek_f32_barrier(float x) {\n"
      "  asm volatile(\"\" : \"+v\"(x));\n"
      "  return x;\n"
      "}\n"
      "#else\n"
      "__device__ __forceinline__ float sarek_f32_barrier(float x) {\n"
      "#error NON_AMD_ARM_TAKEN\n"
      "  return x;\n"
      "}\n"
      "#endif\n"
      "__global__ void k(float* o, const float* i) {\n"
      "  o[0] = sarek_f32_barrier(i[0] * 1.1f);\n"
      "}\n";
  char buf[4096];

  snprintf(buf, sizeof buf,
           "#if defined(__HIP__) || defined(__HIP_PLATFORM_AMD__)\n%s", body);
  try_compile("current: __HIP__ || __HIP_PLATFORM_AMD__", buf);

  snprintf(buf, sizeof buf, "#if defined(__HIP_PLATFORM_AMD__)\n%s", body);
  try_compile("narrow: __HIP_PLATFORM_AMD__ only", buf);

  /* Liveness control: a guard nothing satisfies must take the other arm and
     report NON_AMD_ARM_TAKEN. Without this, every COMPILES above could mean
     "the #error is unreachable for some unrelated reason". */
  snprintf(buf, sizeof buf, "#if defined(__SAREK_NO_SUCH_MACRO__)\n%s", body);
  try_compile("control: guard nothing defines", buf);
  return 0;
}
