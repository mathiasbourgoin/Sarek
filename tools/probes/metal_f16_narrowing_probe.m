/* metal_f16_narrowing_probe.m — #63 slice 5
 *
 * The Metal row of docs/design/f16-relaxed-accuracy.md §2 does not exist:
 * Metal f16 has never been probed (docs/fp-contraction-policy.md §10.9). This
 * probe produces it.
 *
 * Method is the OpenCL probe's (tools/probes/opencl_f16_contraction_probe.c)
 * ported to Metal, and the Objective-C/Metal boilerplate is
 * tools/probes/metal_contraction_barrier_probe.m's. Two differences from the
 * OpenCL probe, both forced by the platform and both load-bearing:
 *
 *   1. MSL HAS NO `double`. The OpenCL positive control computes the fused
 *      answer as `(half)((double)x * (double)1.1f)` — an exactly-representable
 *      product narrowed in one rounding. That is not expressible here. The
 *      control below reconstructs the same value from a double-float pair
 *      (`hi = RN32(x*k)`, `lo = fma(x,k,-hi)` exact) with a round-to-odd step
 *      before the narrowing, which is the standard way to make a two-step
 *      narrowing agree with a single rounding. It is then CHECKED element-wise
 *      against the host `double` reference: the control is only usable if it
 *      agrees on all 63488, and the probe says so either way.
 *
 *   2. Metal's contraction defence is a SOURCE PRAGMA, not a compile option —
 *      no MTLCompileOptions setting stops contraction (§10.4, §10.5). The
 *      pragma is known to fix `a*b+c`. Whether it also governs a multiply
 *      feeding an f32->f16 narrowing is a DIFFERENT question, so every shape is
 *      swept both with and without it.
 *
 * The reference is not a tolerance. Per f16-relaxed-accuracy.md §1.2 the device
 * result must be BIT-IDENTICAL to a member of a named finite set of rounding
 * semantics, so every model below is compared element-wise on binary16 bit
 * patterns, and §1.3's 1-ulp ceiling is reported alongside as the necessary —
 * not sufficient — check.
 *
 * Shapes swept, matching the ones the other backends were swept on:
 *   S1  f16(x * 1.1)                one narrowing
 *   S2  f16(f16(x * 1.1) + 1000)    two narrowings
 *
 * Build (Command Line Tools are enough; no Xcode, no offline `metal` compiler —
 * newLibraryWithSource:options:error: compiles through the driver at runtime):
 *   clang -fobjc-arc -O2 -framework Foundation -framework Metal \
 *       metal_f16_narrowing_probe.m -o metal_f16_narrowing_probe
 *
 * Run: ./metal_f16_narrowing_probe        (no arguments; sweeps everything)
 *
 * ---------------------------------------------------------------------------
 * MEASURED 2026-07-27. Apple M4, macOS 15.6.1 (24G90), arm64, Apple clang
 * 17.0.0 (clang-1700.0.13.5), Metal.framework from the Command Line Tools SDK
 * (no Xcode). 63488 finite binary16 inputs. Evidence tier: EXECUTED.
 *
 * Host calibration: binary16 round-trip 0 failures; the two host models
 * separate on 2912 (S1) and 620 (S2) — 620 being the figure independently
 * reproduced on hiprtc/gfx1100, rusticl/radeonsi and `fusedctl` on Intel Arc.
 *
 *   S1  out = f16(x * 1.1f)                 deviation from S_strict
 *     plain                                        0 / 63488
 *     #pragma METAL fp contract(off)               0 / 63488
 *     #pragma clang fp contract(off)               0 / 63488
 *     volatile thread barrier                      0 / 63488
 *     as_type bitcast barrier                      0 / 63488
 *     FUSEDCTL (positive control)               2912 / 63488  <- control lives
 *     FUSEDCTL + contract(off)                  2912 / 63488
 *
 *   S2  out = f16(f16(x * 1.1f) + 1000.0f)   deviation from S_strict
 *     plain                                        0 / 63488
 *     #pragma METAL fp contract(off)               0 / 63488
 *     #pragma clang fp contract(off)               0 / 63488
 *     volatile thread barrier                      0 / 63488
 *     as_type bitcast barrier                      0 / 63488
 *     FUSEDCTL (positive control)                620 / 63488  <- control lives
 *     FUSEDCTL + contract(off)                   620 / 63488
 *
 * **Metal does not fuse the f16 narrowing.** It meets S_strict element-wise on
 * every finite binary16 input, on both shapes, with and without the pragma, and
 * with and without every barrier. The 0s are readable because the control on
 * the same source, same compile options and same dispatch reproduces
 * S_fuse_mul_into_narrowing on 63488 / 63488 elements and reports 2912 / 620.
 * Metal joins CUDA/nvrtc, HIP/AMDGPU, pocl, IGC and ANV in the strict class;
 * the relaxation of f16-relaxed-accuracy.md is not needed here.
 *
 * TWO SECONDARY FINDINGS, neither of which was the question:
 *
 * 1. **`#pragma METAL fp contract(off)` does not govern this hazard, and does
 *    not need to.** It changes nothing on either shape — not because it is
 *    inert (§10.5 measures it taking `a*b+c` from 8773/8773 to 0/8773) but
 *    because there is no fusion here to prevent. Contraction of `a*b+c` and
 *    absorption of a multiply into an f32->f16 narrowing are separate
 *    behaviours and the pragma is measured on only one of them. Symmetrically
 *    the pragma does not disturb the deliberate fusion in FUSEDCTL: 620 with it
 *    and 620 without.
 *
 * 2. **f16-relaxed-accuracy.md §1.3's 1-ulp ceiling is exceeded by the ADMITTED
 *    model on the two-narrowing shape**, so it cannot be applied to the final
 *    value of a multi-narrowing expression as written. At x = -907.5:
 *      exact x*1.1f = -998.25000216...; RN32 of it is exactly -998.25, which is
 *      a binary16 tie in the binade [512,1024) and rounds to even -> -998, so
 *      S_strict gives -998 + 1000 = 2. The single-rounding model narrows the
 *      exact product instead, is not at the tie, and gives -998.5 -> 1.5.
 *    2 and 1.5 are **512 ulp of binary16 apart at that magnitude**, because the
 *    +1000 cancels away the leading bits. The half-ulp-at-the-elided-step
 *    derivation in §1.3 is sound; what does not follow is the final-value
 *    restatement, since a later cancellation re-scales the ulp. The ceiling has
 *    to be measured at the narrowing where the rounding was elided, not on the
 *    result. This is not a Metal property — it is a property of the contract,
 *    surfaced here because the control computes the admitted model exactly.
 *
 * Determinism (§1.4 a and b): the whole sweep is bit-identical across two runs
 * in separate processes.
 * ---------------------------------------------------------------------------
 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#define NBITS 65536

/* ---- host binary16 helpers --------------------------------------------- */

static float f16_value_of_bits(int b, int *is_finite) {
  int sign = (b & 0x8000) ? -1 : 1;
  int exp = (b >> 10) & 0x1f, man = b & 0x3ff;
  if (exp == 31) { *is_finite = 0; return 0.0f; }
  *is_finite = 1;
  if (exp == 0) return sign * (float)man * ldexpf(1.0f, -24);
  return sign * (float)(1024 + man) * ldexpf(1.0f, exp - 25);
}

static uint16_t bits_of_f16(_Float16 h) { uint16_t u; memcpy(&u, &h, 2); return u; }
static _Float16 f16_of_bits(uint16_t u) { _Float16 h; memcpy(&h, &u, 2); return h; }

/* Signed ordering key on binary16 bit patterns. Adjacent representable values
   differ by exactly 1, including across zero, so |key(a)-key(b)| IS the ulp
   distance of §1.3's ceiling. Finite inputs only. */
static int f16_key(uint16_t u) {
  int mag = u & 0x7fff;
  return (u & 0x8000) ? -mag : mag;
}

/* ---- the named rounding models, computed exactly on the host ------------ */
/* `double` makes every one of these exact: an f16 value carries 11 significant
   bits and the f32 constant 1.1f carries 24, so their product needs 35 — well
   inside double's 53. Adding 1000 to an f16 value is likewise exact in double.
   So `(_Float16)<double expression>` is a SINGLE correctly-rounded narrowing,
   which is what the fused models mean. */

enum { M_STRICT = 0, M_FUSE_MUL, M_FUSE_BOTH, M_FUSE_ADD, M_DROP_MID, N_MODELS };

static const char *MODEL_NAME[N_MODELS] = {
  "S_strict            (every narrowing rounded, the interpreter)",
  "S_fuse_mul_into_narrowing (mul absorbed into the f32->f16 narrowing)",
  "S_fuse_both         (mul AND add each absorbed into their narrowing)",
  "S_fuse_add_only     (only the add absorbed)",
  "S_drop_mid          (intermediate f16 narrowing dropped — the IGC defect)",
};
/* Which models are meaningful for a one-narrowing shape. */
static const int MODEL_IN_S1[N_MODELS] = { 1, 1, 0, 0, 0 };

static void models_S1(float x, uint16_t out[N_MODELS]) {
  const double K = (double)1.1f;
  out[M_STRICT]    = bits_of_f16((_Float16)(x * 1.1f));
  out[M_FUSE_MUL]  = bits_of_f16((_Float16)((double)x * K));
  out[M_FUSE_BOTH] = out[M_FUSE_MUL];
  out[M_FUSE_ADD]  = out[M_STRICT];
  out[M_DROP_MID]  = out[M_STRICT];
}

static void models_S2(float x, uint16_t out[N_MODELS]) {
  const double K = (double)1.1f;
  _Float16 m_strict = (_Float16)(x * 1.1f);
  _Float16 m_fused  = (_Float16)((double)x * K);
  out[M_STRICT]    = bits_of_f16((_Float16)((float)m_strict + 1000.0f));
  out[M_FUSE_MUL]  = bits_of_f16((_Float16)((float)m_fused  + 1000.0f));
  out[M_FUSE_BOTH] = bits_of_f16((_Float16)((double)m_fused  + 1000.0));
  out[M_FUSE_ADD]  = bits_of_f16((_Float16)((double)m_strict + 1000.0));
  /* IGC's signature: the intermediate binary16 narrowing is not performed at
     all and the whole thing runs in binary32. */
  out[M_DROP_MID]  = bits_of_f16((_Float16)(x * 1.1f + 1000.0f));
}

/* ---- device source ------------------------------------------------------ */

static const char *PREAMBLE =
"#include <metal_stdlib>\n"
"using namespace metal;\n";

/* The positive control's single-rounding multiply-then-narrow.
 *
 * `hi` is forced through the integer domain before `lo` is formed, so that the
 * compiler cannot have already fused the multiply into a narrowing (§10.5
 * measures the as_type round-trip defeating contraction on this device); `hi`
 * is therefore RN32(x*k) whatever the compiler does. `lo = fma(x,k,-hi)` is
 * then the exact residual, so hi+lo is the exact product. Round-to-odd on `hi`
 * with respect to the sign of `lo` makes the subsequent narrowing to binary16
 * agree with a single correctly-rounded narrowing of the exact product.
 *
 * There is no `(half)(a*b)` anywhere in this helper, so the value it returns
 * cannot itself be the thing under test. It is validated element-wise against
 * the host double reference before any conclusion is drawn from it. */
static const char *FUSED_HELPER =
"static inline half fused_mul_narrow(float x, float k) {\n"
"  float hi = as_type<float>(as_type<uint>(x * k));\n"
"  float lo = fma(x, k, -hi);\n"
"  if (lo != 0.0f) {\n"
"    uint u = as_type<uint>(hi);\n"
"    if ((u & 1u) == 0u) {\n"
"      bool up = lo > 0.0f, pos = hi > 0.0f;\n"
"      u = (up == pos) ? (u + 1u) : (u - 1u);\n"
"      hi = as_type<float>(u);\n"
"    }\n"
"  }\n"
"  return (half)hi;\n"
"}\n";

typedef struct {
  const char *label;
  const char *pragma;   /* file-scope pragma, or "" */
  const char *body;     /* uses `float x`, writes `out[i]` */
  int is_control;       /* 1 = deliberately fused; must match M_FUSE_* */
} variant;

/* ---- S1: f16(x * 1.1) --------------------------------------------------- */
static variant V_S1[] = {
  { "plain",                      "",
    "  out[i] = (half)(x * 1.1f);\n", 0 },
  { "#pragma METAL fp contract(off)", "#pragma METAL fp contract(off)\n",
    "  out[i] = (half)(x * 1.1f);\n", 0 },
  { "#pragma clang fp contract(off)", "#pragma clang fp contract(off)\n",
    "  out[i] = (half)(x * 1.1f);\n", 0 },
  { "volatile thread barrier",     "",
    "  volatile thread float t = x * 1.1f;\n  out[i] = (half)t;\n", 0 },
  { "as_type bitcast barrier",     "",
    "  float t = as_type<float>(as_type<uint>(x * 1.1f));\n  out[i] = (half)t;\n", 0 },
  { "FUSEDCTL (positive control)", "",
    "  out[i] = fused_mul_narrow(x, 1.1f);\n", 1 },
  { "FUSEDCTL + contract(off)",    "#pragma METAL fp contract(off)\n",
    "  out[i] = fused_mul_narrow(x, 1.1f);\n", 1 },
};
static const int NV_S1 = sizeof(V_S1)/sizeof(V_S1[0]);

/* ---- S2: f16(f16(x * 1.1) + 1000) --------------------------------------- */
static variant V_S2[] = {
  { "plain",                      "",
    "  half m = (half)(x * 1.1f);\n  out[i] = (half)((float)m + 1000.0f);\n", 0 },
  { "#pragma METAL fp contract(off)", "#pragma METAL fp contract(off)\n",
    "  half m = (half)(x * 1.1f);\n  out[i] = (half)((float)m + 1000.0f);\n", 0 },
  { "#pragma clang fp contract(off)", "#pragma clang fp contract(off)\n",
    "  half m = (half)(x * 1.1f);\n  out[i] = (half)((float)m + 1000.0f);\n", 0 },
  { "volatile thread barrier",     "",
    "  volatile thread float t = x * 1.1f;\n  half m = (half)t;\n"
    "  volatile thread float u = (float)m + 1000.0f;\n  out[i] = (half)u;\n", 0 },
  { "as_type bitcast barrier",     "",
    "  half m = as_type<half>(as_type<ushort>((half)(x * 1.1f)));\n"
    "  out[i] = (half)((float)m + 1000.0f);\n", 0 },
  { "half arithmetic (no f32 mul)", "",
    "  half m = (half)x * (half)1.1f;\n  out[i] = m + (half)1000.0f;\n", 0 },
  { "FUSEDCTL (positive control)", "",
    "  half m = fused_mul_narrow(x, 1.1f);\n  out[i] = (half)((float)m + 1000.0f);\n", 1 },
  { "FUSEDCTL + contract(off)",    "#pragma METAL fp contract(off)\n",
    "  half m = fused_mul_narrow(x, 1.1f);\n  out[i] = (half)((float)m + 1000.0f);\n", 1 },
};
static const int NV_S2 = sizeof(V_S2)/sizeof(V_S2[0]);

/* ---- Metal plumbing ----------------------------------------------------- */

static id<MTLBuffer> mkbuf(id<MTLDevice> d, const void *s, size_t n) {
  return s ? [d newBufferWithBytes:s length:n options:MTLResourceStorageModeShared]
           : [d newBufferWithLength:n options:MTLResourceStorageModeShared];
}

int main(void) { @autoreleasepool {
  id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
  if (!dev) { fprintf(stderr, "no Metal device available\n"); return 1; }
  id<MTLCommandQueue> q = [dev newCommandQueue];
  printf("device  = %s\n", dev.name.UTF8String);
  printf("os      = %s\n",
         [NSProcessInfo processInfo].operatingSystemVersionString.UTF8String);

  /* ---- host side: enumerate, build the models, calibrate --------------- */
  int n = 0;
  static uint16_t inbits[NBITS];
  static float xval[NBITS];
  for (int b = 0; b < NBITS; b++) {
    int fin; float v = f16_value_of_bits(b, &fin);
    if (fin) { inbits[n] = (uint16_t)b; xval[n] = v; n++; }
  }
  printf("finite binary16 inputs: %d\n\n", n);

  static uint16_t mdl_s1[NBITS][N_MODELS], mdl_s2[NBITS][N_MODELS];
  for (int i = 0; i < n; i++) {
    models_S1(xval[i], mdl_s1[i]);
    models_S2(xval[i], mdl_s2[i]);
  }

  /* Calibration 1: the host binary16 rounding round-trips every input. If this
     fails, nothing below means anything. */
  int rt_bad = 0;
  for (int i = 0; i < n; i++)
    if (bits_of_f16((_Float16)xval[i]) != inbits[i]) rt_bad++;

  /* Calibration 2: the two host models must SEPARATE, and by the counts other
     stacks have already reproduced. A model set that does not separate is a
     gate that cannot fail. */
  int sep_s1 = 0, sep_s2 = 0;
  for (int i = 0; i < n; i++) {
    if (mdl_s1[i][M_STRICT] != mdl_s1[i][M_FUSE_MUL]) sep_s1++;
    if (mdl_s2[i][M_STRICT] != mdl_s2[i][M_FUSE_MUL]) sep_s2++;
  }
  printf("=== host calibration (runs with no GPU) ===\n");
  printf("  binary16 round-trip failures            : %d  %s\n", rt_bad,
         rt_bad == 0 ? "OK" : "<<< HOST REFERENCE BROKEN");
  printf("  S1 strict vs fuse_mul separation        : %d   (expected 2912)%s\n",
         sep_s1, sep_s1 == 2912 ? "" : "  <<< UNEXPECTED");
  printf("  S2 strict vs fuse_mul separation        : %d    (expected 620)%s\n",
         sep_s2, sep_s2 == 620 ? "" : "  <<< UNEXPECTED");
  if (rt_bad || sep_s1 == 0 || sep_s2 == 0) {
    fprintf(stderr, "\nhost calibration failed; refusing to report device numbers\n");
    return 4;
  }
  printf("\n");

  id<MTLBuffer> bIn = mkbuf(dev, inbits, (size_t)n * 2);

  /* ---- device sweep ---------------------------------------------------- */
  for (int shape = 0; shape < 2; shape++) {
    variant *V = shape == 0 ? V_S1 : V_S2;
    int NV      = shape == 0 ? NV_S1 : NV_S2;
    uint16_t (*mdl)[N_MODELS] = shape == 0 ? mdl_s1 : mdl_s2;
    const int *keep = shape == 0 ? MODEL_IN_S1 : NULL;

    printf("=== shape %s ===\n",
           shape == 0 ? "S1  out = f16(x * 1.1f)"
                      : "S2  out = f16(f16(x * 1.1f) + 1000.0f)");

    for (int v = 0; v < NV; v++) {
      char src[16384];
      snprintf(src, sizeof src,
        "%s%s%s"
        "kernel void midround(device half* out [[buffer(0)]],\n"
        "                     const device half* in [[buffer(1)]],\n"
        "                     uint i [[thread_position_in_grid]]) {\n"
        "  float x = (float)in[i];\n"
        "%s"
        "}\n",
        PREAMBLE, V[v].pragma, FUSED_HELPER, V[v].body);

      MTLCompileOptions *o = [MTLCompileOptions new];
      o.mathMode = MTLMathModeSafe;
      o.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
      NSError *err = nil;
      id<MTLLibrary> lib =
        [dev newLibraryWithSource:[NSString stringWithUTF8String:src]
                          options:o error:&err];
      if (!lib) {
        printf("  %-32s COMPILE FAILED: %s\n", V[v].label,
               err.localizedDescription.UTF8String);
        continue;
      }
      id<MTLFunction> f = [lib newFunctionWithName:@"midround"];
      id<MTLComputePipelineState> ps =
        [dev newComputePipelineStateWithFunction:f error:&err];
      if (!ps) { printf("  %-32s PIPELINE FAILED\n", V[v].label); continue; }

      id<MTLBuffer> bOut = mkbuf(dev, NULL, (size_t)n * 2);
      id<MTLCommandBuffer> cb = [q commandBuffer];
      id<MTLComputeCommandEncoder> en = [cb computeCommandEncoder];
      [en setComputePipelineState:ps];
      [en setBuffer:bOut offset:0 atIndex:0];
      [en setBuffer:bIn  offset:0 atIndex:1];
      [en dispatchThreads:MTLSizeMake(n,1,1)
            threadsPerThreadgroup:MTLSizeMake(64,1,1)];
      [en endEncoding]; [cb commit]; [cb waitUntilCompleted];
      if (cb.error) { printf("  %-32s EXEC FAILED\n", V[v].label); continue; }

      const uint16_t *got = (const uint16_t *)bOut.contents;

      /* element-wise agreement with every named model */
      int miss[N_MODELS]; int firsti[N_MODELS];
      for (int m = 0; m < N_MODELS; m++) { miss[m] = 0; firsti[m] = -1; }
      int none = 0, maxulp = 0, firstnone = -1, maxulpi = -1;
      for (int i = 0; i < n; i++) {
        int any = 0;
        for (int m = 0; m < N_MODELS; m++) {
          if (keep && !keep[m]) continue;
          if (got[i] != mdl[i][m]) {
            if (firsti[m] < 0) firsti[m] = i;
            miss[m]++;
          } else any = 1;
        }
        if (!any) { none++; if (firstnone < 0) firstnone = i; }
        int d = f16_key(got[i]) - f16_key(mdl[i][M_STRICT]);
        if (d < 0) d = -d;
        if (d > maxulp) { maxulp = d; maxulpi = i; }
      }

      printf("  %-32s", V[v].label);
      for (int m = 0; m < N_MODELS; m++) {
        if (keep && !keep[m]) continue;
        printf("  %s=%d", m == M_STRICT ? "strict"
                        : m == M_FUSE_MUL ? "fuse_mul"
                        : m == M_FUSE_BOTH ? "fuse_both"
                        : m == M_FUSE_ADD ? "fuse_add" : "drop_mid",
               miss[m]);
      }
      printf("  | no-model=%d | max-ulp16=%d\n", none, maxulp);

      /* first divergence from the strict discipline — the field that lets a
         later reader tell one rounding model from another */
      if (firsti[M_STRICT] >= 0) {
        int i = firsti[M_STRICT];
        printf("      first divergence from S_strict: x=%.9g  device=%.9g  S_strict=%.9g",
               xval[i], (double)(float)f16_of_bits(got[i]),
               (double)(float)f16_of_bits(mdl[i][M_STRICT]));
        for (int m = 1; m < N_MODELS; m++) {
          if (keep && !keep[m]) continue;
          if (got[i] == mdl[i][m]) printf("  == %s", MODEL_NAME[m]);
        }
        printf("\n");
      }
      /* §1.3's ceiling, and where it is worst. Reported for the controls too,
         because the ceiling is a claim about the ADMITTED class and a control
         that reproduces the admitted class exactly is the right thing to test
         it with. */
      if (maxulp > 1 && maxulpi >= 0) {
        int i = maxulpi;
        printf("      >>> exceeds the 1-ulp16 ceiling of §1.3: x=%.9g "
               "device=%.9g S_strict=%.9g (%d ulp16 apart)\n",
               xval[i], (double)(float)f16_of_bits(got[i]),
               (double)(float)f16_of_bits(mdl[i][M_STRICT]), maxulp);
      }
      if (firstnone >= 0) {
        int i = firstnone;
        printf("      >>> first input matching NO model: x=%.9g device=%.9g (bits %04x)\n",
               xval[i], (double)(float)f16_of_bits(got[i]), got[i]);
      }

      /* A control that does not reproduce the fused model is not a control. */
      if (V[v].is_control) {
        int ok = (miss[M_FUSE_MUL] == 0);
        printf("      CONTROL VALIDITY: %s (%d / %d elements differ from the "
               "host fused model)\n",
               ok ? "VALID — reproduces S_fuse_mul_into_narrowing exactly"
                  : "*** INVALID — do not read any 0 above as a result ***",
               miss[M_FUSE_MUL], n);
      }
    }
    printf("\n");
  }
  return 0;
} }

/* ===========================================================================
 * A note on the "half arithmetic (no f32 mul)" variant, which is not a
 * candidate and is not a Sarek-emittable shape: `(half)x * (half)1.1f` does the
 * multiply IN binary16, so it is a DIFFERENT expression and matches no model on
 * 6229 of 63488 inputs. It is kept because it is the only variant that makes
 * the harness print a nonzero `no-model` count, which is the control for the
 * no-model detector itself — without it, "no-model=0" everywhere else would be
 * indistinguishable from a detector that never fires.
 * ======================================================================== */
