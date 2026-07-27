/* metal_simdgroup_matrix_probe.m — #63 slice 6 prerequisite
 *
 * docs/design/f16-relaxed-accuracy.md §7 slice 6 asks what MSL actually offers
 * on the M4. Apple documents `simdgroup_matrix` for Metal 3, but which
 * (component type, rows, cols) instantiations exist, and whether the device has
 * matrix hardware behind them, is not something the documentation settles. A
 * probe is the only way to know.
 *
 * Two questions, kept apart on purpose:
 *
 *   AVAILABILITY — which `simdgroup_matrix<T,C,R>` instantiations COMPILE, and
 *   which then load / multiply-accumulate / store without failing. Answered by
 *   compiling each candidate in isolation through the runtime compiler and
 *   dispatching it. Evidence tier: executed (or compiler-output for the ones
 *   that only fail to compile).
 *
 *   NUMERICS — what the MulAdd computes. §5's derived bound applies with K = 8
 *   rather than 16. This probe runs the f16xf16->f32 and f16xf16->f16 8x8x8
 *   products against an exact host reference (an f16 x f16 product is exact in
 *   binary32 — 11 + 11 = 22 bits inside binary32's 24 — so a binary64 host
 *   reference is exact for K = 8), with the REQUIRED negative control of §5: a
 *   host reference that accumulates in binary16 must be REJECTED by the
 *   f32-accumulate comparison, or the comparison is a gate that cannot fail.
 *
 * This probe does NOT claim anything about whether dedicated matrix hardware is
 * used. It reports what the API offers and what it computes; a throughput claim
 * would need a benchmark against a scalar kernel and is out of scope here.
 *
 * Build (Command Line Tools; no Xcode, no offline `metal` compiler):
 *   clang -fobjc-arc -O2 -framework Foundation -framework Metal \
 *       metal_simdgroup_matrix_probe.m -o metal_simdgroup_matrix_probe
 *
 * Run: ./metal_simdgroup_matrix_probe
 *
 * ---------------------------------------------------------------------------
 * MEASURED 2026-07-27. Apple M4, macOS 15.6.1 (24G90), arm64, Apple clang
 * 17.0.0 (clang-1700.0.13.5), Metal.framework from the Command Line Tools SDK.
 * GPU families Apple1-9 + Metal3 all supported; MSL 2.4 / 3.0 / 3.1 / 3.2 all
 * accepted by the runtime compiler. Evidence tier: EXECUTED.
 *
 * AVAILABILITY — `simdgroup_matrix<T, Cols, Rows>` exists for exactly THREE
 * instantiations, all 8x8:
 *
 *     half   8x8   YES        float  8x8   YES        bfloat 8x8   YES
 *
 * Everything else is a compile-time refusal with a named static_assert, so the
 * answer is a closed one and not a sampling:
 *   - any size other than 8x8 (16x16, 8x16, 16x8, 4x4, 32x8, 8x32 all tried)
 *     fails `_valid_simdgroup_matrix_size(Cols, Rows)`;
 *   - char / uchar / short / int fail `is_simdgroup_matrix_element<T>::value`.
 * threadExecutionWidth is 32, so one 8x8 fragment is one 32-wide simdgroup.
 *
 * **There is no integer simdgroup_matrix on Metal.** That matters beyond this
 * probe: f16-relaxed-accuracy.md §8 and §7 slice 4b keep an integer coopmat
 * path as the fallback if the ACO scalar shapes match no closed-form model, and
 * 12 of the 14 configurations advertised by RADV are integer. That fallback
 * exists on Vulkan and does NOT exist on Metal. #63 has no strict-contract
 * route; it is f16/bf16 or nothing.
 *
 * **bfloat 8x8 is available and is not in the plan.** Nothing here says what it
 * computes — it was compiled and dispatched, not swept.
 *
 * NUMERICS — 1024 independent 8x8x8 problems, 65536 output elements. Anti-
 * vacuity first: on this input set a host binary16 accumulator is rejected by
 * the f32 bound on 65452/65536, the exact reference is rejected by its own
 * bound on 0, and sequential vs pairwise binary32 accumulation differ on
 * 27597/65536 — so both the bound and the element-wise model test can fail.
 *
 *   simdgroup_half8x8 x half8x8 -> float8x8
 *     bit-equal to SEQUENTIAL binary32 accumulate : 65536 / 65536   <-- EXACT
 *     bit-equal to PAIRWISE   binary32 accumulate : 37939 / 65536
 *     bit-equal to the exact dot product          :   217 / 65536
 *     worst error / sum|p_i| 2.79e-07 against the derived bound 4.17e-07
 *
 *   simdgroup_half8x8 x half8x8 -> half8x8
 *     bit-equal to SEQUENTIAL binary16 accumulate : 65528 / 65536   <-- NOT exact
 *     bit-equal to PAIRWISE   binary16 accumulate : 41109 / 65536
 *     worst error / sum|p_i| 1.75e-03 against the derived bound 3.43e-03
 *
 * **The f32-accumulate configuration matches a NAMED CLOSED-FORM MODEL
 * element-wise on every element** — sequential accumulation k = 0..7 in
 * binary32. That is a materially stronger statement than sitting inside §5's
 * bound, and it is the outcome §6.1's corollary was written for: friction falls
 * as evidence improves, so this configuration belongs in the loud-diagnostic
 * row and not the mandatory-opt-in row. Note the model cannot distinguish
 * "exact products then sequential adds" from an fma chain, and does not need
 * to: an f16 x f16 product is exact in binary32 (11 + 11 = 22 bits inside 24),
 * so the two are the same function. Evidence tier for THAT step:
 * by-construction.
 *
 * **The f16-accumulate configuration matches no closed-form model** — 8
 * elements in 65536 differ from sequential binary16, by 1-2 ulp16, and match
 * neither pairwise binary16 nor a binary32 chain narrowed at the end. It stays
 * bounded-only, which is exactly the case §5 recommends NOT admitting; the
 * recommendation now rests on a measurement rather than on the width of the
 * bound alone.
 *
 * DETERMINISM (§1.4) — bit-identical across two processes, and bit-identical
 * across threadgroup sizes 32 / 64 / 128 / 256 (0 / 65536 differ in each case).
 * §1.4 says a coopmat result may legitimately move with the dispatch shape and
 * that nobody had measured it. On this device, at this tile size, it does not.
 *
 * NOT MEASURED: throughput. Whether the M4 has dedicated matrix hardware behind
 * these three instantiations is not answered by this probe and cannot be read
 * off it — that needs a benchmark against a scalar kernel.
 * ---------------------------------------------------------------------------
 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

static id<MTLBuffer> mkbuf(id<MTLDevice> d, const void *s, size_t n) {
  return s ? [d newBufferWithBytes:s length:n options:MTLResourceStorageModeShared]
           : [d newBufferWithLength:n options:MTLResourceStorageModeShared];
}

/* ---- part 1: which instantiations exist -------------------------------- */

typedef struct { const char *t; int rows, cols; } cfg;

static cfg CFGS[] = {
  { "half",   8,  8 }, { "float",  8,  8 }, { "bfloat", 8,  8 },
  { "half",  16, 16 }, { "float", 16, 16 }, { "bfloat",16, 16 },
  { "half",   8, 16 }, { "half",  16,  8 },
  { "half",   4,  4 }, { "half",  32,  8 }, { "half",   8, 32 },
  { "char",   8,  8 }, { "uchar",  8,  8 },
  { "short",  8,  8 }, { "int",    8,  8 },
};
static const int NCFG = sizeof(CFGS)/sizeof(CFGS[0]);

/* Declared, loaded, multiply-accumulated and stored — a declaration alone can
   compile while the operations do not exist for that instantiation. */
static void avail_source(char *buf, size_t sz, cfg c) {
  snprintf(buf, sz,
    "#include <metal_stdlib>\n"
    "#include <metal_simdgroup_matrix>\n"
    "using namespace metal;\n"
    "kernel void k(device %s* out [[buffer(0)]],\n"
    "              const device %s* a [[buffer(1)]],\n"
    "              const device %s* b [[buffer(2)]],\n"
    "              uint tid [[thread_position_in_grid]]) {\n"
    "  simdgroup_matrix<%s, %d, %d> A, B, C;\n"
    "  C = simdgroup_matrix<%s, %d, %d>(0);\n"
    "  simdgroup_load(A, a, %d);\n"
    "  simdgroup_load(B, b, %d);\n"
    "  simdgroup_multiply_accumulate(C, A, B, C);\n"
    "  simdgroup_store(C, out, %d);\n"
    "}\n",
    c.t, c.t, c.t, c.t, c.cols, c.rows, c.t, c.cols, c.rows,
    c.cols, c.cols, c.cols);
}

/* ---- part 2: what the 8x8x8 f16 MulAdd computes ------------------------ */

/* NT independent 8x8x8 problems, one per simdgroup. */
#define NT 1024

/* f16 operands, f32 accumulate. */
static const char *SRC_MMA_F32 =
"#include <metal_stdlib>\n"
"#include <metal_simdgroup_matrix>\n"
"using namespace metal;\n"
"kernel void k(device float* out [[buffer(0)]],\n"
"              const device half* a [[buffer(1)]],\n"
"              const device half* b [[buffer(2)]],\n"
"              uint tid [[thread_position_in_grid]]) {\n"
"  uint sg = tid / 32;\n"
"  simdgroup_half8x8  A, B;\n"
"  simdgroup_float8x8 C = simdgroup_float8x8(0);\n"
"  simdgroup_load(A, a + sg * 64, 8);\n"
"  simdgroup_load(B, b + sg * 64, 8);\n"
"  simdgroup_multiply_accumulate(C, A, B, C);\n"
"  simdgroup_store(C, out + sg * 64, 8);\n"
"}\n";

/* f16 operands, f16 accumulate — §5 recommends NOT admitting this one; the
   probe measures it so the recommendation rests on a number. */
static const char *SRC_MMA_F16 =
"#include <metal_stdlib>\n"
"#include <metal_simdgroup_matrix>\n"
"using namespace metal;\n"
"kernel void k(device half* out [[buffer(0)]],\n"
"              const device half* a [[buffer(1)]],\n"
"              const device half* b [[buffer(2)]],\n"
"              uint tid [[thread_position_in_grid]]) {\n"
"  uint sg = tid / 32;\n"
"  simdgroup_half8x8 A, B, C = simdgroup_half8x8(0);\n"
"  simdgroup_load(A, a + sg * 64, 8);\n"
"  simdgroup_load(B, b + sg * 64, 8);\n"
"  simdgroup_multiply_accumulate(C, A, B, C);\n"
"  simdgroup_store(C, out + sg * 64, 8);\n"
"}\n";

static id<MTLLibrary> build(id<MTLDevice> dev, const char *src, NSError **err) {
  MTLCompileOptions *o = [MTLCompileOptions new];
  o.mathMode = MTLMathModeSafe;
  o.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
  return [dev newLibraryWithSource:[NSString stringWithUTF8String:src]
                           options:o error:err];
}

int main(void) { @autoreleasepool {
  id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
  if (!dev) { fprintf(stderr, "no Metal device available\n"); return 1; }
  id<MTLCommandQueue> q = [dev newCommandQueue];

  printf("device  = %s\n", dev.name.UTF8String);
  printf("os      = %s\n",
         [NSProcessInfo processInfo].operatingSystemVersionString.UTF8String);
  printf("unified memory = %s\n", dev.hasUnifiedMemory ? "yes" : "no");

  printf("\n=== GPU family support ===\n");
  struct { const char *n; MTLGPUFamily f; } fams[] = {
    { "Apple1", MTLGPUFamilyApple1 }, { "Apple2", MTLGPUFamilyApple2 },
    { "Apple3", MTLGPUFamilyApple3 }, { "Apple4", MTLGPUFamilyApple4 },
    { "Apple5", MTLGPUFamilyApple5 }, { "Apple6", MTLGPUFamilyApple6 },
    { "Apple7", MTLGPUFamilyApple7 }, { "Apple8", MTLGPUFamilyApple8 },
    { "Apple9", MTLGPUFamilyApple9 }, { "Metal3", MTLGPUFamilyMetal3 },
    { "Common1", MTLGPUFamilyCommon1 }, { "Common2", MTLGPUFamilyCommon2 },
    { "Common3", MTLGPUFamilyCommon3 },
  };
  for (unsigned i = 0; i < sizeof(fams)/sizeof(fams[0]); i++)
    printf("  %-8s %s\n", fams[i].n,
           [dev supportsFamily:fams[i].f] ? "yes" : "no");

  printf("\n=== MSL language version accepted by the runtime compiler ===\n");
  struct { const char *n; MTLLanguageVersion v; } vers[] = {
    { "2.4", MTLLanguageVersion2_4 }, { "3.0", MTLLanguageVersion3_0 },
    { "3.1", MTLLanguageVersion3_1 }, { "3.2", MTLLanguageVersion3_2 },
  };
  for (unsigned i = 0; i < sizeof(vers)/sizeof(vers[0]); i++) {
    MTLCompileOptions *o = [MTLCompileOptions new];
    o.languageVersion = vers[i].v;
    NSError *e = nil;
    id<MTLLibrary> l = [dev newLibraryWithSource:
        @"#include <metal_stdlib>\nkernel void k(){}\n" options:o error:&e];
    printf("  metal %-4s %s\n", vers[i].n, l ? "accepted" : "rejected");
  }

  printf("\n=== simdgroup_matrix<T, Cols, Rows> availability ===\n");
  printf("  (declared + simdgroup_load + multiply_accumulate + store)\n");
  for (int i = 0; i < NCFG; i++) {
    char src[4096]; avail_source(src, sizeof src, CFGS[i]);
    NSError *e = nil;
    id<MTLLibrary> lib = build(dev, src, &e);
    if (!lib) {
      /* keep the first diagnostic line that names the failure, not the module
         preamble the compiler prints before it */
      const char *m = e.localizedDescription.UTF8String;
      const char *w = strstr(m, "error:");
      if (!w) w = m;
      char one[240]; size_t k = 0;
      for (const char *p = w; *p && k < sizeof(one)-1; p++)
        one[k++] = (*p == '\n') ? ' ' : *p;
      one[k] = 0;
      printf("  %-7s %2dx%-2d  NO   %.200s\n", CFGS[i].t, CFGS[i].rows, CFGS[i].cols, one);
      continue;
    }
    id<MTLFunction> f = [lib newFunctionWithName:@"k"];
    id<MTLComputePipelineState> ps =
      [dev newComputePipelineStateWithFunction:f error:&e];
    if (!ps) {
      printf("  %-7s %2dx%-2d  compiles, PIPELINE FAILED\n",
             CFGS[i].t, CFGS[i].rows, CFGS[i].cols);
      continue;
    }
    printf("  %-7s %2dx%-2d  YES  (threadExecutionWidth=%lu, "
           "maxTotalThreadsPerThreadgroup=%lu)\n",
           CFGS[i].t, CFGS[i].rows, CFGS[i].cols,
           (unsigned long)ps.threadExecutionWidth,
           (unsigned long)ps.maxTotalThreadsPerThreadgroup);
  }

  /* ---- numerics: 8x8x8, f16 operands ---------------------------------- */
  printf("\n=== 8x8x8 MulAdd numerics (K = 8, %d independent problems) ===\n", NT);

  /* INPUT DESIGN, and it is the part that makes this a check rather than a
     formality. A first attempt drew operands as multiples of 2^-9 in [-1,1].
     Every product was then a multiple of 2^-18 below 8, so the sum of 8 of them
     was EXACTLY representable in binary32 — every accumulation order gives the
     same answer and the f32 comparison could not fail. It reported 64/64 exact
     and measured nothing.
     The operands below carry a full 11-bit significand and an exponent drawn
     from a 13-wide range, so a product needs 22 bits and the 8 products span 24
     binades: the sum needs about 46 significant bits. That is far outside
     binary32, so accumulation order IS observable, and still inside binary64's
     53, so the host reference remains EXACT. Both halves matter. */
  static _Float16 A[NT*64], B[NT*64];
  unsigned s = 12345u;
  for (int i = 0; i < NT*64; i++) {
    s = s * 1103515245u + 12345u;
    int ma = 1024 + (int)((s >> 7) & 0x3ff);       /* 11-bit significand   */
    int ea = -6 + (int)((s >> 19) & 0xf) % 13;     /* exponent in [-6, 6]  */
    int sa = (s & 0x40) ? -1 : 1;
    A[i] = (_Float16)(sa * ldexpf((float)ma, ea - 10));
    s = s * 1103515245u + 12345u;
    int mb = 1024 + (int)((s >> 7) & 0x3ff);
    int eb = -6 + (int)((s >> 19) & 0xf) % 13;
    int sb = (s & 0x40) ? -1 : 1;
    B[i] = (_Float16)(sb * ldexpf((float)mb, eb - 10));
  }

  /* Host models. `refd` is EXACT (see the input design note). The two ordered
     models are named closed forms: element-wise agreement with one of them is a
     far stronger statement than sitting inside a bound, and §6.1 turns exactly
     that distinction into how much friction a user is made to accept. */
  static double refd[NT*64], sabs[NT*64];
  static float  seq32[NT*64], tree32[NT*64];
  static double ref16[NT*64], tree16[NT*64];
  for (int t = 0; t < NT; t++)
    for (int r = 0; r < 8; r++)
      for (int c = 0; c < 8; c++) {
        const _Float16 *a = A + t*64, *b = B + t*64;
        int o = t*64 + r*8 + c;
        double acc = 0.0, sa = 0.0;
        float p[8];
        for (int k = 0; k < 8; k++) {
          p[k] = (float)a[r*8+k] * (float)b[k*8+c];   /* EXACT in binary32 */
          acc += (double)p[k]; sa += fabs((double)p[k]);
        }
        refd[o] = acc; sabs[o] = sa;
        { float f = 0.0f; for (int k = 0; k < 8; k++) f += p[k]; seq32[o] = f; }
        { float l0 = (p[0]+p[1]) + (p[2]+p[3]);
          float l1 = (p[4]+p[5]) + (p[6]+p[7]);
          tree32[o] = l0 + l1; }
        { _Float16 h = (_Float16)0.0f;
          for (int k = 0; k < 8; k++) h = (_Float16)((float)h + p[k]);
          ref16[o] = (double)(float)h; }
        { _Float16 h0 = (_Float16)((float)(_Float16)(p[0]+p[1]) + p[2]);
          h0 = (_Float16)((float)h0 + p[3]);
          _Float16 h1 = (_Float16)((float)(_Float16)(p[4]+p[5]) + p[6]);
          h1 = (_Float16)((float)h1 + p[7]);
          tree16[o] = (double)(float)(_Float16)((float)h0 + (float)h1); }
      }

  /* K-1 = 7 binary32 additions in any order: |err| <= (7u)/(1-7u) * sum|p| */
  const double U32 = ldexp(1.0, -24);
  const double BOUND32 = (7.0 * U32) / (1.0 - 7.0 * U32);
  const double U16 = ldexp(1.0, -11);
  const double BOUND16 = (7.0 * U16) / (1.0 - 7.0 * U16);
  printf("  derived bound, f32 accumulate: %.3g * sum|p_i|\n", BOUND32);
  printf("  derived bound, f16 accumulate: %.3g * sum|p_i|\n", BOUND16);

  id<MTLBuffer> bA = mkbuf(dev, A, sizeof A), bB = mkbuf(dev, B, sizeof B);

  /* ANTI-VACUITY CONTROL. Before any device number is read: does the f32 bound
     reject anything at all on this input set? A host binary16 accumulator must
     be rejected by it, and the exactly-summed reference must be accepted. If
     the first number is 0 the gate cannot fail and no pass below is readable. */
  {
    int rej = 0, self = 0; double worst = 0.0;
    for (int i = 0; i < NT*64; i++) {
      double e = fabs(ref16[i] - refd[i]);
      double rel = sabs[i] > 0 ? e / sabs[i] : 0.0;
      if (rel > worst) worst = rel;
      if (e > BOUND32 * sabs[i]) rej++;
      if (fabs(refd[i] - refd[i]) > BOUND32 * sabs[i]) self++;
    }
    printf("  ANTI-VACUITY CONTROL:\n");
    printf("    host binary16 accumulator rejected by the f32 bound: %d / %d "
           "(worst rel. error %.3g) -> %s\n", rej, NT*64, worst,
           rej > 0 ? "the f32 bound CAN fail"
                   : "*** the f32 bound cannot fail here; do not trust a pass ***");
    printf("    exact reference accepted by its own bound          : %d rejected\n",
           self);
    /* And the f32 accumulation order must actually be observable, or the
       element-wise model comparison below is equally vacuous. */
    int sep = 0;
    for (int i = 0; i < NT*64; i++) if (seq32[i] != tree32[i]) sep++;
    printf("    sequential vs pairwise binary32 order differ on    : %d / %d "
           "-> %s\n", sep, NT*64,
           sep > 0 ? "accumulation order IS observable on this input set"
                   : "*** order is unobservable here; the model test is vacuous ***");
  }

  struct { const char *label; const char *src; int f32out; } runs[] = {
    { "simdgroup_half8x8 x half8x8 -> float8x8", SRC_MMA_F32, 1 },
    { "simdgroup_half8x8 x half8x8 -> half8x8",  SRC_MMA_F16, 0 },
  };
  for (int r = 0; r < 2; r++) {
    NSError *e = nil;
    id<MTLLibrary> lib = build(dev, runs[r].src, &e);
    if (!lib) { printf("  %s: COMPILE FAILED: %s\n", runs[r].label,
                       e.localizedDescription.UTF8String); continue; }
    id<MTLFunction> f = [lib newFunctionWithName:@"k"];
    id<MTLComputePipelineState> ps =
      [dev newComputePipelineStateWithFunction:f error:&e];
    if (!ps) { printf("  %s: PIPELINE FAILED\n", runs[r].label); continue; }
    size_t osz = (size_t)NT * 64 * (runs[r].f32out ? 4 : 2);
    id<MTLBuffer> bO = mkbuf(dev, NULL, osz);
    id<MTLCommandBuffer> cb = [q commandBuffer];
    id<MTLComputeCommandEncoder> en = [cb computeCommandEncoder];
    [en setComputePipelineState:ps];
    [en setBuffer:bO offset:0 atIndex:0];
    [en setBuffer:bA offset:0 atIndex:1];
    [en setBuffer:bB offset:0 atIndex:2];
    [en dispatchThreads:MTLSizeMake(32*NT,1,1)
          threadsPerThreadgroup:MTLSizeMake(32,1,1)];
    [en endEncoding]; [cb commit]; [cb waitUntilCompleted];
    if (cb.error) { printf("  %s: EXEC FAILED: %s\n", runs[r].label,
                           cb.error.localizedDescription.UTF8String); continue; }

    double bound = runs[r].f32out ? BOUND32 : BOUND16;
    int N = NT*64;
    int over = 0, exact = 0, eqseq = 0, eqtree = 0, eq16 = 0, eq16t = 0, eqnar = 0;
    double worst = 0.0; int worsti = -1;
    for (int i = 0; i < N; i++) {
      double g = runs[r].f32out ? (double)((const float *)bO.contents)[i]
                                : (double)(float)((const _Float16 *)bO.contents)[i];
      double err = fabs(g - refd[i]);
      double rel = sabs[i] > 0 ? err / sabs[i] : 0.0;
      if (rel > worst) { worst = rel; worsti = i; }
      if (err > bound * sabs[i]) over++;
      if (g == refd[i]) exact++;
      if (g == (double)seq32[i]) eqseq++;
      if (g == (double)tree32[i]) eqtree++;
      if (g == ref16[i]) eq16++;
      if (g == tree16[i]) eq16t++;
      if (g == (double)(float)(_Float16)seq32[i]) eqnar++;
    }
    printf("  %s\n", runs[r].label);
    printf("    bit-equal to the exact dot product          : %d / %d\n", exact, N);
    printf("    bit-equal to SEQUENTIAL binary32 accumulate : %d / %d\n", eqseq, N);
    printf("    bit-equal to PAIRWISE   binary32 accumulate : %d / %d\n", eqtree, N);
    printf("    bit-equal to SEQUENTIAL binary16 accumulate : %d / %d\n", eq16, N);
    printf("    bit-equal to PAIRWISE   binary16 accumulate : %d / %d\n", eq16t, N);
    printf("    bit-equal to seq. binary32 NARROWED at the end : %d / %d\n",
           eqnar, N);
    /* Element-wise agreement is the contract (§1.2), so when a model is close
       but not exact the residue is the interesting part, not the count. */
    if (eq16 > 0 && eq16 < N) {
      int shown = 0;
      printf("    residue against SEQUENTIAL binary16 (first few):\n");
      for (int i = 0; i < N && shown < 5; i++) {
        double g = runs[r].f32out ? (double)((const float *)bO.contents)[i]
                                  : (double)(float)((const _Float16 *)bO.contents)[i];
        if (g == ref16[i]) continue;
        printf("      [%5d] device=%-14.9g seq16=%-14.9g seq32=%-14.9g "
               "narrowed(seq32)=%-14.9g exact=%.9g\n",
               i, g, ref16[i], (double)seq32[i],
               (double)(float)(_Float16)seq32[i], refd[i]);
        shown++;
      }
    }
    printf("    worst error / sum|p_i|                      : %.3g  (bound %.3g)\n",
           worst, bound);
    printf("    elements outside the bound                  : %d / %d -> %s\n",
           over, N, over == 0 ? "WITHIN BOUND" : "OUTSIDE BOUND");
    if (worsti >= 0 && worst > 0)
      printf("    worst element                               : device=%.17g exact=%.17g\n",
             runs[r].f32out ? (double)((const float *)bO.contents)[worsti]
                            : (double)(float)((const _Float16 *)bO.contents)[worsti],
             refd[worsti]);
  }

  /* §1.4(a) re-run in process, and §1.4(c) vary the dispatch shape. The latter
     is REPORTED, not asserted: for coopmat the mapping of matrix components to
     invocations is implementation-dependent, so a shape-dependent result would
     be legitimate and is exactly what §1.4 says has never been measured. */
  printf("\n=== determinism (§1.4) ===\n");
  {
    NSError *e = nil;
    id<MTLLibrary> lib = build(dev, SRC_MMA_F32, &e);
    if (lib) {
      id<MTLComputePipelineState> ps = [dev newComputePipelineStateWithFunction:
          [lib newFunctionWithName:@"k"] error:&e];
      size_t osz = (size_t)NT * 64 * 4;
      float *ref = malloc(osz);
      int tgs[] = { 32, 32, 64, 128, 256 };
      for (unsigned p = 0; p < sizeof(tgs)/sizeof(tgs[0]); p++) {
        id<MTLBuffer> bO = mkbuf(dev, NULL, osz);
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> en = [cb computeCommandEncoder];
        [en setComputePipelineState:ps];
        [en setBuffer:bO offset:0 atIndex:0];
        [en setBuffer:bA offset:0 atIndex:1];
        [en setBuffer:bB offset:0 atIndex:2];
        [en dispatchThreads:MTLSizeMake(32*NT,1,1)
              threadsPerThreadgroup:MTLSizeMake(tgs[p],1,1)];
        [en endEncoding]; [cb commit]; [cb waitUntilCompleted];
        if (p == 0) { memcpy(ref, bO.contents, osz); continue; }
        int d = 0;
        const float *g = (const float *)bO.contents;
        for (size_t i = 0; i < osz/4; i++) if (memcmp(&g[i], &ref[i], 4)) d++;
        printf("  threadgroup %3d : %d / %zu elements differ from the "
               "threadgroup-32 run%s\n", tgs[p], d, osz/4,
               d == 0 ? "" : "   <<< SHAPE-DEPENDENT");
      }
      free(ref);
    }
  }
  printf("  (a fresh-process re-run is checked by running this binary twice "
         "and diffing its output)\n");
  return 0;
} }
