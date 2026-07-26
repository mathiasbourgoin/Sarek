/* metal_math_mode_probe.m — backlog #125
 *
 * Does MTLCompileOptions' math setting actually CHANGE RESULTS on Metal, or is
 * it accepted and ignored?
 *
 * That distinction is the whole point. A compile that succeeds proves plumbing,
 * not semantics — the same trap that made the OpenCL FP liveness control
 * valuable (docs/fp-contraction-policy.md §9.2: rusticl accepts every
 * -cl-* FP option and honours none of them). Until this probe was run, Sarek
 * asked Metal for non-fast math and had no evidence Metal was listening.
 *
 * Build (no Xcode needed; the Command Line Tools are enough, because
 * newLibraryWithSource:options:error: compiles through the driver at runtime,
 * which is exactly the call under test):
 *
 *   clang -fobjc-arc -O2 -framework Foundation -framework Metal \
 *       metal_math_mode_probe.m -o metal_math_mode_probe
 *
 * METHOD
 *
 * The kernel under test is `out = a*b + c` — the shape a compiler is free to
 * contract into a single fma, removing a rounding the Sarek DSL promises
 * (§1 corollary 1).
 *
 * Three reference values are computed per element:
 *
 *   ref  = fl(fl(a*b) + c)   the separately-rounded value Sarek mandates.
 *                            Computed in TWO KERNEL PASSES with the product
 *                            round-tripped through device memory in between,
 *                            so no compiler at any optimisation level can fuse
 *                            it. An in-kernel "reference" would be contractible
 *                            too and would silently agree with whatever the
 *                            test expression did.
 *   dfma = fma(a, b, c)      the DEVICE's own fma, read from the device, not
 *                            modelled. The policy doc records why this matters
 *                            (§6): RADV's fma is not correctly rounded, so
 *                            inputs chosen against an IEEE model can collide
 *                            with what the hardware actually returns and
 *                            produce a false "they agree".
 *   test = a*b + c           one expression, compiler's choice.
 *
 * Inputs are then SELECTED to the subset where ref != dfma bit-for-bit — the
 * only elements on which contraction is observable at all. On that subset,
 * `test` matching `ref` means not contracted; matching `dfma` means contracted.
 *
 * If `test` is bit-identical across every math setting, Metal accepts the
 * option and ignores it, and Metal's FP options belong in the same class as
 * rusticl's.
 *
 * ---------------------------------------------------------------------------
 * MEASURED 2026-07-26 on Apple M4, macOS 15.6.1 (24G90), arm64, Apple clang
 * 17.0.0 (clang-1700.0.13.5), Metal.framework via the Command Line Tools SDK.
 * 65536 inputs, 8773 of them observable.
 *
 * Defaults, read from a freshly constructed MTLCompileOptions:
 *   mathMode                   = 2  (MTLMathModeFast)
 *   mathFloatingPointFunctions = 0  (MTLMathFloatingPointFunctionsFast)
 * BOTH Metal defaults are the fast one. "Metal defaults to fast math" was
 * previously quoted from Apple's documentation; it is now measured, and it is
 * two knobs rather than one.
 *
 * THE OPTIONS ARE HONOURED (vs default, on sqrt(a) + 1/a, 65536 elements):
 *   mathMode=Safe                              16017 results change
 *   mathMode=Safe + fpFunctions=Precise        22135 results change
 *   fastMathEnabled=NO                         22135 results change
 *   mathMode=Fast / fastMathEnabled=YES            0 (confirms the default)
 * So Metal is NOT in the class of rusticl's OpenCL FP options, which are
 * accepted and discarded (§10.2). Setting mathMode alone is not enough:
 * mathFloatingPointFunctions is a second, independent knob.
 *
 * BUT THEY DO NOT TOUCH CONTRACTION. `a*b+c` is CONTRACTED on all 8773
 * observable elements under EVERY setting above, including mathMode=Safe, and
 * `a*b+c` is bit-identical across all of them (0/65536 differ). The fix for
 * that is source-level: see tools/probes/metal_contraction_barrier_probe.m and
 * Sarek_ir_metal.metal_fp_contract_pragma.
 *
 * Full write-up: docs/fp-contraction-policy.md §10.
 * ---------------------------------------------------------------------------
 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define N 65536

static const char *SRC =
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"kernel void mul_only(device float* p [[buffer(0)]],\n"
"                     const device float* a [[buffer(1)]],\n"
"                     const device float* b [[buffer(2)]],\n"
"                     uint i [[thread_position_in_grid]]) {\n"
"  p[i] = a[i] * b[i];\n"
"}\n"
"kernel void add_only(device float* o [[buffer(0)]],\n"
"                     const device float* p [[buffer(1)]],\n"
"                     const device float* c [[buffer(2)]],\n"
"                     uint i [[thread_position_in_grid]]) {\n"
"  o[i] = p[i] + c[i];\n"
"}\n"
"kernel void explicit_fma(device float* o [[buffer(0)]],\n"
"                         const device float* a [[buffer(1)]],\n"
"                         const device float* b [[buffer(2)]],\n"
"                         const device float* c [[buffer(3)]],\n"
"                         uint i [[thread_position_in_grid]]) {\n"
"  o[i] = fma(a[i], b[i], c[i]);\n"
"}\n"
"kernel void mul_add(device float* o [[buffer(0)]],\n"
"                    const device float* a [[buffer(1)]],\n"
"                    const device float* b [[buffer(2)]],\n"
"                    const device float* c [[buffer(3)]],\n"
"                    uint i [[thread_position_in_grid]]) {\n"
"  o[i] = a[i] * b[i] + c[i];\n"
"}\n"
/* A single-precision math function, to exercise mathFloatingPointFunctions,
   which selects between metal::fast and metal::precise and defaults to fast. */
"kernel void math_fn(device float* o [[buffer(0)]],\n"
"                    const device float* a [[buffer(1)]],\n"
"                    uint i [[thread_position_in_grid]]) {\n"
"  o[i] = sqrt(a[i]) + 1.0f / a[i];\n"
"}\n";

typedef struct {
    const char *label;
    int set_math_mode;       /* -1 = leave default */
    int set_fp_functions;    /* -1 = leave default */
    int set_fast_math;       /* -1 = leave default (deprecated property) */
} variant;

static id<MTLBuffer> buf(id<MTLDevice> dev, const void *src, size_t bytes) {
    return src ? [dev newBufferWithBytes:src length:bytes options:MTLResourceStorageModeShared]
               : [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
}

static void run1(id<MTLDevice> dev, id<MTLCommandQueue> q, id<MTLLibrary> lib,
                 const char *fn, NSArray<id<MTLBuffer>> *bufs) {
    NSError *err = nil;
    id<MTLFunction> f = [lib newFunctionWithName:[NSString stringWithUTF8String:fn]];
    if (!f) { fprintf(stderr, "FATAL: no function %s\n", fn); exit(1); }
    id<MTLComputePipelineState> ps = [dev newComputePipelineStateWithFunction:f error:&err];
    if (!ps) { fprintf(stderr, "FATAL: pipeline %s: %s\n", fn, err.localizedDescription.UTF8String); exit(1); }
    id<MTLCommandBuffer> cb = [q commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    [e setComputePipelineState:ps];
    for (NSUInteger i = 0; i < bufs.count; i++) [e setBuffer:bufs[i] offset:0 atIndex:i];
    [e dispatchThreads:MTLSizeMake(N,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];
    [e endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.error) { fprintf(stderr, "FATAL: exec %s: %s\n", fn, cb.error.localizedDescription.UTF8String); exit(1); }
}

int main(void) {
  @autoreleasepool {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (!dev) { fprintf(stderr, "no Metal device\n"); return 1; }
    printf("device = %s\n", dev.name.UTF8String);

    /* ---- what the DEFAULTS actually are, read from a fresh options object -- */
    MTLCompileOptions *probe = [MTLCompileOptions new];
    printf("\n=== MTLCompileOptions defaults, as constructed ===\n");
    if ([probe respondsToSelector:@selector(mathMode)])
        printf("  mathMode                  = %ld   (Safe=0 Relaxed=1 Fast=2)\n", (long)probe.mathMode);
    else
        printf("  mathMode                  = SELECTOR ABSENT\n");
    if ([probe respondsToSelector:@selector(mathFloatingPointFunctions)])
        printf("  mathFloatingPointFunctions= %ld   (Fast=0 Precise=1)\n", (long)probe.mathFloatingPointFunctions);
    else
        printf("  mathFloatingPointFunctions= SELECTOR ABSENT\n");
    printf("  respondsToSelector(setFastMathEnabled:) = %s\n",
           [probe respondsToSelector:@selector(setFastMathEnabled:)] ? "YES" : "NO");
    printf("  respondsToSelector(setMathMode:)        = %s\n",
           [probe respondsToSelector:@selector(setMathMode:)] ? "YES" : "NO");

    /* ---- inputs -------------------------------------------------------- */
    float *A = malloc(N*4), *B = malloc(N*4), *C = malloc(N*4);
    unsigned s = 987654321u;
    for (int i = 0; i < N; i++) {
        s = s*1103515245u + 12345u; A[i] = (float)((s>>8)&0xFFFFF) / 8192.0f + 0.5f;
        s = s*1103515245u + 12345u; B[i] = (float)((s>>8)&0xFFFFF) / 8192.0f + 0.5f;
        s = s*1103515245u + 12345u; C[i] = (float)((s>>8)&0xFFFFF) / 4096.0f + 0.5f;
    }

    id<MTLCommandQueue> q = [dev newCommandQueue];
    id<MTLBuffer> bA = buf(dev,A,N*4), bB = buf(dev,B,N*4), bC = buf(dev,C,N*4);

    variant variants[] = {
      { "DEFAULT (nil options, what shipped before)", -1, -1, -1 },
      { "mathMode = Safe",                             0, -1, -1 },
      { "mathMode = Fast",                             2, -1, -1 },
      { "mathMode=Safe + fpFunctions=Precise",         0,  1, -1 },
      { "fastMathEnabled = NO  (deprecated)",         -1, -1,  0 },
      { "fastMathEnabled = YES (deprecated)",         -1, -1,  1 },
    };
    int nv = sizeof(variants)/sizeof(variants[0]);

    /* Zero-initialised: a variant whose compile fails `continue`s without
       assigning these, and the cross-variant comparison below tests them for
       NULL. Indeterminate pointers there would compare garbage and report it
       as a measurement. */
    float *test[16] = {0}, *dfma[16] = {0}, *ref[16] = {0}, *mfn[16] = {0};

    for (int v = 0; v < nv; v++) {
        MTLCompileOptions *o = [MTLCompileOptions new];
        /* mathMode / mathFloatingPointFunctions are macOS 15.0+ / iOS 18.0+.
           Assigning them on an older SDK is an unrecognised selector, i.e. a
           crash, which would read as "the probe is broken" rather than "this
           OS cannot express the setting". Report and skip instead. */
        if (variants[v].set_math_mode >= 0
            && ![o respondsToSelector:@selector(setMathMode:)]) {
            printf("  %-58s SKIPPED: setMathMode: unavailable on this OS\n",
                   variants[v].label);
            continue;
        }
        if (variants[v].set_fp_functions >= 0
            && ![o respondsToSelector:@selector(setMathFloatingPointFunctions:)]) {
            printf("  %-58s SKIPPED: setMathFloatingPointFunctions: unavailable\n",
                   variants[v].label);
            continue;
        }
        if (variants[v].set_math_mode >= 0)    o.mathMode = (MTLMathMode)variants[v].set_math_mode;
        if (variants[v].set_fp_functions >= 0) o.mathFloatingPointFunctions = (MTLMathFloatingPointFunctions)variants[v].set_fp_functions;
        if (variants[v].set_fast_math >= 0) {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
            o.fastMathEnabled = variants[v].set_fast_math ? YES : NO;
#pragma clang diagnostic pop
        }
        NSError *err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:SRC]
                                               options:o error:&err];
        if (!lib) { printf("  %-42s -> COMPILE FAILED: %s\n", variants[v].label,
                           err.localizedDescription.UTF8String); continue; }

        id<MTLBuffer> bP = buf(dev,NULL,N*4), bR = buf(dev,NULL,N*4);
        id<MTLBuffer> bF = buf(dev,NULL,N*4), bT = buf(dev,NULL,N*4), bM = buf(dev,NULL,N*4);
        /* ref: two passes, product through memory — unfusable by construction */
        run1(dev,q,lib,"mul_only",     @[bP,bA,bB]);
        run1(dev,q,lib,"add_only",     @[bR,bP,bC]);
        run1(dev,q,lib,"explicit_fma", @[bF,bA,bB,bC]);
        run1(dev,q,lib,"mul_add",      @[bT,bA,bB,bC]);
        run1(dev,q,lib,"math_fn",      @[bM,bA]);

        ref[v]=malloc(N*4); dfma[v]=malloc(N*4); test[v]=malloc(N*4); mfn[v]=malloc(N*4);
        memcpy(ref[v],  bR.contents, N*4);
        memcpy(dfma[v], bF.contents, N*4);
        memcpy(test[v], bT.contents, N*4);
        memcpy(mfn[v],  bM.contents, N*4);

        /* Restrict to elements where contraction is OBSERVABLE on this device:
           the device's own fma differs from the separately-rounded value. */
        int observable=0, matches_ref=0, matches_fma=0, matches_neither=0;
        for (int i=0;i<N;i++){
            if (memcmp(&ref[v][i], &dfma[v][i], 4) == 0) continue;
            observable++;
            if (memcmp(&test[v][i], &ref[v][i], 4)==0) matches_ref++;
            else if (memcmp(&test[v][i], &dfma[v][i], 4)==0) matches_fma++;
            else matches_neither++;
        }
        printf("\n--- %s\n", variants[v].label);
        printf("    observable elements (device fma != separately rounded): %d / %d\n", observable, N);
        printf("    a*b+c matches separately-rounded : %d\n", matches_ref);
        printf("    a*b+c matches device fma (CONTRACTED): %d\n", matches_fma);
        printf("    matches neither                  : %d\n", matches_neither);
        printf("    VERDICT: %s\n",
               observable == 0 ? "INCONCLUSIVE - no observable element, inputs are useless"
             : matches_fma == observable ? "CONTRACTED on every observable element"
             : matches_ref == observable ? "NOT contracted on any observable element"
                                         : "MIXED");
    }

    /* ---- cross-variant comparison: does the setting change anything? ---- */
    printf("\n=== does the setting CHANGE RESULTS? (bit comparison vs DEFAULT) ===\n");
    for (int v = 1; v < nv; v++) {
        if (!test[v] || !test[0]) continue;
        int d_mul=0, d_fn=0;
        for (int i=0;i<N;i++){
            if (memcmp(&test[v][i], &test[0][i], 4)) d_mul++;
            if (memcmp(&mfn[v][i],  &mfn[0][i],  4)) d_fn++;
        }
        printf("  %-42s a*b+c differs %6d/%d | sqrt+recip differs %6d/%d\n",
               variants[v].label, d_mul, N, d_fn, N);
    }
    printf("\nIf every row is 0/%d, Metal accepts these options and ignores them.\n", N);
    free(A); free(B); free(C);
  }
  return 0;
}
