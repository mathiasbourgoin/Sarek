/* metal_contraction_barrier_probe.m — backlog #125
 *
 * metal_math_mode_probe established that MTLCompileOptions' math settings ARE
 * honoured by Metal, and that NONE of them stops Metal contracting `a*b+c`
 * into an fma. This probe answers the follow-up: does anything?
 *
 * Same method as tools/probes/opencl_f16_contraction_probe.c — sweep every
 * source-level barrier expressible on the platform and report which, if any,
 * defeats the fusion. Plus the API-choice question: is the deprecated
 * `fastMathEnabled = NO` equivalent to the two modern properties?
 *
 * The separately-rounded reference is computed in TWO KERNEL PASSES with the
 * product round-tripped through device memory, so it cannot itself be fused;
 * and elements are restricted to those where the DEVICE's own `fma` differs
 * from it, since contraction is unobservable anywhere else. Both points matter
 * — see docs/fp-contraction-policy.md §6 on why inputs must be chosen against
 * a device's measured `fma` rather than an IEEE model.
 *
 * Build (Command Line Tools are enough; no Xcode, no offline `metal` compiler):
 *   clang -fobjc-arc -O2 -framework Foundation -framework Metal \
 *       metal_contraction_barrier_probe.m -o metal_contraction_barrier_probe
 *
 * ---------------------------------------------------------------------------
 * MEASURED 2026-07-26, Apple M4, macOS 15.6.1 (24G90), Apple clang 17.0.0.
 * 65536 inputs, 8773 of them observable. Every variant built with
 * mathMode=Safe + mathFloatingPointFunctions=Precise:
 *
 *   plain a*b+c (no barrier)              CONTRACTED 8773 / 8773
 *   #pragma METAL fp contract(off)        contracted    0 / 8773   <-- ADOPTED
 *   #pragma METAL fp math_mode(safe)      CONTRACTED 8773 / 8773
 *   #pragma clang fp contract(off)        contracted    0 / 8773
 *   volatile thread local                 contracted    0 / 8773
 *   volatile threadgroup local            contracted    0 / 8773
 *   device round-trip                     contracted    0 / 8773
 *   as_type bitcast round-trip            contracted    0 / 8773
 *   precise:: namespace                   does not compile (no such namespace)
 *
 * Note `#pragma METAL fp math_mode(safe)` fails exactly as the `mathMode`
 * PROPERTY does: math mode and contraction are orthogonal on Metal. That is
 * docs/fp-contraction-policy.md §1 corollary 2 — a flag that names the hazard
 * is not a mechanism that prevents it.
 *
 * `#pragma METAL fp contract(off)` is what Sarek_ir_metal now emits: file
 * scoped, no register or memory traffic, no per-expression codegen change.
 *
 * Equivalence result: `fastMathEnabled = NO` and
 * `mathMode=Safe + fpFunctions=Precise` are BIT-IDENTICAL over 65536 elements
 * of sqrt + reciprocal + sin + log + exp (0 differ). So the pre-macOS-15
 * fallback in Metal_bindings.mtl_compile_options_conformant is exact, not
 * degraded.
 * ---------------------------------------------------------------------------
 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define N 65536

static const char *PREAMBLE =
"#include <metal_stdlib>\n"
"using namespace metal;\n";

/* Reference passes, compiled identically in every variant. */
static const char *REFKERNELS =
"kernel void mul_only(device float* p [[buffer(0)]], const device float* a [[buffer(1)]],\n"
"                     const device float* b [[buffer(2)]], uint i [[thread_position_in_grid]]) {\n"
"  p[i] = a[i] * b[i]; }\n"
"kernel void add_only(device float* o [[buffer(0)]], const device float* p [[buffer(1)]],\n"
"                     const device float* c [[buffer(2)]], uint i [[thread_position_in_grid]]) {\n"
"  o[i] = p[i] + c[i]; }\n"
"kernel void explicit_fma(device float* o [[buffer(0)]], const device float* a [[buffer(1)]],\n"
"                         const device float* b [[buffer(2)]], const device float* c [[buffer(3)]],\n"
"                         uint i [[thread_position_in_grid]]) {\n"
"  o[i] = fma(a[i], b[i], c[i]); }\n";

typedef struct { const char *label; const char *pragma; const char *body; } cand;

static cand CANDS[] = {
  { "plain a*b+c (no barrier)", "",
    "  o[i] = a[i] * b[i] + c[i];\n" },
  { "#pragma METAL fp contract(off)", "#pragma METAL fp contract(off)\n",
    "  o[i] = a[i] * b[i] + c[i];\n" },
  { "#pragma METAL fp math_mode(safe)", "#pragma METAL fp math_mode(safe)\n",
    "  o[i] = a[i] * b[i] + c[i];\n" },
  { "#pragma clang fp contract(off)", "#pragma clang fp contract(off)\n",
    "  o[i] = a[i] * b[i] + c[i];\n" },
  { "volatile thread local", "",
    "  volatile thread float p = a[i] * b[i];\n  o[i] = p + c[i];\n" },
  { "volatile threadgroup local", "",
    "  threadgroup volatile float tg[64];\n  tg[i % 64] = a[i] * b[i];\n"
    "  threadgroup_barrier(mem_flags::mem_threadgroup);\n  o[i] = tg[i % 64] + c[i];\n" },
  { "device round-trip (write then read)", "",
    "  o[i] = a[i] * b[i];\n  threadgroup_barrier(mem_flags::mem_device);\n  o[i] = o[i] + c[i];\n" },
  { "as_type bitcast round-trip", "",
    "  float p = as_type<float>(as_type<uint>(a[i] * b[i]));\n  o[i] = p + c[i];\n" },
  { "separate precise:: namespace", "",
    "  float p = precise::float(a[i]) * precise::float(b[i]);\n  o[i] = p + c[i];\n" },
};
static const int NC = sizeof(CANDS)/sizeof(CANDS[0]);

static id<MTLBuffer> mkbuf(id<MTLDevice> d, const void *s, size_t n) {
  return s ? [d newBufferWithBytes:s length:n options:MTLResourceStorageModeShared]
           : [d newBufferWithLength:n options:MTLResourceStorageModeShared];
}
static int run1(id<MTLDevice> dev, id<MTLCommandQueue> q, id<MTLLibrary> lib,
                const char *fn, NSArray<id<MTLBuffer>> *bufs) {
  NSError *e=nil;
  id<MTLFunction> f=[lib newFunctionWithName:[NSString stringWithUTF8String:fn]];
  if(!f) return 0;
  id<MTLComputePipelineState> ps=[dev newComputePipelineStateWithFunction:f error:&e];
  if(!ps) return 0;
  id<MTLCommandBuffer> cb=[q commandBuffer];
  id<MTLComputeCommandEncoder> en=[cb computeCommandEncoder];
  [en setComputePipelineState:ps];
  for(NSUInteger i=0;i<bufs.count;i++)[en setBuffer:bufs[i] offset:0 atIndex:i];
  [en dispatchThreads:MTLSizeMake(N,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];
  [en endEncoding];[cb commit];[cb waitUntilCompleted];
  return cb.error==nil;
}

int main(void){ @autoreleasepool {
  id<MTLDevice> dev=MTLCreateSystemDefaultDevice();
  printf("device = %s\n\n", dev.name.UTF8String);
  id<MTLCommandQueue> q=[dev newCommandQueue];

  float *A=malloc(N*4),*B=malloc(N*4),*C=malloc(N*4);
  unsigned s=987654321u;
  for(int i=0;i<N;i++){
    s=s*1103515245u+12345u; A[i]=(float)((s>>8)&0xFFFFF)/8192.0f+0.5f;
    s=s*1103515245u+12345u; B[i]=(float)((s>>8)&0xFFFFF)/8192.0f+0.5f;
    s=s*1103515245u+12345u; C[i]=(float)((s>>8)&0xFFFFF)/4096.0f+0.5f;
  }
  id<MTLBuffer> bA=mkbuf(dev,A,N*4),bB=mkbuf(dev,B,N*4),bC=mkbuf(dev,C,N*4);

  printf("=== barrier sweep, each built with mathMode=Safe + fpFunctions=Precise ===\n");
  for(int c=0;c<NC;c++){
    char src[8192];
    snprintf(src,sizeof src,
      "%s%s"
      "kernel void test(device float* o [[buffer(0)]], const device float* a [[buffer(1)]],\n"
      "                 const device float* b [[buffer(2)]], const device float* cc [[buffer(3)]],\n"
      "                 uint i [[thread_position_in_grid]]) {\n"
      "  const device float* c = cc;\n"
      "%s"
      "}\n%s", PREAMBLE, CANDS[c].pragma, CANDS[c].body, REFKERNELS);

    MTLCompileOptions *o=[MTLCompileOptions new];
    o.mathMode=MTLMathModeSafe;
    o.mathFloatingPointFunctions=MTLMathFloatingPointFunctionsPrecise;
    NSError *err=nil;
    id<MTLLibrary> lib=[dev newLibraryWithSource:[NSString stringWithUTF8String:src] options:o error:&err];
    if(!lib){ printf("  %-38s -> COMPILE FAILED (%s)\n", CANDS[c].label,
                     err.localizedDescription.UTF8String); continue; }

    id<MTLBuffer> bP=mkbuf(dev,NULL,N*4),bR=mkbuf(dev,NULL,N*4),bF=mkbuf(dev,NULL,N*4),bT=mkbuf(dev,NULL,N*4);
    if(!run1(dev,q,lib,"mul_only",@[bP,bA,bB])||!run1(dev,q,lib,"add_only",@[bR,bP,bC])
       ||!run1(dev,q,lib,"explicit_fma",@[bF,bA,bB,bC])||!run1(dev,q,lib,"test",@[bT,bA,bB,bC])){
      printf("  %-38s -> EXEC FAILED\n", CANDS[c].label); continue; }

    float *R=bR.contents,*F=bF.contents,*T=bT.contents;
    int obs=0,mr=0,mf=0,mn=0;
    for(int i=0;i<N;i++){
      if(!memcmp(&R[i],&F[i],4)) continue;
      obs++;
      if(!memcmp(&T[i],&R[i],4)) mr++;
      else if(!memcmp(&T[i],&F[i],4)) mf++;
      else mn++;
    }
    printf("  %-38s -> observable %5d | separately-rounded %5d | CONTRACTED %5d | other %4d  %s\n",
           CANDS[c].label, obs, mr, mf, mn,
           obs==0?"(inconclusive)": mf==0?"<<< NOT CONTRACTED":"");
  }

  /* Equivalence: is deprecated fastMathEnabled=NO the same as the two modern
     properties set safe/precise, bit for bit? */
  printf("\n=== equivalence: fastMathEnabled=NO  vs  mathMode=Safe + fpFunctions=Precise ===\n");
  const char *mathsrc =
    "#include <metal_stdlib>\nusing namespace metal;\n"
    "kernel void mfn(device float* o [[buffer(0)]], const device float* a [[buffer(1)]],\n"
    "                uint i [[thread_position_in_grid]]) {\n"
    "  o[i] = sqrt(a[i]) + 1.0f/a[i] + sin(a[i]) + log(a[i]) + exp(a[i]*0.01f); }\n";
  float *out[2]={0,0};
  for(int k=0;k<2;k++){
    MTLCompileOptions *o=[MTLCompileOptions new];
    if(k==0){
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
      o.fastMathEnabled=NO;
#pragma clang diagnostic pop
    } else { o.mathMode=MTLMathModeSafe; o.mathFloatingPointFunctions=MTLMathFloatingPointFunctionsPrecise; }
    NSError *err=nil;
    id<MTLLibrary> lib=[dev newLibraryWithSource:[NSString stringWithUTF8String:mathsrc] options:o error:&err];
    if(!lib){ printf("  variant %d compile failed\n",k); continue; }
    id<MTLBuffer> bO=mkbuf(dev,NULL,N*4);
    if(!run1(dev,q,lib,"mfn",@[bO,bA])){ printf("  variant %d exec failed\n",k); continue; }
    out[k]=malloc(N*4); memcpy(out[k],bO.contents,N*4);
  }
  if(out[0]&&out[1]){
    int d=0; for(int i=0;i<N;i++) if(memcmp(&out[0][i],&out[1][i],4)) d++;
    printf("  differ on %d / %d elements -> %s\n", d, N,
           d==0?"BIT-IDENTICAL (the deprecated boolean == both modern properties)"
               :"NOT equivalent - the two spellings mean different things");
  }
  free(A);free(B);free(C);
} return 0; }
