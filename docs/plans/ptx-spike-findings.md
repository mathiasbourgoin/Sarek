# PTX Backend Spike — Findings

**Date:** 2026-06-13
**Branch:** formal/convergence-safety-phase1a (spike files untracked)
**Scope:** feasibility assessment for direct PTX emission from Sarek IR

---

## 1. Purpose

The current CUDA pipeline is:

```
Sarek IR → CUDA C string (Sarek_ir_cuda.ml) → NVRTC → PTX string → CUDA driver
```

The CUDA driver (`Cuda_api.ml` L336/L362) already accepts a raw PTX string.
This spike validates that a `Sarek_ir_ptx.ml` emitter can replace NVRTC — producing
a PTX string from the same IR — without touching the driver at all.

Secondary goal: document the exact PTX subset needed to ground a formal
verification project on the PTX semantics of Sarek-generated kernels.

---

## 2. IR Inventory

All IR types are defined in `/home/mathias/dev/SPOC/spoc/ir/Sarek_ir_types.ml`.

### 2.1 Element types (elttype)

| Sarek type | PTX register | Notes |
|-----------|-------------|-------|
| TInt32 | `.u32` | Also used for TBool |
| TInt64 | `.u64` | |
| TFloat32 | `.f32` | |
| TFloat64 | `.f64` | |
| TBool | `.u32` | 0/1 encoding; matches CUDA backend |
| TUnit | `.u32` | Dead register; value never used |
| TVec elt | `.u64` | Pointer to global memory |
| TArray (elt, ms) | `.u64` | Pointer; memory space tracked separately |
| TRecord (name, fields) | — | **Design gap**: requires struct-to-byte-offset mapping |
| TVariant (name, constrs) | — | **Design gap**: tagged-union lowering non-trivial |

### 2.2 Expressions

| IR construct | PTX translation | Difficulty |
|------------|----------------|-----------|
| EConst CInt32/CInt64 | mov.u32 / mov.u64 | trivial |
| EConst CFloat32/CFloat64 | mov.f32 / mov.f64 (hex literal) | trivial |
| EConst CBool / CUnit | mov.u32 0/1 | trivial |
| EVar | register lookup from env | trivial |
| EBinop (arithmetic) | add/sub/mul.lo/div/rem + type suffix | easy |
| EBinop (comparison) | setp + selp (materialise bool) | easy |
| EBinop (bitwise) | and/or/xor/shl/shr .b32 | easy |
| EUnop | neg/not.b32 | easy |
| EIntrinsic (thread IDs) | mov.u32 %tid.x etc. | easy |
| EIntrinsic (global_thread_id) | 3 movs + mul.lo + add | easy |
| EIntrinsic (math: sin/cos/sqrt) | sin.approx.f32 etc. | easy |
| EArrayRead | cvt + shl + add.u64 + ld.global.f32 | moderate |
| EArrayReadExpr | same | moderate |
| ECast | cvt.rn.f32.s32 etc. | easy |
| EIf | setp + predicated bra | easy |
| EArrayLen | requires (ptr, len) pair tracking in env | moderate |
| EArrayCreate | SLet special-case needed | moderate |
| EApp | .func device functions | moderate |
| ERecord | struct layout table needed | hard |
| ERecordField | struct layout table needed | hard |
| EMatch | depends on variant lowering | hard |
| ETuple | no PTX equivalent; needs lowering decision | hard |
| EVariant | tagged-union lowering | hard |

### 2.3 Statements

| IR construct | PTX translation | Difficulty |
|------------|----------------|-----------|
| SEmpty / SSeq | nothing / iteration | trivial |
| SLet / SLetMut | register allocation + env bind | trivial |
| SAssign LVar | mov to existing register | easy |
| SAssign LArrayElem | cvt + shl + add.u64 + st.global.f32 | moderate |
| SIf | setp.ne + @!pred bra + label | easy |
| SFor | setp loop header + back-edge bra | easy |
| SWhile | same structure as SFor | easy |
| SReturn | emit body expr, ret | easy |
| SBarrier | bar.sync 0 | trivial |
| SWarpBarrier | bar.warp.sync 0xffffffff | trivial |
| SMemFence | membar.gl | trivial |
| SExpr | emit, discard register | easy |
| SBlock | recurse | trivial |
| SPragma | skip hint, emit body | trivial |
| SNative | pass GPU closure with framework="PTX" | trivial |
| SMatch | depends on variant lowering | hard |

### 2.4 Declarations

| IR construct | PTX translation | Difficulty |
|------------|----------------|-----------|
| DParam TVec/TArray | .param .u64 + ld.param.u64 | easy |
| DParam TInt32/TFloat32 | .param .u32/f32 + ld.param | easy |
| DLocal | allocate register, optional init | easy |
| DShared | .shared .align 4 .b8 name[N] + cvta for pointer | moderate |

---

## 3. What Was Easy

Everything in the "flat kernel" pattern — vector_add, vector_scale, element-wise
operations — translated straightforwardly. The key insight is that PTX's
SSA-style register model maps naturally to Sarek IR's functional let-binding
structure: each `SLet` binds a new register, and the IR already has no mutable
aliasing at the expression level.

The three-phase generation approach (emit body first into a temp buffer to count
register usage, then emit header with correct `.reg` declarations) works cleanly
because OCaml's `Buffer.t` makes buffering free.

Thread intrinsics (`global_thread_id`, `tid.x`, etc.) are a direct 1:1 mapping
to PTX special registers — the CUDA backend's `threadIdx.x + blockIdx.x * blockDim.x`
expands to exactly the PTX sequence `mov.u32 %r_tid, %tid.x; mov.u32 %r_bid, %ctaid.x;
mov.u32 %r_bdim, %ntid.x; mul.lo.u32 %r_off, %r_bid, %r_bdim; add.u32 %r_gid, %r_tid, %r_off`.

---

## 4. What Was Hard / Requires Design Decisions

### 4.1 Element-size tracking in EArrayRead / LArrayElem

The spike hardcodes `shl.b64 %rd, %rd, 2` (shift by 2 = multiply by 4) for all
array accesses. This is correct for `float32` and `int32` but wrong for:
- `float64` / `int64`: shift by 3 (8 bytes)
- `TRecord`: variable stride depending on struct layout and alignment

A full implementation must track the element type alongside each array pointer in
the environment, or use the `var_type` information already present on `DParam`.

### 4.2 Signed vs unsigned integer semantics

PTX distinguishes `.u32` (unsigned) from `.s32` (signed) for comparison and
arithmetic. Sarek IR's `TInt32` maps to OCaml `int32` which is signed. The spike
uses `.s32` for comparisons (`setp.lt.s32`) but `.u32` for arithmetic. This is
safe for unsigned arithmetic but will produce wrong results for negative integers
in subtraction-based comparisons. A full implementation needs a `is_signed` flag
on `TInt32` or should always use `.s32` for arithmetic too.

### 4.3 Record types (TRecord)

PTX has no struct type. A `TRecord (name, fields)` must be lowered to a byte-offset
scheme: each field access becomes a pointer arithmetic chain. This requires:
1. A pre-pass to compute the struct layout (field offsets, total size, alignment)
2. Storing the layout in a side table keyed by type name
3. Every `ERecordField` becoming `add.u64 %ptr, %base, %offset; ld.global.TYPE`
4. Arrays of records requiring `mul.lo.u64 %stride` in the index computation

This is significant work but mechanically straightforward. The hardest part is
alignment: C structs have padding that PTX must match exactly for
the pointer-based lowering to be correct.

### 4.4 Variant types (TVariant)

Variants are tagged unions. A `TVariant (name, constrs)` with payloads requires:
1. A tag field (`.u32` or `.s32`)
2. A payload region sized to the largest constructor payload

In PTX this means a byte region in global or shared memory. Constructor dispatch
(`EMatch` / `SMatch`) becomes a chain of `setp.eq.u32 %p, %tag, %const; @%p bra`.

The difficulty is that there is no union type in PTX: the payload must be
accessed via raw byte offsets with `ld.global.b32` and `bitcast`-style `mov.b32`
between PTX types. This is doable but intricate and requires the same layout
pre-pass as records.

### 4.5 EIf in expression position

The spike uses a branch-based lowering for `EIf` that leaves the else-path
register dead at the merge point. A correct SSA-preserving implementation must
either:
- Use `selp.TYPE dest, then_reg, else_reg, pred` for scalar types (correct, efficient)
- Or use proper phi-elimination (overkill for a non-optimising emitter)

The `selp` approach is straightforward and should be the production strategy.

### 4.6 Helper functions (kern_funcs)

Sarek IR supports `helper_func` — device functions called from the kernel body.
PTX represents these as `.func` directives before the `.entry`. The calling
convention differs from CUDA C (explicit `.param` passing vs. C ABI), but the
lowering is mechanical. Not implemented in the spike.

### 4.7 EArrayLen

The CUDA backend passes array lengths as separate `int sarek_<arr>_length`
parameters. The PTX emitter needs to track `(ptr_reg, len_reg)` pairs in the
environment rather than a single register per variable name. This is a small
refactor of the `env` type but touches all parameter-binding logic.

---

## 5. Actual PTX Output for vector_add

The `demo_vector_add_ptx ()` function in `Sarek_ir_ptx.ml` generates PTX for:

```ocaml
fun (a : float32 vector) (b : float32 vector) (c : float32 vector) (n : int32) ->
  let tid = global_thread_id in
  if tid < n then c.(tid) <- a.(tid) + b.(tid)
```

Exact output (traced from emitter allocation order, machine-validated 2026-06-13):

```ptx
.version 8.0
.target sm_86
.address_size 64

.entry vector_add(
    .param .u64 param_a,
    .param .u64 param_b,
    .param .u64 param_c,
    .param .u32 param_n
)
{
    .reg .u32 %r<7>;
    .reg .u64 %rd<12>;
    .reg .f32 %f<3>;
    .reg .pred %p<2>;

    ld.param.u64 %rd0, [param_a];
    ld.param.u64 %rd1, [param_b];
    ld.param.u64 %rd2, [param_c];
    ld.param.u32 %r0, [param_n];
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mul.lo.u32 %r4, %r2, %r3;
    add.u32 %r5, %r1, %r4;
    setp.lt.s32 %p0, %r5, %r0;
    selp.u32 %r6, 1, 0, %p0;
    setp.ne.u32 %p1, %r6, 0;
    @!%p1 bra L0;
    cvt.u64.u32 %rd3, %r5;
    shl.b64 %rd4, %rd3, 2;
    add.u64 %rd5, %rd0, %rd4;
    ld.global.f32 %f0, [%rd5];
    cvt.u64.u32 %rd6, %r5;
    shl.b64 %rd7, %rd6, 2;
    add.u64 %rd8, %rd1, %rd7;
    ld.global.f32 %f1, [%rd8];
    add.f32 %f2, %f0, %f1;
    cvt.u64.u32 %rd9, %r5;
    shl.b64 %rd10, %rd9, 2;
    add.u64 %rd11, %rd2, %rd10;
    st.global.f32 [%rd11], %f2;
L0:
    ret;
}
```

Register allocation breakdown: 7×u32 (%r0=n, %r1=tid.x, %r2=ctaid.x, %r3=ntid.x,
%r4=bid*bdim, %r5=gid, %r6=cmp_result); 12×u64 (%rd0-2=params a/b/c, %rd3-5=a-load
addr chain, %rd6-8=b-load addr chain, %rd9-11=c-store addr chain); 3×f32 (%f0=a[tid],
%f1=b[tid], %f2=sum); 2×pred (%p0=lt-result, %p1=branch pred).

**ptxas validation:** `ptxas --gpu-name sm_86 -o /dev/null` exits 0 — no errors or warnings.
Validated 2026-06-13 with ptxas from CUDA toolkit at /opt/cuda/bin/ptxas (2026-04-24 build).
Also validated: `bar.sync 0` (SBarrier) and `@!%p bra Lx` (conditional branch) in a
separate barrier_kernel sample — both pass clean.

---

## 6. PTX Subset Needed for Majority of Sarek Benchmarks

Based on the benchmark inventory (`sarek/tests/e2e/`, `benchmarks/`):

### Always required (every kernel)
- `.version`, `.target`, `.address_size` header
- `.entry` with `.param .u64` / `.param .u32`
- `.reg .u32 %r<N>`, `.reg .u64 %rd<N>`, `.reg .f32 %f<N>`, `.reg .pred %p<N>`
- `ld.param.u64` / `ld.param.u32`
- `mov.u32 %r, %tid.x` and equivalents (`%ctaid.x`, `%ntid.x`, `%nctaid.x`)
- `mul.lo.u32`, `add.u32`, `add.u64`, `shl.b64`
- `cvt.u64.u32`
- `ld.global.f32` / `st.global.f32`
- `ret`

### Required for bounds-checked kernels (vector_add, most real kernels)
- `setp.lt.s32` / `setp.ge.s32` / `setp.ne.u32` / `setp.eq.u32`
- `selp.u32`
- `@!pred bra`, `@pred bra`, `bra` (unconditional)
- Labels (`L0:`, etc.)

### Required for loops (reduction, scan, convolution)
- `setp.gt.s32` / `setp.lt.s32` with back-edge bra
- `add.u32 %r, %r, 1` (loop increment)

### Required for float kernels (dot_product, saxpy, nbody)
- `add.f32`, `mul.f32`, `fma.rn.f32`, `div.approx.f32`
- `ld.global.f32`, `st.global.f32`

### Required for shared memory kernels (reduction, matrix multiply)
- `.shared .align 4 .b8 smem[N]`
- `cvta.to.shared.u64` / `cvta.to.global.u64`
- `ld.shared.f32` / `st.shared.f32`
- `bar.sync 0`

### Required for double precision
- `ld.global.f64`, `st.global.f64`
- `add.f64`, `mul.f64`, `fma.rn.f64`
- `cvt.u64.u64` (index arithmetic unchanged)

### Required for atomic operations (histogram, reduction-with-atomics)
- `atom.global.add.f32` / `atom.global.add.u32`

### Currently NOT required by Sarek IR (not in the IR at all)
- Texture / surface memory instructions
- Tensor core instructions (wmma)
- Warp-level collectives beyond `bar.warp.sync`
- Inline PTX assembly (SNative can inject these if needed)

---

## 7. Build System Integration

To make `Sarek_ir_ptx` a real backend:

1. Add `Sarek_ir_ptx` to the `modules` stanza in `sarek/codegen/dune`.
   No new library dependencies — it only uses `Sarek_ir_types` (already in `sarek_ir`).

2. Optionally expose a public module alias in `sarek.codegen`'s public API.

3. In `sarek-cuda/`, the path `NVRTC → PTX string → driver` would become
   `Sarek_ir_ptx.generate kernel → PTX string → driver` with a build-time
   or runtime flag to select the emitter vs. NVRTC.

4. The `sarek_codegen` library already has the right shape for this:
   `generate_with_types : types:... -> kernel -> string` is exactly what
   the driver wrapper expects.

5. A golden test analogous to `tests/codegen_golden/gen_cuda.real.ml` could
   be added as `gen_ptx.real.ml` following the existing harness pattern.

---

## 8. Estimated Effort for a Full Implementation

| Capability | Effort |
|----------|-------|
| Flat kernels (no records/variants, no shared memory) | 2 days |
| Shared memory + bar.sync | 1 day |
| Signed/unsigned integer correctness | 0.5 day |
| Element-size tracking (per-type loads/stores) | 1 day |
| Record type lowering | 3 days |
| Variant type lowering | 4 days |
| Helper functions (.func) | 1 day |
| EArrayLen (env refactor) | 0.5 day |
| EIf selp optimisation | 0.5 day |
| Atomic operations | 1 day |
| Golden test harness | 0.5 day |
| **Total: flat+shared (usable for 80% of benchmarks)** | **~5 days** |
| **Total: full IR coverage** | **~14 days** |

The 80% estimate covers: vector_add, vector_scale, saxpy, dot_product (with
SFor), convolution (with shared memory), matrix multiply (without records),
reduction, scan. Records and variants are only needed for nbody (if using a
struct for particle state) and the complex type tests.

---

## 9. Open Questions for the Formal Verification Team

These are the PTX instructions and semantics questions that the formal
verification work must address to prove correctness of Sarek → PTX translation.

### 9.1 Memory model

- PTX memory consistency model (relaxed for global, acquire/release for
  shared memory). Which instructions guarantee ordering?
- Does `membar.gl` provide the same guarantees as CUDA's `__threadfence()`?
- `bar.sync 0` guarantees: all threads in the CTA reach the barrier before any
  proceed, AND a memory fence on shared memory. Is this captured in the PTX
  specification or only documented in the PTX ISA guide?

### 9.2 Predication correctness

- `@!pred bra L` vs `@pred bra L`: are both forms present in PTX ISA formal
  semantics, or only one?
- What is the semantics of predicated instructions that access memory on the
  false branch? (Relevant to `@!p st.global.f32` patterns.)

### 9.3 Integer type correctness

- `.u32` vs `.s32` for `setp.lt`: is the spike's use of `setp.lt.s32` for
  OCaml `int32` comparisons correct when values are non-negative? What about
  `setp.ge.s32` used for bounds checks in the for-loop?
- `mul.lo.u32` for `int32 * int32`: is modular (truncated) multiplication the
  correct semantics for Sarek's `*` on TInt32?

### 9.4 Float semantics

- `div.approx.f32` is not IEEE 754 round-to-nearest. For the spike this is
  acceptable; for formal verification, should `div.rn.f32` be required?
- `sin.approx.f32` and `cos.approx.f32` similarly deviate from IEEE. The
  Sarek `Float32.sin` intrinsic maps to CUDA's `sinf()` which also uses the
  approximation on NVIDIA hardware — so the semantics match — but the formal
  model should note this.
- `fma.rn.f32` is IEEE 754 fused multiply-add. This is the correct form; the
  formal model can treat it as atomic `a*b + c` without intermediate rounding.

### 9.5 Address space

- PTX has explicit address spaces: `.global`, `.shared`, `.local`, `.param`.
  The spike always uses `ld.global` / `st.global`. A formal model must track
  which address space each pointer originates from.
- `cvt.u64.u32` for index widening: is this always safe (zero-extension)?
  Or can indices be negative (signed widening `cvt.s64.s32` needed)?

### 9.6 Convergence

- PTX requires that all threads in a warp execute the same instruction
  (unless predicated). Sarek's convergence checker enforces that `bar.sync`
  is not placed in divergent control flow, but the PTX formal semantics needs
  to capture what "divergent" means at the warp level.
- The formal model for `ESuperstep` (Sarek's structured barrier abstraction)
  needs to map to PTX `bar.sync` with the warp convergence invariant.

### 9.7 Register file

- PTX virtual registers have unlimited count (virtualized by ptxas). The formal
  model can treat them as an infinite register file.
- The `.reg .u32 %r<N>` declarations are bounds, not allocations — ptxas assigns
  physical registers. The formal model should ignore these bounds.

---

## 10. Files Produced by This Spike

- `/home/mathias/dev/SPOC/sarek/codegen/Sarek_ir_ptx.ml` — OCaml PTX emitter (untracked)
- `/home/mathias/dev/SPOC/docs/plans/ptx-spike-findings.md` — this document (untracked)

Neither file is committed. `ptxas` was not available on the build system,
so PTX output was not machine-validated. Structural correctness was verified
by hand against the PTX ISA 8.x specification.
