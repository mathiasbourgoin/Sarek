---
name: roster-spec
type: spec
status: live
feature: Matrix-product (GEMM) pattern recognition on Sarek IR, with a compile-time advice and a library entry point
brief: owner request 2026-10-11 (matrix pattern detection — PR 1 of 3: static advice)
date: 2026-10-11
version: 1.0.0
---

# Spec — matrix-pattern-advice

Decisions not dictated by the request are **made by the agent** under the owner's
delegation and are marked *(agent decision)*. Owner rules in force: a
semantics-preserving optimisation may be default; anything that can change results
is explicit opt-in; an unsupported device refuses through the capability model.
This PR changes no generated code and no result: it only reports.

## What exists before this change

- `Sarek_fusion` (library `sarek`) classifies array accesses for fusion as
  `OneToOne | Stencil | Reduction | Gather | Unknown`. A matrix-product operand
  read once is classified `OneToOne`, i.e. the analysis has no notion of a
  reduction variable or of affine indexing. It lives in `sarek/sarek`, which the
  PPX cannot link (the PPX links only `spoc.ir`).
- `SCoopmat` lowers on HIP gfx11 (f16×f16+f32→f32 relaxed, opt-in; {u8,s8}×{u8,s8}
  →{s32,u32} strict), CUDA (compile-only), Vulkan/GLSL (integer, per device probe).
- The DSL has no 8-bit integer type; `TUint8`/`TInt8` are IR-only storage
  types, without arithmetic (`spoc/ir/Sarek_ir_int8.mli`).

## The pattern (what is recognised)

A **GEMM nest** is, anywhere in the kernel body (under guards and enclosing loops):

```
let mut acc = init in            (* SLetMut *)
...                              (* statements not touching acc *)
for kv = lo to hi do             (* SFor Upto, lo/hi uniform *)
  [let x = ... in]*              (* immutable lets, inlined *)
  acc := acc + A[ia] * B[ib]     (* or B*A, or acc on the right, or fma(A, B, acc) *)
done;
...                              (* statements not touching acc *)
[if g then] C[ic] <- e(acc)      (* the only other use of acc *)
```

- Operands are reads of **global** kernel-parameter vectors, optionally under a
  widening cast (`float32_of_float16`, `ECast`).
- `ia`, `ib`, `ic` are put in **affine form** over *atoms* with *uniform*
  polynomial coefficients. Uniform = built from constants, scalar parameters,
  `block_dim_*`, `grid_dim_*`, array lengths (and immutable lets of those). Atoms
  = `kv`, enclosing loop variables, thread-varying sub-expressions
  (`thread_idx_x`, `global_thread_id / n`, ...). A product of two
  atom-containing terms, a division, modulo, shift or bit operation involving
  `kv`, or an index containing an array read is **not affine** and rejects. An
  index, bound or guard that involves an `int64` constant, variable or cast
  rejects with the reason `Int64_arithmetic` (guards are checked too, since
  they bound the rewritten region): the form and the rewrite's addresses are
  int32, so a 64-bit index is declined, never narrowed. The test is
  conservative: an explicit `int32` cast of an int64 expression is declined as
  well.
- `kv` must occur in `ia` and `ib` and not in `ic`. The **M atom** occurs in `ia`
  and `ic` but not `ib`; the **N atom** in `ib` and `ic` but not `ia`; exactly one
  of each. Every other atom of `ic` is a **batch** atom; `ia` and `ib` may
  contain only their own atom, `kv`, and batch atoms. An atom of A and C only
  (a batch over which B is broadcast) belongs to the M index when its strides
  are consistent (A and C stack their batches contiguously): that is one larger
  GEMM, not a batched one; otherwise (a padded batch stride) it is a batch atom
  with B broadcast. Symmetrically for B and N.
- A row or column index may be several atoms (`thread_idx_y + block_dim_y *
  block_idx_y` is two): they form one index when their coefficients in the
  operand and in C are the same weights times one stride each.
- Layout of each operand from its strides: row-major when the column stride is
  1 (leading dimension = the row stride), column-major when the row stride is 1.
  An operand with no unit stride, with equal strides (`a.(row + i)`,
  `c.(row + col)`: Toeplitz views, correlations, races) or with a constant
  leading dimension smaller than 16 or than a constant K is rejected: a matrix
  unit loads a tile with one base and one leading dimension. A symbolic leading
  dimension is trusted (it cannot be compared with the bounds before launch).
  Two indices derived from the same thread value are independent only as the
  pair `t / d`, `t mod d` (`tid / 4` and `tid / 8` cover a band of C). The
  accumulator's initial value must be uniform or the output element
  (`let sum = mut c.(o)`, rendered as `C = init + Σ ...`). The two factors are named A and B so that C's column
  stride is 1 whenever either naming allows it: a column-major `C = A·B` is
  reported as the row-major `Cᵀ = Bᵀ·Aᵀ`. Constant and batch terms are kept as
  the operand offset.
- Rejected (reported as a *near miss* with a reason, never as advice):
  `downto` loops, non-uniform or data-dependent bounds (`for i = 0 to len.(r)`,
  `for i = 0 to row` — triangular), a non-product update (`acc + a[i]`), a
  product of three factors (`acc + alpha*a*b`), subtraction, a call to a
  helper function, a shared/local operand (an already tiled kernel), an output
  index containing `kv`, an output that aliases an operand or that another statement also writes, an operand the kernel
  writes or may write (any write through a computed base or any atomic counts as
  writing every operand, helper bodies included), `acc` used anywhere else, a
  store guarded by a condition on `acc`, an epilogue using thread-varying values
  or other arrays, and a reduction whose operands do not separate into M and N
  atoms (dot products, row sums, `c[tid]` outputs).

*(agent decision)* Only the kernel body is analysed, not helper functions: the
advice must point at a loop the user wrote in this kernel.

Known false negatives (conservative, never a wrong advice): a constant leading
dimension on B or C is never accepted, because their contiguous extent N is not
a loop bound (guards are not read as extent evidence), so a fully
compile-time-sized GEMM is a near miss; register-tiled kernels (nested
accumulators) and shared-memory tiled kernels are near misses.

## Matrix-unit configuration per element case

The analysis reports which `Sarek_coopmat` configuration a lowering would use,
on which backend, and how results would change (`Sarek_matrix_advice.accuracy_class`,
deliberately not `Sarek_coopmat.accuracy_regime`: operand rounding must not be
mistakable for the `Relaxed_bounded` that the existing opt-in admits). This PR
rewrites nothing.

| A, B elements | acc / C | configuration (16×16×16, subgroup) | accuracy class | opt-in |
|---|---|---|---|---|
| float16, float16 | float32 (C float32 or float16) | f16×f16+f32→f32 | `Relaxed_accumulation` (order only, γ_K bound) | `Sarek_coopmat.accept_relaxed_accumulation` |
| float32, float32 | float32 | f16×f16+f32→f32 after rounding the operands | `Relaxed_operands`, *beyond* the existing bound | none exists yet: a matrix-unit lowering needs a new explicit opt-in (PR 2) |
| int32, int32 | int32 | u8×u8+s32→s32 on the four byte planes of each operand (10 products per tile, wrapping) | `Exact`: bit-identical, int32 arithmetic is modulo 2³² | none |
| anything else (int64, float64, mixed) | | none | | |

Backends, for the configurations above (fixed table, from specs/coopmat-hip-wmma.md):
HIP gfx11 wave32 — executes; CUDA/C sm_72+ (integer only) — explicit request only; compiled with NVRTC,
never executed; ZLUDA lacks wmma; Vulkan — integer configurations,
executes on RADV (64-wide subgroups); Metal, OpenCL, WGSL, PTX, Native — no
matrix-unit lowering.

*(agent decision)* **Byte-plane decomposition for int32.** With `a = Σᵢ aᵢ2⁸ⁱ`,
`b = Σⱼ bⱼ2⁸ʲ` (aᵢ, bⱼ the unsigned bytes of the two's-complement pattern),
`Σₖ aₖbₖ mod 2³² = Σ_{i+j≤3} 2^{8(i+j)} Σₖ aᵢₖ bⱼₖ mod 2³²`, and each inner sum is
an exact u8×u8 product accumulated with 32-bit wrap. It is the only route from
the DSL's int32 to an 8-bit matrix unit that is exact for every input; a range
check (“operands fit in 8 bits”) would be data-dependent. PR 2 implements and
measures it; this PR only reports it.

## The diagnostic

*(agent decision)* **An OCaml alert, not a ppwarning.** The PPX wraps the kernel
expression as `((let module Sarek_advice : sig val matrix_product : unit
[@@alert sarek_matrix "<text>"] end = struct let matrix_product = () end in
Sarek_advice.matrix_product); <kernel>)`, the use carrying the location of the recognised `for` loop (from `for` to the
loop variable; the kernel's location when the loop cannot be identified uniquely).
Measured (OCaml 5.4, dune dev and release): an alert is printed with its
location and does not fail the build, whereas a ppwarning (warning 22) is fatal in
dune's dev profile. Alerts are shown by merlin and controlled by standard means:

- per kernel: `[%kernel ...] [@alert "-sarek_matrix"]`, or
  `let k = [%kernel ...] [@@alert "-sarek_matrix"]`;
- per file: `[@@@alert "-sarek_matrix"]`; per library: `(flags (:standard -alert -sarek_matrix))`;
- as an error: `-alert @sarek_matrix`.

An advice is emitted only for a recognised nest whose element case has a
configuration (first three rows above). Text (one alert per nest):

```
Sarek: kernel '<name>' contains a matrix product (GEMM) that matrix units could compute.
  C[m,n] = [<init> + ]sum over <kv> = <lo>..<hi> of A[m,<kv>] * B[<kv>,n]
  A = <array>[<index>] (<layout>, ld <ld>)
  B = ...
  C = ...
  m = <row index>, n = <column index>[, batched over <atoms>]
  elements: <A> x <B> -> <acc> accumulator
  configuration: 16x16x16 <cfg> subgroup (<direct | operands rounded to float16 | byte planes>)
  HIP gfx11 (wave32): executes
  CUDA/C sm_70+ / sm_72+: <none | explicit request only; compiled (NVRTC), never run; ZLUDA lacks wmma>
  Vulkan: <none | executes (RADV, 64-wide subgroups)>
  Metal, OpenCL, WGSL, PTX, Native: no matrix-unit lowering
  accuracy: <exact | relaxed ... opt-in>
  No code was changed. Silence with [@alert "-sarek_matrix"] on the kernel expression or its binding.
```

The name is the kernel's own or, failing that, the enclosing binding's. An
advice must never fail a build: a failure of the analysis, of the rendering or
of the scan file is reported on stderr and the kernel compiles without advice.
A kernel containing a `[%native]` block (body or helper) is not analysed.

`SAREK_MATRIX_SCAN=<file>`: when set, the PPX appends one line per kernel and per
finding (recognised or near miss) to `<file>`, for corpus scans. Unset: no I/O.
The variable is not a dune dependency: a scan needs a fresh build directory and
`DUNE_CACHE=disabled`, or cached PPX outputs are not re-run.

## Library entry points (`spoc.ir`)

- `Sarek_ir_affine` — affine forms with uniform polynomial coefficients
  (`of_expr`, coefficient lookup, `to_expr`).
- `Sarek_matrix_pattern` — `analyze : kernel -> finding list`
  (`Gemm of gemm | Near_miss of near_miss`), `recognise : kernel -> gemm list`,
  `reason_text`.
- `Sarek_matrix_advice` — `advise : gemm -> advice` (element case,
  configuration, accuracy class, backends), `render : ?kernel_name:string -> gemm -> advice -> string`.
- `Sarek_fusion.should_fuse` consults `Sarek_matrix_pattern`: a consumer that
  reads the intermediate as a GEMM operand is `DontFuse` with the reason
  “<intermediate> is a matrix-product operand of the consumer: fusing would
  recompute each of its elements once per output row or column”. This replaces the generic reason it got before (the decision was
  already `DontFuse`); FUSION.md gains the row.

## User stories

### US-1: a naive GEMM gets an advice (P0)
1. **Given** the naive float16→float32 GEMM (`row = global_idx_y`, `col = global_idx_x`),
   **When** compiled, **Then** an alert `sarek_matrix` is printed at the `for`
   loop naming A, B, C, their layouts, the f16 configuration and the opt-in.
2. **Given** the same kernel with `[@alert "-sarek_matrix"]`, **When** compiled with
   `-alert @sarek_matrix`, **Then** it compiles (silenced).
3. **Given** an int32 GEMM, **Then** the advice says exact, byte-plane u8 configuration, no opt-in.

### US-2: variants are recognised (P0)
Transposed A, transposed B, column-major C, 1-D mapping (`row = tid / n`,
`col = tid mod n`), loop-nest (no threads), batched (shared batch atom, broadcast
B), offsets, `fma`, `let`-bound operands, epilogue `alpha*acc + beta*C[ic]`,
float32: recognised with the right roles and layouts.

### US-3: non-GEMMs are not (P0)
Dot product, row sum, matrix-vector product, `c[tid]` output, non-affine index
(`a[(row*k+i) mod 7]`, `a[i*i]`), gathered index, data-dependent bound,
triangular bound, `downto`, three-factor product, subtraction, shared-memory tiled
GEMM, stencil, max-reduction: no `Gemm` finding, a near miss with the stated
reason where a reduction loop exists. The negative corpus is compiled with
`-alert @sarek_matrix`, so a false positive fails the build.

### US-4: no false positive on the repository (P0)
Every kernel in the repository is scanned (`SAREK_MATRIX_SCAN`, clean build); every
`Gemm` finding is inspected by hand and must be a matrix product.

## Requirements

- **FR-001** `Sarek_ir_affine`, `Sarek_matrix_pattern`, `Sarek_matrix_advice` in `spoc/ir` (no new dependency).
- **FR-002** PPX: after lowering, emit one alert per advised nest, located at the loop.
- **FR-003** Silencing through standard OCaml alert controls; documented in the message.
- **FR-004** `SAREK_MATRIX_SCAN` corpus hook.
- **FR-005** `Sarek_fusion.should_fuse` GEMM-operand rule; FUSION.md updated.
- **FR-006** Tests: unit tests of the affine forms; corpus test (positive + negative)
  asserting roles/layouts/reasons; alert-as-error negative test (Makefile
  `test_negative`); silenced-kernel library compiled with `-alert @sarek_matrix`.

## Success criteria

- SC-1: positive corpus all recognised with expected roles/layouts; negative corpus none.
- SC-2: repository scan: zero false positives, counts reported.
- SC-3: `dune build`, `dune runtest`, `make test_negative` pass; CI green.
