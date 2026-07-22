SPDX-License-Identifier: CECILL-B
SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com>

# Sarek PTX emitter — candidate optimization passes

Grounded against `sarek/codegen/Sarek_ir_ptx_{expr,mem,stmt,kernel,types}.ml`
as of this session (`main`/current worktree). The emitter is a **single-pass,
register-only, source-to-source PTX text generator**: every `emit_*` function
writes instructions directly into a `Buffer.t` and returns a register name;
there is no IR-to-IR rewrite step, no CFG, no liveness analysis, and the
register allocator (`Sarek_ir_ptx_types.ml:91-119`, `new_u32`/`new_u64`/...) is
a monotonically-increasing counter — it never reuses a register once
allocated. That single fact underlies most of the findings below.

All PTX produced is fed to `ptxas` (native CUDA) or ZLUDA's PTX→AMDGPU path
before it ever runs; the honest question for every candidate is **"does
ptxas/ZLUDA already fix this downstream, or is the waste baked in before it
gets there?"** ptxas runs classic SSA-based optimization (CSE, dead-code
elimination, register allocation with spill, some load/store vectorization on
recent versions) on straight-line code with clear scalar def-use chains — most
single-block redundancy *within one basic block* is within its reach. Where it
reliably falls short: cross-warp scheduling decisions that depend on
programmer intent (`.nc`, `.maxntid`), anything requiring whole-kernel dataflow
across our emitted branch-heavy match/if chains, and register-pressure
decisions that were already committed to as separate SSA values before ptxas
sees them (ptxas coalesces, but coalescing 3800 named virtual regs is not the
same as never having emitted them).

## Safety section — passes that must NEVER change semantics

Read first because it constrains everything below.

1. **Convergence (`bar.sync`/`bar.warp.sync`).** `SBarrier`/`SWarpBarrier`
   (`Sarek_ir_ptx_stmt.ml:135-136`) are emitted unconditionally at their
   statement position — never inside a scalarized `selp`-both-branches
   rewrite. `expr_needs_branch_guard` (`Sarek_ir_ptx_expr.ml:42-69`) already
   treats a barrier reached through an expression form (`EIntrinsic
   ("block_barrier"/"warp_barrier"/"memory_fence", _)`) as guard-requiring, so
   it cannot be pulled into a speculative both-sides `selp`. **Any pass that
   hoists, sinks, or duplicates code across a conditional must re-check this
   predicate is still respected** — hoisting a loop-invariant load above a
   barrier, or duplicating a loop body during unrolling across a barrier, can
   silently desync warps. Barrier/divergence code must stay exactly once, at
   its original relative position, on every code path that reaches it today.
2. **Atomics** (`atomic_rmw`, `Sarek_ir_ptx_expr.ml:1020-1074`) are already
   funneled through `expr_needs_branch_guard`'s `is_atomic` check — same rule:
   never speculate, never duplicate, never reorder across another atomic or a
   barrier on the same address space.
3. **ABI / parameter layout.** `emit_params` (`Sarek_ir_ptx_kernel.ml:19-124`)
   emits the `(ptr, u32 length)` pair for every `TVec`/`TArray` parameter
   *unconditionally*, even when the body never reads the length — the comment
   at line 30-35 explains why: `Execute.expand_to_run_source_args` on the host
   side always marshals both slots, so skipping the length param would shift
   every subsequent parameter's `kernelParams` index. **No pass may drop an
   unread parameter or reorder parameters** — this is a host/device contract,
   not a local dead-code opportunity.
4. **Variant vector-element loads/stores load/store every constructor's
   slots**, not just the active one (`emit_variant_elem_load`,
   `Sarek_ir_ptx_mem.ml:250-281`; `store_variant_elem` guards stores with a
   tag-compare branch, `Sarek_ir_ptx_mem.ml:312-357`, but loads are
   unconditional for all ctors). This is deliberate (shape-uniform binding for
   leaf-wise merge, FR-013) — a "dead field" pass must not touch this without
   re-deriving the shape-uniformity invariant it protects.
5. Any CSE/hoisting pass operates on a **single straight-line block with no
   aliasing model for global memory** — the emitter has no alias analysis.
   Caching a load's value across a store to the same array (even under a
   different index) is unsound unless the pass proves non-aliasing; the safe
   default is CSE only for *pure address arithmetic* (candidate 1), never for
   the loaded *value*.

## Candidates

### 1. Address-arithmetic CSE — real, cheap, no safety risk if scoped to addresses only

**Evidence.** `emit_agg_elem_addr` (`Sarek_ir_ptx_mem.ml:191-205`) is called
fresh from `emit_record_field` (`Sarek_ir_ptx_expr.ml:647-665`) and
`emit_agg_array_read` (`Sarek_ir_ptx_expr.ml:610-624`) on **every** field
projection or aggregate-element read — there is no memo table keyed on
`(arr_name, idx_expr)`. Two field reads of the same vector element,
`p.(i).x` then `p.(i).y`, each independently emit:

```
mul.wide.u32 %rd1, %r_idx, 12;   ; p.(i).x
add.u64      %rd2, %rd_base, %rd1;
ld.global.f32 %f0, [%rd2];
mul.wide.u32 %rd3, %r_idx, 12;   ; p.(i).y  — identical to %rd1
add.u64      %rd4, %rd_base, %rd3;          ; identical to %rd2
ld.global.f32 %f1, [%rd4+4];
```
The scalar path (`emit_array_read`, `Sarek_ir_ptx_mem.ml:30-80`) has the same
shape: `cvt.u64.u32` + `shl.b64` + `add.u64` recomputed per read, even for
`a.(i) + a.(i)`. Loop-carried index math (`thread_id`/`global_thread_id`,
`Sarek_ir_ptx_expr.ml:1241-1252`) is likewise re-derived at every occurrence —
`global_thread_id` alone is 5 instructions, repeated verbatim at each call
site.

**What ptxas already recovers.** This is exactly the class of same-block,
no-store-in-between, identical-operand redundancy ptxas's SSA-based local CSE
is built for — `mul.wide.u32 %rd1, %r_idx, 12` appearing twice with the same
operands in the same basic block is close to the textbook case, and ptxas
(and ZLUDA's downstream LLVM pipeline) will typically fold it. The gain from
doing it ourselves is **register-count and PTX-file-size only**, not expected
runtime speedup on NVIDIA — but it matters for compile time on large kernels,
for readability when debugging emitted PTX, and (importantly, per candidate 7)
because *emitting* the redundant instruction still burns a virtual register
slot, feeding register-pressure even when ptxas later coalesces the
instructions away.

Across a *branch* (the two accesses are in different arms of an `if`) or
across a *loop iteration* (same index expression re-evaluated each pass),
ptxas's reach is weaker — same-value hoisting out of a loop is a real,
harder optimization that plain local CSE won't give you for free.

**Cost:** S–M. A same-block hash-consing table keyed on the *source
expression* (arr_name, idx register-producing sub-expr) for address triples
(`base`, `idx`, `stride`) would need: (a) a canonical key for idx_expr (already
have `expr` equality via structural compare, modulo re-evaluation side
effects — must be effect-free, which `EVar`/`EConst`/pure `EBinop` chains are),
(b) invalidation on writes to the same array (points-to is name-based already
via `arr_name`, so this is tractable), (c) scoped to one basic block (no loop
hoisting) to sidestep the harder invariant-motion problem.

**Where it lives:** emitter-local (a small cache table threaded through
`emit_agg_array_read`/`emit_record_field`/`emit_array_read`, keyed per
kernel-generation call, cleared at each label/branch point) — not a separate
IR pass, since the existing structure has no separate IR stage to insert one
into.

**Verdict: Medium priority.** Real but ptxas-shadowed for straight-line code;
its main value is knock-on register-pressure reduction (feeds candidate 7),
not raw throughput. Do it as part of a register-pressure initiative, not
standalone.

### 2. Vectorized memory ops (ld/st.v2/v4) — not emitted, real gain, but layout/ABI-entangled

**Evidence.** `emit_agg_elem_load`/`emit_agg_elem_store`
(`Sarek_ir_ptx_mem.ml:283-305`, `364-427`) walk a record's fields in layout
order and emit exactly **one typed `ld.global`/`st.global` per scalar leaf**
(`emit_field_load`/`emit_field_store`, lines 213-241) — there is no check for
"are these N consecutive leaves same-type and naturally aligned for
`.v2`/`.v4`" anywhere in the module; `grep` for `v2\.` / `v4\.` / `ld.global.v`
across `sarek/codegen/` returns nothing. A record `{x:f32; y:f32; z:f32;
w:f32}` (offsets 0/4/8/12, 16-byte stride) emits 4 independent
`ld.global.f32` at `[addr]`, `[addr+4]`, `[addr+8]`, `[addr+12]` instead of one
`ld.global.v4.f32 {%f0,%f1,%f2,%f3}, [addr]`.

**What ptxas already recovers.** ptxas *does* have a load/store vectorization
pass for exactly this shape on recent CUDA versions (contiguous, same-width,
naturally-aligned loads with no intervening aliasing store), and in practice
often successfully fuses adjacent scalar `ld.global.f32` into `.v2`/`.v4` when
the addresses are provably contiguous and 8/16-byte aligned. It is **not
guaranteed** — it depends on the addressing being simple enough for ptxas's
pattern matcher (a computed offset via `mul.wide.u32` + `add.u64` per field,
as we emit, is more opaque than 4 loads off one already-materialized base with
constant immediate offsets, which is exactly what our `addr_operand` shape
*is* — `[%rd+8]` etc. — so this is actually a reasonably favorable shape for
ptxas to recognize). ZLUDA's downstream translation to AMDGPU is less
predictable for this than native ptxas; AoS vectorized loads are a known
higher-variance case there.

**Expected gain if ptxas misses it:** for AoS record-vector kernels, up to
~4x fewer load instructions and better memory-controller coalescing —
plausibly a real (not just cosmetic) bandwidth win, order 1.2–2x kernel time
on memory-bound AoS-heavy kernels. If ptxas already fuses it (likely for our
exact addressing shape), the win from doing it ourselves is close to zero.

**Cost:** M–L. Requires: (1) grouping consecutive same-type leaves in
`Sarek_ir_layout`'s field layout, (2) alignment proof (the element's base
alignment plus the sub-offset must satisfy `.v2`/`.v4`'s natural-alignment
requirement — nontrivial when records nest or mix `f32`/`i32`/`f64`), (3) a
new SROA binding shape carrying a vector register group instead of N scalar
registers, which ripples into `agg_value`/`mov_binding`/`copy_binding`
(`Sarek_ir_ptx_types.ml`) since those currently assume one PTX register per
leaf.

**Verdict: Low-Medium priority.** The precondition (contiguous
same-type-and-width fields, no intervening writes) is common but not
guaranteed to hold, and the evidence strongly suggests ptxas already recovers
this for our exact addressing shape on NVIDIA. Worth a differential PTX
inspection (emit a 4-float record kernel, run `ptxas -arch=sm_86 --verbose` or
disassemble with `nvdisasm`/`cuobjdump` and check for `.v4` in SASS) before
investing — do the cheap verification experiment before the implementation.

### 3. `ld.global.nc` for read-only parameters — cheap, real, but arch-dependent gain

**Evidence.** `emit_array_read` (`Sarek_ir_ptx_mem.ml:30-80`) always emits
plain `ld.global.<ty>`; there is no `.nc` variant anywhere in the codebase
(confirmed by grep). `DParam` array metadata
(`alloc.arr_elt_types`/`alloc.arr_memspaces`, populated in `emit_params`,
`Sarek_ir_ptx_kernel.ml:19-124`) tracks element type and shared-vs-global, but
nothing tracks **whether the kernel ever writes through that pointer** — no
write-set analysis exists.

**What ptxas already recovers.** Modern ptxas (and `__ldg`/`const __restrict__`
in CUDA C) infers `.nc`-eligibility itself when it can prove no aliasing
write — but that inference is best-effort and commonly fails to fire across
even simple aliasing-unclear cases (multiple pointer params of the same
type — exactly our multi-array-kernel-param signature — defeats the
compiler's alias analysis unless `restrict` is asserted). We *can* prove this
statically and cheaply: at the IR level, a `DParam` array name that never
appears as the target of `EArrayWrite`/`SArraySet`/an atomic op anywhere in
`kern_body` (including inlined helper bodies) is provably read-only for the
whole kernel — strictly stronger and more reliable than what ptxas can infer
without `restrict` hints we don't currently emit.

**Expected gain:** on read-heavy kernels with multiple pointer parameters
(the common case where ptxas's own alias inference is weakest), routing
through the texture/read-only cache path can give a real, measurable win on
architectures where it's wired to a separate cache resource (pre-Volta
`ld.global.nc` maps to the texture cache; Volta+ it's the same L1 but with a
non-coherence hint that still helps scheduling) — order 1.1–1.3x on
bandwidth-bound kernels with repeated same-address reads, near-zero on
compute-bound or single-touch-per-element kernels.

**Cost:** S. A single write-set pass over `kern_body` (and each inlined
`hf_body`) collecting `DParam` names ever appearing as an array-write target,
then a one-line change to `emit_array_read`'s call sites to pass `~nc:true`
when the array is param-sourced and absent from the write set. No SROA/ABI
ripple — purely a load-instruction-suffix decision.

**Verdict: High priority.** Best gain-to-cost ratio of the memory-access
candidates: cheap, provably safe (read-only is a static, decidable property
here), and targets exactly the case (multi-pointer-param kernels) where
ptxas's own inference is weakest without `restrict`.

### 4. `ld.param` re-load elimination / register caching — already done, no-op

**Evidence.** `emit_params` (`Sarek_ir_ptx_kernel.ml:19-124`) loads every
parameter into a register **exactly once**, at kernel entry, and binds that
register into `env` (`env_bind env v.var_name r`); every subsequent
`EVar`/array-name reference is `env_lookup env v.var_name`
(`Sarek_ir_ptx_types.ml:177-185`), a table lookup returning the same register
name — there is no re-emission of `ld.param` anywhere else in the emitter.

**Verdict: N/A — not a real candidate.** This optimization already exists by
construction of the single-pass env-binding design. Listed here only to
close out the checklist; no work needed.

### 5. Strength reduction on index math — scalars already shift; loop induction/div-mod unaddressed

**Evidence.** `elt_shift` (`Sarek_ir_ptx_mem.ml:21-28`) already converts
scalar-array byte-offset multiplication to a compile-time-constant `shl`
(`shl.b64 %rd, %rd_idx64, 2` for f32/i32, `3` for f64/i64) — this is *already*
strength-reduced at emission time, not left as a `mul` for ptxas to fix.
Aggregate-element addressing (`emit_agg_elem_addr`,
`Sarek_ir_ptx_mem.ml:191-205`) uses `mul.wide.u32 %rd, %r_idx, stride` with a
literal `stride` operand — ptxas reliably strength-reduces `mul` by a
compile-time power-of-2 constant to a shift; for non-power-of-2 record sizes
(the common case — `stride=12` for a 3×f32 record) there is no cheaper
integer op than a multiply, so nothing to gain.

What's genuinely **absent**: `SFor` (`Sarek_ir_ptx_stmt.ml:102-123`) emits a
fresh loop-counter register (`r_loop`, incremented `add.u32`/`sub.u32` by
literal `1` each iteration) with no induction-variable strength reduction —
i.e. if the loop body computes `base + i * stride` where `i` is the loop
variable, that's re-derived as a multiply every iteration (see candidate 1)
rather than incrementally updated by `+= stride` (classic induction-variable
strength reduction). `Mod`/`Div` by block-dimension values
(`emit_binop`'s `Div`/`Mod` cases, `Sarek_ir_ptx_expr.ml:800-826`) are emitted
as plain `div.u32`/`rem.u32` — expensive on GPU ALUs — with **no**
special-casing for the extremely common pattern `idx / blockDim.x` or `idx %
blockDim.x` where blockDim is a runtime-but-uniform value; no power-of-2
runtime check, no reciprocal-multiply lowering.

**What ptxas already recovers:** compile-time-constant-divisor strength
reduction, yes (and we already do the power-of-2 array-index case ourselves).
Runtime-divisor (blockDim-based) division/modulo reciprocal tricks — no,
ptxas cannot prove blockDim is a compile-time constant, so `div.u32`/`rem.u32`
against a register operand stays as emitted; these are genuinely expensive
(20-40 cycle latency class) ops on every architecture.

**Expected gain:** loop induction-variable strength reduction: real but small
per-iteration (saves one `mul.wide` per loop pass, replaced by an `add`) —
meaningful only in tight, high-trip-count loops with aggregate-array indexing
inside; low double-digit percent on such loops, ~0 elsewhere. `div`/`rem` by
blockDim: no safe general fix without knowing blockDim is power-of-2 at
launch time, which Sarek doesn't currently expose to codegen — not
actionable without a language-level change (out of scope for an emitter
pass).

**Cost:** M for loop induction-variable strength reduction (needs to detect
"index expression is affine in the loop variable" and thread an
incrementally-updated register through the loop body — this is the classic
loop-optimization dataflow problem, non-trivial in a single-pass emitter
without a loop-body pre-scan).

**Verdict: Low priority.** The genuinely-fixable slice (induction-variable
strength reduction inside tight loops) is a real M-cost project for a
narrow win; the higher-value div/mod case needs upstream compile-time
guarantees the emitter doesn't have. Not worth prioritizing over candidates
1/3/7.

### 6. `selp` vs branch policy — mostly right; one real pessimization case

**Evidence.** `expr_needs_branch_guard` (`Sarek_ir_ptx_expr.ml:42-69`) already
draws the correct safety line: barriers/atomics/array-reads/match/aggregate
constructs force real control flow; everything else (arithmetic, casts,
comparisons) takes the eager both-branches `selp` path
(`EIf` scalar case, `Sarek_ir_ptx_expr.ml:350-374`). This is the right
default for *cheap* scalar branches (avoids a mispredict-prone `bra`/warp
divergence for a one-instruction difference).

**The gap:** the guard is purely *structural* (is this expression kind ever
unsafe to speculate), not *cost-based*. An `EIf` whose both branches are
several chained f32 transcendentals (e.g. `if c then sin(x)*cos(y) else
exp(z)+log(w)`, none of which trip `expr_needs_branch_guard` since they're
plain `EIntrinsic`/`EBinop` chains with no array reads or atomics) will
**unconditionally evaluate both arms** and `selp` the result — genuinely
wasteful when the branches are expensive and divergence-tolerant (warps often
agree on the predicate in practice, in which case a real branch would skip
the untaken arm's cost entirely).

**What ptxas already recovers:** nothing — once both arms are emitted as
straight-line code with a `selp`, ptxas has no way to know it was safe to
skip one at runtime; the cost is baked in at the PTX level, not a downstream
optimization opportunity at all. This is squarely a decision only the
emitter can make.

**Expected gain:** kernel-dependent, potentially large (2x+) on kernels with
expensive divergent branches inside math-heavy code, near-zero on today's
typical Sarek kernels (arithmetic/array-index kernels where each arm is 1-3
cheap ops, where selp genuinely is better than a branch).

**Cost:** M. Needs an instruction-count/cost heuristic on `then_e`/`else_e`
(e.g. count `EIntrinsic` calls with no native 1-op lowering, or a fixed
threshold on estimated PTX instruction count) to flip to the branch path
above some cost, and it must NOT weaken any existing branch-guard case — this
is additive to the guard, not a replacement.

**Verdict: Low priority right now** (no evidence today's kernels hit this
case commonly), but flag it as a **known gap**, not something to leave
undocumented — cite it if a future kernel profile shows expensive divergent
`if`/`else` arithmetic arms.

### 7. Register pressure (no reuse pass) — confirmed structural, highest-leverage single fix

**Evidence.** The allocator (`Sarek_ir_ptx_types.ml:91-119`) is a bare
monotonic counter per PTX type — `new_u32`/`new_u64`/`new_f32`/`new_f64`
never recycle a register once its last use has passed; there is no liveness
tracking at all. Combined with:
- **Full inlining** (`emit_app`, `Sarek_ir_ptx_expr.ml:491-537`): every helper
  call duplicates the callee body's entire register footprint at the call
  site (module docstring at lines 430-436 confirms this is deliberate —
  "PTX .func would need a per-function register frame... helpers are small
  and NVCC inlines them anyway").
- **SROA** (`copy_binding`/`bind_helper_param`,
  `Sarek_ir_ptx_expr.ml:249-263`, `442-473`): every record/variant value is
  represented as N independent scalar registers, and helper-parameter binding
  leaf-wise *copies* every non-array argument (mutation isolation,
  `Sarek_ir_ptx_expr.ml:450`) — a fresh register set per call, never shared
  with the caller's.
- A recursive-but-`[@sarek.inline N]`-pragma'd helper (the `fib` example
  documented in `sarek/ppx/README.md:198-206`) multiplies this: each of the up
  to `N` levels of unrolled recursive inlining duplicates the *entire*
  register footprint of the remaining call tree, so register count grows
  combinatorially with inline depth, not linearly with source size — this is
  the documented mechanism behind the ~3.8k-register `fib` kernel referenced
  in project memory.

**What ptxas already recovers.** ptxas's register allocator *does* do
liveness-based coalescing/allocation from a virtual-register PTX input — this
is exactly its job, and it usually produces a reasonably tight final SASS
register file even from a PTX input with thousands of named virtual
registers, **as long as it can prove short, non-overlapping live ranges**.
The risk is not "ptxas fails to allocate" — it's (a) **compile time**, which
scales worse than linearly with virtual-register count on some ptxas
versions, and (b) **occupancy**: if ptxas's allocator, faced with genuinely
long-lived aggregate SROA fields kept live across the entire inlined helper
body (because we never free/reuse the OCaml-side name once bound), concludes
the *true* live range is wide even after its own analysis, it will allocate
more physical registers per thread than a source PTX with tighter,
recycled virtual registers would have implied — directly reducing
occupancy (fewer resident warps per SM). ZLUDA's downstream path is *more*
exposed here: its PTX→AMDGPU translation historically has had less mature
register-pressure-aware scheduling than mature ptxas, so a bloated virtual
register count is more likely to survive into worse real occupancy on the
ZLUDA/AMD path than on native NVIDIA — this is the project's own stated
interest (`SPOC × ZLUDA`).

**Expected gain:** this is the one candidate where "order of magnitude" is
plausible rather than marginal — for deep-inline recursive kernels (the `fib`
class), reducing virtual-register count by even a naive linear-scan
recycling pass (free a register once its last textual use in the same block
has passed) could plausibly cut declared `.reg` counts by 2-10x on
inline-heavy kernels, with a *knock-on* occupancy improvement that's
architecture- and kernel-dependent but could be the difference between 1 and
4+ resident blocks per SM on register-limited kernels. On today's simple,
non-recursive, non-inline-heavy kernels (the common case), the gain is near
zero — ptxas already handles small virtual-register counts fine.

**Cost:** M-L. A real "linear-scan recycling of dead virtual registers" pass
needs: (1) a last-use pass over the emitted instruction stream per type-class
(scan the buffer's text, or — cleaner — instrument `emit`/register-returning
call sites to record a use-list as they're emitted), (2) a free-list per
register class the allocator consults before minting a new number, (3) care
around control-flow joins (`EIf` `selp`/branch paths, `EMatch` arm merges) —
a register live into a branch join must not be recycled inside one arm and
reused for something else if the other arm still needs the original value
live past the join. This is a nontrivial dataflow problem for a supposedly
"single-pass" emitter — likely needs a lightweight post-pass over the
per-kernel instruction buffer rather than being interleavable with emission
itself.

**Verdict: Highest priority of the register/compute-shape candidates.** Not
because today's typical Sarek kernels need it, but because it's the one
candidate with a plausible order-of-magnitude win specifically on the
recursive-inline kernel class the project already cares about (`fib`), and
because it directly benefits the ZLUDA path where downstream register
allocation is less mature than native ptxas. Should be scoped as its own
project (M-L cost, needs its own design for the CFG-join safety question)
rather than folded into a "quick wins" batch.

### 8. Launch-bounds / `.maxntid` hints — never emitted, real occupancy-tuning gain, needs a source of the bound

**Evidence.** `generate`/`make_ptx_header` (`Sarek_ir_ptx_kernel.ml:209-249`)
never emits `.maxntid`/`.maxnreg`/`.minnctapersm` anywhere; grep across
`sarek/codegen/` confirms zero occurrences. The `.entry` directive
(`Printf.bprintf out ".entry %s(\n" k.kern_name`, line 234) carries no
launch-bounds annotation at all.

**What ptxas already recovers:** nothing — `.maxntid` is precisely the
mechanism *for* telling ptxas "you may allocate more registers per thread
because you know at most N threads will run this block," which lets ptxas
trade register count for occupancy in the direction the programmer knows is
safe; without it, ptxas must assume the architectural max thread-block size,
which is frequently far more conservative than the kernel's actual launch
config, causing ptxas to either allocate fewer registers than it safely could
(leaving performance on the table) or spill unnecessarily on register-heavy
kernels.

**Expected gain:** kernel- and launch-config-dependent, but this is a
well-documented, real lever in CUDA practice — commonly cited gains are in
the 10-30% range on register-pressure-bound kernels when the launch bound
matches actual usage, occasionally more on kernels near an occupancy cliff
(e.g. going from 1 to 2 resident blocks per SM). Zero gain on kernels that
are not register-bound (memory-bound or occupancy-saturated kernels).

**Cost:** S if the launch configuration (block dimensions) is known at
codegen time from the Sarek kernel's launch parameters; the emitter's
`generate` function currently takes only `sm_target` as an override, with no
plumbing for a block-size hint. Would need: (1) a way for the caller
(`Execute`/kernel-launch driver) to pass an expected/maximum block size into
`generate`, (2) emit `.maxntid <x>, <y>, <z>` in the `.entry` header when
known. If block size is only known at launch time (post-compile, dynamic),
this candidate is not applicable without a JIT-recompile step — verify
against how Sarek kernels are actually launched before committing to this.

**Verdict: Medium priority, gated on a scoping question** (is block size
known at PTX-generation time in the current pipeline, or only at launch —
this needs a quick check of the `Execute` launch path before estimating cost
further). If known, this is a cheap, real, well-understood win; if not, it's
out of scope for the emitter alone.

### 9. Loop unrolling (literal-bound `SFor`) + loop-invariant hoisting

**Evidence.** `SFor` (`Sarek_ir_ptx_stmt.ml:102-123`) always emits the general
header/body/increment/branch-back shape regardless of whether `start_e`/
`stop_e` are literal constants — no special-case for
`SFor (v, EConst _, EConst _, _, body)` that could fully or partially unroll.
Nothing in the loop-emission path hoists a loop-invariant sub-expression
(e.g. an aggregate-element base-address computation that doesn't depend on
the loop variable) out of the header/body — every iteration re-evaluates
whatever the body's expression tree says to evaluate, including anything
provably loop-invariant.

**What ptxas already recovers:** unrolling small fixed-trip-count loops and
some invariant hoisting are both classic optimizations ptxas *can* apply, and
for genuinely small literal-bounded loops (say trip count ≤ 4-8) it often
does. Its reach drops for loops with non-trivial bodies (aggregate accesses,
inlined helper calls) since the pattern-matching gets harder as the loop body
grows — the same "our addressing is a few instructions of computed offset,
not a simple indexed load" issue as candidate 2.

**Expected gain:** for small literal-trip-count loops with real per-iteration
overhead (branch/predicate-test cost relative to a tiny body), IR-level
unrolling could meaningfully help — but the ratio of branch overhead to work
is exactly where ptxas's own unrolling threshold is tuned to fire, so the
marginal value of doing it ourselves for *typical* small counts is likely
small. For invariant hoisting specifically (e.g. an array's base pointer add
that doesn't depend on the loop variable — though note the base itself is
already just a register carried in `env`, so there is nothing to hoist there
today; only *address* computation building on the loop variable is the actual
candidate, which is really candidate 1/5's induction-variable case, not a
distinct hoisting problem), most of the real opportunity collapses into
candidates 1 and 5.

**Cost:** M-L for unrolling (needs to duplicate the body N times with
per-copy loop-variable substitution — nontrivial in a text-emitting,
non-IR-rewriting emitter, since "duplicate this body" means re-running
`emit_stmt` N times with a different bound register per copy, which is
actually straightforward given the emitter's structure — but correctness
requires care that `SReturn`/labels inside the body get fresh labels per
unrolled copy, which `new_label` already guarantees since it's called fresh
each `emit_stmt` invocation).

**Verdict: Low priority as a standalone candidate** — most of its real value
is already covered by candidate 1 (address CSE) and candidate 5 (induction
strength reduction) once those exist; unrolling on top would mostly reduce
branch-instruction count, a secondary effect. Don't build a separate
unrolling pass before those land; re-evaluate after.

### 10. Barrier/divergence-aware optimizations — see Safety section

Already covered above as a *constraint*, not a positive-gain candidate: there
is no barrier-adjacent optimization currently missing that's worth adding —
the emitter's current behavior (unconditional emission at statement position,
never inside a `selp` speculative path) is already the safe, correct
baseline. The actionable item here is documentation/enforcement (done above),
not a code change.

### 11. Shared-memory bank-conflict padding — not implemented, real but narrow

**Evidence.** `DShared` (`Sarek_ir_ptx_kernel.ml:149-186`) declares
`.shared .align <n> .<btype> <name>[<size>]` directly from the source-level
size — no stride-padding logic (e.g. bumping a `[32][32]` f32 tile to
`[32][33]` to break the classic 32-way bank conflict on strided
column-access) exists anywhere.

**What ptxas already recovers:** nothing — padding a shared array's
declared size to avoid bank conflicts is a semantic-changing layout decision
(it changes the array's total byte size and every index's physical offset)
that only the emitter (or the source-level programmer) can make; ptxas
operates on the array as declared and cannot infer that padding would help
without knowing the *access pattern*, which requires whole-kernel dataflow
analysis (which indices are ever used, is the access strided by a
bank-conflicting stride) that PTX-level tools generally don't attempt.

**Expected gain:** real and sometimes large (2-32x on the affected shared-load
instructions specifically) **but only for a narrow, specific access pattern**
(strided access across the padded dimension that hits the classic 32-bank
conflict) — most Sarek shared-memory kernels may not have this shape at all;
this needs an access-pattern check per shared array (is it indexed with a
stride that's a multiple of the bank count) before deciding to pad, which
itself requires a static analysis of every index expression touching that
array — nontrivial for a single-pass emitter with no separate analysis stage.

**Cost:** M-L for a general version (access-pattern detection); S for a
narrow blessed case (e.g. always pad 2D `[N][32]`-shaped shared tiles by 1
column) if the language exposes shared arrays with known 2D shape at all —
worth checking whether Sarek's shared-array declarations are 1D-only today
(`DShared` here takes a flat `size_opt : expr option`, no 2D shape), in which
case this candidate may not even apply to the current language surface.

**Verdict: Low priority / possibly not applicable.** `DShared` is 1D as
currently modeled (`Sarek_ir_ptx_kernel.ml:149-169`) — the classic 2D-tile
bank-conflict scenario this optimization targets may not be expressible in
today's Sarek surface at all. Verify against the source language (does Sarek
support 2D shared tiles) before spending any design time here.

### 12. Constant folding/propagation — NOT done anywhere upstream; real, cheap, and currently just missing

**Evidence.** Checked both ends: the emitter (`Sarek_ir_ptx_expr.ml:242-269`)
emits a `mov` for every `EConst` leaf with no folding of `EBinop` on two
`EConst` operands — `emit_binop` (`Sarek_ir_ptx_expr.ml:740-919`) always reads
both operands via `emit_expr` (which for `EConst` always emits a `mov`) and
always emits the arithmetic instruction, regardless of whether both operands
are literals. The lowering stage (`sarek/ppx/Sarek_lower_ir.ml:427`,
`TEBinop (op, a, b) -> Ir.EBinop (ir_binop op te.ty, lower_expr state a, lower_expr state b)`)
does the same — no constant-folding special case; grep for
`fold_const`/`constant_fold`/`ConstFold` across `sarek/` returns nothing.
Concretely, a source expression like `n * 4 + 1` where the multiplier and
addend are literals emits (today, always): `mov.u32 %r0, 4; mov.u32 %r1, 1;`
plus a full `mul.lo.u32`/`add.u32` chain against the *variable* operand only
where genuinely needed — but `2 + 3` (both literal) still emits two `mov`s
and an `add`, never folded to a single `mov.u32 %r, 5`.

**What ptxas already recovers.** This is the single case in this whole
document where ptxas's recovery is closest to complete — constant folding on
two `mov`-then-op instructions with literal immediate operands in the same
block is about as easy as SSA optimization gets, and ptxas reliably folds it.
**Practically zero runtime gain expected from fixing this ourselves.**

**Expected gain:** near-zero for runtime; small for PTX-file-size/readability
and (same knock-on argument as candidate 1) a marginal reduction in emitted
virtual-register count feeding candidate 7 (each folded constant removes 1-2
registers that would otherwise need allocating and — however briefly —
tracking).

**Cost:** S — a `fold_const : expr -> expr` pre-pass over the IR (or inline in
`emit_binop`'s dispatch, matching `EConst _, EConst _` before falling through
to the general case) is a small, mechanical, easily-tested addition (pure
function, no state, trivially unit-testable against the existing `EConst`
variants for int32/int64/float32/float64/bool).

**Verdict: Low priority for runtime, but cheapest possible win for PTX
hygiene.** Good candidate for a "do it because it's free and makes debug PTX
readable" pass, not because it moves any performance number. Sequence it
opportunistically (e.g. bundled with whichever other pass touches
`emit_binop`/lowering next), not as a scheduled priority item on its own.

## Priority matrix (gain × cost)

| # | Pass | Real gain beyond ptxas/ZLUDA | Cost | Priority |
|---|------|-------------------------------|------|----------|
| 3 | `ld.global.nc` for read-only params | Real, arch-dependent (1.1-1.3x on multi-pointer-param kernels) | S | **High** |
| 7 | Register-reuse pass (recycle dead virtual regs) | Potentially order-of-magnitude on inline-heavy/recursive kernels; benefits ZLUDA occupancy specifically | M-L | **High** (own project) |
| 8 | `.maxntid` launch-bounds hint | Real 10-30% on register-bound kernels, IF block size known at codegen time | S (gated on a scoping check) | Medium |
| 1 | Address-arithmetic CSE | Mostly ptxas-shadowed in-block; feeds #7's register-pressure reduction | S-M | Medium |
| 2 | Vectorized `ld/st.v2/v4` | Real if ptxas misses it (likely already recovers our exact addressing shape — verify first) | M-L | Low-Medium (verify before building) |
| 12 | Constant folding | Near-zero runtime (ptxas already does this); free PTX hygiene | S | Low (opportunistic) |
| 5 | Induction-variable strength reduction | Real but small, narrow (tight loops w/ aggregate indexing) | M | Low |
| 6 | Cost-based `selp`-vs-branch | Real only for expensive divergent arms; not evidenced in current kernels | M | Low (documented gap, not scheduled) |
| 9 | Loop unrolling (literal bounds) | Mostly subsumed by #1/#5 once those exist; ptxas already unrolls small loops | M-L | Low |
| 11 | Shared-memory bank-conflict padding | Large but narrow (2D strided tiles); may not apply — `DShared` is 1D today | M-L / possibly N/A | Low (verify language surface first) |
| 4 | `ld.param` re-load elimination | N/A — already done by construction | — | N/A |
| 10 | Barrier/divergence optimization | N/A — current behavior is already the safe baseline | — | N/A (safety constraint, not a gain candidate) |

## Top-3 recommendation, with reasoning

1. **`ld.global.nc` for read-only params (#3).** Best gain/cost ratio in the
   set: a small, provably-safe static write-set check unlocks a real
   cache-path win precisely where ptxas's own alias inference is weakest
   (multi-pointer-param kernels, which is the norm for Sarek kernels reading
   several input arrays).
2. **Register-reuse pass (#7).** Not cheap, but it's the only candidate with
   plausible order-of-magnitude upside, targets a kernel class the project
   already tracks (`fib`, deep `[@sarek.inline]`), and compounds with the
   ZLUDA interest — AMDGPU-path register allocation is less mature than
   native ptxas, so a bloated virtual-register count is more likely to
   survive into real occupancy loss there than on NVIDIA. Scope it as its
   own M-L project with an explicit design for the branch/match-join
   liveness-safety question before writing code.
3. **`.maxntid` launch-bounds hint (#8)**, *conditional on a quick scoping
   check*: confirm whether block/launch dimensions are available at
   PTX-generation time in the current `Execute` pipeline. If yes, this is a
   cheap, well-understood, real win and should be pulled ahead of #7 in
   sequencing (much lower cost for a comparable gain class on register-bound
   kernels). If block size truly is only known at launch time with no
   PTX-generation-time hook, drop it from the near-term queue.

Everything else in the list is either already-done (#4), a safety constraint
rather than a gain (#10), speculative pending a cheap verification experiment
before committing implementation cost (#2, #11), or real-but-marginal enough
that it should ride along with other work rather than being scheduled on its
own (#1, #12, #5, #6, #9).
