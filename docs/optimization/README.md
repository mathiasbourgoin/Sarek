# Sarek/SPOC Optimization Roadmap

_Evaluation document — 2026-07-23. Nothing here is scheduled for immediate implementation;
each item is sized and prioritized for future cycles. Goal: performance and expressivity
better than or comparable to pure CUDA, OpenCL, and Triton._

Three detailed volumes, each grounded in the current code (file:line citations throughout):

| Volume | Scope |
|---|---|
| [opt-ptx-passes.md](opt-ptx-passes.md) | Optimization passes for the PTX emitter (12 candidates + safety constraints) |
| [opt-spoc-runtime.md](opt-spoc-runtime.md) | SPOC runtime: SoA↔AoS, transfers, fusion, allocation |
| [opt-expressivity-gaps.md](opt-expressivity-gaps.md) | Feature gaps vs CUDA/OpenCL/Triton + where Sarek is already ahead |

## Combined priority shortlist (gain × cost)

### Tier 1 — high gain, bounded cost

1. **SoA layout for custom vectors** (runtime, M) — the headline. AoS `point3d vector` is the
   textbook 1/3-efficiency uncoalesced case; SoA makes device addressing *simpler* (independent
   scalar streams). Opt-in per-vector, flat records first; AoS stays for whole-record-copy
   workloads. Prerequisite thinking done; kernel-arity plumbing is the real cost.
2. **Finish warp-primitive codegen** (emitter, M) — shuffle/vote/ballot are already modeled in
   `Sarek_core_primitives` with convergence analysis and tests; **zero backends emit them**.
   Unlocks fast reductions/scans; subgroup equivalents exist on Vulkan/OpenCL/Metal.
3. **Pinned host memory** (runtime, S-M) — transfers are pageable-only today; ~2× transfer
   bandwidth, and the hard prerequisite for async-transfer overlap.
4. **`ld.global.nc` for read-only params** (emitter, S) — static write-set analysis per DParam;
   targets exactly where downstream JIT alias inference is weakest.

### Tier 2 — real but sequenced

5. **Async transfers + compute/copy overlap** (runtime, M) — stream machinery and kernarg
   retention (PR #221) generalize cleanly; sequence after pinned memory.
6. **Register-reuse pass** (emitter, M-L) — the only order-of-magnitude candidate (inlining +
   SROA explode virtual regs; fib = 3.8k). Benefits ZLUDA specifically (less mature downstream
   allocation than ptxas). Needs a liveness design across branch/match joins — own project.
7. **Kernel `printf`/`assert`** (DX, M) — PTX `vprintf` + OpenCL builtin; the single biggest
   debugging-experience gap vs CUDA.
8. **Block-level reduce/scan stdlib** (library, S once warp prims land) — extract from existing
   e2e kernels rather than invent.
9. **`.maxntid` launch-bounds hints** (emitter, S) — gated on block size being known at PTX
   generation time (verify in Execute pipeline first).

### Tier 3 — evaluate-later / conditional

10. **Multi-dim strided views over `Vector.t`** (library-only) — closes the day-to-day Triton
    convenience gap without IR changes. The full Triton tile model is a *different compiler*
    (layout/coalescing passes) — out of scope as architected; documented honestly.
11. **f16/bf16** — ML-facing; cross-backend story exists (fp16 extensions) but scope is real.
12. **Vectorized `ld.global.v2/v4`** — needs a cheap disassembly experiment first to confirm
    ZLUDA/ptxas don't already fuse adjacent loads.
13. **Address-arithmetic CSE / constant folding** — largely shadowed by the downstream JIT for
    straight-line code; value is register-pressure knock-on only. Fold into item 6 if pursued.
14. **Occupancy auto-config** — 108 call sites hand-specify block sizes with no evidence of
    pain; revisit when auto-tuning has a user.

## Already better than CUDA (keep and advertise)

- Algebraic data types (records/variants) + `match` **on the GPU**, with a formally proved
  layout (PtxLayout.v) — CUDA has no equivalent.
- Real parametric polymorphism via OCaml — no template bloat, no SFINAE.
- Static convergence checking — divergent warp-collective misuse is a compile error, not a hang.
- Six backends from one kernel source; per-device JIT specialization.
- Metaprogramming via PPX/functors where CUDA needs template metaprogramming.

## Standing safety constraints (any pass must preserve)

Convergence of `bar.sync` (no code motion across barriers), atomics ordering, the host ABI
byte layout (PtxLayout.v-frozen), and the width-safety invariant (no instruction suffix may
disagree with its operand register class). See the safety section of opt-ptx-passes.md.

## Interactions with the active limits campaign

SoA (item 1) reuses `Sarek_ir_layout` leaf enumeration; the aligned-ABI migration (campaign L8,
GO-M) should land **before** SoA to avoid re-doing layout math; warp primitives (item 2) are
the foundation for stdlib reduce/scan (item 8); Df64/real64 (campaign L11) benefits directly
from fma-correctness guarantees documented in the safety section.
