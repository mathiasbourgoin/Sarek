# Sarek_worklist — portable dynamic parallelism (L16)

Date: 2026-07-23. Branch: `feat/sarek-worklist`. Design source:
`roster/ptx-limits-campaign/L16-dynamic-parallelism.md` (verdict: GO Route B
`Sarek_worklist`, NO-GO native CDP).

## What this delivers
A pure-Sarek work-queue library serving the irregular-work / frontier /
tree-recursion use cases that CUDA dynamic parallelism (CDP) targets, on all
runnable backends, with no per-backend launch mechanism (CDP is absent on
ZLUDA/Metal/WGSL and unreliable on OpenCL 2.0 device-enqueue — see the doc §2).

## Key empirical findings (this environment)
- Global atomics available to kernels: `atomic_add_global_int32`,
  `atomic_inc_global_int32` (Sarek_stdlib/Gpu). **No global CAS/exch** — so the
  queue uses monotonic atomic-add ticket counters, not CAS-retry.
- Fences: `memory_fence_block` / `memory_fence_device`.
- **Atomics cannot be encapsulated in `[@sarek.module]` helpers**: the OCaml
  shim pins the vector to `int32 array`, which mismatches the kernel's vector
  object type (`< … underlying: Vector.t >`). The PPX only reconciles vector
  params through `.()`/`.()<-` rewriting, not through intrinsic calls. So
  push/claim atomics live **inline in the user `[%kernel]`** (doc-sanctioned:
  "provide it as a documented pattern + helpers, not a magic wrapper"). Pure
  non-vector helpers (`wl_ring_index`, `wl_has_work`) remain callable.
- **Persistent-spin worklists deadlock on sequential/pool-bounded executors**:
  the interpreter runs blocks sequentially (or across a fixed domain pool), so a
  kernel that spin-waits across blocks hangs. Only real GPUs with co-resident
  occupancy make forward progress. This is an execution-model reality, not a
  missing primitive.
- Interpreter had **no atomic support** (histogram filtered it out). Added
  trivial single-threaded RMW for the two global atomics + no-op fences
  (mutex-guarded for the parallel interpreter).

## Two portable patterns (both proven on all backends)
1. **Level-synchronous frontier** (headline, multi-thread, all backends incl.
   sequential interpreter): each launch, threads independently claim tickets
   from a fixed `[head, snapshot_tail)` window via the shared atomic HEAD
   counter (**no thread ever waits on another**), process, and push children to
   TAIL for the next level. Host relaunches until drained. Spec-safe: no
   forward-progress assumption. Serves BFS/graph-frontier / variable-fanout
   trees. Memory ordering: the host sync between levels is the barrier.
2. **Persistent single-launch** (documented; safe for single-thread everywhere,
   and multi-thread on GPUs under co-residency): one launch, each thread loops
   pop→work→push until `head >= tail`. Used at grid=1 for the ring-wrap stress
   (single thread → no cross-thread spin → runs on every backend).

## Termination protocol + assumptions
- Level-sync: host-driven. Snapshot TAIL at level start; kernel claims up to the
  snapshot; a level that produces no new pushes (TAIL unchanged, HEAD caught up)
  is the last. Correct under: **capacity ≥ total items enqueued** (monotonic
  ring, no reuse) for the main demo; an OVERFLOW flag signals under-sizing.
- Persistent: a thread that claims `h >= tail` (empty) stops. For multi-thread
  GPU use, add an OUTSTANDING counter (inc on push, dec after a work item's
  pushes complete); quiescence at OUTSTANDING=0 is stable. Ring reuse (wrap) is
  safe only when **capacity ≥ peak simultaneously-live items** — documented.

## Control-vector layout (int32 vector, length 4)
`[0]=HEAD  [1]=TAIL  [2]=OUTSTANDING  [3]=OVERFLOW`; plus a `slots` ring vector.

## CDP-vs-worklist rationale (from the doc)
CDP is frequently slower than a well-written worklist for its own headline
workload (many small irregular tasks), and exists cleanly on only 1 of 6
backends. The worklist serves the same *use cases* portably. Route B's one real
gap: sub-problems needing a genuinely different launch configuration (narrow,
out of scope for FP32/FP64 numeric kernels).

## Test matrix (all verified vs CPU reference)
- BFS variable-fanout tree frontier sum (level-sync) — every device.
- Re-pop stress: more items than threads (claim-loop re-claims).
- Push permutation + overflow-flag unit check.
- Ring wrap-around: persistent single-thread linked-list, capacity ≪ N.
