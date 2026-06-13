# SPOC Formal Verification & PTX Backend — Master Roadmap
# Last updated: 2026-06-13 (session 2)
# Status: LIVE (update this file as work progresses)
# Git: intentionally untracked — local tracking only

---

## Active work streams

### Stream 1 — convergence-safety T3-SEMANTIC (branch: formal/convergence-safety-phase1a)

**Goal**: semantic soundness — prove check_expr is sound w.r.t. a fuel-indexed SIMT evaluator.
**Apparatus**: formal-apparatus v1.2.1 installed.

| Task | Status | Notes |
|------|--------|-------|
| T3-GATE | DONE | Full ladder T3-S1..S8 approved 2026-06-13 |
| T3-S1 Semantic domain + big-step evaluator | **DONE** | commit fbfb3656; ConvergenceSemantics.v: 22 theorems, 0 admits, 0 axioms, coqchk passes |
| T3-S2 Uniformity soundness of is_varying_in_env | IN FLIGHT | workflow wf_315cd2a1 (Opus) running; in-progress changes in working tree |
| T3-S3 Trace silence of barrier-free exprs | PENDING | Blocked by T3-S1 ✓ |
| T3-S4 Core semantic soundness of check_env | PENDING | L effort; blocked by T3-S2+S3 |
| T3-S5 EReturn residual-divergence (F-04) | PENDING | Blocked by T3-S4 |
| T3-S6 ESuperstep semantic grounding | PENDING | Blocked by T3-S4 |
| T3-S7 Warp-collective semantic soundness | PENDING | Blocked by T3-S4 |
| T3-S8 Extraction + differential conformance | PENDING | Blocked by T3-S5; closes T3-SEMANTIC on pass |

**After T3-SEMANTIC**: merge PR, start type-safety formal project (no external semantic foundations needed; natural next step).

---

### Stream 2 — PTX backend for Sarek

**Goal**: direct Sarek IR → PTX emission, bypassing NVCC. Drops NVCC from the TCB. Enables a tractable formal verification story (Sarek IR → PTX, provably correct).

**Why PTX over CUDA C**: NVCC is an unverifiable black box; PTX is a published virtual ISA; bar.sync maps 1-to-1 to abstract EBarrier.

**Phase A — Spike**: **DONE AND VALIDATED** (2026-06-13)
- `sarek/codegen/Sarek_ir_ptx.ml` — PTX emitter from Sarek IR (commit 1da95861)
- `sarek-cuda/test/test_ptx_external.ml` — end-to-end test: load PTX via cuModuleLoadData, verify vector_add results
- Validated by ptxas (static) and on real NVIDIA hardware (GTX 1070, sm_61): **0.186s, correct results**
- `Cuda_api.Kernel.load_from_ptx` auto-adapts .target to device SM — PTX built for sm_86 loads on any SM ≥ 6.1 (commit f6c14c2a)
- Generator parameterized: `Sarek_ir_ptx.generate ?sm_target` (default sm_86)

**Phase B — Full backend implementation** (open):
- Cover full Sarek IR (records, variants, helper functions, shared memory)
- ptxas validation in CI
- Status: PLANNED

**Phase C — PTX backend + formal spec co-design** (open):
- Shape implementation to be provably correct
- Status: PLANNED (after cuda-semantics spec locked)

**Key insight**: Phase A changes the codegen-cuda verification timeline. Proving `Sarek_ir_ptx.ml` correct (OCaml function we own) is far more tractable than reasoning about NVRTC (blackbox). The verification target is now concrete.

---

### Stream 3 — cuda-semantics formal project (spec derisking phase)

**Goal**: Rocq formal semantics for the PTX subset Sarek emits. Enables proving PTX backend correctness.

**Key derisking findings (2026-06-13)**:
- No weak memory model needed — Sarek only uses __syncthreads; collapses to one axiom: CTA-scope acquire-release barrier-visibility
- First theorem: `accepted_block_no_deadlock` — checker accepts ⇒ SIMT block never deadlocks
- Source semantics: ConvergenceSemantics.v already written (T3-S1) — reuse it
- Architecture: sibling `formal/cuda-semantics/` depending on `formal/convergence-safety/`
- Best prior art: GPUVerify TOPLAS SDV semantics; Lustig'19 for barrier axiom

**Status**: BLOCKED on convergence-safety T3-SEMANTIC completing. Resume after T3-S8.

**Phases**:
| Phase | Status | Notes |
|-------|--------|-------|
| Spec plan (PTX-focused) | READY | cuda-semantics-spec-2026-06-13.md |
| PTX spike | DONE | Sarek_ir_ptx.ml validated on real hardware |
| convergence-safety T3-S1 | DONE | ConvergenceSemantics.v = source semantics |
| GPUVerify TOPLAS deadlock def | PENDING (manual) | FlateDecode issue; read PDF manually |
| /formal-init cycle | PENDING | After T3-SEMANTIC done + TOPLAS read |
| First theorem: accepted_block_no_deadlock | PLANNED | |
| Full formal project (T0-T3) | PLANNED | |

---

## Future streams (not in scope yet)

| Project | Priority | Prereqs | Notes |
|---------|----------|---------|-------|
| type-safety of Sarek PPX | **HIGH** | convergence-safety T3 done | No external semantic foundations; natural next step after T3-S8 |
| cuda-semantics: PTX emitter correctness | **HIGH** | cuda-semantics formal project + PTX Phase B | Sarek_ir_ptx.ml is the verification target |
| sarek-opencl formal verification | MEDIUM | cuda-semantics methodology | Same shape, different target ISA |
| sarek-vulkan / sarek-metal | MEDIUM | Above | |
| cuda-semantics: memory model proof (not axiom) | LOW | cuda-semantics formal project | Certifies CUDA not Sarek; separate project |
| transpiler-fidelity (OCaml → PTX semantics preservation) | LOW | cuda-semantics + type-safety | Full compiler correctness; far future |
| WGSL backend formal verification | SKIP | — | User: low priority |

---

## Open decisions

| Decision | Options | Status |
|----------|---------|--------|
| cuda-semantics: extend convergence-safety/ or sibling? | Sibling (recommended) | PENDING |
| PTX backend: new opam package sarek-ptx/ or extend sarek-cuda/? | Extend sarek-cuda (spike landed there) | LEANING toward keeping in sarek-cuda; revisit at Phase B |

---

## How to continue in a new session

1. Read this file first
2. `git log --oneline -5` on `formal/convergence-safety-phase1a`
3. Check `formal/convergence-safety/STATUS.md` for current proof count
4. Workflow `formal/convergence-safety/formal-verif-autopilot.workflow.js` handles T3 ticks autonomously (model: opus)
5. PTX spike complete — next PTX action is Phase B (full IR coverage) or starting cuda-semantics formal project
