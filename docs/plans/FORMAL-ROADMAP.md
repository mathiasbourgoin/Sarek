# SPOC Formal Verification & PTX Backend — Master Roadmap
# Last updated: 2026-07-30 (reconciled against the tree; see the note below)
# Status: LIVE (update this file as work progresses)
# Git: TRACKED, at docs/plans/FORMAL-ROADMAP.md

> **Reconciliation note, 2026-07-30.** This file had not been touched since
> 2026-06-13 and had drifted in the *understating* direction: work that shipped
> was still listed as PENDING / PLANNED / "not in scope yet". Corrections are
> made **in place**, next to the claim they replace, rather than appended — a
> stale line left standing above a changelog entry is still read as current.
>
> Two header lines were themselves wrong. "Last updated: 2026-06-13" was
> accurate but ~6½ weeks old. "Git: intentionally untracked — local tracking
> only" was simply false: this file is tracked and has been committed
> (`git log -- docs/plans/FORMAL-ROADMAP.md` shows `18248b7e docs(ptx): README
> experimental section + formal roadmap update`). A file that says it is
> untracked while being tracked invites edits nobody expects to publish.

---

## Active work streams

### Stream 1 — convergence-safety T3-SEMANTIC — **CLOSED**

**Goal**: semantic soundness — prove check_expr is sound w.r.t. a fuel-indexed SIMT evaluator.
**Apparatus**: formal-apparatus v1.2.1 installed.

**The whole T3-S1..S8 ladder is done and on `main`.** The table below is kept as
the record of the ladder, with every row's real status — not as a to-do list.
`formal/convergence-safety/proof-ledger.json` reports **111 theorems, 0
declared axioms** across `ConvergenceSpec` (28) and `ConvergenceSemantics`
(83), and `formal/convergence-safety/STATUS.md` records `coqchk` passing
("Modules were successfully checked") with conformance 18/18, extraction 7/7,
semantics 4/4.

| Task | Status | Notes |
|------|--------|-------|
| T3-GATE | DONE | Full ladder T3-S1..S8 approved 2026-06-13 |
| T3-S1 Semantic domain + big-step evaluator | DONE | `formal/convergence-safety/theories/ConvergenceSemantics.v`. The commit this row used to cite, `fbfb3656`, is in zero remote branches — unresolvable in a fresh clone. On `main` the file arrives with `21d67ec9` (the T3-S2 commit) |
| T3-S2 Uniformity soundness of is_varying_in_env | **DONE** | was "IN FLIGHT (workflow wf_315cd2a1 running)"; `eval_check_uniform`, `check_env_nonvarying_uniform_*`, `is_strongly_uniform_*` are all in the ledger |
| T3-S3 Trace silence of barrier-free exprs | **DONE** | was PENDING; `eval_concrete_barrier_free_silent`, `barrier_free_no_barriers`, `eval_seq_no_barrier` |
| T3-S4 Core semantic soundness of check_env | **DONE** | was PENDING; `check_env_sound_core` + `core_frag` bridges |
| T3-S5 EReturn residual-divergence (F-04) | **DONE** | was PENDING; `hazard_spec_detects` + `hazard_checker_sound` closed F-04 (the earlier `hazard_checker_blind` was the counterexample) |
| T3-S6 ESuperstep semantic grounding | **DONE** | was PENDING; `check_env_sound_superstep`, `semantic_f01_*` |
| T3-S7 Warp-collective semantic soundness | **DONE** | was PENDING; `check_warp_sound_core`, warp-size closed via abstract `warp_of` |
| T3-S8 Extraction + differential conformance | **DONE** | was PENDING; `eval_concrete` extracted, `eval_concrete_fuel_monotone`, coqchk passes |

**"After T3-SEMANTIC: start type-safety formal project" — done.** That project
exists at `formal/type-safety/`, is itself at T3-SEMANTIC MILESTONE LOCKED
(T3-S1..S8, 2026-06-15), and reports **90 theorems, 0 admits, 0 axioms**. See
the Future-streams table below, where its row is now marked shipped rather than
"not in scope yet".

---

### Stream 2 — PTX backend for Sarek

**Goal**: direct Sarek IR → PTX emission, bypassing NVCC. Drops NVCC from the TCB. Enables a tractable formal verification story (Sarek IR → PTX, provably correct).

**Why PTX over CUDA C**: NVCC is an unverifiable black box; PTX is a published virtual ISA; bar.sync maps 1-to-1 to abstract EBarrier.

**Phase A — Spike**: **DONE AND VALIDATED** (2026-06-13)
- `sarek/codegen/Sarek_ir_ptx.ml` — PTX emitter from Sarek IR. (This row used to
  cite commit `1da95861`, which is in zero remote branches and so unresolvable
  in a fresh clone; the spike lands on `main` as `6cbb0a21` "feat(ptx-spike):
  PTX emitter, external kernel loading, and end-to-end test".) The file is now a
  thin re-export façade over `Sarek_ir_ptx_{types,mem,expr,stmt,kernel}`
- `sarek-cuda/test/test_ptx_external.ml` — end-to-end test: load PTX via cuModuleLoadData, verify vector_add results
- Validated by ptxas (static) and on real NVIDIA hardware (GTX 1070, sm_61): **0.186s, correct results**
- `Cuda_api.Kernel.load_from_ptx` auto-adapts `.target` to the device SM:
  `Cuda_api.with_sm_target` rewrites every `.target` line to the device's own
  `sm_<major><minor>`, unconditionally. So PTX built for sm_86 loads on an older
  device *provided the body uses no instruction newer than that device* — the
  rewrite makes the directive agree, it does not translate instructions. sm_61
  is the version actually validated (above). (This row used to cite commit
  `f6c14c2a`, unresolvable in a fresh clone, and to claim "any SM ≥ 6.1"; no
  lower bound is enforced in the code, so the bound is dropped rather than
  restated.)
- Generator parameterized: `Sarek_ir_ptx.generate ?sm_target` (default `sm_86`)
  — `Sarek_ir_ptx.generate` re-exports `Sarek_ir_ptx_kernel.generate`, whose
  `.mli` carries the optional argument

**Phase B — Full backend implementation**: **DONE**, and it is the shipping path,
not a branch. Both bullets that were "PLANNED" here have landed:

- *Full Sarek IR coverage* — records and variants with the aligned C-ABI layout,
  `match` with static tag erasure, helper functions via `EApp` inlining, static
  **and** dynamic shared memory (module-scope `.extern .shared`, size supplied
  at launch through `~shared_mem`), per-thread `.local` arrays
  (`create_array n Local`), the full atomics family, an f64 softmath library,
  and SoA lowering of record-typed vector parameters. `Cuda_ptx_plugin`
  (framework name `CUDA/PTX`) is the CUDA backend's **default** device path;
  `Cuda_c_plugin` (`CUDA/C`, NVRTC) is the alternative.
- *ptxas validation in CI* — **shipped, and it cannot silently self-skip.**
  `sarek/tests/unit/test_ptx_intrinsic_sweep.ml` builds one kernel per
  registered intrinsic name and width, enumerated from the emitter's own
  `Sarek_ir_ptx_expr.intrinsic_registry` (83 names today) rather than from a
  hand-picked list, and assembles each with `ptxas`; adding an intrinsic without
  a recipe fails `test_every_name_has_a_recipe`.
  `sarek/tests/unit/test_ptx_snapshot.ml` assembles regression and SoA kernels
  on top of its text assertions. `ci/Dockerfile` installs CUDA 12.6
  `ptxas`/`nvdisasm`/NVRTC + headers, and `ci/assert-toolchain.sh` runs before
  any test step and **fails the job** if `ptxas` is absent or cannot assemble a
  probe module — because the gates keep their skip logic (correct on a developer
  machine) and an image without CUDA once made the whole suite green having
  validated nothing.

  What is still *not* in CI is GPU **execution**: no runner has a device, so
  device e2e tests skip there. Device evidence comes from an RX 7900 XTX under
  ZLUDA and a GTX 1070, by hand.

**Phase C — PTX backend + formal spec co-design** (open):
- Shape implementation to be provably correct
- Status: PARTLY DONE, and asymmetric — see `formal/codegen-ptx/STATUS.md`. The
  aggregate byte layout half is co-designed and *extracted*: `PtxLayout.v` is
  extracted to OCaml by `LayoutExtract.v` and the conformance suite runs the
  theory's own definitions against `Sarek_ir_layout`, with CI byte-comparing a
  from-scratch re-extraction. The expression/statement/kernel half is still
  conformance-tested against a hand-written OCaml mirror, and the five files
  `Extract.v` produces do not compile and are linked by nothing. Closing that
  asymmetry — not "starting Phase C" — is the real next step here.

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

**Status**: **NOT STARTED — and no longer blocked.** There is no
`formal/cuda-semantics/` directory; `formal/` holds exactly three projects
(`convergence-safety`, `type-safety`, `codegen-ptx`), which is also what
`scripts/check-formal-proofs.sh` pins as `EXPECTED_PROJECTS=3`. The stated
blocker — "BLOCKED on convergence-safety T3-SEMANTIC completing" — was cleared
when T3-S8 landed (see Stream 1), so this stream is now unblocked and idle, not
waiting. Anyone picking it up should also note that `formal/codegen-ptx/` has
since taken over part of the ground this stream was scoped for (an abstract GPU
semantics + emitter correctness theorems, `AGpuSemantics.v` /
`PtxExprSpec.v` / `PtxStmtSpec.v` / `PtxKernelSpec.v`), so its scope needs
re-cutting rather than resuming as written.

**Phases**:
| Phase | Status | Notes |
|-------|--------|-------|
| Spec plan (PTX-focused) | READY | cuda-semantics-spec-2026-06-13.md |
| PTX spike | DONE | Sarek_ir_ptx.ml validated on real hardware; long since superseded by the shipping Phase B backend |
| convergence-safety T3-S1 | DONE | ConvergenceSemantics.v = source semantics |
| convergence-safety T3-S8 (the stated blocker) | **DONE** | was the reason this stream was BLOCKED |
| GPUVerify TOPLAS deadlock def | NOT DONE | FlateDecode issue; read PDF manually |
| /formal-init cycle | NOT STARTED | prerequisite T3-SEMANTIC is now met; TOPLAS read still outstanding |
| First theorem: accepted_block_no_deadlock | NOT STARTED | |
| Full formal project (T0-T3) | NOT STARTED | re-scope first, per the status note above |

---

## Future streams

_This table was headed "Future streams (not in scope yet)". That heading is now
wrong for its first two rows, so it is gone rather than qualified: the
type-safety project shipped, and emitter-correctness work is partly done in
`formal/codegen-ptx/`. Shipped rows are kept here, marked, instead of being
deleted — deleting them would lose the prereq chain that predicted them._

| Project | Priority | Prereqs | Notes |
|---------|----------|---------|-------|
| type-safety of Sarek PPX | **SHIPPED** | convergence-safety T3 done ✓ | `formal/type-safety/` is on `origin/main` (scaffolded in `73083c1f`, milestone recorded on `main` by `c3f2578b` "T3-SEMANTIC milestone lock on main — both projects", 2026-07-02); its own `STATUS.md` records T3-SEMANTIC MILESTONE LOCKED 2026-06-15. Ledger: **90 theorems, 0 admits, 0 axioms** across 12 modules. Was listed here as "not in scope yet" |
| PTX emitter correctness | **PARTLY DONE** | — | Done as `formal/codegen-ptx/` (79 theorems, 0 admits, 6 sanctioned math `Parameter`s), not as a `cuda-semantics` sub-project. `emit_expr_correct` / `emit_stmt_correct` / `emit_kernel_correct` hold against a Rocq-side model; `PtxLayout.v` covers the aggregate byte ABI and is the one half checked against production via **extraction**. Remaining: extract the expr/stmt/kernel model too, so the conformance suite stops going through a hand mirror |
| sarek-opencl formal verification | MEDIUM | cuda-semantics methodology | Same shape, different target ISA |
| sarek-vulkan / sarek-metal | MEDIUM | Above | |
| cuda-semantics: memory model proof (not axiom) | LOW | cuda-semantics formal project | Certifies CUDA not Sarek; separate project |
| transpiler-fidelity (OCaml → PTX semantics preservation) | LOW | cuda-semantics + type-safety | Full compiler correctness; far future |
| WGSL backend formal verification | SKIP | — | User: low priority |

---

## Open decisions

| Decision | Options | Status |
|----------|---------|--------|
| cuda-semantics: extend convergence-safety/ or sibling? | Sibling (recommended) | OPEN, but moot until Stream 3 is re-scoped — `formal/codegen-ptx/` already exists as a sibling and holds the abstract GPU semantics |
| PTX backend: new opam package sarek-ptx/ or extend sarek-cuda/? | Extend sarek-cuda | **DECIDED — neither, in the end.** The emitter lives in the shared `sarek.codegen` library (`sarek/codegen/Sarek_ir_ptx*.ml`, six modules); `sarek-cuda/Sarek_ir_ptx.ml` is a 9-line re-export kept for source compatibility. No `sarek-ptx` package was created |

---

## How to continue in a new session

1. Read this file first — and check its claims against the tree before acting on
   them. Every stale line corrected on 2026-07-30 was stale in the direction of
   understating what exists, which is the direction that causes shipped work to
   be redone.
2. Read `formal/*/STATUS.md`, but trust `formal/*/proof-ledger.json` over it for
   counts: the ledgers are generated by `scripts/gen-proof-ledger.py` from a
   from-scratch Rocq build and CI fails on drift, while the STATUS files are
   hand-written.
3. Run `scripts/check-formal-proofs.sh` (it needs a Rocq toolchain; CI runs it
   inside `rocq/rocq-prover:9.1.1`). It rebuilds all three projects, runs
   `coqchk`, re-extracts `PtxLayout.v` and byte-compares against the committed
   copy, and enforces `formal/axiom-allowlist.txt` in both directions.
4. Work happens on `main` now — all three projects are there. The old per-phase
   branches are historical, and — checked against `git branch -r` on
   2026-07-30 — only some of them are still fetchable:
   `formal/type-safety-phase1a..d` are on `origin`;
   `formal/codegen-ptx-phase1` exists **only as a local branch here**, so a
   fresh clone will not have it; and `formal/convergence-safety-phase1a` —
   which step 2 of this list used to tell you to `git log` — is on neither.
   `formal/convergence-safety/formal-verif-autopilot.workflow.js` is still in
   the tree, but the ladder it drove is finished, so there are no T3 ticks left
   for it to take.
5. The highest-value formal action is no longer "Phase B" or "start
   cuda-semantics": it is **extracting the codegen-ptx expr/stmt/kernel model**
   so `test_codegen_ptx_conformance.ml` stops checking a hand-written mirror,
   the way the layout suite already does.
