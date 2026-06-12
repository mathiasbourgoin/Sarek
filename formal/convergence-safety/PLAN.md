# ConvergenceSafety — Work Plan

**Last updated**: 2026-06-12 (tick 0 — fresh session)
**Apparatus version**: 1.1.0
**Phase**: Post-T2-RETURN (20 theorems, 0 admits, 0 axioms, coqchk passes)

---

## Open tasks

| ID | Title | Tier | Status | Blocked by |
|---|---|---|---|---|
| T1A-CONF | Extend QCheck conformance tests — ≥3 dedicated ESuperstep false/true QCheck properties | T1 | open | HARD: PR #182 not merged |
| T1-LATEX | Reconcile ConvergenceSafetySpec.tex — add superstep_outer_diverged_error; verify all 12 theorems listed | T1 | **done** (v0.4, 20 theorems, 2026-06-12) | — |
| T1-LEDGER | Sync proof-ledger.json — 20 theorems, summary.total=20 | T1 | **done** (20 entries, summary.total=20) | — |
| T2-F02 | Environment-threaded is_varying in Rocq spec for F-02 | T2 | **done** | — |
| T2-WARP | WarpConvergence error class: EWarpPoint/WarpError/check_warp/warp_diverged_error/warp_mode_monotone/warp_varying_if_flags | T2 | **done** | — |
| T2-RETURN | TEReturn early-return barrier-skip: EReturn constructor, model barrier-skip, safety theorem | T2 | **done** (return_barrier_skip_safe, 20 theorems, 14 conformance, 7 extraction) | — |
| T3-GATE | HUMAN DECISION — surface "All T1+T2 done. Confirm T3-SEMANTIC in scope." | T3 | open | T1A-CONF not done (hard-blocked by PR #182) |
| T3-SEMANTIC | Semantic soundness with execution semantics | T3 | open | T3-GATE |
| DOCS-SYNC | STATUS.md / ASSUMPTIONS.md / proof-ledger.json / ConvergenceSafetySpec.tex drift check | hygiene | **clean** (verified tick 0) | — |

---

## Current task

**T1A-CONF — Extend QCheck conformance tests for ESuperstep**

All T2 work is complete (T2-F02 / T2-WARP / T2-RETURN — 20 theorems, 0 admits, 0 axioms, coqchk passes).
All DOCS-SYNC items are clean: proof-ledger.json at 20 entries / summary.total=20; ConvergenceSafetySpec.tex at v0.4 with 20 theorem environments; STATUS.md current.
PR #182 STILL OPEN — T1A-CONF is hard-blocked this tick.

### Blocker

HARD: PR #182 (`formal(convergence-safety): Phase 1a — ESuperstep in spec, F-01 theorem`) is not yet merged into mathiasbourgoin/Sarek.
This task cannot execute until that PR merges.

### Scope (execute once PR #182 merges)

Add ≥3 new QCheck properties to `test/test_convergence_conformance.ml` covering ESuperstep:

1. **ESuperstep false (outer-diverged error)**: `check Diverged (ESuperstep false body cont) ≠ []`
   — property: for arbitrary body/cont, outer-Diverged mode always produces errors.
2. **ESuperstep true (inner superstep, converged outer)**: `check Converged (ESuperstep true body cont) = []`
   — property: if outer mode is Converged and superstep resets to Converged, body/cont clean.
3. **ESuperstep false/true — body errors propagated**: verify that errors in body/cont
   are included in the overall check output (monotonicity for ESuperstep).

Total conformance properties after this task: ≥17 (currently 14).

### Expected deliverables

- `test/test_convergence_conformance.ml`: ≥3 new QCheck properties for ESuperstep; total ≥17 properties, all GREEN.
- `STATUS.md`: bump test counts.
- `proof-ledger.json`: update `generated` date (no new theorems, just test coverage).

---

## Blocker rules (standing)

- T1A-CONF is hard-blocked until PR #182 merges.
  Check `gh pr list --repo mathiasbourgoin/Sarek --state open` at each tick.
- T3-GATE blocks T3-SEMANTIC; T3-GATE is not unblocked until T1A-CONF is done.
- T3-GATE is a human gate — workflow pauses and surfaces: "All T1+T2 done. Confirm T3-SEMANTIC in scope."

---

## Workflow notes

- Tick 0 (2026-06-12, fresh session): T2-RETURN confirmed DONE (STATUS.md: 20 theorems,
  return_barrier_skip_safe, 0 admits, 0 axioms, coqchk passes; 14 conformance props, 7 extraction,
  all green). T1-LATEX confirmed DONE (ConvergenceSafetySpec.tex v0.4, 20 theorem environments,
  LOCKED 2026-06-12). T1-LEDGER confirmed DONE (proof-ledger.json: 20 entries, summary.total=20).
  DOCS-SYNC clean — no drift detected across ledger/LaTeX/STATUS.
  PR #182 STILL OPEN — T1A-CONF remains hard-blocked.
  currentTask = T1A-CONF (awaiting PR #182 merge to execute).
- Tick 1 (2026-06-12): T2-WARP confirmed DONE. T1-LEDGER confirmed DONE (19 entries).
  T1-LATEX re-opened (drift: 17 LaTeX envs vs 19 in ledger). currentTask promoted to T2-RETURN.
- Tick 0 (2026-06-12): T1-LATEX confirmed DONE (v0.2 LOCKED, 16 theorems). T1-LEDGER DONE.
  T2-F02 DONE. PR #182 STILL OPEN. currentTask promoted to T2-WARP.
- Prior tick 0 (2026-06-11): T2-F02 discovered DONE; T1-LATEX promoted as currentTask; T1-LEDGER queued.
