# ConvergenceSafety — Work Plan

**Last updated**: 2026-06-12 (tick 1 — T1A-CONF done; T3-GATE is current task)
**Apparatus version**: 1.1.0
**Phase**: Post-T1A-CONF (20 theorems, 0 admits, 0 axioms, coqchk passes; 17 conformance, 7 extraction)

---

## Open tasks

| ID | Title | Tier | Status | Blocked by |
|---|---|---|---|---|
| T1A-CONF | Extend QCheck conformance tests — 3 dedicated ESuperstep false/true QCheck properties | T1 | **done** (17/17 green, PR #182 merged 2026-06-12T06:53:15Z) | — |
| T1-LATEX | Reconcile ConvergenceSafetySpec.tex — add superstep_outer_diverged_error; verify all theorems listed | T1 | **done** (v0.4, 20 theorems, 2026-06-12) | — |
| T1-LEDGER | Sync proof-ledger.json — 20 theorems, summary.total=20 | T1 | **done** (20 entries, summary.total=20) | — |
| T2-F02 | Environment-threaded is_varying in Rocq spec for F-02 | T2 | **done** | — |
| T2-WARP | WarpConvergence error class: EWarpPoint/WarpError/check_warp/warp_diverged_error/warp_mode_monotone/warp_varying_if_flags | T2 | **done** | — |
| T2-RETURN | TEReturn early-return barrier-skip: EReturn constructor, model barrier-skip, safety theorem | T2 | **done** (return_barrier_skip_safe, 20 theorems, 17 conformance, 7 extraction) | — |
| T3-GATE | HUMAN DECISION — surface "All T1+T2 done. Confirm T3-SEMANTIC in scope." | T3 | **open (human gate)** | — |
| T3-SEMANTIC | Semantic soundness with execution semantics | T3 | open | T3-GATE |
| DOCS-SYNC | STATUS.md / ASSUMPTIONS.md / proof-ledger.json / ConvergenceSafetySpec.tex drift check | hygiene | **clean** (verified tick 1) | — |

---

## Current task

**T3-GATE — HUMAN DECISION: All T1+T2 done. Confirm T3-SEMANTIC in scope.**

All T1 and T2 work is complete:
- T1A-CONF: DONE — 3 dedicated ESuperstep QCheck properties added
  (`superstep_outer_diverged_error`, `superstep_no_entry_error_converged`,
  `superstep_body_errors_propagate`); 17/17 conformance properties GREEN.
  PR #182 merged 2026-06-12T06:53:15Z.
- T1-LATEX: DONE — ConvergenceSafetySpec.tex v0.4, 20 theorem environments.
- T1-LEDGER: DONE — proof-ledger.json: 20 entries, summary.total=20.
- T2-F02: DONE — env-threaded is_varying, 3 new theorems.
- T2-WARP: DONE — WarpConvergence error class, 3 new theorems.
- T2-RETURN: DONE — EReturn/return_barrier_skip_safe, compositionality theorem.
- DOCS-SYNC: Clean — no drift detected.

### Human gate

This is a **workflow stop**. The agent cannot proceed to T3-SEMANTIC without explicit
human confirmation.

Surface to operator:
> All T1+T2 work is complete. 20 theorems, 0 admits, 0 axioms, coqchk passes.
> Conformance: 17/17 green. Extraction: 7/7 green. Live CMBT: 10/10 green.
> Confirm: is T3-SEMANTIC (semantic soundness — months of work, separate sub-project) in scope?

### What T3-SEMANTIC entails (for operator decision)

T3-SEMANTIC requires:
- Defining an execution semantics (small-step or big-step) for the `expr` AST.
- Proving that `check_expr` is sound with respect to that semantics — i.e., that
  every program accepted by `check_expr` (zero errors) does not exhibit a diverged
  barrier at runtime under the semantics.
- Estimated effort: months; likely a separate Rocq sub-project.
- Not required for the current safety guarantees (which are syntactic/type-theoretic).

---

## Blocker rules (standing)

- T3-GATE is a human gate — workflow pauses and surfaces: "All T1+T2 done. Confirm T3-SEMANTIC in scope."
- T3-SEMANTIC is blocked by T3-GATE.

---

## Workflow notes

- Tick 1 (2026-06-12): T1A-CONF executed and confirmed DONE. PR #182 merged
  (confirmed via `gh pr view 182 --repo mathiasbourgoin/Sarek`; merged 2026-06-12T06:53:15Z).
  Added 3 ESuperstep QCheck properties to test_convergence_conformance.ml:
  `superstep_outer_diverged_error` (F-01 direct), `superstep_no_entry_error_converged`
  (safe-path complement), `superstep_body_errors_propagate` (monotonicity).
  17/17 conformance properties GREEN. STATUS.md updated. DOCS-SYNC clean.
  currentTask promoted to T3-GATE (human gate — workflow stops).
- Tick 0 (2026-06-12, fresh session): T2-RETURN confirmed DONE (STATUS.md: 20 theorems,
  return_barrier_skip_safe, 0 admits, 0 axioms, coqchk passes; 14 conformance props, 7 extraction,
  all green). T1-LATEX confirmed DONE (ConvergenceSafetySpec.tex v0.4, 20 theorem environments,
  LOCKED 2026-06-12). T1-LEDGER confirmed DONE (proof-ledger.json: 20 entries, summary.total=20).
  DOCS-SYNC clean — no drift detected across ledger/LaTeX/STATUS.
  PR #182 STILL OPEN — T1A-CONF remained hard-blocked.
  currentTask = T1A-CONF (awaiting PR #182 merge to execute).
- Prior tick 1 (2026-06-12): T2-WARP confirmed DONE. T1-LEDGER confirmed DONE (19 entries).
  T1-LATEX re-opened (drift: 17 LaTeX envs vs 19 in ledger). currentTask promoted to T2-RETURN.
- Prior tick 0 (2026-06-12): T1-LATEX confirmed DONE (v0.2 LOCKED, 16 theorems). T1-LEDGER DONE.
  T2-F02 DONE. PR #182 STILL OPEN. currentTask promoted to T2-WARP.
- Prior tick 0 (2026-06-11): T2-F02 discovered DONE; T1-LATEX promoted as currentTask; T1-LEDGER queued.
