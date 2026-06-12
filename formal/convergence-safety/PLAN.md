# ConvergenceSafety — Work Plan

**Last updated**: 2026-06-12 (tick 1)
**Apparatus version**: 1.1.0
**Phase**: Post-T2-WARP (19 theorems, 0 admits, 0 axioms, coqchk passes)

---

## Open tasks

| ID | Title | Tier | Status | Blocked by |
|---|---|---|---|---|
| T1A-CONF | Extend QCheck conformance tests — ≥3 dedicated ESuperstep false/true QCheck properties | T1 | open | HARD: PR #182 not merged |
| T1-LATEX | Reconcile ConvergenceSafetySpec.tex — add warp_mode_monotone + warp_varying_if_flags; update header to 19 theorems | T1 | **open** (drifted) | — |
| T1-LEDGER | Sync proof-ledger.json — 19 theorems, summary.total=19 | T1 | **done** | — |
| T2-F02 | Environment-threaded is_varying in Rocq spec for F-02 | T2 | **done** | — |
| T2-WARP | WarpConvergence error class: EWarpPoint/WarpError/check_warp/warp_diverged_error/warp_mode_monotone/warp_varying_if_flags | T2 | **done** | — |
| T2-RETURN | TEReturn early-return barrier-skip: EReturn constructor, model barrier-skip, safety theorem | T2 | open | — |
| T3-GATE | HUMAN DECISION — surface "All T1+T2 done. Confirm T3-SEMANTIC in scope." | T3 | open | T1A-CONF, T2-RETURN not done |
| T3-SEMANTIC | Semantic soundness with execution semantics | T3 | open | T3-GATE |
| DOCS-SYNC | STATUS.md / ASSUMPTIONS.md / proof-ledger.json / ConvergenceSafetySpec.tex drift check | hygiene | open | — |

---

## Current task

**T2-RETURN — TEReturn early-return barrier-skip**

T2-WARP complete (confirmed 2026-06-12): EWarpPoint/WarpError/check_warp/warp_diverged_error/warp_mode_monotone/warp_varying_if_flags — 19 theorems, 13 conformance properties, 7 extraction tests, all green.
PR #182 still open → T1A-CONF remains hard-blocked.
T1-LATEX has drifted (LaTeX has 17 theorem environments; spec has 19) — flagged for DOCS-SYNC.
Next non-hard-blocked open task is T2-RETURN.

### Scope

Extend `ConvergenceSpec.v` with an EReturn constructor that models early-return barrier-skip:

1. **Add EReturn constructor** to the `expr` inductive type.
2. **Model barrier-skip**: `check` on `EReturn` in Diverged mode returns no errors
   (early return exits the current control-flow path before reaching any barrier).
3. **Prove theorem `return_barrier_skip_safe`** (or equivalent name): if an
   expression contains EReturn, barriers after it are unreachable under the
   diverged path, i.e., early return is safe with respect to barrier checks.
4. **Update conformance test**: add ≥1 QCheck property for return_barrier_skip_safe;
   total becomes 14 conformance properties.
5. **Update proof-ledger.json**: add `return_barrier_skip_safe` entry, bump total
   to 20, set `generated` = today's date.
6. **Update STATUS.md**: bump "Last updated", add theorem row, update project
   description and test counts.

### Key question to resolve before coding

Read the current `ConvergenceSpec.v` to confirm:
- The exact name and shape of the `expr` inductive (for adding EReturn).
- How `check` handles sequential composition (`ESeq`) — EReturn likely short-circuits
  the remainder of a sequence.
- Whether a separate `check_return` helper is cleaner, or whether EReturn as a
  constructor in `check` with a simple `[] (* EReturn always safe *)` case suffices.

### Approach sketch

Option A — EReturn is a no-op for check: `check mode (EReturn) = []`. Theorem
`return_barrier_skip_safe` states `check mode EReturn = []` (trivial by simpl).
Then add a stronger compositionality theorem: in `ESeq EReturn rest`, the rest
is unreachable — barrier errors in `rest` are not reported.

Option B — EReturn aborts the sequence: model `ESeq` so that if the first element
is EReturn, subsequent elements are skipped. Stronger but requires a non-trivial
change to `check_seq` or `ESeq` semantics.

Recommended: read `ConvergenceSpec.v` first. If ESeq is defined as
`check mode e1 ++ check mode e2`, Option A (EReturn = []) may be sufficient for
the safety property. If a stronger skip theorem is needed, choose Option B.

### Expected deliverables

- `theories/ConvergenceSpec.v`: `EReturn` constructor, `check` case, theorem
  `return_barrier_skip_safe` (0 admits).
- `test/test_convergence_conformance.ml`: ≥1 new QCheck property; total becomes
  14 properties.
- `proof-ledger.json`: 20 entries, summary.total=20, generated=today.
- `STATUS.md`: updated theorem table, test counts, Last-updated line.
- `ConvergenceSafetySpec.tex`: add EReturn grammar entry and
  `return_barrier_skip_safe` theorem entry; update header theorem count to 20.

---

## DOCS-SYNC note (tick 1)

**ConvergenceSafetySpec.tex drift detected**: LaTeX file has 17 theorem/lemma
environments; proof-ledger.json has 19. Missing entries: `warp_mode_monotone`
and `warp_varying_if_flags` (both added in T2-WARP). The LaTeX header also
reads "17 theorems" (v0.3). This drift should be fixed as part of T2-RETURN
deliverables (fold into the same pass that adds the EReturn theorem entry),
or as a standalone DOCS-SYNC pass if T2-RETURN is delayed.

---

## Blocker rules (standing)

- T1A-CONF is hard-blocked until PR #182 merges.
  Check `gh pr list --repo mathiasbourgoin/Sarek --state open` at each tick.
- T3-GATE blocks T3-SEMANTIC; T3-GATE is not unblocked until all T1+T2 items
  (T1A-CONF, T2-RETURN) are done.

---

## Workflow notes

- Tick 1 (2026-06-12): T2-WARP confirmed DONE (STATUS.md: 19 theorems, 0 admits,
  proof-ledger.json: 19 entries with warp_diverged_error/warp_mode_monotone/
  warp_varying_if_flags, all PROVEN). T1-LEDGER confirmed DONE (19 entries,
  summary.total=19). PR #182 STILL OPEN — T1A-CONF remains hard-blocked.
  T1-LATEX re-opened: LaTeX has 17 theorem environments vs 19 in ledger (drift
  introduced by T2-WARP additions). DOCS-SYNC: LaTeX drift flagged above.
  currentTask promoted to T2-RETURN (first non-hard-blocked open task).
- Tick 0 (2026-06-12): T1-LATEX confirmed DONE (ConvergenceSafetySpec.tex is
  v0.2 LOCKED, 16 theorems, ESuperstep/EVar grammar, env-threaded section
  present). T1-LEDGER confirmed DONE (proof-ledger.json: 16 entries,
  summary.total=16). T2-F02 confirmed DONE (STATUS.md + git log).
  PR #182 STILL OPEN — T1A-CONF remains hard-blocked.
  currentTask promoted to T2-WARP (first non-hard-blocked open task).
- Prior tick 0 (2026-06-11): T2-F02 discovered DONE; T1-LATEX promoted as
  currentTask; T1-LEDGER queued.
