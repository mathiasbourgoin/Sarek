# ConvergenceSafety — Work Plan

**Last updated**: 2026-06-11 (tick 0)
**Apparatus version**: 1.1.0
**Phase**: Post-1a (12/12 theorems, 0 admits, 0 axioms)

---

## Open tasks

| ID | Title | Tier | Status | Blocked by |
|---|---|---|---|---|
| T1A-CONF | Extend QCheck conformance tests to cover ESuperstep patterns | T1 | open | HARD: PR #182 not merged |
| T2-F02 | Environment-threaded is_varying in Rocq spec for F-02 | T2 | open | — |
| T2-WARP | WarpConvergence error class in spec | T2 | open | — |
| T2-RETURN | TEReturn early-return barrier-skip analysis | T2 | open | — |
| T3-SEMANTIC | Semantic soundness with execution semantics | T3 | open | — |
| DOCS-SYNC | STATUS.md / ASSUMPTIONS.md / proof-ledger.json drift check | hygiene | open | — |

---

## Current task

**T2-F02 — Environment-threaded is_varying in Rocq spec for F-02**

PR #182 is still open (not merged), so T1A-CONF is hard-blocked this tick.
The first non-hard-blocked open task is T2-F02.

### Approach

F-02 finding: `is_thread_varying` is binding-blind — let-aliased thread-varying
values are not propagated. The OCaml fix (PR #181, merged) introduces a
`varying_vars` environment threaded through `check_expr`. The Rocq spec
currently mirrors the *old* (unfixed) checker. This task brings the spec up to
date with the fix:

1. Extend `ConvergenceSpec.v` with an `Env` type (a finite map `var → bool`)
   and a `is_varying_in_env` predicate.
2. Thread `Env` through the `check` function in the abstract model.
3. Handle `TELet` / `TEVar` binding cases in the spec.
4. Extend the conformance abstract model
   (`test/test_convergence_conformance.ml`) to mirror the env-threaded checker.
5. Add at least one new conformance property exercising the let-alias scenario.
6. Verify `coqchk` still passes with 0 axioms.

### Expected deliverables

- Updated `theories/ConvergenceSpec.v` with env-threaded `is_varying`.
- Updated abstract OCaml model in conformance test.
- At least one new QCheck property for F-02 pattern.
- `coqchk` GREEN, 0 admits, 0 axioms.
- STATUS.md updated to reflect T2-F02 progress.

---

## Blocker rules (standing)

- T1A-CONF is hard-blocked until PR #182 merges into the default branch.
  Check `gh pr list --repo mathiasbourgoin/Sarek --state open` at each tick.

---

## Workflow notes (tick 0 — initial plan)

- PLAN.md created from scratch at tick 0 (file did not previously exist).
- PR #182 is OPEN as of 2026-06-11; T1A-CONF remains hard-blocked.
- T2-F02 promoted to currentTask as first non-blocked T2 item.
- DOCS-SYNC is always a background hygiene item; check at each tick whether
  STATUS.md "Last updated" matches the most recent commit and whether
  proof-ledger.json (if it exists) reflects current theorem count.
