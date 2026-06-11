# ConvergenceSafety — Work Plan

**Last updated**: 2026-06-11 (tick 0, re-evaluated)
**Apparatus version**: 1.1.0
**Phase**: Post-T2-F02 (16/16 theorems, 0 admits, 0 axioms, coqchk passes)

---

## Open tasks

| ID | Title | Tier | Status | Blocked by |
|---|---|---|---|---|
| T1A-CONF | Extend QCheck conformance tests to cover ESuperstep patterns | T1 | open | HARD: PR #182 not merged |
| T1-LATEX | Reconcile ConvergenceSafetySpec.tex — add superstep_outer_diverged_error + T2-F02 theorems; remove stale elision note; verify all 16 theorems listed | T1 | open | — |
| T1-LEDGER | Sync proof-ledger.json — add missing 5 theorems; update total count to 16 | T1 | open | — |
| T2-F02 | Environment-threaded is_varying in Rocq spec for F-02 | T2 | **done** | — |
| T2-WARP | WarpConvergence error class in spec | T2 | open | — |
| T2-RETURN | TEReturn early-return barrier-skip analysis | T2 | open | — |
| T3-GATE | HUMAN DECISION — surface "All T1+T2 done. Confirm T3-SEMANTIC in scope." | T3 | open | T1A-CONF, T1-LATEX, T1-LEDGER, T2-WARP, T2-RETURN all done |
| T3-SEMANTIC | Semantic soundness with execution semantics | T3 | open | T3-GATE |
| DOCS-SYNC | STATUS.md / ASSUMPTIONS.md / proof-ledger.json drift check | hygiene | open | — |

---

## Current task

**T1-LATEX — Reconcile ConvergenceSafetySpec.tex**

PR #182 is still open → T1A-CONF remains hard-blocked.
T2-F02 is done (committed da7ae592).
First non-hard-blocked open task is T1-LATEX.

### Observed drift (tick 0 re-evaluation)

`ConvergenceSafetySpec.tex` is locked at v0.1 (2026-06-11) and reflects only
the original 11 theorems. Since then, Phase 1a and T2-F02 added 5 more:

| Theorem | Added in | Missing from LaTeX |
|---|---|---|
| `superstep_outer_diverged_error` | Phase 1a | yes |
| `env_lookup_extend_same` | T2-F02 | yes |
| `env_let_alias_varying` | T2-F02 | yes |
| `env_var_diverged_clean` | T2-F02 | yes |
| `env_check_let_alias_catches` | T2-F02 | yes |

Additional stale content:
- Abstract says "eleven properties" — must become sixteen.
- Section 4 "Out of scope" still lists `TESuperstep` as elided — ESuperstep
  was brought in scope in Phase 1a; remove that entry.
- Section 4 "Elided constructors" still shows `TESuperstep` — remove or annotate.
- No `ESuperstep` constructor in the abstract `expr` grammar (Definition 2.3).
- No `EVar` constructor in the abstract `expr` grammar (added in T2-F02).
- `check(m, ESuperstep(...))` rule is absent from the semantics section.
- `check_env` / `is_varying_in_env` / `Env` definitions are absent.

### Approach

1. Update the abstract grammar (Definition 2.3) to add `ESuperstep(b, body,
   cont)` and `EVar(x)` constructors.
2. Add the `check(m, ESuperstep(...))` semantic rule (resets mode to Converged,
   then checks body + cont; outer-Diverged case yields BarrierError).
3. Add `Env`, `env_extend`, `env_lookup`, `is_varying_in_env`, `check_env`
   definitions (new subsection for T2-F02 env-threaded model).
4. Add all 5 missing theorems in the Properties section (T1 subsection).
5. Update abstract to say "sixteen properties" (or use a symbolic count macro).
6. Remove `TESuperstep` from the elided-constructors list; update the
   type-correspondence table to include `ESuperstep → TESuperstep`.
7. Update the version/date stamp to v0.2 (post-T2-F02).

### Expected deliverables

- `ConvergenceSafetySpec.tex` updated to v0.2 with all 16 theorems listed,
  ESuperstep and EVar in grammar, new semantic rules, env-threaded definitions,
  stale elision note removed.
- `STATUS.md` "Last updated" field bumped if changed.

---

## Upcoming task (after T1-LATEX)

**T1-LEDGER — Sync proof-ledger.json**

`proof-ledger.json` currently lists 11 theorems (summary.total = 11).
Five theorems added since then are absent:

- `superstep_outer_diverged_error` (T1, Phase 1a)
- `env_lookup_extend_same` (T1, T2-F02)
- `env_let_alias_varying` (T1, T2-F02)
- `env_var_diverged_clean` (T1, T2-F02)
- `env_check_let_alias_catches` (T1, T2-F02)

Update `proof-ledger.json`: add all 5 entries, set summary.total = 16,
summary.proven_clean = 16, update `generated` timestamp, and re-verify that
`vo_sha256` matches the current compiled `.vo` (or note it needs refresh).

---

## Blocker rules (standing)

- T1A-CONF is hard-blocked until PR #182 merges.
  Check `gh pr list --repo mathiasbourgoin/Sarek --state open` at each tick.
- T3-GATE blocks T3-SEMANTIC; T3-GATE is not unblocked until all T1+T2 items
  are done.

---

## Workflow notes

- Tick 0 re-evaluation (2026-06-11): T2-F02 discovered DONE (commit da7ae592).
  PLAN.md updated: T2-F02 marked done; T1-LATEX and T1-LEDGER added from
  roadmap; currentTask promoted to T1-LATEX.
- proof-ledger.json confirmed stale: 11 theorems listed vs 16 in STATUS.md.
- ConvergenceSafetySpec.tex confirmed stale: locked at v0.1, 11 theorems, no
  ESuperstep/EVar grammar, no env-threaded definitions.
- PR #182 is OPEN as of 2026-06-11; T1A-CONF remains hard-blocked.
- DOCS-SYNC is a background hygiene item; drift confirmed this tick (see above).
