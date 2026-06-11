# ConvergenceSafety formal — METHODOLOGY

Specialises the apparatus methodology in
`~/.claude/skills/formal-apparatus/docs/methodology.md`
for the convergence-safety project.

## Architecture

**3-layer** (no rate-aware / issuance / compound-fee logic detected in
Phase 0 scope read).

| Layer | Artifact | Role |
|---|---|---|
| L1 | `theories/ConvergenceSpec.v` | Abstract Rocq spec — types, functions, proofs |
| L2 | `extraction/ConvergenceModel.ml` | Extracted OCaml model — oracle for conformance tests |
| L3 | `test/test_convergence_{conformance,extraction}.ml` | Differential PBT harness |

## Spec source-of-truth

The authoritative specification is `ConvergenceSafetySpec.tex` (LaTeX). The
Rocq theories under `theories/` are the implementation that backs the LaTeX.
Post-lock (current state): every spec change goes into
`ConvergenceSafetySpec.tex` FIRST; Rocq follows. LaTeX-Rocq reconciliation pass
mandatory before any lock revision.

## Conformance shape

The project uses an **abstract-model differential** pattern:

1. **Inline abstract model** (`test_convergence_conformance.ml`): an OCaml
   re-implementation of the 6 semantic functions that mirrors
   `ConvergenceSpec.v`. QCheck2 tests 10 properties on it — this validates
   the inline model against the theorem statements.

2. **Extraction conformance** (`test_convergence_extraction.ml`): generates
   random `expr` values, translates them to `ConvergenceModel.expr` via
   `to_extracted`, runs both the inline and extracted model, compares
   results. 6 properties × 1500–2000 samples each. This validates that
   `coqc` extraction did not introduce bugs.

3. **texpr conformance** (future, T2): translate real `texpr` AST nodes via
   `texpr_to_expr`, run `Sarek_convergence.check_expr`, compare with
   `ConvergenceModel.check`. This is the true CMBT link against the
   implementation. Blocked on: SPOC test suite integration (needs the full
   sarek library available in dune deps).

## Testing strategy

- **Domain axis**: `test_helpers/coverage_probe.ml` instruments every PBT
  scenario with an eq-class tuple `(mode, varying_root, has_barrier, has_diverging, depth)`.
- **Source axis**: `bisect_ppx` when feasible (no known blockages for this project).

PBT layers:
- Quick: QCheck2, 1000–2000 samples, runs in dev and CI
- Long: not yet configured (single-layer, stateless spec — Monolith-native
  does not apply)

Hand-written conformance:
- None yet; any future regression test goes under `test/regressions/`
  and MUST link to a `findings/DIVERGENCE_FINDINGS.md` entry.

## Trust root

| Assumption | Rationale |
|---|---|
| Rocq kernel soundness | Standard across all apparatus projects |
| OCaml extraction + compiler | Standard TCB |
| Abstract model faithful to real implementation | Verified by code inspection; documented in `ASSUMPTIONS.md` |
| `is_thread_varying` completeness for `TEVar`/`TEIntrinsicConst` | Depends on `Sarek_core_primitives` — outside this spec's scope |

## Failure triage

Two-phase per `policy/CONFORMANCE_POLICY.md §"Conformance test fails"`:
- Phase 1 (harness-bug triage, autonomous)
- Phase 2 (spec-vs-proto classification, user-gated)

## CMBT completeness chain status

Required 7 links per apparatus methodology.

- [x] `theories/` with Rocq spec (11 theorems, coqchk PASS)
- [x] `extraction/` mapping spec to OCaml via `coqc` extraction
- [x] Zero `Admitted` + zero `Axiom` across `theories/`
- [x] `test/` with conformance harness (10 + 6 QCheck2 properties)
- [x] `test_helpers/coverage_probe.ml` (eq-class probe present)
- [x] `findings/` with audit-doc stubs (`DIVERGENCE_FINDINGS.md`, `UNCOVERED_EDGE_CASES.md`)
- [x] `report/` with dual-axis coverage table (`BENCHMARKS.md`)

**All 7 links present. Grade A achieved.**

## Specialisations

- No Monolith-native long tier (stateless spec; documented in `report/BENCHMARKS.md §Tool choice rationale`).
- texpr-level conformance test deferred to T2 (blocked on SPOC test suite integration).
  Rationale recorded in `ASSUMPTIONS.md §Assumed / unmodeled`.
