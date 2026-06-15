# TypeSafety — Policy Overrides

## LaTeX spec (Rule 2 / `/formal-init` Phase 3) — WAIVED

**Policy**: Grade A projects must have a LaTeX spec as source of truth, produced
during `/formal-init` Phase 3 before Rocq work begins.

**Actual state**: No LaTeX spec exists. This project was template-bootstrapped
from convergence-safety (copying the apparatus scaffold) rather than going through
`/formal-init`. The Rocq theories were developed directly, with no prior LaTeX elicitation phase.

**Rationale**: The Rocq spec IS the source of truth. The 12 theory files cover
the full post-unification type inference surface of `Sarek_typer.ml` (1154 lines).
All 90 theorem statements are machine-checked; the spec is not draft material that
needs a human-readable LaTeX precursor for alignment purposes. A retroactive LaTeX
document would duplicate the Rocq statements without adding verification value.

**Commitment**: If the project scope expands (new AST forms, unification revision,
or a second lock revision), a `TypeSafetySpec.tex` will be produced before Rocq
work begins on that revision, per the standard apparatus flow.

**Reviewer acknowledgement**: This waiver was reviewed and accepted at milestone-lock
(2026-06-15, T3-SEMANTIC complete, 90 theorems, 0 admits, 0 axioms).
