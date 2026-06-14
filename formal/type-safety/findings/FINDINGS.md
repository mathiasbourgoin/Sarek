# TypeSafety — Findings

No findings yet. Type-safety properties (soundness, completeness, preservation,
unification correctness) are **TBD after T1-SPEC** — they emerge once the
Admitted lemmas in `theories/TypeSafetySpec.v` are proved (T1-SOUND) and the
extraction conformance harness runs against `Sarek_typer.infer` (T1-CMBT).

## Finding template

| ID | Severity | Target (file:line) | Property | Status |
|----|----------|--------------------|----------|--------|

Findings discovered during proof work (e.g. a soundness lemma that fails to
prove, revealing a checker bug or an over-strong spec) are recorded here with a
concrete reproduction and a pointer to the failing theorem.
