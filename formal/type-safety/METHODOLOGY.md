# TypeSafety formal — METHODOLOGY

Specialises the apparatus methodology in
`~/.claude/skills/formal-apparatus/docs/methodology.md`
for the type-safety project.

## Architecture

**3-layer** (post-unification model; no rate-aware / issuance / compound-fee logic).

| Layer | Artifact | Role |
|---|---|---|
| L1 | `theories/*.v` (12 files) | Abstract Rocq spec — types, functions, proofs |
| L2 | `extraction/*Model.ml` (12 files) | Extracted OCaml models — oracles for conformance tests |
| L3 | `test/test_type_safety_conformance.ml` | Differential PBT + smoke harness |

## Spec source-of-truth

This project was template-bootstrapped from convergence-safety (no `/formal-init` Phase 3 was run). The Rocq theories under `theories/` are the primary source of truth. A retroactive LaTeX spec is deferred; see `policy_overrides.md` for the formal waiver.

The 12 spec files form a dependency stack:

```text
TypeSafetySpec.v (top, Require Imports all 11 below)
  ├── UnifySpec.v      ← HM unification foundations
  ├── VecSpec.v        ← vector/array layer
  ├── RegistrySpec.v   ← record layer
  ├── ControlFlowSpec.v ← if/for/while/seq
  ├── OperatorSpec.v   ← arith/bool/cmp operators
  ├── FunSpec.v        ← function application + let-rec
  ├── MutSpec.v        ← mutable let / assign
  ├── PatternSpec.v    ← pattern matching / variant branches
  ├── ConstrSpec.v     ← constructor / record construction
  ├── SpecialSpec.v    ← SEReturn / SECreateArray / SETyped
  └── GPUSpec.v        ← GELetShared / GESuperstep
```

## Conformance shape

**Abstract-model differential** pattern (same as convergence-safety):

1. **Extraction**: each `*Spec.v` is extracted to a corresponding `*Model.ml`.
   The extracted OCaml is the oracle.

2. **Conformance harness** (`test_type_safety_conformance.ml`): generates random
   `expr` / `env` inputs, runs the extracted model, compares against expected
   outcomes. Organised in tiers:
   - **T1-CMBT**: top-level `infer_type` differential, 2000 samples, 20 smoke tests.
   - **T2-{UNIFY,VEC,REGISTRY}**: 1000 samples each; per-layer differential.
   - **T3-S1..S8**: 10–13 smoke tests per layer (control flow through GPU forms).

3. **Mutation tests** (`test_mutation.ml`): two targeted mutants:
   - M1 (type-erasure): `infer_type` always returns `Inl (TPrim TUnit)` — caught by the float-literal property.
   - M2 (variable-blind): `infer_type` returns `Inr (UnboundVar x)` for every `EVar` — caught by the let-bound-var property.

## Testing strategy

- **Domain axis**: random `expr` generation with type-directed shrinking via QCheck2.
- **Source axis**: `bisect_ppx` when feasible (no known blockages).

PBT layers:
- Quick: QCheck2, 1000–2000 samples, runs in dev and CI.
- Long: not configured (Monolith-native does not apply to a stateless post-unification spec).

Hand-written conformance:
- Smoke tests (10–13 per layer) are deterministic; any future regression test goes under
  `test/regressions/` and MUST link to a `findings/FINDINGS.md` entry.

## Trust root

| Assumption | Rationale |
|---|---|
| Rocq kernel soundness | Standard across all apparatus projects |
| OCaml extraction + compiler | Standard TCB |
| Post-unification model faithful to real type checker | Verified by code inspection; `Sarek_typer.ml` resolves all TVars before emitting `texpr` — the spec models the resolved state |
| Mutable unification variables abstracted away | Documented in PLAN.md §"Termination design": `follow`/`follow_pvar` model the chain-following; the mutation is transparent post-unification |

## Failure triage

Two-phase per apparatus `policy/CONFORMANCE_POLICY.md §"Conformance test fails"`:
- Phase 1 (harness-bug triage, autonomous)
- Phase 2 (spec-vs-proto classification, user-gated)

## CMBT completeness chain status

Required 7 links per apparatus methodology.

- [x] `theories/` with 12 Rocq spec files (90 declarations, coqchk PASS)
- [x] `extraction/` mapping each spec to OCaml via `coqc` extraction (12 *Model.ml)
- [x] Zero `Admitted` + zero `Axiom` across all `theories/*.v`
- [x] `test/` with conformance harness (T1-CMBT + T2 + T3-S1..S8, all GREEN)
- [x] Mutation tests present (`test/test_mutation.ml`, M1 + M2 caught)
- [x] `coqchk` passes (0 axioms) across all 12 spec files
- [x] Open findings documented in `findings/FINDINGS.md` (F-TS-01 RESOLVED)

**All 7 links present. Grade A achieved.**

## Specialisations

- No Monolith-native long tier (stateless spec; no replay-able mutable transitions).
- No `test_helpers/coverage_probe.ml` (eq-class probe not added; this is an optional Grade A artifact per apparatus docs).
- LaTeX spec deferred; Rocq is source of truth — see `policy_overrides.md`.
