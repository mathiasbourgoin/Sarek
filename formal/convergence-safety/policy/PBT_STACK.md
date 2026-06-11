# PBT Stack — Nightly / On-Demand Coverage

**Normative document** (locked stack 2026-Q2; generic rewrite
2026-06-11, apparatus v1.1 — locked semantics unchanged).

A formal project's conformance tests are **not part of CI-default**;
they run on a nightly schedule or on developer demand.  This gives
4–12 h/night of wall-clock budget instead of the ~60 s CI cap.  This
document defines the tooling stack that turns that budget into the
strongest achievable empirical coverage.

The stack below was selected empirically by the staking 2026-Q2
bakeoff (run under `policy/BENCHMARK_POLICY.md`); the full verdict
table, the original staking planning record it superseded, and the
AFL methodology lessons are archived in
`examples/tezos/pbt_stack_2026Q2.md` (cited below as "the archive").

## The locked stack

| Tier | Tool | Role |
|---|---|---|
| **Quick** | **QCheck2** (run via `QCheck_alcotest.to_alcotest`) | Pre-push / suite-default stateful PBT — seconds-to-minutes budget; part of the project's normal test suite. |
| **Long** | **Monolith-native** (`declare` + `Monolith.main` + `Gen`, /dev/urandom-driven random mode) | Nightly / on-demand stateful conformance — ~30 min to multi-hour runs; eq-class coverage to saturation.  Pareto-optimal at every duration band in the 2026-Q2 matrix (+28 % eq-classes over QCheck2 at 3600 s). |
| **Differential** | model-vs-target lockstep | One representative per observed eq-class, replayed through the real target and the certified extraction; tie-break-robust observables. |

No other PBT framework is used on new apparatus projects unless the
host profile declares one (and then in addition, never instead).
Specifically NOT in the stack:

- **AFL-guided drivers on stateful workloads** — strictly dominated
  by the random rows at every duration in the matrix (input-encoding
  tax + bitmap saturation; eq-classes plateau early).  If AFL is ever
  used (e.g. for a pure-function fuzzing side-tier), the **fork-exec
  pattern is mandatory**: let the harness exit per input rather than
  using a persistent loop — on the staking workload this lifted AFL
  bitmap stability from ~41 % to ~98 % and roughly doubled edges
  found per wall-clock (methodology in the archive).
- **Crowbar** — reserved for arithmetic-unit fuzzing of pure
  functions, where straight random over small tuples covers quickly;
  not a stateful-tier candidate (weakest eq-class results in the
  matrix).
- **Hand-written Alcotest** as a coverage tool — forbidden by
  `policy/CONFORMANCE_POLICY.md §Tool discipline` (allowed only as
  the runner of PBT cases and for `test/regressions/`).

## Role of each layer

```
┌──────────────────────────────────────────────────────────────────┐
│  Proofs: spec ≡ model (simulation) and spec ≡ target arithmetic  │
│  (Conforms lemmas) — mathematical guarantee on the primitives    │
├──────────────────────────────────────────────────────────────────┤
│  Monolith-native (long tier, nightly / on-demand)                │
│  — random stateful scenarios over the extracted certified model  │
│    vs the project's validation machine; eq-class probe attached  │
├──────────────────────────────────────────────────────────────────┤
│  QCheck2 (quick tier, pre-push / suite-default)                  │
│  — same workload, small budget; phantom-typed harness            │
├──────────────────────────────────────────────────────────────────┤
│  Differential (model vs target, per eq-class representative)     │
│  — pins extraction ≡ target on every reached equivalence class   │
└──────────────────────────────────────────────────────────────────┘
```

- **Proofs (top, strongest).**  Every function with a Qed simulation
  or Conforms lemma can be REMOVED from the empirical testing budget
  — the long tier focuses on the *un-proved* parts (boundary logic,
  queue management, multi-step drift).
- **Long tier.**  Discovers eq-classes and divergences at scale;
  run to coverage saturation (decelerating eq-class curve — see
  `docs/coverage.md` for the plateau threshold).
- **Quick tier.**  Keeps the suite-default green signal cheap;
  phantom types (project-specific `Mutez.t`-style wrappers) make
  unit-confusion bugs unrepresentable in the harness.
- **Differential.**  One representative scenario per eq-class
  replayed against the live target; observables chosen tie-break
  robust (win-counts / final-credits style, not raw traces).

## Cadence rules

- **Generator-update smoke run** — any extension of the long-tier
  generator triggers a mandatory 5-minute smoke run, per
  `policy/BENCHMARK_POLICY.md §Generator-update smoke run`.
- **Bakeoff staleness** — `/formal-check` rule 6 flags a project
  whose last Monolith-native run predates its last generator update.
- **Fail-loud threshold** — ANY divergence found by any tier opens a
  Finding entry (per `policy/CONFORMANCE_POLICY.md §Failure triage`)
  and the scenario becomes a permanent regression test under
  `test/regressions/`.  There is no "rerun and see" route.

## Provenance

The 2026-Q2 matrix verdict that selected this stack — including the
per-tool verdict table, the partial vindication of the earlier
"Monolith primary, AFL-guided" plan, and the original staking
roadmap/layer record this document replaced — is archived in
`examples/tezos/pbt_stack_2026Q2.md`.
