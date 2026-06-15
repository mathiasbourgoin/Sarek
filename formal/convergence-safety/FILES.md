# ConvergenceSafety formal — FILES

Per-file inventory of the project. Update at every structural change.

## Directory layout

```
convergence-safety/
├── AGENT_PREAMBLE.md
├── ASSUMPTIONS.md                # abstract↔real correspondence; elided constructors
├── COVERAGE_INSTRUCTIONS.md
├── ConvergenceSafetySpec.tex     # LaTeX source of truth (LOCKED v0.1)
├── FILES.md
├── METHODOLOGY.md
├── STATUS.md
├── _CoqProject                   # Rocq build config
├── proof-ledger.json             # machine-readable proof state
├── theories/
│   └── ConvergenceSpec.v         # Rocq spec — 11 theorems, 0 admits, 0 axioms
├── extraction/
│   ├── ConvergenceSafetyExtraction.v  # extraction config
│   ├── ConvergenceModel.ml            # extracted OCaml (committed; auto-generated)
│   ├── ConvergenceModel.mli           # extracted interface (committed)
│   └── dune                           # library stanza for convergence_model
├── test/
│   ├── dune
│   ├── test_convergence_conformance.ml   # 10 QCheck2 properties on inline model
│   └── test_convergence_extraction.ml    # 6 QCheck2 properties: extracted vs inline
├── test_helpers/
│   └── coverage_probe.ml         # eq-class probe instrument
├── policy/                       # 5 apparatus policies (copied from skill/policy/)
│   ├── BENCHMARK_POLICY.md
│   ├── CONFORMANCE_POLICY.md
│   ├── DIVERGENCE_POLICY.md
│   ├── PBT_STACK.md
│   └── SESSION_POLICY.md
├── findings/
│   ├── DIVERGENCE_FINDINGS.md    # F-01: TESuperstep mode-reset (open)
│   └── UNCOVERED_EDGE_CASES.md
├── report/
│   ├── BENCHMARKS.md
│   └── REPORT.md
└── history/
    └── JOURNAL.md
```

## Rocq theories

| File | Lines | Purpose |
|---|---|---|
| `theories/ConvergenceSpec.v` | 350 | Abstract `expr` type, 6 functions, 11 theorems, `expr_list_rect` induction principle |

## OCaml test code

| File | Lines | Purpose |
|---|---|---|
| `test_helpers/coverage_probe.ml` | 40 | Hashtbl-backed eq-class probe |
| `test/test_convergence_conformance.ml` | 240 | PBT — 10 QCheck2 properties on inline abstract model |
| `test/test_convergence_extraction.ml` | 180 | PBT — 6 properties: extracted `ConvergenceModel` vs inline |
| `extraction/ConvergenceModel.ml` | 136 | Rocq-extracted OCaml (do not edit by hand) |

## Policies

5 policies live under `policy/` as stamped copies of apparatus `policy/`
(BENCHMARK / CONFORMANCE / DIVERGENCE / SESSION / PBT\_STACK).
