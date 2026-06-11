# ConvergenceSafety — Benchmarks

Dual-axis coverage table per apparatus `policy/BENCHMARK_POLICY.md`.

## Dual-axis coverage table

Last updated: 2026-06-11

| Axis | Metric | Value | Notes |
|---|---|---|---|
| Domain (eq-class) | Eq-classes observed | (not yet measured) | Run `coverage_probe` to baseline |
| Source (bisect_ppx) | Line coverage % | N/A | `bisect_blockages: none` — run when needed |

## Test intensity

| Suite | Tests | Samples per test | Total samples |
|---|---|---|---|
| `test_convergence_conformance` | 10 QCheck2 | 1000–2000 | ~16 000 |
| `test_convergence_extraction` | 6 QCheck2 | 1500–2000 | ~10 500 |

## Monolith-native bake-off

Per `policy/BENCHMARK_POLICY.md §S2 Level B`:

- Last run: (none yet)
- Eq-classes found: —
- Source coverage %: —
- Stale relative to last generator update: —

## Tool choice rationale

QCheck2 chosen over Monolith-native for the quick tier: the abstract `expr`
type has no stateful transitions, so the stateful long-run Monolith pattern
does not apply. The extraction conformance test (`test_convergence_extraction`)
substitutes for the Monolith-native long tier: it validates the extracted OCaml
against the inline model for 6 properties × 1500–2000 samples.

## Coverage findings

| Finding | Status |
|---|---|
| UC-01: TESuperstep diverged outer | Gap — see `findings/UNCOVERED_EDGE_CASES.md` |
| UC-02: nested diverged EIf | Gap |
