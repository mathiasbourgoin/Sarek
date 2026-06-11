# ConvergenceSafety formal — COVERAGE INSTRUCTIONS

Task templates to strengthen coverage. Cross-references apparatus-level coverage policy at `~/.claude/skills/formal-apparatus/docs/coverage.md`.

## Coverage targets

- **Domain (eq-class) coverage**: track via `test_helpers/coverage_probe.ml`. Plateau = lock-eligibility threshold.
- **Source coverage**: track via `bisect_ppx` when feasible. Reported in `report/BENCHMARKS.md` dual-axis table.

## Standard tasks

### Adding a new eq-class

1. Identify the gap via `report/BENCHMARKS.md §coverage findings` or `findings/UNCOVERED_EDGE_CASES.md`.
2. Extend the Monolith-native generator in `test_helpers/convergence-safety_machine.ml` to produce inputs hitting the new eq-class.
3. Run mandatory **5-min smoke** via `dune exec test/...monolith_native...`. Apparatus expects this on every generator update.
4. Optionally run full duration (1h or 4h) to re-baseline.
5. Update `report/BENCHMARKS.md` dual-axis table with new eq-class count.
6. Append `history/JOURNAL.md` one-liner.

### Adding a regression test for a finding

1. Confirm finding has been classified by user (`a`, `a'`, `b`, `c`).
2. Create `test/regressions/test_convergence-safety_F<NN>_<short_name>.ml`.
3. File MUST start with comment linking to `findings/DIVERGENCE_FINDINGS.md` §F-NN.
4. File MUST exercise at least one `sarek` symbol.
5. Run; verify GREEN.
6. Update `STATUS.md` §"Test intensity" + §"Open findings".

### Running `/formal-check`

Before any project-level lock or milestone:

1. Invoke `/formal-check` in Claude Code.
2. Resolve any BLOCKERs.
3. Resolve or document any WARNINGs.
4. Verify CMBT completeness chain (7 links) — `/formal-check` step 5.

## Coverage saturation criterion

Eq-class plateau definition: three consecutive long-duration bake-off runs produce eq-class counts within ±2% of each other AND no new generator path added between runs. See apparatus `docs/coverage.md` for details.

## Project-specific notes

<!-- TODO: any project-specific coverage targets, e.g., "must cover all Tier 1-6 scenarios in UNCOVERED_EDGE_CASES.md before lock". -->
