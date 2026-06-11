# policy/BENCHMARK_POLICY.md — Level B bakeoff discipline

**Standing policy** (2026-Q2; generic core extracted 2026-06-11,
apparatus v1.1 — locked semantics unchanged).

**Scope.** Multi-duration bakeoff that compares PBT tools at the
project's stateful conformance workload. Enforces identical
budgets, canonical binaries, dual-axis coverage, scenario sampling,
and a ledger-backed audit trail so benchmark rows can't be
fabricated, misattributed, or silently truncated.

Every rule below is a **hard requirement**, not a "best effort" — the
runner script MUST refuse to produce a final table when any gate
fails.

The canonical instantiation (the staking 2026-Q2 matrix: 6 tools ×
4 durations + a 7th-tool generator-fairness extension) is archived in
`examples/tezos/bakeoff_2026Q2.md`; it is the worked example for every
section below.

## 1. Canonical matrix

The project defines its bakeoff as a **canonical matrix** — N tools ×
M durations at a fixed per-scenario step budget — in a runner script
checked into `scripts/`.  Requirements:

- **Identical workload and budget per row.** Every tool drives the
  same stateful machine with the same step budget; only the generator
  / driver differs.  (Per `memory/feedback_benchmark_uniform_workload.md`:
  a new tool joins with identical workload/budget/artifact format —
  tool-specific specialisation goes in a driver wrapper, counted
  toward the ergonomics axis.)
- **Per-duration batching** (all rows at the shortest duration → all
  rows at the next → …) so any partial result is a coherent slice.
- **Single quiescent host** (no parallel builds), one seed file,
  recorded in the ledger.
- **Methodology disclosures in the write-up.** Any row whose name
  could be read as measuring something it doesn't (e.g. a library
  linked but not driving generation, a fork-exec vs persistent
  execution model, an env-var-controlled variant) MUST carry a
  disclosure paragraph in `report/BENCHMARKS.md`.  The staking
  matrix's four disclosures are the model (see the archive).
- **Variant rows differ by declared env only.** When two rows are
  variants of one binary, the difference is controlled ONLY by a
  documented environment variable — no code revert, no duplicate
  binary.

## 2. Artifacts every row must produce

Under `coverage/<bakeoff-id>/row_{tool}_{budget_s}s/`:

- `runner_stats.json` — `{tool, iter, time_s, scen_per_s, divergences, eq_classes, step_budget}`
  (random tools) or AFL's `fuzzer_stats` file copied here (AFL rows).
- `bisect_*.coverage` files — written by `bisect_ppx` instrumentation
  of the target modules in `{protocol_src}`.
- `bisect_report.txt` — output of
  `bisect-ppx-report summary --per-file --coverage-path …` plus the
  global `Coverage: X/Y (Z%)` line. Branch coverage if supported by
  the installed bisect_ppx; line coverage at minimum.
- `bisect_audit_modules.txt` — coverage filtered to the project's
  audit-walked module list (declared in the project's audit doc; the
  staking 10-module list is in the archive).
- `scenarios_sample.json` — first 4 scenarios with
  `{iter_idx, codes, steps}`.
- `scenarios_extremes.json` — longest scenario (by step count) and
  slowest scenario (by wall-clock duration_us), each with full
  `{iter_idx, length, duration_us, codes, steps}` reproducer. Random
  drivers update in-process via `SCENARIO_EXTREMES_FILE`; AFL rows
  derive these post-run by replaying the corpus.
- `coverage_curve.tsv` — every 30 s, a sample of
  `(elapsed_s, bisect_global_pct, bisect_audit_pct, edges_found)`.
  Used for the time-to-N% coverage analysis and the convergence
  epoch metric.
- `op_mix_histogram.json` — frequency of each step constructor
  across all sampled scenarios.
- `binary.md5`, `source.md5` — md5 of the exe and the source file(s)
  used for this row.
- `cmdline.txt`, `env.txt` — exact invocation and environment
  captured at start.

AFL-driven rows additionally produce `corpus/` (copy of AFL's queue
directory at end of run) and `divergence_N.json` for each crash
discovered (full repro: scenario codes + spec trace + target trace +
diff).

## 3. Ledger — `coverage/<bakeoff-id>/ledger.jsonl`

One JSON line per row. Written BEFORE the run starts (`status: started`)
and updated at completion (`status: completed` /
`preflight_failed` / `inflight_failed` / `postflight_failed`).
Schema (field values illustrative):

```json
{
  "row_idx": 1,
  "tool": "qcheck",
  "duration_s": 300,
  "model_revision": "0f7c26d63b…",
  "binary": "/path/stateful_qcheck.exe",
  "binary_md5": "…",
  "source_md5": "…",
  "switch": "<opam switch name>",
  "cmdline": "…",
  "env": {…},
  "seed_file_md5": "…",
  "start_time": 1776990000,
  "end_time": 1776990300,
  "measured_duration_s": 298.4,
  "iter": 1493,
  "execs_per_sec": 5.0,
  "eq_classes": 130,
  "divergences": 0,
  "afl_stability_pct": null,
  "afl_crashes": null,
  "src_cvg_global_pct": 12.55,
  "src_cvg_audit_pct": 38.12,
  "branch_cvg_audit_pct": 31.4,
  "longest_scenario": {"iter_idx": 87, "length": 50, "duration_us": 712014, ...},
  "slowest_scenario": {"iter_idx": 142, "length": 50, "duration_us": 1842977, ...},
  "time_to_first_divergence_s": null,
  "convergence_epoch_s": 280,
  "generator_wastage_rate": 0.16,
  "rss_peak_mb": 412,
  "artifact_dir": "coverage/<bakeoff-id>/row_qcheck_300s/",
  "status": "completed"
}
```

For AFL rows, `afl_stability_pct` and `afl_crashes` are populated
from afl-fuzz's `fuzzer_stats`. For random rows, both are `null`.

The `report/BENCHMARKS.md` table is generated FROM the ledger by `jq`.
The agent does not type benchmark numbers by hand. If a field is
missing or invalid, the corresponding cell is `—` or `INCOMPLETE`,
never extrapolated or back-computed from a shorter run.

## 4. Preflight gates (per row, before the run starts)

1. Canonical binary exists at the expected path.
2. The workload source file(s) and driver source exist.
3. Binary md5 + source md5 captured into the ledger row.
4. `model_revision` captured (git rev-parse HEAD).
5. Switch name captured (`opam switch show`).
6. Env vars echoed into `env.txt`. Per-row required / forbidden env
   vars, as declared in the matrix definition, are checked here (a
   variant row's controlling variable must appear; it must NOT appear
   in the baseline row).
7. `SCENARIOS_SAMPLE_FILE`, `SCENARIO_EXTREMES_FILE` set to per-row
   artifact paths.

Any failure → ledger status `preflight_failed`; row skipped; runner
continues with the next row but marks the cell `INCOMPLETE (preflight)`.

## 5. Inflight gates (during the run)

- Background watcher polls the runner PID every 60 s. If the PID
  dies before the run has seen at least 0.95 × `duration_s` of
  elapsed wall clock, the row is marked `inflight_failed`; runner
  moves on.
- AFL rows additionally require `fuzzer_stats:run_time` to be
  monotonically increasing; a frozen stat is treated as death.
- Coverage curve sampling: every 30 s, the watcher snapshots
  bisect counters and AFL `edges_found` to `coverage_curve.tsv`.

## 6. Postflight gates (per row, before writing anything)

Hard checks; all must pass for `status: completed`:

1. **Duration.** `measured_duration_s ≥ 0.95 × duration_s`.
2. **Artifacts.** All files in §2 exist and are non-empty.
   `scenarios_sample.json` must contain ≥ 4 scenarios.
   `scenarios_extremes.json` must have both `longest` and `slowest`
   non-null.
3. **Coverage report.** `bisect_report.txt` parses to a non-empty
   per-file table, and the global `Coverage:` line is present.
4. **AFL stability (AFL rows only).** `afl_stability_pct ≥ 80`.
   Below 80% → `postflight_failed` with `reason: afl_stability_low`;
   row's cell marked accordingly. (Fork-exec drivers measure ≈ 98%;
   see the archive for the stability methodology.)

A row that fails any postflight gate is written to the ledger as
`postflight_failed` with an explicit `reason` field. The cell in
`report/BENCHMARKS.md` is written as `INCOMPLETE (postflight:<reason>)`
— never silently replaced by a shorter-window value.

## 7. Ledger access path — surfaced at every checkpoint

Each runner output line MUST contain the ledger path and the row
offset, e.g.:

```
[row 7/24 qcheck 900s] starting; ledger row written → coverage/<bakeoff-id>/ledger.jsonl#L7
[row 7/24 qcheck 900s] duration=897s eq_classes=412 src_cvg=18.21% audit_cvg=51.7% OK
             ledger=coverage/<bakeoff-id>/ledger.jsonl#L7
             artifacts=coverage/<bakeoff-id>/row_qcheck_900s/
```

Final summary MUST include the ledger full path, a file-tree listing
of `coverage/<bakeoff-id>/`, and per-row artifact paths so every
number in the final table can be re-derived by `jq` / `cat` on the
artifacts — independently of any prose.

## 8. "Task done" rule

The N-row matrix is `completed` if and only if **N of N rows have
`status: completed`** in the ledger with all postflight gates passed.
Partial matrices (e.g. N−1 / N) are reported as such, leaving the
task `in_progress`, awaiting a decision. The agent does not decide to
ship partial data unilaterally.

For the per-duration batching, partial completion is reported per
duration band (e.g. "all rows at 300 s + 900 s complete; 1800 s +
3600 s pending").

## 9. Prohibited fabrications

Anti-patterns explicitly forbidden:

- Copying a value from a shorter-duration row (e.g. 900 s) into a
  longer-duration row (e.g. 3600 s).
- Estimating / extrapolating any field.
- Running a row without the canonical env (e.g. claiming a variant
  row without its controlling env var set).
- Re-using an artifact directory from a previous run.
- Writing the final table before the ledger is complete.
- Including a superseded driver binary as a canonical row (a
  superseded driver may remain in the codebase as a code archive but
  is NOT a matrix row; the staking example is in the archive).

If any of these would be necessary to "finish" the task, the
correct action is to stop and report the blocker, not to fudge.

## 10. Generator-update smoke run

When the project's long-tier (Monolith-native) generator is extended —
a new operation, a new eq-class target, a distribution change — a
**5-minute smoke run is mandatory** before the change is considered
landed: it validates that the extended generator still runs the full
workload and actually reaches the eq-classes it was extended to hit
(`docs/coverage.md` invokes this rule when coverage findings surface
unhit eq-classes).  A full matrix refresh is NOT required per
extension; bakeoff staleness relative to the last generator update is
tracked by `/formal-check` rule 6.

## 11. Related policies

- `policy/CONFORMANCE_POLICY.md §"Dual-axis coverage"` — why we always
  report both domain-space (`coverage_probe.ml`) and source-code
  (`bisect_ppx`) axes.
- `policy/DIVERGENCE_POLICY.md §The atomicity rule` — benchmark
  write-up commits as atomic units (the staking matrix used three:
  prior-artifact wipe, AFL stability mitigation, matrix + analysis +
  writeup).
- `policy/PBT_STACK.md` — which tools sit in which tier; the locked
  stack the matrix verdict produced.
- `AGENT_PREAMBLE.md` — formal-agent workflow.
