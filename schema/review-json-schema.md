# `briefs/<task>-review.json` — the review verdict envelope

The contract for the review verdict. Cited as authoritative by
`schema/review-finding.schema.json`, by `roster-review` (five times), and by
`scripts/lib/review/*.js`. Until this file existed, every verdict was hand-derived from skill
prose, and the keys the convergence gate needs were routinely absent — see
[Why this document exists](#why-this-document-exists).

The nested shapes live in real JSON Schema files and are not restated here:

| Shape | File |
|---|---|
| one entry of `findings[]` / `cross_runtime_findings[]` | `schema/review-finding.schema.json` |
| one line of `briefs/<task>-review-trace.jsonl` | `schema/review-trace.schema.json` |

The envelope itself has no machine-checkable schema. `scripts/check-review-convergence.js` is its
only enforcer. It reads ten keys for its own decisions (below) and, since the fail-closed hardening
(`scripts/lib/review-gate-hardening.js`), it also *validates* the routing keys it does not act on —
`status`, `no_go_reason`, `cross_runtime_findings` — because an unvalidated routing key is a step
that silently does nothing. Everything still outside that set is enforced by prose alone.

---

## Pipeline order

**The order is load-bearing.** Fingerprints must be stable before "novel finding" means anything,
and `strike` is an *output* of the gate, so it cannot be present when the gate runs.

```
1. specialists            each emits a JSON ARRAY of findings
                          (schema/review-finding.schema.json)
        │
2. normalize              node scripts/review-normalize.js <array-file...> --ledger <prior findings>
                          --round N --cycle N --task <slug> [--gate-report] [--prior]
                          → canonical `fingerprint`, `fingerprint_v2`, `fid`; ledger carry-forward;
                            reobserved / reopened / pending-check dispositions
        │
3. assemble the verdict   node scripts/review-verdict-assemble.js --task … --round … --cycle …
                          → briefs/<task>-review.json.draft, with `strike` ABSENT
        │
4. gate                   node scripts/check-review-convergence.js <draft> --max-rounds N --strikes N
                          → JSON report on stdout; persist verbatim to
                            briefs/<task>-gate-report.json
        │
5. write strikes back     node scripts/review-verdict-assemble.js --write-strike
                          --verdict <draft> --gate-report <report>
                          → copies the gate's boolean `current_round_strike` into
                            rounds_audit[round].strike
        │
6. persist + route        rename the draft to briefs/<task>-review.json, then route on
                          `status` / `no_go_reason` (roster-review "Output Contract")
```

Two ordering traps, both of which have already cost real rounds:

- **`review-normalize.js` takes a JSON ARRAY of finding objects, never the verdict envelope.**
  Passing the envelope yields `finding file must be a JSON array` and nothing at the call site says
  which is expected. `review-verdict-assemble.js` checks this per-file and names the offending file.
- **Skipping step 2 makes step 4 unable to detect a novel finding at all.** Normalization is what
  makes fingerprints stable, and stable fingerprints are what separate a novel finding from a
  carried-forward one. Note that `first_seen_round` itself is **not** stamped by the normalizer —
  `canonicalizeFindings()` rewrites only `fingerprint`, `fingerprint_v2`, `fid` and `status`, and
  `schema/review-finding.schema.json` does not require the field — yet `isNovelStrikeFinding()`
  returns false unless it is numeric and equals the round. A HIGH+ finding that arrives without it
  is therefore permanently unstrikeable, which is why `review-verdict-assemble.js` refuses one.

---

## Order of authority

Where sources disagree, the earlier one wins for *what happens*, and the disagreement is recorded
in [Prose/enforcer discrepancies](#proseenforcer-discrepancies) rather than silently resolved.

1. **`scripts/check-review-convergence.js` and `scripts/lib/review/*.js`** — the enforcer. Its
   expectations *are* the schema for the ten keys it reads.
2. **`schema/review-finding.schema.json`, `schema/review-trace.schema.json`** — machine-checked
   nested shapes.
3. **`roster-review` prose** (`.agents/skills/roster-review/SKILL.md`, mirrored to
   `.claude/commands/roster-review.md` and `.harness/skills/roster-review.md`) — field semantics
   and intent, plus every key the gate does not read.

---

## Why this document exists

`scripts/check-review-convergence.js` used to degrade silently, not loudly, on an under-populated
verdict. Reproduced at the time on a real hand-written verdict in this repo:

```console
$ node scripts/check-review-convergence.js briefs/make-tests-actually-run-review.json --static
{
  "round": null,
  "legacy_round": true,
  "current_round_strike": null,
  "cause": null,
  "warnings": [
    "legacy review.json: no_go_round key absent — treating as round 0",
    "legacy review.json: round key absent — skipping strike and rounds_audit checks (B-8)",
    "legacy review.json: round key absent — trace checks skipped (B-8)"
  ],
  "violations": [],
  ...
}
$ echo $?
0
```

`violations: []`, exit `0`. The gate passed because it was never given anything to check, and that
outcome was byte-identical to a genuine pass everywhere a caller looks (exit code, `violations`,
`cause`).

**That is now a refusal.** The same verdict exits `2`, prints no report at all, and says why:

```console
$ node scripts/check-review-convergence.js briefs/make-tests-actually-run-review.json --static
check-review-convergence: review.json has no `round` key — the gate would skip strike
classification, the rounds_audit completeness check and every trace check, then exit 0 with
violations: [] (B-8 vacuity, schema/review-json-schema.md §round). Assemble the verdict with
scripts/review-verdict-assemble.js, or pass --allow-legacy to authorize the skip — it is then
RECORDED as legacy_skip_authorized in the report and --write-strike refuses that report.
$ echo $?
2
```

The escape hatch survives, but it is a **recorded** skip rather than a silent one: `--allow-legacy`
stamps `legacy_skip_authorized: true` into the report, and
`review-verdict-assemble.js --write-strike` refuses to journal a strike from it. An unrecorded skip
is indistinguishable from a pass; a recorded one is not. Read [`round`](#round) and
[`rounds_audit[].strike`](#rounds_auditstrike) before writing a verdict by hand — better still,
don't write one by hand.

---

## Keys the gate reads (enforced)

Ten drive decisions. **Every "Absent" and "Malformed" cell below is a refusal, not a default.** The
old behaviour is kept in the last column, because most of these were silent — and knowing which
green you used to be getting is the point of the table.

| Key | Absent | Malformed | Was (silent) |
|---|---|---|---|
| `round` | exit 2 unless `--allow-legacy` | non-number / negative / non-finite → exit 2 | skipped strike + `rounds_audit` + trace checks, warned, exit 0 |
| `no_go_round` | exit 2 unless `--allow-legacy` | non-number / negative → exit 2 | defaulted to `0`; the cap could never fire |
| `cycle` | exit 2 (non-legacy) | non-number / `< 1` → exit 2 | `null`; every trace line classified as prior-cycle |
| `findings` | `[]`, legacy-safe | present but not an array → exit 2 | already fail-closed (FIX-A/CGF-1) |
| `rounds_audit` | treated as `[]` | non-array → exit 2; a gap in rounds `2..round-1` → exit 3 | non-array read as `[]`; a gap erased the streak |
| `task` | exit 2 (non-legacy) | not a `validSlug` → exit 2 | load-bearing only on a trace-obligated round |
| `mode` | no scope-gate trace line required | outside `express\|fast\|full` → exit 2 | any other value read exactly like absent |
| `normalized_by` | exit 2 (non-legacy) | non-string / empty → exit 2 | a conditional warning |
| `cross_runtime` | no corroboration | non-object → exit 2; unknown `status` → exit 2; `digest` without `config_digest` → exit 2 | non-object ignored; statuses unvalidated |
| `streak_override` | no override | anything not `{round: <this round>, by: "human"}` → no override | unchanged |

Three more keys are **validated but not acted on** — they route, and an unvalidated routing key is a
step that silently does nothing:

| Key | Rule |
|---|---|
| `status` | outside `GO\|NO-GO` → exit 2. `GO` while the gate reports a design violation → `go-with-design-violation`, exit 1 |
| `no_go_reason` | absent on a `NO-GO` → exit 2; `type` outside the routing enum → exit 2; `cause` outside the cause enum → exit 2 |
| `cross_runtime_findings` | non-array → exit 2; entries are ratchet-checked like `findings`; a `CRITICAL`/`HIGH` `OPEN` entry not mirrored into `findings[]` → `unmirrored-cross-runtime-finding`, exit 1 |

Findings in **both** arrays must carry a `status` inside `OPEN\|RESOLVED\|ACCEPTED`; anything else is
exit 2, because the gate only ever compares against `RESOLVED` and `ACCEPTED` and a near-miss
escapes both the ratchet and the waiver.

Still unread and prose-enforced only: `date`, `summary`, `auto_fixes_applied`, `escalation_needed`,
`escalation_reason`.

---

### `round`

Integer ≥ 1 (the gate accepts `0`, but round 1 is the first round of a cycle). The **physical
per-cycle verdict counter**. Derived by the lifecycle witness, never by hand:

```bash
node scripts/lib/review/review-lifecycle.js --prior briefs/<task>-review.json   # → {round, cycle, fresh_cycle}
```

Two events only (INV-3): a persisted `GO` verdict keeps its cycle-final `round` / `rounds_audit` /
`cross_runtime` for auditability; the *next* cycle then initializes fresh — `round: 1`, full
fan-out, empty `rounds_audit`, new cross-runtime probe, `cycle + 1`.

**Absence was the vacuity hole (B-8).** With no `round` key the gate skipped strike classification,
the `rounds_audit` completeness check and all trace checks, emitted three warnings, and exited 0.
It is now **exit 2 unless `--allow-legacy`** is passed, which records the skip (see
[Gate report](#gate-report)). `scripts/review-verdict-assemble.js` still refuses to emit a verdict
without it, so the flag is for pre-schema fixtures, never for new work.

**Skipping a round was the quieter half of the same hole.** `computeStreakViolation()` walks
`round, round-1, …` and stops at the first entry that is not `true`; `computeStrikeMap()` only
knows about rounds that have a `rounds_audit` entry. Jumping from round 3 to round 7 left rounds
4–6 absent from the map and erased the streak — with **no** warning, because
`computeMissingStrikeWarnings()` only inspects entries that exist. A gap anywhere in
`2..round-1` is now a `missing-past-audit` violation (`process-incomplete`, exit 3), repairable by
journaling the missing round. The assembler additionally refuses any `--round` other than the one
the lifecycle witness derives.

Never conflate `round` with `no_go_round` — separate counters, separate reset rules.

### `no_go_round`

Integer ≥ 0. The **qualifying-only** round-cap backstop: reset to `0` on `GO`, incremented on a
finding-driven `NO-GO` outside `category: "scope"`. Compared against `--max-rounds`
(`tunables.max_no_go_rounds`, default 5) to produce the `round-cap` violation. Absent → `0` with a
warning, which means the cap can never fire; that is a second, quieter vacuity path. A counter
that is *present* but never advances is the same hole with no warning at all, so the assembler
requires each NO-GO verdict's `--no-go-round` to either hold or advance by exactly one relative to
the prior verdict.

### `cycle`

Integer ≥ 1. Scopes trace lines. Absent or non-numeric is legacy-safe (`null`) and never fatal on
its own — but on a **trace-obligated** round, `null` makes every numerically-stamped trace line
classify as prior-cycle, leaving zero current-round lines and producing a `missing-trace`
violation (exit 3). If you obligate the trace, supply `cycle`.

### `findings[]`

Cumulative across rounds within a cycle. Base shape: `schema/review-finding.schema.json`. Carry
prior entries forward **verbatim**, updating only `status` and the round-tracking fields; reset to
`[]` on `GO` after promoting red-verified checks.

Keys the gate reads from a finding: `severity`, `category`, `status`, `first_seen_round`,
`resolved_round`, `reopened_at_round`, `check`, `check_encodable`, `fingerprint`, `fid`, `path`,
`line`, `pre_fix_sha`, `red_verified`, `check_blob`.

Enforced rules, all scoped to `severity ∈ {CRITICAL, HIGH}` (`HIGH_PLUS`):

- **ratchet** — `RESOLVED` with `resolved_round > first_seen_round` and `check` null/absent →
  `resolved-without-check` / `cause: unencodable-finding`.
- **provenance** — `RESOLVED` with a missing or non-numeric `first_seen_round` or `resolved_round`
  → `missing-round-provenance` / `cause: unencodable-finding`. Missing provenance cannot be waved
  through as a same-round raise+resolve.
- **unencodable** — `check_encodable: false` and not `ACCEPTED` → `cause: unencodable-finding`.
- **strike** — counts toward this round's strike when `first_seen_round == round`, `category !=
  "scope"`, `status != "ACCEPTED"`, and not same-round-resolved; **or** when
  `reopened_at_round == round` (E-4).
- **red/green** — in full (non-`--static`) mode, every finding with a string `check` is executed:
  red against `pre_fix_sha` in a `git archive` scratch tree, green on the current tree. `check`
  must be a **node-runnable file path** (`node <path>`); a spec-level `CHECK-N` id with no file is
  recorded but not executed. `pre_fix_sha: null` (dirty tree) → flagged, `red_verified` stays
  `null`, not a violation.

`ACCEPTED` is a permanent waiver — state it that way to the human, never "skip for now".

### `rounds_audit[]`

Append-only, one entry per round **including `GO` rounds**, retained on `GO`. Prior entries are
never rewritten (`review-verdict-assemble.js` refuses).

```json
{
  "round": 2,
  "reviewed_sha": "17e9bbc7…",
  "fix_sha": "55148469…",
  "fix_sha_reason": "dirty-tree",
  "specialists_run": [{ "name": "reviewer", "selection_reason": "always (owner), loop-back round 2" }],
  "strike": false,
  "trace_schema_version": "1.0"
}
```

| Field | Requirement (enforced from `round` ≥ 2) |
|---|---|
| `round` | number; the entry is found by `round === review.round` |
| `reviewed_sha` | must be **defined** (any value); absent → `process-incomplete`, exit 3 |
| `fix_sha` | must be **defined**; `null` is allowed only with a non-empty `fix_sha_reason` (EC-8, dirty tree) |
| `fix_sha_reason` | required non-empty string when `fix_sha` is `null` |
| `specialists_run` | non-empty array; **every** element needs a non-empty `selection_reason` |
| `strike` | boolean — see below. Not part of the completeness check |
| `trace_schema_version` | `"1.0"` on new rounds. **Presence obligates the trace checks** |

On `round == 1` none of this is enforced (`computeMissingAuditViolation` returns early) — a missing
round-1 entry is a warning in prose, not a violation. Write the entry anyway.

`trace_schema_version` is a *commitment*: stamping it obligates the round to have a matching
`briefs/<task>-review-trace.jsonl` with ≥1 line for `(task, cycle, round)`, plus one line per
claimed invocation. Omit it only for a round that genuinely predates the trace mechanism. Never
fabricate a trace line for a tool that did not run (FR-177/C-3) — running the missed tool is the
only repair. Because omitting the stamp *removes* the trace obligation rather than failing it,
`review-verdict-assemble.js` stamps `"1.0"` unconditionally and rejects any other
`--trace-schema-version`: a round it assembles is a round that ran now, so it always commits.

#### `rounds_audit[].strike`

**Boolean. Never `null`, never absent once the round has been gated.** This is the single most
dangerous field in the envelope, and a non-boolean is now **exit 2**.

`computeStrikeMap()` reads past strikes with
`if (typeof entry.strike === "boolean") strikeByRound.set(entry.round, entry.strike)`. A `null`
therefore never lands in the map, `Map.get()` returns `undefined`, and
`computeStreakViolation()`'s `strikeByRound.get(r) !== true` test **silently reset the streak**.

Verified behaviour, on a five-round verdict where every round carries a novel HIGH finding —
before, and after:

| Verdict | Was | Now |
|---|---|---|
| `strike` boolean throughout | exit **1**, `cause: novel-finding-streak`, firing at **round 3** not round 5 | unchanged |
| `strike: null` on round 4 only | exit 1, `cause: round-cap`; the streak gone; one warning | **exit 2**, no report, `rounds_audit entry for round 4 has strike null` |
| `strike: null` on every round, `no_go_round: 2` | exit **0**, `violations: []`, `cause: null` — total escape, three warnings | **exit 2**, no report |

The third row was the real-world failure: five rounds, every round striking,
`current_round_strike: true` in the report, and the gate still reporting no violation. A live
example of the shape that produced it is `briefs/f16-dsl-slice1-review.json` (`round: 3`,
`strike: null` on all three entries) — it now fails the gate, and the repair is to re-gate each
round and journal its strike.

The two sharp edges that made it so quiet, both now closed:

- A `null` on the **current** round's entry produced **no warning at all** —
  `computeMissingStrikeWarnings()` skips entries with `round >= currentRound`, and the current
  round's strike is recomputed fresh anyway. It became a silent streak reset one round later, once
  that round was a past round. The boolean check covers the current round's entry too, whenever
  `strike` is *present*; **absent** on the round being gated is still correct and required.
- The only signal for a past-round `null` was a warning inside a JSON report, on a run whose exit
  code was often `0`.

**Therefore: `strike` is written by the gate, not the author.** Assemble the draft with the key
ABSENT, run the gate, then copy the report's boolean `current_round_strike` into the entry:

```bash
node scripts/review-verdict-assemble.js --write-strike \
  --verdict briefs/<task>-review.json.draft --gate-report briefs/<task>-gate-report.json
```

That command refuses any non-boolean `current_round_strike` — which is also how it catches a gate
report produced from a verdict that had no `round` (the gate reports `null` there). It additionally
refuses a report whose `round` does not match the verdict's, and one missing `config.strikes` or
`trace`: the gate emits both on every exit code it reports on (see §"Gate report") precisely so a
caller can detect a stale gate script, and a boolean strike computed under superseded rules is
indistinguishable from a real one once journaled.

Round 1 never strikes. Escalation fires when the last `--strikes` consecutive rounds (all ≥ 2) each
struck; a strike-free round resets the streak, and non-consecutive strikes never accumulate.

### `task`

The task slug, validated by `validSlug()` from `scripts/lib/xruntime/xruntime-journal.js`. Used to
derive two sibling paths from `path.dirname(<review.json path>)` — never by suffix-stripping:

- `briefs/<task>-review-trace.jsonl`
- `briefs/<task>-xruntime.jsonl`

**Load-bearing, and the failure is a hard exit 2.** On a round with a `round` key, an invalid or
absent `task` fails closed *if* the round is trace-obligated — either by
`trace_schema_version` on the current entry, or by *any* `*-review-trace.jsonl` already existing in
the review directory (A-2: blanking `task` must not disguise an obligated round as a legacy skip).
With neither prong, an absent `task` falls through to the B-8 skip so the ~40 pre-existing
round-based fixtures keep passing.

### `mode`

`"express" | "fast" | "full"`, read from `briefs/<task>-impl.md`. The gate reads it for exactly one
thing: `mode === "full"` requires a `scope-gate` trace line on the round. Any other value, including
absent, requires nothing — a typo (`"Full"`) reads to the gate exactly like an omission, so the
assembler rejects anything outside the three-value enum instead of passing it through.

> **Name collision.** The gate's own *report* also has a `mode`, whose values are
> `"static" | "full"` (whether `--static` was passed). Same key name, different enum, and `"full"`
> means different things in the two documents. Do not copy one into the other.

### `normalized_by`

The `normalizer_version` string from `scripts/review-normalize.js` (currently `"2.0.0"`). Any
truthy value **obligates a `normalizer` trace line** for the round — the normalizer self-appends it
when given `--task`, `--round` and `--cycle` together (all three, or it appends nothing). A
non-legacy round with neither a trace obligation nor `normalized_by` triggers the loud
"omit-everything posture" warning (FR-180) — `computeOmitEverythingWarning()` returns null as soon
as *either* is present, so it is not an absent-`normalized_by` detector on its own. An
assembler-produced round always stamps `trace_schema_version` and is therefore always obligated,
which means it can never raise this warning; it is a signal about hand-written verdicts.

### `cross_runtime`

An object keyed by runtime name (`codex`, `opencode`, …), each value a state object:

```json
{ "codex": { "status": "healthy", "reason": null, "config_digest": "…", "round": 2, "ts": "…", "actor": "…" } }
```

Prose statuses: `healthy`, `degraded`, `skipped-degraded`, `skipped-human`, `blocked`. The gate
handles a strict subset:

- **warning** — a `status: "degraded"` entry missing `reason` or missing `config_digest`. Nothing
  else is warned about, and no status value is validated.
- **corroboration** — for `status ∈ {healthy, degraded, skipped-human}` **and**
  `entry.round === review.round`, there must be a line in `briefs/<task>-xruntime.jsonl` with a
  matching `runtime`, `cycle`, and `digest`. No match → `unattested-invocation`, exit 1.
  `skipped-degraded` and `blocked` are never corroborated.

- **`fault` (SPOC-local, backlog #102)** — present on `status: "degraded"` entries and on the
  corresponding journal line. `"runtime"` means the runtime or its environment failed (never
  started, never finished, mutated the tree, or answered with nothing). `"caller"` means the
  runtime ran and answered but the answer did not satisfy the output contract — a prompt defect,
  not a runtime defect.

  Only `fault: "runtime"` may arm the cross-runtime circuit breaker. A degraded entry with
  `fault: "caller"` never suppresses a later probe at the same digest; the fix is to append
  `node scripts/xruntime-review.js --emit-contract` to the probe prompt and re-probe. An entry
  written before this field existed carries no `fault` key and is read as `"runtime"`, so the
  change never silently un-arms a degradation already on disk.

> **Key asymmetry, easy to get wrong:** the verdict writes `config_digest`; the journal writes
> `digest`. Same value, different key. The gate matches `cross_runtime[rt].config_digest` against
> `journalEntry.digest` — never `config_digest` against `config_digest`.

Cross-runtime runs are **never** listed in `specialists_run` — the gate corroborates them through
this field and the `cross-runtime` trace event instead.

> **Assembler gap (known, not fixed here).** `review-verdict-assemble.js` can only *inherit*
> `cross_runtime` from the prior verdict — `{}` on a fresh cycle — because it has no flag for
> probe state. A round that really ran a `codex`/`opencode` probe cannot record it without editing
> the verdict by hand, which is the hand-derivation this tool exists to remove, on the one field
> that drives `unattested-invocation` (exit 1). Adding a validated probe-state input belongs with
> the D1 work, not here.

### `streak_override`

```json
{ "round": 3, "by": "human" }
```

Valid only when `round === review.round` **and** `by === "human"`. A valid override forces this
round's `strike` to `false` instead of recomputing it, so a later `--static` re-check still passes.
It is current-round-only — the next verdict's `round` increment retires it, and a new streak must
fully re-accumulate. **The round-cap escalation is never overridable.** Offer the override only
when the gate's `cause` is `novel-finding-streak`, never for `round-cap`, and never for
`unattested-invocation` (INV-8).

---

## Keys the gate does not read (prose-enforced)

| Key | Type | Notes |
|---|---|---|
| `task` | string | also in the enforced set — listed there |
| `date` | string | full ISO-8601 (`2026-07-25T10:00:22Z`), not a day-only string |
| `status` | `"GO" \| "NO-GO"` | **the gate never reads this.** `scripts/lib/review/review-lifecycle.js` does: `status === "GO"` is what makes the next cycle fresh |
| `summary` | string | human-facing |
| `auto_fixes_applied` | integer | count of step-2 mechanical corrections |
| `no_go_reason` | object \| null | see below — the routing key, and entirely unenforced |
| `cross_runtime_findings` | array | canonical finding shape, augment-only, never merged into `findings` and never rewritten after intake. A CRITICAL/HIGH OPEN entry here is supposed to force `NO-GO`, but the gate never looks at this array |
| `escalation_needed` | boolean | Express/Fast mode-escalation signal; informational, never blocks `GO` |
| `escalation_reason` | string \| null | required in spirit when `escalation_needed` is true; nothing checks it |

### `no_go_reason`

`null` on `GO`. On `NO-GO`:

```json
{ "type": "design-not-converging", "cause": "novel-finding-streak", "failed_acs": ["AC-3", "FR-021"] }
```

- **`type`** — the routing key. Values recognized by `roster-review`'s Output Contract and
  `roster-run`'s routing table:

  | `type` | Route |
  |---|---|
  | `out-of-scope-change` | `/roster-implement` (a human `ACCEPT` on the scope finding is the escape hatch) |
  | `spec-ac-failure` | `/roster-spec` — populate `failed_acs` from each finding's `acs` |
  | `cross-runtime-finding` | `/roster-implement`; mirror the entry into `findings` so it enters the ratchet |
  | `design-not-converging` | `/roster-spec`, minimal-freeze profile |
  | `review-integrity-failure` | re-run the claimed tooling for real — **never** `/roster-spec`, never streak-override (FR-175/INV-8) |

- **`cause`** — mirrors the gate report's top-level `cause` when the gate drove the `NO-GO`:
  `unencodable-finding`, `unattested-invocation`, `novel-finding-streak`, `round-cap`.
  `process-incomplete` is **never** a top-level `cause` and never routes.
- **`failed_acs`** — `AC-N` / `FR-NNN` for a `specs/<task-slug>.md` contract, `S<N>` claim ids for
  `kb/spec.md`.

Nothing validates either enum. A typo in `type` routes nowhere; `roster-run` reads it with
`jq -r '.no_go_reason.type // "none"'` and falls through.

---

## Gate report

Persisted verbatim to `briefs/<task>-gate-report.json` each round. Emitted on every exit code the
gate *reports* on — 0, 1, 3, the legacy skip, and the inconclusive-red/green flavour of 2.

> **Not on an input-rejection exit 2.** `main()` prints the report only after `validateArgsAndReview()`
> and the `trace.fail` check have passed, so an absent/malformed verdict, an unknown flag, or an
> invalid `task` slug on a trace-obligated round produces **no report at all**:
> ```
> $ node scripts/check-review-convergence.js briefs/nope.json --static 2>/dev/null | wc -c
> 0
> ```
> That matters because `--write-strike` now hard-depends on a report. There is nothing to write back
> from on those paths — fix the input and re-gate; do not hand-write a report to satisfy the tool.

```json
{
  "mode": "static | full",
  "no_go_round": 2, "max_rounds": 5, "legacy_no_go_round": false,
  "round": 3, "legacy_round": false,
  "current_round_strike": true,
  "legacy_skip_authorized": false,
  "config": { "max_rounds": 5, "strikes": 2, "static": true },
  "hardening": { "version": "1.0.0" },
  "cause": "novel-finding-streak",
  "warnings": [], "violations": [], "checks": [],
  "trace": { "obligated": true, "lines_seen": 9, "schema_version": "1.0", "skipped": false }
}
```

Before trusting any other field, confirm **all three** of `config.strikes`, `trace` and
`hardening.version` are present. Any one absent means a stale gate script: do not persist, surface
"gate script out of date", stop.

### `legacy_skip_authorized` — the recorded skip

`true` when `--allow-legacy` suppressed the fail-closed refusal for an absent `round` or an absent
`no_go_round`. It means: *this run was authorized to skip checks it could not perform, and is not
evidence that they passed.* `review-verdict-assemble.js --write-strike` refuses such a report
outright — journaling a strike from a gate run that skipped strike classification would convert a
recorded skip straight back into an indistinguishable pass.

Pass the flag only for a genuinely pre-schema fixture. For anything else, the fix is to assemble
the verdict rather than to authorize the skip.

### Exit codes

| Exit | Meaning | Action |
|---|---|---|
| 0 | no violations, no degraded input | proceed |
| 1 | design violation — `cause` is `unencodable-finding` / `unattested-invocation` / `novel-finding-streak` / `round-cap` | `NO-GO` regardless of the human verdict; route per `cause` |
| 2 | degraded input: absent/malformed verdict, unknown flag, inconclusive red/green, malformed current-cycle trace line, **or any of the fail-closed envelope rules above** | fail closed; block the route-back, surface to the human. **No report is emitted** — there is nothing to persist |
| 3 | `process-incomplete` only — incomplete/absent `rounds_audit` entry for the current round, a **gap** in rounds `2..round-1`, or `missing-trace` | repair per `violations[].detail`, re-gate, max 2 attempts, never bump `round`, never route |

Two violation types are newer than the four causes above and reuse them:

| Type | Cause | Exit |
|---|---|---|
| `unmirrored-cross-runtime-finding` | `unencodable-finding` | 1 |
| `go-with-design-violation` | inherited from the first design violation found | 1 |
| `missing-past-audit` | `process-incomplete` | 3 |

`cause` precedence (FR-059/B-5, extended FR-174):

```
unencodable-finding  >  unattested-invocation  >  novel-finding-streak  >  round-cap
```

**This precedence surprises people.** On a verdict that violates both the streak rule and the
round cap, `cause` is `novel-finding-streak` and *not* `round-cap`, even though `round-cap` is in
`violations[]`. In practice a genuinely non-converging review escalates at **round 3** on the
streak, so `round-cap` is only ever the top-level cause when the streak has been broken — which,
given the previous section, most often means a `strike: null`.

---

## Prose/enforcer discrepancies — resolved

Each row was a place where `roster-review` prose implied an enforcement that did not exist, or
where the enforcer required something the prose never stated. **All eleven are now enforced**, by
`scripts/lib/review-gate-hardening.js`, and each has a proof-of-rejection case in
`scripts/check-review-convergence-hardening.test.js` — the gate has been watched failing on every
one of them.

| # | Discrepancy | Was (silent) | Now |
|---|---|---|---|
| D1 | Prose: "any `cross_runtime_findings` entry that is CRITICAL or HIGH (OPEN) sets `status: NO-GO`". The gate never read the array. | A HIGH cross-runtime finding was invisible; only the prose-level mirror into `findings` put it under the ratchet, and forgetting the mirror was unenforced. | The array is ratchet-checked like `findings`, and a `CRITICAL`/`HIGH` `OPEN` entry with no mirror is `unmirrored-cross-runtime-finding` (exit 1). |
| D2 | Prose treats `no_go_reason.type` as the routing key; the gate never read `no_go_reason` or `status`. | A typo routed nowhere. Caught only for assembler-produced verdicts. | `type`/`cause` enums and the `status` enum are validated (exit 2); a `GO` asserted over a design violation is `go-with-design-violation` (exit 1). |
| D3 | Prose says `strike` is "populated after the gate reports it" but never that it must be a **boolean**. | `strike: null` passed the completeness check and defeated streak escalation. Live here: `briefs/f16-dsl-slice1-review.json`. | Non-boolean on any gated round → exit 2. Absent on the round being gated is still correct. |
| D4 | Prose lists `task` as an ordinary field; the gate made it load-bearing only once the round was trace-obligated. | On any other round, a blanked `task` fell through to the B-8 skip. | Required and `validSlug`-checked on every non-legacy round (exit 2). |
| D5 | `mode` means `express\|fast\|full` in the verdict and `static\|full` in the gate report. | Copying one into the other silently changed what the gate demanded. | Outside the verdict enum → exit 2, with the collision named in the message. |
| D6 | Prose does not state that stamping `normalized_by` creates a `normalizer` trace obligation, nor that the normalizer self-appends only when `--task`, `--round` **and** `--cycle` are all given. | Absent `normalized_by` was a conditional warning, on a verdict whose fingerprints were therefore unstable. | Absent/empty on a non-legacy round → exit 2: strike classification on an unnormalized verdict is not a result. |
| D7 | `findings[].status` is `OPEN\|RESOLVED\|ACCEPTED` per `schema/review-finding.schema.json`; the gate accepted any string (it only tests `=== "RESOLVED"` / `=== "ACCEPTED"`). | `status: "OPEN-FOR-HUMAN"` (live here) matched neither branch. Enforcement depended on which tool you ran. | Enum-checked in **both** finding arrays → exit 2. |
| D8 | Prose describes the ratchet's exemptions but not the FIX-2 hardening: *missing* round provenance on a `RESOLVED` HIGH+ finding is itself an `unencodable-finding` violation. | — | **Not a hole**: the enforcer is *stricter* than the prose, so nothing escapes. Resolved by this documentation only. |
| D9 | Prose says `cross_runtime` entries persist `config_digest`; the journal writes the same value under `digest`. | Writing the journal's key into the verdict broke corroboration silently; no status value was validated at all. | `digest` without `config_digest` → exit 2 with the asymmetry explained; unknown `status` → exit 2. |
| D10 | Prose does not mention that an absent `no_go_round` defaults to `0` with a warning. | The round cap could never fire — a second, quieter vacuity path alongside `round`. | Exit 2 unless `--allow-legacy` records the skip. |
| D11 | Prose does not mention that an absent/non-numeric `cycle` filters every numerically-stamped trace line out as prior-cycle. | A trace-obligated round with no `cycle` failed as `missing-trace` even with the lines on disk. | Absent or non-numeric/`< 1` on a non-legacy round → exit 2. |

Three further holes, found auditing the same file and not previously recorded:

| # | Hole | Now |
|---|---|---|
| G3 | An absent `round` skipped strike, `rounds_audit` and trace checks and exited 0. | Exit 2 unless `--allow-legacy` records the skip. |
| G5 | A **gap** in `rounds_audit` (e.g. round 3 → round 7) left the skipped rounds out of the strike map and erased the streak, with no warning, because `computeMissingStrikeWarnings()` only inspects entries that exist. | `missing-past-audit` over `2..round-1` → exit 3. |
| G13 | `rounds_audit` and `cross_runtime` were coerced (`Array.isArray(x) ? x : []`, `typeof x === "object" \|\| return`), so a wrong type emptied the check instead of failing it. | Present-but-wrong-type → exit 2. |

Two more notes on the enforcer itself, neither a discrepancy:

- In `computeFindingViolations()`, `const accepted = f.status === "ACCEPTED"` is then tested as
  `if (f.status === "RESOLVED" && !accepted)`. `!accepted` is necessarily true inside that branch —
  redundant, harmless, and it does match the prose (an `ACCEPTED` finding is never ratchet-checked).
- `check_encodable: false` is only a violation on `CRITICAL`/`HIGH`. On a `MEDIUM` it is ignored.

---

## Distribution

`scripts/check-review-convergence.js`, `scripts/review-normalize.js`, `scripts/lib/review/*`,
`scripts/lib/xruntime/*`, `schema/review-finding.schema.json` and
`schema/review-trace.schema.json` are **upstream-owned bundle files**, sentinelled by
`scripts/review-bundle.manifest.json` and verified by `node scripts/review-bundle-verify.js`. Do
not hand-edit them; see `scripts/REVIEW-BUNDLE.md`.

This document, `scripts/review-verdict-assemble.js`, `scripts/check-mandated-steps.js`, their rule
layers `scripts/lib/review-verdict-rules.js`, `scripts/lib/review-gate-hardening.js` and
`scripts/lib/mandated-steps-rules.js`, and the three tests are **repo-local** and deliberately
absent from the manifest, so a bundle upgrade neither overwrites nor flags them. Note the path: the
rule layers sit directly in `scripts/lib/`, *not* in the bundle-owned `scripts/lib/review/`.

`check-review-convergence.js` requires `scripts/lib/review-gate-hardening.js`. That is the one
place a bundle file depends on a repo-local one, and it is deliberate: the hardening rules are this
repo's contract with the gate, and keeping them out of the bundle means a bundle upgrade cannot
silently restore any of the eleven holes.

The checks run standalone, no test framework. Each constructs the input that should be rejected and
asserts the refusal:

```bash
node scripts/review-verdict-assemble.test.js              # assembler ↔ real gate, red-on-mutation
node scripts/check-review-convergence-hardening.test.js   # D1..D11 + G3/G5/G13 proof-of-rejection
node scripts/check-mandated-steps.test.js                 # recorded-skip ledger
node scripts/review-bundle-verify.js                      # bundle integrity, no network
```

They are deliberately not wired into `make test-all`: that target runs the GPU suites, and this
contract must stay checkable on a machine with no GPU.

> **Known gap:** the bundle files themselves are untracked in this repository — present in the
> working tree, absent from git. `review-verdict-assemble.js` requires
> `scripts/lib/review/review-lifecycle.js` and shells out to `scripts/review-normalize.js`, so a
> fresh clone gets this document and the assembler but not their dependencies, and must run the
> bundle installer first. Committing 22 upstream-owned generated files is a separate decision from
> shipping this contract.

---

## Minimal valid verdict

Round 1, `GO`, nothing outstanding — the smallest envelope that makes the gate run its real checks
rather than take the legacy skip. Produce it with the assembler, not by hand.

```json
{
  "task": "example-task",
  "date": "2026-07-25T10:00:22.000Z",
  "status": "GO",
  "mode": "full",
  "round": 1,
  "cycle": 1,
  "no_go_round": 0,
  "auto_fixes_applied": 0,
  "findings": [],
  "cross_runtime_findings": [],
  "cross_runtime": {},
  "rounds_audit": [
    {
      "round": 1,
      "reviewed_sha": "64f5d605a4952cc4036e5cff5d4437cdd17e465f",
      "fix_sha": "64f5d605a4952cc4036e5cff5d4437cdd17e465f",
      "specialists_run": [{ "name": "reviewer", "selection_reason": "always (owner), round 1 full fan-out" }],
      "trace_schema_version": "1.0",
      "strike": false
    }
  ],
  "no_go_reason": null,
  "summary": "No findings.",
  "escalation_needed": false,
  "escalation_reason": null,
  "normalized_by": "2.0.0"
}
```

With `trace_schema_version: "1.0"` and `mode: "full"` this round is trace-obligated, so
`briefs/example-task-review-trace.jsonl` must carry, for `(task, cycle 1, round 1)`, at least a
`scope-gate` line, a `specialist` line with `actor: "reviewer"`, and — because `normalized_by` is
set — a `normalizer` line.
