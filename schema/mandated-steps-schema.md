# `briefs/<task>-steps.jsonl` — the mandated-step ledger

The contract for recording that a mandated pipeline step ran, or that it was skipped and why.
Enforced by `scripts/check-mandated-steps.js`; rules in `scripts/lib/mandated-steps-rules.js`;
proof-of-rejection cases in `scripts/check-mandated-steps.test.js`.

## Why this exists

A mandated step that is skipped without leaving a record is **indistinguishable from one that ran
and passed**. Nothing in the artifacts separates the two, so the skip reads as a pass forever after.

Three real instances, all from the same friction log:

- `roster-run` Step 4 mandates `/roster-doctor preflight` before any phase that builds or tests. It
  was not run across an entire day of implementation work — and the environment had three real
  problems it would have surfaced: a local `main` three commits stale, a wedged Docker daemon, and
  a partially-upgraded opam switch breaking a bytecode link. Nothing recorded that it had not run.
- The `roster-ship` human gate was skipped under a standing autonomy delegation. Legitimate — but
  it entered the record as though a human had answered, erasing exactly the distinction the gate
  exists to preserve.
- The cross-runtime pass, mandatory on every Fast/Full PR, was disabled by its own circuit breaker
  on a caller-side fault; the round then looked like one on which cross-runtime simply had nothing
  to say.

This is the same defect as the review gate's silent degradation
(`schema/review-json-schema.md` §"Why this document exists"), one level up: a green that means
"not checked", not "checked out".

**Skipping is allowed. Skipping silently is not.**

## Record shape

One JSON object per line. Append with the tool — never hand-write the file:

```bash
node scripts/check-mandated-steps.js --record --task <slug> \
  --step preflight --outcome ran --result READY --actor agent

node scripts/check-mandated-steps.js --record --task <slug> \
  --step human-gate --outcome skipped --actor agent \
  --reason "standing autonomy delegation for session 2026-07-25; user AFK"
```

| Field | Rule |
|---|---|
| `ts` | full ISO-8601, stamped by the writer |
| `task` | the task slug; a record stamped with another task is rejected |
| `step` | one of the known ids below. An **unknown id is rejected** — a typo can never stand in for the step it resembles |
| `outcome` | `ran` \| `skipped` |
| `actor` | `human` \| `agent` — a skip must be attributable |
| `result` | required when `outcome: ran`: `READY` \| `NOT-READY` \| `PASS` \| `FAIL` \| `N/A` |
| `reason` | required when `outcome: skipped`, and when `result` is `NOT-READY` or `FAIL` |
| `phase` | optional, informational |

`NOT-READY` is a first-class recordable result: a preflight that ran and found the environment
broken is evidence, and must not be indistinguishable from one that found it healthy.

### Known steps

| Step | Meaning |
|---|---|
| `preflight` | `/roster-doctor preflight` — required before any phase that builds or tests |
| `human-gate` | the human validation quiz (`rules/human-validation.md`). **Human-only** |
| `xruntime` | the cross-runtime review pass (mandatory on Fast/Full PRs) |
| `scope-gate` | `check-scope-diff.sh` — the out-of-scope-change gate |
| `degraded-specialist` | a conditional specialist selected but unable to run. Repeatable |

**`human-gate` may not be recorded as `ran` by `actor: "agent"`.** An agent proceeding under a
standing delegation records `outcome: "skipped"`, `actor: "agent"`, with the delegation as its
reason — so an agent's decision is never journaled as a human's.

## Checking

```bash
node scripts/check-mandated-steps.js --task <slug> --phase implement
```

| Phase | Mandates |
|---|---|
| `plan` | `human-gate` |
| `implement` | `preflight` |
| `review` | — (see below) |
| `qa` | `preflight` |
| `ship` | `preflight`, `human-gate` |
| `intake` `question` `research` `spec` | — (records are still validated) |

### Why `review` mandates nothing — measured, not assumed

The first draft mandated `scope-gate` and `xruntime` on `review`. Measuring against the real
artifacts in `briefs/` before shipping killed that:

| Signal | Count |
|---|---|
| verdict files | 36 |
| journaled `rounds_audit` rounds | 30 |
| review trace files | 16 |
| `scope-gate` trace events | 9 |
| `cross-runtime` trace events | **1** |
| verdicts recording `cross_runtime` | 3 (0 with `status: "healthy"`) |

The requirement would have fired on nearly every round. Under this ledger a `skipped` record
discharges it, so it would never have been *unsatisfiable* — but a requirement discharged by
boilerplate nine rounds out of ten is an alarm people learn to paste past, which is the same
failure as one they learn to ignore.

The decisive reason is the second one: **both obligations already have an authoritative enforcer.**
`review-trace-rules.js` requires a `scope-gate` trace line whenever `mode === "full"`
(`computeRequiredCoverage`) and corroborates every claimed probe against the xruntime journal
(`computeCrossRuntimeCorroboration`, exit 1 `unattested-invocation`). Mandating them here as well
would be a second, weaker record of a fact that already has a stronger one — and two records of the
same fact drift apart.

Both remain **recordable**, opt-in via `--require scope-gate,xruntime`. `degraded-specialist` is
recordable but never required: a step that may legitimately never occur cannot have its absence
read as a skip.

(Method borrowed from PR #305's F9, which measured a proposed phase-order rule against 34 real
ledgers, found it flagged 7 legitimately, and shipped it as a documented limitation rather than an
alarm.)

`--require <step,step>` overrides the table. An **unknown phase is rejected**, never defaulted to
"nothing required" — that default would itself be a gate that cannot fail, since a typo'd phase
would demand nothing.

| Exit | Meaning |
|---|---|
| 0 | every mandated step has exactly one valid record |
| 1 | a mandated step has no record, or a record is invalid — the gate's "no" |
| 2 | usage error, unknown phase/step, or an unreadable/malformed ledger — fail closed |

An **absent ledger is exit 1** with every requirement listed, never exit 0: "no file" is the
strongest possible form of "nothing was recorded". A malformed line is exit 2, never a silently
dropped record — dropping it would reintroduce the defect on the checker's own side.

Deliberately **not** enforced: whether a `reason` is a *good* reason. That is a human judgement.
What is enforced is that one exists, that it is attributable, and that an agent's decision is never
recorded as a human's.
