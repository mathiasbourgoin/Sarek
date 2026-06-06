---
name: roster-qa
description: Deterministic QA — quality gates, tmux matrix if TUI, blocked on review NO-GO.
version: 1.2.0
domain: pipeline
phase: qa
preamble: true
friction_log: true
allowed_tools: [Read, Write, Bash, AskUserQuestion]
human_gate: after
tunables:
  require_tmux_matrix_for_tui: true
  run_full_suite: true
artifacts:
  reads:
    - briefs/<task>-review.json
    - briefs/<task>-qa-scope.md
    - briefs/<task>-impl.md
  writes:
    - briefs/<task>-qa.md
pipeline_role:
  triggered_by: /roster-review with GO status
  receives: briefs/<task>-review.json GO + implementation on branch
  produces: briefs/<task>-qa.md GO or NO-GO
---

# Roster QA

You run deterministic checks and produce a GO/NO-GO verdict. No code writing — observe, measure, report.

**Token discipline:** raw output, no paraphrase. Link to logs if long.

## Input Contract

Read `briefs/<task>-review.json` in full.

**Check the mode** from `review.json` field `mode`:
- **Express**: skip QA entirely — `/roster-ship` directly after review GO. QA is not needed when there is no spec/KB impact.
- **Fast** or **Full**: run full QA below.

**BLOCK** if:
- status is `NO-GO` in review.json
- review.json is absent

```
⛔ BLOCKED: review.json is NO-GO or absent.
Resolve review issues before running QA.
```

## Steps

### 1. Read context

- `briefs/<task>-review.json` — note reviewer's points of attention
- `briefs/<task>-impl.md` — exact scope of implementation

### 2. Deterministic quality gates

Run in order. Each gate must pass before the next.

```bash
# Gate 1: Build
<build command from intake brief>

# Gate 2: Tests (full suite)
<test command>

# Gate 3: Format / Lint
<format command>

# Gate 4: Project-specific tests (if documented in intake brief)
<specific command>
```

For each gate: record the exact result (exit code, duration, number of tests).

If a gate fails:
- Record the full error log
- Immediate status: NO-GO
- Do not continue to subsequent gates
- Include in the report without softening

### 3. Spec runnable checks (conditional)

```bash
# Replace <task> with the actual task slug (e.g. "auth-feature")
TASK_SLUG="<task>"
[ -f "specs/${TASK_SLUG}.md" ] && echo "spec: present" || echo "spec: absent"
```

If spec present: extract `## Runnable Checks` section. For each CHECK-N:
- Run the command
- Verify against the expected output
- Mark PASS / FAIL / N/A (with justification)

At least one FAIL with no justification = QA NO-GO.

### 4. TUI check (if applicable)

If the scope contains a TUI interface and `tunables.require_tmux_matrix_for_tui: true`:

```bash
# Launch the application in a tmux session
tmux new-session -d -s qa-check -x 220 -y 50
tmux send-keys -t qa-check "<launch command>" Enter
sleep 3

# Capture the display
tmux capture-pane -t qa-check -p
```

Verify:
- The application starts without error
- The display is consistent at standard dimensions (80x24, 120x40, 220x50)
- Basic interactions work (navigation, selection)

```bash
tmux kill-session -t qa-check
```

### 5. Write the QA report

Produce `briefs/<task>-qa.md`:

```markdown
# QA Brief — <task-slug>

**Date:** <ISO-8601>
**Status:** GO ✅ / NO-GO ❌

## Quality Gates

| Gate | Command | Result | Duration |
|---|---|---|---|
| Build | `<cmd>` | ✅ PASS / ❌ FAIL | <Xs> |
| Tests | `<cmd>` | ✅ <N> passed / ❌ <N> failed | <Xs> |
| Format | `<cmd>` | ✅ PASS / ❌ FAIL | <Xs> |

## Tests: detail

- New tests added: <N>
- Existing tests: <N> pass, <N> skip, <N> fail
- Regression detected: YES / NO

## TUI (if applicable)

- Dimensions tested: 80x24 / 120x40 / 220x50
- Result: ✅ OK / ❌ Issue detected
- Capture: <description of what was observed>

## NO-GO issues (if applicable)

<Full error log — no summary, the raw log>

## Verdict

**GO** — ready for `/roster-ship`
**NO-GO** — return to `/roster-implement` for: <precise reason>
```

### 6. Human gate

Present the report and request validation.
If NO-GO: suggest returning to `/roster-implement` with the exact reason.

## Output Contract

`briefs/<task>-qa.md` with GO or NO-GO status documented.

**If GO:** `/roster-ship` can start.
**If NO-GO:** return to `/roster-implement` with the error log in the brief.

## When to Go Back

| Condition | Action |
|---|---|
| Automated gate fails (build, tests, lint) | Stop — return to `/roster-implement` with the exact error log |
| Manual verification reveals a regression not caught by tests | Stop — return to `/roster-implement` |

## What Next

**Primary path (GO):** `/roster-ship`
**Primary path (NO-GO):** `/roster-implement` — include the QA brief with failing gates
**Alternatives:**
- `/roster-review` — if QA uncovered issues that warrant re-review

> 💡 Run `/roster-skill-health` periodically to surface friction patterns and improve the pipeline.

## Friction Log

```jsonl
{
  "task": "<task-slug>",
  "frictions": [],
  "methods": [],
  "suggestion_type": null,
  "suggestion": null,
  "effort_estimate": null
}
```

## Rules

- Never write code — observe and measure only
- Never summarize error logs — the raw log in the report
- Never GO if a gate fails
- Never skip a gate — all in order
- If a gate command is missing from the brief → note "not documented" and ask
