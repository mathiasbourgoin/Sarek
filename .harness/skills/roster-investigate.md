---
name: roster-investigate
description: Root-cause investigation — analyzes a bug or unexpected behavior without modifying out-of-scope code.
version: 1.2.0
domain: pipeline
phase: null
preamble: true
friction_log: true
allowed_tools: [Read, Bash, AskUserQuestion]
human_gate: before
tunables:
  auto_freeze_scope: true
  max_hypothesis: 5
artifacts:
  reads: []
  writes:
    - briefs/<task>-investigation.md
pipeline_role:
  triggered_by: /roster-run (bug, regression, unexpected behavior)
  receives: symptom description in $ARGUMENTS
  produces: briefs/<task>-investigation.md with root cause and fix plan
---

# Roster Investigate

You analyze a bug or unexpected behavior. Your job is to **understand**, not to fix.
No code modification without an explicit human gate.

**Fundamental rule:** never fix without a complete investigation. A fix without a root cause is debt disguised as a solution.

## Input Contract

`$ARGUMENTS`: description of the observed symptom (can be short).

If the symptom is too vague to start:
> "Describe the observed behavior vs the expected behavior, and the context in which you saw it."

## Steps

### 1. Gate before — freeze scope

If `tunables.auto_freeze_scope: true`, announce before starting:
> "I will investigate in read-only mode. I will not modify any file without explicitly asking you. Investigation scope: [what is relevant from the description]."

Wait for confirmation before starting.

### 2. Understand the symptom

- Restate the symptom in precise terms:
  - Observed behavior
  - Expected behavior
  - Reproduction conditions (always / sometimes / once)
  - Context (environment, data, state)
- Identify the module / file / function most likely involved

### 3. Reproduce (if possible)

```bash
# Attempt to reproduce the symptom
<reproduction command if known>
```

If not reproducible → note and continue with static analysis.

When the symptom involves state or accounting, anchor on **ground-truth state** (the authoritative store, DB, or service), not on logs or intermediate reports — those record what was *attempted*, not the durable result, and can disagree with reality. Before theorizing, find the **empirical discriminator**: the smallest observable that separates a failing case from a passing sibling. A reproduction that looks correct may be the *wrong scenario* (wrong trigger, missing setup), not the absence of the bug.

### 4. Formulate hypotheses

Formulate up to `tunables.max_hypothesis` root cause hypotheses, ordered by probability.

For each hypothesis:
```
H1: <description>
  Probability: high / medium / low
  Evidence: <what supports this hypothesis in the code>
  Test: <how to confirm or refute>
```

### 5. Test hypotheses (read-only)

For each hypothesis, in probability order:
- Read the relevant code
- Trace the execution flow
- Look for proof or refutation

```bash
# Read-only tools
git log --oneline -20 -- <file>
git blame <file>
grep -n "<pattern>" <file>
```

Distinguish **code-confirmed** (the code provably supports the hypothesis when read) from **observed** (the symptom reproduced live). If the symptom is reproducible, do not stop at code-confirmed — confirm by observation: code-reading can be right about the mechanism yet wrong about the runtime effect.

### 6. Identify the root cause

State the root cause precisely:
```
Root cause: <description>
Evidence: <file:line — exact quote>
Introduced: <commit or date if traceable>
Impact scope: <what is affected>
```

If multiple hypotheses remain open → list them with their confidence level.

### 7. Propose a fix plan

Without touching code:
```
Fix plan:
1. <step 1 — affected file>
2. <step 2>

Fix risks:
- <what could regress>

Tests to add:
- <test that would have caught this bug>
```

### 8. Write the report

Produce `briefs/<task>-investigation.md`:

```markdown
# Investigation — <task-slug>

**Date:** <ISO-8601>
**Symptom:** <precise restatement>
**Status:** ROOT CAUSE IDENTIFIED / HYPOTHESES IN PROGRESS

## Root Cause

<precise description>
**Evidence:** `<file>:<line>` — `<exact quote>`
**Introduced:** <commit or "undetermined">

## Tested hypotheses

| # | Hypothesis | Result | Evidence |
|---|---|---|---|
| H1 | ... | CONFIRMED / REFUTED | `file:line` |

## Fix plan

<proposed steps>

## Tests to add

<tests that would have caught this bug>

## Impact scope

<what is affected — modules, users, data>
```

Present the report and ask:
> "Root cause identified. Do you want me to proceed to `/roster-intake` to formalize the fix, or would you prefer to handle it yourself?"

## Output Contract

`briefs/<task>-investigation.md` with documented root cause or hypotheses in progress.

**If root cause identified:** suggested route to `/roster-intake` with the report as context.

**Persist what was learned (if a KB exists).** Before considering the investigation closed, fold the durable facts into the KB (`/kb-update` or hand them to `kb-agent`): the confirmed root cause **and** the hypotheses that were ruled out — so the next investigator does not re-walk the same dead-ends. A refutation is knowledge worth keeping, not just a discarded branch.

## When to Go Back

| Condition | Action |
|---|---|
| Root cause cannot be determined from code alone | Stop — report hypotheses to human, ask for more context or logs |
| Investigation reveals the bug is in an external dependency | Stop — report findings, do not attempt a fix in this run |

## What Next

**Primary path:** `/roster-intake` — formalize the fix plan using the investigation report as context
**Alternatives:**
- `/roster-plan` — if root cause and fix are unambiguous and intake is not needed

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

- Never modify code without an explicit human gate
- Never propose a fix without an identified root cause
- Every causal claim must cite the file and line
- "Looks like" is not a root cause — confirm or refute
- If reproducible: reproduce before analyzing statically
- Code-confirmed is not observed: if the symptom is reproducible, confirm by observation, not by reading alone
- Anchor causal claims on ground-truth state, not on logs or intermediate reports
- Find the empirical discriminator before theorizing; a clean reproduction may be the wrong scenario, not the absence of the bug
- An investigation is not closed until its result — including ruled-out hypotheses — is folded into the KB (when one exists)
