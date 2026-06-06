---
name: roster-plan
description: Dual-voice decomposition — reads the intake brief, produces per-role sub-briefs.
version: 1.2.0
domain: pipeline
phase: plan
preamble: true
friction_log: true
allowed_tools: [Read, Write, Agent, AskUserQuestion]
human_gate: after
artifacts:
  reads:
    - briefs/<task>-intake.md
  writes:
    - briefs/<task>-plan.md
    - briefs/<task>-implementer.md
    - briefs/<task>-reviewer.md
    - briefs/<task>-qa-scope.md
pipeline_role:
  triggered_by: /roster-intake with validated brief
  receives: briefs/<task>-intake.md (single source of truth)
  produces: per-role sub-briefs + sequenced plan
---

# Roster Plan

You decompose a validated brief into executable sub-briefs. You have no research context — the brief is your single source of truth. What the brief does not say does not exist for you.

**Token discipline:** precise decomposition. No invention beyond the brief.

## Input Contract

Read `briefs/<task>-intake.md` **in full** before doing anything.

If the brief is absent or does not have VALIDATED status:
> ⛔ Intake brief absent or not validated. Re-run `/roster-intake` first.

If required sections are missing (Goal, Scope Boundary, Relevant Files, Quality Gates):
> ⛔ Incomplete brief — missing section(s): <list>. Complete the brief before planning.

## Steps

### 0. KB ambiguity pre-check (conditional)

```bash
[ -d kb ] && ([ -f kb/spec.md ] || [ -f kb/index.md ]) && echo "KB present" || echo "KB absent"
```

If KB is **present**:
→ Invoke `skills/kb/ambiguity-auditor.md` on the KB.
→ If the audit returns **Critical** findings: present them to the human and ask:
  - "These KB contradictions may corrupt the plan. Fix KB first, or continue knowing these risks?"
  - If "fix first": STOP — return to user.
  - If "continue": annotate the plan's "Identified risks" table with each contradiction.
→ If only Warnings/Info: log them in the plan's "Identified risks" table. Continue.
→ If no findings: continue silently.

If KB is **absent**: skip silently.

### 1. Read the brief

Read `briefs/<task>-intake.md` in its entirety. Read nothing else.

Extract:
- The goal and its scope boundary
- The files involved
- The exact quality gates
- Any unresolved open questions

### 2. Dual-voice: two independent analyses

Run **sequentially** two independent analyses of the plan.

#### Voice 1 — Claude sub-agent (fresh context)

Spawn a sub-agent with this exact prompt (do not inject the current conversation context):

```
You are a software architect. You are given a task brief.
You must propose a decomposition plan in sequential steps.

Be adversarial: look for unverified assumptions, hidden dependencies,
implementation risks, and edge cases not covered by the brief.
Do not compliment the brief — find its flaws.

Brief:
<full content of briefs/<task>-intake.md>

Produce:
1. Sequenced plan (numbered steps with dependencies)
2. Assumptions you had to make (what the brief does not clearly state)
3. Identified risks
4. Questions you would ask before starting
```

#### Voice 2 — Second model or adversarial fallback

**If a second model (codex, o3, etc.) is available:**
→ Run the same analysis via that model.

**If not available or on error:**
→ Spawn a second Claude sub-agent with this prompt (different — more adversarial):

```
You are a skeptical senior engineer. You are asked to challenge an implementation plan.
Your role: find why this plan will fail.

Starting assumption: the plan is too optimistic.
Questions to ask yourself:
- What is not said in the brief but will cause problems?
- Which dependencies will break?
- Where is the real risk (not the apparent risk)?
- What will take 3x longer than expected?

Brief:
<full content of briefs/<task>-intake.md>

Do not propose an alternative plan — only give argued objections.
```

### 3. Consensus table

Build a synthesis table:

```markdown
## Consensus Table

| Point | Voice 1 | Voice 2 | Status |
|---|---|---|---|
| Step 1: <description> | ✅ | ✅ | AGREE |
| Risk: <description> | ⚠️ | ✅ | AGREE |
| Approach for X | Option A | Option B | DISAGREE |
| Direction on Y | Keep | Change | USER-CHALLENGE |

Statuses:
- AGREE: both voices converge → auto-decided
- DISAGREE: disagreement on an approach → present both options to the human
- USER-CHALLENGE: both voices think the brief's direction should change → NEVER auto-decided
```

**USER-CHALLENGE rule:** if both analyses agree to change a direction from the brief:
- Clearly present the recommendation
- Explain why both analyses converge
- State what context might be missing
- Ask — never act

### 4. Resolve DISAGREE items

For each DISAGREE, present both options to the human with:
- Voice 1's position (and why)
- Voice 2's position (and why)
- Recommendation if one option is clearly better

Wait for the decision before continuing.

### 5. Write the plan

Produce `briefs/<task>-plan.md`:

```markdown
# Plan — <task-slug>

**Date:** <ISO-8601>
**Status:** DRAFT

## Sequential steps

1. **<step>** — <description, files involved, completion criterion>
2. **<step>** — ...

## Dependencies

<Step N must precede step M because ...>

## Identified risks

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|

## Decisions made

| Point | Decision | Reason |
|---|---|---|

## Assumptions

<What was assumed because the brief did not specify it>
```

### 6. Write sub-briefs

Produce one sub-brief per execution role:

**`briefs/<task>-implementer.md`** — for `/roster-implement`:
- Goal, scope boundary, files to modify with snippets
- Sequential steps from the plan
- Exact quality gates
- Points of attention from voices (risks, assumptions)

**`briefs/<task>-reviewer.md`** — for `/roster-review`:
- What was implemented (summary from the plan)
- Files to audit first
- Identified risks to verify
- Expected behaviors to confirm

**`briefs/<task>-qa-scope.md` (brief, not the report)** — for `/roster-qa`:
- Exact quality gates with commands
- Behaviors to validate
- If TUI scope: scenarios to test in tmux matrix

### 7. Human validation quiz

Before presenting the sub-briefs, ask 3 consistency questions:

1. "The plan sequences steps in this order: [list]. Is the order correct?"
2. "The identified risks are: [list]. Are there other important ones?"
3. "The implementer sub-brief covers [scope]. Does that match what you want in this iteration?"

Wait for answers before finalizing the sub-briefs.

### 8. Final human gate

Present the sub-briefs with their paths. Request validation before spawning execution agents.

Set `**Status:** VALIDATED` in each sub-brief after approval.

## Output Contract

- `briefs/<task>-plan.md` (VALIDATED)
- `briefs/<task>-implementer.md` (VALIDATED)
- `briefs/<task>-reviewer.md` (VALIDATED)

**Next:** `/roster-implement` reads `briefs/<task>-implementer.md`.

## When to Go Back

| Condition | Action |
|---|---|
| Brief has unresolvable ambiguity or missing required sections | Stop — re-run `/roster-intake` to complete the brief |
| Both voices agree the brief's direction should change | Stop — present to user, re-run `/roster-intake` if approved |
| KB has Critical contradictions and user chose to fix KB first | Stop — fix KB with `/ambiguity-auditor`, then re-run `/roster-plan` |

## What Next

**Primary path:** `/roster-implement` (reads `briefs/<task>-implementer.md`)
**Alternatives:**
- Re-run `/roster-intake` if the brief was incomplete

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

- The brief is the single source of truth — do not read the codebase
- USER-CHALLENGE is never auto-decided — always present to the human
- Do not spawn execution agents — produce sub-briefs only
- If a plan step is not covered by the brief → note as assumption, do not invent
- Sub-briefs must be self-contained: the receiving agent cannot assume access to the current context
