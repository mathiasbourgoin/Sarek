---
name: roster-question
description: Decompose a task into neutral research questions — blind research prep, task intent not revealed.
version: 1.0.0
domain: pipeline
phase: question
preamble: true
friction_log: true
allowed_tools: [Read, Write, Agent, AskUserQuestion, Bash]
human_gate: after
artifacts:
  reads:
    - AGENTS.md
    - README.md
  writes:
    - roster/<task-slug>/questions.md
pipeline_role:
  triggered_by: /roster-run (always, as first step)
  receives: task description in $ARGUMENTS
  produces: roster/<task-slug>/questions.md (neutral questions, task intent hidden)
---

# Roster Question

You decompose a task into neutral research questions. Your output will be handed to a blind researcher who must never know what is being built — only what exists.

**Critical principle:** `questions.md` must contain zero information about what to build, what to add, or what to change. Questions describe what IS, not what SHOULD BE.

## Input Contract

- `$ARGUMENTS`: task description (any length)
- `AGENTS.md`, `README.md` for project context

## Steps

### 1. Read context

Read `AGENTS.md` and `README.md` silently. Do not read the codebase yet.

### 2. Derive the task slug

Kebab-case, max 4 words from $ARGUMENTS. Example: "add webhook retry logic" → `webhook-retry-logic`.

### 3. Spawn question-generation sub-agent

Spawn a sub-agent with this exact prompt:

```
You are a research planner. You will receive a task description.
Your job: produce 3–7 neutral research questions about the EXISTING codebase.

Rules:
- Questions must describe what EXISTS — never what to BUILD
- No question may reveal the feature or change being requested
- Each question must be answerable by reading code (grep, glob, read)
- Questions must be specific enough to direct a code reader to the right areas

Good: "How does the middleware chain handle request authentication, and where are auth policies defined?"
Bad: "What's the best way to add a new authenticated endpoint?"

Good: "Where are retry mechanisms currently implemented, and what interfaces do they use?"
Bad: "How should we implement webhook retry logic?"

Task description (DO NOT include this in the output):
<$ARGUMENTS>

Produce: a numbered list of 3–7 neutral research questions only.
No introduction, no conclusion, no mention of what is being built.
```

### 4. Create output directory and write questions.md

```bash
mkdir -p roster/<task-slug>
```

Write `roster/<task-slug>/questions.md`:

```markdown
# Research Questions — <task-slug>

_Generated: <ISO-8601>_
_DO NOT include the task description in this file or share it with the researcher._

1. <neutral question>
2. <neutral question>
3. <neutral question>
...
```

**Do not include the task description in this file.**

### 5. Human review gate

Present the questions to the user:

> "Research questions ready for `<task-slug>`. Review before I hand them to the researcher:
>
> <list questions>
>
> Approve, edit, or ask me to regenerate?"

Apply any corrections. Wait for explicit approval before proceeding.

### 6. Announce next step

> "Questions approved. Run `/roster-research roster/<task-slug>/questions.md` to continue."

## Output Contract

`roster/<task-slug>/questions.md` — neutral questions only, no task intent, human-approved.

**Next:** `/roster-research` reads this file as its only input.

## When to Go Back

| Condition | Action |
|---|---|
| Task too vague to form any question | Stop — ask the user to clarify the task before proceeding |
| Questions keep revealing the solution intent | Regenerate with stricter prompt; if still failing, reformulate the task with the user |

## What Next

**Primary path:** `/roster-research roster/<task-slug>/questions.md`
**Alternatives:**
- Skip research and go directly to `/roster-intake` — only for trivial single-file tasks with no codebase exploration needed

> 💡 Run `/roster-skill-health` periodically to surface friction patterns and improve the pipeline.

## Friction Log

```jsonl
{
  "date": "<ISO-8601>",
  "skill": "roster-question",
  "task": "<task-slug>",
  "frictions": [],
  "methods": [],
  "suggestion_type": null,
  "suggestion": null,
  "effort_estimate": null
}
```

## Rules

- Never include the task description or solution intent in `questions.md`
- Never skip the human review gate — questions shape the quality of all downstream research
- If the task is in a domain with no existing codebase (greenfield), note this and generate architectural questions about conventions and tooling instead
