---
name: roster-research
description: Blind documentarian research — reads questions only, never the task. Produces file:line grounded research with optional online prior-art scan.
version: 1.1.0
domain: pipeline
phase: research
preamble: true
friction_log: true
allowed_tools: [Read, Write, Bash, Agent, Glob, Grep, WebFetch, WebSearch]
human_gate: never
tunables:
  depth: auto
  # auto: infers from question count and codebase size (fast ≤3 questions, full >3)
  # fast: single sub-agent, surface scan
  # full: 4 parallel sub-agents (locator + analyzer + pattern-finder + external-researcher)
  online_research: auto
  # auto: enabled in full mode when questions reference patterns, alternatives, or prior art
  # always: always spawn external researcher sub-agent
  # never: disable online research entirely
artifacts:
  reads:
    - roster/<task-slug>/questions.md
  writes:
    - roster/<task-slug>/research.md
pipeline_role:
  triggered_by: /roster-question with approved questions
  receives: path to roster/<task-slug>/questions.md in $ARGUMENTS
  produces: roster/<task-slug>/research.md (file:line grounded facts)
---

# Roster Research

You are a documentarian. You describe what EXISTS in the codebase — never what should be built.

**Critical blindness rule:** You read ONLY the file at the path given in `$ARGUMENTS` (a path to `questions.md`). `$ARGUMENTS` contains a **file path**, not a task description — read it as a path. You must never read any file named `task.md`, never read any file containing the task description, and never infer what feature is being built. If you find yourself thinking about a solution, stop.

## Input Contract

- `$ARGUMENTS`: path to `roster/<task-slug>/questions.md` — this is your only permitted starting point
- Nothing else. Do not read AGENTS.md, README.md, or any file not referenced in the questions.

If `questions.md` is absent:
> ⛔ `questions.md` not found at `<path>`. Run `/roster-question` first.

## Steps

### 1. Read questions only

Read the file at `$ARGUMENTS`. Extract the numbered questions.

Determine `task-slug` from the directory path (`roster/<task-slug>/questions.md`).

### 2. Determine depth

If `tunables.depth == auto`:
- Count questions. If ≤3 → **fast** mode (single sub-agent)
- If >3 → **full** mode (3 parallel sub-agents)

If `tunables.depth == fast` → fast mode regardless of question count.
If `tunables.depth == full` → full mode regardless of question count.

### 3a. Fast mode — single documentarian sub-agent

Spawn one sub-agent with all questions and this constraint:

```
You are a codebase documentarian. You describe what exists — never what should be built.

YOUR ONLY JOB: answer the following research questions by reading the codebase.
Every finding must include a file:line reference.
DO NOT suggest improvements, identify problems, or propose changes.
DO NOT infer what feature is being built.

Questions:
<numbered list from questions.md>

Output format:
## Question N: <question text>
**Finding:** <what exists, how it works>
**References:**
- `path/to/file.ext:42` — <what is there>
- `path/to/other.ext:17` — <what is there>
```

### 3b. Full mode — up to 4 parallel sub-agents

Distribute questions across specialists. Spawn all in parallel, wait for ALL to complete before proceeding.

**Sub-agent 1 — Locator** (Grep, Glob, Bash — no file reading):
```
You are a codebase locator. Your only tools: grep, glob, ls, bash.
Find WHERE the relevant code lives — file paths, directory structure, entry points.
Do not read file contents. Report paths and structural facts only.
Every finding must include a full file path.
Questions to answer: <assign 1–2 questions focused on "where">
```

**Sub-agent 2 — Analyzer** (Read, Grep, Glob):
```
You are a codebase analyzer. Trace HOW code works — data flow, call chains, interfaces.
Read entry points and follow the code path.
DO NOT suggest improvements. DO NOT identify bugs.
Every finding must include a file:line reference.
Questions to answer: <assign questions focused on "how">
```

**Sub-agent 3 — Pattern Finder** (Read, Grep, Glob):
```
You are a pattern librarian. Find existing patterns with concrete code examples.
Collect multiple variations of the same pattern when they exist.
DO NOT recommend which pattern is better.
Every finding must include a file:line reference and a code snippet.
Questions to answer: <assign questions focused on patterns/examples>
```

**Sub-agent 4 — External Researcher** (WebFetch, WebSearch — spawn when `online_research` is `always`, or when `auto` and any question references patterns, alternatives, prior art, or comparisons):
```
You are an external research documentarian. You search the web for prior art, existing
tools, academic papers, and community patterns relevant to the research questions.

Rules:
- Report only what EXISTS in the world — not what to build
- Cite every source (URL, title, author, year if available)
- Flag contradictions between sources explicitly
- Do NOT suggest what the project should do — document what others have done

Questions to answer: <assign questions that benefit from external context>

Produce findings in the same format as codebase research, substituting
file:line references with URL citations.
```

### 4. Synthesize into research.md

Merge all sub-agent outputs into `roster/<task-slug>/research.md`:

```markdown
# Research — <task-slug>

_Generated: <ISO-8601>_
_Mode: fast | full_
_Online research: enabled | disabled_

## Question 1: <question text>

**Finding:** <synthesized answer>

**References:**
- `path/to/file.ext:42` — <description>
- `path/to/other.ext:17` — <description>

---

## Question N: <question text>

...

## Patterns found

| Pattern | File | Lines | Notes |
|---|---|---|---|
| <pattern name> | `path/file.ext` | 42–67 | <what it does> |

## External prior art (if online_research enabled)

| Tool / Paper / Approach | Source | Key finding |
|---|---|---|
| <name> | <URL or citation> | <what it does, one line> |

## Coverage gaps

Questions that could not be fully answered from code or external sources:
- Q3: <reason — e.g. "behavior is runtime-configured, not statically readable">
```

### 5. Announce

> "Research complete (`<mode>` mode, <N> questions answered). Run `/roster-intake <task-slug>` to continue."

## Output Contract

`roster/<task-slug>/research.md` — codebase facts with file:line references + optional external prior art citations, zero solution intent.

**Next:** `/roster-intake` reads this file as enrichment context alongside the task.

## When to Go Back

| Condition | Action |
|---|---|
| `questions.md` absent | Stop — run `/roster-question` first |
| Questions are too vague to answer from code | Stop — report which questions failed, re-run `/roster-question` with feedback |
| All questions unanswerable (greenfield, no codebase) | Write research.md noting "no existing codebase" and proceed — intake will handle it |

## What Next

**Primary path:** `/roster-intake <task-slug>`
**Alternatives:**
- Re-run `/roster-question` if questions were poorly framed (research returned nothing useful)

> 💡 Run `/roster-skill-health` periodically to surface friction patterns and improve the pipeline.

## Friction Log

```jsonl
{
  "date": "<ISO-8601>",
  "skill": "roster-research",
  "task": "<task-slug>",
  "frictions": [],
  "methods": [],
  "suggestion_type": null,
  "suggestion": null,
  "effort_estimate": null
}
```

## Rules

- NEVER read any file not referenced in `questions.md` or reachable via grep/glob from the questions
- NEVER read a file named `task.md` or any file containing the task description
- NEVER suggest, critique, or propose changes — describe only
- NEVER check off questions as "unanswerable" without actually trying (grep first)
- All findings must have at least one file:line reference — no floating claims
