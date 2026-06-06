---
name: improvement-loop-planner
description: Propose bounded self-improvement loops from KB, code, tests, issues, and CI signals.
version: 1.1.0
domain: workflow
phase: null
preamble: true
allowed_tools: [Read, Bash, AskUserQuestion]
human_gate: after
pipeline_role:
  triggered_by: human (when improvement targets are unclear)
  receives: $ARGUMENTS — area or project to analyze
  produces: loop spec(s) approved by human → passed to improvement-loop
  pairs_with: improvement-loop
---

# Improvement Loop Planner

**Pair:** this skill proposes loops; `/improvement-loop` executes them. Run this first when you don't have a loop spec yet — once the human approves a proposal, pass it as `$ARGUMENTS` to `/improvement-loop`.

Propose a small set of high-value, bounded improvement loops for the project or area described in $ARGUMENTS.

This skill is **KB-aware but not KB-dependent**:

- If `kb/` exists, use it as the highest-priority source of intent and constraints
- If no KB exists, fall back to repository evidence: tests, CI, issues, TODOs, docs, and code structure

Your job is to **discover and define** candidate loops, not to run them.

## Steps

### 1. Read Intent And Constraints

- Read `AGENTS.md`, `README.md`, `CLAUDE.md`, and any architecture/spec docs that exist
- If `kb/` exists, read at least:
  - `kb/spec.md` if present
  - `kb/properties.md` if present
  - `kb/architecture.md` if present
  - `kb/index.md` if present
- Extract:
  - desired behavior
  - non-negotiable constraints
  - forbidden areas
  - quality properties

### 2. Inspect Mechanical Signals

Prioritize evidence that can support a deterministic loop:

- failing tests
- flaky tests
- lint/type/build failures
- CI failures
- coverage gaps
- performance hotspots with measurable baselines
- issue backlog with concrete acceptance criteria
- repeated TODO/FIXME clusters
- code areas that appear inconsistent with KB or repo docs

If tools exist, use them. Examples:

- `gh issue list`
- `gh run view --log-failed`
- project test command
- lint/typecheck/build commands

### 3. Identify Candidate Loop Targets

A good loop target has all of these:

- narrow writable scope
- measurable success signal
- low to moderate blast radius
- repeatable verification command
- clear keep/discard decision

Bad loop targets include:

- vague “improve architecture”
- unbounded refactors
- subjective UI polish with no acceptance criteria
- changes that require irreversible side effects

## Proposal Format

Propose **1 to 5** loops, ordered by expected value and safety.

For each loop, use exactly this structure:

```markdown
## Loop <N> — <short name>

- Objective: <what this loop is trying to improve>
- Why now: <evidence from KB, tests, issues, CI, or code>
- Confidence: high | medium | low
- Writable scope: <specific files, directories, or globs>
- Read-only context: <files/docs/tests/issues to consult but not modify>
- Metric: <single primary metric or binary pass condition>
- Verify: `<command>`
- Guard: `<command or none>`
- Max iterations: <integer between 3 and 5; pick lower for high-risk scopes>
- Risk: low | medium | high
- Keep rule: <when a change is kept>
- Discard rule: <when a change is reverted or abandoned>
- KB basis: <spec/properties/architecture refs, or “none”>
```

After the proposals, add:

```markdown
## Recommendation

- Best starting loop: <Loop N>
- Why: <why this one is the best first candidate>
- Missing setup: <anything the user should define before execution, or “none”>

## Tool Opportunities

For each loop proposed above, identify patterns that could become deterministic tools
instead of LLM judgment. Optional section — only include if a genuine opportunity exists.

Format:
```
[TOOL] <tool description> — replaces: <the LLM judgment or manual step it eliminates>
       Trigger: <when this tool would run — CI, pre-commit, post-edit>
       Output: <what it produces — exit code, report, annotation>
```

Examples:
- [TOOL] Custom linter rule for missing auth guards — replaces: reviewer manually checking auth on each new endpoint
- [TOOL] Schema diff checker — replaces: LLM comparing API responses to spec definitions
```

## Rules

- Do **not** start editing code as part of this skill
- Do **not** propose an unbounded loop by default; bounded loops only
- Prefer deterministic metrics over subjective judgment
- If no trustworthy verification signal exists, say so explicitly and refuse to propose an execution-ready loop
- If KB exists and conflicts with issues or code, note the contradiction instead of papering over it
- If confidence is low because KB is absent, say that explicitly

## When to Go Back

| Condition | Action |
|---|---|
| No measurable verification signal exists for any candidate | Stop — do not propose a loop; report to human |
| KB contradicts the proposed improvement area | Stop — surface the contradiction, do not paper over it |

## What Next

**Primary path:** pass the approved loop spec as `$ARGUMENTS` to `/improvement-loop`.
