---
name: roster-spec
description: Adversarial spec phase — derives user stories with concrete GWT scenarios, surfaces challenges, formalizes FR-NNN requirements, produces structured contract with runnable checks.
version: 2.0.0
domain: pipeline
phase: spec
preamble: true
friction_log: true
allowed_tools: [Read, Write, Bash, Agent, AskUserQuestion, WebFetch]
human_gate: after
tunables:
  min_user_stories: 2
  min_gwtscenarios_per_story: 3
  min_challenges_per_story: 1
  max_questions_to_user: 5
  require_runnable_checks: true
artifacts:
  reads:
    - briefs/<task>-intake.md
    - specs/*.md (existing specs for consistency check)
    - kb/ (if present)
  writes:
    - specs/<task-slug>.md
    - briefs/<task>-spec.md
pipeline_role:
  triggered_by: roster-run (feature/api-change tasks only)
  receives: briefs/<task>-intake.md (VALIDATED)
  produces: specs/<task-slug>.md + briefs/<task>-spec.md
---

# Roster Spec

You produce a structured contract for the feature described in the intake brief.
This is not a transcription phase — it is an adversarial process. Your job is to
find what is wrong with the requirements before any implementation begins.

**Anti-sycophancy rule:** Never validate a requirement just because it was stated.
Challenge every assumption. Surface every gap. The spec is only complete when
challenges are resolved, not when requirements are listed.

## Trigger Check

Read `briefs/<task>-intake.md`. Find the `**Type:**` line.

- If Type is `fix`, `chore`, `docs`, or `refactor`:
  > ℹ️ Spec phase skipped for `<type>` tasks. Writing completion marker.
  Write `briefs/<task>-spec.md` with `**Status:** SKIPPED — type: <type>`.
  Stop.

- If Type is `feature` or `api-change`: continue.

- If `**Type:**` line is missing:
  > ⛔ Intake brief has no Type field. Re-run `/roster-intake` and ensure
  > the Type field is set before proceeding.
  Stop.

## Idempotency Check

If `specs/<task-slug>.md` already exists:
> "A spec already exists for `<task-slug>`. Overwrite, review, or skip?"
Wait for user decision. If skip: write completion marker and stop.

## Steps

### 1. Research Sub-Agent

Spawn a sub-agent (fresh context) with this prompt:

```
You are a codebase researcher. You will be given a task brief.
Your job: read the actual codebase and surface everything relevant to this feature.

Do NOT suggest what to build. Only document what EXISTS.

Find and report (with file:line citations):
1. All existing code paths touched by or relevant to this feature
2. Similar features already implemented — what patterns do they use?
3. Existing tests that cover adjacent behavior
4. Any existing specs in specs/ that reference the same entities or feature area
5. KB entries (kb/) relevant to this feature's domain
6. Potential conflicts: code that assumes the opposite of what this feature proposes

Brief:
<full content of briefs/<task>-intake.md>
```

Read the research output. Use it to pre-populate the spec's context — do not re-investigate
what the research covers.

### 2. Clarification Elicitor

Spawn a sub-agent with this prompt:

```
You are a requirements clarifier. You receive a task brief and codebase research.
Your job: surface and resolve every material ambiguity before story drafting begins.

For each ambiguity you find:
1. State the question precisely.
2. Answer it from the brief or research if the answer is already knowable.
3. If NOT answerable from context, mark it [OPEN].

Produce a Q&A table of 4–8 items. Cover:
- Scope boundaries that are unclear (what is explicitly in vs out)
- Behavioral edge cases the brief does not address
- Naming or definition inconsistencies between the brief and existing code
- Constraints implied by the architecture but not stated in the brief

Format:
| Q | A |
|---|---|
| <precise question> | <answer from context, or "[OPEN]"> |

Brief:
<full content of briefs/<task>-intake.md>

Research:
<research sub-agent output summary>
```

Collect the Q&A table. Track `questions_asked_step2` = number of [OPEN] items resolved
by asking the user. Resolve all [OPEN] items before proceeding to story generation —
either by re-reading sources or asking the user (one question at a time).

### 3. Story Generation

From the brief's Goal, clarification Q&A, and research findings, derive 2–N user stories. Each story must be:
- Independent: delivers value without requiring other stories in this brief
- Specific: names the actor, the action, and the observable outcome
- Falsifiable: there exists a test that could prove it works or fails

For each story, supply:
- **Priority** (P0 = must-have, P1 = important, P2 = nice-to-have) with justification
- **Scope**: explicitly state what this story does NOT cover (1–2 sentences)
- **Independent Test**: one sentence describing how this story can be tested in isolation
- **Acceptance Scenarios**: at least `tunables.min_gwtscenarios_per_story` concrete Given/When/Then
  scenarios. Each scenario must:
  - Name a specific actor or system state (not "a user" — "a logged-in contributor", "an empty project directory")
  - Specify a concrete action with real values where applicable
  - Describe an observable outcome (visible output, file written, exit code, error message)

Format:
```
### US-1: <Brief title> (Priority: P0|P1|P2)
As a [role], I want [action] so that [outcome].
**Why this priority**: ...
**Scope**: This story does NOT cover [explicit exclusion].
**Independent Test**: ...
**Acceptance Scenarios**:
1. **Given** [concrete state with real values], **When** [concrete action], **Then** [observable outcome]
2. **Given** [concrete state with real values], **When** [concrete action], **Then** [observable outcome]
3. **Given** [error / boundary state], **When** [concrete action], **Then** [observable outcome]
```

If you cannot derive at least 2 independent user stories from the brief, write
`briefs/<task>-spec.md` with `**Status:** BOUNCED` and return to the user:
> ⛔ Spec bounced: cannot derive independent user stories from the brief.
> The brief describes [what it describes]. Missing: [what would make it specifiable].
Stop.

### 4. Challenge Sub-Agent (adversarial)

Spawn a sub-agent with this prompt:

```
You are an adversarial requirements engineer. You receive user stories and research context.
Your job: find every challenge that must be resolved before these stories can be implemented.

Rules:
- Never accept a story as valid without a challenge
- A challenge is a question, contradiction, edge case, or missing constraint
  that — if unresolved — would make the implementation ambiguous or wrong
- Challenges must be specific: reference the story number and the exact ambiguity
- At least 1 challenge per story

After listing challenges, also produce an ## Edge Cases section:
For each story, enumerate boundary conditions, error paths, and unexpected inputs
that are NOT covered by the acceptance scenarios. Format:
- EC-N [US-N]: [edge condition] → [expected behavior if known, or "behavior unspecified"]

Stories:
<list of US-N with their acceptance scenarios>

Research context:
<summary of research sub-agent output>

Brief:
<goal + scope boundary + architecture notes from intake brief>

Produce:
1. Numbered challenges: C-1, C-2, ... each citing its story. No solutions. Only challenges.
2. ## Edge Cases: EC-1, EC-2, ... each citing its story.
```

### 5. Adversarial Resolution (human gate)

Review the challenges. Classify each:

- **Resolvable from research/KB**: resolve immediately, document resolution.
- **Resolvable from brief**: resolve immediately, document resolution.
- **Requires user input**: add to questions list.

Ask the user questions **one at a time**. Remaining budget:
`max_questions_to_user − questions_asked_step2` (questions already used in Step 2).
Do NOT ask questions answerable by reading code or the brief.

If unresolved challenges exceed the remaining budget:
> ⛔ Spec bounced: <N> challenges remain unresolved after research. Returning to intake.
> Unresolved: [list challenges]
Write `briefs/<task>-spec.md` with `**Status:** BOUNCED — unresolved challenges: [list]`.
Stop.

### 6. Requirements Formalizer

Spawn a sub-agent with this prompt:

```
You are a requirements formalizer. You receive resolved user stories, their acceptance
scenarios, and the challenge resolution table.

Your job: produce FR-NNN MUST/MUST NOT statements — one normative requirement per
distinct behavioral obligation. Group by story. Each FR must:
- Be normative (use MUST, MUST NOT, SHALL, SHALL NOT)
- Be testable (maps to at least one acceptance scenario or runnable check)
- Be specific (names the actor, trigger, and observable outcome)
- Be marked [US-N] to show which story it belongs to

Format:
#### <Feature Area from US-N>
- **FR-001** [US-1]: System MUST <normative statement>
- **FR-002** [US-1]: <actor> MUST be able to <action> resulting in <observable>
- **FR-003** [US-2]: System MUST NOT <prohibited behavior> when <condition>

Do NOT include implementation details. Requirements are behavioral, not technical.
Do NOT add FRs not traceable to an accepted story or resolved challenge.

Stories + acceptance scenarios:
<all US-N with their scenarios>

Challenge resolutions:
<challenge/resolution table>
```

### 7. Cross-Spec Consistency Check

```bash
ls specs/*.md 2>/dev/null | head -20
```

If existing specs found: grep their `## Entities` sections for entity names that also
appear in your draft entities. For each match, compare the definitions. If they differ:

> ⚠️ Entity conflict: `<EntityName>` defined as `<X>` in `specs/<existing>.md` but
> as `<Y>` in this spec. Resolve before proceeding.

Ask user which definition is canonical. Update accordingly.

### 8. Write Spec File

Write `specs/<task-slug>.md`:

```markdown
---
name: <task-slug>
type: spec
status: live
feature: <feature name from brief goal>
brief: briefs/<task>-intake.md
date: <ISO-8601>
version: 1.0.0    # spec artifact version — not the roster-spec skill version
---

# Spec — <feature name>

## Clarifications

<!-- Q&A from the Clarification Elicitor step. Resolved [OPEN] items added here after user answers. -->
| Q | A |
|---|---|
| <question> | <answer> |

## User Stories

<!-- Each story includes GWT acceptance scenarios with concrete actors and values -->
### US-1: <Brief title> (Priority: P0|P1|P2)

As a [role], I want [action] so that [outcome].

**Why this priority**: <value justification — what breaks or degrades without this>

**Scope**: This story does NOT cover [explicit exclusion].

**Independent Test**: <how this story can be verified in isolation, without requiring other stories>

**Acceptance Scenarios**:
1. **Given** [concrete state with real values], **When** [concrete action], **Then** [observable outcome]
2. **Given** [concrete state with real values], **When** [concrete action], **Then** [observable outcome]
3. **Given** [error / boundary state], **When** [concrete action], **Then** [observable outcome]

### US-2: ...

## Challenges

| ID | Story | Challenge | Resolution |
|---|---|---|---|
| C-1 | US-1 | <challenge description> | <how resolved> |
| C-2 | US-2 | <challenge description> | <how resolved> |

## Functional Requirements

<!-- Derived from accepted stories + challenge resolutions by the Requirements Formalizer. -->
#### <Feature Area from US-1>
- **FR-001** [US-1]: System MUST <normative statement>
- **FR-002** [US-1]: <actor> MUST be able to <action> resulting in <observable outcome>

#### <Feature Area from US-2>
- **FR-003** [US-2]: System MUST NOT <prohibited behavior> when <condition>

## Acceptance Criteria

<!-- One AC per resolved challenge + one per story's happy path -->
- AC-1 [US-1, C-1]: [behavior] → [expected observable outcome]
- AC-2 [US-1 happy path]: [behavior] → [expected observable outcome]
- AC-3 [US-2, C-2]: ...

## Edge Cases

<!-- From the Challenge sub-agent's EC-N output -->
- EC-1 [US-1]: [error path / auth boundary / concurrency scenario] → [expected behavior]
- EC-2 [US-2]: [limit / rate / size constraint] → [expected behavior]

## Runnable Checks

<!-- Concrete shell/curl/pytest assertions — one per AC -->
<!-- Format: CHECK-N [AC-N]: `<exact command>` → expected: <what success looks like> -->
- CHECK-1 [AC-1]: `<curl/pytest/bash command>` → expected: <exit code / response / output>
- CHECK-2 [AC-2]: `<command>` → expected: <what success looks like>

## Entities

<!-- Domain entities referenced by this spec — for cross-spec consistency -->
- `<EntityName>`: <one-sentence definition>
- `<EntityName2>`: <one-sentence definition>
```

If `tunables.require_runnable_checks` is true and no concrete checks can be written
(e.g., feature is pure documentation): mark them as `CHECK-N: manual — <description>`.

### 9. Write Completion Artifact

Write `briefs/<task>-spec.md`:

```markdown
# Spec Brief — <task-slug>

**Date:** <ISO-8601>
**Status:** VALIDATED
**Spec file:** specs/<task-slug>.md
**User stories:** <count>
**Clarifications:** <count Q&A pairs>
**Challenges resolved:** <count>/<count total>
**Functional requirements:** <count FR-NNN>
**ACs:** <count>
**Runnable checks:** <count>
```

### 10. Announce

> "Spec complete for `<task-slug>`. [N] user stories, [N] clarifications, [N] challenges resolved,
> [N] FRs, [N] ACs, [N] runnable checks. Run `/roster-plan` to continue."

## Output Contract

- `specs/<task-slug>.md` — permanent indexed contract
- `briefs/<task>-spec.md` — pipeline completion marker (VALIDATED | SKIPPED | BOUNCED)

## When to Go Back

| Condition | Action |
|---|---|
| Cannot derive ≥2 independent user stories | BOUNCED → user must clarify brief with `/roster-intake` |
| >max_questions challenges unresolved after research | BOUNCED → re-run `/roster-intake` with challenge list |
| Entity conflict with existing spec, user cannot resolve | STOP — ask user to amend existing spec first |
| Type field missing from intake brief | STOP — re-run `/roster-intake` |

## What Next

**Primary path:** `/roster-plan`
**If bounced:** `/roster-intake` — brief must be enriched before re-running roster-spec

## Friction Log

```jsonl
{
  "date": "<ISO-8601>",
  "skill": "roster-spec",
  "task": "<task-slug>",
  "frictions": [],
  "methods": [],
  "suggestion_type": null,
  "suggestion": null,
  "effort_estimate": null
}
```

## Rules

- Never write the spec before resolving all challenges
- Never ask questions answerable by reading code or the brief
- Never produce a spec with 0 runnable checks unless explicitly marked manual
- Anti-sycophancy: challenge every requirement, including ones that seem obvious
- If both sub-agents agree a brief direction is wrong → USER-CHALLENGE, never auto-change
