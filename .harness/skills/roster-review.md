---
name: roster-review
description: Fix-first review with conditional specialists — produces a structured GO/NO-GO verdict.
version: 1.4.0
domain: pipeline
phase: review
preamble: true
friction_log: true
allowed_tools: [Read, Edit, Bash, Agent, AskUserQuestion]
human_gate: after
tunables:
  auto_fix_threshold_lines: 20
  always_run_spec_compliance: true
artifacts:
  reads:
    - briefs/<task>-impl.md
    - briefs/<task>-reviewer.md
    - git diff (current)
  writes:
    - briefs/<task>-review.json
pipeline_role:
  triggered_by: /roster-implement completed
  receives: briefs/<task>-impl.md + current diff
  produces: briefs/<task>-review.json GO or NO-GO
---

# Roster Review

You conduct a structured, fix-first review. Mechanical corrections are applied without asking. Ambiguities are grouped into one single question. You produce a structured JSON verdict.

**Golden rule:** every claim ("it's handled", "tests cover that") must cite the file and line. Never "probably" or "likely".

## Mode Awareness

**Read the mode from `briefs/<task>-impl.md`** (field `mode: express|fast|full`). If absent, infer from context.

| Mode | Review scope | Specialist invocation | Escalation check |
|---|---|---|---|
| **Express** | Correctness + security only. Skip spec/KB compliance — no spec impact expected. | `reviewer` agent only. Skip `spec-compliance`, `code-quality-auditor`, `architect` unless diff > 5 files. | Mandatory — see below |
| **Fast** | Full review. Spec/KB compliance only if KB exists. | `reviewer` + conditionals per normal rules. | Mandatory — see below |
| **Full** | Full review. All specialists per normal rules. | All conditionals apply. | N/A |

### Mode Escalation Check (Express and Fast only)

After reading the diff, check for signs the task scope exceeded its mode:

| Signal | Escalation |
|---|---|
| New public API, interface, or exported function | Recommend upgrading to Full (spec needed) |
| Behaviour change affecting callers beyond the reported fix | Recommend upgrading to Full |
| Design decision made implicitly in the code (no brief, no spec) | Recommend upgrading to Fast if Express, or Full if Fast |
| Spec or KB update is clearly needed but was not done | Flag as `escalation_needed: true` in verdict |

If escalation is needed: set `escalation_needed: true` and `escalation_reason` in the verdict. **Do not block GO** for this — it is informational. The human decides whether to loop back.

## Input Contract

Read in order:
1. `briefs/<task>-reviewer.md` — context and points of attention
2. `briefs/<task>-impl.md` — modified files and decisions made
3. `git diff main...HEAD` — the complete diff

If `briefs/<task>-impl.md` is absent:
> ⛔ Impl brief missing. Review cannot start without knowing the implementation scope.

## Steps

### 1. Read the diff

```bash
git diff main...HEAD
git log main...HEAD --oneline
```

Read each modified file in its entirety — not just the diff lines.

### 2. Fix-first: auto corrections

Apply the following mechanical corrections without asking:

| Category | Examples | Auto-fix threshold |
|---|---|---|
| Dead code | Unused variables, unused imports | Always |
| Magic numbers | Inline constants → named constants | Always |
| Stale comments | Comments that contradict the code | Always |
| Style / format | Local style inconsistencies, trailing whitespace | Always |
| Obvious DRY | Identical copy-paste block 3+ lines | If < `tunables.auto_fix_threshold_lines` |

**Do not auto-fix:**
- Security (auth, injection, XSS) → always in findings
- Race conditions → always in findings
- Visible behavior changes → always ask
- Refactors > `tunables.auto_fix_threshold_lines` lines → always ask

After each auto-fix, verify that quality gates still pass.

### 3. Conditional specialists

Spawn specialists based on scope. Each specialist receives:
- The complete diff
- The `briefs/<task>-reviewer.md`
- Their own instructions (path below)

| Specialist | Condition | Path / Invocation |
|---|---|---|
| `spec-compliance` (per-feature) | `specs/<task-slug>.md` exists | Invoke `spec-compliance-auditor` with spec path as `$ARGUMENTS` |
| `spec-compliance` | Always if KB exists (`kb/spec.md` present) | Skill — read `skills/kb/spec-compliance-auditor.md` and invoke via `Skill` tool or spawn as sub-agent with this content |
| `code-quality-auditor` | Always if KB exists (`kb/properties.md` present) | Skill — read `skills/kb/code-quality-auditor.md`; provide diff + `kb/properties.md` + `kb/glossary.md` + reviewer.md |
| `architect` | Medium or large blast radius (>3 files modified or public module) | `.claude/agents/architect.md` |
| `terminal-ux-reviewer` | TUI scope detected in diff or brief | `.claude/agents/terminal-ux-reviewer.md` |
| `reviewer` (agent) | Always | `.claude/agents/reviewer.md` |

**KB-conditional check:**

```bash
[ -d kb ] && [ -f kb/properties.md ] && echo "KB present" || echo "KB absent"
```

If KB is present, `code-quality-auditor` findings are merged into the review table alongside other specialists. Critical KB violations are auto-classified as HIGH severity.

When findings have `category: "spec"` and severity CRITICAL or HIGH:
- Set `no_go_reason.type = "spec-ac-failure"` in the verdict
- Populate `no_go_reason.failed_acs` with the AC identifiers from those findings

**Expected findings format from each specialist:**

```json
{
  "severity": "CRITICAL|HIGH|MEDIUM|LOW|INFO",
  "confidence": 1-5,
  "path": "file/path.ml",
  "line": 42,
  "category": "correctness|security|architecture|ux|spec|style",
  "summary": "Short problem description",
  "evidence": "File X line Y — exact code quote",
  "fix": "What to do",
  "fingerprint": "path:line:category",
  "specialist": "architect|reviewer|spec-compliance|terminal-ux-reviewer"
}
```

### 4. Deduplication

If two specialists report the same finding (same `fingerprint` or same path+line+category):
- Keep the finding with the highest severity
- Note that both specialists converged (confidence signal)

### 5. Group ambiguities

Collect all findings that require a human decision (severity HIGH+ on behavior changes, security, design).

Present in **one single** `AskUserQuestion`:

```
I have questions on [N] points before finalizing the review:

1. [path:line] — <finding summary> — <option A vs option B>
2. [path:line] — ...

For each point: A, B, or free-form answer.
```

Never ask multiple separate questions. One single pass.

### 6. Write the verdict

Produce `briefs/<task>-review.json`:

```json
{
  "task": "<task-slug>",
  "date": "<ISO-8601>",
  "status": "GO|NO-GO",
  "auto_fixes_applied": [
    {
      "path": "file.ml",
      "line": 10,
      "category": "dead-code",
      "description": "Removed unused variable `x`"
    }
  ],
  "findings": [
    {
      "severity": "HIGH",
      "confidence": 4,
      "path": "file.ml",
      "line": 42,
      "category": "correctness",
      "summary": "...",
      "evidence": "...",
      "fix": "...",
      "fingerprint": "file.ml:42:correctness",
      "specialist": "reviewer",
      "status": "OPEN|RESOLVED|ACCEPTED"
    }
  ],
  "summary": {
    "critical": 0,
    "high": 0,
    "medium": 0,
    "low": 0,
    "auto_fixed": 0
  },
  "no_go_reason": {
    "type": null,
    "failed_acs": []
  },
  "mode": "express|fast|full",
  "escalation_needed": false,
  "escalation_reason": null
  // type values: null | "spec-ac-failure" | "code-plan-failure"
  // escalation_reason: null | "new-public-api" | "implicit-design-decision" | "spec-update-needed" | "behaviour-change"
}
```

**GO status if:** no CRITICAL or HIGH OPEN finding.
**NO-GO status if:** at least one CRITICAL or HIGH OPEN finding not resolved or explicitly accepted.

### 7. Human gate

Present a summary:
```
Review complete.
Auto-fixes applied: <N>
Findings: <N> critical, <N> high, <N> medium, <N> low
Status: GO ✅ / NO-GO ❌

[If NO-GO]: resolve HIGH+ findings before proceeding to QA.
[If GO]: ready for /roster-qa.
```

## Output Contract

`briefs/<task>-review.json` with GO or NO-GO status and all findings documented.

**If GO:** `/roster-qa` can start.
**If NO-GO:** return to `/roster-implement` with OPEN findings.
**If NO-GO with `no_go_reason.type == "spec-ac-failure"`:** return to `/roster-spec` — the spec ACs were not met by the implementation.

## When to Go Back

| Condition | Action |
|---|---|
| NO-GO verdict — fixes required | Stop — return to `/roster-implement` with OPEN findings listed |
| Research reveals a design flaw missed in planning | Stop — re-run `/roster-plan` or `/roster-intake` before fixes |
| `code-quality-auditor` returns Critical KB violations | Auto-classify as HIGH finding → NO-GO unless immediately auto-fixable |
| `escalation_needed: true` in Express/Fast mode | Present to human — they decide whether to loop back to `/roster-spec` or accept as-is |

## What Next

**Primary path (GO):** `/roster-qa`
**Primary path (NO-GO):** `/roster-implement` — pass `briefs/<task>-review.json` as context
**Alternatives:**
- `/roster-audit` — if broader code quality concerns were flagged beyond this task

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

- Every coverage claim must cite the file and line — never "probably"
- "Looks good" is not a finding — if it's good, don't mention it
- A caught error or a rejected input is not automatically safe — check its blast radius: does the error abort a whole request, transaction, job, or batch rather than failing just the one operation? If so, flag it as a correctness/security finding
- One single grouped AskUserQuestion — never multiple separate questions
- Auto-fixes: verify quality gates after each fix
- Specialists must produce JSON findings — do not accept free-form text as output
- Do not auto-fix visible behavior changes even if under the line threshold
