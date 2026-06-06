---
name: roster-audit
description: Quality and compliance audit — combines code-quality and spec-compliance into one actionable report.
version: 1.1.0
domain: pipeline
phase: null
preamble: true
friction_log: true
allowed_tools: [Read, Bash, AskUserQuestion]
human_gate: after
tunables:
  max_function_lines: 50
  require_kb: false
  check_spec_compliance: true
  check_code_quality: true
  check_naming: true
artifacts:
  reads:
    - kb/spec.md
    - kb/properties.md
    - kb/glossary.md
  writes:
    - briefs/audit-<date>.md
pipeline_role:
  triggered_by: human or /roster-skill-health
  receives: optional scope in $ARGUMENTS (files / modules / entire repo)
  produces: briefs/audit-<date>.md with actionable findings
---

# Roster Audit

You audit code quality and its compliance with the KB. You produce actionable findings, not a style report. Every finding must cite the file and line.

**Token discipline:** concise findings. Do not paraphrase the KB — point to violations.

## Input Contract

- `$ARGUMENTS`: scope (e.g. `ocaml/agent-manager/src/` or empty for the entire repo)
- KB if it exists (`kb/spec.md`, `kb/properties.md`, `kb/glossary.md`)
- If `tunables.require_kb: true` and KB absent → block and say so

Default scope if $ARGUMENTS is empty: all source code (excluding `_build/`, `node_modules/`, `dist/`).

## Steps

### 1. Load references

If KB exists:
- Read `kb/properties.md` → invariants, thresholds, constraints
- Read `kb/glossary.md` → canonical naming
- Read `kb/spec.md` → specified behaviors

If KB absent and `tunables.require_kb: false` → continue with defaults (thresholds in tunables).

### 2. Check: function size (if `check_code_quality: true`)

```bash
# Identify long functions
grep -n "^let \|^  let \|^and " <scope>/**/*.ml | head -100
# (adapt pattern to the language)
```

Threshold: `tunables.max_function_lines` lines (default 50).
Report each function that exceeds this with: file, line, estimated size.

### 3. Check: DRY violations

Look for duplicated code blocks (≥ 5 identical or near-identical lines).

```bash
# Search for repeated patterns
grep -rn "<suspect pattern>" <scope>
```

Report with both locations.

### 4. Check: naming (if `check_naming: true` and glossary available)

For each term in `kb/glossary.md`:
- Search for variants (abbreviations, synonyms, different casing)
- Report inconsistencies with both forms (canonical vs found)

### 5. Check: spec compliance (if `check_spec_compliance: true` and spec available)

For each behavior specified in `kb/spec.md`:
1. Locate the implementation
2. Verify the match
3. Verify that a test covers this behavior

Classification:
| Status | Meaning |
|---|---|
| **PASS** | Code compliant + test exists |
| **PARTIAL** | Code compliant + no test |
| **DIVERGE** | Code behaves differently |
| **MISSING** | No implementation found |

### 6. Check: invariants

For each invariant in `kb/properties.md`:
- Verify it is preserved in the code
- If not statically verifiable → note "not statically verifiable"

### 7. Report

Produce `briefs/audit-<YYYY-MM-DD>.md`:

```markdown
# Audit — <date>

**Scope:** <audited scope>
**KB used:** YES / NO (reason if no)

## Summary

| Category | Findings | Actionable |
|---|---|---|
| Function size | N | N |
| DRY | N | N |
| Naming | N | N |
| Spec compliance | PASS: N / PARTIAL: N / DIVERGE: N / MISSING: N | N |
| Invariants | N | N |

## Actionable findings

### CRITICAL / HIGH
<findings that block or risk regressions>

### MEDIUM
<important quality findings>

### LOW / INFO
<minor findings>

## Non-actionable (for reference)
<findings not statically verifiable or accepted>
```

### 8. Human gate

Present the report and ask:
> "Which findings do you want to address now? I can create a `/roster-intake` for each group."

## Output Contract

`briefs/audit-<date>.md` with classified and actionable findings.

## When to Go Back

| Condition | Action |
|---|---|
| Findings reveal the current brief or plan is mis-scoped | Stop — re-run `/roster-intake` or `/roster-plan` with findings as context |
| Audit is blocked by missing KB or spec | Stop — ask human to run `/roster-init` or provide the missing spec |

## What Next

**Primary path:** `/roster-review` or `/roster-plan` — depending on whether findings are review-level or require re-planning
**Alternatives:**
- `/roster-intake` — if findings reveal a new task worth tackling separately

> 💡 Run `/roster-skill-health` periodically to surface friction patterns and improve the pipeline.

## Friction Log

```jsonl
{
  "task": "audit",
  "frictions": [],
  "methods": [],
  "suggestion_type": null,
  "suggestion": null,
  "effort_estimate": null
}
```

## Rules

- Every finding must cite file and line — never a generality
- "The code looks clean" is not a finding
- Without KB → apply tunable thresholds, do not invent rules
- Not statically verifiable → say so explicitly, do not assume
- Never modify code during the audit
