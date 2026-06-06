---
name: harness-validator
description: Meta-auditor — validate the KB harness itself (structure, auditors, rules coherence, feedback loops).
version: 1.0.0
---

# Harness Validator

Verify the integrity of the knowledge base harness: KB structure, installed auditors, rule coherence, hook configuration, and feedback loop wiring.

## Steps

### 1. Check: KB Exists and Has Valid Structure

- Verify `kb/index.md` exists and lists all KB files.
- Verify core files exist: `spec.md`, `properties.md`, `glossary.md`.
- For each file in the index, confirm it actually exists on disk.
- Check each KB file has `last-updated` frontmatter.
- Flag missing core files as Critical. Flag missing indexed files as Warning.

### 2. Check: Auditor Skills Installed

- Verify each required auditor skill exists in `skills/kb/`:
  - `ambiguity-auditor.md` — always required
  - `code-quality-auditor.md` — always required
  - `harness-validator.md` — always required (self-referential check)
  - `spec-compliance-auditor.md` — required only if `specs/` directory exists in project root
    ```bash
    [ -d specs ] && echo "spec-compliance required" || echo "spec-compliance optional"
    ```
- Check each has valid frontmatter with `description`.
- Verify installed copies exist in `.harness/skills/` and any Claude compatibility copies in `.claude/commands/` (if install has been run).
- Flag missing auditors as Critical.

### 3. Check: Rules Coherence

- Read `.harness/rules/` first, then `.claude/rules/` compatibility copies if present.
- Read agent definitions (e.g., `agents/` directory or `AGENTS.md`).
- Verify: rules reference KB files that exist, agent instructions don't contradict rules, no orphaned rules referencing deleted agents.
- Flag contradictions as Critical. Flag orphaned rules as Warning.

### 4. Check: Hook Configuration

- Read `.harness/hooks/` and any runtime hook projection such as `.claude/settings.json`.
- Verify hooks don't block tools that agents need (e.g., a pre-commit hook that blocks Bash would break all agents).
- Check that hook commands reference valid scripts/paths.
- Flag blocking hooks as Critical. Flag missing expected hooks as Info.

### 5. Check: CLAUDE.md References KB

- Read `CLAUDE.md` (project root).
- Verify it references the KB (`kb/` directory, key files, or the KB update process).
- If CLAUDE.md exists but doesn't mention KB, flag as Warning.
- If CLAUDE.md doesn't exist, flag as Info (not required but recommended).

### 6. Check: Feedback Loop Wired (Ralph Loop)

- Verify the tech-lead agent (or equivalent) has eval criteria and a workflow that triggers auditors.
- Check for: a defined evaluation step that invokes at least one auditor, results feed back to planning.
- Look for evidence in agent definitions, AGENTS.md, or workflow configs.
- Flag missing feedback loop as Critical — the harness isn't closed without it.

### 7. Generate Report

Write to `kb/reports/harness-validation-report.md`:

```markdown
---
auditor: harness-validator
date: <today YYYY-MM-DD>
status: N critical, N warnings, N info
harness-health: HEALTHY | DEGRADED | BROKEN
---

## Harness Health: <HEALTHY|DEGRADED|BROKEN>

- **HEALTHY**: All checks pass, feedback loop closed.
- **DEGRADED**: Non-critical issues, harness functional but incomplete.
- **BROKEN**: Critical issues, harness cannot function correctly.

## Critical

### [C1] <Issue title>
- **Check**: <which check found this>
- **Issue**: <description>
- **Fix**: <specific remediation>

## Warnings

### [W1] <Issue title>
- **Check**: <which check found this>
- **Issue**: <description>
- **Recommendation**: <action>

## Info

### [I1] <Issue title>
- **Check**: <which check found this>
- **Note**: <description>

## Checklist Summary

- [ ] KB exists with valid structure
- [ ] All auditor skills installed
- [ ] Rules coherent with agents
- [ ] Hooks don't block needed tools
- [ ] CLAUDE.md references KB
- [ ] Ralph Loop (feedback loop) wired
```

## Rules

- This auditor checks the harness, not the project code — stay in scope.
- Run this after any structural change to the harness (new agents, changed rules, modified hooks).
- Create `kb/reports/` directory if it doesn't exist.
- A single Critical finding sets harness-health to BROKEN. Warnings without Criticals = DEGRADED. No issues = HEALTHY.
